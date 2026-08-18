# The deployment image — one container, N virtualenvs

```
/opt/sadt/{.venv,server,runner.py}   the API: newest Python, fastapi, no torch
/tools/<name>/{.venv,.schema.json,src}
/DATA/<name>/{models,testfiles}      read-only mount, outside the image
/jobs/<uuid>/                        ephemeral
```

`import numpy` in the API venv is a `ModuleNotFoundError`, and that is the
point rather than an oversight: the server holds fastapi, uvicorn,
python-multipart and pydantic-settings, and every heavy dependency lives in a
tool's own `.venv`. Measured in the built image, `numpy_old` runs numpy 1.26.4
on Python 3.12 and `numpy_new` runs numpy 2.5.2 on Python 3.13, both answering
over HTTP from one container.

**The container does not run as root.** Third-party code from fifteen upstreams
runs in it, on confidential imaging. That is also why `.schema.json` is
generated into `SCHEMA_CACHE_DIR` at startup rather than rewritten in place:
`/tools` is read-only to the user serving it.

A tool is a **virtualenv, not a service**. Fifteen images would mean fifteen
copies of the same CUDA stack, fifteen things to schedule, and a network hop
between a tool and the file it was just handed. One image, one container, and
`exec()`.

## Building

```bash
docker buildx build -f docker/Dockerfile --build-context tools=<dir> -t sadt .
```

`<dir>` holds one folder per tool — `pyproject.toml`, `uv.lock`, `src/` — which
is what the `sadt-tools` repository holds. `.schema.json` is **generated during
the build**, by running `describe.py` with that tool's own freshly-synced
interpreter, because a schema read from a tool's source can only be produced by
something that can import it.

It is a **named build context** rather than a path inside this repo because the
tools genuinely live in another one; a named context overrides the stage of the
same name, so omitting it builds a server with no tools, which is valid.

The context may be a working checkout, `.venv/` directories and all — those are
excluded rather than copied, so pointing this at `~/code/sadt-tools/tools`
works without a `dist/` staging step. A tool nested one level deeper (a
grouping folder holding several related tools) is **not** discovered by the
server today: `registry/` and `execution/dispatch.py` look one level down from
`TOOLS_DIR`, and only the supervisor's lookup walks a group.

Through compose, where it is under a profile so nothing else changes:

```bash
docker compose --profile venvs up -d --build inference-venvs          # fixtures
TOOLS_CONTEXT=../sadt-tools/dist docker compose --profile venvs build # real tools
```

## Checking that it worked

```bash
docker run --rm sadt /opt/sadt/.venv/bin/python /opt/sadt/verify_dedup.py
```

uv installs a package by **hardlinking** it out of its cache, so the same wheel
in eleven virtualenvs costs the disk once. When that falls back to copying,
nothing observable fails — every tool still runs, every test still passes — and
the image is tens of gigabytes larger. Measured on the current tool set: one
isolated torch 2.2 CUDA stack is 4.9 GB; eleven torch tools are ~63 GB with
deduplication broken, ~26 GB with it working, ~17 GB if the pins are aligned
onto two runtimes.

The raw form of the same check, on an image that has torch:

```bash
stat -c '%h %n' /tools/*/.venv/lib/python*/site-packages/torch/lib/libtorch_cuda.so
```

A link count of 1 everywhere means deduplication is broken.

## The three things about uv that cost tens of gigabytes

1. **`UV_CACHE_DIR` must be on the same filesystem as the virtualenvs.**
   `link()` fails with `EXDEV` across filesystems and uv falls back to a full
   copy, silently. A BuildKit `--mount=type=cache` pointed straight at
   `UV_CACHE_DIR` breaks exactly this, because the mount *is* a separate
   filesystem. The cache mount is therefore transport only: copy in, sync,
   copy back out, prune — all in one `RUN`.
2. **Every `uv sync` in a single `RUN` layer.** overlayfs copies a file up when
   a later layer touches it, which breaks hardlinks between tools installed in
   different layers; and a cache deleted in a later layer frees nothing, it
   only writes a whiteout.
3. **`COPY` the manifests before the sources.** Docker's `COPY` cannot glob a
   directory structure, so the `pyproject.toml`/`uv.lock` files are extracted
   in a stage of their own. That stage re-runs on any change, but its *output*
   is content-addressed: unchanged lock files mean an identical layer digest,
   so `uv sync` stays cached and editing a tool's code does not reinstall its
   dependencies.

One more, which is why there is a single base image: `nvidia-*` wheels are
`py3-none-manylinux` — no Python ABI tag — so they deduplicate across
virtualenvs **even when the tools pin different Pythons**. A torch wheel
bundles its own CUDA runtime and the driver lives on the host, so a CUDA base
image per torch version buys nothing.

## Mounting tools instead of building them in

The image builds each tool's virtualenv **in place**, at the path it will be
used from, which is why none of what follows applies to it. It applies to every
other arrangement — a dev server pointed at a checkout, a CI job, anything that
mounts `sadt-tools` rather than baking it — and both constraints were found by
hitting them.

**Mount paths must match host paths exactly.** A virtualenv is not relocatable:
`bin/python`, the shebangs and `pyvenv.cfg` all hold absolute paths from where
`uv sync` ran. Mounted anywhere else the interpreter is simply not found, and
the tool fails to load with a message about a missing virtualenv rather than
about the mount.

    -v /home/you/code/sadt-tools:/home/you/code/sadt-tools:ro   # same path

**`~/.local/share/uv` must be mounted too.** uv does not copy an interpreter
into the venv; `bin/python` is a symlink to a uv-managed one that lives outside
the tool tree entirely. Mounting `sadt-tools` alone gives you a venv whose
python points at nothing.

    -v /home/you/.local/share/uv:/home/you/.local/share/uv:ro

**And the schemas have to come from somewhere.** `.schema.json` is a cache, not
a committed file. Without `DESCRIBE_PATH` pointing at `sadt-tools/scripts/describe.py`
and a writable `SCHEMA_CACHE_DIR`, every packaged tool fails to load. The server
now refuses to start in that case rather than serving only its in-process
fixtures — a registry of two reads as a small deployment, not a broken one.

## `docker/fixtures/` — what the image is proven with

Three tools that need no GPU, no model and no network:

| | pins | resolves to |
|---|---|---|
| `numpy_old` | `numpy<2`, Python `<3.13` | numpy 1.26.4 on 3.12 |
| `numpy_new` | `numpy>=2`, Python `>=3.13` | numpy 2.5.2 on 3.13 |
| `numpy_twin` | `numpy>=2`, Python `>=3.13` | numpy 2.5.2 on 3.13 |

The first two are the conflict the whole architecture exists for, in miniature
— `SurgMovPred` wants `numpy==2.4.0` while AREG/MedX/CLIC want `numpy<2.0.0`,
and no single interpreter can hold both. Note that the old pin drags an old
interpreter behind it: numpy 1.26 ships no wheel for 3.13, so uv installs
Python 3.12 into the image for that tool alone.

The third exists so `verify_dedup.py` has something to check: it resolves to
the *same* wheel as `numpy_new`, so those files must be one inode. Measured on
this image: 925 shared files, 54.4 MB not spent, and 610 MB against 686 MB for
the same image built with `UV_LINK_MODE=copy`.
