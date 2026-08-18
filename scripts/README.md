# `scripts/` — standing the server up, and filling `DATA/`

Two jobs, one folder. **Deployment**: go from a bare machine to a running
server, and keep it current. **Data**: fill `DATA/`, which is gitignored (it
holds confidential data and hundreds of GB of weights), from the public GitHub
releases the original Slicer modules already used — so nobody hand-copies
model bundles around.

| File | Role |
| --- | --- |
| [`setup-server.sh`](setup-server.sh) | Clone, check docker, start. One command from nothing. Runnable straight from GitHub. |
| [`install-docker.sh`](install-docker.sh) | Docker Engine + compose plugin (+ the NVIDIA toolkit on request). Linux, needs root. |
| [`install-docker-gpu.sh`](install-docker-gpu.sh) | The same from a bare Ubuntu, driver included: Docker, the NVIDIA driver if a card is present and unserved, then the container toolkit. |
| [`server_ctl.py`](server_ctl.py) | The deployment engine: `status` / `up` / `update` / `down` / `logs` / `catalog` / `models`. |
| [`setup-models.sh`](setup-models.sh) | Fetch AI models. Runnable straight from GitHub. |
| [`setup-testfiles.sh`](setup-testfiles.sh) | Fetch reference test files. Runnable straight from GitHub. |
| [`fetch_data.py`](fetch_data.py) | The download engine the wrappers call. Standard library only. |
| [`data-manifest.yml`](data-manifest.yml) | What exists, where it comes from, where it goes. |

Everything here is **standard library only**, on purpose: it runs on a host
before any `requirements.txt` is installed, and — for `server_ctl.py` — inside
Slicer's own interpreter, where nothing may be pip-installed on a user's behalf.

## What this folder does NOT do yet: put the tools on the machine

Worth knowing before reading the rest. `setup-server.sh` and `server_ctl.py`
start the `inference` (or `inference-cpu`) service, which mounts `./server` and
serves whatever `TOOLS_DIR` points at — by default `server/tools/`, i.e. the two
in-process demos. **The packaged tools are not fetched by anything here.**

Today they reach a server one of two ways:

- **the deployment image**, built with the tools as a named build context —
  `TOOLS_CONTEXT=../sadt-tools/dist docker compose --profile venvs build
  inference-venvs` (see [`docker/README.md`](../docker/README.md)). One image,
  N virtualenvs, port 8001;
- **a checkout**, for development — `./run-local.sh` with `SADT_TOOLS` pointing
  at a `sadt-tools` clone, which sets `TOOLS_DIR` and `DESCRIBE_PATH` and
  generates each schema at startup.

Neither is `curl | sh`, and that is the gap. The intended answer is a
`tools.lock` + volume bootstrap — tool folders fetched at pinned commit SHAs
into `/srv/sadt/tools/<name>/revs/<sha>/` behind a `current` symlink, under a
lock, provisioned by an entrypoint — so the image stays small, never rebuilds
when a tool changes, and a rollback is a symlink move. Until that lands, a
deployment stood up by these scripts serves `Test_Tool` and `Example_Tool`.

`DATA/` is the half that IS handled: the model bundles and test files below are
fetched independently of how the tools themselves arrive.

## Standing a server up

From nothing, on the machine that will host it:

```bash
curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-server.sh | sh
```

That clones the repo, checks docker (telling you exactly what to run if it is
missing), generates an API token into a `0600` `.env`, starts the right compose
service, waits for `/health`, and prints the URL and token to paste into the
Slicer client. Re-running updates the clone instead of re-cloning, and **keeps
the existing token** so clients already configured against this server keep
working.

Add `--tool NAME` (repeatable) to also download that tool's data. Nothing is
downloaded by default — the full set is ~29 GB and which tools a site uses is
not something to assume.

### Afterwards, from the clone

```bash
python3 scripts/server_ctl.py status            # prerequisites, clone drift, container, health
python3 scripts/server_ctl.py status --check-remote   # ... after a git fetch
python3 scripts/server_ctl.py update            # pull new commits, relaunch if anything changed
python3 scripts/server_ctl.py logs -n 200
python3 scripts/server_ctl.py down
python3 scripts/server_ctl.py token             # the API key, for a client
```

`update` **fast-forwards only** and refuses to pull over uncommitted changes.
When something did change it relaunches with `up -d --force-recreate`, never
`restart`: the container installs `requirements.txt` as part of its *command*,
into a writable layer `restart` keeps — so a new dependency that is never
re-resolved would be a silent no-op update, which is worse than a failed one.

### GPU or not

`docker-compose.yml` defines two server services: `inference` (with the nvidia
device reservation) and `inference-cpu` (without). A compose device
reservation is all-or-nothing and cannot be removed by an override file, so a
single service asking for a GPU simply cannot start on a laptop. `server_ctl.py`
picks between them from whether **docker** has an `nvidia` runtime — not from
whether `nvidia-smi` exists on the host, which says nothing about whether the
container toolkit is installed. Force it with `--device gpu|cpu`.

### A word on where it listens

`server_ctl.py up` publishes the port on **`127.0.0.1`** by default, because
this deployment speaks plain HTTP. `docker compose up` by hand still binds
every interface — IPv4 *and* IPv6 — unchanged, for the lab server that sits
behind a TLS terminator. Putting a plain-HTTP deployment on a network address
means medical images crossing the network in the clear — see `SECURITY.md`.

`BIND_ADDR` unset is not the same as `BIND_ADDR=0.0.0.0`: with no host address
docker publishes on both stacks, while an explicit `0.0.0.0` is IPv4 only.
Leave it out unless you mean to restrict.

`--port` moves the **host** side only (the container always serves 8000
internally) and is remembered in `.env`, so it is passed once and not on every
start. It exists because "something else already holds 8000" is an ordinary
thing to walk into on a workstation, and the alternative is a compose error
about a port buried in the output of a button press.

### Starting without a network

A start re-runs `pip install -r requirements.txt` inside the container, and
that install **cannot succeed offline** — `nnunetv2` → `batchgenerators` →
`unittest2` → `argparse`, and pip never treats `argparse` as satisfied because
the stdlib module shadows the distribution. It re-downloads that one 23 kB
wheel every time.

So the install does not gate the server: pip's failure is logged as
`DEPENDENCY-INSTALL-SKIPPED` and startup continues on what is already
installed. `server_ctl.py` repeats that as a note, because it is the one state
where a change to `requirements.txt` has *not* taken effect while the server
looks perfectly healthy.

A container that genuinely has nothing installed still fails, on purpose and
in one line (`DEPENDENCY-INSTALL-FATAL`): it needs the network once.

### `DATA/` has to exist before docker starts

`./DATA:/data:ro` is a bind mount, so a missing host path is created by the
**docker daemon** — owned by root. Every later download then fails with
"Permission denied" against the very directory the server reads, on a
brand-new install. `server_ctl.py` creates `DATA/` as the invoking user before
compose ever runs; if docker already won that race, it says so and gives the
one-line `chown` that fixes it.

### From Slicer instead

The **Slicer Cloud** module in `SlicerAutomatedDentalToolsCloud` is a panel
over exactly these subcommands: install, update, start/stop, per-tool model
selection, and it configures the extension's server URL and token for you when
the server comes up.

## Filling `DATA/`

On a machine with nothing checked out — run it from the directory that should
end up holding `DATA/`:

```bash
curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-models.sh | sh
curl -fsSL https://raw.githubusercontent.com/Jules-GP/slicer-remote-tool-server/main/scripts/setup-testfiles.sh | sh
```

From a clone, the same thing without the network round trip (the wrappers
detect `./scripts/` and use the local manifest, so a local edit takes effect):

```bash
./scripts/setup-models.sh
./scripts/setup-testfiles.sh
```

**Everything is about 29 GB** across 14 tools, 12 GB of which is ALI's CBCT
models. Ask for one tool at a time instead:

```bash
./scripts/setup-models.sh --tool AMASSS --tool SurgMovPred
python3 scripts/fetch_data.py --list          # sizes per tool, downloads nothing
```

Arguments reach the engine through `sh -s --` when piping:

```bash
curl -fsSL .../setup-models.sh | sh -s -- --tool AMASSS
```

Useful flags: `--tool` (repeatable), `--kind models|testfiles`, `--data-dir`,
`--force`, `--list`, `--progress`. `DATA_DIR` works as an environment variable
too, and `REPO`/`REF` point the wrappers at a fork or a branch other than
`main`.

`--progress always` prints a new download-progress line every few seconds
instead of redrawing one in place with `\r`. That is for a caller reading whole
lines out of a pipe — a GUI streaming this into a log pane — where `\r` is
invisible and a 12 GB bundle would otherwise print nothing at all for an hour.
On a terminal the default is unchanged.

### Picking tools, and coming back later

```bash
python3 scripts/server_ctl.py catalog     # per tool: on disk, still to fetch, free space
python3 scripts/server_ctl.py models --tool AMASSS --tool ALI
```

`catalog` compares the manifest against what is actually on disk, so a partial
install is legible: it reports **what a download would really transfer**, not
the tool's total size. That distinction is the point — "ALI: 12.3 GB" next to
an already-complete ALI is the number that makes someone skip a tool they
already have.

Anything already present is skipped, so adding one more tool six months later
costs exactly that tool. There is no separate "resume": re-running *is* the
resume.

## What you get

The layout is exactly the one `server/data_store.py` reads, so a file landing
here is immediately offered by `GET /tools/<tool>/data` — nothing else to
configure:

```
DATA/
├── AMASSS/
│   ├── models/AMASSS_Models/{CB,CBMASK,CV,MAND,MANDMASK,MAX,MAXMASK,SKIN,UAW}/…
│   └── testfiles/MG_test_scan.nii.gz
└── SurgMovPred/
    ├── models/all_models/<target>_Pred/stacking_package.pkl
    └── testfiles/TestFiles/patients_to_predict.xlsx
```

Re-running is cheap and safe: anything already on disk is skipped, so an
interrupted 12 GB download resumes by restarting the command. A download in
flight lives in a temporary folder and is only moved into place once complete,
so a killed run never leaves a truncated model that the next run would mistake
for a finished one.

## Adding an entry

Append to [`data-manifest.yml`](data-manifest.yml) — the header there documents
every field. The short version:

```yaml
  MyTool:
    models:
      - name: weights.zip        # what to download
        url: https://…           # where from
        size: 12345678           # optional, for the size report
        extract: true            # unpack, keep the folder, drop the archive
        dest: MyBundle/weights   # optional, overrides the destination name
```

`dest` exists for two situations worth knowing about before you hit them:

- **Name collisions.** Every nnUNet bundle ships a file literally called
  `checkpoint_final.pth`; without `dest` the second one downloaded would
  overwrite the first.
- **Grouping.** `server_selectable="model"` shows one dropdown entry per
  top-level name and hands the tool exactly one. ALI's 112 landmark archives
  are therefore filed under a single `ALI_CBCT_Models/` bundle rather than
  appearing as 112 separate choices.

## Checksums

**71 of the 77 entries carry no checksum, and that is a gap rather than a
design.** For almost everything under `DATA/` the only thing verified is the
byte count, so an upstream artefact replaced in place — or a download truncated
at exactly the right size — passes. These are model weights and patient test
data on a server built for confidential imaging. `fetch_data.py` proceeds on a
missing `sha256` with a log line rather than a refusal, which reads as a check
that happened.

**Backlog, not blocking:** every entry gets a `sha256`, and `fetch_data.py`
warns loudly or refuses rather than logging quietly when one is absent. Raised
2026-08-18, when `IOSCBCT_TestFile` downloaded with `no sha256 in the manifest`
and nothing stopped.


`sha256` is optional and mostly absent today. When present it is verified and
a mismatch discards the download rather than installing it. To pin an entry,
run the fetch once, take the hash the script prints, and paste it into the
manifest — the hashes are not published by GitHub, so inventing them would be
worse than leaving the field out.
