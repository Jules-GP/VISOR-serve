# CLAUDE.md — GPU Inference Server (tool-registry architecture) + 3D Slicer client

## Project context

We are offloading heavy computation from a 3D Slicer extension to a remote GPU
server. The Slicer module is a **thin client**: it sends inputs to the server, the
server runs the selected tool, and returns the result.

The extension will expose **at least 15 tools**, and this number will grow. The
whole design must therefore be **scalable and low-friction to extend**: adding a new
tool must require writing one self-contained class and nothing else — no edits to
the server core, no new route, no manual registration list to keep in sync.

**The data is confidential medical imaging.** Confidentiality and transport security
(TLS, auth, temp-file cleanup) are first-class requirements.

The HTTP contract is **blocking request/response** (the client gets its result in
the same response), but the server itself executes tools **in parallel**: each
`tool.invoke` runs in a worker thread, capped by `MAX_CONCURRENT_TOOLS` (see
`config.py`), so a long inference never freezes the event loop or other requests.
Do **not** add Celery/Redis/async job queues yet.

## Core design: a `Tool` base class

> **Historical, and still half true.** This section is the original brief, and
> it is what `base.py` still implements — `ArgSpec`, `validate()` before
> `run()`, `ToolArgumentError` → 422. What changed is where a tool's
> declaration comes from: a clinical tool is no longer a subclass here, it is a
> `.schema.json` generated from a `run()` signature in `sadt-tools`, which the
> registry turns into exactly this object (`registry/schema_tool.py`). Read the
> contract below as the contract the SERVER honours; read `ADDING_A_TOOL.md`
> for how a tool is actually written today.

Every tool is a **class deriving from a common `Tool` base class**. A tool declares:

1. **A unique name** (used to select it over the wire, e.g. `"Test_Tool"`).
2. **The arguments it expects**, as a typed schema (each argument has a name, a type,
   and whether it is required). This schema is what lets the server **validate**
   incoming arguments *before* running the tool.
3. **A `run(...)` method** that performs the actual work and returns the result.

The base class provides the shared machinery: argument validation against the
declared schema, a uniform way to be invoked, and metadata for discovery. Subclasses
only declare their schema and implement `run`.

### Validation contract (important)
- When a tool is invoked with a set of arguments, the base class **validates those
  arguments against the tool's declared schema** first:
  - every **required** argument is present,
  - no **unknown** arguments are passed,
  - each argument **matches its declared type** (coerce where sensible, e.g. form
    strings → int/float/bool; otherwise reject).
- If validation fails, **raise a clear, specific error** (a dedicated exception type,
  e.g. `ToolArgumentError`) naming what is wrong (missing arg, unexpected arg, wrong
  type). The server maps this to an HTTP `422` with the message.
- Only if validation passes is `run(...)` called. `run` can then trust its inputs.

### Suggested base-class shape (guidance, adapt as needed)
```python
@dataclass
class ArgSpec:
    type: type          # str, int, float, bool, "file", ...
    required: bool = True
    description: str = ""

class Tool(ABC):
    name: str                      # unique tool id, set on each subclass
    arguments: dict[str, ArgSpec]  # declared schema, set on each subclass
    output_kind: str = "text"      # "text" | "file" | "segmentation" | ...

    def validate(self, args: dict) -> dict:
        """Check args against self.arguments; return cleaned/coerced args or
        raise ToolArgumentError."""
        ...

    def invoke(self, args: dict):
        cleaned = self.validate(args)
        return self.run(**cleaned)

    @abstractmethod
    def run(self, **kwargs):
        """Do the actual work. Trusts that args are already validated."""
        ...
```
Use whatever concrete mechanism is cleanest (dataclasses, pydantic models, or a
simple dict of `ArgSpec`). The **hard requirement** is: declared typed schema +
validation-before-run + clear error on mismatch.

## Scalable tool discovery (the registry)

`server/registry/` discovers **two kinds of tool**, side by side, and
`GET /tools` publishes them identically — a client cannot tell which is which.

1. **A folder under `TOOLS_DIR` holding a `.schema.json`** (every clinical tool).
   The server reads the JSON, checks it against the hash of the `src/` beside
   it, and builds a `Tool` from it. **It imports nothing.** That is the whole
   point: the tool's dependencies have nothing to agree with the server's, and
   two tools wanting incompatible versions of torch stop being each other's
   problem. A folder that has a `.schema.json` is never imported.
2. **A `Tool` subclass under `server/tools/<name>/<name>.py`** (the two demos).
   Imported into this process at startup, the way every tool used to be.

Rules that hold for both:

- **Duplicate names are rejected** at startup, comparing case- and
  separator-insensitively (`Batch_Dental_Seg` and `BatchDentalSeg` are the same
  tool written two ways).
- **A tool that will not load is SKIPPED, never fatal.** With 15+ tools, one
  missing model must not block the others. It is logged in a banner, kept in
  `FAILED_TOOLS`, and named again by `get_tool()`.
- **The one exception is a `source_hash` that cannot be resolved.** A stale
  hash is a stale cache and is REGENERATED (`registry/schema_tool.resolve_schema`,
  which runs `describe.py` with the tool's own interpreter). A schema with no
  hash at all is unverifiable and is skipped.
- **A leading `_` excludes a folder from discovery**, which is what keeps
  `_dispatch_probe/` and `_AREG/` out of `GET /tools`.
- `TOOLS: dict[str, Tool]`, and `get_tool(name)` raises → `404`.

## The registered tools

Every clinical tool now lives in **`sadt-tools`**, one isolated project each,
and is served from `TOOLS_DIR` without this server importing a line of it.
Names are what a client sends to `/run/<name>`, and they are the folder names
on that side:

- `AMASSS` — CBCT skull structure segmentation (nnUNet v2, GPU).
- `ALI` — automatic landmark identification, on CBCT volumes (deep-RL agents)
  or intraoral surface scans (multi-view rendering + 2D UNet). The engine is
  chosen from the data, not from an argument.
- `ASO` — automated standardized orientation, CBCT and intra-oral scans. Its
  fully-automated CBCT mode calls `ALI` **mid-run**, through the supervisor.
- `AREG` — registration of two timepoints. Drives `AMASSS`, `ASO`, `Crown_Seg`
  and (through ASO) `ALI`, all through the supervisor.
- `Crown_Seg` — per-tooth labelling of intraoral scans (shapeaxi). Its own tool
  rather than a helper inside ALI, because ASO, AREG and FlexReg need it too.
- `Batch_Dental_Seg` — teeth and jaw structures on dental CT/CBCT, one scan or a
  whole cohort (nnUNet v2, GPU). Four trained models that label different
  things; the hosted bundle name selects the weights and their label table
  together.
- `Surg_Mov_Pred` — surgical movement prediction from tabular measurements
  (stacking models, server-side model bundles).

Two in-process tools stay in this repository, and only these two. They are the
demonstration of the `Tool`/`ArgSpec` path, not clinical tools:

- `Test_Tool` — two required strings in, their concatenation out. Proves the
  round trip end to end with no dependency at all.
- `Example_Tool` — the feature showcase: multi-type input (`csv_file` or
  `folder`), `choice`/`multichoice` arguments, `output_kind = "files"`.

The extension will eventually expose ~15+ tools; the architecture must
accommodate them without change to the core. See `ADDING_A_TOOL.md`.

## Target architecture

Three repositories, and the seams between them are the design:

```
 SlicerAutomatedDentalToolsCloud        slicer-remote-tool-server              sadt-tools
 ┌────────────────────────┐  HTTPS  ┌───────────────────────────┐        ┌────────────────────┐
 │ 3D Slicer modules      │ ──────► │ FastAPI (uvicorn)         │        │ tools/<Name>/      │
 │  - build the panel     │ POST    │  - verify token           │        │   pyproject.toml   │
 │    from GET /tools     │ /run/X  │  - registry.get(name)     │        │   uv.lock          │
 │  - upload in parallel  │         │  - tool.validate(args)    │        │   src/sadt_<name>/ │
 │  - load the result     │ ◄────── │  - dispatch → subprocess ─┼──exec──►   run(...)         │
 └────────────────────────┘  200    │  - stream / reference     │        │   .venv/  (its own │
                                    │  - delete every temp file │        │           torch)   │
                                    └───────────────────────────┘        └────────────────────┘
        knows only the schema            knows no dental tool               knows no server
```

The middle box imports nothing from the right-hand one. It runs it:

```
<TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json
```

`runner.py` ships with the SERVER and is injected by path, never installed into
a tool's venv — so runner and server are always the same version and there is
no cross-repo skew to negotiate. It reads `job.json`, imports the tool from its
`src/`, calls `run(**params)`, and writes `result.json`. A tool that declares
`*, sup` also receives a **supervisor** and can call another tool; that call
re-enters the same file with the sibling's interpreter, so chaining and nesting
(`AREG → ASO → ALI`) are one recursion rather than a feature.

## Repo structure

```
.
├── CLAUDE.md
├── ADDING_A_TOOL.md         # the contract for writing a tool (in sadt-tools)
├── MIGRATING_A_TOOL.md      # the record of how the tools left this repo
├── docker-compose.yml       # inference (GPU) + inference-cpu + inference-venvs + test services
├── docker/                  # the deployment image: one container, N tool virtualenvs
│   ├── Dockerfile           #   /opt/sadt (API, no torch) + /tools/<name>/.venv
│   ├── verify_dedup.py      #   are the venvs hardlinked, or silently copied?
│   ├── fixtures/            #   three tools with incompatible pins, no GPU needed
│   └── README.md
├── .env.example             # the three variables compose interpolates
├── .githooks/pre-push       # runs `docker compose run --rm test` before a push
├── .github/workflows/       # the same suite on every push and PR, plus the image build
├── run-local.sh             # a local server serving a sadt-tools checkout, port 8001
├── scripts/                 # stand the server up, and populate DATA/
│   ├── setup-server.sh      #   curl-pipeable: clone, check docker, start
│   ├── install-docker.sh    #   Docker Engine + compose plugin (Linux, root)
│   ├── install-docker-gpu.sh#   ... plus the NVIDIA container toolkit
│   ├── server_ctl.py        #   the deployment engine: status/up/update/down/catalog/models
│   ├── setup-models.sh      #   curl-pipeable entry points
│   ├── setup-testfiles.sh
│   ├── fetch_data.py        #   the download engine (stdlib only)
│   └── data-manifest.yml    #   what to download, and where it goes
├── server/
│   ├── main.py              # FastAPI app: /run/{tool_name}, /uploads, /results
│   ├── base.py              # Tool base class, ArgSpec, ToolArgumentError
│   ├── config.py            # every setting, from the environment
│   ├── data_store.py        # DataStore interface + LocalDataStore (server-side models/testfiles)
│   ├── file_utils.py        # shared helpers: scratch dirs, zip, scan extensions, tabular loading
│   ├── registry/            # what this server serves, and how it decided
│   │   ├── __init__.py      #   discovery: .schema.json folders, then tools/ subclasses
│   │   ├── schema_tool.py   #   a Tool built from a .schema.json, never imported
│   │   ├── schema_hash.py   #   source_hash: the reference implementation, executable
│   │   ├── conventions.py   #   what a tool gets with NO configuration at all
│   │   └── deployment.py    #   deployment.toml: the per-tool exceptions
│   ├── execution/           # running a tool, out of process
│   │   ├── dispatch.py      #   server side: job dir, GPU slot, timeout, error mapping
│   │   ├── runner.py        #   tool side: stdlib only, executed BY a tool's venv, injects `sup`
│   │   └── parity.py        #   run a tool both ways and compare what a caller receives
│   ├── wire/                # the HTTP edge that is not routing
│   │   ├── transfer.py      #   chunked resumable uploads, range-served results
│   │   └── security.py      #   Bearer token verification
│   ├── deployment.toml      # per-tool overrides; empty, because the conventions cover them
│   ├── deployment.toml.example
│   ├── tools/               # NOT where the tools are any more — see below
│   │   ├── _dispatch_probe/ # test fixture: underscore = never discovered
│   │   ├── _AREG/           # the parked in-process AREG, kept for its history
│   │   ├── Test_Tool/       # in-process demo: the minimal round trip
│   │   └── Example_Tool/    # in-process demo: multi-type input, choices, files out
│   ├── tests/               # HTTP-layer + integration tests, GPU- and weight-free
│   ├── requirements.txt     # what a dev checkout installs
│   ├── requirements-api.txt # what the API itself needs: fastapi, uvicorn, and nothing heavier
│   ├── requirements-dev.txt
│   ├── .env.example
│   └── README.md
└── DATA/                    # DATA_DIR mount, read-only, gitignored: <tool_name>/{models,testfiles}/
```

**The real tools are not in this repository.** They live in `sadt-tools`, one
isolated project each, and reach this server through `TOOLS_DIR` — a folder of
`<Tool_Name>/{.schema.json,.venv,src}`. `server/tools/` keeps only the two
in-process demos, the dispatch fixture, and the parked `_AREG`.

The Slicer client (thin modules + the generic inference client) lives in its
own repo, `SlicerAutomatedDentalToolsCloud` — not here. Its **Slicer Cloud**
module is a panel over `scripts/server_ctl.py`: it clones this repository,
checks Docker, starts the container, reports when the clone has fallen behind
and relaunches it, and picks which tools' bundles land in `DATA/`. The logic
stays here on purpose — a deployment fix ships with the server rather than
needing an extension release.

---

## PART 1 — Server (`server/` directory)

### Required stack
- **FastAPI** + **Uvicorn**, `python-multipart`.
- **No** database, queue, Celery, or Redis in this version.
- Python 3.10+.

### `base.py`
- `ArgSpec`, `Tool` (abstract base with `name`, `arguments`, `output_kind`,
  `validate`, `invoke`, abstract `run`), and `ToolArgumentError`.
- `validate` enforces: required present, no unknowns, type match/coercion; raises
  `ToolArgumentError` with a precise message otherwise.

### `registry/`
- `__init__.py` — discovery (both kinds, see above), `TOOLS`, `FAILED_TOOLS`,
  `get_tool(name)`.
- `schema_tool.py` — a `.schema.json` turned into a `Tool`, plus the schema
  vocabulary the two repositories agree on. An unknown key is a **warning**,
  never a refusal: this is the seam between two repositories, and a field one
  side adds must not stop the other from starting.
- `schema_hash.py` — `source_hash`, executable as
  `python server/registry/schema_hash.py <src>` so the generator on the other
  side can be checked against it byte for byte.
- `conventions.py` — **what a tool gets with no configuration at all.** An
  argument named `model`, `*_model` or `*_reference` is published as a name
  picked from `DATA/<tool>/models/` and can never be uploaded; any other `path`
  may also be filled from `DATA/<tool>/testfiles/`; an argument in `TECHNICAL`
  (`device`, `tile_step_size`, `num_workers`, `seed`, …) is not rendered to a
  clinician. Adding a tool needs no edit to this repository.
- `deployment.py` — `deployment.toml`, the exceptions to those conventions,
  merged per argument over them.

### `execution/`
- `dispatch.py` — the server half of a run: job directory, `job.json`, the GPU
  slot, the timeout, and the mapping from a tool's exception class NAME to a
  status code.
- `runner.py` — the tool half, executed **by the tool's own interpreter**.
  Standard library only, 3.9 → 3.13, injected by absolute path so runner and
  server are always the same version. It also injects the **supervisor**.
- `parity.py` — run one tool both ways and compare what a caller receives.

### `data_store.py` — server-side models and test files
Lets a tool argument be satisfied by a file already present on the server (an AI
model, a reference test dataset) instead of the client uploading it every call.
- `ArgSpec.server_selectable: Optional[str]` — `"model"` or `"testfile"` on a
  file-typed argument opts it into this; `None` (default) means upload-only.
- Layout on disk: `DATA_DIR/<tool_name>/models/` and
  `DATA_DIR/<tool_name>/testfiles/` (`DATA_DIR` from config, mounted **read-only**).
- `GET /tools/{tool_name}/data` (Bearer-protected) lists the available file names
  in both folders, so a client can present them (e.g. a dropdown) instead of a file
  picker.
- `GET /tools/{tool_name}/testfiles/{filename}` (Bearer-protected) streams one of
  the listed **test files** to the client (a folder entry arrives zipped) — backs
  the Slicer client's "Test file" button. Models are deliberately not
  downloadable: selected by name, used in place.
- In `POST /run/{tool_name}`, a `server_selectable` argument sent as a plain form
  value (the file name) instead of an upload is resolved against `data_store`
  rather than streamed to a temp dir.
- **Backend abstraction (`DataStore`):** `main.py` and every `Tool` only ever call
  `data_store.list_models/list_testfiles/resolve_model/resolve_testfile` — never
  the filesystem directly. `LocalDataStore` is the only implementation today. To
  swap to an external database or object store later: add a new `DataStore`
  subclass in `data_store.py` returning a `ResolvedFile(path, is_temporary)` from
  each `resolve_*` (`is_temporary=True` if the backend had to materialize a temp
  copy, so `main.py`'s cleanup deletes it — persistent local paths must never be
  deleted), then select it in `build_data_store()` via `settings.DATA_BACKEND`.
  No change needed anywhere else.

### `tools/Test_Tool/Test_Tool.py`
- Defines `TestTool(Tool)` with `name = "Test_Tool"`, the two required string
  args, and a `run` returning a str. It is the minimal proof that the HTTP
  round trip works with no dependency in the way — **not** the template for a
  new tool any more. A new tool is a package in `sadt-tools`; see
  `ADDING_A_TOOL.md`.

### Endpoints (`main.py`)
1. `GET /health` → `{"status": "ok"}`, no auth.
2. `GET /tools` → list of `{name, arguments, output_kind}` from the registry, no
   auth. Lets clients discover tools and their expected arguments.
3. `POST /run/{tool_name}` — generic, Bearer-token protected:
   - `404` if `tool_name` not in registry.
   - Collect arguments from the request (form fields for scalars; an optional
     uploaded `file` streamed to a temp dir in chunks — never fully into RAM).
   - Call `tool.invoke(args)` → this runs `validate` then `run`.
   - On `ToolArgumentError` → `422` with the message.
   - Return the result: JSON for text/scalar outputs, `FileResponse` for file
     outputs (correct media type).
   - **Guaranteed cleanup** of any temp files (input and output), including on error
     (`BackgroundTask` for the streamed output, `try/finally` for the input).

### Chunked transfer (`transfer.py`)

A file arriving in one request rides one TCP connection, which is bound by its
congestion window long before it is bound by bandwidth — hence a 100 MB CBCT
taking minutes, and a connection dropped at 95% starting again from zero. These
endpoints let a client use several at once. All Bearer-protected, all optional:
a client that ignores them still works, and one that uses them against an older
server falls back on the `404`.

4. `POST /uploads` `{filename, size, chunk_size?}` → `{upload_id, chunk_size,
   part_count}`. The server clamps `chunk_size` to [1, 64] MB and **answers with
   what it used**: part `n` is always `[n * chunk_size, (n+1) * chunk_size)`, and
   both sides compute offsets from that one number. A file over `MAX_UPLOAD_MB`
   is refused here, before a byte of it travels, which the multipart path
   cannot do.
5. `PUT /uploads/{id}/parts/{n}`, raw body, no multipart framing. `os.pwrite`
   at the part's offset into a pre-`truncate`d (sparse) blob, so concurrent
   parts write disjoint ranges of one file and there is **no reassembly pass**:
   each uploaded byte is written to disk exactly once. Idempotent, re-sending a
   part is how a client resumes. `X-Part-SHA256` is verified before anything is
   written, so a bad part is one retried part; since the parts tile the file,
   that verifies the whole upload without either side making a second pass.
   `Content-Encoding: gzip` is honoured (worth ~3x on an uncompressed `.nii` or
   a `.vtk`), and the checksum covers the *decompressed* bytes, what lands on
   disk, not what travelled.
6. `GET /uploads/{id}` → `missing_parts`. What makes a transfer resumable.
7. `DELETE /uploads/{id}`.
8. `GET /results/{id}`, honouring `Range` → `206`. `POST /run` hands back a
   *reference* instead of the bytes when the client sends
   `X-Result-Delivery: reference` **and the result is at least
   `RESULT_REFERENCE_MIN_MB`**; `DELETE /results/{id}` releases it.

   That threshold is about cleanup, not speed. A streamed `FileResponse`
   deletes its file through a `BackgroundTask` when the response ends, which
   fires even when the client disconnects mid-body (measured), so it depends on
   nothing the client does; a reference waits for a `DELETE` or for the reaper.
   Parallel ranges buy nothing under 16 MB, so the stronger guarantee is kept
   for the overwhelming majority of runs.
9. In `POST /run/{tool_name}`, an input that came up this way is named in the
   reserved `__uploads__` form field (`{argument name: upload id}`) instead of
   being sent as bytes. Its blob is **renamed** into the request's work dir, not
   copied, same filesystem, so a 2 GB upload becomes a tool's input in
   microseconds. Extensions are validated identically on both routes.

State lives on disk (a `meta.json` written once and never mutated, plus a
zero-byte marker file per received part), not in a module global: part `n` and
part `n+1` of one upload may legitimately be served by different `uvicorn
--workers`, and a session has to survive the `--reload` a code edit triggers
mid-transfer. It also means no lock anywhere, parts never overlap, and a marker
is created with `O_EXCL`. Ids are `secrets.token_urlsafe(24)`, matched against
`[A-Za-z0-9_-]{16,64}` *before* any path is built from them.

**Cleanup, and why it is a timer.** A reference is the one thing here that
survives its request, so it needs a bound that does not depend on the client
behaving. Three layers, in the order they normally fire:

1. The client `DELETE`s the result as soon as it has it, from a `finally`, so a
   download that failed halfway or an archive that failed its integrity check
   releases it too. Retried once.
2. `transfer.reap_expired` runs **on a timer** (`_reaper_loop`, every
   `TRANSFER_SWEEP_SECONDS`), not only opportunistically when a session is
   created: an abandoned transfer sits longest exactly when no new request
   arrives to trigger an opportunistic sweep.
3. `TRANSFER_TTL_SECONDS` is an **idle** timeout, not an age limit. Every part
   written and every range read stamps its directory (`transfer.touch`), so a
   transfer still in flight is never at risk however long it takes, while one
   whose client vanished expires 15 minutes later. That is what lets the number
   be minutes rather than the hours an age limit would need.

Worst case for patient data left on disk by a client that died mid-download:
`TRANSFER_TTL_SECONDS + TRANSFER_SWEEP_SECONDS`, about 16 minutes, with no
dependency on when the next request arrives.

### `wire/security.py`, `config.py`
- Bearer token from env (`API_TOKEN`), constant-time compare, `401` on failure.
- Config from env, grouped as `config.py` groups them:
  - **core** — `API_TOKEN`, `DEVICE`, `TEMP_DIR`, `DATA_DIR`, `DATA_BACKEND`.
  - **running a tool** — `SADT_DISPATCH_MODE`, `TOOLS_DIR`, `RUNNER_PATH`,
    `DESCRIBE_PATH`, `SCHEMA_CACHE_DIR`, `DEPLOYMENT_CONFIG`, `SADT_API`,
    `MAX_CONCURRENT_TOOLS`, `MAX_CONCURRENT_GPU_JOBS`, `TOOL_TIMEOUT_SECONDS`.
  - **uploads and results** — `MAX_UPLOAD_MB`, `MAX_EXTRACTED_MB`,
    `UPLOAD_CHUNK_MB`, `TRANSFER_TTL_SECONDS`, `TRANSFER_SWEEP_SECONDS`,
    `RESULT_REFERENCE_MIN_MB`, `ZIP_COMPRESSLEVEL`, `ALLOWED_EXTENSIONS`.
- **`MAX_CONCURRENT_GPU_JOBS` is one counter ACROSS tools.** The per-tool
  semaphores went with the tools that held them: a packaged tool is its own
  process, so an in-process semaphore would cap nothing, and an AMASSS run and
  a `Crown_Seg` run want the same card. A run is **assumed** to want the GPU
  unless it declares `device` and resolves it to a CPU value — the safe default
  is the strict one, because a tool that quietly imports torch without
  declaring `device` would otherwise never queue at all.
- **Every setting goes through `config.Settings`** — nothing reads `os.getenv`
  directly, so the whole configuration stays discoverable in one file and
  documented in `.env.example`. A *tool* now reads no setting at all: what used
  to be `settings.AMASSS_TILE_STEP_SIZE` is an argument of `run()` with the
  same default, and the server passes it only to override.

### Security / confidentiality — hard requirements
- **TLS mandatory**; README documents HTTPS + self-signed cert for dev, real cert
  for prod, never plain HTTP.
- Upload size limit (`MAX_UPLOAD_MB` → `413`).
- Client archives extracted for `"folder"` arguments are untrusted: zip slip,
  symlink members, and zip bombs (`MAX_EXTRACTED_MB`) all rejected with `400`.
- Delete all temp files after processing.
- Never log file contents, arguments values, or patient metadata. Logs limited to
  timestamp, endpoint, tool name, status, duration, size.
- Comment in `main.py`: deploy in the appropriate jurisdiction; de-identification
  happens client-side.

### `.env.example`, server `README.md`
- All env vars with dummy values.
- Install, self-signed cert generation, HTTPS run command, `curl` examples hitting
  `/tools` and `/run/Test_Tool` (with token). A short "how to add a tool" section:
  create a file in `tools/`, subclass `Tool`, done.

---

## PART 2 — Slicer client (`slicer_client/inference_client.py`)

Runs inside Slicer's Python interpreter. Available: `requests`, `slicer`, `qt`,
`vtk`, `os`, `tempfile`. No other external deps.

Provide a small, generic client mirroring the server:
- `check_server(server_url, verify_tls) -> bool` (`GET /health`).
- `list_tools(server_url, verify_tls) -> list` (`GET /tools`), so the client can see
  each tool's expected arguments.
- `run_tool(server_url, token, tool_name, args: dict, file_path: str = None,
  verify_tls=True) -> response`: POST to `/run/{tool_name}` with Bearer token,
  sending `args` as form fields and optionally streaming `file_path`. Blocking, with
  a generous configurable timeout. Returns the parsed result (JSON text or a written
  output file path depending on content type).
- Clean error handling mapping `401` / `404` / `422` / `413` / timeout / network
  errors to clear messages (via `slicer.util.errorDisplay` when wired into UI).

### Client-side security
- `verify_tls` defaults to **True**; False only for local dev, documented as such.
- Token not hardcoded: read from Slicer settings / env, passed as a parameter.
- Never log token or argument contents.

### UI integration (guidance only)
- Show how a module's `onApplyButton` calls `run_tool` with its own `tool_name` and
  an `args` dict, and how the same client serves all tools.
- Use `showStatusMessage` / wait cursor during the blocking call.

---

## Code style and conventions
- Clear code; comments at security, cleanup, and `# TODO` extension points.
- Explicit error handling, no silent `except: pass`.
- Type hints throughout the server.
- No hardcoded secrets.
- **All code, identifiers, comments, docstrings, and log messages in English.**

## Definition of done
- Server runs over HTTPS; `/health`, `/tools`, `/run/{tool_name}` all work.
- `Test_Tool` round-trips: send `text_1` + `text_2`, get a string back.
- Passing wrong/missing/extra args to a tool yields a `422` with a clear message
  (validation happens in the `Tool` base class before `run`).
- Unknown tool → `404`; no token → `401`; oversized file → `413`.
- Adding a new tool requires only a new file in `tools/` subclassing `Tool` — no
  core changes, no manual registration.
- No temp files left behind.

## Out of scope for this iteration (do not implement)
- Job queue / Celery / Redis / async polling. The contract stays blocking
  request/response.
- Scaling across machines, and a database.
- **VRAM budgeting.** `MAX_CONCURRENT_GPU_JOBS` is a job counter, not a memory
  one, and a supervised chain is invisible to it (a nested call is a subprocess
  of its parent, not a new admission). `runner.py` records
  `peak_vram_bytes` per run precisely so a real budget can later be set from
  measurements rather than guesses — that instrumentation is in, the policy is
  not.

Already implemented, despite earlier versions of this list: real GPU inference,
out-of-process execution, and in-process parallelism (tool runs execute
concurrently in worker threads, capped by `MAX_CONCURRENT_TOOLS`).

## Changelog

### 2026-08-12 — Read the other repository; seven ways nothing would have worked

`sadt-tools` has six tools packaged, and **not one of them would have loaded**.
The architecture matched — same reasoning, sometimes the same sentences — and
every wire between the two was wrong. Found by reading it, and by hashing a
real tool both ways.

- **`source_hash` was a different algorithm.** They hash the file's raw bytes
  and sort `Path` objects; this side hashed each file's digest and sorted
  strings, and `a/b.py` sorts before `a.py` one way and after it the other.
  Every tool would have looked stale. `schema_hash.py` is now a port of
  theirs, checked against `amasss` and `surgmovpred` and equal to the digit.
- **The module is `src/sadt_<tool>/`**, found as "the one package under src/",
  which is the rule their own generator uses. The runner looked for
  `src/<tool>.py` and would have imported nothing.
- **`.schema.json` is not a file they ship.** `scripts/describe.py` emits it
  from `run()`'s signature, run with THAT TOOL's interpreter, so the schema
  cannot drift from the code. It is a **cache**, and `source_hash` is what
  says the cache is behind — so a mismatch now REGENERATES rather than
  refusing to start, which is what that field is for. The image generates
  them at build; the server regenerates at startup into `SCHEMA_CACHE_DIR`,
  because `/tools` is read-only to the user it runs as.
- **They already publish `choices`**, from `Literal[...]` annotations:
  `list[Literal[...]]` is several-of, a bare `Literal[...]` exactly-one. The
  migration cost measured in `MIGRATING_A_TOOL.md` — "AMASSS loses 2
  multichoices" — does not exist; it was solved on their side while this side
  was throwing the field away with a warning.
- **`output_dir` is a required argument of every tool**, and no client can
  supply a directory on the server. It is taken out of the published schema
  and filled in at dispatch with the job's own `output/`. Without this every
  run is a 422 for a missing required argument.
- **Nothing serialises the GPU any anymore.** Every tool used to hold its own
  semaphore, which worked only because they shared one process; a packaged
  tool is its own process, so they have all been removed. New
  `MAX_CONCURRENT_GPU_JOBS`, and it is one counter ACROSS tools — an AMASSS
  run and a CrownSeg run want the same card. A run counts as GPU work when
  the tool declares a `device` argument whose effective value is a CUDA one,
  so a tabular prediction never queues behind a segmentation.
- **Errors map by exception class NAME**, there being no shared package to
  define a base class in: `ToolInputError`/`ValueError`/`FileNotFoundError`
  answer 422 with the message passed through (they are written to be read by
  whoever sent the request), `ToolUnavailableError` answers 503, anything else
  500 with a fixed message. The runner records the name in `result.json` —
  which reverses the earlier "on failure, write nothing".

Also: `device` is injected from `settings.DEVICE` when the caller picks none
(a tool that no longer reads the environment would otherwise run on cuda on a
CPU server); a `.zip` sent for a `path` argument is unpacked by the server,
cap and single-root strip included, since no tool unpacks archives any more;
and `deployment.toml` grows `data_dir`, because packaged tools are lowercase
(`amasss`) while the bundles staged under `DATA/` are not (`AMASSS/`).

**Verified against the real thing**: `amasss`, `batchdentalseg`, `crownseg`
and `surgmovpred` all load, publish their check boxes, hide their output
directory, and are correctly classified as GPU or not — `test_tool_contract.py`
runs their generator with their interpreters and skips where the checkout is
absent. `ali` and `aso` have no virtualenv yet, which their own document says.

**Tests:** 461 server tests (+27).

### 2026-08-12 — The gate phase 4 has to pass through: running a tool both ways

Phase 4 deletes `server/tools/`, and every deletion there is one line and no
way back. What licenses it is not that the subprocess path *works* — phases 1
to 3 showed that — but that a given tool produces the same thing on both
sides. `server/parity.py` runs one tool in both forms and compares **what a
caller receives**:

- every file produced, keyed by name and hashed. Absolute paths are never
  compared: one run wrote into a job directory, the other into a scratch
  directory, and neither name means anything to a client;
- the returned value, with each path replaced by the artifact it names, so
  `{"outputs": {"mandible": "/jobs/ab12/output/x.nii.gz"}}` and
  `/tmp/inference_server/tool_9f/x.nii.gz` compare equal;
- minus the keys that differ between two runs of the *same* code — duration,
  timestamp, job id.

**It does not claim a difference is a defect.** The packaged tool runs against
its own pinned dependencies; a different numpy moves voxels and a newer
SimpleITK writes a different header. It makes the difference visible, per
file, so it is read rather than discovered by a clinician. Exit code 1 on any
difference, and the report says which file and which side.

Tested in both directions on `_dispatch_probe`, which now exists in both forms
— packaged with its own venv, and an in-process twin in the test: agreement is
reported as agreement, a twin returning a different total is caught, and a twin
writing a *different file with the same answer* is caught. That last one is
the failure a smoke test misses.

`file_utils.output_paths` (moved out of `main.py`) is now what both the
response builder and the harness use to find what a run produced, so "what
counts as an output" is written once.

**`MIGRATING_A_TOOL.md`**: the loop, per tool — package, translate the schema,
move `server_selectable` to `deployment.toml`, build, prove, flip — plus what
the translation *costs*, measured against the current registry. `test_tool`,
`SurgMovPred` and `BatchDentalSeg` are nearly free; `ASO` loses 3 choices, 4
multichoices and **7 `visible_when`** rules, whose whole job is hiding the
inert half of its four modes, so it should go last. Rolling a tool back is
deleting its `.schema.json`, which is why the in-process copies stay until the
very end.

**Tests:** 434 server tests (+7).

### 2026-08-12 — One image, N virtualenvs, and the server imports none of them (phase 3)

`docker/Dockerfile`: `/opt/sadt/{.venv,server,runner.py}` for the API on the
newest Python, `/tools/<name>/{.venv,.schema.json,src}` for the tools, `/DATA`
read-only, `/jobs` ephemeral. A tool is a **virtualenv, not a service** —
fifteen images would mean fifteen copies of the same CUDA stack, fifteen things
to schedule, and a network hop between a tool and the file it was just handed.

**Measured, in the built image:** `numpy_old` runs numpy 1.26.4 on Python
3.12.13 and `numpy_new` runs numpy 2.5.2 on Python 3.13.15, both answering over
HTTP from one container, while `import numpy` in the API venv is a
`ModuleNotFoundError`. That is the entire point of the migration, demonstrated
rather than described.

- **The uv cache is transport, never `UV_CACHE_DIR`.** uv installs by
  hardlinking out of its cache; `link()` fails with `EXDEV` across filesystems
  and uv falls back to copying, silently. A BuildKit cache mount *is* a
  separate filesystem, so pointing `UV_CACHE_DIR` at one breaks exactly what
  the mount was added to help. It is copied in, synced against, copied back
  and pruned inside one `RUN`.
- **Every `uv sync` in ONE layer**, because overlayfs copies a file up when a
  later layer touches it, and a cache deleted in a later layer frees nothing.
- **`docker/verify_dedup.py`, and it had to be written twice.** The naive
  check — same path, different inode — reported 114 failures on a correct
  image: numpy 1.26 and 2.5 share a few dozen byte-identical test fixtures
  that come from *different wheels*, which uv could not share if it wanted to.
  It compares files of the same distribution **at the same version**, from
  each venv's `RECORD`, minus what an installer writes rather than unpacks
  (console scripts embed their own venv's interpreter path). Verified in both
  directions: 0 failures on the real image, 925 duplicated files and 54.4 MB
  wasted on the same image built with `UV_LINK_MODE=copy` — 610 MB against
  686 MB, on nothing heavier than numpy.
- **Manifests are extracted in a stage of their own.** `COPY` cannot glob a
  directory structure, and that stage's *output* is content-addressed:
  unchanged lock files mean an identical digest, so editing a tool's source
  leaves `uv sync` CACHED. Confirmed by rebuilding after a source edit.
- **The tools arrive through a named build context** (`--build-context
  tools=<dir>`), because they genuinely live in another repository. A named
  context overrides the stage of the same name, so omitting it builds a server
  with no tools rather than failing.
- **An old pin drags an old interpreter behind it.** numpy 1.26 ships no wheel
  for 3.13, so the first build tried to compile it from source and failed. uv
  installs Python 3.12 into the image for that tool alone — one base image
  still, because `nvidia-*` wheels are `py3-none-manylinux` and deduplicate
  across Python versions anyway.
- **`registry.py` no longer requires the `tools/` package to exist**, which is
  what lets the image ship without the in-process tools — and is the shape
  phase 4 leaves behind.
- **`file_utils` imports pandas lazily now.** It was the one module-level
  heavy import in the server core, and `main.py` imports `file_utils`: the API
  venv could not have been slim while it stayed. `load_tabular_*` are tool
  helpers.
- **The container does not run as root.** Third-party code from fifteen
  upstreams runs in it, on confidential imaging.

`server/requirements-api.txt` is the API's whole dependency list — fastapi,
uvicorn, python-multipart, pydantic-settings — and a test asserts it stays that
way, because an API that quietly regrows numpy is pinned to what the tools can
agree on all over again.

**Tests:** 427 server tests (+4). The image itself is verified by building it,
which the suite does not do.

### 2026-08-12 — A tool can be declared without being imported (phase 2)

Phase 1 gave a tool its own process. This gives it its own *declaration*: a
folder holding `.schema.json`, `.venv/` and `src/`, from which the server
builds a `Tool` without importing a line of it. `registry.py` now discovers
both kinds, and dropping a `.schema.json` into a folder is what moves a tool
from one to the other — a folder that has one is never imported.

**Both kinds at once, and that is not a compromise.** The two bullets of the
phase — "registry reads schemas instead of importing" and "the golden
`GET /tools` test must still pass" — are only simultaneously satisfiable that
way: the eight tools here have no schema, and the contract's type vocabulary
(`path`, `str`, `int`, `float`, `bool`, `list[str]`) *cannot* express what they
publish — `volume_or_zip_file` and its extensions, a `multichoice` over 119
landmarks, `visible_when`, `ui`, `groups`. A schema-only registry today means
either no tools or a different response. The server stops importing tools when
the tools leave this repo, which is phase 4.

- **`source_hash` is fatal, alone among discovery failures.** Everything else
  costs one tool (skipped, reported, `FAILED_TOOLS`); a schema that no longer
  matches the source beside it takes the server down, because it would
  otherwise keep serving — validating requests against a signature that has
  changed under it, accepting arguments `run()` no longer takes and refusing
  ones it now does. A schema with NO hash is unverifiable and skipped instead:
  it must not serve, but it endangers only itself.
- **`schema_hash.py` is executable**, `python server/schema_hash.py <src>`.
  The generator lives in the other repository and the two must agree byte for
  byte or nothing starts, so this ships a reference implementation to check
  against rather than a description to reimplement. The rule: sha256 over
  `<relative posix path>\0<sha256 of contents>\n` per file, sorted,
  `__pycache__` and `*.pyc` excluded — every clause of which exists to make the
  hash reproducible on another machine.
- **The folder must be named after the tool.** Not cosmetic: `dispatch.py`
  looks the interpreter up at `<TOOLS_DIR>/<tool name>/.venv/bin/python`, so a
  mismatch registers a tool that cannot be run.
- **`deployment.toml` is the server's half of the declaration.** A schema is
  generated from the tool's source and is the same wherever it is installed;
  which arguments may be filled from *this* machine's `DATA_DIR`, and how much
  *this* machine accepts as an upload, are not properties of the tool. Absent
  is the normal case. A `server_selectable` entry naming an argument the tool
  does not declare, or one that is not a `path`, is a startup error — the
  failure is otherwise a dropdown that silently never appears.
- **`max_upload_mb` is enforced where the tool is known**: on the multipart
  body, and when a chunked upload is *claimed*. `POST /uploads` opens a session
  for a file, not for a tool, and keeps applying the global `MAX_UPLOAD_MB`
  while the transfer runs.
- **An unknown key in a schema is a WARNING, not a refusal.** This is the seam
  between two repositories: a field one side adds must not stop the other from
  starting, and must not vanish in silence either. An unknown key in
  `deployment.toml` is the opposite — that file is ours, and
  `server_selectible` would just leave every argument upload-only.
- **`base.LIST_TYPE`** (`"list[str]"`), the one argument shape the schema can
  declare that nothing here could express. Not `multichoice`, which picks from
  a catalog the tool declares; this is free text. A type-system addition of the
  same kind as the choice types, made once.

**Where the two repositories could disagree, this server takes both.** Each of
these is read if present and costs the generator one optional field; none of
them changes anything for a tool that omits it:

- **`extensions` on a `path` argument** (`ArgSpec.accepts`, new). The contract's
  types cannot say more than "a path", so the client's file dialog would fall
  back to `ALLOWED_EXTENSIONS` where AMASSS today offers exactly
  `.nii/.nii.gz/.nrrd/...`. This is how a schema tool narrows its own picker
  without a `FILE_TYPES` entry being invented for it.
- **`description` per argument.** The client renders it under the field.
- **`{"outputs": {name: path}}` as a return value, and it is the form to
  write.** The contract shows `"returns": "path"` next to a `result.json` of
  `{"result": {"outputs": {...}}}` — a mapping, not a path, which
  `_output_paths` refused and which answered 500. Both work now, and the
  mapping is canonical: the names buy nothing over HTTP (the response is one
  file or one archive either way) but they are what `depends_on` sequencing
  will wire on. Feeding AMASSS's mandible into AREG means
  `params["scan"] = result["outputs"]["mandible"]`; with a list of paths the
  server picks by extension or by position, which is a guess. When that lands,
  the names have to be declared in `.schema.json` too, so a wiring naming an
  output no tool produces fails at startup like every other schema mistake.
- And a single-file tool returning SEVERAL paths is now a failure rather than
  the first of them streamed as if it were the result.

**Still needing a client release, so left alone:** the schema's tool-level
`description` is read and kept but not published — `GET /tools` has no field
for it, and adding one is a shape change.

**Tests:** 423 server tests (+43), no GPU, no weights and no network.

### 2026-08-12 — A tool can run in its own interpreter (phase 1: the path, no tool on it)

`registry.py` imports every tool INTO the server, which pins the server's
Python to the lowest common denominator across all of them and makes two
incompatible pins simply unresolvable — `SurgMovPred` wants `numpy==2.4.0`
while AREG/MedX/CLIC want `numpy<2.0.0`, and two versions of torch cannot
coexist in one process at all. Holding torch in the API process also keeps a
CUDA context alive for the life of the server (VRAM is never fully released
between jobs) and turns a segfault in a CUDA kernel into a dead API rather than
a failed job.

`dispatch.py` + `runner.py` are the two halves of the way out:

    <TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json

with `SADT_API`, `SADT_JOB_ID`, `SADT_JOB_DIR` in the environment. The server
writes `job.json` (`job_id`, `tool`, `job_dir`, `params`), the runner imports
the tool from its `src/`, calls `run(**params)` and writes
`{"result": ...}` to `result.json`. On failure it writes NOTHING and exits
non-zero — an absent `result.json` is the failure signal, which is why it is
serialized in full and `os.replace`d into place rather than streamed.

- **Nothing switches over yet.** `SADT_DISPATCH_MODE` defaults to `inprocess`,
  and the flag is temporary. `Tool.invoke` validates first either way, so a bad
  request still costs no process; `main.py`, `registry.py` and every tool are
  untouched.
- **`runner.py` ships with the SERVER and is injected by path**, never
  installed into the tool venvs: runner and server are then always the same
  version and there is no cross-repo skew to negotiate. It is standard-library
  only and has to run on 3.9 through 3.13, since each tool pins its own
  interpreter. `tools/_dispatch_probe/` proves exactly that — its venv holds no
  third-party package at all, not even pip.
- **The job directory is a tracked scratch dir** (new
  `file_utils.register_scratch_dir`, for a caller that must name its own), so
  the request handler already removes it — outputs included — once the response
  has streamed, and `_discard` takes it on every error path. A failed run is
  deleted immediately instead of waiting: its inputs are confidential and its
  outputs are worthless.
- **stdout/stderr go to files, not pipes.** nnUNet prints for hours and a pipe
  means holding all of it in the server's memory. Only the last 8 kB of stderr
  travels with the failure, into the server log — `ToolExecutionError` is
  deliberately not one of `base.py`'s typed errors, so it falls through to the
  generic 500 and the client is told only that the run failed, exactly as an
  in-process crash is today. A missing venv is the opposite case and answers
  501 through `ToolUnavailableError`.
- **`API_TOKEN` is stripped from the child environment.** The rest is
  inherited (PATH, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES), but the server's
  bearer token has no business in a venv full of third-party code.
- **`cwd` is the job directory**, so a tool writing a relative path lands
  there rather than in the server's source tree — which is what ALI's original
  module did, into the extension's own sources.

**The golden fixture comes first.** `tests/golden/tools_response.json` is
`GET /tools` as the in-process server produces it, captured before a line of
this was written and asserted per tool, per argument, in order. The Slicer
client builds its entire UI from that response; the point of the whole
migration is that it cannot tell the difference. If that test fails, the client
breaks — the fixture is not what gets updated.

**Tests:** 380 server tests (+25), no GPU, no weights and no network.

### 2026-08-11 — BatchDentalSeg ported: four models, and a manifest that could not load

Port `BATCHDENTALSEG` (teeth and jaw structures on dental CT/CBCT, nnUNet v2).
Upstream is a 2940-line Qt widget, and most of it is not this pipeline: a queue
table, a RAM watchdog, killing nnUNet processes a crashed scan left behind, a
"free memory" button, a cool-down between scans, restoring the queue from disk.
All of it exists because the widget runs inside Slicer on a clinician's laptop
and has to survive being out of memory. Here the queue is a folder argument,
concurrency is `MAX_CONCURRENT_TOOLS` plus a GPU semaphore, and a failure is an
exception. None of it is ported.

**Four models, and they do not label the same things.** DentalSegmentator and
PediatricDentalSeg label 5 segments with the maxilla inside Upper Skull;
NasoMaxillaDentSeg separates the maxilla, which shifts every later value;
UniversalLab labels all 32 permanent teeth, 20 deciduous ones and 3 structures.
The values are what the networks emit, so a wrong table does not fail, it
renames anatomy — `AMASSS_report.json`'s counterpart publishes `labels` with
every run for that reason.

**The hosted bundle name IS the model.** `model` is a scalar
`server_selectable`, and its directory name selects the weights and the label
table together. The first version had a second `dental_model` choice beside it,
which meant a caller could pair bundle X with the table of Y and get a
plausible volume with every structure named wrong. One argument cannot express
that.

**The manifest could not have loaded.** `scripts/data-manifest.yml` already
carried the four bundles, marked NOT PORTED YET, with the checkpoint written
flat beside `dataset.json` and `plans.json`. nnUNet reads its fold from a
subdirectory named after it, so every bundle would have downloaded perfectly
(2.3 GB) and then no model would ever have been found. `dest` now puts the
checkpoint under `<bundle>/fold_0/`, and `find_model_folder` confirms all three
files before accepting a candidate — a half-downloaded bundle reports "not
installed" rather than failing inside nnUNet's loader. A test pins the code and
the manifest to the same folder names, which is why `docker-compose.yml`'s test
services now mount `scripts/` read-only.

**A defect the tests found, in this port's own code.** The scan-to-NIfTI
conversion loop ran before inference with no per-scan guard, so one corrupt
file in a cohort aborted the whole run before a single scan had been
segmented. It is guarded now and reported per scan; a batch where *nothing*
could be read is a 422 rather than a successful run of zero scans.

**Also not ported, each for a stated reason:** the runtime model download from
GitHub releases (a server holding patient data does not make outbound calls
mid-request; bundles are staged with `setup-models.sh --tool BatchDentalSeg`);
the auto-crop (upstream applies it only when its RAM preflight fails, as a
laptop mitigation, and it changes what the network sees); the mirroring
resolution (a button pressed after looking at the result, not part of the
automatic pipeline); and the mesh exports (STL/OBJ/GLTF/VTK), which are the
obvious next addition.

`nnunet_runner.py` is deliberately a second copy of AMASSS's rather than an
import: `registry.py` imports every tool at startup, so importing another
tool's module would make one tool's missing dependency take both out of the
registry — the same reason ASO and AREG each carry their own dicom.py. It does
NOT carry AMASSS's GPU-resampling swap: that drops the input resampling from
spline order 3 to order 1, and nothing has measured what that costs *these*
models.

**Core changes, both sanctioned:** two `config.Settings` fields
(`BATCHDENTALSEG_MAX_GPU_JOBS`, `BATCHDENTALSEG_TILE_STEP_SIZE`). `main.py`,
`registry.py` and `base.py` untouched, and no new dependency — the image
already carries nnUNet v2 for AMASSS.

**Tests:** 354 server tests (+26), no GPU, no weights and no network: inference
is stubbed, everything around it runs for real. Not yet verified against the
real bundles — they are 2.3 GB and have not been fetched on this machine.

### 2026-08-07 — `scripts/` stands the server up, not just fills `DATA/`

`scripts/server_ctl.py`: `status` / `up` / `update` / `down` / `logs` /
`catalog` / `models` / `token`, standard library only — it also runs inside
Slicer's interpreter, where nothing may be pip-installed on a user's behalf.
Two conventions carry the GUI integration: progress goes to **stderr** and
machine-readable output to **stdout**, so `--json` prints exactly one object;
and nothing prints the API token except `server_ctl.py token`, because a status
dump lands in log panes and bug reports.

`install-docker.sh` (Linux, root) runs Docker's own `get.docker.com` and adds
the user to the `docker` group, saying loudly that group membership only
applies to a **new login session**. `--nvidia` adds the container toolkit, not
a GPU driver. `setup-server.sh` is the curl-pipeable one-liner: clone, check
docker, optionally fetch tools, start, print the URL and token — re-running
**keeps the existing token**, so configured clients keep working.

Decisions worth keeping:

- **`inference-cpu` under a `cpu` profile**, sharing an anchor with
  `inference`. A compose device reservation is all-or-nothing, and an override
  file cannot rescue it: compose **merges** the `devices` list rather than
  replacing it, so `devices: []` in a second `-f` leaves the reservation in
  place (measured). Which one runs is decided by whether docker has an `nvidia`
  runtime, not by whether `nvidia-smi` exists — the container toolkit is a
  separate install.
- **`BIND_ADDR` unset by default**, written as `127.0.0.1` by `server_ctl.py
  up`: a local deployment speaks plain HTTP, which must not leave the machine.
  Unset, NOT `0.0.0.0`: with no host address docker publishes on both stacks,
  while an explicit `0.0.0.0` is IPv4 only and silently drops IPv6 clients. The
  `${BIND_ADDR:+${BIND_ADDR:-}:}` form keeps that distinction; the inner `:-`
  stops compose warning on every command. `docker compose config` for
  `inference` is byte-identical to before.
- **`HOST_PORT`**, with a preflight that refuses to start when anything is
  listening — checked by connecting, not by `compose ps`, since a second clone
  is its own compose project and `ps` reports nothing while it holds the port.
- **`--branch` on `status` and `update`.** `clone()` runs only on an empty
  directory, so a deployment repointed at another branch kept following the old
  one in silence. The git half of `cmd_update` runs **before** the docker
  preflight: fetching new code needs neither docker nor a free port, which are
  what someone may be updating in order to fix.
- **The dependency install no longer gates uvicorn.** `pip install -r
  requirements.txt` cannot succeed offline even when everything is installed:
  `nnunetv2 → batchgenerators → unittest2 → argparse`, and pip never treats
  `argparse` as satisfied (the stdlib module shadows the distribution), so it
  re-downloads that 23 kB wheel on every start. With `&&`, a machine off the
  network had a server that started once and never again. It warns and
  continues, hard-gating on `python -c 'import fastapi, uvicorn'` instead;
  `server_ctl.py` greps for both markers. Verified with `--network none`.
- **Every operator in that command sits at the END of its line.** A YAML `>`
  folded scalar keeps its continuation newlines, so a line starting with `||`
  reaches `sh` as its own command: `sh: 2: Syntax error: "||" unexpected`, in a
  restart loop.
- **`wait_for_health` treats `restarting` as a failure.** `restart:
  unless-stopped` turns a boot failure into a loop, and a loop never becomes
  healthy — the caller used to sit out the full 30-minute timeout.
- **`DATA/` is created before docker can.** `./DATA:/data:ro` is a bind mount,
  so a missing host path is created by the daemon, owned by root, and every
  later download died on `Permission denied` on a brand-new install.
- **`update` is `up -d --force-recreate`, never `restart`**: the container
  installs `requirements.txt` in its command, into a layer `restart` keeps. It
  fast-forwards only and refuses to pull over uncommitted changes.
- **`--progress always` on `fetch_data.py`**: its `\r` progress was invisible
  to a reader collecting whole lines, so a 12 GB bundle printed nothing for an
  hour into the client's log pane.
- **`catalog`** reports, per tool, the manifest size *and* what a download would
  actually transfer, cross-referenced against what is on disk. Everything
  present is skipped, so re-running is both "resume" and "add one more tool".

Also: a root `.env.example` for the three variables compose interpolates
(distinct from `server/.env.example`, which documents what the application
reads), and `scripts/README.md` rewritten around the folder's two jobs.

### 2026-08-06 — ALI can be asked for named landmarks, which is what ASO needs

ASO's fully-automated CBCT mode calls ALI in-process and checks that its schema
exposes `("input", "model", "landmarks")`. ALI declared `cbct_regions` and
`ios_networks` and no `landmarks`, so the call failed on the contract check.

The name was the smaller half. ASO registers on seven points (Ba, S, N, RPo,
LPo, ROr, LOr) straddling the Cranial base and Upper regions, so asking by
region runs **58 agents to use 7** — and one agent is a full two-scale walk of
the volume. The engine always worked at landmark granularity internally; only
the schema was coarser.

- `landmarks` is a multichoice over all 119 catalog labels, **every option off
  by default** — unlike `cbct_regions`, whose options are all on. "All off" is
  what an omitted multichoice arrives as, so the default state means "nothing
  said here, the regions decide", which is what every earlier request keeps
  meaning.
- Naming a landmark **REPLACES** the region selection rather than narrowing it.
  Narrowing would agree for ASO only because it leaves the regions all on, and
  would silently drop landmarks for a caller that set both. The run report says
  which drove the run: `regions` is empty when `landmarks_selected` is not.
- The 119 options are readable because the schema says how to group them:
  `ui="tabs"` with `groups=LANDMARK_GROUPS`, which is `GROUP_LABELS` — the same
  table the engine names its output files by, published rather than restated,
  so a landmark added to it gets its tab with no client release. ALI also
  gained sections and a `label` on every argument.

**Tests:** 197 server tests (+3), including ASO's exact argument dict surviving
`tool.validate` with `input` as a resolved directory. Client-side, 34 ALI tests.

### 2026-08-06 — Presentation hints: the schema says how to lay a panel out

ASO's panel was unusable: four modes share one schema, so a generic client
rendered 130 CBCT landmarks, 32 teeth, 8 landmark types and 2 jaws as a single
column of ~180 check boxes with CBCT and IOS options interleaved, while any run
uses one half or the other. ALI has the same shape for a different reason: 119
landmark options and no `mode` field to hide the inert selection behind. The
old Slicer modules solved this with hand-written QStackedWidgets and ~700 lines
of checkbox plumbing, with the anatomy written inside the widget — exactly what
the ports removed.

**Five optional `ArgSpec` fields, published by `GET /tools`, ignored by
`validate()` and `run()`:** `label`, `section`, `visible_when`
(`{other_arg: value}`), `ui` (`"tabs"`/`"grid"`/`"inline"`) and `groups`. Every
one is `null` on a tool that declares none, so existing panels render unchanged.

- `label` closes the last thing the client was inventing. Labels were built
  client-side by two different rules in the same panel, so ASO showed
  "Reference" above "cbct_landmarks". No naming rule can produce "Scan /
  Landmark Folder" from `input`. Every user-visible word describing a tool is
  now the tool's; the client keeps only its own chrome.
- **None of the layout fields names an anatomical concept**: `groups` says what
  to group, `ui` how to lay it out, `visible_when` when it applies. ASO's
  `TOOTH_GROUPS` is derived from `TOOTH_IDS`, and ALI's tabs are
  `GROUP_LABELS`.
- The two tools use different amounts of it. ASO has a `modality` choice, so
  `visible_when` makes its two selections mutually exclusive; ALI has no such
  field on purpose, so it gets `section` only.
- `check_schema` rejects them at startup, and that matters more here than for a
  real type: a wrong `visible_when` hides a field for good, and a client cannot
  tell that from a field the tool never declared. An option no group mentions
  is *not* an error — the client renders the leftovers.
- `visible_when` is presentation, not validation: a hidden argument is not
  sent, so its declared default applies, and cross-argument checks still run
  for a direct API call. What it fixes is a real wire problem — a multichoice
  is read back as the complete `{option: checked}` dict, so a panel was sending
  the inert mode's selection, frozen at whatever the invisible widget held.

### 2026-07-31 — ALI's model bundle is matched to the detected mode; a wrong bundle is a 422

Found by running ALI IOS from Slicer with the dropdown left on
`ALI_CBCT_Models`: the IOS engine listed all 119 CBCT files as unrecognized and
Slicer showed `500 — The tool failed on the server`, the one message written
for the user buried in the log.

- **`model` is now optional and the mode picks it** (`ALILogic.select_bundle`).
  Each engine recognises its own bundle layout through `discover_weights`
  (`<landmark>/<scale>/*.pth` folders vs flat jaw/network-token checkpoints;
  mutually exclusive, file-name parsing only, so probing costs a directory walk
  and never a model load). No match is a 422 naming `setup-models.sh`; several
  matches is a 422 naming the candidates rather than a silent pick — which
  model vintage ran must never be a surprise. The report gains `model_bundle`.
  A temp copy materialized for a probe (`ResolvedFile.is_temporary`) is deleted
  whether or not it was picked.
- **A named-but-wrong bundle answers 422, not 500.** The five
  `FileNotFoundError`s the engines raised are `ToolArgumentError` now, so the
  message reaches Slicer verbatim. The two "not a directory" messages named the
  full server path; a 422 body travels, so they name the basename only.

**Client:** the dropdown of an optional scalar `server_selectable` argument
leads with an "(automatic)" entry whose item data is `""`, so the default
selection sends no `model` at all. Generic in `base_widget`/`formgen`.

**Tests:** 7 new server-side, plus an end-to-end CBCT run with no `model`.
Verified live: an IOS request with no `model` returned 200 in 17s with
`model_bundle: ALI_IOS_Models`.

### 2026-07-31 — 501 for "this server cannot do that", instead of a blank 500

Found by running ALI in IOS mode: the preflight raised immediately with a
message naming pytorch3d, and that message went to the server log while the
Slicer user got `500 — The tool failed on the server.`

500 hides its detail rightly: a crash inside a tool can name server-side paths.
A missing optional dependency is the opposite — the request was valid, nothing
the caller changes will help, and the reason names a package.

`base.ToolUnavailableError` plus a `501 Not Implemented` mapping in `main.py`.
Every dependency-import failure across `ALI`, `CrownSeg` and `AMASSS` raises it
(twelve sites), because the same condition answering 500 in one tool and 501 in
another is worse than either. **No client release needed:** `error_for_status`
shows the server's `detail` verbatim for any unmapped status.

### 2026-07-31 — Test files are downloadable: `GET /tools/{tool}/testfiles/{filename}`

The Slicer client grows a per-input "Test file" button filling a file input
with reference data. The hosted-name route runs a tool on a test file without
it travelling, but cannot put the file in the user's hands.

One Bearer-protected endpoint streams a test file by name, resolved through
`data_store.resolve_testfile` (so the backend abstraction and its traversal
checks apply; unknown name → 404). A folder entry is zipped on the fly into a
staging dir under `TEMP_DIR` and removed by background task once streamed; a
backend temp copy is likewise removed. **Test files only** — models are
selected by name and used in place. The log line carries tool, status, duration
and size, never the file name.

Also: `AMASSS`'s `input` is now `server_selectable="testfile"`. The client grays
its button off the actual `GET /tools/{tool}/data` listing, so an empty
`testfiles/` folder is a grayed button explaining itself, not a 404.

**Tests:** 401 without a token, 404 for unknown tool/file, a plain file streamed
with the right headers, a folder zipped with `TEMP_DIR` clean afterward, and an
`is_temporary` copy removed after streaming.

### 2026-07-31 — `monai` pinned: an unpinned entry was replacing the image's torch

Caught by reading a `pip install` log, not by a failing test. Adding `monai`
unpinned made every container start resolve `monai 1.6.0`, which requires
`torch>=2.8.0` — so pip downloaded `torch 2.13.0` plus the whole CUDA 13 stack
**over the image's `2.5.1+cu124`**, on every start. Three consequences, none of
which fail a test: ~3 GB per container start, `torchaudio 2.5.1+cu124` left
unsatisfiable, and the image's purpose-built CUDA torch shadowed by a wheel
that happened to still find the card here.

`monai==1.5.1` asks for `torch>=2.4.1`, which the image satisfies, so pip
leaves torch alone. Every transform and network ALI uses exists there. Move to
1.6 only together with an image rebuild to torch >= 2.8.

**The general rule:** an unpinned dependency can upgrade torch transitively.
When adding one, check its torch requirement against the image.

**And an operational one.** The `inference` service installs requirements as
part of its *command*, so a container up for days runs whatever
`requirements.txt` said when it last started — uvicorn's `--reload` picks up
new Python code but never re-runs pip. Worse, `pip --user` writes into the
container's writable layer, which `docker compose restart` keeps. After
changing `requirements.txt`:

    docker compose up -d --force-recreate inference

**A dependency failure is a run-level failure** (`check_dependencies()` in both
ALI engines). The missing `itk` surfaced through the per-scan `try/except`, so
it was reported as if one patient's data were at fault: every scan failed
identically, each only after a complete histogram correction, and the run ended
on "ALI produced no landmarks for any scan". Both engines now import their
whole lazy stack once, before the loop.

### 2026-07-31 — Real-data tests are opt-in; `test-gpu` service; ALI's GPU cap off 1

`inference` already runs on the GPU and every tool reads `settings.DEVICE`.
Nothing hardcodes a device; the one service deliberately on CPU was `test`.

- **`test-gpu`.** The unit tests stub every model and gain nothing from a card,
  but `tests/test_data_integration.py` runs each tool end to end against the
  real bundles — minutes on a GPU, hours on a CPU. A compose device reservation
  is all-or-nothing, so putting it on `test` would make the pre-push hook fail
  on any clone without a card. A second service instead, sharing everything
  through a YAML anchor. The hook keeps pointing at `test`.
- **The real-data suite is opt-in (`RUN_REAL_DATA_TESTS`).** It was written to
  skip when `DATA/` is empty, but the moment a full ALI bundle lands "skip"
  turns into eleven minutes of GPU inference, or hours on the CPU the hook
  uses. It now skips at collection and stays ~10s. A pre-push hook that takes
  hours is a hook people disable.
- **`ALI_MAX_GPU_JOBS` 1 → 4.** Measured on the real bundle (RTX 6000 Ada): an
  ALI CBCT run peaks at **256 MiB** of VRAM on a 48 GB card. At a limit of 1,
  two concurrent requests fully serialized for a resource neither was close to
  exhausting. The figure is a property of the models, not the card.
  `AMASSS_MAX_GPU_JOBS` stays at 1: a 3d_fullres nnUNet is a different order of
  magnitude and nothing here measured it.

### 2026-07-31 — ALI (both engines) + CrownSeg

Port `ALI` — automatic landmark identification — from a pair of Slicer CLI
modules. The first tool with *two* engines sharing nothing but their output
format, and the first whose IOS half depends on a library the image lacks.

**One tool, two engine folders (`tools/ALI/src/{ALI_CBCT,ALI_IOS}/`).** One
entry in `GET /tools`, one `DATA/ALI/`, one Slicer module. `ALILogic.py` owns
everything before inference (unpacking, DICOM conversion, mode detection, the
run report) so each engine only has to place landmarks. `src/markups/` holds
the Slicer `.mrk.json` writer both engines use.

**The mode is detected, not declared.** There is deliberately no `mode`
argument: a `.zip` can hold either kind of data and a DICOM series has no
extension, so only the data distinguishes them. An archive holding both kinds
is a 422 rather than a guess. The accepted cost is that the schema cannot say
"this argument only applies in mode X": `cbct_regions` and `ios_networks` are
both optional and one is inert on any run. Emptying the selection for the mode
that actually ran is a 422 naming the argument to fill in.

**`CrownSeg` is a tool, not a helper.** ALI's IOS engine needs a mesh carrying
per-tooth labels. The Slicer module got them by running the `dentalmodelseg`
executable out of Slicer's bin — which is only the console-script entry point
of the `shapeaxi` PyPI package, so nothing needed porting: `tools/CrownSeg/src/`
calls `shapeaxi.dental_model_seg.main()` directly. It lives in its own tool
because ASO, AREG and FlexReg call it too, and because ALI's IOS half needs
pytorch3d — inside ALI, one absent dependency would take four tools out of the
registry instead of one. `model` is optional there and falls back to
`settings.CROWNSEG_MODEL`; the library's own fallback downloads the checkpoint
from GitHub mid-request, and a server holding patient data does not make
outbound calls. shapeaxi's stdout is swallowed — it prints the patient's own
file name.

**Defects fixed by construction**, all of which lost results silently:

- **One unknown landmark cost the whole patient.** `LABEL_GROUPS[landmark]` was
  indexed with no guard inside the save loop, and its `KeyError` was caught far
  above — so nothing at all was written for that scan. The two spellings that
  triggered it (`UR3OI…` in the UI, `UR3OIP…` in the CLI) are aliases of one
  vocabulary now, and `group_of()` cannot raise.
- **Homonyms overwrote each other in batch.** The patient key was `file.name`,
  so two `scan.nii.gz` in different subfolders collided twice over. Scans are
  keyed by path relative to the input root, and the output mirrors the tree.
- **A missing mandibular IOS model was a silently-caught `KeyError`.** Reported
  now. The jaw must be named explicitly in the checkpoint's name: "not Lower ⇒
  Upper" meant a bundle missing its mandibular model quietly predicted the
  lower arch with the maxillary one. One naming rule replaces the two the UI
  and CLI disagreed on, verified against the published archive.
- **DICOM conversion wrote into the user's own folder** (`<input>/NIFTI/`),
  which the next run re-ingested as input scans. Everything goes to the request
  scratch dir, as does the segmentation CSV the module wrote into the
  extension's own source tree.
- **`.stl` was accepted then ignored**: the UI counted them, the CLI globbed
  for `.vtk` only. `surface_or_zip_file` (new `FILE_TYPES` entry — the only
  core edit) advertises exactly what discovery walks.
- **`R`, `RIP`, `OIP`** were selectable and predicted by nothing. Not offered.
- **`SaveId` was read by nothing**; `prediction_ID` is a real argument.
- **Output extensions disagreed** (`.mrk.json` vs `.json` for identical
  content, only the first of which Slicer recognises). Uniform, and one file
  per scan instead of one per region — the split forced every downstream tool
  to recombine them by hand.
- **`display.visibility: false`**, in both CLIs. It switches the markups
  *display* node off, so Slicer loads the file, builds the node and draws
  nothing. Invisible inside the old module, fatal the moment anyone opens a
  result file — which is what a returned archive is for. Two tests pin it.
- Two latent search bugs: `new_pos.all() > 0` reduced the array to one boolean
  *before* comparing, letting negative coordinates through; and `Focus`'s
  convergence loop had no bound, which in a worker thread is a request that
  never returns. The IOS masks were also argmax'd over logits cast to `int16`,
  turning near ties into real ones resolving toward the background channel.

**Sequencing:** the CBCT engine runs today on `monai` + `itk`. The IOS engine
and CrownSeg are written and tested but cannot execute until the base image is
rebuilt on torch ≥ 2.8 with pytorch3d compiled in — pytorch3d has no PyPI
distribution at all. Both are imported lazily, so ALI loads, publishes its
schema, and fails only an IOS *run*.

**Tests:** 37 for ALI, 20 for CrownSeg, no GPU, weights or network.

### 2026-07-31 — ASO ported: four modes, one tool, and the defects it inherited

Port ASO (Automated Standardized Orientation) from a 2587-line Slicer widget
plus four CLI modules. ASO is the step every longitudinal study runs before
anything else, and AREG needs it programmatically.

|          | Semi-Automated | Fully-Automated |
|---|---|---|
| **CBCT** | your landmarks, ICP onto a gold set | landmarks predicted first, then the same ICP |
| **IOS**  | your landmarks, ICP per jaw | tooth centroids of an already segmented mesh |

`modality` and `automation` are explicit `choice` arguments, never inferred: a
`.zip` can hold either kind of data, and guessing wrong orients a patient
against the wrong reference and calls it a success. Every mode-specific
argument is `required=False`, with cross-argument rules raised as
`ToolArgumentError` **before** any file is read.

The call into the landmark tool is **in-process, not HTTP to our own /run/ALI**:
a tool run holds one of `MAX_CONCURRENT_TOOLS` slots for its whole duration, so
four concurrent ASO runs each waiting on a fifth slot would deadlock the
server, `/health` included. `Tool.invoke` is the same entry point `main.py`
uses, validation included.

**Fully-Automated IOS takes already-segmented meshes only** (crown segmentation
is CrownSeg's job; `segment_unlabelled()` is where it plugs in). **No** labelled
mesh in the batch is a 422 (wrong mode); *some* unlabelled is a per-patient
report entry and the rest of the batch is processed.

**The defects that cost data**, each with a named test:

- **`SEMI_ASO_CBCT` could not work at all.** It read `data["tfm"]`
  unconditionally, but only the fully-automated chain produced one, so every
  semi-automated patient died on a `KeyError` caught 90 lines above.
  Recentring always runs now, and the landmarks are moved with it.
- **One landmark could lose a patient.** `GetDistDifference` indexed the
  reference's pairwise table with the *input's* keys. The two sides are
  intersected first and what was dropped is reported.
- **Patient keys collided.** `GetPatients` keyed on the base name, and stripped
  `_T1`/`_T2`, collapsing two timepoints into one patient.
- **`MergeJson` merged a patient's landmark files by writing into the caller's
  input folder and DELETING the sources.** The merge is in memory.
- **A second run re-ingested the first.** `patient1_Or.nii.gz` sorts before
  `patient1_scan.nii.gz`. Previous outputs are set aside and used only when a
  patient has nothing else.
- **`UpperOrLower` defaulted to Lower**, so a maxillary mesh named
  `patient1.vtk` was registered against the mandibular reference and returned
  as a success. A file whose name does not say its jaw is refused.
- **`Files_vtk_json.organise` paired with `vtk_name in json_name`**, so patient
  `1` matched patient `10` — and padded its list with a literal
  `"Upper_nioegfjhdfjkdffdhjmndfhnmdfhj"` sentinel. Exact stem, per directory.
- **Both jaws wrote the same `.tfm`.** Named per jaw now.
- **The published IOS reference was rejected outright.** Refusing a mesh whose
  name does not say its jaw is right, but the first version also required an
  identifier *before* the jaw token — and the published `Gold_file.zip` is
  `Upper_gold.vtk` / `Lower_gold.vtk`, jaw first. Found by reading the real
  archive rather than assuming its shape.

**Concurrency, which only matters because this is a shared server:**

- `InitIcp` wrote `source.npy`/`target.npy` into **its own installed package
  directory** and re-`np.load`ed one on every iteration of a 2500-iteration
  search — a write into the install tree, thousands of round trips per patient,
  and two concurrent requests overwriting each other's landmarks. The search is
  pure and in memory (`src/geometry.py`, shared by both engines, which had
  carried two drifted copies).
- The triplet search drew from the **global** numpy generator, so the same
  request gave a different orientation every time. Every ordered triplet is now
  enumerated when there are at most `ASO_ICP_MAX_TRIPLETS` of them (7 landmarks
  is 210) — deterministic, faster *and* better than sampling; above that a
  local generator seeded with `ASO_ICP_SEED` is used.

**Latent bugs found while reading, each now a test:** `np.arccos` of a dot
product rounding past 1.0 gave NaN propagating through the rotation matrix;
`RotationMatrix` divided by a zero-length axis for two parallel vectors;
`ASO_IOS_utils/utils.py` defined `PatientNumber` twice; `WriteSurf` used
`vtkPolyDataWriter`'s ASCII default; the `.off` reader referenced an undefined
`line`; and `SEMI_ASO_IOS` wrote every transform as `matrix_file_0.npy`.
**The IOS matrix composition order was also backwards** (`M_init @ M_icp`,
where the ICP runs on points the initialisation already moved); the CBCT engine
always had it right, which makes it a transcription slip.

**Also removed rather than ported:** `PRE_ASO_CBCT`'s `model_folder`, `SmallFOV`
and `temp_folder` arguments (read, never used), the `<filter-progress>` prints,
`sys.exit()`, `tqdm`, the `time.sleep(0.2)` progress theatre, the `*Error.txt`
files written into the output folder, the skip-if-exists guards, and the
reference *scan* the CBCT ICP read and never used — which it nonetheless
required, so a reference bundle holding only landmarks died on an `IndexError`.

**Core changes, both sanctioned:** one `FILE_TYPES` entry (`surface_file`) and
three `config.Settings` fields. No GPU: everything ASO computes is
SimpleITK/VTK/numpy. `main.py` and `registry.py` untouched.

**The two published references carry DISJOINT landmark sets**, and the schema's
defaults only match one: Frankfurt Horizontal + Midsagittal has
`Ba, S, N, RPo, LPo, ROr, LOr`; Occlusal + Midsagittal has
`ANS, IF, PNS, UL6O, UR1O, UR6O`. Picking the second and leaving the defaults
would drop every landmark as "not in the reference" and fail all forty patients
separately. `_check_selection_against_reference` turns that into a single 422
naming what the reference offers, raised after discovery but before a scan is
read.

**Tests:** 201 server tests, 71 new. Two of them cover the outputs a clinician
relies on, and both hold to the float: the written landmarks land exactly where
the resampling put the voxels (volume and markups move by two different code
paths — if they disagree the markups file opens floating beside its scan), and
the `.tfm` maps ORIENTED → ORIGINAL, recentring included. That direction is
asserted rather than assumed because getting it backwards is silent.

### 2026-07-30 — AMASSS surfaces: binary, and decimated by default

A five-structure run with surfaces returned a 41.9 MB archive Slicer could not
open — the client froze on the main thread and the user read it as "the server
never sent the .vtk". It had: `curl` pulled all 41,889,544 bytes and every mesh
re-read cleanly. The geometry was the problem.

- **Marching cubes runs on the original scan grid**, so a 0.33 mm CBCT produces
  a triangle per voxel face: 1.6M for a cranial base, 3.5M across five
  structures, 11.8M for a merged nine-structure volume. The mask underneath is
  only accurate to about half a voxel, so that is resolution nobody asked for.
- **`vtkPolyDataWriter` was writing ASCII** (its default): 848.5 MB for the
  merged surface against 6.4 MB for every segmentation in the same run. Binary
  is the same geometry and the *more* accurate of the two — it round-trips the
  float32 vertices exactly, while ASCII prints ~6 significant digits and moved
  points by up to 5e-05 mm on read-back. It is also 133x faster to parse (a
  1.6M triangle cranial base: 2.67s ASCII, 0.02s binary).
- **`surface_decimation` (new argument, default 90).** `vtkDecimatePro` with
  `PreserveTopologyOn`, applied after smoothing and *before* the per-cell colour
  array is built. Measured on the cranial base against a 0.33 mm voxel:

  | reduction | triangles | mean dev | p95 | max |
  |---|---|---|---|---|
  | 50% | 811,222 | 0.0034mm | 0.004mm | 0.277mm |
  | 80% | 324,488 | 0.0338mm | 0.125mm | 0.493mm |
  | **90%** | **162,244** | **0.0590mm** | **0.171mm** | 0.692mm |
  | 95% | 81,122 | 0.0951mm | 0.264mm | 1.223mm |

  90 costs a fifth of a voxel on average and buys a factor of ten. 0 keeps the
  raw mesh. The value is recorded in `AMASSS_report.json`, these surfaces being
  lossy by default now.

**End to end, real HTTP:** archive 41,889,544 → **5,417,443 bytes** (7.7x),
triangles 3,519,420 → **351,938** (10x), client-side mesh parsing 2.7s+ →
**0.01s**. Decimation adds ~12s of server time for five structures.

**A caveat worth keeping:** binary alone did NOT shrink the download. DEFLATE
was already squeezing ASCII at 6.2:1 and binary only compresses 2.7:1, so the
archive went 223.4 MB → 227.8 MB on a nine-structure run. Binary pays off in
disk, RAM, write time, zip time and parse time — not on the wire. Only removing
geometry moved the download.

**Still on the table, in the client repo:** `AMASSS.py`'s
`MAX_RESULTS_TO_LOAD = 12` caps by *file count* while the cost is in triangles.

**Tests:** 119 server tests (+4).

### 2026-07-30 — AMASSS: the GPU was idle seven eighths of the run

Profiling one structure on a 512x512x365 CBCT at 0.33 mm: **14.6s** resampling
the input to the model's 0.4 mm grid, **4.5s** of inference, **6.9s** resampling
the logits back. Both resamplings are scipy splines pinned to a single core.

**The tempting fix was measured and discarded.** At a 128³ patch the network
already saturates the SMs at batch 1: throughput is flat from batch 1 through
12. Cutting the GPU's work 5x (`tile_step_size` 0.5 → 1.0) moved the total from
37.3s to 34.5s and dropped utilisation from 36% to 11% — direct evidence the
GPU was never the constraint. **Free VRAM is not convertible into speed here;
idle time is.**

- **GPU resampling** (`nnunet_runner._enable_gpu_resampling`, new setting
  `AMASSS_GPU_RESAMPLING`, default on). nnUNet already ships torch versions of
  both resamplers, so nothing is reimplemented — only selected, and selected by
  NAME: nnUNet resolves them out of the configuration dict via
  `recursive_find_resampling_fn_by_name`. Two things make that mutation safe:
  `PlansManager` hands out a `deepcopy`, so it touches neither the shared plans
  nor a concurrent request — and consequently the `torch.device` never reaches
  the `plans.json` nnUNet writes, which `json.dump` could not serialize. Both
  properties are `@property @lru_cache`, so the swap clears them.
- The GPU path uses `predict_from_files_sequential`: `predict_from_files` fans
  preprocessing and export out to *spawned* processes, each of which would need
  its own CUDA context. That trades away the CPU/GPU overlap on multi-scan
  batches; recovering it with a reader thread is the obvious next step.

**Measured end to end, real models, five structures, one scan: 195.9s → 77.0s
(2.5x).** Per structure 34.2s → 13.2s.

**Not numerically free.** torch has no 3D cubic interpolation, so the input
resampling drops from spline order 3 to order 1. Dice against the scipy
pipeline: MAND 0.998, UAW 0.997, MAX 0.995, CB 0.991, **CV 0.978**. The
cervical vertebra is consistently the outlier — thinnest structure, closest to
the edge of the field of view. `AMASSS_GPU_RESAMPLING=false` restores
bit-identical output, and a bundle whose plans pin a non-default resampler opts
itself out. `AMASSS_report.json` records `gpu_resampling` and `tile_step_size`.

**`AMASSS_TILE_STEP_SIZE` (default 0.5, unchanged).** The one knob here that
moves the segmentation for a *pure* speed gain, so it is exposed rather than
tuned: 0.7 measured Dice 0.995 against 0.5 and saves ~2.5s of GPU per structure.

**`_convert_to_nifti` stopped casting to float32.** The cast was never what made
the conversion real, and it doubled the bytes gzipped per scan and gunzipped
again by nnUNet, costing 2.4s + 0.4s for nothing.

**`vtk_export` cleanups.** Cell colours are built in numpy instead of a
`SetTuple` per cell. The marching-cubes temp file had a *fixed* name, so every
surface in a run wrote over the same path; it is unique per call now and
removed after use.

**Tests:** 115 server tests (+5).

### 2026-07-30 — Dead-code and duplication cleanup

- `base.py` had an entire block (`FOLDER_TYPE`, `SCALAR_TYPES`, `CHOICE_TYPES`,
  `Selection`, `ResolvedPath`) declared TWICE, plus the remnants of the retired
  `SELECTION_TYPE` API. All removed. **The current API is
  `"choice"`/`"multichoice"`** (one `choices` dict of option → default).
- `main.py._describe_argument` (unused, referenced the removed fields) and
  `file_utils.zip_directory` (unused duplicate of `make_zip`) removed.
- `requirements-amasss.txt` removed: torch/nnunetv2/vtk had been added to
  `requirements.txt`, leaving the file a pure duplicate. The heavy stack stays
  lazily imported, and torch stays unpinned so the image's CUDA build is never
  shadowed.
- The `test` service in `docker-compose.yml` is back under a
  `profiles: ["test"]` guard.
- Docs realigned with the code: `claude.md` → `CLAUDE.md`, `surgMovPred` →
  `SurgMovPred`, README's selection-argument section rewritten for
  `choice`/`multichoice`, `ADDING_A_TOOL.md` §7 now describes the real
  requirements layout.

### 2026-07-28 — AMASSS tool + grouped selection arguments

Port `AMASSS_CLI.py` (CBCT skull structure segmentation, nnUNet v2). The first
tool that is genuinely an *API* — AREG already calls it programmatically — the
first to need an argument the schema could not express, and the first with a
GPU deep-learning stack.

**Core additions (`base.py`, `main.py`):**

- Choice arguments: `ArgSpec.choices`, with the `"choice"` (exactly one) and
  `"multichoice"` (any number) types. One `{option: on by default}` dict
  declares the options **and** their initial state, so a client renders the
  widget straight from `GET /tools` and the defaults are written down once.
  Accepted on the wire as `"MAND,MAX"` or `{"MAND": true}`; an invalid option
  is a 422 naming what is allowed. This is a change to the *type system*, made
  once — the same category as adding a `FILE_TYPES` entry.
- `FILE_TYPES["volume_or_zip_file"]`: one argument accepting either a single
  volume or a zip of a folder of them, since the schema cannot express "exactly
  one of these two arguments".
- `file_utils.zip_directory`, the counterpart of `extract_zip`.

**The tool (`tools/AMASSS/`):** `AMASSS.py` declares the schema;
`src/AMASSSLogic.py` holds the pipeline; `src/nnunet_runner.py` isolates
inference; `src/vtk_export.py` handles surfaces. `segment()` is the reusable
API; `main()` is the thin HTTP adapter.

Three defects mattered specifically *because* this is a shared server:

- the CLI set `os.environ['nnUNet_results']` before shelling out. Tools run
  concurrently in worker threads and `os.environ` is process-global, so two
  overlapping requests would have silently swapped model paths.
  `initialize_from_trained_model_folder` takes an explicit path.
- the CLI polled the output file's size and killed the predictor once it
  stopped growing for 3s, which could interrupt nnUNet mid-postprocessing.
- GPU work is serialized by AMASSS's own semaphore (`AMASSS_MAX_GPU_JOBS`,
  default 1), independently of `MAX_CONCURRENT_TOOLS`.

Also corrected: NRRD/GIPL inputs are really converted via SimpleITK instead of
being renamed; folder scanning is recursive and excludes previous outputs;
structures with no shipped model are no longer offered; a missing or failed
structure is reported in `AMASSS_report.json`; `sys.exit()` is gone. Inference
loads each checkpoint once per structure instead of once per (scan × structure).

**Dependencies:** `numpy` + `SimpleITK` in `requirements.txt`;
`torch`/`nnunetv2`/`vtk` imported **lazily**, since `registry.py` imports every
tool at startup and a missing heavy stack must not stop the server booting.

**Tests:** 35 with `nnunet_runner.predict_folder` stubbed, so no GPU and no
real models are needed.

### 2026-07-27 — Parallel request handling (threadpool execution of tools)

`run_tool` called `tool.invoke(args)` synchronously inside an `async def`
endpoint, i.e. directly on the uvicorn event loop. Any inference in progress
froze the entire server — a second `/run`, or even `/health`, could not be
answered until it finished.

`tool.invoke` now runs via `anyio.to_thread.run_sync(...)` in a worker thread,
with a dedicated `anyio.CapacityLimiter` capping simultaneous executions at
`MAX_CONCURRENT_TOOLS` (new setting, default 4). The limiter is created lazily
(anyio needs a running event loop) and is dedicated to tool runs so queued
inference cannot starve the default threadpool. Safe because tools are
stateless, each request has its own `work_dir`, and `DATA_DIR` is read-only.
The HTTP contract is unchanged.

**Test:** a probe tool whose `run()` blocks on a 2-party `threading.Barrier`,
fired from two requests through ONE shared event loop (`TestClient` as a
context manager — two bare `client.post` calls from separate threads would each
get their own loop and pass even against a serial server).

### 2026-07-27 — SurgMovPred: the model is server-side only, selected by name

The model should live exclusively in the server's data store: the client asks
for the list (`GET /tools/SurgMovPred/data`) and sends only the *name*.

The `model` argument changed from `ArgSpec(type="zip_file",
server_selectable="model")` to `ArgSpec(type=str, server_selectable="model")`.
The resolution path is unchanged — `main.py` already resolves any
`server_selectable` argument sent as a form value — but the contract is: a
scalar type means "name only". To enforce it, `main.py` rejects with a 400 any
file *upload* targeting a non-file-typed argument, which previously would have
passed the temp path through as the argument's string value.

**Tests:** an upload for `model` is a 400; an unknown model name is a 404; a
synthetic str-typed `server_selectable` argument resolves through `data_store`.

### 2026-07-27 — Pre-push test gate + real-data integration tests

The suite only ran on synthetic fixtures and only when someone remembered to
invoke it. A new `docker-compose.yml` service, `test`, runs the same image as
`inference` without its GPU reservation (`docker compose run --rm test`). A git
hook, `.githooks/pre-push`, runs it before every push; it is opt-in per clone
via `git config core.hooksPath .githooks` and bypassable with `--no-verify`.

`server/tests/test_data_integration.py` complements the synthetic tests: for
every tool whose required arguments are all `server_selectable`, it looks up
real files via `data_store` and runs the tool end to end. `DATA/` is gitignored,
so a tool with no matching file is **skipped**, never failed.

### 2026-07-24 — Server-side data store: models and test files without re-upload

Tools like `SurgMovPred` required the client to re-upload the same model on
every call, and there was no way to say "run this against the server's
reference data". Confidential-data constraints rule out a generic upload cache,
so this is explicit, per-tool, read-only server-side storage.

`server/data_store.py` introduces a `DataStore` interface with a
`LocalDataStore` reading `DATA_DIR/<tool_name>/{models,testfiles}/`. `ArgSpec`
gained `server_selectable` (`"model"` | `"testfile"`); `GET
/tools/{tool_name}/data` lists what is available. In `POST /run/{tool_name}`, a
`server_selectable` argument sent as a plain form value is resolved through
`data_store` and excluded from the temp-file cleanup that applies to uploads.

**Deliberately abstracted for a future external database/object store:**
neither `main.py` nor any `Tool` touches the filesystem directly. Each
`resolve_*` returns a `ResolvedFile(path, is_temporary)`; `is_temporary` lets a
future backend mark a materialized temp copy for cleanup, while
`LocalDataStore`'s persistent paths are never deleted. Swapping backends is
contained entirely to `data_store.py`.

Also: `docker-compose.yml` now mounts a single `./DATA:/data:ro` (previously two
inconsistent mounts), and `.gitignore` excludes `DATA/`.

### 2026-07-24 — Correct `Content-Type` for file-kind tool outputs

`POST /run/{tool_name}` responses with `output_kind in ("file", "segmentation")`
always sent `application/octet-stream` (or `application/gzip`), regardless of
the real format. An `.xlsx` is internally a zip container, so a client deciding
whether to unzip by sniffing magic bytes could not tell it from a real archive
— it silently extracted the Excel file's internal XML parts.

The `FileResponse` now derives `media_type` from the extension via
`mimetypes.guess_type()`, falling back to the previous logic only when the type
cannot be guessed (still the case for bare `.gz` files, e.g. `.nii.gz`). This
also fixes `.zip`, `.csv` and `.ods`.

**Client-side follow-up (not in this repo):** a client deciding whether to unzip
by sniffing magic bytes must trust `Content-Type`/`Content-Disposition`
instead — sniffing can never distinguish a real `.xlsx`/`.docx`/`.pptx` from an
actual zip archive, those formats being zip containers by design.
