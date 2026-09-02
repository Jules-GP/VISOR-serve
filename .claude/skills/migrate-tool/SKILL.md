---
name: migrate-tool
description: Migrate a Slicer module into a packaged SADT tool, or split an existing tool whose two engines have incompatible dependencies, then update the Slicer client module that calls it. Use when porting a tool from SlicerAutomatedDentalTools, when a tool needs its own pinned torch, or when adding a tool to sadt-tools.
---

# Migrating a tool

A tool is a folder with its own virtualenv. The server never imports it: it
reads the tool's schema, validates the request, and runs the tool's own Python
as a subprocess. Everything below follows from that.

## 1. Read upstream before writing anything

Open the upstream module and **enumerate** what it declares. Do not filter with
a regex you wrote from memory: parameter names mix case (`DCMInput`,
`lm_type`), and a pattern like `[a-z_]+` silently drops half of them.

```bash
# method classes, when there are any
grep -h "^class " <upstream>/<Tool>_Method/*.py

# CLI parameters, when it is a CLI module
grep -oE "<name>[^<]+</name>" <upstream>/<Tool>_CLI/<Tool>_CLI.xml
```

Set that list against what the port exposes and sort every difference into three
piles: ported, deliberately dropped with the reason written down, and missing by
oversight. Only the third is work. The first two exist to isolate it.

## 2. One tool, or several?

Split when the engines have **incompatible dependencies**, not when the code
looks separable. The test is concrete:

```bash
grep -rhoE "^\s*(import|from) (torch|pytorch3d|monai|itk|vtk)[a-z0-9_.]*" <engine>/*.py \
  | awk '{print $2}' | cut -d. -f1 | sort -u
```

If one engine needs pytorch3d (and therefore an exact torch) and the other does
not, a shared virtualenv forces one pin onto the other. That has been measured
to move results: bumping ALI_CBCT to its sibling's torch shifted a landmark by
4 voxels and dropped two others entirely.

Split into `tools/<Family>/<Family>_<ENGINE>/`, with `tools/<Family>/` holding
no `pyproject.toml` of its own.

## 3. What goes in a shared `common/`

**Duplicate the implementation, share the formats.** Two copies of an algorithm
cost a divergence; a coupling costs an entire class of failure, and the copy
usually wins. The exception is anything defining the shape of bytes leaving the
repository, or a convention both engines must agree on:

- output file formats (a Slicer `.mrk.json` writer);
- input vocabularies (which extensions are volumes, which are surfaces);
- identity derivation (how a patient key is read from a filename).

A shared package declares **`dependencies = []`** and must keep doing so: it
installs into environments whose pins are deliberately incompatible.

Never share anything containing `sup.run()`. `describe.py` derives the schema's
`calls` field by reading each tool's own `src/`, and the server refuses to start
on a call naming an unserved tool. Shared orchestration is invisible to both.

## 4. Write `run()`

One public callable. Annotations are stdlib only: `Path`, `str`, `int`, `float`,
`bool`, `Literal[...]`, `list[...]` of those. No `Optional`, no unions. No
default means required.

Heavy imports go **inside** `run()` or the engine, never at module level: CI
imports the package on every PR to publish the schema, and that must not cost a
CUDA stack.

If the tool needs another tool, declare `*, sup=None` -- keyword-only and
unannotated, which is what keeps it out of the published schema.

## 5. Declare it

```toml
[tool.sadt]
tool = true
name = "Crown_Seg"
```

The section is what makes the directory a tool. The name is the API identity:
what a client sends, what `deployment.toml` is keyed by, what `sup.run()` names.
The directory must still be named after the tool, because the interpreter is
looked up by name.

Pin sources explicitly, never as extra index URLs:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

**A `[tool.uv.sources]` entry only applies to a DECLARED dependency.** Left
transitive it is ignored, and the package comes from PyPI built against a
different torch: it imports fine and dies on the first CUDA kernel.

## 6. Verify, in this order

Each step catches what the previous one cannot.

```bash
# 1. it resolves
cd tools/<Tool> && uv lock

# 2. it installs and imports
uv sync && .venv/bin/python -c "import sadt_<tool>"

# 3. no name is undefined on a path only a run reaches
uv run --no-project --with pyflakes -- python -m pyflakes src

# 4. the schema generates, with the tool's own interpreter
.venv/bin/python ../../scripts/describe.py .

# 5. it runs, on real data, and the result is compared
```

Step 3 is not optional. An import proves the module loads and says nothing about
a name inside a branch only a real run reaches -- four such defects survived
import AND schema generation in one split.

Step 5 is what "done" means. Run it directly in its venv, then through
`POST /run/{tool}`, and diff the two. State what matched exactly and what varied
within a measured tolerance. If the tool is non-deterministic, measure its
spread by running it twice on the same input: the reference is that spread, not
zero.

## 7. After a split, check the callers

A rename breaks every `sup.run()` naming the old tool, and the failure is at run
time.

```bash
grep -rn "sup.run(" tools/*/src tools/*/*/src
```

Also check `deployment.toml`: a tool's data folder is looked up by name, so a
split tool no longer matches `DATA/<old name>/` and needs `data_dir`.

And the Slicer module, which calls the tool by name and is the caller a
clinician sees fail. A split renames by definition, so section 8 is not
optional after one.

## 8. Update the client

`GET /tools` builds the panel, so a migrated tool renders with no client work at
all. Four things are **not** derived from the schema, and each of them fails at
run time rather than at build time.

The client is `SlicerAutomatedDentalToolsCloud`. A module is
`<Name>/{<Name>.py, CMakeLists.txt, Resources/Icons/<Name>.png, Testing/}`, and
its own `ARCHITECTURE.md` describes the machinery -- what follows is only what a
migration has to touch.

**The name the module calls.**

```bash
grep -rn '^\s*TOOL_NAME = "' <cloud> --include='*.py'
```

Each must name a tool still served. A pure respelling is already absorbed:
`client._canonical_tool_name` matches on case and separators, which is why
`SurgMovPred` still finds `Surg_Mov_Pred`. A split is not absorbed -- a module
holding `TOOL_NAME = "AREG"` after AREG became `AREG_CBCT`/`AREG_IOS`/
`AREG_IOSCBCT` shows "Unknown tool", which is what a typo shows.

**The module count.** One upstream module becoming N tools is a decision, not a
consequence: either N modules, or one module over the tool that dispatches. A
new module also needs `add_subdirectory(<Name>)` in the **root**
`CMakeLists.txt` -- without it the module is written, committed and never ships
(`MedX` sits commented out there today).

**The three overrides**, declared on the widget for the things a schema cannot
state:

| | declare one when |
|---|---|
| `FILE_INPUTS` | the picker the type implies is the wrong one -- a `zip_file` the user holds as a folder, a volume that should come from the scene, an optional file argument to leave out |
| `RESULT_KIND` | `output_kind: "file"` says a file comes back, not whether to load it into the scene (`volume`/`model`) or offer to save it (`save_as`) |
| `TEST_DATA` | the original extension published test data; the URL puts a "Test data" button on that argument's row |

A wrong `RESULT_KIND` is a 200 that displays the wrong thing, which no status
code catches: AMASSS loaded its segmentation as a surface-rendered model until
it was fixed.

**The test.** `<Name>/Testing/Python/test_<name>_client.py`, driving
`ServerToolsCoreLib` against the tool's own `GET /tools` schema as a fixture
with `qt`/`ctk`/`slicer` stubbed, and registered as a plain Python3 ctest in
the `CMakeLists.txt` beside it -- no Slicer interpreter launched.
Assert the module's *declarations*, not `<Name>.py` itself: importing it needs a
real Slicer, and stubbing that far means the test measures the stub.
`ALI/Testing/Python/` is the shape to copy.

## Commits

One sentence saying what the commit does, prefixed `ADD:`, `FIX:`, `DEL:`,
`UPDATE:` or `CLEAN:`. Several commits rather than one large one. No AI
attribution trailers.
