"""Does the packaged tool do what the imported one did?

A tool exists twice during the migration: as a `Tool` subclass this server
imports (`server/tools/<Name>/<Name>.py`), and as a folder it never imports
(`<TOOLS_DIR>/<name>/{.schema.json,.venv,src}`). Before the first is deleted,
something has to have compared them on real data. This is that something.

    python parity.py --imported AMASSS --args case.json

It runs both, and compares what a caller would actually receive:

- the **artifacts**: every file produced, keyed by name and hashed. Absolute
  paths are meaningless across the two runs (different job directories) and
  are never compared;
- the **result**, with any path in it replaced by the artifact it points at,
  so `{"outputs": {"mandible": "/jobs/ab12/output/x.nii.gz"}}` and the
  in-process run's `/tmp/inference_server/tool_9f/x.nii.gz` compare equal.

**A difference is not automatically a failure**, and this script does not
pretend otherwise. The packaged tool runs against its own pinned dependencies:
a different numpy can move a voxel, a newer SimpleITK can write a different
header. What it does is make the difference visible, per file, so it is looked
at instead of discovered by a clinician. It exits non-zero on any difference.

Not a unit test: it needs the real models under DATA_DIR, the real
virtualenvs, and it takes as long as the tool takes. `tests/test_parity.py`
exercises the comparison itself, on a fixture that runs in milliseconds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import file_utils
from base import Tool
from config import settings
from registry.deployment import deployment_config
from registry.schema_tool import load_tool

# Keys whose value differs between two runs of the SAME code and says nothing
# about the port: how long it took, when it ran, where it ran.
DEFAULT_IGNORED_KEYS = ("duration", "duration_seconds", "elapsed", "timestamp", "date", "job_id")

_READ_CHUNK = 1024 * 1024


@dataclass
class RunSnapshot:
    """What one run produced, in a form that survives being run somewhere else."""

    result: Any
    artifacts: dict = field(default_factory=dict)  # {name: sha256}
    sizes: dict = field(default_factory=dict)  # {name: bytes}


@dataclass
class ParityReport:
    identical: list = field(default_factory=list)
    differing: list = field(default_factory=list)
    only_imported: list = field(default_factory=list)
    only_packaged: list = field(default_factory=list)
    result_imported: Any = None
    result_packaged: Any = None

    @property
    def results_match(self) -> bool:
        return self.result_imported == self.result_packaged

    @property
    def ok(self) -> bool:
        return (
            not self.differing
            and not self.only_imported
            and not self.only_packaged
            and self.results_match
        )


# ----------------------------------------------------------------------
# Building each side
# ----------------------------------------------------------------------

def imported_tool(folder: str) -> Tool:
    """Instantiate the `Tool` subclass in `server/tools/<folder>/<folder>.py`.

    Loaded by path rather than taken from the registry on purpose: while both
    forms of a tool exist, they share a name, and the registry keeps exactly
    one of them (the schema wins). Comparing them means holding both at once.
    """
    name = os.path.basename(folder.rstrip(os.sep))
    module_path = os.path.join(folder, f"{name}.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"No imported tool at {module_path}")

    # Imported under the package name it expects, so its own relative imports
    # (`from .src import ...`) resolve.
    module_name = f"tools.{name}.{name}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        specification = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)

    for attribute in vars(module).values():
        if isinstance(attribute, type) and issubclass(attribute, Tool) and attribute is not Tool:
            return attribute()
    raise LookupError(f"{module_path} defines no Tool subclass")


def packaged_tool(folder: str) -> Tool:
    """Build the tool declared by `<folder>/.schema.json`, importing nothing."""
    return load_tool(folder, deployment_config)


# ----------------------------------------------------------------------
# Running and snapshotting
# ----------------------------------------------------------------------

def _digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(paths: list) -> tuple:
    """{name: sha256}, {name: size} for everything the run produced.

    A directory contributes every file under it, keyed by its path relative to
    that directory; a file contributes its base name. Which is what makes the
    two runs comparable at all -- one wrote into a job directory, the other
    into a scratch directory, and neither name means anything to a caller.
    """
    artifacts = {}
    sizes = {}
    for path in paths:
        if os.path.isdir(path):
            for dir_path, _, file_names in os.walk(path):
                for file_name in sorted(file_names):
                    absolute = os.path.join(dir_path, file_name)
                    name = os.path.relpath(absolute, path)
                    artifacts[name] = _digest(absolute)
                    sizes[name] = os.path.getsize(absolute)
        elif os.path.isfile(path):
            name = os.path.basename(path)
            artifacts[name] = _digest(path)
            sizes[name] = os.path.getsize(path)
    return artifacts, sizes


def _normalize(value, roots: list, ignored_keys: tuple):
    """Replace absolute paths with what they point at, and drop noisy keys.

    Every path in a result is specific to the run that produced it. What a
    caller cares about is which artifact it names, so that is what is compared.
    """
    if isinstance(value, dict):
        return {
            key: _normalize(item, roots, ignored_keys)
            for key, item in value.items()
            if key not in ignored_keys
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, roots, ignored_keys) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        for root in roots:
            if value == root:
                return "<output root>"
            if value.startswith(root.rstrip(os.sep) + os.sep):
                return f"<artifact:{os.path.relpath(value, root)}>"
        return "<path>"
    return value


def run_once(tool: Tool, arguments: dict, ignored_keys: tuple = DEFAULT_IGNORED_KEYS) -> RunSnapshot:
    """Invoke a tool and snapshot everything it produced.

    Scratch directories are tracked exactly as a request would track them, so
    what a tool writes is found whichever side it ran on -- and cleaned up
    afterwards, since a parity run on a cohort produces as much data as a real
    one.
    """
    scratch_dirs = file_utils.track_scratch_dirs()
    try:
        result = tool.invoke(dict(arguments))
        paths = file_utils.output_paths(result)
        artifacts, sizes = _collect(paths)
        roots = [path if os.path.isdir(path) else os.path.dirname(path) for path in paths]
        return RunSnapshot(
            result=_normalize(result, roots, ignored_keys), artifacts=artifacts, sizes=sizes
        )
    finally:
        for directory in scratch_dirs:
            shutil.rmtree(directory, ignore_errors=True)


def compare(imported: RunSnapshot, packaged: RunSnapshot) -> ParityReport:
    report = ParityReport(result_imported=imported.result, result_packaged=packaged.result)
    for name, digest in sorted(imported.artifacts.items()):
        if name not in packaged.artifacts:
            report.only_imported.append(name)
        elif packaged.artifacts[name] == digest:
            report.identical.append(name)
        else:
            report.differing.append(name)
    report.only_packaged = sorted(set(packaged.artifacts) - set(imported.artifacts))
    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_report(report: ParityReport, imported: RunSnapshot, packaged: RunSnapshot) -> None:
    print(f"\nIdentical artifacts : {len(report.identical)}")
    print(f"Differing artifacts : {len(report.differing)}")
    print(f"Only imported       : {len(report.only_imported)}")
    print(f"Only packaged       : {len(report.only_packaged)}")

    for name in report.differing:
        before, after = imported.sizes.get(name, 0), packaged.sizes.get(name, 0)
        delta = "same size" if before == after else f"{before} -> {after} bytes"
        print(f"  DIFFERS  {name}  ({delta})")
    for name in report.only_imported:
        print(f"  MISSING  {name}  (the packaged tool did not produce it)")
    for name in report.only_packaged:
        print(f"  EXTRA    {name}  (the imported tool did not produce it)")

    if not report.results_match:
        print("\nThe returned value differs:")
        print(f"  imported: {json.dumps(report.result_imported, sort_keys=True)[:600]}")
        print(f"  packaged: {json.dumps(report.result_packaged, sort_keys=True)[:600]}")

    if report.ok:
        print("\nOK: the packaged tool produced exactly what the imported one did.")
    else:
        print(
            "\nNOT IDENTICAL. This is not automatically a defect -- the packaged tool runs "
            "against its own pinned dependencies, and a different numpy moves voxels -- but "
            "every line above has to be looked at before the imported tool is deleted."
        )



def _packaged_folder(name: str) -> str:
    """`<TOOLS_DIR>/<name>`, or `<TOOLS_DIR>/<group>/<name>` when it is nested.

    The fifth place this flat-depth assumption was found. Harmless here in a way
    it was not elsewhere -- this is a command-line comparison tool, and
    `--packaged` already lets a caller name the folder outright -- but leaving
    one site behind is how the assumption survived three rounds of fixing it.
    """
    direct = os.path.join(settings.TOOLS_DIR, name)
    if os.path.isdir(direct):
        return direct
    try:
        groups = sorted(os.listdir(settings.TOOLS_DIR))
    except OSError:
        return direct
    for group in groups:
        nested = os.path.join(settings.TOOLS_DIR, group, name)
        if os.path.isdir(nested):
            return nested
    return direct


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imported", required=True, help="Folder name under server/tools/")
    parser.add_argument(
        "--packaged",
        help="Folder name under TOOLS_DIR. Defaults to the imported name, lowercased.",
    )
    parser.add_argument("--args", required=True, help="JSON file of arguments for both runs")
    parser.add_argument(
        "--packaged-args",
        help="JSON file of arguments for the packaged run, when the two schemas differ "
        "(a multichoice becomes a list[str], for instance).",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Result key to leave out of the comparison; repeatable.",
    )
    arguments = parser.parse_args(argv)

    imported_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tools", arguments.imported
    )
    packaged_folder = _packaged_folder(arguments.packaged or arguments.imported.lower())
    ignored = tuple(DEFAULT_IGNORED_KEYS) + tuple(arguments.ignore)

    with open(arguments.args) as handle:
        imported_args = json.load(handle)
    if arguments.packaged_args:
        with open(arguments.packaged_args) as handle:
            packaged_args = json.load(handle)
    else:
        packaged_args = imported_args

    print(f"imported: {imported_folder}")
    print(f"packaged: {packaged_folder}")

    print("\nrunning the imported tool ...")
    imported = run_once(imported_tool(imported_folder), imported_args, ignored)
    print("running the packaged tool ...")
    packaged = run_once(packaged_tool(packaged_folder), packaged_args, ignored)

    report = compare(imported, packaged)
    _print_report(report, imported, packaged)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
