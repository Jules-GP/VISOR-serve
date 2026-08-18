#!/usr/bin/env python3
"""The half of a tool run that happens inside the tool's own interpreter.

    /tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<id>/job.json

Read the job file, import the tool, call its run(), write result.json. That is
the whole job -- plus one thing: a tool declaring `*, sup` is handed a
**supervisor**, and can call another tool through it. That call re-enters this
same file with the sibling's interpreter, so chaining and nesting are one
recursion rather than a feature.

Three constraints shape this file, and all three are load-bearing:

- **Standard library only.** It runs inside a TOOL's virtualenv, which contains
  whatever that tool needs and nothing of the server's. It cannot import
  fastapi, pydantic, or anything else from server/.
- **Python 3.9 through 3.13.** Each tool pins its own interpreter, so this file
  is executed by all of them. No match statements, no `X | Y` annotations
  outside the `from __future__` below.
- **It ships with the SERVER, not with the tools, and is injected by path.** It
  is deliberately not a package installed into each venv: runner and server are
  then always the same version, and there is no cross-repo version skew to
  negotiate.

On failure it exits non-zero, prints the traceback to stderr, and writes the
exception's CLASS NAME into the result file -- `{"error": {"type": ...}}`.
There is no shared exception type to catch, because there is no shared package,
so the name is what tells the server whether the caller sent something wrong
(422) or the tool itself broke (500). A successful result is written whole and
replaced into place, so a half-written one is never read as a result.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import typing
from pathlib import Path

# Where the tool's code lives inside its folder, and the environment variable
# that overrides the folder itself (tests and dev checkouts, where the venv is
# not necessarily next to the sources).
SRC_DIR_NAME = "src"
TOOL_DIR_ENV = "SADT_TOOL_DIR"

RESULT_FILE = "result.json"
JOB_FILE = "job.json"

# The supervisor: how a tool calls ANOTHER tool. Keyword-only and unannotated,
# which is what `describe.py` reads to keep it out of the published schema -- a
# client never sends one, because it is not data.
SUPERVISOR_ARGUMENT = "sup"

# Guards a tool that names itself, directly or through a cycle. Five is past
# anything real: the deepest chain today is AREG_IOSCBCT -> ASO -> ALI_CBCT.
SUPERVISOR_DEPTH_ENV = "SADT_SUPERVISOR_DEPTH"
MAX_SUPERVISOR_DEPTH = 5

# The tools already on the stack, innermost last, as one comma-separated value.
# It travels in the environment rather than in job.json because it belongs to
# the CALL, not to the job: the same tool run directly and run as a child reads
# a different chain, and job.json is what a caller writes.
SUPERVISOR_CHAIN_ENV = "SADT_SUPERVISOR_CHAIN"

# The tool's package under src/, when there is exactly one. sadt-tools names it
# `sadt_<tool>`, but the rule is the one its own describe.py uses -- the single
# importable package -- rather than the prefix, so a tool that names its
# package something else still loads.
_FALLBACK_PREFIX = "sadt_"


class RunnerError(Exception):
    """Anything that stops this script before the tool's run() is reached."""


def _tool_dir() -> str:
    """The tool's folder: /tools/<name>/, holding src/ and .venv/.

    Derived from the interpreter we are running in rather than from the job
    file, because that is the one thing the invocation already fixes: the
    server picked this venv precisely because it is the tool's. Deriving it
    keeps job.json to the four fields the contract declares.
    """
    override = os.environ.get(TOOL_DIR_ENV)
    if override:
        return os.path.abspath(override)

    if sys.prefix == sys.base_prefix:
        raise RunnerError(
            "This interpreter is not a virtualenv, so the tool folder cannot be derived "
            "from it. Run the tool's own /tools/<name>/.venv/bin/python, or set "
            f"{TOOL_DIR_ENV}."
        )
    # <tool dir>/.venv/bin/python -> sys.prefix is <tool dir>/.venv
    return os.path.dirname(os.path.abspath(sys.prefix))


def _package_under(src_dir: str):
    """The one importable package under src/, or None.

    Same rule as sadt-tools' own describe.py: exactly one directory holding an
    __init__.py. Deriving it rather than hardcoding `sadt_<tool>` means the
    runner and the schema generator agree on what they load, which is the only
    way the schema can describe what actually runs.
    """
    candidates = sorted(
        entry
        for entry in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, entry, "__init__.py"))
    )
    return candidates[0] if len(candidates) == 1 else None


def _import_tool(tool_name: str, src_dir: str):
    """Import the module defining run(), from the tool's src/.

    `uv sync` installs the tool into its own virtualenv, so the package is
    importable without any of this; src/ goes on the path first anyway, so the
    runner also works against a checkout that was never synced.
    """
    if not os.path.isdir(src_dir):
        raise RunnerError(f"Tool '{tool_name}' has no '{SRC_DIR_NAME}/' directory at {src_dir}")

    sys.path.insert(0, src_dir)
    package = _package_under(src_dir)
    candidates = [name for name in (package, f"{_FALLBACK_PREFIX}{tool_name}", tool_name, "tool") if name]

    tried = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.isidentifier():
            tried.append(f"{candidate} (not a valid module name)")
            continue
        try:
            module = importlib.import_module(candidate)
        except ImportError as exc:
            # Only a MISSING module moves on to the next candidate. An
            # ImportError raised from inside the module (a missing dependency
            # of the tool itself) is the answer, not a reason to keep looking.
            if getattr(exc, "name", None) != candidate:
                raise
            tried.append(f"{candidate} ({exc})")
            continue
        if package is not None:
            _reject_module_from_elsewhere(module, candidate, src_dir)
        return module

    raise RunnerError(
        f"Tool '{tool_name}': no module defining run() found in {src_dir}. Tried: "
        + ", ".join(tried)
    )


def _reject_module_from_elsewhere(module, name: str, src_dir: str) -> None:
    """Refuse a module that came from anywhere but the tool's src/.

    Without this, a tool whose name collides with a standard-library module
    imports the standard library one and fails much further down, with an error
    naming neither.
    """
    origin = getattr(module, "__file__", None)
    if origin is None or os.path.commonpath(
        [os.path.abspath(src_dir), os.path.abspath(origin)]
    ) != os.path.abspath(src_dir):
        raise RunnerError(
            f"Module '{name}' was imported from {origin!r}, outside the tool's "
            f"{SRC_DIR_NAME}/ directory. Rename the tool's module so it does not collide."
        )


def _run_function(module, tool_name: str):
    run = getattr(module, "run", None)
    if not callable(run):
        raise RunnerError(
            f"Tool '{tool_name}': module '{module.__name__}' defines no callable run()."
        )
    return run


def _coerce(value, annotation):
    """Turn a JSON value into what run()'s annotation asks for.

    Only paths need it: JSON has no path type, so they travel as strings and a
    tool annotating `scans: Path` would receive a `str` and fail on the first
    `.glob()`. Everything else the schema allows -- str, int, float, bool and
    lists of them -- arrives as the right type already.

    Deliberately identical to sadt-tools' testkit driver
    (`testkit/src/sadt_testkit/_driver.py`): every tool's integration tests run
    against that one, so a difference here is a suite that passes while
    production fails.

    An empty string is ABSENCE and stays a string. `Path("")` is
    `PosixPath(".")` -- the current directory, and truthy -- so coercing the
    "not supplied" default of an optional path hands the tool a real directory
    to walk. `ASO` read an unset `landmarks=""` as a supplied landmark folder
    and walked its whole checkout. Calling `run()` directly in Python keeps the
    `""` the signature declares, so coercing it is also what makes the two call
    paths disagree.
    """
    if annotation is Path:
        return Path(value) if value != "" else value
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (Path,):
        return [Path(item) for item in value]
    return value


def _call_arguments(run, params: dict) -> dict:
    try:
        hints = typing.get_type_hints(run)
    except Exception:  # noqa: BLE001 - an unresolvable annotation is not fatal
        hints = {}
    return {name: _coerce(value, hints.get(name)) for name, value in params.items()}


def _jsonable(value):
    """json.dump fallback: a returned Path is a path, anything else is a bug.

    Narrow on purpose. `default=str` would serialize any object at all as its
    repr, so a tool returning something the server cannot use would look like a
    successful run right up until the server tried to open it.
    """
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"run() returned a {type(value).__name__}, which cannot be written to "
        f"{RESULT_FILE}. Return JSON-serializable values (paths as strings)."
    )


def _peak_vram_bytes():
    """Peak CUDA memory this process allocated, or None.

    `sys.modules.get` rather than an import: a tool that never touched torch
    stays untouched, and importing it here would cost seconds and a CUDA
    context on every tabular run. If torch is not in sys.modules, the tool did
    not use it, by definition.

    Every failure is swallowed on purpose. This is instrumentation -- it exists
    so a real VRAM budget can be set later from measurements instead of
    guesses -- and instrumentation must never be able to fail a run that
    otherwise succeeded.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - see above
        return None


def _write_result(job_dir: str, result) -> None:
    """Write {"result": ...} atomically.

    Serialized in full BEFORE anything touches the disk, so an unserializable
    result is a clean failure rather than a truncated result.json; and replaced
    into place, so the server never reads one being written.
    """
    body = {"result": result}
    peak = _peak_vram_bytes()
    if peak is not None:
        body["peak_vram_bytes"] = peak
    payload = json.dumps(body, default=_jsonable)
    final_path = os.path.join(job_dir, RESULT_FILE)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(temp_path, final_path)


def _write_error(job_dir: str, exc: Exception) -> None:
    """Record WHICH failure happened, next to where a result would have gone.

    There is no shared exception type -- there is no shared package -- so the
    server maps the class NAME: ToolInputError/ValueError/FileNotFoundError are
    the caller's problem (422, message passed through), ToolUnavailableError is
    the deployment's (503), anything else is opaque (500). The exit code stays
    non-zero either way, so a caller that reads nothing but that still sees a
    failure.
    """
    payload = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    try:
        with open(os.path.join(job_dir, RESULT_FILE), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass  # the traceback on stderr is still the record


def _configure_logging() -> None:
    """The runner owns handlers, levels and formatting.

    Tools attach none -- a library that configures logging takes the decision
    away from whatever runs it -- so without this their log records go nowhere
    at all.
    """
    logging.basicConfig(
        level=os.environ.get("SADT_LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

def _takes_supervisor(run) -> bool:
    """Does this run() ask for a supervisor?

    The same rule `describe.py` uses to keep it out of the schema: a parameter
    named `sup`, keyword-only, and UNANNOTATED. Unannotated is the marker rather
    than an accident -- every other parameter must be annotated, so there is
    nothing else this shape could be, and a tool cannot grow a supervisor by
    forgetting a type.
    """
    try:
        parameter = inspect.signature(run).parameters.get(SUPERVISOR_ARGUMENT)
        hints = typing.get_type_hints(run)
    except Exception:  # noqa: BLE001 - an unresolvable annotation is not fatal
        return False
    return (
        parameter is not None
        and parameter.kind is parameter.KEYWORD_ONLY
        and SUPERVISOR_ARGUMENT not in hints
    )


class _Supervisor:
    """What a tool receives as `sup`: five members, duck-typed, nothing shared.

    A tool never imports this class. It cannot -- the tool's venv holds none of
    the server -- and that is the point: the same object shape is produced by
    `sadt-tools`' `scripts/run_tool.py` and faked in its tests, and a tool
    cannot tell the three apart.

    `run()` re-enters THIS file with the sibling's interpreter, so the callee
    gets its own venv, its own dependency set and its own supervisor. Nesting
    (`AREG -> ASO -> ALI`) is that same recursion and needs no special case.

    **It does not go back through the server.** A nested call is a subprocess of
    the parent, so it never queues for a slot the parent is already holding --
    which is exactly the deadlock the in-process version had, where four
    concurrent ASO runs each waited on a fifth slot. The cost is that nested
    work is invisible to MAX_GPU_JOBS: a supervised chain can put two tools on
    the card at once. Chains are serial today (ASO waits for ALI before it
    registers), so the peak is one tool at a time per chain, but a deployment
    running several supervised jobs concurrently has to size for that.
    """

    def __init__(self, tools_dir: str, job_dir: str, depth: int, chain=(), job_id=None, root=None):
        self._tools_dir = tools_dir
        self._job_dir = job_dir
        self._depth = depth
        # Innermost last. `chain` is what refuses a cycle by NAME; `depth` is
        # only a backstop for a chain that grows without repeating.
        self._chain = tuple(chain)
        self._job_id = job_id
        # The job at the top of this tree. Carried so every record in a
        # supervised run can be attributed to the request that started it --
        # traceability, and what a VRAM budget would later be applied to. It is
        # NOT what makes nesting deadlock-free: a nested call is a subprocess of
        # its parent and never re-enters the server's admission queue at all,
        # so there is no queue it could wait in.
        self._root = root or job_id
        self._calls = 0
        self.out = Path(job_dir) / "output"
        # Removed with the job directory, by whoever owns it. The tool is held
        # to writing only under `output/`, so its scratch sits beside it rather
        # than inside what the caller keeps.
        self.tmp = Path(job_dir) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)

    # -- the frozen interface ----------------------------------------------

    def run(self, tool: str, **params):
        """Run another tool, blocking, and return what its run() returned."""
        # Checked BEFORE the depth cap, because it is the more precise answer to
        # the same question. A cycle hits the depth limit eventually anyway, but
        # only after starting four processes and whatever they each loaded, and
        # the message it produces names a number rather than the mistake.
        if tool in self._chain:
            raise RunnerError(
                "Supervised call cycle: {} -> {}. A tool cannot call one that is "
                "already running above it.".format(" -> ".join(self._chain), tool)
            )
        if self._depth >= MAX_SUPERVISOR_DEPTH:
            raise RunnerError(
                "Supervised calls nested more than {} deep ({} -> {}).".format(
                    MAX_SUPERVISOR_DEPTH, " -> ".join(self._chain), tool
                )
            )

        interpreter = self._interpreter(tool)
        self._calls += 1
        nested_dir = os.path.join(self._job_dir, "sup", f"{self._calls:02d}_{tool}")
        os.makedirs(os.path.join(nested_dir, "output"), exist_ok=True)

        child_id = f"{os.path.basename(self._job_dir)}.{self._calls}"
        job_path = os.path.join(nested_dir, JOB_FILE)
        with open(job_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job_id": child_id,
                    "tool": tool,
                    "job_dir": nested_dir,
                    "params": params,
                    "parent": self._job_id,
                    "root": self._root,
                },
                handle,
                default=_jsonable,
            )

        self.log(f"running '{tool}'")
        environment = dict(os.environ)
        environment[SUPERVISOR_DEPTH_ENV] = str(self._depth + 1)
        environment[SUPERVISOR_CHAIN_ENV] = ",".join(self._chain + (tool,))
        # SADT_TOOL_DIR points at the PARENT's folder; the callee derives its
        # own from its interpreter, and inheriting ours would send it to the
        # wrong sources.
        environment.pop(TOOL_DIR_ENV, None)

        completed = subprocess.run(
            [interpreter, os.path.abspath(__file__), "--job", job_path],
            # Not captured: a nested tool's log is the only sign of life during
            # an hour-long run, and it is already on stderr where the server
            # collects it. Its result never travels on stdout anyway.
            cwd=nested_dir,
            env=environment,
        )
        if completed.returncode != 0:
            # Its own result file is where the useful half is. The child's
            # traceback goes to stderr, which whatever runs the PARENT may be
            # capturing and trimming -- so "see above" is a promise this cannot
            # keep, and the message has to carry the reason itself.
            raise RunnerError(
                f"Supervised tool '{tool}' failed (exit {completed.returncode}). "
                f"{self._failure(tool, nested_dir)}"
            )
        return self._result(tool, nested_dir)

    def _failure(self, tool: str, nested_dir: str) -> str:
        """What the callee said went wrong, read back from its result file."""
        path = os.path.join(nested_dir, RESULT_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                error = json.load(handle).get("error") or {}
        except (OSError, ValueError):
            return f"It wrote no readable result; see its output above and {path}."
        kind = error.get("type", "Error")
        message = error.get("message", "").strip()
        return f"{kind}: {message}" if message else f"{kind} (no message)."

    def progress(self, fraction: float, message: str) -> None:
        try:
            self.log(f"{float(fraction):.0%} {message}")
        except (TypeError, ValueError):
            self.log(str(message))

    def log(self, message: str) -> None:
        # Through logging, not print: the runner owns handlers and the server
        # reads stderr. The depth prefix is what makes a nested chain readable.
        logging.getLogger("sadt.supervisor").info("%s%s", "  " * self._depth, message)

    # -- internals ----------------------------------------------------------

    def _interpreter(self, tool: str) -> str:
        """The tool's own python, wherever its folder sits.

        Searched one level deep as well as at the top, because a tool may live
        under a GROUPING folder: `ALI_CBCT` and `ALI_IOS` are two tools inside
        `tools/ALI/`, which holds no tool of its own. The folder name is the
        tool name either way -- only its depth varies -- so this stays a lookup
        rather than a scan of every pyproject, which this file could not read
        anyway (stdlib only, and tomllib does not exist before 3.11).

        Depth is capped at 2 on purpose: deeper would start matching a tool's
        own vendored directories.
        """
        for folder in self._candidate_folders(tool):
            for relative in (os.path.join("bin", "python"), os.path.join("Scripts", "python.exe")):
                candidate = os.path.join(folder, ".venv", relative)
                if os.path.isfile(candidate):
                    return candidate
        raise RunnerError(
            f"Supervised tool '{tool}' has no virtualenv under {self._tools_dir} "
            f"(looked for '{tool}/.venv' and '*/{tool}/.venv'). It is not deployed here."
        )

    def _candidate_folders(self, tool: str):
        """`<root>/<tool>` then `<root>/*/<tool>`, for each root we might be under.

        Two roots, because the CALLER may itself be nested. `_tool_dir()` gives
        this tool's folder and the root was taken as its parent -- right for
        tools/AMASSS, wrong for tools/ALI/ALI_CBCT, where the parent is the
        grouping folder and its siblings are the only tools reachable. That is
        latent today, ALI_CBCT calling nothing, and it points straight at
        AREG_IOSCBCT: a nested tool whose whole job is calling the others.
        """
        seen = set()
        for root in self._roots:
            direct = os.path.join(root, tool)
            if direct not in seen:
                seen.add(direct)
                yield direct
            try:
                groups = sorted(os.listdir(root))
            except OSError:
                continue
            for group in groups:
                if group == tool:
                    continue
                nested = os.path.join(root, group, tool)
                if nested not in seen and os.path.isdir(nested):
                    seen.add(nested)
                    yield nested

    @property
    def _roots(self):
        """The tools directory, and its parent when we are a nested tool."""
        roots = [self._tools_dir]
        parent = os.path.dirname(self._tools_dir)
        if parent and parent != self._tools_dir:
            roots.append(parent)
        return roots

    def _result(self, tool: str, nested_dir: str):
        path = os.path.join(nested_dir, RESULT_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise RunnerError(f"Supervised tool '{tool}' wrote no readable result: {exc}")
        if "error" in payload:
            error = payload["error"]
            raise RunnerError(
                f"Supervised tool '{tool}' failed: "
                f"{error.get('type', 'Error')}: {error.get('message', '')}"
            )
        value = payload.get("result")
        if isinstance(value, str):
            return Path(value)
        if isinstance(value, dict):
            return {key: Path(item) for key, item in value.items()}
        return value


def _supervisor_for(run, job: dict):
    """A supervisor for this job, or None when the tool does not ask for one."""
    if not _takes_supervisor(run):
        return None
    tools_dir = os.path.dirname(_tool_dir())
    depth = 0
    try:
        depth = int(os.environ.get(SUPERVISOR_DEPTH_ENV, "0"))
    except ValueError:
        pass
    # The chain the parent handed us, plus ourselves. A root job has an empty
    # environment variable and a chain of just its own name, which is what makes
    # a tool calling ITSELF the first thing refused rather than the fifth.
    inherited = tuple(
        name for name in os.environ.get(SUPERVISOR_CHAIN_ENV, "").split(",") if name
    )
    chain = inherited or (job["tool"],)
    return _Supervisor(
        tools_dir=tools_dir,
        job_dir=job["job_dir"],
        depth=depth,
        chain=chain,
        job_id=job.get("job_id"),
        root=job.get("root"),
    )


def _load_job(job_path: str) -> dict:
    try:
        with open(job_path, encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"Cannot read the job file {job_path}: {exc}")

    for field in ("tool", "job_dir", "params"):
        if field not in job:
            raise RunnerError(f"Job file {job_path} has no '{field}' field.")
    if not isinstance(job["params"], dict):
        raise RunnerError("Job field 'params' must be an object of argument name -> value.")
    if not os.path.isdir(job["job_dir"]):
        raise RunnerError(f"Job directory does not exist: {job['job_dir']}")
    return job


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one SADT tool job.")
    parser.add_argument("--job", required=True, help="Path to the job.json written by the server")
    arguments = parser.parse_args(argv)

    _configure_logging()

    job = None
    try:
        job = _load_job(arguments.job)
        module = _import_tool(job["tool"], os.path.join(_tool_dir(), SRC_DIR_NAME))
        run = _run_function(module, job["tool"])
        arguments = _call_arguments(run, job["params"])
        supervisor = _supervisor_for(run, job)
        if supervisor is not None:
            # Injected, never taken from job.json: it is not data, and a client
            # that could name one would be naming a process to start.
            arguments[SUPERVISOR_ARGUMENT] = supervisor
        result = run(**arguments)
        _write_result(job["job_dir"], result)
    except RunnerError as exc:
        # Ours, and already precise: the traceback would only point back here.
        print(str(exc), file=sys.stderr)
        if job:
            _write_error(job["job_dir"], exc)
        return 1
    except Exception as exc:
        # The tool's own failure. The whole traceback goes to stderr because
        # the server keeps only its tail, and that tail is all anyone will have
        # to work from; the class name goes in the result file, because it is
        # what decides the status code.
        traceback.print_exc()
        if job:
            _write_error(job["job_dir"], exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
