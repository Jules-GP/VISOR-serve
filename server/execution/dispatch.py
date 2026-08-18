"""Run a tool in its own interpreter, in its own process.

This is the server half of what runner.py does on the other side:

    <TOOLS_DIR>/<tool>/.venv/bin/python <RUNNER_PATH> --job <job dir>/job.json

Why a subprocess at all, when importing the tool works today: importing it
pins the SERVER's Python to the lowest common denominator of every tool, and
two tools wanting different versions of torch (or of numpy: SurgMovPred asks
for numpy==2.4.0, AREG for numpy<2.0.0) cannot coexist in one interpreter at
all. Holding torch in the API process also keeps a CUDA context alive for the
lifetime of the server, so VRAM is never fully released between jobs, and a
segfault inside a CUDA kernel takes the API down with the job.

**Still synchronous.** The HTTP request blocks for exactly as long as it does
today; `subprocess.run` is called from the same worker thread `tool.invoke`
already ran in, so MAX_CONCURRENT_TOOLS keeps arbitrating how many run at
once. No queue, no polling, no change to the contract with the client.

The job directory is handed out by file_utils, which means the request handler
already removes it once the response has been streamed -- outputs included,
and on the error paths too. Nothing here needs its own cleanup timer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import uuid
from typing import Any, Optional

import file_utils
from base import ToolUnavailableError
from config import settings
from registry.deployment import deployment_config

logger = logging.getLogger("inference_server")

JOB_FILE = "job.json"
RESULT_FILE = "result.json"
# How long a TERMed process group gets before SIGKILL.
_KILL_GRACE_SECONDS = 10.0

STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

# Subdirectories every job gets. `input/` is where the server will stage inputs
# once tools stop being handed paths into the request's own work dir; `output/`
# is what a tool writes into and what survives long enough to be streamed back.
JOB_SUBDIRS = ("input", "output")

# How much of the tool's stderr travels with the failure. Enough for a
# traceback, bounded because a failing tool can print megabytes.
_STDERR_TAIL_BYTES = 8192


# The GPU is one device and the tools no longer arbitrate for it themselves
# (see settings.MAX_CONCURRENT_GPU_JOBS). Created lazily so the setting can be
# monkeypatched in a test before the first run.
_gpu_slots: Optional[threading.BoundedSemaphore] = None
_gpu_lock = threading.Lock()

# The argument a tool declares when it can be told where to run, and the values
# that mean "not on the card".
DEVICE_ARGUMENT = "device"
_CPU_DEVICES = ("cpu", "mps")


def _gpu_semaphore() -> threading.BoundedSemaphore:
    global _gpu_slots
    with _gpu_lock:
        if _gpu_slots is None:
            _gpu_slots = threading.BoundedSemaphore(settings.MAX_CONCURRENT_GPU_JOBS)
        return _gpu_slots


def uses_the_gpu(tool, params: dict) -> bool:
    """Will this run take the card?

    **Assumed yes**, unless the run is explicitly on something else. Only a
    tool that declares `device` can say otherwise, and only by resolving it to
    a CPU value.

    The safe default is the strict one. A tool that quietly imports torch
    without declaring `device` -- and several do; it was `settings.DEVICE`
    inside them until they were packaged -- would otherwise never queue, and
    two of them would meet on the card with nothing between them. The cost of
    being wrong the other way is a CPU-only run waiting for a slot it did not
    need; the cost of being wrong this way is an out-of-memory in the middle of
    somebody's cohort.
    """
    spec = tool.arguments.get(DEVICE_ARGUMENT)
    if spec is None:
        return True
    value = params.get(DEVICE_ARGUMENT)
    if value is None:
        value = spec.default if spec.is_choice else settings.DEVICE
    return not str(value).lower().startswith(_CPU_DEVICES)


class ToolFailure(RuntimeError):
    """The tool itself raised, and said which exception it was.

    There is no shared exception type to catch -- there is no shared package --
    so the runner records the class NAME and main.py maps it: the caller's
    fault (422), this deployment's (503), or opaque (500).
    """

    def __init__(self, error_type: str, message: str):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


class ToolExecutionError(RuntimeError):
    """The tool's process failed: non-zero exit, a timeout, or no usable
    result.json.

    Deliberately NOT one of base.py's typed errors: main.py maps those to 422 /
    501, and a tool that crashed is neither the caller's fault nor something
    this deployment can be reconfigured to fix. It falls through to the generic
    500 handler, which logs the detail server-side and answers the client with
    "Tool execution failed." -- exactly what an in-process crash does today.
    """


def tool_interpreter(tool_name: str) -> str:
    """Path to the interpreter of this tool's virtualenv.

    Searched one level below TOOLS_DIR as well as directly under it, because a
    tool may live inside a GROUPING folder: ALI_CBCT and ALI_IOS are two tools
    in tools/ALI/, which is not a tool itself. The folder is still named after
    the tool -- only its depth varies -- so this stays a lookup.

    The direct path is returned when nothing matches, so the caller's error
    names where it looked first.
    """
    direct = os.path.join(settings.TOOLS_DIR, tool_name, ".venv", "bin", "python")
    if os.path.isfile(direct):
        return direct
    try:
        groups = sorted(os.listdir(settings.TOOLS_DIR))
    except OSError:
        return direct
    for group in groups:
        if group == tool_name:
            continue
        nested = os.path.join(settings.TOOLS_DIR, group, tool_name, ".venv", "bin", "python")
        if os.path.isfile(nested):
            return nested
    return direct


def _checked_interpreter(tool_name: str) -> str:
    interpreter = tool_interpreter(tool_name)
    if not os.path.isfile(interpreter):
        # The path names a server-side directory, so it goes to the log and not
        # into the exception: a 501 body travels to the client.
        logger.error(
            "Tool '%s' has no interpreter at %s -- its virtualenv is missing.",
            tool_name,
            interpreter,
        )
        raise ToolUnavailableError(
            f"Tool '{tool_name}' is not installed on this server (no virtualenv). "
            f"See the server logs."
        )
    return interpreter


def _checked_runner() -> str:
    runner = settings.RUNNER_PATH
    if not os.path.isfile(runner):
        logger.error("RUNNER_PATH does not point at a file: %s", runner)
        raise ToolUnavailableError("This server is misconfigured: the tool runner is missing.")
    return runner


def _create_job_dir(job_id: str) -> str:
    """A fresh directory for this job, registered for request-scoped cleanup.

    Named after the job so a directory left behind by a crash can be traced
    back to a log line, and created with exist_ok=False so two jobs can never
    end up sharing one.
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    job_dir = os.path.join(settings.TEMP_DIR, f"job_{job_id}")
    os.makedirs(job_dir)
    for subdir in JOB_SUBDIRS:
        os.makedirs(os.path.join(job_dir, subdir))
    return file_utils.register_scratch_dir(job_dir)


def _server_provided(tool, params: dict, job_dir: str) -> dict:
    """The arguments the SERVER fills in, not the caller.

    `output_dir` is the job's own output/: every packaged tool takes it as a
    required argument and writes only there, and a client has no business
    naming a directory on the server.

    `device` is the deployment's, when the caller did not pick one. It used to
    be `settings.DEVICE` read inside each tool; a tool that no longer reads the
    environment would otherwise always run on its own default (cuda), on a
    server configured for CPU.
    """
    filled = dict(params)
    if getattr(tool, "wants_output_dir", False):
        output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        filled["output_dir"] = output_dir
    if DEVICE_ARGUMENT in tool.arguments and DEVICE_ARGUMENT not in filled:
        filled[DEVICE_ARGUMENT] = settings.DEVICE
    return filled


def _write_job_file(job_dir: str, job_id: str, tool_name: str, params: dict) -> str:
    """Write job.json. Every path in `params` is already resolved by the server
    (uploads streamed to disk, server_selectable names looked up through
    data_store): the tool never sees a reference it would have to resolve, and
    never reads DATA_DIR by itself."""
    job = {
        "job_id": job_id,
        "tool": tool_name,
        "job_dir": job_dir,
        "params": params,
    }
    job_path = os.path.join(job_dir, JOB_FILE)
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, default=_jsonable)
    return job_path


def _jsonable(value):
    """Same narrow rule as the runner's: a Path is a path, anything else is a
    schema the wire cannot carry and must fail here rather than halfway."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"Argument value of type {type(value).__name__} cannot be written to {JOB_FILE}."
    )


def _child_environment(job_id: str, job_dir: str) -> dict:
    """The environment the tool process runs in.

    Inherited rather than rebuilt: tools legitimately need PATH, HOME,
    LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES and whatever else the deployment
    sets. API_TOKEN is removed -- it is the server's credential, a tool has no
    use for it, and the tool venvs hold third-party code.
    """
    environment = dict(os.environ)
    environment.pop("API_TOKEN", None)
    environment.update(
        {
            "SADT_API": settings.SADT_API,
            "SADT_JOB_ID": job_id,
            "SADT_JOB_DIR": job_dir,
        }
    )
    return environment


def _stderr_tail(job_dir: str) -> str:
    """The last few KB the tool wrote to stderr, for the exception message.

    May well contain a file path, so it is treated exactly like an in-process
    traceback: logged server-side, never returned to the client (main.py
    answers 500 with a fixed message).
    """
    path = os.path.join(job_dir, STDERR_LOG)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > _STDERR_TAIL_BYTES:
                handle.seek(size - _STDERR_TAIL_BYTES)
            tail = handle.read()
    except OSError:
        return "(no stderr captured)"
    return tail.decode("utf-8", errors="replace").strip() or "(empty stderr)"


def _execute(command: list, job_dir: str, environment: dict, timeout: Optional[float],
             tool_name: str = "") -> int:
    """Run the tool process to completion; return its exit code.

    stdout and stderr go to FILES, not to pipes: a tool can print for hours
    (nnUNet does), and a pipe means holding all of it in the server's memory.
    Their contents are never logged either -- shapeaxi prints the patient's own
    file name.

    The process gets its own SESSION (`start_new_session`), and a timeout kills
    the whole PROCESS GROUP rather than the one PID we know about. That
    distinction is the entire point: nnUNet, torch's DataLoader and shapeaxi all
    fork workers, and killing the parent leaves those workers running and
    holding VRAM -- a card that stays full after the job that filled it is gone,
    with nothing left to attribute it to. `killpg` reaches them because the
    session made them one group.
    """
    with open(os.path.join(job_dir, STDOUT_LOG), "wb") as out_stream, open(
        os.path.join(job_dir, STDERR_LOG), "wb"
    ) as error_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=out_stream,
                stderr=error_stream,
                env=environment,
                # A tool writing a relative path lands in its own job directory
                # rather than in the server's source tree.
                cwd=job_dir,
                # POSIX only, which the deployment image is. On Windows this
                # would need CREATE_NEW_PROCESS_GROUP and a different kill.
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolExecutionError(f"Could not start the tool process: {exc}")

        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            # Naming both knobs, because which one applied is not visible from
            # the outside and the operator's next move differs: a per-tool entry
            # raises it for this tool alone, the global setting for everything.
            raise ToolExecutionError(
                f"Tool '{tool_name}' did not finish within its timeout ({timeout}s). "
                f"Raise it with [tools.{tool_name}] timeout_seconds in deployment.toml, "
                f"or TOOL_TIMEOUT_SECONDS for every tool."
            )


def _kill_group(process: subprocess.Popen) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left.

    TERM first so a tool with a handler can release the card and flush; KILL
    after a grace period because most will not. `os.killpg` needs the group id,
    which is the child's pid precisely because `start_new_session` made it a
    group leader.

    Every failure here is swallowed deliberately: the process may have exited
    between the timeout and the signal, and a ProcessLookupError then is the
    normal case, not an error to propagate over a timeout that already is one.
    """
    for signal_number, grace in ((signal.SIGTERM, _KILL_GRACE_SECONDS), (signal.SIGKILL, None)):
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
        except (ProcessLookupError, PermissionError, OSError):
            break
        if grace is None:
            break
        try:
            process.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # Unreapable after SIGKILL means the process is stuck in the kernel
        # (uninterruptible I/O). Nothing further is possible from here.
        logger.error("Tool process %s survived SIGKILL; it is now a zombie.", process.pid)


def _read_result(job_dir: str, tool_name: str) -> Any:
    """The value run() returned, out of result.json.

    A missing file after a zero exit code is its own failure: the runner writes
    the file last and atomically, so "exited fine but produced nothing" means
    the tool never got as far as returning.
    """
    path = os.path.join(job_dir, RESULT_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise ToolExecutionError(
            f"Tool '{tool_name}' exited successfully but wrote no {RESULT_FILE}."
        )
    except (OSError, ValueError) as exc:
        raise ToolExecutionError(f"Tool '{tool_name}' wrote an unreadable {RESULT_FILE}: {exc}")

    if not isinstance(payload, dict):
        raise ToolExecutionError(f"Tool '{tool_name}': {RESULT_FILE} must be an object.")

    error = payload.get("error")
    if isinstance(error, dict):
        # The tool raised and named its exception class. Which one it was is
        # the difference between "you sent the wrong thing" and "this broke".
        raise ToolFailure(str(error.get("type", "")), str(error.get("message", "")))

    if "result" not in payload:
        raise ToolExecutionError(
            f"Tool '{tool_name}': {RESULT_FILE} must be an object with a 'result' field."
        )
    return payload["result"]


def dispatch(tool, params: dict, job_id: Optional[str] = None) -> Any:
    """Run `tool` on already-validated `params` in the tool's own interpreter.

    Returns exactly what the tool's run() returned, so callers -- Tool.invoke,
    and through it main.py -- cannot tell which side of the flag they are on.
    """
    interpreter = _checked_interpreter(tool.name)
    runner = _checked_runner()
    job_id = job_id or uuid.uuid4().hex
    job_dir = _create_job_dir(job_id)

    try:
        params = _server_provided(tool, params, job_dir)
        job_path = _write_job_file(job_dir, job_id, tool.name, params)
        command = [interpreter, runner, "--job", job_path]
        environment = _child_environment(job_id, job_dir)
        # Per tool, falling back to the global setting. 0 means no limit, which
        # a cohort legitimately needs.
        timeout = deployment_config.timeout_seconds(tool.name) or None

        if uses_the_gpu(tool, params):
            # Held for the whole run, and released by leaving the block on any
            # path. Blocking on purpose: the request already waits for the tool,
            # and a queue is what a single card wants.
            with _gpu_semaphore():
                exit_code = _execute(command, job_dir, environment, timeout, tool.name)
        else:
            exit_code = _execute(command, job_dir, environment, timeout, tool.name)

        if exit_code != 0:
            # A tool that recorded WHICH exception it was gets to say so; the
            # tail of stderr is the fallback for one that died without writing
            # anything (a segfault in a CUDA kernel, an OOM kill).
            _read_result(job_dir, tool.name)
            raise ToolExecutionError(
                f"Tool '{tool.name}' exited with code {exit_code}:\n{_stderr_tail(job_dir)}"
            )
        return _read_result(job_dir, tool.name)
    except Exception:
        # No response will ever be streamed from this job, so nothing has to
        # survive: take the directory down now rather than leaving confidential
        # inputs waiting for the request handler's cleanup.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
