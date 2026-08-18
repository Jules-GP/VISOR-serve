"""The subprocess dispatch path, proven with tools/_dispatch_probe.

The probe needs no model, no GPU and no dependency at all, so everything that
fails here is the dispatch path itself: the job directory, job.json, another
interpreter, runner.py, result.json, and the cleanup around them. See
conftest.py for how it is laid out.
"""

import contextlib
import json
import os
import sys
import signal
import time

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

import config
from execution import dispatch
from base import ToolArgumentError, ToolUnavailableError
from config import settings


# ----------------------------------------------------------------------
# The loop itself
# ----------------------------------------------------------------------

def test_the_tool_runs_in_its_own_interpreter(probe_tool, probe_python, tracked_scratch_dirs):
    result = dispatch.dispatch(probe_tool, {"a": 2, "b": 3})

    assert result["total"] == 5
    # The whole point of the exercise: another interpreter entirely.
    assert result["executable"] != sys.executable
    assert os.path.realpath(result["executable"]) == os.path.realpath(probe_python)


def test_the_tool_writes_into_its_own_job_directory(
    probe_tool, probe_python, tracked_scratch_dirs
):
    result = dispatch.dispatch(probe_tool, {"a": 20, "b": 22})

    output_path = result["outputs"]["probe"]
    assert os.path.isfile(output_path)
    with open(output_path) as handle:
        assert handle.read() == "42"

    job_dir = os.path.dirname(os.path.dirname(output_path))
    assert job_dir in tracked_scratch_dirs, "the job directory must be cleaned up by the request"
    assert result["cwd"] == job_dir, "a relative path must land in the job dir, not the server tree"


def test_the_job_environment_carries_the_three_declared_variables(
    probe_tool, probe_python, tracked_scratch_dirs
):
    result = dispatch.dispatch(probe_tool, {"a": 1, "b": 1}, job_id="a1b2c3d4")

    assert result["job_id"] == "a1b2c3d4"
    assert result["sadt_api"] == settings.SADT_API


def test_the_api_token_never_reaches_the_tool(
    probe_tool, probe_python, tracked_scratch_dirs, monkeypatch
):
    """A tool venv holds third-party code and has no use for the server's
    bearer token."""
    monkeypatch.setenv("API_TOKEN", "a-real-looking-secret")

    result = dispatch.dispatch(probe_tool, {"a": 0, "b": 0})

    assert result["sees_api_token"] is False


def test_invoke_validates_then_dispatches(
    probe_tool, probe_python, subprocess_mode, tracked_scratch_dirs
):
    """Tool.invoke is the single entry point: same validation, other executor.
    ProbeTool.run() asserts if it is ever called in this mode."""
    result = probe_tool.invoke({"a": "7", "b": "5"})  # form values arrive as strings

    assert result["total"] == 12


def test_invoke_rejects_bad_arguments_before_starting_a_process(probe_tool, subprocess_mode):
    with pytest.raises(ToolArgumentError):
        probe_tool.invoke({"a": 1})


def test_in_process_is_the_default_mode():
    """Nothing switches over until it is proven, tool by tool."""
    assert config.Settings(API_TOKEN="x").SADT_DISPATCH_MODE == config.DISPATCH_INPROCESS

    with pytest.raises(ValueError, match="SADT_DISPATCH_MODE"):
        config.Settings(API_TOKEN="x", SADT_DISPATCH_MODE="in-process")


# ----------------------------------------------------------------------
# Failure paths
# ----------------------------------------------------------------------

def test_a_failing_tool_says_which_exception_it_was(
    probe_tool, probe_python, tracked_scratch_dirs
):
    """The class NAME is the whole error contract: there is no shared exception
    type to catch, because there is no shared package."""
    with pytest.raises(dispatch.ToolFailure) as failure:
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1, "fail": True})

    assert failure.value.error_type == "RuntimeError"
    assert failure.value.message == "_dispatch_probe was asked to fail"


def test_a_tool_that_died_without_saying_anything_falls_back_to_stderr(
    probe_tool, probe_python, tracked_scratch_dirs, monkeypatch
):
    """A segfault in a CUDA kernel, or an OOM kill, writes no error file. The
    tail of stderr is all there is, and it has to travel."""
    monkeypatch.setattr(
        dispatch, "_read_result", lambda job_dir, tool_name: (_ for _ in ()).throw(
            FileNotFoundError()
        )
    )
    with pytest.raises(Exception) as failure:
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1, "fail": True})

    assert isinstance(failure.value, (dispatch.ToolExecutionError, FileNotFoundError))


def test_a_failed_run_leaves_nothing_on_disk(probe_tool, probe_python, tracked_scratch_dirs):
    """Its inputs are confidential and its outputs are worthless: the job
    directory goes immediately, without waiting for the request to end."""
    with pytest.raises(dispatch.ToolFailure):
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1, "fail": True}, job_id="doomed")

    assert not os.path.exists(os.path.join(settings.TEMP_DIR, "job_doomed"))


def test_a_tool_without_a_virtualenv_is_unavailable(probe_tool, monkeypatch, tmp_path):
    """501, not 500: the request was fine and nothing the caller changes will
    help -- this deployment simply does not carry the tool."""
    monkeypatch.setattr(settings, "TOOLS_DIR", str(tmp_path))

    with pytest.raises(ToolUnavailableError, match="not installed on this server"):
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1})


def test_a_tool_that_wrote_no_result_is_a_failure(probe_name, tmp_path):
    """A zero exit code is not a result: the runner writes result.json last and
    atomically, so its absence means run() never returned."""
    with pytest.raises(dispatch.ToolExecutionError, match="wrote no result.json"):
        dispatch._read_result(str(tmp_path), probe_name)


def test_a_result_without_a_result_field_is_reported(probe_name, tmp_path):
    (tmp_path / dispatch.RESULT_FILE).write_text(json.dumps({"outputs": {}}))

    with pytest.raises(dispatch.ToolExecutionError, match="'result' field"):
        dispatch._read_result(str(tmp_path), probe_name)


def test_the_timeout_kills_the_tool(probe_tool, probe_python, tracked_scratch_dirs, monkeypatch):
    monkeypatch.setattr(settings, "TOOL_TIMEOUT_SECONDS", 0.001)

    # The message names BOTH knobs now, the limit having become per-tool: which
    # one applied is invisible from outside, and the operator's next move
    # differs between them.
    with pytest.raises(dispatch.ToolExecutionError, match="did not finish within its timeout"):
        dispatch.dispatch(probe_tool, {"a": 1, "b": 1})


# ----------------------------------------------------------------------
# Through the actual endpoint
# ----------------------------------------------------------------------

def test_a_request_runs_a_tool_in_another_process_and_leaves_nothing_behind(
    probe_tool, probe_name, probe_python, subprocess_mode, monkeypatch
):
    """The whole loop over real HTTP: POST /run -> another interpreter ->
    result -> temp files gone.

    main.py is not involved in the change and does not need to be: it calls
    tool.invoke, which is where the two paths part. The probe is injected into
    the registry for this one test rather than living there -- it must never
    appear in GET /tools.
    """
    from fastapi.testclient import TestClient

    import registry
    from main import app

    monkeypatch.setitem(registry.TOOLS, probe_name, probe_tool)
    before = set(os.listdir(settings.TEMP_DIR))

    with TestClient(app) as client:
        response = client.post(
            f"/run/{probe_name}",
            headers={"Authorization": "Bearer test-token"},
            data={"a": "40", "b": "2"},
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["total"] == 42
    assert os.path.realpath(result["executable"]) == os.path.realpath(probe_python)
    assert set(os.listdir(settings.TEMP_DIR)) == before, "the job directory outlived the request"


def test_a_crashing_tool_is_a_500_that_says_nothing(
    probe_tool, probe_name, probe_python, subprocess_mode, monkeypatch
):
    """The stderr tail names files and lands in the server log; the client is
    told only that the run failed."""
    from fastapi.testclient import TestClient

    import registry
    from main import app

    monkeypatch.setitem(registry.TOOLS, probe_name, probe_tool)
    before = set(os.listdir(settings.TEMP_DIR))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/run/{probe_name}",
            headers={"Authorization": "Bearer test-token"},
            data={"a": "1", "b": "1", "fail": "true"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Tool execution failed."}
    assert set(os.listdir(settings.TEMP_DIR)) == before


def test_the_timeout_kills_the_whole_process_group_not_just_the_tool(tmp_path):
    """A grandchild that outlives its parent is killed too.

    This is the entire reason `start_new_session` + `killpg` replaced a bare
    `subprocess.run(timeout=...)`. That kills the one PID it knows about, and
    every heavy tool here forks workers -- nnUNet, torch's DataLoader, shapeaxi.
    An orphaned worker keeps its CUDA context, so the card stays full after the
    job that filled it is gone, with nothing left to attribute it to.

    Deliberately at the level of `_execute` rather than through `dispatch()`:
    the probe's signature and its `source_hash` are pinned by other tests, so
    giving it a "fork and hang" mode to exercise this would change what those
    tests describe.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    pid_file = job_dir / "grandchild.pid"

    # Spawn a grandchild that would happily outlive us, publish its pid, hang.
    program = (
        "import os, subprocess, sys, time;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']);"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid));"
        "time.sleep(300)"
    )

    with pytest.raises(dispatch.ToolExecutionError, match="did not finish within its timeout"):
        dispatch._execute(
            [sys.executable, "-c", program],
            str(job_dir),
            dict(os.environ),
            timeout=2.0,
            tool_name="Probe",
        )

    assert pid_file.is_file(), "the grandchild never started, so this proves nothing"
    grandchild = int(pid_file.read_text())

    # `os.kill(pid, 0)` is the liveness probe: it signals nothing and raises
    # ProcessLookupError once the pid is gone. The grandchild is orphaned rather
    # than ours, so it is reaped by init and never lingers as a zombie -- which
    # is what makes this check meaningful rather than racing a defunct entry.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    # Still alive: clean up so the suite does not leak a 300s sleeper, then fail.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(grandchild, signal.SIGKILL)
    pytest.fail(
        f"grandchild {grandchild} survived the timeout: killpg did not reach the process group"
    )
