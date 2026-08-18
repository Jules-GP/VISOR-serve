"""The runner hands a tool a supervisor, and it calls another tool through it.

Everything here runs the REAL runner as a subprocess, against tools built on
the fly in a temporary TOOLS_DIR -- no fixtures pretending to be venvs, because
the thing under test is precisely that the callee gets its own interpreter.

The tools are one-file packages with no dependencies, so `.venv` is a symlink
farm around `sys.executable`; that is enough for the runner, which only ever
asks for `<tool>/.venv/bin/python`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "execution" / "runner.py"


def make_tool(tools_dir: Path, name: str, body: str) -> Path:
    """A runnable tool: `src/sadt_<name>/__init__.py` plus a venv pointing here."""
    package = tools_dir / name / "src" / f"sadt_{name.lower()}"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n\n" + textwrap.dedent(body), encoding="utf-8"
    )
    binaries = tools_dir / name / ".venv" / "bin"
    binaries.mkdir(parents=True)
    # The runner derives the tool folder from sys.prefix, so the interpreter has
    # to LOOK like it lives in this venv. A symlink does that without building
    # one: sys.prefix follows the link's directory, not its target.
    (binaries / "python").symlink_to(sys.executable)
    (tools_dir / name / ".venv" / "pyvenv.cfg").write_text(
        "home = {}\ninclude-system-site-packages = true\n".format(
            os.path.dirname(sys.executable)
        ),
        encoding="utf-8",
    )
    return tools_dir / name


def run_job(tools_dir: Path, name: str, job_dir: Path, params: dict):
    """Invoke the runner exactly as the server does, and hand back result.json."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {"job_id": "t", "tool": name, "job_dir": str(job_dir), "params": params}
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(tools_dir / name / ".venv" / "bin" / "python"), str(RUNNER),
         "--job", str(job_path)],
        capture_output=True, text=True, cwd=str(job_dir),
    )
    result = {}
    if (job_dir / "result.json").is_file():
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    return completed, result


LEAF = """
    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        \"\"\"Write one file and return where it went.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "leaf.txt").write_text(tag + ":" + str(scans))
        return output_dir
"""

CALLER = """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call the leaf tool, then write what it produced.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if sup is None:
            raise RuntimeError("no supervisor was injected")
        sup.progress(0.5, "calling Leaf")
        produced = sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf", tag="called")
        (output_dir / "chained.txt").write_text((Path(produced) / "leaf.txt").read_text())
        return output_dir
"""


@pytest.fixture
def tools_dir(tmp_path):
    folder = tmp_path / "tools"
    folder.mkdir()
    return folder


def test_a_tool_that_asks_for_no_supervisor_is_given_none(tools_dir, tmp_path):
    make_tool(tools_dir, "Leaf", LEAF)

    completed, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "leaf.txt").read_text().startswith("leaf:")


def test_a_tool_declaring_sup_receives_one_and_reaches_the_other_tool(tools_dir, tmp_path):
    """The whole point: `*, sup` in the signature, a real second venv on the
    other end, and no import between the two."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    completed, result = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "chained.txt").read_text() == "called:{}".format(
        tmp_path / "in"
    )


def test_the_nested_run_gets_its_own_job_directory(tools_dir, tmp_path):
    """So a chain can be read afterwards: which tool ran, in what order, and
    what it was asked for are all on disk."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    run_job(tools_dir, "Caller", tmp_path / "job",
            {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")})

    nested = tmp_path / "job" / "sup" / "01_Leaf"
    assert (nested / "job.json").is_file()
    assert json.loads((nested / "job.json").read_text())["tool"] == "Leaf"
    assert (nested / "result.json").is_file()


def test_progress_and_log_reach_stderr(tools_dir, tmp_path):
    """The runner owns logging, so a supervised call has to surface through it
    -- a nested tool's output is the only sign of life during a long run."""
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Caller", CALLER)

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert "50% calling Leaf" in completed.stderr
    assert "running 'Leaf'" in completed.stderr


def test_a_failing_nested_tool_names_itself(tools_dir, tmp_path):
    """A chain that breaks has to say which link broke."""
    make_tool(tools_dir, "Leaf", """
    def run(scans: Path, output_dir: Path, tag: str = "leaf") -> Path:
        \"\"\"Always fail.\"\"\"
        raise ValueError("the leaf refused")
    """)
    make_tool(tools_dir, "Caller", CALLER)

    job_dir = tmp_path / "job"
    completed, result = run_job(
        tools_dir, "Caller", job_dir,
        {"scans": str(tmp_path / "in"), "output_dir": str(job_dir / "output")},
    )

    assert completed.returncode != 0
    assert "Leaf" in completed.stderr
    # The reason travels in the PARENT's error, not only in the child's output:
    # whatever runs the parent may be capturing and trimming stderr, so "see
    # above" is a promise the supervisor cannot keep. This was found by running
    # a real chain through the server, where the child's traceback vanished.
    assert "ValueError: the leaf refused" in completed.stderr
    assert (job_dir / "result.json").is_file()
    error = json.loads((job_dir / "result.json").read_text())["error"]
    assert "the leaf refused" in error["message"]


def test_an_undeployed_tool_says_it_is_not_installed(tools_dir, tmp_path):
    make_tool(tools_dir, "Caller", CALLER)  # no Leaf at all

    completed, _ = run_job(
        tools_dir, "Caller", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode != 0
    assert "not deployed here" in completed.stderr


def test_a_tool_calling_itself_is_stopped(tools_dir, tmp_path):
    """A cycle would otherwise fork until the machine gives out."""
    make_tool(tools_dir, "Loop", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call itself forever.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Loop", scans=scans, output_dir=sup.tmp / "again")
        return output_dir
    """)

    completed, _ = run_job(
        tools_dir, "Loop", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )

    assert completed.returncode != 0
    # Refused by NAME on the first call, not by the depth cap on the fifth:
    # four processes and whatever each of them loaded are not spent finding out
    # what the chain already says.
    assert "Supervised call cycle" in completed.stderr
    assert "Loop -> Loop" in completed.stderr


def test_an_empty_optional_path_stays_empty(tools_dir, tmp_path):
    """`Path("")` is `PosixPath(".")` -- the current directory, and truthy.

    Coercing the "not supplied" default of an optional path therefore hands the
    tool a real directory: ASO read an unset `landmarks=""` as a supplied
    landmark folder and walked its whole checkout.
    """
    make_tool(tools_dir, "Optional", """
    def run(output_dir: Path, extra: Path = "") -> Path:
        \"\"\"Report whether the optional path arrived as absence.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "seen.txt").write_text("absent" if not extra else str(extra))
        return output_dir
    """)

    completed, result = run_job(
        tools_dir, "Optional", tmp_path / "job",
        {"output_dir": str(tmp_path / "job" / "output"), "extra": ""},
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(result["result"], "seen.txt").read_text() == "absent"


# ---------------------------------------------------------------------------
# parent / root, and the chain that refuses a cycle by name
# ---------------------------------------------------------------------------

def test_a_child_job_records_its_parent_and_its_root(tools_dir, tmp_path):
    """Every nested job says which call made it and which request started it.

    Not what makes nesting deadlock-free -- a nested call is a subprocess of its
    parent and never re-enters the server's admission queue, so there is no
    queue it could wait in. This is traceability, and the key a VRAM budget
    would later be applied to.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    make_tool(tools_dir, "Parent", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call one child.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Leaf", scans=scans, output_dir=sup.tmp / "leaf")
        return output_dir
    """)

    job_dir = tmp_path / "job"
    completed, _ = run_job(
        tools_dir, "Parent", job_dir,
        {"scans": str(tmp_path / "in"), "output_dir": str(job_dir / "output")},
    )
    assert completed.returncode == 0, completed.stderr

    child_jobs = sorted(job_dir.glob("sup/*/job.json"))
    assert len(child_jobs) == 1
    record = json.loads(child_jobs[0].read_text(encoding="utf-8"))
    assert record["parent"] == "t"
    assert record["root"] == "t"
    assert record["tool"] == "Leaf"


def test_a_cycle_two_tools_long_is_named_not_counted(tools_dir, tmp_path):
    """A -> B -> A is refused at the third call, naming all three.

    The depth cap would also stop it, but only after five processes and
    whatever each of them imported, and its message names a number rather than
    the mistake.
    """
    make_tool(tools_dir, "Ping", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call Pong.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Pong", scans=scans, output_dir=sup.tmp / "pong")
        return output_dir
    """)
    make_tool(tools_dir, "Pong", """
    def run(scans: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Call Ping back.\"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sup.run("Ping", scans=scans, output_dir=sup.tmp / "ping")
        return output_dir
    """)

    completed, _ = run_job(
        tools_dir, "Ping", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert completed.returncode != 0
    assert "Supervised call cycle" in completed.stderr
    assert "Ping -> Pong -> Ping" in completed.stderr


def test_the_result_carries_no_vram_figure_when_the_tool_never_touched_torch(tools_dir, tmp_path):
    """`sys.modules.get("torch")` and nothing else: a tabular tool pays nothing.

    The absence IS the assertion. Importing torch to ask would cost seconds and
    a CUDA context on every run of every tool that has no use for either.
    """
    make_tool(tools_dir, "Leaf", LEAF)
    _, result = run_job(
        tools_dir, "Leaf", tmp_path / "job",
        {"scans": str(tmp_path / "in"), "output_dir": str(tmp_path / "job" / "output")},
    )
    assert "result" in result
    assert "peak_vram_bytes" not in result
