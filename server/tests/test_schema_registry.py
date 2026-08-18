"""Tools declared by a `.schema.json`, which the server never imports.

Everything here is about the seam between this repo and the one that packages
the tools: the hash that says a schema still describes its source, the
translation of a narrow type vocabulary into what GET /tools publishes, and
deployment.toml, which is the server's half of the declaration and
deliberately not the tool's.
"""

import json
import os
import sys

import pytest

os.environ.setdefault("API_TOKEN", "test-token")

from registry import deployment
import registry
from registry import schema_hash
from registry import schema_tool
from base import LIST_TYPE, ToolArgumentError
from config import settings


def _schema_path(folder: str) -> str:
    return os.path.join(folder, schema_tool.SCHEMA_FILE)


def _rewrite(folder: str, **changes) -> None:
    with open(_schema_path(folder)) as handle:
        schema = json.load(handle)
    schema.update(changes)
    with open(_schema_path(folder), "w") as handle:
        json.dump(schema, handle)


# ----------------------------------------------------------------------
# source_hash
# ----------------------------------------------------------------------

def test_the_hash_is_stable_and_does_not_depend_on_where_the_tool_lives(make_tool_folder):
    """Two installs of the same source hash the same. If they did not, the
    check would fire on every deployment rather than on a stale schema."""
    first = make_tool_folder("tool_one")
    second = make_tool_folder("tool_two")

    assert schema_hash.hash_source_tree(
        os.path.join(first, "src")
    ) == schema_hash.hash_source_tree(os.path.join(second, "src"))


def test_editing_the_source_changes_the_hash(make_tool_folder):
    folder = make_tool_folder("edited")
    source = os.path.join(folder, "src", "_dispatch_probe.py")
    before = schema_hash.hash_source_tree(os.path.dirname(source))

    with open(source, "a") as handle:
        handle.write("\n# one more comment\n")

    assert schema_hash.hash_source_tree(os.path.dirname(source)) != before


def test_renaming_a_file_changes_the_hash(make_tool_folder):
    """The path is hashed as well as the content: a tool that moved run() into
    another module has a different signature even with identical bytes."""
    folder = make_tool_folder("renamed")
    src = os.path.join(folder, "src")
    before = schema_hash.hash_source_tree(src)

    os.rename(os.path.join(src, "_dispatch_probe.py"), os.path.join(src, "elsewhere.py"))

    assert schema_hash.hash_source_tree(src) != before


def test_compiled_python_does_not_count(make_tool_folder):
    """Importing a tool once would otherwise change its own hash."""
    folder = make_tool_folder("compiled")
    src = os.path.join(folder, "src")
    before = schema_hash.hash_source_tree(src)

    os.makedirs(os.path.join(src, "__pycache__"), exist_ok=True)
    with open(os.path.join(src, "__pycache__", "_dispatch_probe.cpython-311.pyc"), "wb") as handle:
        handle.write(b"\x00compiled")

    assert schema_hash.hash_source_tree(src) == before


def _fake_venv(folder: str) -> None:
    """A .venv whose python is this interpreter, so describe.py can be run."""
    bin_dir = os.path.join(folder, ".venv", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    os.symlink(sys.executable, os.path.join(bin_dir, "python"))


def _stub_describe(tmp_path, schema: dict) -> str:
    """Stands in for sadt-tools' scripts/describe.py: prints a schema."""
    script = tmp_path / "describe.py"
    # The JSON is embedded as a STRING literal, not as a Python dict: `true`
    # is not a Python name.
    script.write_text("print({!r})".format(json.dumps(schema)))
    return str(script)


def test_a_stale_schema_is_regenerated_rather_than_served(make_tool_folder, monkeypatch, tmp_path):
    """The schema is a CACHE of what the source says, and `source_hash` is how
    the server notices the cache is behind. Regenerating is what it is for --
    refusing to start would be right for a schema a human wrote, and this one
    is derived."""
    folder = make_tool_folder("stale", source_hash="0" * 64)
    _fake_venv(folder)
    actual = schema_hash.hash_source_tree(os.path.join(folder, "src"))
    monkeypatch.setattr(
        settings,
        "DESCRIBE_PATH",
        _stub_describe(
            tmp_path,
            {
                "name": "stale",
                "description": "regenerated",
                "arguments": {"a": {"type": "int", "required": True}},
                "returns": "text",
                "source_hash": actual,
            },
        ),
    )

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert tool.description == "regenerated"
    assert set(tool.arguments) == {"a"}


def test_a_regenerated_schema_is_cached_for_the_next_start(make_tool_folder, monkeypatch, tmp_path):
    folder = make_tool_folder("cached", source_hash="0" * 64)
    _fake_venv(folder)
    actual = schema_hash.hash_source_tree(os.path.join(folder, "src"))
    schema = {"name": "cached", "arguments": {}, "returns": "text", "source_hash": actual}
    monkeypatch.setattr(settings, "DESCRIBE_PATH", _stub_describe(tmp_path, schema))

    schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert os.path.isfile(os.path.join(settings.SCHEMA_CACHE_DIR, "cached.schema.json"))


def test_a_tool_that_cannot_describe_itself_is_skipped(make_tool_folder, monkeypatch):
    """describe.py exits 2 on anything it cannot represent rather than emitting
    a schema that is almost right, and a tool that cannot be described must not
    be served."""
    folder = make_tool_folder("undescribable", source_hash="0" * 64)

    with pytest.raises(schema_tool.SchemaError, match="cannot be generated"):
        schema_tool.load_tool(folder, deployment.DeploymentConfig({}))


def test_a_failing_generator_reports_what_it_printed(make_tool_folder, monkeypatch, tmp_path):
    folder = make_tool_folder("broken_generator", source_hash="0" * 64)
    _fake_venv(folder)
    script = tmp_path / "describe.py"
    script.write_text("import sys; print('torch is imported at module level', file=sys.stderr); sys.exit(2)")
    monkeypatch.setattr(settings, "DESCRIBE_PATH", str(script))

    with pytest.raises(schema_tool.SchemaError, match="module level"):
        schema_tool.load_tool(folder, deployment.DeploymentConfig({}))


# ----------------------------------------------------------------------
# The schema as a Tool
# ----------------------------------------------------------------------

def test_a_schema_tool_is_built_without_importing_it(make_tool_folder):
    folder = make_tool_folder("built")

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert tool.name == "built"
    assert set(tool.arguments) == {"a", "b", "out_name", "fail", "tags"}
    assert tool.arguments["a"].required and tool.arguments["out_name"].required is False
    # The declared default is a client pre-fill hint, exactly as for a ported
    # tool: it is NOT applied server-side, so an omitted argument stays out of
    # job.json and the tool's own Python default remains the only one.
    assert tool.arguments["out_name"].initial == "probe.txt"


def test_validation_is_the_same_validation(make_tool_folder):
    tool = schema_tool.load_tool(make_tool_folder("validated"), deployment.DeploymentConfig({}))

    assert tool.validate({"a": "2", "b": "3"}) == {"a": 2, "b": 3}
    with pytest.raises(ToolArgumentError, match="Missing required argument 'b'"):
        tool.validate({"a": 1})
    with pytest.raises(ToolArgumentError, match="Unexpected argument"):
        tool.validate({"a": 1, "b": 2, "c": 3})


def test_a_list_argument_crosses_the_wire_both_ways(make_tool_folder):
    """The one shape the server had no type for. A form field carries the
    comma-separated shorthand, a JSON client the array."""
    tool = schema_tool.load_tool(make_tool_folder("listy"), deployment.DeploymentConfig({}))

    assert tool.arguments["tags"].types[0] == LIST_TYPE
    assert tool.validate({"a": 1, "b": 1, "tags": "one, two"})["tags"] == ["one", "two"]
    assert tool.validate({"a": 1, "b": 1, "tags": '["one","two"]'})["tags"] == ["one", "two"]
    assert tool.validate({"a": 1, "b": 1, "tags": ["one"]})["tags"] == ["one"]


def test_a_path_argument_is_published_as_a_generic_file(make_tool_folder):
    """A schema says an argument is a path and nothing about extensions, so the
    server falls back to ALLOWED_EXTENSIONS rather than inventing a
    restriction the tool never asked for."""
    folder = make_tool_folder(
        "pathy", arguments={"scan": {"type": "path", "required": True}}
    )

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert tool.arguments["scan"].is_file
    assert tool.arguments["scan"].extensions is None


def test_a_path_argument_can_narrow_its_own_file_picker(make_tool_folder):
    """`extensions` is optional and not in the contract's example, but read if
    present: it is the only way a schema tool's file dialog can be as precise
    as an imported tool's, which declares a FILE_TYPES entry instead."""
    folder = make_tool_folder(
        "precise",
        arguments={
            "scan": {"type": "path", "required": True, "extensions": [".nii", ".nii.gz"]}
        },
    )

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert tool.arguments["scan"].extensions == (".nii", ".nii.gz")


def test_extensions_on_something_that_is_not_a_path_is_refused(make_tool_folder):
    folder = make_tool_folder("odd", arguments={"a": {"type": "int", "extensions": [".nii"]}})

    with pytest.raises(schema_tool.SchemaError, match="extensions"):
        schema_tool.load_tool(folder, deployment.DeploymentConfig({}))


def test_returns_decides_how_the_response_is_built(make_tool_folder):
    """A packaged tool returns the output DIRECTORY it was given, or a
    dict[str, Path] of named files -- both of which are zipped."""
    for declared, expected in (
        ("path", "files"),
        ("dict[str, path]", "files"),
        ("text", "text"),
    ):
        folder = make_tool_folder(f"returns_{declared}", returns=declared)
        assert schema_tool.load_tool(folder, deployment.DeploymentConfig({})).output_kind == expected


def test_an_unknown_type_is_refused(make_tool_folder):
    folder = make_tool_folder("weird", arguments={"x": {"type": "complex"}})

    with pytest.raises(schema_tool.SchemaError, match="unknown type"):
        schema_tool.load_tool(folder, deployment.DeploymentConfig({}))


def test_an_unknown_key_is_warned_about_not_refused(make_tool_folder, caplog):
    """The seam between two repositories: a field one side adds must not stop
    the other from starting, and must not vanish silently either."""
    folder = make_tool_folder("forward", arguments={"a": {"type": "int", "units": "mm"}})

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    assert "a" in tool.arguments
    assert "units" in caplog.text


def test_the_folder_must_be_named_after_the_tool(make_tool_folder):
    """dispatch.py looks the interpreter up at <TOOLS_DIR>/<tool name>/.venv,
    so a mismatch registers a tool that cannot be run."""
    folder = make_tool_folder("folder_name", name="another_name")

    with pytest.raises(schema_tool.SchemaError, match="must be named after the tool"):
        schema_tool.load_tool(folder, deployment.DeploymentConfig({}))


def test_a_schema_tool_never_runs_in_process(make_tool_folder):
    tool = schema_tool.load_tool(make_tool_folder("never"), deployment.DeploymentConfig({}))

    with pytest.raises(RuntimeError, match="no in-process implementation"):
        tool.run(a=1, b=1)


def test_a_schema_tool_runs_through_dispatch(probe_python, tracked_scratch_dirs):
    """End to end on the real thing: the probe's own .schema.json, its own
    interpreter, no import anywhere in this process."""
    folder = os.path.join(settings.TOOLS_DIR, "_dispatch_probe")

    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))
    result = tool.invoke({"a": "19", "b": "23", "tags": "a,b"})

    assert result["total"] == 42
    assert result["tags"] == ["a", "b"]
    assert os.path.realpath(result["executable"]) == os.path.realpath(probe_python)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def test_discovery_reads_schemas_and_imports_nothing(make_tool_folder):
    make_tool_folder("alpha")
    make_tool_folder("beta")

    discovered = registry._discover_schema_tools(make_tool_folder.root)

    assert sorted(name for name, _ in discovered) == ["alpha", "beta"]
    assert all(isinstance(tool, schema_tool.SchemaTool) for _, tool in discovered)


def test_an_underscored_folder_is_not_discovered(make_tool_folder):
    """How a fixture stays out of GET /tools."""
    make_tool_folder("_hidden")

    assert registry._discover_schema_tools(make_tool_folder.root) == []


def test_a_broken_schema_is_skipped_and_reported(make_tool_folder):
    make_tool_folder("fine")
    broken = make_tool_folder("broken")
    with open(_schema_path(broken), "w") as handle:
        handle.write("{not json")

    try:
        discovered = registry._discover_schema_tools(make_tool_folder.root)
        assert [name for name, _ in discovered] == ["fine"]
        assert "broken" in registry.FAILED_TOOLS
    finally:
        registry.FAILED_TOOLS.pop("broken", None)


def test_a_folder_with_a_schema_is_never_imported(make_tool_folder, monkeypatch):
    """Dropping a .schema.json is what moves a tool off the in-process path."""
    import tools as tools_package

    package_dir = tools_package.__path__[0]
    assert "Test_Tool" in [name for name, _ in registry._discover_tool_classes(package_dir)]

    monkeypatch.setattr(registry, "is_packaged", lambda folder: folder.endswith("Test_Tool"))

    assert "Test_Tool" not in [name for name, _ in registry._discover_tool_classes(package_dir)]


def test_a_schema_tool_is_published_in_the_shape_the_client_reads(make_tool_folder, monkeypatch):
    """A tool the server never imported has to arrive on the wire looking like
    any other. Every key here is one the Slicer client's form generator reads;
    the golden fixture pins the same shape for the imported ones."""
    from fastapi.testclient import TestClient

    from main import app

    folder = make_tool_folder(
        "published",
        arguments={
            "scan": {"type": "path", "required": True, "description": "A CBCT scan"},
            "structures": {"type": "list[str]", "required": False, "default": ["Mandible"]},
        },
        returns="paths",
    )
    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))
    monkeypatch.setitem(registry.TOOLS, tool.name, tool)

    entry = next(
        item for item in TestClient(app).get("/tools").json() if item["name"] == "published"
    )

    assert entry["output_kind"] == "files"
    assert entry["arguments"]["scan"] == {
        # "path", not "file": a .zip sent for one is unpacked before run() is
        # called, which a generic "file" argument must never be.
        "type": "path",
        "types": ["path"],
        "required": True,
        "description": "A CBCT scan",
        # By convention: a path argument that is not a model may be filled from
        # DATA/<tool>/testfiles/, and can still be uploaded.
        "server_selectable": "testfile",
        "choices": None,
        "initial": None,
        # null, so the client falls back to ALLOWED_EXTENSIONS: the schema
        # says "a path" and nothing about which extensions are acceptable.
        "extensions": {"path": None},
        "label": None,
        "section": None,
        "visible_when": None,
        # Narrows a choice argument's own options; null on everything else.
        "options_when": None,
        # True for a technical argument (a CUDA device, a tile step size),
        # which the tool still declares and still defaults for.
        "hidden": False,
        "ui": None,
        "groups": None,
    }
    structures = entry["arguments"]["structures"]
    assert structures["type"] == LIST_TYPE and structures["initial"] == ["Mandible"]


def test_the_registry_holds_both_kinds_at_once():
    """The migration state: tools this server imported, next to tools it never
    will. Nothing here has a schema yet, so all eight are imported ones."""
    assert set(registry.TOOLS) >= {"Test_Tool", "Example_Tool"}
    assert not any(isinstance(tool, schema_tool.SchemaTool) for tool in registry.TOOLS.values())


def test_a_packaged_path_argument_accepts_the_tool_s_own_formats(make_tool_folder, monkeypatch):
    """A schema can only say "a path", so falling back to ALLOWED_EXTENSIONS
    left every packaged tool accepting .nii alone: Surg_Mov_Pred could not be
    sent its own .csv, Crown_Seg not a .vtk, and no tool a .zip of a cohort."""
    import main

    folder = make_tool_folder("anyformat", arguments={"scan": {"type": "path", "required": True}})
    tool = schema_tool.load_tool(folder, deployment.DeploymentConfig({}))

    for filename in ("cohort.zip", "mesh.vtk", "measures.csv", "volume.nii.gz"):
        assert main._checked_extension(tool, "scan", filename), filename


# ---------------------------------------------------------------------------
# A supervised call must name a tool this server actually serves
# ---------------------------------------------------------------------------

def test_a_call_naming_a_missing_tool_stops_the_server(make_tool_folder, monkeypatch):
    """Fatal, like a stale source_hash, and for the same reason.

    Everything else discovery gets wrong costs one tool. This costs a tool that
    starts, accepts a request, runs for an hour and THEN fails on a name that
    was already wrong at deploy time -- which is what renaming ALI to
    ALI_CBCT/ALI_IOS did to its two callers.
    """
    make_tool_folder("Caller", supervisor=True, calls=["NoSuchTool"])
    monkeypatch.setattr(settings, "TOOLS_DIR", make_tool_folder.root)

    with pytest.raises(registry.UnknownSupervisedCall, match="NoSuchTool"):
        registry._build_registry()


def test_a_call_naming_a_served_tool_is_accepted(make_tool_folder, monkeypatch):
    """The other half: a chain whose members are all deployed starts."""
    make_tool_folder("Caller", supervisor=True, calls=["Callee"])
    make_tool_folder("Callee")
    monkeypatch.setattr(settings, "TOOLS_DIR", make_tool_folder.root)

    built = registry._build_registry()
    # A subset check, not equality: the in-process Test_Tool and Example_Tool
    # are always discovered too, and this test is about the chain.
    assert {"Caller", "Callee"} <= set(built)
    assert built["Caller"].calls == ("Callee",)


def test_a_tool_declaring_no_calls_is_unaffected(make_tool_folder, monkeypatch):
    """Absent is the normal case: most tools call nothing."""
    make_tool_folder("Solo")
    monkeypatch.setattr(settings, "TOOLS_DIR", make_tool_folder.root)

    built = registry._build_registry()
    assert built["Solo"].calls == ()


def test_the_server_refuses_to_start_when_every_packaged_tool_failed(
    make_tool_folder, monkeypatch
):
    """One tool failing costs one tool; ALL of them failing is the deployment.

    The shape that made this necessary: with DESCRIBE_PATH unset and no cached
    schemas, all eight packaged tools failed, the server started anyway, and it
    served Example_Tool and Test_Tool. A registry of 2 reads as a small
    deployment rather than a broken one, and a client asking for AMASSS got the
    same 404 it would get for a typo.
    """
    make_tool_folder("Broken", source_hash="0" * 64)  # never matches the source
    monkeypatch.setattr(settings, "TOOLS_DIR", make_tool_folder.root)

    with pytest.raises(registry.NoToolsLoaded, match="packaged tool"):
        registry._build_registry()


def test_some_tools_failing_still_starts(make_tool_folder, monkeypatch):
    """The other half: a broken tool beside a working one is skipped, not fatal."""
    make_tool_folder("Broken", source_hash="0" * 64)
    make_tool_folder("Fine")
    monkeypatch.setattr(settings, "TOOLS_DIR", make_tool_folder.root)

    built = registry._build_registry()
    assert "Fine" in built
    assert "Broken" not in built
