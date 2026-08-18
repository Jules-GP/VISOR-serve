"""Discovery of the tools this server serves, from two sources.

**A tool declared by a `.schema.json`** (the one this is moving to). A folder
under TOOLS_DIR holding `.schema.json`, `.venv/` and `src/`. The server reads
the JSON, checks it against the hash of the source next to it, and builds the
tool from it -- it imports NOTHING. That is the whole point: a tool's
dependencies then have nothing to agree with the server's, and two tools
wanting incompatible versions of numpy or torch are no longer each other's
problem.

**A tool imported into this process** (what every tool here still is). A
folder under tools/ containing exactly one recognized file,
tools/<name>/<name>.py -- the file name must match the folder name -- defining
one or more Tool subclasses. Any other file in the folder is ignored by
discovery, though the recognized file is free to import from them.

Dropping a `.schema.json` into a folder is what moves a tool from the second
to the first: a folder that has one is never imported.

**A tool that will not load is SKIPPED, never fatal** -- missing dependency,
syntax error, a schema that cannot be generated. With 15+ tools, one
unavailable model must not block all the others. The failure is reported as
loudly as possible (see _record_failure), kept in FAILED_TOOLS, and surfaced
again by get_tool().

A schema whose `source_hash` no longer matches the source beside it is not a
failure at all: it is a stale cache, and it is regenerated (see
schema_tool.resolve_schema). Only a tool that can neither be read nor
described is skipped.
"""

import importlib
import logging
import os

try:  # tomllib is standard from 3.11; the server targets the newest Python
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10 and older
    import tomli as tomllib
import traceback

try:
    import tools as tools_package
except ModuleNotFoundError:
    # There is no tools/ package at all, which is not a failure: the
    # deployment image ships the server WITHOUT the in-process tools (its
    # virtualenv holds fastapi and nothing heavier), and serves everything
    # from TOOLS_DIR instead. This is also what the end state looks like once
    # every tool has been repackaged.
    tools_package = None

from base import Tool
from config import settings
from .deployment import deployment_config
from .schema_tool import SchemaTool, is_packaged, load_tool

# This module runs at import time, BEFORE main.py reaches its own
# logging.basicConfig call: without this one, the INFO summary below would be
# swallowed entirely. basicConfig is a no-op once handlers exist, so main.py's
# identical call later changes nothing.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

_BANNER = "=" * 78
_RULE = "-" * 78

# Tool FOLDER name -> one-line reason it is missing from TOOLS. Keyed by folder
# rather than by tool name because a module that fails to import never gets far
# enough to tell us the name it would have declared.
FAILED_TOOLS: dict = {}


def _record_failure(folder: str, exc: Exception) -> None:
    """Report a tool that won't be registered. Must be called from an `except`
    block: the full traceback is what makes the failure actionable."""
    reason = f"{type(exc).__name__}: {exc}"
    FAILED_TOOLS[folder] = reason
    logger.error(
        "\n%s\n"
        "  TOOL FAILED TO LOAD:  %s\n"
        "  %s\n\n"
        "  This tool is NOT registered. Every other tool still works.\n"
        "%s\n%s%s",
        _BANNER,
        folder,
        reason,
        _RULE,
        traceback.format_exc(),
        _BANNER,
    )


# What marks a directory as a tool, and where its API name is declared.
#
#     [tool.sadt]
#     tool = true
#     name = "Crown_Seg"
#
# Keyed on the SECTION rather than on the presence of a pyproject.toml, because
# a shared path dependency has one too and must not be served: tools/ALI/common/
# holds the markups writer both ALI engines import, and testkit is installed
# into every tool's dev environment. Both are importable packages, neither is a
# tool.
SADT_SECTION = "sadt"
SADT_NAME_KEY = "name"
PYPROJECT = "pyproject.toml"

# How deep discovery walks below TOOLS_DIR. 2 covers a grouping folder
# (tools/ALI/ALI_CBCT); deeper would start descending into a tool's own vendored
# directories, and a tool is never nested inside another tool.
MAX_TOOL_DEPTH = 2


def _declares_tool(folder: str):
    """`(True, declared name or None)` when this folder's pyproject says so.

    A pyproject that cannot be parsed is reported and skipped rather than
    raising: it costs one tool, like every other discovery failure, and taking
    the server down for a malformed file in one folder would be out of
    proportion. The one failure that IS fatal stays the stale source_hash.
    """
    path = os.path.join(folder, PYPROJECT)
    if not os.path.isfile(path):
        return False, None
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError) as exc:
        _record_failure(os.path.basename(folder), exc)
        return False, None
    section = (data.get("tool") or {}).get(SADT_SECTION)
    if not isinstance(section, dict) or not section.get("tool"):
        return False, None
    declared = section.get(SADT_NAME_KEY)
    return True, declared if isinstance(declared, str) and declared.strip() else None


def _tool_folders(root: str) -> list:
    """`(api name, folder path)` for every tool under `root`, in a stable order.

    Walked to MAX_TOOL_DEPTH rather than listed one level deep: ALI_CBCT and
    ALI_IOS are two tools inside tools/ALI/, which holds no tool of its own.
    Discovery does NOT descend into a folder that is already a tool.

    The API name comes from `[tool.sadt] name` when declared, and falls back to
    the directory name otherwise. That fallback is what keeps every tool that
    predates the key working; a new tool should declare it, so its API identity
    is a decision rather than an accident of directory casing.

    A leading underscore or dot still excludes a folder, which is how
    tools/_dispatch_probe/ stays out of GET /tools while remaining a usable
    fixture.
    """
    if not os.path.isdir(root):
        return []

    found = []

    def walk(directory: str, depth: int) -> None:
        try:
            entries = sorted(os.listdir(directory))
        except OSError as exc:
            _record_failure(os.path.basename(directory), exc)
            return
        for entry in entries:
            if entry.startswith("_") or entry.startswith("."):
                continue
            path = os.path.join(directory, entry)
            if not os.path.isdir(path):
                continue
            is_tool, declared = _declares_tool(path)
            if is_tool:
                # Not descended into: a tool is never nested inside another.
                found.append((declared or entry, path))
                continue
            if depth == 1:
                # A folder directly under TOOLS_DIR with no `[tool.sadt]` is
                # still a candidate: that is where an IN-PROCESS tool lives, a
                # `<name>.py` defining a Tool subclass, which has no pyproject at
                # all. `[tool.sadt]` is what a PACKAGED tool declares, and the
                # two kinds are told apart further down by is_packaged().
                found.append((entry, path))
            if depth < MAX_TOOL_DEPTH:
                walk(path, depth + 1)

    walk(root, 1)
    return sorted(found)


def _is_leftover(folder: str) -> bool:
    """Is this directory the remains of a tool git could not delete?

    Renaming tools/example_tool/ to tools/Example_Tool/ leaves the old path
    behind on every checkout where the server or the tests have run: git
    removes the tracked files but cannot remove a directory that still holds an
    untracked __pycache__/. The result is a folder with no source in it at all,
    which discovery would otherwise report as a tool that FAILED TO LOAD -- in
    a full-width banner, at every startup, on a deployment where all the tools
    are in fact fine and `git status` is clean.

    A folder with Python in it but not its <name>.py is the opposite case and
    keeps failing loudly: that one is a real tool, misnamed.
    """
    try:
        return not any(entry.endswith(".py") for entry in os.listdir(folder))
    except OSError:
        # Unreadable is not empty. Let discovery reach it and report why.
        return False


def _discover_schema_tools(root: str) -> list:
    """(folder name, SchemaTool) for every folder under `root` declaring one.

    Nothing here imports the tool: the schema is read, checked against the
    source it claims to describe, and turned into a Tool object. A mismatch is
    re-raised -- it is the one failure that takes the server with it.
    """
    discovered = []
    for name, folder in _tool_folders(root):
        if not is_packaged(folder):
            continue
        try:
            discovered.append((name, load_tool(folder, deployment_config, name=name)))
        except Exception as exc:
            _record_failure(name, exc)
    return discovered


def _discover_tool_classes(root: str) -> list:
    """(folder name, Tool subclass) pairs found under tools/.

    The folder name is carried along so a failure further down -- during
    instantiation or schema validation -- can still name the tool it came from.
    Empty when the image ships no in-process tools at all.
    """
    discovered = []
    if tools_package is None:
        return discovered

    for entry, folder_path in _tool_folders(root):
        # A tool that declares a schema is served from it and never imported.
        if is_packaged(folder_path):
            continue
        if _is_leftover(folder_path):
            continue

        try:
            expected_file = os.path.join(folder_path, f"{entry}.py")
            if not os.path.isfile(expected_file):
                raise RuntimeError(
                    f"Tool folder 'tools/{entry}/' is missing its 'tools/{entry}/{entry}.py' file."
                )
            module = importlib.import_module(f"{tools_package.__name__}.{entry}.{entry}")
        except Exception as exc:
            # Anything at all: a missing dependency, a syntax error, a
            # module-level statement that raises. None of it is worth a dead
            # server.
            _record_failure(entry, exc)
            continue

        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                discovered.append((entry, attr))

    return discovered


def _instantiate(folder: str, cls, registry: dict):
    instance = cls()
    if not instance.name:
        raise RuntimeError(f"Tool class '{cls.__name__}' has no 'name' set.")
    _reject_duplicate(instance.name, registry)
    # Catch a malformed argument declaration here, at import time, rather
    # than on the first request that happens to reach that tool.
    instance.check_schema()
    return instance


def _comparable(name: str) -> str:
    """One tool's name reduced to what does not vary between spellings.

    `Batch_Dental_Seg` and `BatchDentalSeg` are the same tool written two ways,
    so neither case nor separators may decide whether they collide.
    """
    return "".join(character for character in name.casefold() if character.isalnum())


def _reject_duplicate(name: str, registry: dict) -> None:
    if name in registry:
        raise RuntimeError(f"Duplicate tool name detected: '{name}'")


def _check_deployment_config(registry: dict) -> None:
    """Every [tools.X] in deployment.toml must name a tool this server carries.

    A typo'd or leftover section is otherwise dead config that looks live: the
    dropdown it was meant to add simply never appears, and nothing says why.
    """
    unknown = [name for name in deployment_config.configured_tools if name not in registry]
    if unknown:
        logger.warning(
            "deployment.toml configures %s, which this server does not serve. "
            "Check the tool name against GET /tools.",
            unknown,
        )
    for name in deployment_config.configured_tools:
        tool = registry.get(name)
        if tool is None or not deployment_config.for_tool(name).server_selectable:
            continue
        if not isinstance(tool, SchemaTool):
            # An imported tool declares server_selectable in its own ArgSpec,
            # where it travels with the code. Honouring it from here as well
            # would make GET /tools depend on a file the tool knows nothing
            # about.
            logger.warning(
                "deployment.toml sets server_selectable for '%s', which is an imported tool: "
                "it declares that in its own ArgSpec and this entry is ignored.",
                name,
            )



class NoToolsLoaded(RuntimeError):
    """Every packaged tool on disk failed to load. The deployment is broken."""


class UnknownSupervisedCall(RuntimeError):
    """A tool asks the supervisor for a tool this server does not serve."""


def _check_supervised_calls(registry: dict) -> None:
    """Every `sup.run("X", ...)` must name a tool this server carries.

    FATAL, and deliberately so -- the same reasoning as a schema whose
    source_hash no longer matches its source. Everything else discovery can go
    wrong about costs one tool, which is skipped and reported; this one costs a
    tool that starts, accepts a request, runs for an hour and THEN fails on a
    name that was already wrong at deploy time. Refusing to start turns that
    into a message before anyone sends a patient's scan.

    It is the check that the free-standing call name was always missing. A tool
    cannot import another -- separate virtualenvs are the reason the split
    exists -- so the name has to be a string; what it lacked was anything
    connecting that string to reality. Renaming ALI to ALI_CBCT/ALI_IOS broke
    two callers, and neither said so until run time.
    """
    broken = {}
    for name, tool in sorted(registry.items()):
        for wanted in getattr(tool, "calls", ()):
            if wanted not in registry:
                broken.setdefault(name, []).append(wanted)
    if not broken:
        return

    detail = "; ".join(
        "{} calls {}".format(caller, ", ".join(sorted(missing)))
        for caller, missing in sorted(broken.items())
    )
    raise UnknownSupervisedCall(
        "These tools ask the supervisor for tools this server does not serve: "
        "{}. Serving them would mean accepting requests that cannot finish. "
        "Either deploy the missing tools, or fix the call name.".format(detail)
    )



def _refuse_if_nothing_packaged_loaded(registry: dict) -> None:
    """Refuse to start when EVERY packaged tool on disk failed to load.

    One tool failing costs one tool -- it is skipped, reported, and the server
    serves the rest. That is right when the fault is the tool's. It is wrong
    when every one of them failed, because then the fault is not in any tool: it
    is the deployment. A missing DESCRIBE_PATH, an unwritable SCHEMA_CACHE_DIR,
    a TOOLS_DIR mounted at a path the virtualenvs were not built for -- each of
    those takes out all eight at once.

    What made that dangerous is what it looked like. The server started, logged
    its failures, and went on to serve `Example_Tool` and `Test_Tool` -- two
    fixtures. A registry of 2 reads as a small deployment, not a broken one, and
    a client asking for AMASSS got a 404 that says "unknown tool", which is the
    same answer it would get for a typo.

    Same principle as the stale-schema refusal, one level up: refusing to start
    is the only response that cannot be mistaken for working.
    """
    on_disk = [name for name, _ in _tool_folders(settings.TOOLS_DIR)]
    packaged = [
        name for name, folder in _tool_folders(settings.TOOLS_DIR) if is_packaged(folder)
    ]
    loaded_packaged = [name for name in packaged if name in registry]
    if packaged and not loaded_packaged:
        raise NoToolsLoaded(
            "None of the {} packaged tool(s) on disk could be loaded: {}. The server "
            "would start serving only its in-process fixtures, which reads as a small "
            "deployment rather than a broken one. Check DESCRIBE_PATH, that "
            "SCHEMA_CACHE_DIR is writable, and that TOOLS_DIR is mounted at the path "
            "the virtualenvs were built for. Failures: {}".format(
                len(packaged),
                ", ".join(packaged),
                "; ".join("{}: {}".format(k, v) for k, v in sorted(FAILED_TOOLS.items())),
            )
        )
    if packaged and len(loaded_packaged) < len(packaged):
        logger.error(
            "Only %d of %d packaged tool(s) on disk loaded. Missing: %s",
            len(loaded_packaged),
            len(packaged),
            ", ".join(sorted(set(packaged) - set(loaded_packaged))),
        )


def _build_registry() -> dict:
    registry: dict = {}

    for folder, tool in _discover_schema_tools(settings.TOOLS_DIR):
        try:
            _reject_duplicate(tool.name, registry)
        except Exception as exc:
            _record_failure(folder, exc)
            continue
        registry[tool.name] = tool

    schema_tools = len(registry)

    packaged = {_comparable(name): name for name in registry}

    legacy_root = tools_package.__path__[0] if tools_package is not None else ""
    for folder, cls in _discover_tool_classes(legacy_root):
        # Checked BEFORE instantiating, and that ordering is the point. When the
        # two names are identical -- which is what the naming convention makes
        # normal, `AMASSS` on both sides -- _instantiate's duplicate check fires
        # first and the tool is reported as FAILED TO LOAD, in a banner, at
        # every startup. It has not failed: it has been replaced.
        superseded = packaged.get(_comparable(getattr(cls, "name", "") or ""))
        if superseded is not None:
            logger.info(
                "Not serving imported tool '%s': superseded by the packaged '%s'. "
                "Delete server/tools/%s once that one is merged.",
                getattr(cls, "name", cls.__name__), superseded, os.path.basename(folder),
            )
            continue

        try:
            instance = _instantiate(folder, cls, registry)
        except Exception as exc:
            _record_failure(folder, exc)
            continue
        registry[instance.name] = instance

    _check_deployment_config(registry)
    _check_supervised_calls(registry)

    logger.info(
        "Tool registry: %d loaded, %d of them from a schema (%s)",
        len(registry),
        schema_tools,
        ", ".join(sorted(registry)) or "none",
    )
    _refuse_if_nothing_packaged_loaded(registry)

    if FAILED_TOOLS:
        # Repeated here, at the very end of startup, so it survives the wall of
        # uvicorn/framework output that follows and is still on screen.
        logger.error(
            "\n%s\n"
            "  %d TOOL(S) FAILED TO LOAD: %s\n"
            "  Scroll up for the full traceback of each one.\n"
            "%s",
            _BANNER,
            len(FAILED_TOOLS),
            ", ".join(sorted(FAILED_TOOLS)),
            _BANNER,
        )
    return registry


TOOLS: dict = _build_registry()


def get_tool(name: str):
    try:
        return TOOLS[name]
    except KeyError:
        if name in FAILED_TOOLS:
            # The tool exists in the source tree but didn't load. Say so rather
            # than "unknown tool", which reads like a typo. The reason stays in
            # the server logs, where it may name internal paths.
            raise KeyError(
                f"Tool '{name}' failed to load at server startup and is unavailable. "
                f"See the server logs."
            )
        raise KeyError(f"Unknown tool: '{name}'")
