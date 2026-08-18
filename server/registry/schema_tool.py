"""A tool the server has never imported.

    /tools/<name>/
    ├── .schema.json     what run() takes, and the hash of the src/ it was read from
    ├── .venv/           the tool's own interpreter and dependencies
    └── src/             the tool's code -- which this process never touches

`SchemaTool` turns that JSON into the same `Tool` object the registry has
always held, so `GET /tools`, `validate()`, `main.py`'s upload handling and
`data_store` resolution all work on it unchanged. The only thing it cannot do
is run in-process: there is nothing to import, so `invoke` always dispatches.

**The two declarations are kept apart on purpose.** `.schema.json` is
generated from the tool's source and is the same wherever the tool is
installed; `deployment.toml` is what THIS server does with it (see
deployment.py). Which arguments can be filled from this server's DATA_DIR is
not a property of the tool.

The type vocabulary is narrow by design -- `path`, `str`, `int`, `float`,
`bool`, `list[str]` -- and maps onto the server's existing one:

| schema      | ArgSpec                | GET /tools                          |
|-------------|------------------------|-------------------------------------|
| `path`      | `"file"`               | extensions null -> ALLOWED_EXTENSIONS |
| `str` ...   | `str`, `int`, ...      | unchanged                           |
| `list[str]` | `base.LIST_TYPE`       | a new type string on the wire       |

A schema cannot express what the ported tools declare today -- a specific file
type and its extensions, a catalog of choices, a section, a label, a
`visible_when`. Those arguments are published as a generic file or a plain
scalar, which is honest but plainer: a client renders a file picker with no
extension filter instead of one that only offers .nii.gz.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Optional

from base import (
    CHOICE_TYPE,
    LIST_TYPE,
    MULTICHOICE_TYPE,
    PATH_TYPE,
    ArgSpec,
    Selection,
    Tool,
    ToolSchemaError,
)
from . import conventions
from .deployment import ToolDeployment
from .schema_hash import hash_source_tree

logger = logging.getLogger("inference_server")

SCHEMA_FILE = ".schema.json"
SRC_DIR_NAME = "src"

# schema type -> the type an ArgSpec declares. "path" becomes the GENERIC file
# type: the schema says an argument is a path and nothing about which
# extensions are acceptable, so the server falls back to ALLOWED_EXTENSIONS
# rather than inventing a restriction the tool never asked for.
ARGUMENT_TYPES = {
    "path": PATH_TYPE,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    # Every list the generator can emit. Only list[str] has a widget of its
    # own; the others are accepted so a tool declaring one loads and runs
    # rather than being refused for a type the client will simply render as
    # text.
    "list[str]": LIST_TYPE,
    "list[int]": LIST_TYPE,
    "list[float]": LIST_TYPE,
    "list[bool]": LIST_TYPE,
    "list[path]": LIST_TYPE,
}

# A schema type carrying `choices` is a fixed set, and the two widgets fall
# straight out of whether it is a list: `list[Literal[...]]` is several-of,
# a bare `Literal[...]` is exactly-one. The generator derives both from the
# annotation, so there is no second declaration here that could drift.
CHOICE_BASE_TYPES = ("str",)
MULTICHOICE_BASE_TYPES = ("list[str]",)

# What `returns` means for main.py's response handling.
#   "path"  -> one output file, streamed
#   "paths" -> several, or a directory: zipped into an archive
#   "text"  -> any JSON-serializable value, returned as JSON
RETURN_KINDS = {
    # What sadt-tools' describe.py actually emits: run() returns a Path, or a
    # dict[str, Path] when it has several named outputs.
    "path": "files",
    "dict[str, path]": "files",
    # Older/other spellings, kept so a schema written by hand still loads.
    "paths": "files",
    "text": "text",
    "json": "text",
}
DEFAULT_RETURN_KIND = "text"

# Per-argument keys this server reads. `description` and `extensions` are not
# in the frozen contract's example but are read if present, so a generator can
# start emitting them without anything here changing:
#   description  a client shows it under the field; without it a panel is a
#                list of unexplained inputs.
#   extensions   [".nii", ".nii.gz"] for a path argument -- what the client's
#                file dialog offers. Without it the schema says only "a path"
#                and the dialog falls back to ALLOWED_EXTENSIONS.
#   section      the collapsible box a panel puts the field in.
#   ui           how to render a choice: "tabs", "inline", "slider".
#   groups       {tab: [options]} for a multichoice with too many options to
#                stack in one column -- ALI publishes 119 landmarks.
#   visible_when {other argument: value} -- the field is shown only then. This
#                is what keeps a tool with two modalities from asking a CBCT
#                user about intra-oral meshes.
#   label        what the field is called, when the argument name is not it.
#   hidden       never rendered, whatever the panel holds.
#
# The five presentation keys were dropped here for a while, and the effect was
# invisible from both ends: sadt-tools published them, the client read them, and
# nothing arrived. A key this server does not list is a key that silently does
# not exist, so adding one is a deliberate act -- which is why they are named
# and explained rather than passed through wholesale.
_ARGUMENT_KEYS = (
    "type", "required", "default", "description", "extensions", "choices",
    "section", "ui", "groups", "visible_when", "options_when", "label", "hidden",
)

# The output directory every tool takes as a required argument. The SERVER owns
# it -- it is the job's own output/ -- so it is filled in at dispatch time and
# never published: a client has no business picking a directory on the server,
# and a file picker for one is the single fastest way to make every run a 422.
OUTPUT_DIR_ARGUMENT = "output_dir"

# `supervisor` is a flag, not an argument: the tool calls another tool and the
# runner injects the object that lets it. Read here only so it is not reported
# as unknown, and so a deployment can be told which tools need siblings present.
_TOP_LEVEL_KEYS = (
    "name", "description", "arguments", "returns", "source_hash", "supervisor",
    "calls",
)


class SchemaError(Exception):
    """The schema cannot be used: unreadable, malformed, or declaring a type
    this server does not know. The tool is skipped and reported, the way any
    tool that fails to load is."""


def is_packaged(folder: str) -> bool:
    """Is this folder a tool the server must NOT import?

    Its own interpreter and its own source is the layout, and either half is
    enough to say so: a folder with a cached schema is one whose venv has not
    been built yet, and one with a venv is packaged whether or not its schema
    has been generated.
    """
    return os.path.isdir(os.path.join(folder, SRC_DIR_NAME)) and (
        os.path.isfile(os.path.join(folder, SCHEMA_FILE))
        or os.path.isdir(os.path.join(folder, ".venv"))
    )


def generate_schema(folder: str) -> dict:
    """Ask the tool to describe itself, with its own interpreter.

    `scripts/describe.py` lives in the sadt-tools repository and reads the
    schema out of `run()`'s signature, so the two cannot drift. It has to run
    inside the tool's virtualenv: importing a tool needs the tool's
    dependencies, which is the entire reason this server does not do it.

    A non-zero exit means the tool is not loadable -- describe.py exits 2 on
    anything it cannot represent rather than emitting a schema that is almost
    right -- and is reported as such rather than served.
    """
    from config import settings

    describe = settings.DESCRIBE_PATH
    interpreter = os.path.join(folder, ".venv", "bin", "python")
    if not os.path.isfile(describe) or not os.path.isfile(interpreter):
        raise SchemaError(
            f"{os.path.basename(folder)}: no cached {SCHEMA_FILE}, and it cannot be generated "
            f"(the tool's virtualenv or the schema generator is missing)."
        )

    completed = subprocess.run(
        [interpreter, describe, folder], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise SchemaError(
            f"{os.path.basename(folder)}: describe.py exited {completed.returncode}: "
            f"{completed.stderr.strip()[:500]}"
        )
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise SchemaError(f"{os.path.basename(folder)}: describe.py printed no usable JSON ({exc})")


def _cached_path(folder: str) -> str:
    from config import settings

    return os.path.join(settings.SCHEMA_CACHE_DIR, f"{os.path.basename(folder)}{SCHEMA_FILE}")


def _fresh(schema, actual_hash: str) -> bool:
    return isinstance(schema, dict) and schema.get("source_hash") == actual_hash


def resolve_schema(folder: str) -> dict:
    """The schema for this tool, generated if what is cached has gone stale.

    This is what `source_hash` is FOR: the tool's source is the truth, the
    schema is a derived artifact, and the hash says whether the derivation is
    still current. A stale one is regenerated rather than served -- and rather
    than taken as a reason to refuse to start, which it was when the schema
    was something a human wrote by hand.
    """
    actual = hash_source_tree(os.path.join(folder, SRC_DIR_NAME))

    for path in (os.path.join(folder, SCHEMA_FILE), _cached_path(folder)):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    schema = json.load(handle)
            except (OSError, ValueError):
                continue
            if _fresh(schema, actual):
                return schema
            logger.info(
                "%s: the cached schema describes another source; regenerating.",
                os.path.basename(folder),
            )

    schema = generate_schema(folder)
    if not _fresh(schema, actual):
        raise SchemaError(
            f"{os.path.basename(folder)}: describe.py returned a schema whose source_hash does "
            f"not match its own src/. The two sides compute it differently."
        )
    _cache(folder, schema)
    return schema


def _cache(folder: str, schema: dict) -> None:
    """Keep the generated schema, so the next start does not pay for it again.
    Failing to write one is not a failure: it costs a subprocess, not a run."""
    path = _cached_path(folder)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(schema, handle)
    except OSError as exc:
        logger.warning("Could not cache the schema for %s: %s", os.path.basename(folder), exc)


def read_schema(folder: str) -> dict:
    path = os.path.join(folder, SCHEMA_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            schema = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SchemaError(f"Cannot read {path}: {exc}")
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: expected a JSON object.")
    return schema


# Presentation, carried straight through from the tool's schema. `base.ArgSpec`
# has had these fields all along -- what was missing was anyone handing them
# over, so a tool could publish `section` and `ui`, a client could read them,
# and nothing arrived. In one function because there are four places an ArgSpec
# is built and three of them are easy to forget.
_PRESENTATION_KEYS = ("section", "ui", "groups", "visible_when", "options_when", "label")


def _presentation(declaration: dict) -> dict:
    """The layout hints a declaration carries, if any."""
    return {
        key: declaration[key]
        for key in _PRESENTATION_KEYS
        if declaration.get(key) is not None
    }


def _argument_spec(
    tool_name: str, argument_name: str, declaration, deployment: ToolDeployment
) -> ArgSpec:
    where = f"Tool '{tool_name}', argument '{argument_name}'"
    if not isinstance(declaration, dict):
        raise SchemaError(f"{where}: expected an object with at least a 'type'.")

    unknown = sorted(set(declaration) - set(_ARGUMENT_KEYS))
    if unknown:
        # Warned, not refused: this is the seam between two repositories, and a
        # field one side adds must not stop the other from starting. It still
        # has to be visible, because a silently dropped field is a feature that
        # simply never appears.
        logger.warning(
            "%s: ignoring unknown schema key(s) %s -- this server reads %s.",
            where,
            unknown,
            list(_ARGUMENT_KEYS),
        )

    declared_type = declaration.get("type")
    if declared_type not in ARGUMENT_TYPES:
        raise SchemaError(
            f"{where}: unknown type {declared_type!r}. Expected one of {list(ARGUMENT_TYPES)}."
        )

    required = declaration.get("required", True)
    if not isinstance(required, bool):
        raise SchemaError(f"{where}: 'required' must be true or false.")

    selectable = deployment.server_selectable.get(argument_name)
    if selectable is not None and declared_type != "path":
        # A server-side file standing in for a str/int argument is exactly the
        # SurgMovPred case and is legitimate -- but only the deployment can say
        # so, and only for an argument that ends up as a path in run(). Anything
        # else means the two files disagree about what this argument is.
        raise SchemaError(
            f"{where}: deployment.toml marks it server_selectable, but the tool declares it "
            f"as {declared_type!r} rather than 'path'."
        )

    # A deployment decision, not the tool's: which arguments a client renders.
    # The spec still exists and the tool still applies its own default -- this
    # only says nobody is asked. See ArgSpec.hidden.
    hidden = argument_name in deployment.hidden

    choices = declaration.get("choices")
    if choices:
        narrowed = _choice_spec(where, declared_type, declaration, choices)
        if narrowed is not None:
            narrowed.hidden = hidden
            return narrowed

    # A model is chosen from what the server hosts, never uploaded: the weights
    # do not travel, in either direction. What travels is a NAME, so the
    # argument is published as a scalar and main.py resolves it to a local path
    # before run() is called -- which is what the tool's `Path` annotation then
    # receives, unchanged.
    #
    # Published as a path instead, it becomes a file argument: the client gives
    # every file argument an input row with a local picker (see
    # formgen.file_input_modes -- "which arguments those are is the schema's
    # answer, not a module's"), and a clinician is offered to upload a model
    # bundle from their laptop.
    if selectable == "model" and declared_type == "path":
        return ArgSpec(
        **_presentation(declaration),
            type=str,
            required=required,
            description=declaration.get("description", ""),
            server_selectable=selectable,
            hidden=hidden,
        )

    accepts = declaration.get("extensions")
    if accepts is not None and declared_type != "path":
        raise SchemaError(
            f"{where}: 'extensions' describes what a path argument accepts, and this one is "
            f"declared as {declared_type!r}."
        )

    return ArgSpec(
        **_presentation(declaration),
        type=ARGUMENT_TYPES[declared_type],
        required=required,
        description=declaration.get("description", ""),
        server_selectable=selectable,
        accepts=tuple(accepts) if accepts else None,
        # Advisory, exactly as for a ported tool: the value a client pre-fills
        # its widget with. Not applied server-side -- an omitted optional
        # argument is left out of job.json entirely, so the tool's own Python
        # default applies and stays the single source of truth. Dropped for a
        # path, where there is a file picker to pre-fill and nothing to put in
        # it; a schema declaring one is not an error.
        initial=None if declared_type == "path" else declaration.get("default"),
        hidden=hidden,
    )


def _choice_spec(where: str, declared_type: str, declaration: dict, choices: list) -> ArgSpec:
    """An argument narrowed to a fixed set: a picker rather than a text field.

    The generator derives `choices` from a `Literal[...]` annotation, so the
    options live in the tool's signature and nowhere else -- which is exactly
    what the old `ArgSpec.choices` tables failed at, being a second
    declaration that drifted.

    `ArgSpec` wants {option: on by default} rather than a list plus a default,
    so the two are folded together here.
    """
    if not isinstance(choices, list) or not all(isinstance(option, str) for option in choices):
        # int options (Literal[1, 2]) are legal in a schema and have no widget
        # here: published as a plain int, which a client renders as a spin box.
        logger.warning("%s: options %r are not strings; publishing the plain type.", where, choices)
        return None

    default = declaration.get("default")
    if declared_type in MULTICHOICE_BASE_TYPES:
        selected = default if isinstance(default, list) else []
        return ArgSpec(
        **_presentation(declaration),
            type=MULTICHOICE_TYPE,
            required=declaration.get("required", True),
            description=declaration.get("description", ""),
            choices={option: option in selected for option in choices},
        )

    # Exactly one, and a combo box always shows something: with no declared
    # default the first option is the one selected, since a schema that offers
    # options offers them in a meaningful order.
    chosen = default if default in choices else choices[0]
    return ArgSpec(
        **_presentation(declaration),
        type=CHOICE_TYPE,
        required=declaration.get("required", True),
        description=declaration.get("description", ""),
        choices={option: option == chosen for option in choices},
    )


class SchemaTool(Tool):
    """A tool declared by its `.schema.json` and run in its own interpreter."""

    def __init__(self, folder: str, schema: dict, deployment: ToolDeployment):
        name = schema.get("name")
        if not name or not isinstance(name, str):
            raise SchemaError(f"{os.path.join(folder, SCHEMA_FILE)}: no 'name'.")

        unknown = sorted(set(schema) - set(_TOP_LEVEL_KEYS))
        if unknown:
            logger.warning(
                "Tool '%s': ignoring unknown schema key(s) %s -- this server reads %s.",
                name,
                unknown,
                list(_TOP_LEVEL_KEYS),
            )

        arguments = schema.get("arguments", {})
        if not isinstance(arguments, dict):
            raise SchemaError(f"Tool '{name}': 'arguments' must be an object.")

        undeclared = sorted(set(deployment.server_selectable) - set(arguments))
        if undeclared:
            raise SchemaError(
                f"Tool '{name}': deployment.toml marks {undeclared} as server_selectable, but "
                f"the tool declares no such argument(s)."
            )

        self.name = name
        self.folder = folder
        # Read and kept, but NOT published: GET /tools has no tool-level
        # description field, and adding one changes the response shape the
        # Slicer client is built against. It waits for a client release.
        self.description = schema.get("description", "")
        # The output directory is the SERVER's, not the caller's: it is the
        # job's own output/, filled in by dispatch. Taken out of the published
        # schema entirely -- a client offered a file picker for a directory on
        # the server would send something meaningless, and offered nothing
        # would make every run a 422 for a missing required argument.
        self.wants_output_dir = OUTPUT_DIR_ARGUMENT in arguments
        # The runner builds one when it sees `*, sup` in the signature; this is
        # the same fact, published, so `/tools` can say a chain is involved and
        # a deployment check can verify the siblings are actually installed.
        self.needs_supervisor = bool(schema.get("supervisor"))
        # The tools this one asks the supervisor for, as it declared them. A
        # tool cannot import another -- separate virtualenvs are the reason the
        # split exists -- so a call name is necessarily a free string, and this
        # is what lets the registry check it names something real. Verified at
        # startup, not here: it is a statement about the whole registry, and
        # this object only knows itself.
        self.calls = tuple(schema.get("calls") or ())
        self.arguments = {
            argument_name: _argument_spec(name, argument_name, declaration, deployment)
            for argument_name, declaration in arguments.items()
            if argument_name != OUTPUT_DIR_ARGUMENT
        }
        self.output_kind = RETURN_KINDS.get(schema.get("returns"), DEFAULT_RETURN_KIND)
        self.source_hash = schema.get("source_hash", "")

    def invoke(self, args: dict) -> Any:
        """Validate, then run in the tool's own interpreter.

        SADT_DISPATCH_MODE is not consulted: it decides how a tool the server
        IMPORTED is executed, and this one was never imported. There is no
        in-process path to fall back to.
        """
        cleaned = self.validate(args)
        from execution.dispatch import dispatch

        return dispatch(self, self.for_the_wire(cleaned))

    @staticmethod
    def for_the_wire(cleaned: dict) -> dict:
        """What validate() produced, in the shapes run() actually declares.

        A "multichoice" reaches an imported tool as a `Selection` -- every
        option mapped to true/false -- because that is what its `run()` took.
        A packaged tool declares `list[Literal[...]]`, so it takes the enabled
        options and nothing else. Sending the mapping would hand a dict to a
        parameter annotated as a list, and the tool would fail somewhere deep
        in its own code.
        """
        return {
            name: list(value.selected) if isinstance(value, Selection) else value
            for name, value in cleaned.items()
        }

    def run(self, **kwargs):
        raise RuntimeError(
            f"Tool '{self.name}' is declared by its {SCHEMA_FILE} and has no in-process "
            f"implementation; it runs through dispatch (see invoke)."
        )


def load_tool(folder: str, config, name: str = None) -> SchemaTool:
    """Build the tool declared by `folder`, after checking its schema is the
    one its source produced.

    `config` is the whole DeploymentConfig rather than one tool's entry,
    because which entry applies is only known once the schema has been read:
    deployment.toml is keyed by TOOL name, the name the client sends and the
    contract's `[tools.amasss]` uses.
    """
    schema = resolve_schema(folder)

    # The API name, in order: what the registry read from `[tool.sadt] name`,
    # then what the schema says, then the folder. Declaring it is what lets a
    # folder move -- under a grouping folder, say -- without the name a client
    # sends moving with it.
    folder_name = os.path.basename(folder.rstrip(os.sep))
    schema_name = schema.get("name")
    name = name or (schema_name if isinstance(schema_name, str) and schema_name else folder_name)

    if isinstance(schema_name, str) and schema_name and schema_name != name:
        # The schema is generated from the source; `[tool.sadt] name` is the
        # declared API identity. They describe the same tool, so a disagreement
        # is a mistake in one of them, and guessing which would publish a name
        # nobody chose.
        raise SchemaError(
            f"Folder '{folder_name}' declares the tool name '{name}' in its pyproject "
            f"but its schema says '{schema_name}'. Regenerate the schema, or fix "
            f"[tool.sadt] name."
        )

    if name != folder_name:
        # Still required, and `[tool.sadt] name` does not relax it: the
        # interpreter is looked up at <TOOLS_DIR>/[<group>/]<tool name>/.venv, so
        # a folder named anything else registers a tool that cannot be run. What
        # declaring the name buys is that the DEPTH may change -- a tool can move
        # under a grouping folder -- and that renaming the folder without meaning
        # to change the API name now fails here instead of silently renaming the
        # tool a client asks for.
        raise SchemaError(
            f"Tool '{name}' is installed in a folder named '{folder_name}'. The folder "
            f"must be named after the tool: its interpreter is looked up by tool name."
        )

    arguments = schema.get("arguments") or {}
    deployment = conventions.derive(arguments if isinstance(arguments, dict) else {}, config.for_tool(name))
    tool = SchemaTool(folder, schema, deployment)
    try:
        tool.check_schema()
    except ToolSchemaError as exc:
        raise SchemaError(str(exc))
    return tool


