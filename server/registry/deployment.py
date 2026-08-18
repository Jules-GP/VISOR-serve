"""Per-tool server-side configuration: `deployment.toml`.

The split with `.schema.json` is deliberate and worth stating once. A tool's
schema says what its `run()` takes -- the same everywhere the tool is ever
installed, generated from its source, and hashed. This file says what THIS
deployment does with it: which arguments may be satisfied by a file already on
this server, and how large an upload this server accepts for this tool. Move
either of those into the schema and every deployment inherits one server's
paths and limits.

    [tools.amasss]
    server_selectable = { model = "model", scan = "testfile" }
    max_upload_mb = 500

Absent is the normal case: with no file at all, every `path` argument is
upload-only and MAX_UPLOAD_MB applies. Nothing here is required for a tool to
work.

The file is validated at startup rather than on first use: an entry naming an
argument that does not exist, or a `server_selectable` kind that is neither
"model" nor "testfile", would otherwise be a dropdown that silently never
appears.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

try:  # tomllib is standard from 3.11; the server targets the newest Python
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10 and older
    import tomli as tomllib

from config import settings

logger = logging.getLogger("inference_server")

# The two kinds data_store.py serves (DATA_DIR/<tool>/{models,testfiles}/).
SERVER_SELECTABLE_KINDS = ("model", "testfile")

_TOOL_KEYS = ("server_selectable", "max_upload_mb", "data_dir", "hidden", "timeout_seconds")


class DeploymentConfigError(Exception):
    """Raised at startup when deployment.toml cannot be trusted."""


@dataclass(frozen=True)
class ToolDeployment:
    """What this deployment says about one tool. All-defaults means "nothing
    said", which is the same as having no entry at all."""

    # {argument name: "model" | "testfile"}
    server_selectable: dict = field(default_factory=dict)
    # None falls back to settings.MAX_UPLOAD_MB.
    max_upload_mb: Optional[int] = None

    # The folder under DATA_DIR holding this tool's models and test files, when
    # it is not named after the tool. Packaged tools are lowercase (`amasss`)
    # while the data staged by scripts/setup-models.sh is not (`AMASSS/`), and
    # a case-insensitive lookup would be a guess -- on a case-sensitive
    # filesystem both can exist.
    data_dir: Optional[str] = None

    # Argument names a client must not render. The tool still declares them and
    # still applies its own defaults; this only says a clinician has no
    # business being asked. A deployment decision, which is why it lives here
    # and not in the tool: the tool knows nothing about who is looking at it.
    hidden: tuple = ()

    # How long this tool may run before it is killed, in seconds. None falls
    # back to settings.TOOL_TIMEOUT_SECONDS, and 0 there means "no limit".
    #
    # Per tool rather than global because the right number differs by two orders
    # of magnitude: a SurgMovPred prediction is seconds, an AMASSS cohort is
    # hours. One global value has to be set for the slowest tool, which means
    # every fast tool that hangs holds a slot until then.
    timeout_seconds: Optional[float] = None


_NOTHING_DECLARED = ToolDeployment()


class DeploymentConfig:
    def __init__(self, tools: dict):
        self._tools = tools

    def for_tool(self, tool_name: str) -> ToolDeployment:
        return self._tools.get(tool_name, _NOTHING_DECLARED)

    @property
    def configured_tools(self) -> tuple:
        return tuple(sorted(self._tools))

    def data_slug(self, tool_name: str) -> str:
        """The DATA_DIR folder to look this tool's models up in.

        The naming convention spells words out (`Batch_Dental_Seg`) while the
        data was staged run-together (`DATA/BatchDentalSeg/`). The literal name
        wins wherever it exists, so a folder that really does carry underscores
        is never mis-resolved.
        """
        declared = self.for_tool(tool_name).data_dir
        if declared:
            return declared
        if os.path.isdir(os.path.join(settings.DATA_DIR, tool_name)):
            return tool_name
        return tool_name.replace("_", "")

    def upload_limit_mb(self, tool_name: str) -> int:
        """The upload limit for this tool, in MB. Per-tool config wins; the
        global MAX_UPLOAD_MB is the default, not a ceiling."""
        limit = self.for_tool(tool_name).max_upload_mb
        return settings.MAX_UPLOAD_MB if limit is None else limit

    def timeout_seconds(self, tool_name: str) -> float:
        """How long this tool may run, in seconds; 0 means no limit.

        Per-tool config wins, and the global TOOL_TIMEOUT_SECONDS is the
        default rather than a ceiling -- an AMASSS cohort legitimately runs
        longer than anything else here, and capping it at whatever suits
        SurgMovPred would kill real work.
        """
        declared = self.for_tool(tool_name).timeout_seconds
        return settings.TOOL_TIMEOUT_SECONDS if declared is None else declared


def _tool_deployment(tool_name: str, table) -> ToolDeployment:
    where = f"deployment.toml, [tools.{tool_name}]"
    if not isinstance(table, dict):
        raise DeploymentConfigError(f"{where}: expected a table.")

    unknown = sorted(set(table) - set(_TOOL_KEYS))
    if unknown:
        # A typo here is silent otherwise: `server_selectible` would simply
        # leave every argument upload-only, with no dropdown and no error.
        raise DeploymentConfigError(
            f"{where}: unknown key(s) {unknown}. Expected any of {list(_TOOL_KEYS)}."
        )

    selectable = table.get("server_selectable", {})
    if not isinstance(selectable, dict):
        raise DeploymentConfigError(
            f"{where}: 'server_selectable' must be a table of argument name -> "
            f"{' | '.join(SERVER_SELECTABLE_KINDS)}."
        )
    for argument, kind in selectable.items():
        if kind not in SERVER_SELECTABLE_KINDS:
            raise DeploymentConfigError(
                f"{where}: argument '{argument}' is declared server_selectable as {kind!r}; "
                f"expected one of {list(SERVER_SELECTABLE_KINDS)}."
            )

    data_dir = table.get("data_dir")
    if data_dir is not None and (not isinstance(data_dir, str) or not data_dir.strip()):
        raise DeploymentConfigError(f"{where}: 'data_dir' must be a non-empty string.")

    limit = table.get("max_upload_mb")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise DeploymentConfigError(f"{where}: 'max_upload_mb' must be a positive integer.")

    hidden = table.get("hidden", ())
    if not isinstance(hidden, (list, tuple)) or not all(
        isinstance(argument, str) and argument.strip() for argument in hidden
    ):
        raise DeploymentConfigError(
            f"{where}: 'hidden' must be a list of argument names."
        )

    timeout = table.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0
    ):
        raise DeploymentConfigError(
            f"{where}: 'timeout_seconds' must be a non-negative number (0 means no limit)."
        )

    return ToolDeployment(
        server_selectable=dict(selectable),
        max_upload_mb=limit,
        data_dir=data_dir,
        hidden=tuple(hidden),
        timeout_seconds=float(timeout) if timeout is not None else None,
    )


def load(path: Optional[str] = None) -> DeploymentConfig:
    path = settings.DEPLOYMENT_CONFIG if path is None else path
    if not path or not os.path.isfile(path):
        # The normal case. Explicitly not an error: a server with no
        # deployment.toml serves every tool with upload-only inputs.
        return DeploymentConfig({})

    try:
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DeploymentConfigError(f"Cannot read {path}: {exc}")

    unknown = sorted(set(document) - {"tools"})
    if unknown:
        raise DeploymentConfigError(f"{path}: unknown top-level table(s) {unknown}. Expected [tools].")

    tools = document.get("tools", {})
    if not isinstance(tools, dict):
        raise DeploymentConfigError(f"{path}: [tools] must be a table of tool name -> settings.")

    configured = {name: _tool_deployment(name, table) for name, table in tools.items()}
    logger.info("Deployment config: %d tool(s) configured (%s)", len(configured), path)
    return DeploymentConfig(configured)


deployment_config: DeploymentConfig = load()
