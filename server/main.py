# This server processes confidential medical imaging data.
# It must be deployed in an appropriate jurisdiction (EU / a certified health
# host, depending on context) and only ever reached over TLS (see README.md).
# De-identification of patient data happens on the client side before upload;
# this server never logs file contents, argument values, or patient metadata.

import contextlib
import functools
import gzip
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
import zlib
from typing import Optional

import anyio.to_thread
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from execution import dispatch
import file_utils
from wire import transfer
from base import (
    FILE_TYPES,
    FOLDER_TYPE,
    PATH_TYPE,
    ResolvedPath,
    ToolArgumentError,
    ToolUnavailableError,
)
from config import settings
from data_store import DataNotFoundError, data_store
from registry.deployment import deployment_config
from registry import TOOLS, get_tool
from wire.security import verify_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

os.makedirs(settings.TEMP_DIR, exist_ok=True)


async def _reaper_loop() -> None:
    """Sweep expired transfer directories for as long as the server runs.

    A timer, not only the opportunistic sweep transfer.py does when a session
    is created: an abandoned upload sits longest exactly when no new request
    comes in to trigger that sweep.
    """
    while True:
        await anyio.sleep(settings.TRANSFER_SWEEP_SECONDS)
        try:
            await anyio.to_thread.run_sync(transfer.reap_expired)
        except Exception:  # noqa: BLE001 - one bad sweep must not end the loop
            logger.exception("transfer reaper sweep failed")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_reaper_loop)
        try:
            yield
        finally:
            # The loop never returns on its own: without this cancel the task
            # group would wait for it forever on shutdown.
            task_group.cancel_scope.cancel()


app = FastAPI(lifespan=_lifespan)

_CHUNK_SIZE_BYTES = 1024 * 1024  # read/write in 1 MB chunks, never load the full file into RAM
_MAX_EXTRACTED_BYTES = settings.MAX_EXTRACTED_MB * 1024 * 1024
_RESULT_REFERENCE_MIN_BYTES = settings.RESULT_REFERENCE_MIN_MB * 1024 * 1024


class _UploadTooLargeError(Exception):
    pass


_ACCEPT_ALL_EXTENSIONS = "*"

# A packaged tool's exception class NAME -> the status it means. The tools do
# not share a base class, so this maps by name, which is the convention
# sadt-tools documents:
#
#   the caller's fault, and every message is written to be read by whoever
#   sent it -- a bad structure code, a table with no patient column;
#   ToolUnavailableError -- the tool is installed, its engine is not (crownseg
#   without its `segmentation` extra), which no request can fix;
#   anything else is opaque and answers 500 with a fixed message, because a
#   crash inside a tool can name server-side paths.
TOOL_ERROR_STATUS = {
    "ToolInputError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "ValueError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "FileNotFoundError": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "ToolUnavailableError": status.HTTP_503_SERVICE_UNAVAILABLE,
}

# Form field carrying {argument name: upload id} for inputs that travelled
# through the chunked-upload endpoints instead of this request's body (see
# transfer.py). Double-underscored so it can never collide with a tool's own
# argument name, and popped before anything looks at `args`.
_UPLOADS_FIELD = "__uploads__"

# Sent by a client that would rather be handed a reference to the result and
# fetch the bytes over parallel range requests. A client that does not send it
# gets exactly the response it always got.
_RESULT_DELIVERY_HEADER = "X-Result-Delivery"
_DELIVER_BY_REFERENCE = "reference"

# Caps how many tool executions run at once (settings.MAX_CONCURRENT_TOOLS).
# Dedicated to tool runs, so waiting inference jobs never starve the threadpool
# used for everything else. Created lazily: anyio needs a running event loop to
# instantiate a CapacityLimiter.
_tool_limiter: Optional[anyio.CapacityLimiter] = None


def _get_tool_limiter() -> anyio.CapacityLimiter:
    global _tool_limiter
    if _tool_limiter is None:
        _tool_limiter = anyio.CapacityLimiter(settings.MAX_CONCURRENT_TOOLS)
    return _tool_limiter


def _extract_extension(filename: str) -> str:
    """Return the file's extension, preserving compound extensions like .nii.gz."""
    lower = filename.lower()
    parts = lower.split(".")
    if len(parts) >= 3 and parts[-1] in ("gz", "bz2", "xz"):
        return "." + ".".join(parts[-2:])
    if len(parts) >= 2:
        return "." + parts[-1]
    return ""


def _expected_extensions(tool, field_name: str) -> Optional[tuple]:
    """Return the specific extensions expected for this argument, across every
    file type it declares (see base.FILE_TYPES). None means "no specific type
    declared" -- fall back to settings.ALLOWED_EXTENSIONS.
    """
    spec = tool.arguments.get(field_name)
    if spec is None or not spec.is_file:
        return None
    # A packaged tool's "path" takes whatever the tool reads -- a .vtk mesh, a
    # .csv of measurements, a .zip of a whole cohort, which the server unpacks.
    # Its schema cannot say more than "a path", and falling back to
    # ALLOWED_EXTENSIONS here would leave every packaged tool accepting .nii
    # only: Surg_Mov_Pred could not be sent its own .csv.
    if spec.accepts is None and PATH_TYPE in spec.types:
        return (_ACCEPT_ALL_EXTENSIONS,)
    return spec.extensions


def _matched_extension(filename: str, expected: Optional[tuple]) -> Optional[str]:
    """Return the extension to use for the saved file, or None to reject it.

    `expected` is the specific extension tuple for this argument (from the
    tool's own schema) if any; otherwise settings.ALLOWED_EXTENSIONS is used.
    "*" in either list accepts every extension, preserved as-is.
    """
    candidates = expected if expected is not None else settings.ALLOWED_EXTENSIONS

    if _ACCEPT_ALL_EXTENSIONS in candidates:
        return _extract_extension(filename)

    lower = filename.lower()
    for extension in candidates:
        if lower.endswith(extension):
            return extension
    return None


async def _stream_to_disk(upload: UploadFile, destination: str, max_bytes: int) -> int:
    """Write the upload to disk in chunks, never buffering the whole file in RAM."""
    size = 0
    with open(destination, "wb") as out_file:
        while chunk := await upload.read(_CHUNK_SIZE_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise _UploadTooLargeError()
            out_file.write(chunk)
    return size


def _upload_limit_mb(tool) -> int:
    """The upload limit for this tool: deployment.toml's `max_upload_mb` when
    it declares one, MAX_UPLOAD_MB otherwise.

    Applied here rather than in POST /uploads because that endpoint opens a
    session for a file, not for a tool, and does not know which tool the bytes
    are for. The chunked path is therefore bounded by the global limit while
    the transfer runs, and by the tool's own the moment it is claimed below.
    """
    return deployment_config.upload_limit_mb(tool.name)


def _type_name(arg_type) -> str:
    return arg_type if isinstance(arg_type, str) else arg_type.__name__


def _resolved_kind(spec, path: str) -> str:
    """Which declared type a path on disk corresponds to, for ResolvedPath.kind."""
    if os.path.isdir(path):
        return FOLDER_TYPE
    if not spec.is_file:
        return "file"
    return spec.match_type(_extract_extension(os.path.basename(path)))


def _extract_folder_argument(spec, archive_path: str, work_dir: str, field_name: str) -> str:
    """Extract an archive sent for a "folder"-typed argument, so run() gets a
    directory. HTTP has no notion of a folder: the client zips it, the server
    unpacks it here, and the tool never sees the archive step at all.
    """
    try:
        return file_utils.extract_zip(
            archive_path,
            os.path.join(work_dir, f"{field_name}_folder"),
            strip_single_root=True,
            max_total_bytes=_MAX_EXTRACTED_BYTES,
        )
    except file_utils.BadArchiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Argument '{field_name}': {exc}",
        )


def _temp_root_of(path: str) -> Optional[str]:
    """Top-level folder under settings.TEMP_DIR containing `path`, or None if
    `path` lives outside TEMP_DIR. Used to clean up a tool's own scratch dir
    (see file_utils.make_scratch_dir) once its output has been streamed."""
    temp_dir = os.path.realpath(settings.TEMP_DIR)
    resolved = os.path.realpath(path)
    if os.path.commonpath([temp_dir, resolved]) != temp_dir or resolved == temp_dir:
        return None
    relative = os.path.relpath(resolved, temp_dir)
    return os.path.join(temp_dir, relative.split(os.sep)[0])


def _discard(work_dir: Optional[str], scratch_dirs: list) -> None:
    """Remove everything this request created, right now. Used on the error
    paths, where no response will ever stream and background tasks won't run."""
    for directory in ([work_dir] if work_dir else []) + list(scratch_dirs):
        shutil.rmtree(directory, ignore_errors=True)


def _media_type_of(path: str) -> str:
    """Content-Type for a file about to be streamed. Derived from the real
    extension so an .xlsx is never mislabeled as a generic zip; the
    gzip/octet-stream fallback covers bare .gz files (e.g. .nii.gz), which
    mimetypes cannot name."""
    media_type, _ = mimetypes.guess_type(str(path))
    if media_type is None:
        media_type = "application/gzip" if str(path).endswith(".gz") else "application/octet-stream"
    return media_type


def _human_bytes(size: int) -> str:
    """Byte count in the largest unit that keeps it readable. Logged alongside
    the exact figure, never instead of it."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def _log_served(tool_name: str, start_time: float, received: int, sent: Optional[int]) -> None:
    """One line per successfully served request.

    Called at each return point rather than before them, so `duration` covers
    packing the response too -- zipping a multi-GB segmentation is not free.
    `sent` is None for a "text" tool, whose result travels as JSON. Nothing
    here may name a file, an argument value, or patient metadata (see the note
    at the top of this module).
    """
    duration = time.monotonic() - start_time
    sent_field = (
        "" if sent is None else f" sent={sent}B ({_human_bytes(sent)})"
    )
    logger.info(
        "endpoint=/run/%s status=200 duration=%.2fs received=%dB (%s)%s",
        tool_name,
        duration,
        received,
        _human_bytes(received),
        sent_field,
    )


def _output_roots(outputs: list, work_dir: str) -> set:
    """TEMP_DIR folders holding the tool's outputs, excluding the request's own
    work dir (already scheduled for cleanup by the caller)."""
    work_dir_real = os.path.realpath(work_dir)
    roots = set()
    for path in outputs:
        root = _temp_root_of(path if os.path.isdir(path) else os.path.dirname(path))
        if root is not None and root != work_dir_real:
            roots.add(root)
    return roots


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _extensions_of(spec) -> Optional[dict]:
    """{type name: [extension, ...]} for every file type an argument accepts.

    Published so a client never has to mirror FILE_TYPES. Keyed by type rather
    than flattened because the caller needs the split: the extensions of
    "folder" are what a zipped folder may be uploaded as, not what its file
    picker should offer.
    """
    per_type = {}
    for declared in spec.types:
        name = _type_name(declared)
        if name in FILE_TYPES:
            # ArgSpec.accepts overrides the declared type's own list: it is how
            # an argument declared as a generic file (a .schema.json can only
            # say "path") still tells the client what it reads.
            extensions = FILE_TYPES[name] if spec.accepts is None else spec.accepts
            per_type[name] = list(extensions) if extensions else None
    return per_type or None


@app.get("/tools")
def list_tools() -> list:
    """Let clients discover every registered tool and its expected arguments."""
    return [
        {
            "name": tool.name,
            "arguments": {
                arg_name: {
                    # "type" stays a single string for clients that predate
                    # multi-type arguments; "types" is the full list and is
                    # what a client should read to build its file picker.
                    "type": _type_name(spec.types[0]),
                    "types": [_type_name(declared) for declared in spec.types],
                    "required": spec.required,
                    "description": spec.description,
                    "server_selectable": spec.server_selectable,
                    # For "choice"/"multichoice": the options to render, each
                    # with its initial state. null for every other type.
                    "choices": spec.choices,
                    # For a SCALAR argument: the value a client should pre-fill
                    # its widget with, so a spin box does not start at Qt's 0
                    # while the tool's own default reads 5.
                    "initial": spec.initial,
                    # {type name: accepted extensions}, so a client builds its
                    # file dialog filters without a copy of FILE_TYPES on its
                    # side. null for the generic "file" type (which falls back
                    # to ALLOWED_EXTENSIONS) and for a non-file argument.
                    "extensions": _extensions_of(spec),
                    # Presentation hints (see ArgSpec): how a client lays this
                    # argument out and when to show it. All null on a tool that
                    # declares none, so its panel renders exactly as before.
                    "label": spec.label,
                    "section": spec.section,
                    "visible_when": spec.visible_when,
                    "options_when": spec.options_when,
                    # True: do not render this at all. Still published, because
                    # a client that hides it must still know it exists rather
                    # than treat it as an argument the server invented.
                    "hidden": spec.hidden,
                    "ui": spec.ui,
                    # Listed explicitly so the wire shape does not depend on
                    # whether a tool spelled its catalog as a tuple or a list.
                    "groups": (
                        {name: list(options) for name, options in spec.groups.items()}
                        if spec.groups
                        else None
                    ),
                }
                for arg_name, spec in tool.arguments.items()
            },
            "output_kind": tool.output_kind,
        }
        for tool in TOOLS.values()
    ]


@app.get("/tools/{tool_name}/data", dependencies=[Depends(verify_token)])
def list_tool_data(tool_name: str) -> dict:
    """List models and test files available on the server for this tool, so
    a client can pick one instead of uploading its own (see ArgSpec.server_selectable).
    """
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    slug = deployment_config.data_slug(tool.name)
    return {
        "models": data_store.list_models(slug),
        "testfiles": data_store.list_testfiles(slug),
    }


def _remove_path(path: str) -> None:
    """Remove a file or a whole directory tree — for backend-materialized temp
    copies (ResolvedFile.is_temporary), which can be either."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


@app.get("/tools/{tool_name}/testfiles/{filename}", dependencies=[Depends(verify_token)])
async def download_testfile(tool_name: str, filename: str, background_tasks: BackgroundTasks):
    """Stream one of the tool's hosted test files, so a user can fill an input
    with reference data. The valid names are what GET /tools/{name}/data lists.

    Only test files are downloadable. Models are deliberately NOT: they are
    selected by name and used in place (see ArgSpec.server_selectable).

    A test entry that is a FOLDER is zipped on the fly and the client unpacks
    it back into a directory on its side.
    """
    start_time = time.monotonic()
    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        resolved = data_store.resolve_testfile(deployment_config.data_slug(tool.name), filename)
    except DataNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if resolved.is_temporary:
        background_tasks.add_task(_remove_path, resolved.path)

    path = resolved.path
    if os.path.isdir(path):
        # DATA_DIR is read-only: the archive is built in its own staging dir
        # under TEMP_DIR, which must outlive the response stream — hence the
        # background task, and the inline cleanup on the one path where no
        # response (and so no background task) will ever run.
        staging_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
        archive_name = f"{os.path.basename(filename)}.zip"
        try:
            path = await anyio.to_thread.run_sync(
                file_utils.make_zip, path, os.path.join(staging_dir, archive_name)
            )
        except Exception:
            logger.exception("endpoint=/tools/%s/testfiles status=500 (packing)", tool_name)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Could not package the test folder.")
        background_tasks.add_task(shutil.rmtree, staging_dir, ignore_errors=True)

    # Same log shape as /run: tool, status, duration, size — never the file
    # name (see the confidentiality note at the top of this module).
    size = os.path.getsize(path)
    logger.info(
        "endpoint=/tools/%s/testfiles status=200 duration=%.2fs sent=%dB (%s)",
        tool_name,
        time.monotonic() - start_time,
        size,
        _human_bytes(size),
    )
    return FileResponse(
        path,
        media_type=_media_type_of(path),
        filename=os.path.basename(path),
        background=background_tasks,
    )


# ----------------------------------------------------------------------
# Chunked upload / range-served results (see transfer.py for the why)
# ----------------------------------------------------------------------

class _NewUpload(BaseModel):
    filename: str
    size: int
    chunk_size: Optional[int] = None


def _transfer_error(exc: transfer.TransferError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@app.post("/uploads", dependencies=[Depends(verify_token)])
async def create_upload(spec: _NewUpload) -> dict:
    """Open a session the client then fills with parallel PUTs.

    Answering with `chunk_size` rather than accepting the client's keeps the
    layout single-sourced: part n is always
    `[n * chunk_size, (n+1) * chunk_size)`, computed by both sides from the one
    number returned here.
    """
    try:
        session = await anyio.to_thread.run_sync(
            functools.partial(
                transfer.create_upload, spec.filename, spec.size, spec.chunk_size
            )
        )
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    return {
        "upload_id": session.upload_id,
        "chunk_size": session.chunk_size,
        "part_count": session.part_count,
    }


@app.get("/uploads/{upload_id}", dependencies=[Depends(verify_token)])
async def upload_status(upload_id: str) -> dict:
    """What is still missing, this is what makes a transfer resumable: a
    client coming back after a dropped connection sends only these parts."""
    try:
        session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
        missing = await anyio.to_thread.run_sync(session.missing_parts)
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    return {
        "upload_id": session.upload_id,
        "size": session.size,
        "chunk_size": session.chunk_size,
        "part_count": session.part_count,
        "missing_parts": missing,
    }


@app.put("/uploads/{upload_id}/parts/{index}", dependencies=[Depends(verify_token)])
async def upload_part(upload_id: str, index: int, request: Request) -> dict:
    """Receive one part, verify it, write it at its offset.

    The body is the raw bytes, no multipart framing: there is exactly one thing
    in it. `Content-Encoding: gzip` is honoured for inputs not already
    compressed (an uncompressed .nii or .vtk is 3-4x smaller deflated), and
    `X-Part-SHA256` is checked against what lands on disk either way.
    """
    body = await request.body()
    if request.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            body = await anyio.to_thread.run_sync(gzip.decompress, body)
        except (OSError, EOFError, zlib.error) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Part {index} is not readable gzip: {exc}",
            )
    try:
        session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
        remaining = await anyio.to_thread.run_sync(
            functools.partial(
                transfer.write_part,
                session,
                index,
                body,
                request.headers.get("X-Part-SHA256"),
            )
        )
    except transfer.TransferError as exc:
        raise _transfer_error(exc)
    # An upload that is still moving must never be reaped, however long it
    # takes. This is what makes TRANSFER_TTL_SECONDS an idle timeout.
    await anyio.to_thread.run_sync(transfer.touch, session.directory)
    return {"received": index, "missing_count": remaining}


@app.delete("/uploads/{upload_id}", dependencies=[Depends(verify_token)])
async def delete_upload(upload_id: str) -> dict:
    await anyio.to_thread.run_sync(transfer.discard_upload, upload_id)
    return {"status": "ok"}


@app.get("/results/{result_id}", dependencies=[Depends(verify_token)])
async def download_result(result_id: str, request: Request):
    """Serve a stored result, honouring `Range`.

    That header is the whole point: it lets the client pull one file down over
    several connections at once. A client that sends no Range still gets the
    entire file in one response.
    """
    try:
        stored = await anyio.to_thread.run_sync(transfer.get_result, result_id)
    except transfer.TransferError as exc:
        raise _transfer_error(exc)

    try:
        span = transfer.parse_range(request.headers.get("Range"), stored.size)
    except transfer.TransferError as exc:
        # The size is what the client got wrong, so the real one has to travel
        # with the refusal, otherwise it can only guess again.
        return JSONResponse(
            {"detail": str(exc)},
            status_code=exc.status_code,
            headers={"Content-Range": f"bytes */{stored.size}"},
        )

    # Stamped before the body streams, not after: a download in progress is a
    # download that must survive the reaper, and the next range may be minutes
    # away on a slow link.
    await anyio.to_thread.run_sync(transfer.touch, stored.directory)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{stored.filename}"',
    }
    if span is None:
        start, end, code = 0, stored.size - 1, status.HTTP_200_OK
    else:
        start, end = span
        code = status.HTTP_206_PARTIAL_CONTENT
        headers["Content-Range"] = f"bytes {start}-{end}/{stored.size}"
    headers["Content-Length"] = str(max(0, end - start + 1))

    return StreamingResponse(
        transfer.read_range(stored.blob_path, start, end),
        status_code=code,
        media_type=stored.media_type,
        headers=headers,
    )


@app.delete("/results/{result_id}", dependencies=[Depends(verify_token)])
async def delete_result(result_id: str) -> dict:
    """Sent by a client that has the whole file. Not required for correctness
    -- the reaper collects what is never claimed -- but it keeps TEMP_DIR flat
    under load instead of holding every result for the full TTL."""
    await anyio.to_thread.run_sync(transfer.discard_result, result_id)
    return {"status": "ok"}


def _upload_references(raw) -> dict:
    """{argument name: upload id} from the request's `__uploads__` field."""
    if not raw:
        return {}
    try:
        references = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed '{_UPLOADS_FIELD}' field: {exc}",
        )
    if not isinstance(references, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in references.items()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{_UPLOADS_FIELD}' must be an object of argument name -> upload id.",
        )
    return references



# Characters kept from a client-supplied filename. Everything else is dropped
# rather than escaped: this string becomes a path component on the server's
# disk, and a whitelist is the only form of that decision which cannot be
# reasoned around.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
# How much of the name survives. Long enough for a real patient identifier,
# short enough that it cannot push the path past a filesystem limit.
_MAX_STEM = 64


def _safe_stem(filename: str, extension: str) -> str:
    """The patient-identifying part of an upload's name, made safe to write.

    Sanitized, NOT discarded, and the distinction is clinical. Naming the temp
    file after the form field alone -- which is what this replaces -- meant every
    scan in a batch arrived as `scans.nii.gz`, so every tool that names its
    outputs after its input handed back `scans_Pred_MAND.nii.gz` for every
    patient. Identity survived only in the order the requests were made. Measured
    on three unrelated tools (AMASSS, Batch_Dental_Seg, Crown_Seg) before being
    fixed here, once, where the name is chosen.

    The real risk was never the name's presence, it is writing an unsanitized
    client string to disk. So:

    - the declared extension is removed first, not `Path.stem`, which only strips
      the last suffix and would leave `.nii` on a `.nii.gz`;
    - everything outside `[A-Za-z0-9_.-]` is dropped, which removes separators
      and control characters outright;
    - leading dots go, so nothing becomes a hidden file or a relative path;
    - a result that is empty, or that is all dots (`.`, `..`), returns "" and the
      caller falls back to the field name alone. Traversal cannot survive a
      whitelist that excludes `/`, but `..` is refused explicitly because it is
      the one leftover that is still a meaningful path.
    """
    name = os.path.basename(filename or "")
    if extension and name.lower().endswith(extension.lower()):
        name = name[: -len(extension)]
    cleaned = _SAFE_STEM.sub("_", name).strip("._")
    if not cleaned or set(cleaned) <= {"."}:
        return ""
    return cleaned[:_MAX_STEM]


def _checked_extension(tool, field_name: str, filename: str) -> str:
    """The extension an input will be saved under, or a 400 naming what was
    allowed. Shared by the multipart path and the chunked one so an upload is
    validated identically however its bytes arrived."""
    expected = _expected_extensions(tool, field_name)
    extension = _matched_extension(filename or "", expected)
    if extension is None:
        allowed = expected if expected is not None else settings.ALLOWED_EXTENSIONS
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension for '{field_name}'. Allowed: {allowed}",
        )
    return extension


def _reject_upload_for_scalar(spec, field_name: str) -> None:
    """A scalar-typed argument must never arrive as a file: a server-side-only
    model (ArgSpec(type=str, server_selectable="model")) is selected by name.
    Without this check the uploaded file's temp path would silently become the
    argument's string value."""
    if spec is not None and not spec.is_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Argument '{field_name}' expects a plain value, not an uploaded file.",
        )


def _unpacks_to_a_folder(kind: str, extension: str) -> bool:
    """Does this input arrive as an archive that must be extracted first?

    Always for a "folder" argument, and for a "path" argument that received a
    .zip: no packaged tool unpacks archives any more -- each used to carry its
    own extraction, zip-bomb cap and scratch directory, which is exactly the
    duplication moving them out removed.
    """
    return kind == FOLDER_TYPE or (kind == PATH_TYPE and extension.lower() == ".zip")


async def _as_resolved_path(spec, input_path: str, extension: str, work_dir: str, field_name: str):
    """Tag an input with the declared type it actually is, unpacking an archive
    so run() only ever sees a real file or directory."""
    kind = spec.match_type(extension) if spec is not None and spec.is_file else "file"
    if not _unpacks_to_a_folder(kind, extension):
        return ResolvedPath(input_path, kind)
    extracted = await anyio.to_thread.run_sync(
        functools.partial(_extract_folder_argument, spec, input_path, work_dir, field_name)
    )
    return ResolvedPath(extracted, FOLDER_TYPE)


@app.post("/run/{tool_name}", dependencies=[Depends(verify_token)])
async def run_tool(tool_name: str, request: Request, background_tasks: BackgroundTasks):
    start_time = time.monotonic()

    try:
        tool = get_tool(tool_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # Generic argument collection: whatever scalar fields and/or files the
    # caller sends, whichever tool it targets. Each uploaded file is matched to
    # the tool's argument of the same name; the tool's own schema (validated in
    # tool.invoke) decides what is actually accepted.
    form = await request.form()
    args: dict = {}
    uploaded_files: dict = {}
    for key, value in form.multi_items():
        if isinstance(value, StarletteUploadFile):
            uploaded_files[key] = value
        else:
            args[key] = value

    # Inputs that came up through the chunked-upload endpoints reference their
    # session here instead of carrying their bytes in this request. Popped
    # before `args` is looked at, so it can never reach a tool as an argument.
    upload_references = _upload_references(args.pop(_UPLOADS_FIELD, None))

    # An argument declared with ArgSpec(server_selectable=...) can be sent as a
    # plain form value (the file name) instead of an upload, resolved below
    # into a path already on the server (see data_store.py). Pulled out of
    # `args` before the upload loop so a genuine upload for the same field name
    # is never mistaken for one.
    server_file_args: dict = {}
    for field_name in list(args):
        spec = tool.arguments.get(field_name)
        if spec is not None and spec.server_selectable:
            server_file_args[field_name] = args.pop(field_name)

    work_dir = None
    input_paths = []
    resolved_files = []
    size = 0
    upload_limit_mb = _upload_limit_mb(tool)
    upload_limit_bytes = upload_limit_mb * 1024 * 1024

    if uploaded_files or upload_references:
        work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

    try:
        for field_name, upload in uploaded_files.items():
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            extension = _checked_extension(tool, field_name, upload.filename or "")
            # The field name stays as a prefix so the argument a file belongs to
            # is still readable; the patient's own name follows it, so a batch's
            # outputs can be told apart without counting requests.
            stem = _safe_stem(upload.filename or "", extension)
            base = f"{field_name}_{stem}" if stem else field_name
            input_path = os.path.join(work_dir, f"{base}{extension}")
            try:
                size += await _stream_to_disk(upload, input_path, upload_limit_bytes)
            except _UploadTooLargeError:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {upload_limit_mb} MB limit.",
                )
            input_paths.append(input_path)
            # An argument can accept several types (e.g. ("csv_file",
            # "folder")): decide here which one this upload is and tag the path
            # with it. A "folder" arrives zipped and is unpacked now.
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )

        # Same treatment, for the inputs whose bytes are already on disk: the
        # session's blob is RENAMED into the work dir rather than copied, so a
        # chunked upload costs no extra pass over the file at all.
        for field_name, upload_id in upload_references.items():
            spec = tool.arguments.get(field_name)
            _reject_upload_for_scalar(spec, field_name)
            try:
                session = await anyio.to_thread.run_sync(transfer.get_upload, upload_id)
                extension = _checked_extension(tool, field_name, session.filename)
                # The tool is only known here, so this is where a per-tool
                # limit lower than the global one is enforced -- before the
                # blob is claimed, never after.
                if session.size > upload_limit_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {upload_limit_mb} MB limit.",
                    )
                input_path = os.path.join(work_dir, f"{field_name}{extension}")
                await anyio.to_thread.run_sync(transfer.claim_upload, upload_id, input_path)
            except transfer.TransferError as exc:
                raise _transfer_error(exc)
            size += session.size
            input_paths.append(input_path)
            args[field_name] = await _as_resolved_path(
                spec, input_path, extension, work_dir, field_name
            )
    except HTTPException:
        # Nothing is queued for cleanup yet and no response will stream, so the
        # work dir goes now. So do the unclaimed sessions: the reaper would get
        # them eventually, but that is a TTL's worth of confidential imaging
        # sitting on disk.
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        for upload_id in upload_references.values():
            await anyio.to_thread.run_sync(transfer.discard_upload, upload_id)
        raise

    for field_name, filename in server_file_args.items():
        spec = tool.arguments[field_name]
        resolver = data_store.resolve_model if spec.server_selectable == "model" else data_store.resolve_testfile
        try:
            # Not tool.name: the packaged tools are lowercase while the data
            # staged under DATA/ is not, so deployment.toml maps the two.
            resolved = resolver(deployment_config.data_slug(tool.name), filename)
        except DataNotFoundError as exc:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        resolved_files.append(resolved)

        # Server-side data can be a real folder or a single file; tag it the
        # same way as an upload so run() branches on .kind either way. An
        # archive standing in for a "folder" argument is unpacked here too, so
        # the two routes stay indistinguishable from the tool's point of view.
        kind = _resolved_kind(spec, resolved.path)
        path = resolved.path
        if not os.path.isdir(path) and _unpacks_to_a_folder(
            kind, _extract_extension(os.path.basename(path))
        ):
            # DATA_DIR is read-only: extract into the request's own work dir.
            if work_dir is None:
                work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)
            try:
                path = await anyio.to_thread.run_sync(
                    functools.partial(_extract_folder_argument, spec, path, work_dir, field_name)
                )
            except HTTPException:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise
        args[field_name] = ResolvedPath(path, kind)

    # Anything the tool creates through file_utils.make_scratch_dir() lands
    # here, so it can be removed even if run() raises before returning a path.
    scratch_dirs = file_utils.track_scratch_dirs()

    try:
        # Run the tool in a worker thread, NOT on the event loop: tool.invoke
        # is synchronous CPU-bound work and would otherwise freeze the whole
        # server -- even /health -- for its entire duration. Concurrency is
        # bounded by MAX_CONCURRENT_TOOLS and safe: tools are stateless
        # (everything arrives via args), each request gets its own work_dir,
        # and DATA_DIR is read-only.
        result = await anyio.to_thread.run_sync(
            tool.invoke, args, limiter=_get_tool_limiter()
        )
    except ToolArgumentError as exc:
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    except dispatch.ToolFailure as exc:
        # The tool raised and named its exception class. There is no shared
        # exception type to isinstance-check -- there is no shared package --
        # so the NAME decides, and only the names that mean "the caller can fix
        # this" let their message through.
        code = TOOL_ERROR_STATUS.get(exc.error_type, 500)
        logger.warning("endpoint=/run/%s status=%d error=%s", tool_name, code, exc.error_type)
        if code == 500:
            # The ONLY place this exists. A 4xx carries its message to the
            # caller, but a 500 answers "Tool execution failed." and the job
            # directory holding stderr.log is discarded on the next line, so
            # without this the tool's traceback is gone -- which is exactly
            # what makes a failing tool undiagnosable from the outside.
            # Server-side only, as _stderr_tail intends.
            logger.error("endpoint=/run/%s failure:\n%s", tool_name, exc.message)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(
            status_code=code,
            detail=exc.message if code != 500 else "Tool execution failed.",
        )
    except ToolUnavailableError as exc:
        # The request is fine; this deployment cannot serve it (a dependency the
        # image does not carry). 501 rather than 500: nothing the caller changes
        # will help, and the reason names a missing package, never a path.
        logger.warning("endpoint=/run/%s status=501", tool_name)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))
    except Exception:
        logger.exception("endpoint=/run/%s status=500", tool_name)
        _discard(work_dir, scratch_dirs)
        raise HTTPException(status_code=500, detail="Tool execution failed.")
    finally:
        # Uploaded inputs are never needed again past this point.
        for input_path in input_paths:
            if os.path.exists(input_path):
                os.remove(input_path)
        # Server-side data (DATA_DIR) is persistent and must never be
        # deleted; only backend-materialized temp copies are (see
        # ResolvedFile.is_temporary in data_store.py).
        for resolved in resolved_files:
            if resolved.is_temporary and os.path.exists(resolved.path):
                os.remove(resolved.path)

    if tool.output_kind in ("file", "segmentation", "files"):
        # `result` is a path to the output file the tool wrote -- or, for
        # "files", a list of paths / a single directory to bundle into a zip.
        if work_dir is None:
            work_dir = tempfile.mkdtemp(dir=settings.TEMP_DIR)

        # A tool whose inputs all came from the read-only data store writes its
        # output in its own scratch dir under TEMP_DIR, which must be cleaned up
        # too -- whether the response goes out or the packing below fails.
        # `scratch_dirs` covers file_utils.make_scratch_dir(); _output_roots
        # also catches a tool that wrote under TEMP_DIR by hand.
        try:
            outputs = file_utils.output_paths(result)
            if not outputs:
                raise ValueError(
                    f"Tool '{tool.name}' declares output_kind={tool.output_kind!r} but "
                    f"run() returned {type(result).__name__}, not a path (or a list of paths)."
                )
            if tool.output_kind != "files":
                # One file goes back, so there has to be exactly one. Taking
                # the first of several silently would return part of a result
                # and call it a success.
                if len(outputs) > 1:
                    raise ValueError(
                        f"Tool '{tool.name}' declares output_kind={tool.output_kind!r} but "
                        f"returned {len(outputs)} paths. Declare 'files' to return several."
                    )
                # Normalized rather than reused as-is: `result` may have been a
                # mapping of named outputs, and what is streamed is a path.
                result = outputs[0]
            output_roots = _output_roots(outputs, work_dir) | set(scratch_dirs)
            if tool.output_kind == "files":
                # Built inside work_dir, never inside the tool's own scratch
                # dir: the archive has to outlive the files it was made from,
                # right up until the response has finished streaming.
                archive_name = (
                    f"{os.path.basename(outputs[0].rstrip(os.sep))}.zip"
                    if len(outputs) == 1 and os.path.isdir(outputs[0])
                    else f"{tool.name}_output.zip"
                )
                result = await anyio.to_thread.run_sync(
                    file_utils.make_zip, outputs, os.path.join(work_dir, archive_name)
                )
        except Exception:
            logger.exception("endpoint=/run/%s status=500 (packing output)", tool_name)
            _discard(
                work_dir,
                _output_roots(file_utils.output_paths(result), work_dir) | set(scratch_dirs),
            )
            raise HTTPException(status_code=500, detail="Tool execution failed.")

        media_type = _media_type_of(str(result))

        # A client asking for reference delivery gets the result MOVED out of
        # the work dir (a rename, not a copy) and a JSON pointer to it, so it
        # can pull the bytes over several range requests. Done before the
        # cleanup tasks are queued, which would otherwise take the file away.
        #
        # Only above RESULT_REFERENCE_MIN_MB, for cleanup rather than speed: a
        # streamed response deletes its file the moment the response ends, with
        # no dependency on the client, while a reference waits for a DELETE or
        # for the reaper. Parallel ranges buy nothing on a small result.
        deliver_by_reference = (
            request.headers.get(_RESULT_DELIVERY_HEADER, "").lower() == _DELIVER_BY_REFERENCE
            and os.path.getsize(result) >= _RESULT_REFERENCE_MIN_BYTES
        )
        stored = None
        if deliver_by_reference:
            try:
                stored = await anyio.to_thread.run_sync(
                    transfer.store_result, str(result), media_type
                )
            except OSError:
                # Reference delivery is an optimisation; failing it must not
                # fail a run that has already done the expensive part. Falls
                # through to streaming the file the way it always did.
                logger.exception("endpoint=/run/%s (storing result by reference)", tool_name)

        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
        for output_root in output_roots:
            background_tasks.add_task(shutil.rmtree, output_root, ignore_errors=True)

        if stored is not None:
            _log_served(tool_name, start_time, size, stored.size)
            return JSONResponse({"result_ref": stored.as_reference()}, background=background_tasks)

        # The size of the file about to be streamed. Measured rather than
        # accumulated: for output_kind="files" what goes out is the archive
        # built just above, not the sum of what run() produced.
        _log_served(tool_name, start_time, size, os.path.getsize(result))
        return FileResponse(
            result,
            media_type=media_type,
            filename=os.path.basename(result),
            background=background_tasks,
        )

    # A "text" tool can still have written scratch files along the way.
    for directory in ([work_dir] if work_dir else []) + list(scratch_dirs):
        background_tasks.add_task(shutil.rmtree, directory, ignore_errors=True)
    _log_served(tool_name, start_time, size, None)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
