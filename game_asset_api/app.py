"""Local HTTP API for submitting and retrieving game asset jobs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import mimetypes
import os
from pathlib import Path, PureWindowsPath
import re
import stat
from typing import Any
from uuid import UUID

from aiohttp import ContentTypeError, web

from game_asset_api.contracts import RequestError, parse_asset_request
from game_asset_api.jobs import Job, JobRunner, JobStatus


_MAX_PUBLIC_ERROR_LENGTH = 500
_PUBLIC_ERROR_FALLBACK = "generation failed due to an internal error"
_ABSOLUTE_PATH_MARKER = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
    r"|(?<![A-Za-z0-9])\\\\"
    r"|(?<![A-Za-z0-9:])//"
    r"|(?<![A-Za-z0-9\\/])[\\/](?!/)"
    r")"
)
_ASSET_READ_ATTEMPTS = 3


def create_app(runner: JobRunner, client: Any | None = None) -> web.Application:
    """Create the local game asset API bound to *runner*."""
    app = web.Application()
    app["game_asset_api.runner"] = runner
    app["game_asset_api.client"] = client

    app.router.add_post("/v1/game-assets", _create_job)
    app.router.add_get("/v1/jobs/{job_id}", _get_job)
    app.router.add_get("/assets/{job_id}/{path:.*}", _get_asset)
    app.on_startup.append(_start_runner)
    app.on_cleanup.append(_stop_runner_and_client)
    return app


async def _create_job(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return web.json_response({"error": "request body must be valid JSON"}, status=400)

    try:
        asset_request = parse_asset_request(data)
    except (RequestError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)

    job = _runner(request.app).enqueue(asset_request)
    return web.json_response({"job_id": job.id, "status": _status_value(job.status)}, status=202)


async def _get_job(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    if not _is_uuid(job_id):
        return _not_found()

    job = _read_job(_runner(request.app), job_id)
    if job is None:
        return _not_found()

    payload: dict[str, object] = {"job_id": job.id, "status": _status_value(job.status)}
    if job.error is not None:
        payload["error"] = _public_error(job.error)
    if job.status is JobStatus.COMPLETED:
        _add_completed_outputs(payload, job)
    return web.json_response(payload)


async def _get_asset(request: web.Request) -> web.StreamResponse:
    job_id = request.match_info["job_id"]
    relative_path = _safe_relative_path(request.match_info["path"])
    if not _is_uuid(job_id) or relative_path is None:
        return _not_found()

    runner = _runner(request.app)
    if _read_job(runner, job_id) is None:
        return _not_found()

    jobs_root = Path(runner.jobs_root).absolute()
    candidate = jobs_root / job_id / relative_path
    body = _read_asset_bytes(jobs_root, candidate)
    if body is None:
        return _not_found()
    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return web.Response(body=body, content_type=content_type)


async def _start_runner(app: web.Application) -> None:
    _runner(app).start()


async def _stop_runner_and_client(app: web.Application) -> None:
    client = app.get("game_asset_api.client")
    try:
        await _runner(app).stop()
    finally:
        if client is not None:
            await client.close()


def _runner(request_or_app: web.Request | web.Application) -> JobRunner:
    return request_or_app["game_asset_api.runner"]


def _read_job(runner: JobRunner, job_id: str) -> Job | None:
    try:
        return runner.store.read(job_id)
    except (OSError, ValueError):
        return None


def _public_error(error: str) -> str:
    normalized = " ".join(error.split())
    if _ABSOLUTE_PATH_MARKER.search(normalized):
        return _PUBLIC_ERROR_FALLBACK
    return normalized[:_MAX_PUBLIC_ERROR_LENGTH] or _PUBLIC_ERROR_FALLBACK


def _add_completed_outputs(payload: dict[str, object], job: Job) -> None:
    character = _public_asset_url(job.id, job.outputs.get("character"))
    spritesheet = _public_asset_url(job.id, job.outputs.get("spritesheet"))
    metadata = _public_asset_url(job.id, job.outputs.get("metadata"))
    if character is not None:
        payload["character_design"] = character
    if spritesheet is not None:
        payload["spritesheet"] = spritesheet
    if metadata is not None:
        payload["metadata"] = metadata
    frames = [
        public_url
        for _, output in sorted(
            (
                (key, value)
                for key, value in job.outputs.items()
                if key.startswith("frame_")
            ),
            key=_frame_sort_key,
        )
        if (public_url := _public_asset_url(job.id, output)) is not None
    ]
    payload["frames"] = frames


def _frame_sort_key(item: tuple[str, str]) -> tuple[int, int | str]:
    suffix = item[0].removeprefix("frame_")
    return (0, int(suffix)) if suffix.isdecimal() else (1, suffix)


def _public_asset_url(job_id: str, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    prefix = f"/assets/{job_id}/"
    if not value.startswith(prefix):
        return None
    relative_path = _safe_relative_path(value.removeprefix(prefix))
    if relative_path is None:
        return None
    return f"{prefix}{relative_path.as_posix()}"


def _safe_relative_path(value: str) -> Path | None:
    if not value or "\\" in value:
        return None
    windows_path = PureWindowsPath(value)
    path = Path(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        return None
    return path


def _read_asset_bytes(jobs_root: Path, candidate: Path) -> bytes | None:
    with _pinned_jobs_root_final_path(jobs_root) as jobs_root_final_path:
        if jobs_root_final_path is None:
            return None
        for _ in range(_ASSET_READ_ATTEMPTS):
            before = _asset_component_stats(jobs_root, candidate)
            if before is None or not _is_regular_file_stat(before[-1]):
                continue
            try:
                with candidate.open("rb") as asset_file:
                    opened = os.fstat(asset_file.fileno())
                    after = _asset_component_stats(jobs_root, candidate)
                    if (
                        after is not None
                        and _same_regular_file(before[-1], opened, after[-1])
                        and _opened_file_is_within_jobs_root(
                            asset_file, jobs_root_final_path
                        )
                    ):
                        return asset_file.read()
            except OSError:
                continue
    return None


def _asset_component_stats(jobs_root: Path, candidate: Path) -> list[os.stat_result] | None:
    try:
        relative_path = candidate.relative_to(jobs_root)
    except ValueError:
        return None
    components = [jobs_root]
    current = jobs_root
    for part in relative_path.parts:
        current /= part
        components.append(current)
    try:
        stats = [component.lstat() for component in components]
    except OSError:
        return None
    if any(_is_reparse_stat(path_stat) for path_stat in stats):
        return None
    return stats


def _is_reparse_stat(path_stat: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(path_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(path_stat.st_mode) or bool(reparse_point and attributes & reparse_point)


def _is_regular_file_stat(path_stat: os.stat_result) -> bool:
    return stat.S_ISREG(path_stat.st_mode)


def _same_regular_file(*stats: os.stat_result) -> bool:
    if not all(_is_regular_file_stat(path_stat) for path_stat in stats):
        return False
    first = stats[0]
    for path_stat in stats[1:]:
        for attribute in ("st_dev", "st_ino"):
            first_value = getattr(first, attribute, None)
            other_value = getattr(path_stat, attribute, None)
            if first_value is not None and other_value is not None and first_value != other_value:
                return False
    return True


def _opened_file_is_within_jobs_root(asset_file: Any, jobs_root_final_path: Path) -> bool:
    final_path = _final_handle_path(asset_file)
    if final_path is None:
        return False
    try:
        final_path.relative_to(jobs_root_final_path)
    except ValueError:
        return False
    return True


def _final_handle_path(asset_file: Any) -> Path | None:
    if os.name == "nt":
        try:
            import msvcrt

            handle = msvcrt.get_osfhandle(asset_file.fileno())
        except (ImportError, OSError):
            return None
        return _windows_final_path_from_handle(handle)
    return _posix_final_path_from_fd(asset_file.fileno())


@contextmanager
def _pinned_jobs_root_final_path(jobs_root: Path) -> Iterator[Path | None]:
    if os.name == "nt":
        handle = _open_windows_directory(jobs_root)
        if handle is None:
            yield None
            return
        try:
            if _windows_handle_is_reparse_point(handle):
                yield None
            else:
                yield _windows_final_path_from_handle(handle)
        finally:
            _close_windows_handle(handle)
        return

    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow_flag = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow_flag is None:
        yield None
        return
    flags = os.O_RDONLY | directory_flag | no_follow_flag
    try:
        descriptor = os.open(jobs_root, flags)
    except OSError:
        yield None
        return
    try:
        yield _posix_final_path_from_fd(descriptor)
    finally:
        os.close(descriptor)


def _posix_final_path_from_fd(descriptor: int) -> Path | None:
    try:
        final_path = os.readlink(Path("/proc/self/fd") / str(descriptor))
    except OSError:
        return None
    if not os.path.isabs(final_path):
        return None
    return Path(os.path.normpath(final_path))


def _open_windows_directory(path: Path) -> int | None:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        handle = kernel32.CreateFileW(
            str(path),
            0x0080,
            0x00000001 | 0x00000002,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
    except AttributeError:
        return None
    if handle in (None, ctypes.c_void_p(-1).value):
        return None
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
    except AttributeError:
        pass


def _windows_handle_is_reparse_point(handle: int) -> bool:
    try:
        import ctypes

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [
                ("file_attributes", ctypes.c_uint32),
                ("reparse_tag", ctypes.c_uint32),
            ]

        info = FileAttributeTagInfo()
        kernel32 = ctypes.windll.kernel32
        kernel32.GetFileInformationByHandleEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.GetFileInformationByHandleEx.restype = ctypes.c_int
        if not kernel32.GetFileInformationByHandleEx(
            ctypes.c_void_p(handle),
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            return True
    except AttributeError:
        return True
    return bool(info.file_attributes & 0x00000400)


def _windows_final_path_from_handle(handle: int) -> Path | None:
    try:
        import ctypes

        buffer_size = 32_768
        buffer = ctypes.create_unicode_buffer(buffer_size)
        result = ctypes.windll.kernel32.GetFinalPathNameByHandleW(
            ctypes.c_void_p(handle), buffer, buffer_size, 0
        )
    except AttributeError:
        return None
    if result == 0 or result >= buffer_size:
        return None
    final_path = buffer.value
    if final_path.startswith("\\\\?\\UNC\\"):
        final_path = "\\\\" + final_path[8:]
    elif final_path.startswith("\\\\?\\"):
        final_path = final_path[4:]
    return Path(os.path.normpath(final_path))


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _status_value(status: JobStatus | str) -> str:
    return status.value if isinstance(status, JobStatus) else str(status)


def _not_found() -> web.Response:
    return web.json_response({"error": "not found"}, status=404)
