"""Durable, serialized two-stage game asset generation jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import shutil
from typing import Protocol
from uuid import uuid4

from PIL import Image

from game_asset_api.comfy_client import image_records
from game_asset_api.contracts import AssetRequest, parse_asset_request
from game_asset_api.postprocess import copy_selected_frames, frame_indices, write_sprite_sheet
from game_asset_api.prompting import build_action_prompt, build_character_prompt
from game_asset_api.workflows import (
    build_action_workflow,
    build_character_workflow,
    reference_input_path,
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_CHARACTER = "generating_character"
    GENERATING_ACTION = "generating_action"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: {JobStatus.GENERATING_CHARACTER, JobStatus.FAILED},
    JobStatus.GENERATING_CHARACTER: {JobStatus.GENERATING_ACTION, JobStatus.FAILED},
    JobStatus.GENERATING_ACTION: {JobStatus.POSTPROCESSING, JobStatus.FAILED},
    JobStatus.POSTPROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}
_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED}


@dataclass(frozen=True, slots=True)
class Job:
    """Typed representation of the durable job manifest."""

    id: str
    request: AssetRequest
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    prompt_ids: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


class ComfyPromptClient(Protocol):
    async def submit(self, graph: Mapping[str, object]) -> str: ...

    async def wait_for_prompt(
        self, prompt_id: str, timeout_seconds: float = 1800
    ) -> dict[str, object]: ...


class JobStore:
    """Atomic JSON-backed job state transitions."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, request: AssetRequest) -> Job:
        """Create a queued job with a random UUID4 identifier."""
        job_id = str(uuid4())
        now = _timestamp()
        job = Job(
            id=job_id,
            request=request,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self._job_dir(job_id).mkdir(parents=True, exist_ok=False)
        self._write(job)
        return job

    def read(self, job_id: str) -> Job:
        """Read and validate one typed job manifest."""
        path = self._manifest_path(job_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"job manifest for {job_id} is unreadable") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"job manifest for {job_id} is malformed")
        return _job_from_manifest(raw)

    def transition(
        self,
        job_id: str,
        status: JobStatus | str,
        *,
        error: str | None = None,
        outputs: Mapping[str, str] | None = None,
    ) -> Job:
        """Move a job through its legal state machine and persist it atomically."""
        job = self.read(job_id)
        next_status = JobStatus(status)
        if job.status in _TERMINAL_STATUSES:
            raise ValueError("terminal jobs cannot transition")
        if next_status not in _ALLOWED_TRANSITIONS[job.status]:
            raise ValueError(f"invalid job transition: {job.status.value} -> {next_status.value}")
        updated = Job(
            id=job.id,
            request=job.request,
            status=next_status,
            created_at=job.created_at,
            updated_at=_timestamp(),
            error=str(error) if error is not None else None,
            prompt_ids=job.prompt_ids,
            outputs=dict(outputs) if outputs is not None else job.outputs,
        )
        self._write(updated)
        return updated

    def record_prompt_id(self, job_id: str, stage: str, prompt_id: str) -> Job:
        """Persist a ComfyUI prompt identifier without changing job status."""
        job = self.read(job_id)
        prompt_ids = dict(job.prompt_ids)
        prompt_ids[stage] = prompt_id
        updated = Job(
            id=job.id,
            request=job.request,
            status=job.status,
            created_at=job.created_at,
            updated_at=_timestamp(),
            error=job.error,
            prompt_ids=prompt_ids,
            outputs=job.outputs,
        )
        self._write(updated)
        return updated

    def _write(self, job: Job) -> None:
        manifest = self._manifest_path(job.id)
        temporary = manifest.with_name(f"{manifest.name}.tmp")
        temporary.write_text(
            json.dumps(_job_manifest(job), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(manifest)

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _manifest_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"


class JobRunner:
    """Single-worker orchestrator for character, action, and sprite-sheet generation."""

    def __init__(
        self,
        project_root: Path,
        client: ComfyPromptClient,
        poll_timeout_seconds: float = 1800,
    ) -> None:
        self.project_root = Path(project_root)
        self.jobs_root = self.project_root / "output" / "game_assets"
        self.comfy_output_root = self.project_root / "output"
        self.comfy_input_root = self.project_root / "input"
        self.store = JobStore(self.jobs_root)
        self.client = client
        self.poll_timeout_seconds = poll_timeout_seconds
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the one worker if it is not already running."""
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="game-asset-job-runner")

    async def stop(self) -> None:
        """Drain queued jobs and stop the worker."""
        if self._worker is None:
            return
        await self.join()
        await self._queue.put(None)
        await self._worker
        self._worker = None

    def enqueue(self, request: AssetRequest) -> Job:
        """Persist a queued job and add it to the serialized work queue."""
        job = self.store.create(request)
        self._queue.put_nowait(job.id)
        return job

    async def join(self) -> None:
        """Wait until every job currently queued has been processed."""
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                if job_id is None:
                    return
                await self._process(job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        try:
            job = self.store.transition(job_id, JobStatus.GENERATING_CHARACTER)
            character_prompt_id = await self.client.submit(
                build_character_workflow(job.request, job.id)
            )
            job = self.store.record_prompt_id(job.id, "character", character_prompt_id)
            character_history = await self.client.wait_for_prompt(
                character_prompt_id, self.poll_timeout_seconds
            )
            self._copy_character_outputs(job, character_history)

            job = self.store.transition(job.id, JobStatus.GENERATING_ACTION)
            action_prompt_id = await self.client.submit(
                build_action_workflow(job.request, job.id, reference_input_path(job.id))
            )
            job = self.store.record_prompt_id(job.id, "action", action_prompt_id)
            action_history = await self.client.wait_for_prompt(
                action_prompt_id, self.poll_timeout_seconds
            )
            source_paths = self._source_paths(action_history)
            selected_indices = frame_indices(len(source_paths), job.request.frame_count)

            job = self.store.transition(job.id, JobStatus.POSTPROCESSING)
            copied_paths = copy_selected_frames(
                source_paths, selected_indices, self.jobs_root / job.id / "frames"
            )
            sprite_path, columns, rows = self._write_sprite_sheet(copied_paths, job.id)
            metadata_path = self._write_metadata(
                job,
                source_paths,
                selected_indices,
                copied_paths,
                sprite_path,
                columns,
                rows,
            )
            self.store.transition(
                job.id,
                JobStatus.COMPLETED,
                outputs=_asset_outputs(job.id, copied_paths, sprite_path, metadata_path),
            )
        except (RuntimeError, TimeoutError, ValueError, OSError) as error:
            self._fail(job_id, error)

    def _copy_character_outputs(
        self, job: Job, history: Mapping[str, object]
    ) -> None:
        rgb_path = _resolve_output_image(
            self.comfy_output_root, image_records(history, "11")[0]
        )
        alpha_path = _resolve_output_image(
            self.comfy_output_root, image_records(history, "12")[0]
        )
        reference_path = self.comfy_input_root / reference_input_path(job.id)
        character_path = self.jobs_root / job.id / "character.png"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        character_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rgb_path, reference_path)
        shutil.copyfile(alpha_path, character_path)

    def _source_paths(self, history: Mapping[str, object]) -> list[Path]:
        paths = [
            _resolve_output_image(self.comfy_output_root, record)
            for record in image_records(history, "16")
        ]
        return sorted(paths, key=lambda path: path.as_posix())

    def _write_sprite_sheet(
        self, copied_paths: list[Path], job_id: str
    ) -> tuple[Path, int, int]:
        frames: list[Image.Image] = []
        for frame_path in copied_paths:
            with Image.open(frame_path) as frame:
                frames.append(frame.convert("RGBA").copy())
        return write_sprite_sheet(frames, self.jobs_root / job_id / "spritesheet.png")

    def _write_metadata(
        self,
        job: Job,
        source_paths: list[Path],
        selected_indices: list[int],
        copied_paths: list[Path],
        sprite_path: Path,
        columns: int,
        rows: int,
    ) -> Path:
        metadata_path = self.jobs_root / job.id / "metadata.json"
        metadata = {
            "request": asdict(job.request),
            "prompts": {
                "character": build_character_prompt(job.request),
                "action": build_action_prompt(job.request),
            },
            "seed": job.request.seed if job.request.seed is not None else 0,
            "source_frame_count": len(source_paths),
            "source_frame_indices": selected_indices,
            "source_frame_map": [
                {
                    "output_filename": copied_paths[index].name,
                    "source_index": source_index,
                    "source_filename": source_paths[source_index].name,
                }
                for index, source_index in enumerate(selected_indices)
            ],
            "sprite_sheet": {
                "filename": sprite_path.name,
                "columns": columns,
                "rows": rows,
            },
            "output_frames": [path.name for path in copied_paths],
            "prompt_ids": job.prompt_ids,
            "model_filenames": _model_filenames(),
            "timestamps": {
                "created_at": job.created_at,
                "postprocessed_at": _timestamp(),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return metadata_path

    def _fail(self, job_id: str, error: Exception) -> None:
        try:
            job = self.store.read(job_id)
            if job.status not in _TERMINAL_STATUSES:
                self.store.transition(job_id, JobStatus.FAILED, error=str(error))
        except (ValueError, OSError):
            pass


def _resolve_output_image(output_root: Path, record: Mapping[str, object]) -> Path:
    if record.get("type") != "output":
        raise ValueError("image record type must be output")
    filename = _relative_record_path(record.get("filename"), "filename", allow_empty=False)
    subfolder = _relative_record_path(record.get("subfolder"), "subfolder", allow_empty=True)
    root = output_root.resolve()
    candidate = (root / subfolder / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("image record escapes the ComfyUI output directory") from None
    if not candidate.is_file():
        raise ValueError("ComfyUI output image is missing")
    return candidate


def _relative_record_path(value: object, field: str, *, allow_empty: bool) -> Path:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"image record {field} is malformed")
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"image record {field} must be a relative path")
    return path


def _asset_outputs(
    job_id: str, copied_paths: list[Path], sprite_path: Path, metadata_path: Path
) -> dict[str, str]:
    base = f"/assets/{job_id}"
    outputs = {
        "character": f"{base}/character.png",
        "spritesheet": f"{base}/{sprite_path.name}",
        "metadata": f"{base}/{metadata_path.name}",
    }
    outputs.update(
        {
            f"frame_{index:03d}": f"{base}/frames/{path.name}"
            for index, path in enumerate(copied_paths)
        }
    )
    return outputs


def _model_filenames() -> list[str]:
    return [
        "sd_xl_base_1.0.safetensors",
        "pixel-art-xl.safetensors",
        "BiRefNet-general-epoch_244.safetensors",
        "wan2.2_ti2v_5B_fp16.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan2.2_vae.safetensors",
    ]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_manifest(job: Job) -> dict[str, object]:
    return {
        "id": job.id,
        "request": asdict(job.request),
        "status": job.status.value,
        "timestamps": {"created_at": job.created_at, "updated_at": job.updated_at},
        "error": job.error,
        "prompt_ids": job.prompt_ids,
        "outputs": job.outputs,
    }


def _job_from_manifest(manifest: Mapping[str, object]) -> Job:
    try:
        job_id = manifest["id"]
        request_data = manifest["request"]
        if not isinstance(request_data, Mapping):
            raise ValueError("job request is malformed")
        request = parse_asset_request(
            {key: value for key, value in request_data.items() if value is not None}
        )
        status = JobStatus(manifest["status"])
        timestamps = manifest["timestamps"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("job manifest is malformed") from error
    if not isinstance(job_id, str) or not isinstance(timestamps, Mapping):
        raise ValueError("job manifest is malformed")
    created_at = timestamps.get("created_at")
    updated_at = timestamps.get("updated_at")
    error = manifest.get("error")
    prompt_ids = manifest.get("prompt_ids", {})
    outputs = manifest.get("outputs", {})
    if (
        not isinstance(created_at, str)
        or not isinstance(updated_at, str)
        or error is not None
        and not isinstance(error, str)
        or not isinstance(prompt_ids, Mapping)
        or not isinstance(outputs, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in prompt_ids.items())
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in outputs.items())
    ):
        raise ValueError("job manifest is malformed")
    return Job(
        id=job_id,
        request=request,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        error=error,
        prompt_ids=dict(prompt_ids),
        outputs=dict(outputs),
    )
