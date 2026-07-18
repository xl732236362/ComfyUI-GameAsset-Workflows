"""Durable, serialized two-stage game asset generation jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import secrets
import shutil
from typing import Protocol, cast
from uuid import uuid4

from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest, parse_animation_request
from game_asset_api.comfy_client import image_records
from game_asset_api.contracts import AssetRequest, parse_asset_request
from game_asset_api.godot_export import GodotArtifacts
from game_asset_api.postprocess import copy_selected_frames, frame_indices, write_sprite_sheet
from game_asset_api.prompting import build_action_prompt, build_character_prompt
from game_asset_api.workflows import (
    build_action_workflow,
    build_character_workflow,
    reference_input_path,
)


class JobKind(str, Enum):
    GAME_ASSET = "game_asset"
    PRODUCTION_ANIMATION = "production_animation"


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_CHARACTER = "generating_character"
    GENERATING_ACTION = "generating_action"
    POSTPROCESSING = "postprocessing"
    VALIDATING_INPUTS = "validating_inputs"
    MOTION_PLANNING = "motion_planning"
    TEMPORAL_GENERATION = "temporal_generation"
    CHARACTER_STABILIZATION = "character_stabilization"
    WEAPON_COMPOSITE = "weapon_composite"
    GODOT_EXPORT = "godot_export"
    VALIDATING_OUTPUTS = "validating_outputs"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: {
        JobStatus.GENERATING_CHARACTER,
        JobStatus.VALIDATING_INPUTS,
        JobStatus.FAILED,
    },
    JobStatus.GENERATING_CHARACTER: {JobStatus.GENERATING_ACTION, JobStatus.FAILED},
    JobStatus.GENERATING_ACTION: {JobStatus.POSTPROCESSING, JobStatus.FAILED},
    JobStatus.POSTPROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.VALIDATING_INPUTS: {JobStatus.MOTION_PLANNING, JobStatus.FAILED},
    JobStatus.MOTION_PLANNING: {JobStatus.TEMPORAL_GENERATION, JobStatus.FAILED},
    JobStatus.TEMPORAL_GENERATION: {
        JobStatus.CHARACTER_STABILIZATION,
        JobStatus.FAILED,
    },
    JobStatus.CHARACTER_STABILIZATION: {JobStatus.WEAPON_COMPOSITE, JobStatus.FAILED},
    JobStatus.WEAPON_COMPOSITE: {JobStatus.GODOT_EXPORT, JobStatus.FAILED},
    JobStatus.GODOT_EXPORT: {JobStatus.VALIDATING_OUTPUTS, JobStatus.FAILED},
    JobStatus.VALIDATING_OUTPUTS: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}
_TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED}


@dataclass(frozen=True, slots=True)
class Job:
    """Typed representation of the durable job manifest."""

    id: str
    kind: JobKind
    request: AssetRequest | AnimationRequest
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    failed_stage: str | None = None
    prompt_ids: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


class ComfyPromptClient(Protocol):
    async def submit(self, graph: Mapping[str, object]) -> str: ...

    async def wait_for_prompt(
        self, prompt_id: str, timeout_seconds: float = 1800
    ) -> dict[str, object]: ...


class AnimationProcessorProtocol(Protocol):
    def validate_inputs(self, request: AnimationRequest, job_id: str) -> object: ...

    def plan_motion(
        self, request: AnimationRequest, job_id: str, prepared: object
    ) -> object: ...

    async def generate(
        self,
        request: AnimationRequest,
        job_id: str,
        prepared: object,
        plan: object,
        on_prompt: Callable[[int, str], None] | None = None,
    ) -> tuple[str, object]: ...

    def stabilize(
        self, request: AnimationRequest, plan: object, generated: object
    ) -> object: ...

    def composite(self, plan: object, stabilized: object, prepared: object) -> object: ...

    def export(
        self,
        request: AnimationRequest,
        job_id: str,
        plan: object,
        stabilized: object,
        composited: object,
    ) -> object: ...

    def validate_and_publish(
        self, request: AnimationRequest, job_id: str, staged: object
    ) -> GodotArtifacts: ...

    def cleanup(self, job_id: str) -> None: ...


class JobStore:
    """Atomic JSON-backed job state transitions."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, request: AssetRequest) -> Job:
        """Create a queued job with a random UUID4 identifier."""
        return self._create(JobKind.GAME_ASSET, request)

    def create_animation(self, request: AnimationRequest) -> Job:
        """Create a queued production animation job with a durable seed."""
        if request.seed is None:
            request = replace(request, seed=secrets.randbits(64))
        return self._create(JobKind.PRODUCTION_ANIMATION, request)

    def _create(
        self, kind: JobKind, request: AssetRequest | AnimationRequest
    ) -> Job:
        job_id = str(uuid4())
        now = _timestamp()
        job = Job(
            id=job_id,
            kind=kind,
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
            kind=job.kind,
            request=job.request,
            status=next_status,
            created_at=job.created_at,
            updated_at=_timestamp(),
            error=str(error) if error is not None else None,
            failed_stage=job.status.value if next_status is JobStatus.FAILED else job.failed_stage,
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
            kind=job.kind,
            request=job.request,
            status=job.status,
            created_at=job.created_at,
            updated_at=_timestamp(),
            error=job.error,
            failed_stage=job.failed_stage,
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
        animation_processor: AnimationProcessorProtocol | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.jobs_root = self.project_root / "output" / "game_assets"
        self.comfy_output_root = self.project_root / "output"
        self.comfy_input_root = self.project_root / "input"
        self.store = JobStore(self.jobs_root)
        self.client = client
        self.poll_timeout_seconds = poll_timeout_seconds
        self.animation_processor = animation_processor
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

    def enqueue_animation(self, request: AnimationRequest) -> Job:
        """Persist a queued production animation in the serialized work queue."""
        job = self.store.create_animation(request)
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
            job = self.store.read(job_id)
            if job.kind is JobKind.PRODUCTION_ANIMATION:
                if self.animation_processor is None:
                    raise ValueError("production animation processor is not configured")
                await self._process_animation(job)
            else:
                await self._process_game_asset(job)
        except (RuntimeError, TimeoutError, ValueError, OSError) as error:
            try:
                job = self.store.read(job_id)
                if job.kind is JobKind.PRODUCTION_ANIMATION and self.animation_processor is not None:
                    try:
                        self.animation_processor.cleanup(job_id)
                    except Exception:
                        pass
            except (OSError, ValueError):
                pass
            self._fail(job_id, error)

    async def _process_game_asset(self, job: Job) -> None:
        job = self.store.transition(job.id, JobStatus.GENERATING_CHARACTER)
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

    async def _process_animation(self, job: Job) -> None:
        processor = cast(AnimationProcessorProtocol, self.animation_processor)
        request = cast(AnimationRequest, job.request)
        job = self.store.transition(job.id, JobStatus.VALIDATING_INPUTS)
        prepared = processor.validate_inputs(request, job.id)
        job = self.store.transition(job.id, JobStatus.MOTION_PLANNING)
        plan = processor.plan_motion(request, job.id, prepared)
        job = self.store.transition(job.id, JobStatus.TEMPORAL_GENERATION)
        _, generated = await processor.generate(
            request,
            job.id,
            prepared,
            plan,
            on_prompt=lambda index, prompt_id: self.store.record_prompt_id(
                job.id, f"animation_{index:03d}", prompt_id
            ),
        )
        job = self.store.transition(job.id, JobStatus.CHARACTER_STABILIZATION)
        stabilized = processor.stabilize(request, plan, generated)
        job = self.store.transition(job.id, JobStatus.WEAPON_COMPOSITE)
        composited = processor.composite(plan, stabilized, prepared)
        job = self.store.transition(job.id, JobStatus.GODOT_EXPORT)
        staged = processor.export(request, job.id, plan, stabilized, composited)
        job = self.store.transition(job.id, JobStatus.VALIDATING_OUTPUTS)
        artifacts = processor.validate_and_publish(request, job.id, staged)
        self.store.transition(
            job.id,
            JobStatus.COMPLETED,
            outputs=_animation_outputs(job.id, artifacts),
        )

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


def _animation_outputs(job_id: str, artifacts: GodotArtifacts) -> dict[str, str]:
    base = f"/assets/{job_id}/production_action"
    return {
        **{
            f"frame_{index:03d}": f"{base}/frames/{path.name}"
            for index, path in enumerate(artifacts.frames)
        },
        "spritesheet": f"{base}/{artifacts.spritesheet.name}",
        "sprite_frames": f"{base}/{artifacts.sprite_frames.name}",
        "metadata": f"{base}/{artifacts.metadata.name}",
        "preview": f"{base}/{artifacts.preview.name}",
    }


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
        "kind": job.kind.value,
        "request": asdict(job.request),
        "status": job.status.value,
        "timestamps": {"created_at": job.created_at, "updated_at": job.updated_at},
        "error": job.error,
        "failed_stage": job.failed_stage,
        "prompt_ids": job.prompt_ids,
        "outputs": job.outputs,
    }


def _job_from_manifest(manifest: Mapping[str, object]) -> Job:
    try:
        job_id = manifest["id"]
        kind = JobKind(manifest.get("kind", JobKind.GAME_ASSET.value))
        request_data = manifest["request"]
        if not isinstance(request_data, Mapping):
            raise ValueError("job request is malformed")
        parser = (
            parse_asset_request
            if kind is JobKind.GAME_ASSET
            else parse_animation_request
        )
        request = parser({key: value for key, value in request_data.items() if value is not None})
        status = JobStatus(manifest["status"])
        timestamps = manifest["timestamps"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("job manifest is malformed") from error
    if not isinstance(job_id, str) or not isinstance(timestamps, Mapping):
        raise ValueError("job manifest is malformed")
    created_at = timestamps.get("created_at")
    updated_at = timestamps.get("updated_at")
    error = manifest.get("error")
    failed_stage = manifest.get("failed_stage")
    prompt_ids = manifest.get("prompt_ids", {})
    outputs = manifest.get("outputs", {})
    if (
        not isinstance(created_at, str)
        or not isinstance(updated_at, str)
        or error is not None
        and not isinstance(error, str)
        or failed_stage is not None
        and not isinstance(failed_stage, str)
        or not isinstance(prompt_ids, Mapping)
        or not isinstance(outputs, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in prompt_ids.items())
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in outputs.items())
    ):
        raise ValueError("job manifest is malformed")
    if failed_stage is not None:
        try:
            failed_status = JobStatus(failed_stage)
        except ValueError as error:
            raise ValueError("job manifest is malformed") from error
        if status is not JobStatus.FAILED or failed_status in _TERMINAL_STATUSES:
            raise ValueError("job manifest is malformed")
    return Job(
        id=job_id,
        kind=kind,
        request=request,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        error=error,
        failed_stage=failed_stage,
        prompt_ids=dict(prompt_ids),
        outputs=dict(outputs),
    )
