"""Local processing and atomic publication for production animations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Protocol

from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest
from game_asset_api.animation_inputs import AnimationInputs, load_animation_inputs
from game_asset_api.animation_motion import MotionPlan, plan_sword_attack, write_pose_images
from game_asset_api.animation_stabilization import StabilizedSequence, stabilize_character_frames
from game_asset_api.animation_workflow import OUTPUT_NODE_ID, build_production_animation_workflow
from game_asset_api.comfy_client import image_records
from game_asset_api.godot_export import GodotArtifacts, write_godot_bundle
from game_asset_api.weapon_composite import CompositedSequence, composite_weapons


class AnimationPromptClient(Protocol):
    async def submit(self, graph: Mapping[str, object]) -> str: ...

    async def wait_for_prompt(
        self, prompt_id: str, timeout_seconds: float = 1800
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PreparedAnimation:
    inputs: AnimationInputs
    reference: Path
    pose_directory: Path


class AnimationProcessor:
    """Coordinate production stages without taking ownership of job state."""

    def __init__(
        self,
        project_root: Path,
        client: AnimationPromptClient,
        timeout_seconds: float = 1800,
    ) -> None:
        self.project_root = Path(project_root)
        self.input_root = self.project_root / "input"
        self.output_root = self.project_root / "output"
        self.jobs_root = self.output_root / "game_assets"
        self.client = client
        self.timeout_seconds = timeout_seconds

    def validate_inputs(
        self, request: AnimationRequest, job_id: str
    ) -> PreparedAnimation:
        """Load inputs and replace this job's staged ComfyUI input files."""
        job_id = _job_id(job_id)
        inputs = load_animation_inputs(self.input_root, request)
        staging = self.input_root / "game_assets" / job_id / "production"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        reference = staging / "reference.png"
        inputs.character.save(reference, format="PNG")
        return PreparedAnimation(inputs, reference, staging / "poses")

    def plan_motion(
        self,
        request: AnimationRequest,
        job_id: str,
        prepared: PreparedAnimation,
    ) -> MotionPlan:
        """Build and render the pose batch used by temporal generation."""
        _job_id(job_id)
        plan = plan_sword_attack(request.frame_count)
        write_pose_images(plan, prepared.pose_directory)
        return plan

    async def generate(
        self,
        request: AnimationRequest,
        job_id: str,
        prepared: PreparedAnimation,
        plan: MotionPlan,
    ) -> tuple[str, tuple[Path, ...]]:
        """Submit one batched graph and resolve its returned source frames."""
        job_id = _job_id(job_id)
        pose_paths = tuple(sorted(prepared.pose_directory.glob("*.png")))
        graph = build_production_animation_workflow(
            request,
            job_id,
            reference_image=_input_relative(self.input_root, prepared.reference),
            pose_images=tuple(
                _input_relative(self.input_root, path) for path in pose_paths
            ),
        )
        prompt_id = await self.client.submit(graph)
        history = await self.client.wait_for_prompt(prompt_id, self.timeout_seconds)
        records = sorted(
            image_records(history, OUTPUT_NODE_ID), key=_record_sort_key
        )
        if len(records) != request.frame_count:
            raise ValueError("generated frame count must equal requested frame_count")
        paths = tuple(
            _resolve_output_image(self.output_root, record) for record in records
        )
        if len(paths) != len(plan.frames):
            raise ValueError("generated frame count must equal motion frame count")
        return prompt_id, paths

    def stabilize(
        self,
        request: AnimationRequest,
        plan: MotionPlan,
        generated: tuple[Path, ...],
    ) -> StabilizedSequence:
        """Validate and ground the generated character frames."""
        return stabilize_character_frames(generated, plan.frames, request.sprite_size)

    def composite(
        self,
        plan: MotionPlan,
        stabilized: StabilizedSequence,
        prepared: PreparedAnimation,
    ) -> CompositedSequence:
        """Apply deterministic weapon transforms over stabilized characters."""
        return composite_weapons(stabilized.frames, plan.frames, prepared.inputs.weapon)

    def export(
        self,
        request: AnimationRequest,
        job_id: str,
        plan: MotionPlan,
        stabilized: StabilizedSequence,
        composited: CompositedSequence,
    ) -> GodotArtifacts:
        """Write the bundle exclusively beneath this job's temporary directory."""
        temporary = self._temporary_bundle(job_id)
        if temporary.exists():
            shutil.rmtree(temporary)
        return write_godot_bundle(
            temporary,
            request,
            composited.frames,
            plan.frames,
            stabilized.translations,
            composited.transforms,
        )

    def validate_and_publish(
        self, request: AnimationRequest, job_id: str, staged: GodotArtifacts
    ) -> GodotArtifacts:
        """Accept the complete staged bundle, then publish it with one rename."""
        temporary = self._temporary_bundle(job_id)
        final = self._final_bundle(job_id)
        if final.exists():
            raise ValueError("production action is already published")
        _validate_artifacts(staged, temporary, request.frame_count, request.sprite_size)
        os.replace(temporary, final)
        self._remove_generation_work(job_id)
        return _rebased_artifacts(staged, final)

    def cleanup(self, job_id: str) -> None:
        """Remove unpublished output and ComfyUI generation work only."""
        shutil.rmtree(self._temporary_bundle(job_id), ignore_errors=True)
        self._remove_generation_work(job_id)

    def _temporary_bundle(self, job_id: str) -> Path:
        return self.jobs_root / _job_id(job_id) / ".production_action.tmp"

    def _final_bundle(self, job_id: str) -> Path:
        return self.jobs_root / _job_id(job_id) / "production_action"

    def _remove_generation_work(self, job_id: str) -> None:
        shutil.rmtree(
            self.output_root / ".animation_work" / _job_id(job_id),
            ignore_errors=True,
        )


def _input_relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _job_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in value)
    ):
        raise ValueError("job_id is invalid")
    return value


def _record_sort_key(record: Mapping[str, object]) -> tuple[str, str]:
    return (
        "/".join(_record_path(record.get("subfolder"), "subfolder", allow_empty=True)),
        "/".join(_record_path(record.get("filename"), "filename", allow_empty=False)),
    )


def _resolve_output_image(output_root: Path, record: Mapping[str, object]) -> Path:
    if record.get("type") != "output":
        raise ValueError("image record type must be output")
    filename = _record_path(record.get("filename"), "filename", allow_empty=False)
    subfolder = _record_path(record.get("subfolder"), "subfolder", allow_empty=True)
    try:
        root = Path(output_root).resolve(strict=True)
    except OSError as error:
        raise ValueError("ComfyUI output directory is unreadable") from error
    candidate = root.joinpath(*subfolder, *filename)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("ComfyUI output image is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("image record escapes the ComfyUI output directory") from None
    if not resolved.is_file():
        raise ValueError("ComfyUI output image is missing")
    return resolved


def _record_path(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if allow_empty and value == "":
        return ()
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"image record {field} must be a relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = tuple(value.split("/"))
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(not part or part in {".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"image record {field} must be a relative path")
    return parts


def _validate_artifacts(
    staged: GodotArtifacts,
    temporary: Path,
    frame_count: int,
    sprite_size: int,
) -> None:
    temporary.resolve(strict=True)
    expected_frames = tuple(
        temporary / "frames" / f"{index:03d}.png" for index in range(frame_count)
    )
    expected_artifacts = GodotArtifacts(
        expected_frames,
        temporary / "spritesheet.png",
        temporary / "sprite_frames.tres",
        temporary / "animation.json",
        temporary / "preview.gif",
    )
    if staged != expected_artifacts:
        raise ValueError("staged animation artifacts are invalid")
    actual_frames = tuple(sorted((temporary / "frames").glob("*.png")))
    if actual_frames != expected_frames:
        raise ValueError("production frame names do not match requested frame_count")
    for path in expected_frames:
        try:
            with Image.open(path) as frame:
                frame.load()
                if frame.size != (sprite_size, sprite_size):
                    raise ValueError("production frame size does not match request")
                if frame.mode != "RGBA":
                    raise ValueError("production frame must be RGBA")
        except ValueError:
            raise
        except (OSError, SyntaxError, Image.DecompressionBombError) as error:
            raise ValueError("production frame is unreadable") from error
    try:
        metadata = json.loads(expected_artifacts.metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("animation metadata is unreadable") from error
    if not isinstance(metadata, dict):
        raise ValueError("animation metadata is malformed")
    if metadata.get("frame_count") != frame_count:
        raise ValueError("animation metadata frame_count does not match request")
    canvas = metadata.get("canvas")
    if not isinstance(canvas, dict) or canvas != {
        "height": sprite_size,
        "width": sprite_size,
    }:
        raise ValueError("animation metadata canvas does not match request")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError("animation metadata frame entries do not match request")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("animation metadata artifacts are malformed")
    expected_names = {
        "frames": [f"frames/{index:03d}.png" for index in range(frame_count)],
        "preview": "preview.gif",
        "sprite_frames": "sprite_frames.tres",
        "spritesheet": "spritesheet.png",
    }
    for name, expected in expected_names.items():
        value = artifacts.get(name)
        if isinstance(value, list):
            names = [_artifact_name(item) for item in value]
        else:
            names = _artifact_name(value)
        if names != expected:
            raise ValueError("animation metadata artifact names are invalid")
    _verify_image(expected_artifacts.spritesheet, "spritesheet", "PNG", 1)
    _verify_sprite_frames(expected_artifacts.sprite_frames)
    _verify_image(expected_artifacts.preview, "preview", "GIF", frame_count)


def _verify_image(
    path: Path, label: str, expected_format: str, expected_frames: int
) -> None:
    try:
        with Image.open(path) as image:
            if image.format != expected_format:
                raise ValueError(f"production {label} is unreadable")
            image.verify()
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != expected_frames:
                raise ValueError(f"production {label} is unreadable")
            for index in range(expected_frames):
                image.seek(index)
                image.load()
    except ValueError:
        raise
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"production {label} is unreadable") from error


def _verify_sprite_frames(path: Path) -> None:
    try:
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError("production sprite frames are unreadable")
    except ValueError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("production sprite frames are unreadable") from error


def _artifact_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("animation artifact path must be relative")
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(
            not part or part in {".", ".."} or ":" in part
            for part in value.split("/")
        )
    ):
        raise ValueError("animation artifact path must be relative")
    return value


def _rebased_artifacts(staged: GodotArtifacts, final: Path) -> GodotArtifacts:
    source = staged.metadata.parent
    return GodotArtifacts(
        tuple(final / path.relative_to(source) for path in staged.frames),
        final / staged.spritesheet.relative_to(source),
        final / staged.sprite_frames.relative_to(source),
        final / staged.metadata.relative_to(source),
        final / staged.preview.relative_to(source),
    )
