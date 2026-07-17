"""Sequential execution and validation for pose-controlled action frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from PIL import Image

from game_asset_api.comfy_client import image_records
from game_asset_api.contracts import AssetRequest
from game_asset_api.jobs import ComfyPromptClient, _resolve_output_image
from game_asset_api.pose_sequence import write_sword_attack_pose_sequence
from game_asset_api.pose_workflow import build_pose_controlled_workflow
from game_asset_api.postprocess import write_sprite_sheet
from game_asset_api.workflows import reference_input_path


@dataclass(frozen=True, slots=True)
class PoseRunResult:
    """Stable output paths from one complete pose-controlled action run."""

    frames: tuple[Path, ...]
    sprite_sheet: Path
    prompt_ids: tuple[str, ...]


async def run_pose_controlled_action(
    project_root: Path,
    client: ComfyPromptClient,
    request: AssetRequest,
    job_id: str,
    reference_source: Path,
    *,
    controlnet_strength: float = 0.9,
    ipadapter_weight: float = 0.65,
    ipadapter_weight_type: str = "style transfer",
    timeout_seconds: float = 1800,
) -> PoseRunResult:
    """Generate, validate, and assemble one authored sword-attack sequence."""
    project_root = Path(project_root)
    reference_source = Path(reference_source)
    reference_destination = project_root / "input" / reference_input_path(job_id)
    reference_destination.parent.mkdir(parents=True, exist_ok=True)
    if reference_source.resolve() != reference_destination.resolve():
        shutil.copyfile(reference_source, reference_destination)

    pose_directory = reference_destination.parent / "poses"
    if pose_directory.exists():
        shutil.rmtree(pose_directory)
    write_sword_attack_pose_sequence(pose_directory, request.frame_count)

    output_directory = project_root / "output" / "game_assets" / job_id / "pose_action"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    frame_directory = output_directory / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    prompt_ids = []
    copied_frames = []

    for frame_index in range(request.frame_count):
        graph = build_pose_controlled_workflow(
            request,
            job_id,
            frame_index,
            controlnet_strength=controlnet_strength,
            ipadapter_weight=ipadapter_weight,
            ipadapter_weight_type=ipadapter_weight_type,
        )
        prompt_id = await client.submit(graph)
        prompt_ids.append(prompt_id)
        history = await client.wait_for_prompt(prompt_id, timeout_seconds)
        records = image_records(history, "20")
        if len(records) != 1:
            raise ValueError("pose frame output count must equal one")
        source = _resolve_output_image(project_root / "output", records[0])
        destination = frame_directory / f"{frame_index:03d}.png"
        _copy_validated_rgba_frame(source, destination, request.sprite_size)
        copied_frames.append(destination)

    frames = []
    for path in copied_frames:
        with Image.open(path) as image:
            frames.append(image.convert("RGBA").copy())
    sprite_sheet, _, _ = write_sprite_sheet(
        frames, output_directory / "spritesheet.png"
    )
    return PoseRunResult(tuple(copied_frames), sprite_sheet, tuple(prompt_ids))


def _copy_validated_rgba_frame(
    source: Path, destination: Path, sprite_size: int
) -> None:
    with Image.open(source) as image:
        frame = image.convert("RGBA")
    if frame.size != (sprite_size, sprite_size):
        raise ValueError("pose frame dimensions do not match sprite size")
    minimum_alpha, maximum_alpha = frame.getchannel("A").getextrema()
    if minimum_alpha != 0 or maximum_alpha == 0:
        raise ValueError("pose frame alpha must contain foreground and background")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.save(destination, format="PNG")
