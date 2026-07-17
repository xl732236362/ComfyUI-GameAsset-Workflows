"""Frame selection and sprite-sheet composition helpers."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

from PIL import Image


def wan_source_frame_count(frame_count: int) -> int:
    """Return the source frame count required by WAN for *frame_count* outputs."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    return 4 * ceil((frame_count - 1) / 4) + 1


def frame_indices(source_count: int, target_count: int) -> list[int]:
    """Select *target_count* evenly distributed source frame indices."""
    if source_count < 1:
        raise ValueError("source_count must be at least 1")
    if target_count < 2:
        raise ValueError("target_count must be at least 2")
    if source_count < target_count:
        raise ValueError("source_count must be at least target_count")
    return [
        round(index * (source_count - 1) / (target_count - 1))
        for index in range(target_count)
    ]


def write_sprite_sheet(
    frames: list[Image.Image], path: Path
) -> tuple[Path, int, int]:
    """Write equally sized frames into an RGBA sprite sheet in row-major order."""
    if not frames:
        raise ValueError("at least one frame is required")

    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("all frames must have the same dimensions")

    columns = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / columns)
    sheet = Image.new("RGBA", (columns * width, rows * height))
    for index, frame in enumerate(frames):
        position = ((index % columns) * width, (index // columns) * height)
        sheet.paste(frame.convert("RGBA"), position)

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return path, columns, rows


def copy_selected_frames(
    source_paths: list[Path], selected_indices: list[int], destination: Path
) -> list[Path]:
    """Copy selected source frames to sequential RGBA PNG files."""
    if len(source_paths) < len(selected_indices):
        raise ValueError("not enough source paths for selected frames")

    for source_index in selected_indices:
        if source_index < 0 or source_index >= len(source_paths):
            raise ValueError("invalid source frame index")

    selected_frames = []
    for source_index in selected_indices:
        with Image.open(source_paths[source_index]) as frame:
            selected_frames.append(frame.convert("RGBA").copy())

    destination.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for output_index, frame in enumerate(selected_frames):
        output_path = destination / f"{output_index:03d}.png"
        frame.save(output_path, format="PNG")
        copied_paths.append(output_path)
    return copied_paths
