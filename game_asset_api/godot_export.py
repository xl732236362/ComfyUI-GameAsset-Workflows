"""Godot 4 animation bundle export for production character actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest
from game_asset_api.animation_motion import MotionFrame
from game_asset_api.postprocess import write_sprite_sheet
from game_asset_api.weapon_composite import WeaponTransform


@dataclass(frozen=True, slots=True)
class GodotArtifacts:
    frames: tuple[Path, ...]
    spritesheet: Path
    sprite_frames: Path
    metadata: Path
    preview: Path


def write_godot_bundle(
    output: Path,
    request: AnimationRequest,
    frames: tuple[Image.Image, ...],
    motion: tuple[MotionFrame, ...],
    translations: tuple[tuple[int, int], ...],
    weapon_transforms: tuple[WeaponTransform, ...],
) -> GodotArtifacts:
    if not (
        len(frames)
        == len(motion)
        == len(translations)
        == len(weapon_transforms)
    ):
        raise ValueError("animation artifact counts must match")

    output.mkdir(parents=True, exist_ok=False)
    frame_directory = output / "frames"
    frame_directory.mkdir()
    paths = []
    for index, frame in enumerate(frames):
        path = frame_directory / f"{index:03d}.png"
        frame.save(path, format="PNG")
        paths.append(path)

    spritesheet, columns, rows = write_sprite_sheet(
        list(frames), output / "spritesheet.png"
    )
    metadata = _write_metadata(
        output,
        request,
        frames[0].size,
        motion,
        translations,
        weapon_transforms,
        columns,
        rows,
    )
    sprite_frames = _write_sprite_frames(
        output, request, frames[0].size, motion, columns
    )
    preview = _write_preview(output, frames, motion)
    return GodotArtifacts(tuple(paths), spritesheet, sprite_frames, metadata, preview)


def _write_metadata(
    output: Path,
    request: AnimationRequest,
    frame_size: tuple[int, int],
    motion: tuple[MotionFrame, ...],
    translations: tuple[tuple[int, int], ...],
    weapon_transforms: tuple[WeaponTransform, ...],
    columns: int,
    rows: int,
) -> Path:
    width, height = frame_size
    entries = []
    events = []
    for index, (frame, translation, transform) in enumerate(
        zip(motion, translations, weapon_transforms, strict=True)
    ):
        for event in frame.events:
            events.append({"frame": index, "name": event})
        entries.append(
            {
                "alignment_translation": list(translation),
                "duration": frame.duration,
                "events": list(frame.events),
                "index": index,
                "phase": frame.phase,
                "region": _region(index, columns, width, height),
                "root": list(frame.root),
                "weapon": {
                    "target_grip": list(frame.weapon_grip),
                    "target_tip": list(frame.weapon_tip),
                    "transform": {
                        "rotation": transform.rotation,
                        "scale": transform.scale,
                        "source_digest": transform.source_digest,
                        "transformed_grip": list(transform.transformed_grip),
                        "transformed_tip": list(transform.transformed_tip),
                        "translation": list(transform.translation),
                    },
                },
            }
        )

    metadata = {
        "action": request.action,
        "artifacts": {
            "frames": [f"frames/{index:03d}.png" for index in range(len(entries))],
            "preview": "preview.gif",
            "sprite_frames": "sprite_frames.tres",
            "spritesheet": "spritesheet.png",
        },
        "asset": request.asset_name,
        "canvas": {"height": height, "width": width},
        "events": events,
        "frame_count": len(entries),
        "frames": entries,
        "godot_resource_prefix": request.godot_resource_prefix,
        "grid": {"columns": columns, "rows": rows},
        "loop": False,
        "schema_version": 1,
    }
    path = output / "animation.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_sprite_frames(
    output: Path,
    request: AnimationRequest,
    frame_size: tuple[int, int],
    motion: tuple[MotionFrame, ...],
    columns: int,
) -> Path:
    width, height = frame_size
    sections = [
        f'[gd_resource type="SpriteFrames" load_steps={len(motion) + 2} format=3]',
        "",
        '[ext_resource type="Texture2D" '
        f'path="{request.godot_resource_prefix}/spritesheet.png" id="1_atlas"]',
    ]
    for index in range(len(motion)):
        region = _region(index, columns, width, height)
        sections.extend(
            (
                "",
                f'[sub_resource type="AtlasTexture" id="AtlasTexture_{index:03d}"]',
                'atlas = ExtResource("1_atlas")',
                "region = Rect2("
                f'{region["x"]}, {region["y"]}, {region["width"]}, {region["height"]}'
                ")",
            )
        )
    frames = ",\n".join(
        "{\n"
        f'"duration": {frame.duration * 12.0},\n'
        f'"texture": SubResource("AtlasTexture_{index:03d}")\n'
        "}"
        for index, frame in enumerate(motion)
    )
    sections.extend(
        (
            "",
            "[resource]",
            "animations = [{",
            '"frames": [' + frames + "],",
            '"loop": false,',
            '"name": &"sword_attack",',
            '"speed": 12.0',
            "}]",
        )
    )
    path = output / "sprite_frames.tres"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return path


def _write_preview(
    output: Path, frames: tuple[Image.Image, ...], motion: tuple[MotionFrame, ...]
) -> Path:
    path = output / "preview.gif"
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=[round(frame.duration * 1000) for frame in motion],
        disposal=2,
    )
    return path


def _region(index: int, columns: int, width: int, height: int) -> dict[str, int]:
    return {
        "height": height,
        "width": width,
        "x": index % columns * width,
        "y": index // columns * height,
    }
