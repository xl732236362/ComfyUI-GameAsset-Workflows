"""Validate and align generated character frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from game_asset_api.animation_motion import MotionFrame


@dataclass(frozen=True, slots=True)
class StabilizedSequence:
    frames: tuple[Image.Image, ...]
    translations: tuple[tuple[int, int], ...]


def stabilize_character_frames(
    paths: tuple[Path, ...],
    motion: tuple[MotionFrame, ...],
    sprite_size: int,
) -> StabilizedSequence:
    if len(paths) != len(motion):
        raise ValueError("generated frame count must match motion frame count")

    frames = []
    translations = []
    frame_size = None
    for path, target in zip(paths, motion):
        with Image.open(path) as source:
            frame = source.convert("RGBA").copy()

        if frame_size is None:
            frame_size = frame.size
        elif frame.size != frame_size:
            raise ValueError("generated frames must have the same dimensions")

        alpha = frame.getchannel("A")
        bounds = alpha.getbbox()
        if bounds is None:
            raise ValueError("generated frame has no foreground")
        if alpha.getextrema()[0] != 0:
            raise ValueError("generated frame must contain a transparent background")

        left, top, right, bottom = bounds
        anchor = ((left + right - 1) / 2.0, bottom - 1.0)
        desired = (
            target.root[0] * frame.width / 512.0,
            target.root[1] * frame.height / 512.0,
        )
        translation = (
            round(desired[0] - anchor[0]),
            round(desired[1] - anchor[1]),
        )
        shifted_bounds = (
            left + translation[0],
            top + translation[1],
            right + translation[0],
            bottom + translation[1],
        )
        if (
            shifted_bounds[0] < 0
            or shifted_bounds[1] < 0
            or shifted_bounds[2] > frame.width
            or shifted_bounds[3] > frame.height
        ):
            raise ValueError("aligned character would be clipped")

        aligned = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        aligned.alpha_composite(frame, translation)
        frames.append(
            aligned.resize((sprite_size, sprite_size), Image.Resampling.NEAREST)
        )
        translations.append(translation)

    return StabilizedSequence(tuple(frames), tuple(translations))
