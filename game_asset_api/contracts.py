"""Normalized request contracts for game asset generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


CAMERAS = {"side", "front", "top_down", "three_quarter", "custom"}
SPRITE_SIZES = {64, 96, 128, 256}
MAX_SEED = 18_446_744_073_709_551_615


class RequestError(ValueError):
    """Raised when a game asset request cannot be normalized."""


@dataclass(frozen=True, slots=True)
class AssetRequest:
    character_prompt: str
    action_prompt: str
    frame_count: int = 8
    camera: str | None = None
    camera_prompt: str | None = None
    seed: int | None = None
    sprite_size: int = 128


def parse_asset_request(data: object) -> AssetRequest:
    """Validate *data* and return the normalized asset generation request."""
    if not isinstance(data, Mapping):
        raise RequestError("request body must be an object")

    character_prompt = _optional_text(data, "character_prompt")
    action_prompt = _optional_text(data, "action_prompt")
    if not character_prompt or not action_prompt:
        raise RequestError("character_prompt and action_prompt are required")

    camera = _optional_text(data, "camera")
    if camera is not None and camera not in CAMERAS:
        raise RequestError("camera is invalid")

    camera_prompt = _optional_text(data, "camera_prompt")
    if camera == "custom" and not camera_prompt:
        raise RequestError("camera_prompt is required when camera is custom")
    if camera != "custom" and camera_prompt:
        raise RequestError("camera_prompt is only valid when camera is custom")

    frame_count = data.get("frame_count", 8)
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or not 2 <= frame_count <= 16
    ):
        raise RequestError("frame_count must be between 2 and 16")

    sprite_size = data.get("sprite_size", 128)
    if (
        not isinstance(sprite_size, int)
        or isinstance(sprite_size, bool)
        or sprite_size not in SPRITE_SIZES
    ):
        raise RequestError("sprite_size must be one of 64, 96, 128, 256")

    seed = _parse_seed(data["seed"]) if "seed" in data else None
    return AssetRequest(
        character_prompt=character_prompt,
        action_prompt=action_prompt,
        frame_count=frame_count,
        camera=camera,
        camera_prompt=camera_prompt,
        seed=int(seed) if seed is not None else None,
        sprite_size=sprite_size,
    )


def _optional_text(data: Mapping[object, object], field: str) -> str | None:
    if field not in data:
        return None

    value = data[field]
    if not isinstance(value, str):
        raise RequestError(f"{field} must be a string")
    return value.strip()


def _parse_seed(value: object) -> int:
    if isinstance(value, bool):
        raise RequestError("seed must be an integer")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.isdecimal():
        try:
            seed = int(value)
        except ValueError:
            raise RequestError("seed must be an integer") from None
    else:
        raise RequestError("seed must be an integer")

    if not 0 <= seed <= MAX_SEED:
        raise RequestError("seed must be between 0 and 18446744073709551615")
    return seed
