"""Request contract for production character animations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re

from game_asset_api.contracts import RequestError, SPRITE_SIZES, parse_seed


ANIMATION_FRAME_COUNTS = {8, 12, 16}
_ASSET_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class AnimationRequest:
    asset_name: str
    character_image: str
    character_prompt: str
    weapon: str
    action: str
    frame_count: int = 12
    sprite_size: int = 128
    seed: int | None = None
    godot_resource_prefix: str = ""


def parse_animation_request(data: object) -> AnimationRequest:
    if not isinstance(data, Mapping):
        raise RequestError("request body must be an object")

    asset_name = _required_text(data, "asset_name")
    if _ASSET_NAME.fullmatch(asset_name) is None:
        raise RequestError("asset_name is invalid")
    character_image = _safe_relative_input(data, "character_image")
    character_prompt = _required_text(data, "character_prompt")
    weapon = _safe_relative_input(data, "weapon")
    action = _required_text(data, "action")
    if action != "sword_attack":
        raise RequestError("action must be sword_attack")

    frame_count = data.get("frame_count", 12)
    if type(frame_count) is not int or frame_count not in ANIMATION_FRAME_COUNTS:
        raise RequestError("frame_count must be one of 8, 12, 16")
    sprite_size = data.get("sprite_size", 128)
    if type(sprite_size) is not int or sprite_size not in SPRITE_SIZES:
        raise RequestError("sprite_size must be one of 64, 96, 128, 256")
    seed = parse_seed(data["seed"]) if "seed" in data else None

    prefix = data.get(
        "godot_resource_prefix", f"res://game_assets/{asset_name}"
    )
    if not isinstance(prefix, str) or not prefix.startswith("res://"):
        raise RequestError("godot_resource_prefix must begin with res://")
    suffix = prefix.removeprefix("res://")
    if suffix.endswith("/"):
        suffix = suffix[:-1]
    if not suffix or "\\" in suffix or _unsafe_path(suffix):
        raise RequestError("godot_resource_prefix must be safe")

    return AnimationRequest(
        asset_name=asset_name,
        character_image=character_image,
        character_prompt=character_prompt,
        weapon=weapon,
        action=action,
        frame_count=frame_count,
        sprite_size=sprite_size,
        seed=seed,
        godot_resource_prefix=f"res://{suffix}",
    )


def _required_text(data: Mapping[object, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{field} is required")
    return value.strip()


def _safe_relative_input(data: Mapping[object, object], field: str) -> str:
    value = _required_text(data, field)
    if "\\" in value or _unsafe_path(value):
        raise RequestError(f"{field} must be a safe relative path")
    return PurePosixPath(value).as_posix()


def _unsafe_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        "\x00" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(
            not part or part in {".", ".."} or ":" in part
            for part in value.split("/")
        )
    )
