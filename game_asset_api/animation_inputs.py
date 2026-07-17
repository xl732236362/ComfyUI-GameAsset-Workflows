"""Validated external inputs for production character animations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat

from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest


@dataclass(frozen=True, slots=True)
class WeaponAsset:
    image: Image.Image
    grip: tuple[float, float]
    tip: tuple[float, float]
    default_layer: str


@dataclass(frozen=True, slots=True)
class AnimationInputs:
    character: Image.Image
    weapon: WeaponAsset


def load_animation_inputs(input_root: Path, request: AnimationRequest) -> AnimationInputs:
    """Load validated character and weapon assets from *input_root*."""
    root, resolved_root = _input_root(input_root)
    character_path = _confined_path(
        root,
        resolved_root,
        _request_path_parts(request.character_image, "character image"),
        "character image",
    )
    descriptor_path = _confined_path(
        root,
        resolved_root,
        _request_path_parts(request.weapon, "weapon descriptor"),
        "weapon descriptor",
    )
    descriptor = _load_descriptor(descriptor_path)
    image_parts, grip, tip, default_layer = _weapon_fields(descriptor)
    descriptor_parts = descriptor_path.relative_to(root).parts
    weapon_path = _confined_path(
        root,
        resolved_root,
        descriptor_parts[:-1] + image_parts,
        "weapon image",
    )

    character = _load_rgba_image(character_path, "character image")
    weapon_image = _load_rgba_image(weapon_path, "weapon image")
    if weapon_image.getchannel("A").getbbox() is None:
        raise ValueError("weapon image alpha is empty")

    return AnimationInputs(
        character=character,
        weapon=WeaponAsset(
            image=weapon_image,
            grip=grip,
            tip=tip,
            default_layer=default_layer,
        ),
    )


def _input_root(input_root: Path) -> tuple[Path, Path]:
    root = Path(input_root)
    _reject_reparse_points(root, "input root")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("input root is unreadable") from error
    if not resolved_root.is_dir():
        raise ValueError("input root is unreadable")
    return root, resolved_root


def _request_path_parts(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} path is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} path is invalid")
    parts = tuple(value.split("/"))
    if not parts or any(not part or part == "." or ":" in part for part in parts):
        raise ValueError(f"{label} path is invalid")
    return parts


def _confined_path(
    root: Path, resolved_root: Path, parts: tuple[str, ...], label: str
) -> Path:
    candidate = root.joinpath(*parts)
    _reject_reparse_points(candidate, label)
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is unreadable") from error
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"{label} escapes input root") from None
    return candidate


def _reject_reparse_points(path: Path, label: str) -> None:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute_path.anchor)
    try:
        for part in absolute_path.parts[1:]:
            current /= part
            if _is_reparse_stat(current.lstat()):
                raise ValueError(f"{label} contains a reparse point")
    except OSError as error:
        raise ValueError(f"{label} is unreadable") from error


def _is_reparse_stat(path_stat: object) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(path_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        reparse_point and attributes & reparse_point
    )


def _load_descriptor(path: Path) -> dict[str, object]:
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("weapon descriptor is unreadable") from error
    except OSError as error:
        raise ValueError("weapon descriptor is unreadable") from error
    except json.JSONDecodeError as error:
        raise ValueError("weapon descriptor is malformed") from error
    if not isinstance(descriptor, dict):
        raise ValueError("weapon descriptor is malformed")
    return descriptor


def _weapon_fields(
    descriptor: dict[str, object],
) -> tuple[tuple[str, ...], tuple[float, float], tuple[float, float], str]:
    if type(descriptor.get("schema_version")) is not int or descriptor["schema_version"] != 1:
        raise ValueError("weapon descriptor schema is invalid")
    image_parts = _descriptor_image_parts(descriptor.get("image"))
    grip = _normalized_point(descriptor.get("grip"))
    tip = _normalized_point(descriptor.get("tip"))
    if grip == tip:
        raise ValueError("weapon descriptor point is invalid")
    default_layer = descriptor.get("default_layer")
    if not isinstance(default_layer, str) or default_layer not in {
        "behind_character",
        "in_front_of_character",
    }:
        raise ValueError("weapon descriptor layer is invalid")
    return image_parts, grip, tip, default_layer


def _descriptor_image_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or "\x00" in value or "\\" in value:
        raise ValueError("weapon descriptor image is invalid")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = tuple(value.split("/"))
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not parts
        or any(not part or part in {".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError("weapon descriptor image is invalid")
    return parts


def _normalized_point(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("weapon descriptor point is invalid")
    if any(type(coordinate) not in {int, float} or not isfinite(coordinate) for coordinate in value):
        raise ValueError("weapon descriptor point is invalid")
    point = (float(value[0]), float(value[1]))
    if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
        raise ValueError("weapon descriptor point is invalid")
    return point


def _load_rgba_image(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"{label} is unreadable") from error
