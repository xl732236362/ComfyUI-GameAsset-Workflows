import json
from pathlib import Path

from PIL import Image
import pytest

from game_asset_api.animation_contracts import AnimationRequest
from game_asset_api.animation_inputs import (
    AnimationInputs,
    WeaponAsset,
    load_animation_inputs,
)


def _request(
    *,
    character_image: str = "characters/hero.png",
    weapon: str = "weapons/swords/sword.json",
) -> AnimationRequest:
    return AnimationRequest(
        asset_name="hero_attack",
        character_image=character_image,
        character_prompt="hero",
        weapon=weapon,
        action="sword_attack",
    )


def _save_image(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 8), color).save(path)


def _descriptor(**changes: object) -> dict[str, object]:
    descriptor = {
        "schema_version": 1,
        "image": "art/sword.png",
        "grip": [0.25, 0.75],
        "tip": [0.75, 0.25],
        "default_layer": "behind_character",
    }
    descriptor.update(changes)
    return descriptor


def _write_inputs(input_root: Path, descriptor: object | None = None) -> Path:
    _save_image(input_root / "characters" / "hero.png", (10, 20, 30, 255))
    _save_image(
        input_root / "weapons" / "swords" / "art" / "sword.png",
        (40, 50, 60, 255),
    )
    descriptor_path = input_root / "weapons" / "swords" / "sword.json"
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        json.dumps(_descriptor() if descriptor is None else descriptor),
        encoding="utf-8",
    )
    return descriptor_path


def test_load_animation_inputs_returns_rgba_copies_and_weapon_metadata(tmp_path):
    input_root = tmp_path / "input"
    _write_inputs(input_root)

    inputs = load_animation_inputs(input_root, _request())

    assert isinstance(inputs, AnimationInputs)
    assert isinstance(inputs.weapon, WeaponAsset)
    assert inputs.character.mode == "RGBA"
    assert inputs.weapon.image.mode == "RGBA"
    assert inputs.character.getpixel((0, 0)) == (10, 20, 30, 255)
    assert inputs.weapon.image.getpixel((0, 0)) == (40, 50, 60, 255)
    assert inputs.weapon.grip == (0.25, 0.75)
    assert inputs.weapon.tip == (0.75, 0.25)
    assert inputs.weapon.default_layer == "behind_character"
    assert not hasattr(inputs, "__dict__")
    assert not hasattr(inputs.weapon, "__dict__")

    inputs.character.putpixel((0, 0), (0, 0, 0, 0))
    with Image.open(input_root / "characters" / "hero.png") as source:
        assert source.convert("RGBA").getpixel((0, 0)) == (10, 20, 30, 255)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "weapon descriptor schema is invalid"),
        ({"image": "../sword.png"}, "weapon descriptor image is invalid"),
        ({"image": "/sword.png"}, "weapon descriptor image is invalid"),
        ({"image": "C:/sword.png"}, "weapon descriptor image is invalid"),
        ({"image": "art\\sword.png"}, "weapon descriptor image is invalid"),
        ({"image": "art/\x00sword.png"}, "weapon descriptor image is invalid"),
        (
            {"grip": [0.5, 0.5], "tip": [0.5, 0.5]},
            "weapon descriptor point is invalid",
        ),
        ({"grip": [1.1, 0.5]}, "weapon descriptor point is invalid"),
        ({"tip": [float("nan"), 0.5]}, "weapon descriptor point is invalid"),
        ({"grip": [True, 0.5]}, "weapon descriptor point is invalid"),
        (
            {"default_layer": "above_everything"},
            "weapon descriptor layer is invalid",
        ),
    ],
)
def test_load_animation_inputs_rejects_invalid_descriptor_fields(
    tmp_path, changes, message
):
    input_root = tmp_path / "input"
    _write_inputs(input_root, _descriptor(**changes))

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_animation_inputs(input_root, _request())


def test_load_animation_inputs_rejects_malformed_weapon_descriptor(tmp_path):
    input_root = tmp_path / "input"
    descriptor_path = _write_inputs(input_root)
    descriptor_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="^weapon descriptor is malformed$"):
        load_animation_inputs(input_root, _request())


def test_load_animation_inputs_rejects_unreadable_weapon_descriptor(tmp_path):
    input_root = tmp_path / "input"
    descriptor_path = _write_inputs(input_root)
    descriptor_path.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="^weapon descriptor is unreadable$"):
        load_animation_inputs(input_root, _request())


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("character", "character image is unreadable"),
        ("weapon", "weapon image is unreadable"),
    ],
)
def test_load_animation_inputs_rejects_unreadable_images(tmp_path, target, message):
    input_root = tmp_path / "input"
    _write_inputs(input_root)
    path = (
        input_root / "characters" / "hero.png"
        if target == "character"
        else input_root / "weapons" / "swords" / "art" / "sword.png"
    )
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_animation_inputs(input_root, _request())


def test_load_animation_inputs_rejects_weapon_without_visible_alpha(tmp_path):
    input_root = tmp_path / "input"
    _write_inputs(input_root)
    _save_image(
        input_root / "weapons" / "swords" / "art" / "sword.png",
        (40, 50, 60, 0),
    )

    with pytest.raises(ValueError, match="^weapon image alpha is empty$"):
        load_animation_inputs(input_root, _request())


@pytest.mark.parametrize(
    ("character_image", "weapon", "message"),
    [
        ("../outside.png", "weapons/swords/sword.json", "character image escapes input root"),
        ("characters/hero.png", "../outside.json", "weapon descriptor escapes input root"),
    ],
)
def test_load_animation_inputs_rejects_paths_outside_input_root(
    tmp_path, character_image, weapon, message
):
    input_root = tmp_path / "input"
    _write_inputs(input_root)
    _save_image(tmp_path / "outside.png", (10, 20, 30, 255))
    (tmp_path / "outside.json").write_text(json.dumps(_descriptor()), encoding="utf-8")

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_animation_inputs(
            input_root,
            _request(character_image=character_image, weapon=weapon),
        )


def _create_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")


@pytest.mark.parametrize("target", ["character", "descriptor"])
def test_load_animation_inputs_rejects_in_root_symlinks(tmp_path, target):
    input_root = tmp_path / "input"
    descriptor_path = _write_inputs(input_root)
    if target == "character":
        character_path = input_root / "characters" / "hero.png"
        link = input_root / "characters" / "linked.png"
        _create_symlink_or_skip(link, character_path)
        request = _request(character_image="characters/linked.png")
        message = "character image contains a reparse point"
    else:
        link = input_root / "weapons" / "swords" / "linked.json"
        _create_symlink_or_skip(link, descriptor_path)
        request = _request(weapon="weapons/swords/linked.json")
        message = "weapon descriptor contains a reparse point"

    with pytest.raises(ValueError, match=f"^{message}$"):
        load_animation_inputs(input_root, request)


def test_load_animation_inputs_rejects_descriptor_image_path_traversal(tmp_path):
    input_root = tmp_path / "input"
    _write_inputs(input_root, _descriptor(image="../art/sword.png"))

    with pytest.raises(ValueError, match="^weapon descriptor image is invalid$"):
        load_animation_inputs(input_root, _request())
