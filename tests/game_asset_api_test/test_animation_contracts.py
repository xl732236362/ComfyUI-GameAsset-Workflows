import pytest

from game_asset_api.animation_contracts import (
    AnimationRequest,
    parse_animation_request,
)
from game_asset_api.contracts import RequestError


VALID_REQUEST = {
    "asset_name": "cultivator_attack",
    "character_image": "characters/cultivator.png",
    "character_prompt": "white-robed cultivator",
    "weapon": "weapons/sword.json",
    "action": "sword_attack",
}


def test_minimal_animation_request_uses_production_defaults():
    request = parse_animation_request(
        {
            **VALID_REQUEST,
            "character_prompt": " white-robed cultivator ",
        }
    )

    assert request == AnimationRequest(
        asset_name="cultivator_attack",
        character_image="characters/cultivator.png",
        character_prompt="white-robed cultivator",
        weapon="weapons/sword.json",
        action="sword_attack",
        frame_count=12,
        sprite_size=128,
        seed=None,
        godot_resource_prefix="res://game_assets/cultivator_attack",
    )


@pytest.mark.parametrize("frame_count", [7, 9, 17, True])
def test_animation_request_rejects_unsupported_frame_counts(frame_count):
    with pytest.raises(
        RequestError, match="^frame_count must be one of 8, 12, 16$"
    ):
        parse_animation_request({**VALID_REQUEST, "frame_count": frame_count})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("asset_name", "../attack", "asset_name is invalid"),
        (
            "character_image",
            "../character.png",
            "character_image must be a safe relative path",
        ),
        ("weapon", "C:/weapon.json", "weapon must be a safe relative path"),
        ("action", "walk", "action must be sword_attack"),
        (
            "godot_resource_prefix",
            "user://attack",
            "godot_resource_prefix must begin with res://",
        ),
        (
            "godot_resource_prefix",
            "res://../attack",
            "godot_resource_prefix must be safe",
        ),
    ],
)
def test_animation_request_rejects_unsafe_or_unsupported_values(
    field, value, message
):
    with pytest.raises(RequestError, match=f"^{message}$"):
        parse_animation_request({**VALID_REQUEST, field: value})


def test_animation_request_rejects_non_object_body():
    with pytest.raises(RequestError, match="^request body must be an object$"):
        parse_animation_request([])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_name", None),
        ("character_image", ""),
        ("character_prompt", "   "),
        ("weapon", None),
        ("action", ""),
    ],
)
def test_animation_request_rejects_missing_or_blank_required_values(field, value):
    data = {**VALID_REQUEST, field: value}

    with pytest.raises(RequestError, match=rf"^{field} is required$"):
        parse_animation_request(data)


def test_animation_request_rejects_missing_required_value():
    data = dict(VALID_REQUEST)
    del data["weapon"]

    with pytest.raises(RequestError, match="^weapon is required$"):
        parse_animation_request(data)


@pytest.mark.parametrize("sprite_size", [63, 512, True])
def test_animation_request_rejects_unsupported_sprite_sizes(sprite_size):
    with pytest.raises(
        RequestError, match="^sprite_size must be one of 64, 96, 128, 256$"
    ):
        parse_animation_request({**VALID_REQUEST, "sprite_size": sprite_size})


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(0, 0), ("42", 42), (18_446_744_073_709_551_615, 18_446_744_073_709_551_615)],
)
def test_animation_request_reuses_unsigned_64_bit_seed_contract(seed, expected):
    request = parse_animation_request({**VALID_REQUEST, "seed": seed})

    assert request.seed == expected
    assert isinstance(request.seed, int)


@pytest.mark.parametrize("seed", [-1, 18_446_744_073_709_551_616])
def test_animation_request_rejects_seed_outside_unsigned_64_bit_range(seed):
    with pytest.raises(
        RequestError, match="^seed must be between 0 and 18446744073709551615$"
    ):
        parse_animation_request({**VALID_REQUEST, "seed": seed})


@pytest.mark.parametrize("seed", [None, True, 1.5, "invalid"])
def test_animation_request_rejects_non_integer_seed(seed):
    with pytest.raises(RequestError, match="^seed must be an integer$"):
        parse_animation_request({**VALID_REQUEST, "seed": seed})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("character_image", "characters\\cultivator.png"),
        ("character_image", "/characters/cultivator.png"),
        ("character_image", "characters//cultivator.png"),
        ("character_image", "characters/./cultivator.png"),
        ("weapon", "C:\\weapons\\sword.json"),
        ("weapon", "/weapons/sword.json"),
        ("weapon", "weapons/../sword.json"),
        ("weapon", "weapons/sword:variant.json"),
    ],
)
def test_animation_request_rejects_unsafe_input_path_syntax(field, value):
    with pytest.raises(
        RequestError, match=rf"^{field} must be a safe relative path$"
    ):
        parse_animation_request({**VALID_REQUEST, field: value})


@pytest.mark.parametrize(
    "prefix",
    [
        "res://",
        "res:///game_assets/attack",
        "res://game_assets//attack",
        "res://game_assets/./attack",
        "res://game_assets/../attack",
        "res://C:/game_assets/attack",
        "res://game_assets\\attack",
        "res://game_assets/attack:variant",
    ],
)
def test_animation_request_rejects_unsafe_godot_prefixes(prefix):
    with pytest.raises(RequestError, match="^godot_resource_prefix must be safe$"):
        parse_animation_request(
            {**VALID_REQUEST, "godot_resource_prefix": prefix}
        )


def test_animation_request_normalizes_explicit_safe_godot_prefix():
    request = parse_animation_request(
        {
            **VALID_REQUEST,
            "godot_resource_prefix": "res://actors/cultivator_attack/",
        }
    )

    assert request.godot_resource_prefix == "res://actors/cultivator_attack"
