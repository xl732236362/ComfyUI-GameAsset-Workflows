import pytest

from game_asset_api.contracts import RequestError, parse_asset_request
from game_asset_api.prompting import (
    PIXEL_STYLE,
    build_action_prompt,
    build_character_prompt,
)


def test_custom_camera_without_prompt_is_rejected():
    with pytest.raises(RequestError, match="camera_prompt is required"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "camera": "custom",
            }
        )


def test_minimal_valid_request_uses_defaults_and_trims_prompts():
    request = parse_asset_request(
        {
            "character_prompt": "  knight  ",
            "action_prompt": "  run  ",
        }
    )

    assert request.character_prompt == "knight"
    assert request.action_prompt == "run"
    assert request.frame_count == 8
    assert request.sprite_size == 128
    assert request.camera is None
    assert request.camera_prompt is None
    assert request.seed is None


@pytest.mark.parametrize("data", [{}, {"character_prompt": "knight"}, {"action_prompt": "run"}])
def test_missing_required_prompts_are_rejected(data):
    with pytest.raises(RequestError, match="^character_prompt and action_prompt are required$"):
        parse_asset_request(data)


def test_non_object_request_body_is_rejected():
    with pytest.raises(RequestError, match="^request body must be an object$"):
        parse_asset_request([])


@pytest.mark.parametrize("seed", [None, True, 1.5, [], "", "invalid"])
def test_invalid_seed_values_are_rejected(seed):
    with pytest.raises(RequestError, match="^seed must be an integer$"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "seed": seed,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("character_prompt", ["knight"]),
        ("action_prompt", ["run"]),
        ("camera", ["side"]),
        ("camera_prompt", ["front view"]),
    ],
)
def test_non_string_prompt_fields_are_rejected(field, value):
    data = {"character_prompt": "knight", "action_prompt": "run", field: value}

    with pytest.raises(RequestError, match=rf"^{field} must be a string$"):
        parse_asset_request(data)


def test_camera_prompt_cannot_override_a_fixed_camera():
    with pytest.raises(RequestError, match="^camera_prompt is only valid when camera is custom$"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "camera": "side",
                "camera_prompt": "front view",
            }
        )


def test_invalid_camera_is_rejected():
    with pytest.raises(RequestError, match="^camera is invalid$"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "camera": "overhead",
            }
        )


@pytest.mark.parametrize("frame_count", [1, 17])
def test_frame_count_outside_supported_range_is_rejected(frame_count):
    with pytest.raises(RequestError, match="^frame_count must be between 2 and 16$"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "frame_count": frame_count,
            }
        )


def test_invalid_sprite_size_is_rejected():
    with pytest.raises(RequestError, match="^sprite_size must be one of 64, 96, 128, 256$"):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "sprite_size": 512,
            }
        )


def test_optional_values_are_normalized_and_preserved():
    request = parse_asset_request(
        {
            "character_prompt": "knight",
            "action_prompt": "run",
            "camera": "custom",
            "camera_prompt": "  low angle  ",
            "frame_count": 16,
            "sprite_size": 64,
            "seed": 42,
        }
    )

    assert request.camera_prompt == "low angle"
    assert request.frame_count == 16
    assert request.sprite_size == 64
    assert request.seed == 42
    assert isinstance(request.seed, int)


def test_string_seed_is_coerced_to_an_integer():
    request = parse_asset_request(
        {
            "character_prompt": "knight",
            "action_prompt": "run",
            "seed": "42",
        }
    )

    assert request.seed == 42
    assert isinstance(request.seed, int)


@pytest.mark.parametrize(
    "seed",
    [-1, 18_446_744_073_709_551_616, "18446744073709551616"],
)
def test_seed_outside_ksampler_range_is_rejected(seed):
    with pytest.raises(
        RequestError, match="^seed must be between 0 and 18446744073709551615$"
    ):
        parse_asset_request(
            {
                "character_prompt": "knight",
                "action_prompt": "run",
                "seed": seed,
            }
        )


@pytest.mark.parametrize("seed", [18_446_744_073_709_551_615, "18446744073709551615"])
def test_ksampler_maximum_seed_is_accepted(seed):
    request = parse_asset_request(
        {
            "character_prompt": "knight",
            "action_prompt": "run",
            "seed": seed,
        }
    )

    assert request.seed == 18_446_744_073_709_551_615
    assert isinstance(request.seed, int)


def test_side_camera_character_prompt_uses_requested_character_and_pixel_style():
    request = parse_asset_request(
        {
            "character_prompt": "clockwork ranger with a brass rifle",
            "action_prompt": "walk",
            "camera": "side",
        }
    )

    prompt = build_character_prompt(request)

    assert request.character_prompt in prompt
    assert "side view" in prompt
    assert "pixel art" in prompt
    assert PIXEL_STYLE in prompt


def test_action_prompt_uses_camera_style_and_identity_constraints():
    request = parse_asset_request(
        {
            "character_prompt": "knight",
            "action_prompt": "swing a sword",
            "camera": "front",
        }
    )

    prompt = build_action_prompt(request)

    assert request.action_prompt in prompt
    assert "fixed front view" in prompt
    assert PIXEL_STYLE in prompt
    assert "locked camera, consistent character identity" in prompt
