from hashlib import sha256

from PIL import Image
import pytest

from game_asset_api.animation_inputs import WeaponAsset
from game_asset_api.animation_motion import MotionFrame
from game_asset_api.weapon_composite import composite_weapons


def _character(
    color: tuple[int, int, int] = (20, 120, 220),
    partial_alpha: int | None = None,
) -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(20, 58):
        for x in range(22, 42):
            image.putpixel((x, y), (*color, 255))
    if partial_alpha is not None:
        image.putpixel((22, 20), (*color, partial_alpha))
    return image


def _gradient_character(offset: int) -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(8, 56):
        for x in range(8, 56):
            image.putpixel(
                (x, y),
                ((x * 3 + offset) % 256, (y * 5 + offset) % 256, (x + y + offset) % 256, 255),
            )
    return image


def _weapon() -> WeaponAsset:
    image = Image.new("RGBA", (32, 8), (0, 0, 0, 0))
    for x in range(2, 30):
        image.putpixel((x, 4), (220, 230, 240, 255))
    return WeaponAsset(
        image,
        grip=(2 / 31, 4 / 7),
        tip=(29 / 31, 4 / 7),
        default_layer="behind_character",
    )


def _motion(
    grip: tuple[float, float],
    tip: tuple[float, float],
    layer: str = "behind_character",
) -> MotionFrame:
    return MotionFrame(
        0.0,
        "contact",
        ((0, 0),) * 18,
        (256.0, 448.0),
        grip,
        tip,
        layer,
        1 / 12,
        ("hit",),
    )


def test_weapon_transform_maps_grip_and_tip_within_one_output_pixel():
    result = composite_weapons(
        (_character(),),
        (_motion((160.0, 240.0), (400.0, 240.0)),),
        _weapon(),
    )

    transform = result.transforms[0]

    assert transform.transformed_grip == pytest.approx((20.0, 30.0), abs=1.0)
    assert transform.transformed_tip == pytest.approx((50.0, 30.0), abs=1.0)
    assert result.frames[0].mode == "RGBA"


def test_weapon_digest_is_shared_without_mutating_the_source_image():
    weapon = _weapon()
    source = weapon.image.tobytes()

    result = composite_weapons(
        (_character(), _character((30, 125, 225))),
        (
            _motion((160.0, 240.0), (400.0, 240.0)),
            _motion((180.0, 220.0), (360.0, 360.0)),
        ),
        weapon,
    )

    assert weapon.image.tobytes() == source
    assert {transform.source_digest for transform in result.transforms} == {
        sha256(source).hexdigest()
    }


def test_weapon_layering_places_the_weapon_behind_or_in_front_of_character():
    result = composite_weapons(
        (_character((40, 100, 180)), _character((40, 100, 180))),
        (
            _motion((160.0, 240.0), (400.0, 240.0), "behind_character"),
            _motion((160.0, 240.0), (400.0, 240.0), "in_front_of_character"),
        ),
        _weapon(),
    )

    assert result.frames[0].getpixel((30, 30)) == (40, 100, 180, 255)
    assert result.frames[1].getpixel((30, 30)) == (220, 230, 240, 255)


def test_weapon_composite_rejects_mismatched_character_and_motion_counts():
    with pytest.raises(
        ValueError, match="^character and motion frame counts must match$"
    ):
        composite_weapons((_character(),), (), _weapon())


def test_weapon_composite_rejects_zero_length_source_segment():
    weapon = _weapon()
    degenerate = WeaponAsset(
        weapon.image,
        grip=(0.5, 0.5),
        tip=(0.5, 0.5),
        default_layer=weapon.default_layer,
    )

    with pytest.raises(ValueError, match="^weapon source segment must have length$"):
        composite_weapons(
            (_character(),),
            (_motion((160.0, 240.0), (400.0, 240.0)),),
            degenerate,
        )


def test_weapon_composite_rejects_zero_length_target_segment():
    with pytest.raises(ValueError, match="^weapon target segment must have length$"):
        composite_weapons(
            (_character(),),
            (_motion((160.0, 240.0), (160.0, 240.0)),),
            _weapon(),
        )


def test_weapon_composite_rejects_clipped_target():
    with pytest.raises(ValueError, match="^weapon would be clipped$"):
        composite_weapons(
            (_character(),),
            (_motion((500.0, 240.0), (620.0, 240.0)),),
            _weapon(),
        )


def test_weapon_composite_uses_a_sequence_palette_of_at_most_255_rgb_colors():
    result = composite_weapons(
        (_gradient_character(0), _gradient_character(97)),
        (
            _motion((160.0, 240.0), (400.0, 240.0)),
            _motion((180.0, 220.0), (360.0, 360.0)),
        ),
        _weapon(),
    )

    rgb_colors = {
        pixel[:3]
        for frame in result.frames
        for pixel in frame.get_flattened_data()
    }

    assert len(rgb_colors) <= 255
    assert all(
        len(set(frame.get_flattened_data())) <= 256 for frame in result.frames
    )


def test_weapon_composite_restores_original_alpha_after_palette_quantization():
    result = composite_weapons(
        (_character(partial_alpha=96),),
        (_motion((160.0, 240.0), (400.0, 240.0)),),
        _weapon(),
    )

    frame = result.frames[0]

    assert frame.getpixel((0, 0))[3] == 0
    assert frame.getpixel((22, 20))[3] == 96
    assert frame.getpixel((30, 30))[3] == 255
