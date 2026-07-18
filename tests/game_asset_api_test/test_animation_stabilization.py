from dataclasses import FrozenInstanceError
from pathlib import Path

from PIL import Image
import pytest

from game_asset_api.animation_motion import MotionFrame
from game_asset_api.animation_stabilization import (
    StabilizedSequence,
    stabilize_character_frames,
)


def _motion(root: tuple[float, float] = (240.0, 416.0)) -> MotionFrame:
    return MotionFrame(
        time=0.0,
        phase="anticipation",
        pose=((0, 0),) * 18,
        root=root,
        weapon_grip=(200.0, 200.0),
        weapon_tip=(300.0, 200.0),
        weapon_layer="behind_character",
        duration=1 / 12,
    )


def _write_frame(
    path: Path,
    box: tuple[int, int, int, int] = (4, 2, 12, 14),
    size: tuple[int, int] = (16, 16),
) -> None:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            image.putpixel((x, y), (20, 120, 220, 255))
    image.save(path)


def test_stabilization_aligns_bottom_center_to_planned_root(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source)

    result = stabilize_character_frames((source,), (_motion(),), sprite_size=64)

    assert result.translations == ((0, 0),)
    assert result.frames[0].mode == "RGBA"
    assert result.frames[0].size == (64, 64)
    assert result.frames[0].getchannel("A").getbbox() == (16, 8, 48, 56)


def test_stabilization_uses_integer_translation_for_root_jitter(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_frame(first, box=(4, 2, 12, 14))
    _write_frame(second, box=(5, 2, 13, 14))

    result = stabilize_character_frames(
        (first, second), (_motion(), _motion()), sprite_size=64
    )

    assert result.translations == ((0, 0), (-1, 0))


def test_stabilization_rejects_a_frame_without_transparency(tmp_path):
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (16, 16), (20, 30, 40, 255)).save(opaque)

    with pytest.raises(
        ValueError,
        match="^generated frame must contain a transparent background$",
    ):
        stabilize_character_frames((opaque,), (_motion(),), 64)


def test_stabilization_rejects_a_frame_without_foreground(tmp_path):
    empty = tmp_path / "empty.png"
    Image.new("RGBA", (16, 16), (20, 30, 40, 0)).save(empty)

    with pytest.raises(ValueError, match="^generated frame has no foreground$"):
        stabilize_character_frames((empty,), (_motion(),), 64)


def test_stabilization_rejects_mismatched_generated_and_motion_counts(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source)

    with pytest.raises(
        ValueError,
        match="^generated frame count must match motion frame count$",
    ):
        stabilize_character_frames((source,), (), 64)


def test_stabilization_rejects_mixed_source_frame_dimensions(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_frame(first)
    _write_frame(second, size=(32, 16))

    with pytest.raises(
        ValueError,
        match="^generated frames must have the same dimensions$",
    ):
        stabilize_character_frames((first, second), (_motion(), _motion()), 64)


def test_stabilization_scales_non_square_source_roots_per_axis(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source, box=(6, 2, 14, 10), size=(32, 16))

    result = stabilize_character_frames(
        (source,), (_motion(root=(176.0, 320.0)),), 64
    )

    assert result.translations == ((2, 1),)
    assert result.frames[0].getchannel("A").getbbox() == (16, 12, 32, 44)


def test_stabilization_rejects_aligned_foreground_that_would_clip(tmp_path):
    clipped = tmp_path / "clipped.png"
    _write_frame(clipped, box=(0, 2, 8, 14))

    with pytest.raises(ValueError, match="^aligned character would be clipped$"):
        stabilize_character_frames(
            (clipped,), (_motion(root=(480.0, 416.0)),), 64
        )


def test_stabilization_ignores_low_alpha_background_when_finding_the_anchor(tmp_path):
    source = tmp_path / "frame.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 1))
    image.putpixel((0, 0), (0, 0, 0, 0))
    for y in range(2, 14):
        for x in range(4, 12):
            image.putpixel((x, y), (20, 120, 220, 255))
    image.save(source)

    result = stabilize_character_frames((source,), (_motion(),), sprite_size=64)

    assert result.translations == ((0, 0),)
    assert result.frames[0].getpixel((16, 8)) == (20, 120, 220, 255)


def test_stabilization_preserves_partial_alpha_after_alignment(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source)
    with Image.open(source) as opened:
        image = opened.copy()
    image.putpixel((4, 2), (20, 120, 220, 128))
    image.save(source)

    result = stabilize_character_frames(
        (source,), (_motion(root=(272.0, 416.0)),), 64
    )

    assert result.translations == ((1, 0),)
    assert result.frames[0].getpixel((20, 8)) == (20, 120, 220, 128)


def test_stabilization_returns_copied_nearest_resized_frames(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source)
    with Image.open(source) as image:
        image.putpixel((4, 2), (255, 10, 30, 255))
        image.save(source)

    result = stabilize_character_frames((source,), (_motion(),), sprite_size=64)

    assert result.frames[0].getpixel((16, 8)) == (255, 10, 30, 255)
    assert result.frames[0].getpixel((19, 11)) == (255, 10, 30, 255)
    result.frames[0].putpixel((16, 8), (0, 0, 0, 0))
    with Image.open(source) as original:
        assert original.getpixel((4, 2)) == (255, 10, 30, 255)


def test_stabilized_sequence_is_immutable():
    sequence = StabilizedSequence((), ())

    with pytest.raises(FrozenInstanceError):
        sequence.frames = ()
