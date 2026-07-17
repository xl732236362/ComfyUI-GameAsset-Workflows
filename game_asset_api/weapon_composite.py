"""Rigid weapon placement and shared-palette frame composition."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, cos, hypot, pi, sin

from PIL import Image

from game_asset_api.animation_inputs import WeaponAsset
from game_asset_api.animation_motion import MotionFrame


PointF = tuple[float, float]


@dataclass(frozen=True, slots=True)
class WeaponTransform:
    scale: float
    rotation: float
    translation: PointF
    transformed_grip: PointF
    transformed_tip: PointF
    source_digest: str


@dataclass(frozen=True, slots=True)
class CompositedSequence:
    frames: tuple[Image.Image, ...]
    transforms: tuple[WeaponTransform, ...]


def composite_weapons(
    characters: tuple[Image.Image, ...],
    motion: tuple[MotionFrame, ...],
    weapon: WeaponAsset,
) -> CompositedSequence:
    if len(characters) != len(motion):
        raise ValueError("character and motion frame counts must match")
    if not characters:
        return CompositedSequence((), ())

    digest = sha256(weapon.image.tobytes()).hexdigest()
    frames = []
    transforms = []
    for character, target in zip(characters, motion):
        transformed, record = _transform_weapon(
            character.size, target, weapon, digest
        )
        if target.weapon_layer == "behind_character":
            frames.append(Image.alpha_composite(transformed, character))
        else:
            frames.append(Image.alpha_composite(character, transformed))
        transforms.append(record)
    return CompositedSequence(_shared_palette(tuple(frames)), tuple(transforms))


def _transform_weapon(
    output_size: tuple[int, int],
    target: MotionFrame,
    weapon: WeaponAsset,
    digest: str,
) -> tuple[Image.Image, WeaponTransform]:
    source_grip = _source_point(weapon.grip, weapon.image.size)
    source_tip = _source_point(weapon.tip, weapon.image.size)
    target_grip = _target_point(target.weapon_grip, output_size)
    target_tip = _target_point(target.weapon_tip, output_size)

    source_length = hypot(
        source_tip[0] - source_grip[0], source_tip[1] - source_grip[1]
    )
    if source_length == 0.0:
        raise ValueError("weapon source segment must have length")
    target_length = hypot(
        target_tip[0] - target_grip[0], target_tip[1] - target_grip[1]
    )
    if target_length == 0.0:
        raise ValueError("weapon target segment must have length")

    source_angle = atan2(
        source_tip[1] - source_grip[1], source_tip[0] - source_grip[0]
    )
    target_angle = atan2(
        target_tip[1] - target_grip[1], target_tip[0] - target_grip[0]
    )
    rotation = _shortest_angle(target_angle - source_angle)
    scale = target_length / source_length
    rotation_cosine = cos(rotation)
    rotation_sine = sin(rotation)
    translation = (
        target_grip[0]
        - scale
        * (rotation_cosine * source_grip[0] - rotation_sine * source_grip[1]),
        target_grip[1]
        - scale
        * (rotation_sine * source_grip[0] + rotation_cosine * source_grip[1]),
    )
    transformed_grip = _forward_point(
        source_grip, scale, rotation_cosine, rotation_sine, translation
    )
    transformed_tip = _forward_point(
        source_tip, scale, rotation_cosine, rotation_sine, translation
    )
    if (
        hypot(
            transformed_grip[0] - target_grip[0],
            transformed_grip[1] - target_grip[1],
        )
        > 1.0
        or hypot(
            transformed_tip[0] - target_tip[0], transformed_tip[1] - target_tip[1]
        )
        > 1.0
    ):
        raise ValueError("weapon endpoint transform is inaccurate")

    _reject_clipped_weapon(
        weapon.image, output_size, scale, rotation_cosine, rotation_sine, translation
    )

    inverse_scale = 1.0 / scale
    a = rotation_cosine * inverse_scale
    b = rotation_sine * inverse_scale
    d = -rotation_sine * inverse_scale
    e = rotation_cosine * inverse_scale
    c0 = source_grip[0] - a * target_grip[0] - b * target_grip[1]
    f0 = source_grip[1] - d * target_grip[0] - e * target_grip[1]
    transformed = weapon.image.transform(
        output_size,
        Image.Transform.AFFINE,
        (a, b, c0, d, e, f0),
        resample=Image.Resampling.NEAREST,
        fillcolor=(0, 0, 0, 0),
    )
    return transformed, WeaponTransform(
        scale,
        rotation,
        translation,
        transformed_grip,
        transformed_tip,
        digest,
    )


def _source_point(point: PointF, size: tuple[int, int]) -> PointF:
    return (point[0] * (size[0] - 1), point[1] * (size[1] - 1))


def _target_point(point: PointF, size: tuple[int, int]) -> PointF:
    return (point[0] * size[0] / 512.0, point[1] * size[1] / 512.0)


def _shortest_angle(angle: float) -> float:
    while angle > pi:
        angle -= 2 * pi
    while angle <= -pi:
        angle += 2 * pi
    return angle


def _forward_point(
    point: PointF,
    scale: float,
    rotation_cosine: float,
    rotation_sine: float,
    translation: PointF,
) -> PointF:
    return (
        scale * (rotation_cosine * point[0] - rotation_sine * point[1])
        + translation[0],
        scale * (rotation_sine * point[0] + rotation_cosine * point[1])
        + translation[1],
    )


def _reject_clipped_weapon(
    image: Image.Image,
    output_size: tuple[int, int],
    scale: float,
    rotation_cosine: float,
    rotation_sine: float,
    translation: PointF,
) -> None:
    bounds = image.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("weapon image has no foreground")
    left, top, right, bottom = bounds
    corners = (
        (left, top),
        (right - 1, top),
        (left, bottom - 1),
        (right - 1, bottom - 1),
    )
    for corner in corners:
        transformed = _forward_point(
            corner, scale, rotation_cosine, rotation_sine, translation
        )
        if not (
            0.0 <= transformed[0] <= output_size[0] - 1
            and 0.0 <= transformed[1] <= output_size[1] - 1
        ):
            raise ValueError("weapon would be clipped")


def _shared_palette(frames: tuple[Image.Image, ...]) -> tuple[Image.Image, ...]:
    rgb_frames = tuple(_normalized_rgb(frame) for frame in frames)
    palette = _palette_source(rgb_frames, frames).quantize(
        colors=255,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    quantized = []
    for rgb, original in zip(rgb_frames, frames):
        frame = rgb.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
        frame.putalpha(original.getchannel("A"))
        quantized.append(frame)
    return tuple(quantized)


def _normalized_rgb(frame: Image.Image) -> Image.Image:
    rgb = frame.convert("RGB")
    transparent = frame.getchannel("A").point(
        lambda alpha: 255 if alpha == 0 else 0
    )
    rgb.paste((0, 0, 0), mask=transparent)
    return rgb


def _palette_source(
    rgb_frames: tuple[Image.Image, ...], frames: tuple[Image.Image, ...]
) -> Image.Image:
    colors = []
    for rgb, frame in zip(rgb_frames, frames):
        alpha = frame.getchannel("A").get_flattened_data()
        colors.extend(
            color
            for color, opacity in zip(rgb.get_flattened_data(), alpha)
            if opacity != 0
        )
    if not colors:
        return Image.new("RGB", (1, 1), (0, 0, 0))

    width = min(rgb_frames[0].width, len(colors))
    height = (len(colors) + width - 1) // width
    palette_source = Image.new("RGB", (width, height), colors[0])
    palette_source.putdata(colors + [colors[0]] * (width * height - len(colors)))
    return palette_source
