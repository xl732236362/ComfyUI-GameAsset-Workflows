"""Deterministic OpenPose control images for a side-view sword attack."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


Point = tuple[int, int]
Pose = tuple[Point, ...]

LIMBS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


def _pose(
    nose: Point,
    neck: Point,
    right_shoulder: Point,
    right_elbow: Point,
    right_wrist: Point,
    left_shoulder: Point,
    left_elbow: Point,
    left_wrist: Point,
    right_hip: Point,
    right_knee: Point,
    right_ankle: Point,
    left_hip: Point,
    left_knee: Point,
    left_ankle: Point,
) -> Pose:
    x, y = nose
    return (
        nose,
        neck,
        right_shoulder,
        right_elbow,
        right_wrist,
        left_shoulder,
        left_elbow,
        left_wrist,
        right_hip,
        right_knee,
        right_ankle,
        left_hip,
        left_knee,
        left_ankle,
        (x - 7, y - 4),
        (x + 7, y - 4),
        (x - 15, y),
        (x + 15, y),
    )


SWORD_ATTACK_PHASES = (
    "anticipation",
    "draw_back",
    "wind_up",
    "acceleration",
    "contact",
    "follow_through",
    "overshoot",
    "recovery",
)

SWORD_ATTACK_POSES = (
    _pose(
        (250, 118), (245, 160),
        (225, 165), (208, 218), (220, 266),
        (265, 164), (240, 220), (230, 270),
        (230, 270), (203, 350), (160, 430),
        (260, 271), (292, 350), (340, 428),
    ),
    _pose(
        (245, 116), (240, 158),
        (220, 164), (196, 194), (183, 233),
        (260, 162), (222, 196), (192, 240),
        (226, 269), (202, 349), (162, 430),
        (256, 270), (286, 350), (333, 428),
    ),
    _pose(
        (248, 120), (243, 164),
        (222, 170), (190, 157), (204, 106),
        (264, 168), (220, 153), (214, 113),
        (226, 274), (201, 353), (158, 430),
        (258, 275), (293, 354), (343, 427),
    ),
    _pose(
        (260, 119), (254, 161),
        (233, 167), (268, 177), (302, 194),
        (275, 165), (260, 193), (294, 202),
        (237, 271), (205, 350), (160, 430),
        (269, 272), (305, 347), (354, 425),
    ),
    _pose(
        (286, 129), (276, 170),
        (253, 176), (320, 188), (392, 210),
        (296, 173), (329, 203), (382, 219),
        (258, 279), (220, 356), (170, 430),
        (291, 280), (333, 350), (390, 422),
    ),
    _pose(
        (292, 137), (280, 178),
        (257, 183), (330, 223), (393, 284),
        (300, 181), (338, 239), (383, 293),
        (260, 285), (222, 357), (174, 430),
        (294, 286), (338, 351), (393, 422),
    ),
    _pose(
        (280, 141), (270, 182),
        (247, 187), (293, 272), (326, 339),
        (291, 185), (305, 279), (317, 347),
        (251, 287), (216, 358), (170, 430),
        (285, 288), (328, 354), (382, 424),
    ),
    _pose(
        (258, 122), (253, 164),
        (232, 169), (241, 220), (262, 258),
        (274, 167), (278, 220), (252, 267),
        (237, 274), (211, 350), (170, 430),
        (269, 275), (299, 350), (345, 428),
    ),
)


def write_sword_attack_pose_sequence(
    output_directory: Path, frame_count: int = 8
) -> list[Path]:
    """Write the two-frame smoke or full eight-frame attack pose sequence."""
    if frame_count == 2:
        poses = (SWORD_ATTACK_POSES[0], SWORD_ATTACK_POSES[4])
    elif frame_count == 8:
        poses = SWORD_ATTACK_POSES
    else:
        raise ValueError("frame_count must be 2 or 8")

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, pose in enumerate(poses):
        image = render_pose(pose)
        path = output_directory / f"{index:03d}.png"
        image.save(path, format="PNG")
        paths.append(path)
    return paths


def render_pose(pose: Pose) -> Image.Image:
    image = Image.new("RGB", (512, 512), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for (start, end), color in zip(LIMBS, COLORS):
        limb_color = tuple(int(channel * 0.6) for channel in color)
        draw.line((pose[start], pose[end]), fill=limb_color, width=8)
    for point, color in zip(pose, COLORS):
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    return image
