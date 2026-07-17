"""Dense, deterministic motion planning for production sword attacks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from pathlib import Path

from game_asset_api.pose_sequence import (
    Pose,
    SWORD_ATTACK_PHASES,
    SWORD_ATTACK_POSES,
    render_pose,
)


PointF = tuple[float, float]
PHASE_TIMES = (0.0, 0.16, 0.30, 0.42, 0.50, 0.64, 0.78, 1.0)
WEAPON_SEGMENTS = (
    ((230.0, 270.0), (143.4, 220.0)),
    ((192.0, 240.0), (142.0, 153.4)),
    ((214.0, 113.0), (214.0, 13.0)),
    ((298.0, 198.0), (368.7, 127.3)),
    ((382.0, 219.0), (482.0, 219.0)),
    ((388.0, 289.0), (469.9, 346.4)),
    ((321.0, 343.0), (363.3, 433.6)),
    ((258.0, 262.0), (258.0, 362.0)),
)
PHASE_DURATION_MULTIPLIERS = (1.5, 1.25, 1.0, 0.5, 1.5, 0.75, 1.0, 1.25)
AUTHORED_ROOTS = tuple(
    (
        (pose[10][0] + pose[13][0]) / 2.0,
        (pose[10][1] + pose[13][1]) / 2.0,
    )
    for pose in SWORD_ATTACK_POSES
)
WEAPON_LENGTHS = tuple(
    ((tip[0] - grip[0]) ** 2 + (tip[1] - grip[1]) ** 2) ** 0.5
    for grip, tip in WEAPON_SEGMENTS
)


@dataclass(frozen=True, slots=True)
class MotionFrame:
    time: float
    phase: str
    pose: Pose
    root: PointF
    weapon_grip: PointF
    weapon_tip: PointF
    weapon_layer: str
    duration: float
    events: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MotionPlan:
    dense_frames: tuple[MotionFrame, ...]
    frames: tuple[MotionFrame, ...]


def plan_sword_attack(frame_count: int) -> MotionPlan:
    if type(frame_count) is not int or frame_count not in {2, 8, 12, 16}:
        raise ValueError("sword attack frame count must be 2, 8, 12, or 16")

    dense = tuple(
        _sample_motion(index / (frame_count * 3 - 1))
        for index in range(frame_count * 3)
    )
    selected_times = (
        (0.0, PHASE_TIMES[4])
        if frame_count == 2
        else tuple(index / (frame_count - 1) for index in range(frame_count))
    )
    contact_index = min(
        range(frame_count),
        key=lambda index: abs(selected_times[index] - PHASE_TIMES[4]),
    )
    selected = []
    for index, selected_time in enumerate(selected_times):
        frame = _sample_motion(
            PHASE_TIMES[4] if index == contact_index else selected_time
        )
        events = ("hit",) if index == contact_index else ()
        phase = "contact" if events else frame.phase
        multiplier = PHASE_DURATION_MULTIPLIERS[SWORD_ATTACK_PHASES.index(phase)]
        selected.append(
            MotionFrame(
                time=frame.time,
                phase=phase,
                pose=frame.pose,
                root=frame.root,
                weapon_grip=frame.weapon_grip,
                weapon_tip=frame.weapon_tip,
                weapon_layer=frame.weapon_layer,
                duration=multiplier / 12.0,
                events=events,
            )
        )
    return MotionPlan(dense, tuple(selected))


def write_pose_images(plan: MotionPlan, directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(plan.frames):
        path = directory / f"{index:03d}.png"
        render_pose(frame.pose).save(path, format="PNG")
        paths.append(path)
    return tuple(paths)


def _sample_motion(time: float) -> MotionFrame:
    phase_index = _phase_index(time)
    phase = SWORD_ATTACK_PHASES[
        min(range(len(PHASE_TIMES)), key=lambda index: abs(PHASE_TIMES[index] - time))
    ]
    pose = tuple(
        (
            round(_hermite_scalar(tuple(point[0] for point in joint), time)),
            round(_hermite_scalar(tuple(point[1] for point in joint), time)),
        )
        for joint in zip(*SWORD_ATTACK_POSES)
    )
    if time <= PHASE_TIMES[4]:
        pose = (*pose[:10], SWORD_ATTACK_POSES[0][10], *pose[11:])

    weapon_grip = _hermite_point(
        tuple(segment[0] for segment in WEAPON_SEGMENTS), time
    )
    weapon_angle = _hermite_scalar(_unwrapped_weapon_angles(), time)
    weapon_length = _hermite_scalar(WEAPON_LENGTHS, time)
    weapon_tip = (
        weapon_grip[0] + weapon_length * cos(weapon_angle),
        weapon_grip[1] + weapon_length * sin(weapon_angle),
    )
    return MotionFrame(
        time=time,
        phase=phase,
        pose=pose,
        root=_hermite_point(AUTHORED_ROOTS, time),
        weapon_grip=weapon_grip,
        weapon_tip=weapon_tip,
        weapon_layer=(
            "behind_character"
            if phase_index <= SWORD_ATTACK_PHASES.index("contact")
            else "in_front_of_character"
        ),
        duration=1 / 12.0,
    )


def _phase_index(time: float) -> int:
    for index, phase_time in enumerate(PHASE_TIMES[1:], start=1):
        if time < phase_time:
            return index - 1
    return len(PHASE_TIMES) - 1


def _hermite_scalar(values: tuple[float, ...], time: float) -> float:
    phase_index = _phase_index(time)
    if phase_index == len(values) - 1:
        return values[-1]

    start_time = PHASE_TIMES[phase_index]
    end_time = PHASE_TIMES[phase_index + 1]
    interval = end_time - start_time
    position = (time - start_time) / interval
    start_tangent = _catmull_tangent(values, phase_index)
    end_tangent = _catmull_tangent(values, phase_index + 1)
    position_squared = position * position
    position_cubed = position_squared * position
    return (
        (2 * position_cubed - 3 * position_squared + 1) * values[phase_index]
        + (position_cubed - 2 * position_squared + position)
        * interval
        * start_tangent
        + (-2 * position_cubed + 3 * position_squared) * values[phase_index + 1]
        + (position_cubed - position_squared) * interval * end_tangent
    )


def _hermite_point(points: tuple[PointF, ...], time: float) -> PointF:
    return (
        _hermite_scalar(tuple(point[0] for point in points), time),
        _hermite_scalar(tuple(point[1] for point in points), time),
    )


def _catmull_tangent(values: tuple[float, ...], index: int) -> float:
    if index == 0:
        return (values[1] - values[0]) / (PHASE_TIMES[1] - PHASE_TIMES[0])
    if index == len(values) - 1:
        return (values[-1] - values[-2]) / (PHASE_TIMES[-1] - PHASE_TIMES[-2])
    return (values[index + 1] - values[index - 1]) / (
        PHASE_TIMES[index + 1] - PHASE_TIMES[index - 1]
    )


def _unwrapped_weapon_angles() -> tuple[float, ...]:
    angles = []
    for grip, tip in WEAPON_SEGMENTS:
        angle = atan2(tip[1] - grip[1], tip[0] - grip[0])
        if angles:
            while angle - angles[-1] > pi:
                angle -= 2 * pi
            while angle - angles[-1] < -pi:
                angle += 2 * pi
        angles.append(angle)
    return tuple(angles)
