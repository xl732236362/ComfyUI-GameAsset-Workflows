from math import atan2, pi

from PIL import Image
import pytest

from game_asset_api.animation_motion import (
    PHASE_DURATION_MULTIPLIERS,
    plan_sword_attack,
    write_pose_images,
)
from game_asset_api.pose_sequence import SWORD_ATTACK_PHASES, SWORD_ATTACK_POSES


@pytest.mark.parametrize("frame_count", [8, 12, 16])
def test_sword_attack_plan_has_dense_and_selected_frames(frame_count):
    plan = plan_sword_attack(frame_count)

    assert len(plan.dense_frames) == frame_count * 3
    assert len(plan.frames) == frame_count
    assert plan.frames[0].phase == "anticipation"
    assert plan.frames[-1].phase == "recovery"
    assert plan.frames[0].pose == SWORD_ATTACK_POSES[0]
    assert plan.frames[-1].pose == SWORD_ATTACK_POSES[-1]
    assert plan.frames[0].weapon_grip == pytest.approx((230.0, 270.0))
    assert plan.frames[-1].weapon_tip == pytest.approx((258.0, 362.0))
    assert sum("hit" in frame.events for frame in plan.frames) == 1
    assert all(frame.duration > 0 for frame in plan.frames)


def test_two_frame_sword_attack_selects_anticipation_and_contact():
    plan = plan_sword_attack(2)

    assert len(plan.dense_frames) == 6
    assert len(plan.frames) == 2
    anticipation, contact = plan.frames
    assert anticipation.time == 0.0
    assert anticipation.phase == "anticipation"
    assert anticipation.pose == SWORD_ATTACK_POSES[0]
    assert anticipation.weapon_grip == pytest.approx((230.0, 270.0))
    assert anticipation.weapon_tip == pytest.approx((143.4, 220.0))
    assert anticipation.events == ()
    assert contact.time == 0.50
    assert contact.phase == "contact"
    assert contact.pose[:10] == SWORD_ATTACK_POSES[4][:10]
    assert contact.pose[10] == SWORD_ATTACK_POSES[0][10]
    assert contact.pose[11:] == SWORD_ATTACK_POSES[4][11:]
    assert contact.weapon_grip == pytest.approx((382.0, 219.0))
    assert contact.weapon_tip == pytest.approx((482.0, 219.0))
    assert contact.events == ("hit",)
    assert sum("hit" in frame.events for frame in plan.frames) == 1


def test_selected_frame_durations_follow_their_phases():
    plan = plan_sword_attack(12)

    for frame in plan.frames:
        multiplier = PHASE_DURATION_MULTIPLIERS[
            SWORD_ATTACK_PHASES.index(frame.phase)
        ]
        assert frame.duration == pytest.approx(multiplier / 12.0)

    contact = next(frame for frame in plan.frames if frame.events == ("hit",))
    assert contact.duration == pytest.approx(PHASE_DURATION_MULTIPLIERS[4] / 12.0)


def test_sword_attack_plan_locks_planted_foot_through_contact():
    plan = plan_sword_attack(16)

    planted = [frame.pose[10] for frame in plan.dense_frames if frame.time <= 0.50]

    assert len(set(planted)) == 1
    assert planted[0] == SWORD_ATTACK_POSES[0][10]


def test_sword_attack_weapon_angle_uses_one_continuous_arc():
    frames = plan_sword_attack(16).dense_frames
    angles = [
        atan2(
            frame.weapon_tip[1] - frame.weapon_grip[1],
            frame.weapon_tip[0] - frame.weapon_grip[0],
        )
        for frame in frames
    ]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        while angle - unwrapped[-1] > pi:
            angle -= 2 * pi
        while angle - unwrapped[-1] < -pi:
            angle += 2 * pi
        unwrapped.append(angle)

    assert all(
        next_angle >= angle
        for angle, next_angle in zip(unwrapped, unwrapped[1:])
    )
    assert max(
        next_angle - angle for angle, next_angle in zip(unwrapped, unwrapped[1:])
    ) < 0.35


def test_sword_attack_selected_contact_maps_to_authored_contact():
    plan = plan_sword_attack(12)

    contact = next(frame for frame in plan.frames if "hit" in frame.events)

    assert contact.phase == "contact"
    assert contact.weapon_grip == pytest.approx((382.0, 219.0), abs=2.0)


@pytest.mark.parametrize("frame_count", [1, 7, 9, 17, 2.0, True])
def test_sword_attack_plan_rejects_unsupported_frame_counts(frame_count):
    with pytest.raises(
        ValueError, match="sword attack frame count must be 2, 8, 12, or 16"
    ):
        plan_sword_attack(frame_count)


def test_write_pose_images_writes_selected_frames(tmp_path):
    paths = write_pose_images(plan_sword_attack(2), tmp_path)

    assert [path.name for path in paths] == ["000.png", "001.png"]
    for path in paths:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (512, 512)
