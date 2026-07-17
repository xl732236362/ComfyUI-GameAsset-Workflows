# Production Animation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce temporally coherent sword-attack sprites with a deterministic external weapon and export a validated Godot 4.x animation bundle.

**Architecture:** A new animation request enters the existing serialized job queue. Focused modules own input validation, dense motion planning, one batched AnimateDiff graph, character stabilization, rigid weapon compositing, shared palette quantization, and Godot export; the job runner owns only durable state transitions and atomic publication.

**Tech Stack:** Python 3.13, aiohttp, Pillow, pytest, ComfyUI HTTP API, SDXL, AnimateDiff-Evolved, OpenPose ControlNet, IP-Adapter Plus, BiRefNet, Godot 4.x headless.

---

## File Map

Create these focused modules:

- `game_asset_api/animation_contracts.py`: parse the production HTTP request without reading the filesystem.
- `game_asset_api/animation_inputs.py`: confine and stage the character image and weapon descriptor below ComfyUI `input/`.
- `game_asset_api/animation_motion.py`: build dense sword-attack poses, weapon targets, durations, and events.
- `game_asset_api/animation_workflow.py`: construct one batched AnimateDiff/OpenPose/IP-Adapter API graph.
- `game_asset_api/animation_stabilization.py`: validate, align, and resize generated RGBA character frames.
- `game_asset_api/weapon_composite.py`: apply the rigid weapon transform and one shared final palette.
- `game_asset_api/godot_export.py`: write PNG frames, sprite sheet, metadata, GIF, and Godot `SpriteFrames`.
- `game_asset_api/animation_pipeline.py`: coordinate filesystem staging and atomic bundle publication without owning job persistence.

Modify these owner modules only where needed:

- `game_asset_api/contracts.py`: expose the existing seed parser for reuse.
- `game_asset_api/jobs.py`: add production job kinds/statuses and dispatch within the existing single GPU queue.
- `game_asset_api/app.py`: add `POST /v1/animations` and production output URLs.
- `game_asset_api/model_manifest.py`: add the verified SDXL motion adapter and explicit fallback URL support.
- `game_asset_api/node_manifest.py`: add the pinned AnimateDiff-Evolved source.
- `game_asset_api/deployment.py`: publish and discover the production workflow artifact.
- `scripts/export_production_animation_workflow.py`: regenerate the representative API workflow.
- `scripts/run_production_animation.py`: run the two-frame preflight and manual production jobs.
- `scripts/deploy.py`: run the new preflight after deployment.
- `README.md`: document inputs, endpoint, deployment, Godot copy path, and validation.

### Task 1: Add The Production Request Contract

**Files:**
- Create: `game_asset_api/animation_contracts.py`
- Modify: `game_asset_api/contracts.py:87-102`
- Create: `tests/game_asset_api_test/test_animation_contracts.py`

- [ ] **Step 1: Write failing request-contract tests**

```python
import pytest

from game_asset_api.animation_contracts import (
    AnimationRequest,
    parse_animation_request,
)
from game_asset_api.contracts import RequestError


def test_minimal_animation_request_uses_production_defaults():
    request = parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": " white-robed cultivator ",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
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
    with pytest.raises(RequestError, match="frame_count must be one of 8, 12, 16"):
        parse_animation_request({**VALID_REQUEST, "frame_count": frame_count})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("asset_name", "../attack", "asset_name is invalid"),
        ("character_image", "../character.png", "character_image must be a safe relative path"),
        ("weapon", "C:/weapon.json", "weapon must be a safe relative path"),
        ("action", "walk", "action must be sword_attack"),
        ("godot_resource_prefix", "user://attack", "godot_resource_prefix must begin with res://"),
        ("godot_resource_prefix", "res://../attack", "godot_resource_prefix must be safe"),
    ],
)
def test_animation_request_rejects_unsafe_or_unsupported_values(field, value, message):
    with pytest.raises(RequestError, match=message):
        parse_animation_request({**VALID_REQUEST, field: value})


VALID_REQUEST = {
    "asset_name": "cultivator_attack",
    "character_image": "characters/cultivator.png",
    "character_prompt": "white-robed cultivator",
    "weapon": "weapons/sword.json",
    "action": "sword_attack",
}
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_contracts.py' -q
```

Expected: collection fails because `game_asset_api.animation_contracts` does not exist.

- [ ] **Step 3: Expose seed parsing and implement the animation contract**

Rename `_parse_seed` to `parse_seed` in `game_asset_api/contracts.py` and update
`parse_asset_request` to call it. Add this implementation:

```python
"""Request contract for production character animations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re

from game_asset_api.contracts import RequestError, SPRITE_SIZES, parse_seed


ANIMATION_FRAME_COUNTS = {8, 12, 16}
_ASSET_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class AnimationRequest:
    asset_name: str
    character_image: str
    character_prompt: str
    weapon: str
    action: str
    frame_count: int = 12
    sprite_size: int = 128
    seed: int | None = None
    godot_resource_prefix: str = ""


def parse_animation_request(data: object) -> AnimationRequest:
    if not isinstance(data, Mapping):
        raise RequestError("request body must be an object")

    asset_name = _required_text(data, "asset_name")
    if _ASSET_NAME.fullmatch(asset_name) is None:
        raise RequestError("asset_name is invalid")
    character_image = _safe_relative_input(data, "character_image")
    character_prompt = _required_text(data, "character_prompt")
    weapon = _safe_relative_input(data, "weapon")
    action = _required_text(data, "action")
    if action != "sword_attack":
        raise RequestError("action must be sword_attack")

    frame_count = data.get("frame_count", 12)
    if type(frame_count) is not int or frame_count not in ANIMATION_FRAME_COUNTS:
        raise RequestError("frame_count must be one of 8, 12, 16")
    sprite_size = data.get("sprite_size", 128)
    if type(sprite_size) is not int or sprite_size not in SPRITE_SIZES:
        raise RequestError("sprite_size must be one of 64, 96, 128, 256")
    seed = parse_seed(data["seed"]) if "seed" in data else None

    prefix = data.get(
        "godot_resource_prefix", f"res://game_assets/{asset_name}"
    )
    if not isinstance(prefix, str) or not prefix.startswith("res://"):
        raise RequestError("godot_resource_prefix must begin with res://")
    suffix = prefix.removeprefix("res://")
    if not suffix or "\\" in suffix or _unsafe_path(suffix):
        raise RequestError("godot_resource_prefix must be safe")

    return AnimationRequest(
        asset_name=asset_name,
        character_image=character_image,
        character_prompt=character_prompt,
        weapon=weapon,
        action=action,
        frame_count=frame_count,
        sprite_size=sprite_size,
        seed=seed,
        godot_resource_prefix=prefix.rstrip("/"),
    )


def _required_text(data: Mapping[object, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{field} is required")
    return value.strip()


def _safe_relative_input(data: Mapping[object, object], field: str) -> str:
    value = _required_text(data, field)
    if "\\" in value or _unsafe_path(value):
        raise RequestError(f"{field} must be a safe relative path")
    return PurePosixPath(value).as_posix()


def _unsafe_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} or ":" in part for part in posix.parts)
    )
```

- [ ] **Step 4: Run contract and existing contract tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_contracts.py' `
  'tests\game_asset_api_test\test_contracts.py' -q
```

Expected: both files pass.

- [ ] **Step 5: Commit the contract**

```powershell
git add game_asset_api/contracts.py game_asset_api/animation_contracts.py `
  tests/game_asset_api_test/test_animation_contracts.py
git commit -m 'Add production animation request contract'
```

### Task 2: Validate And Stage External Weapon Inputs

**Files:**
- Create: `game_asset_api/animation_inputs.py`
- Create: `tests/game_asset_api_test/test_animation_inputs.py`

- [ ] **Step 1: Write failing descriptor and confinement tests**

```python
import json

import pytest
from PIL import Image

from game_asset_api.animation_inputs import load_animation_inputs
from game_asset_api.animation_contracts import parse_animation_request


def _request():
    return parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
        }
    )


def test_load_animation_inputs_reads_rgba_character_and_weapon(tmp_path):
    input_root = tmp_path / "input"
    (input_root / "characters").mkdir(parents=True)
    (input_root / "weapons").mkdir()
    Image.new("RGBA", (32, 32), (20, 40, 60, 255)).save(
        input_root / "characters" / "cultivator.png"
    )
    Image.new("RGBA", (32, 8), (220, 230, 240, 255)).save(
        input_root / "weapons" / "sword.png"
    )
    (input_root / "weapons" / "sword.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": "sword.png",
                "grip": [0.1, 0.5],
                "tip": [0.9, 0.5],
                "default_layer": "behind_character",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_animation_inputs(input_root, _request())

    assert loaded.character.mode == "RGBA"
    assert loaded.weapon.image.mode == "RGBA"
    assert loaded.weapon.grip == (0.1, 0.5)
    assert loaded.weapon.tip == (0.9, 0.5)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"schema_version": 2, "image": "sword.png", "grip": [0.1, 0.5], "tip": [0.9, 0.5], "default_layer": "behind_character"},
        {"schema_version": 1, "image": "../outside.png", "grip": [0.1, 0.5], "tip": [0.9, 0.5], "default_layer": "behind_character"},
        {"schema_version": 1, "image": "sword.png", "grip": [0.2, 0.5], "tip": [0.2, 0.5], "default_layer": "behind_character"},
        {"schema_version": 1, "image": "sword.png", "grip": [-0.1, 0.5], "tip": [0.9, 0.5], "default_layer": "behind_character"},
    ],
)
def test_load_animation_inputs_rejects_invalid_weapon_descriptors(tmp_path, descriptor):
    # Create valid PNG fixtures, then replace only descriptor content.
    input_root = _valid_input_tree(tmp_path)
    (input_root / "weapons" / "sword.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="weapon descriptor"):
        load_animation_inputs(input_root, _request())


def test_load_animation_inputs_rejects_reparse_or_resolved_escape(tmp_path):
    input_root = _valid_input_tree(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(outside)
    (input_root / "characters" / "cultivator.png").unlink()
    try:
        (input_root / "characters" / "cultivator.png").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    with pytest.raises(ValueError, match="input directory"):
        load_animation_inputs(input_root, _request())
```

Add this shared fixture constructor to the same test file:

```python
def _valid_input_tree(tmp_path):
    input_root = tmp_path / "input"
    (input_root / "characters").mkdir(parents=True)
    (input_root / "weapons").mkdir()
    Image.new("RGBA", (32, 32), (20, 40, 60, 255)).save(
        input_root / "characters" / "cultivator.png"
    )
    Image.new("RGBA", (32, 8), (220, 230, 240, 255)).save(
        input_root / "weapons" / "sword.png"
    )
    (input_root / "weapons" / "sword.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": "sword.png",
                "grip": [0.1, 0.5],
                "tip": [0.9, 0.5],
                "default_layer": "behind_character",
            }
        ),
        encoding="utf-8",
    )
    return input_root
```

- [ ] **Step 2: Run the input tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_inputs.py' -q
```

Expected: collection fails because `animation_inputs.py` does not exist.

- [ ] **Step 3: Implement confined loading and typed weapon data**

```python
"""Confined filesystem inputs for production animations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import stat

from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest


PointF = tuple[float, float]


@dataclass(frozen=True, slots=True)
class WeaponAsset:
    image: Image.Image
    grip: PointF
    tip: PointF
    default_layer: str


@dataclass(frozen=True, slots=True)
class AnimationInputs:
    character: Image.Image
    weapon: WeaponAsset


def load_animation_inputs(input_root: Path, request: AnimationRequest) -> AnimationInputs:
    root = Path(input_root).resolve(strict=True)
    character_path = _confined_file(root, PurePosixPath(request.character_image))
    descriptor_path = _confined_file(root, PurePosixPath(request.weapon))
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("weapon descriptor is unreadable") from error
    weapon_path, grip, tip, layer = _parse_weapon_descriptor(
        root, descriptor_path, descriptor
    )
    character = _load_rgba(character_path, "character image")
    weapon = _load_rgba(weapon_path, "weapon image")
    if weapon.getchannel("A").getbbox() is None:
        raise ValueError("weapon image has no visible pixels")
    return AnimationInputs(
        character=character,
        weapon=WeaponAsset(weapon, grip, tip, layer),
    )


def _parse_weapon_descriptor(root: Path, descriptor_path: Path, value: object):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("weapon descriptor schema is invalid")
    image = value.get("image")
    layer = value.get("default_layer")
    if (
        not isinstance(image, str)
        or "\\" in image
        or layer not in {"behind_character", "in_front_of_character"}
    ):
        raise ValueError("weapon descriptor fields are invalid")
    grip = _point(value.get("grip"))
    tip = _point(value.get("tip"))
    if grip == tip:
        raise ValueError("weapon descriptor grip and tip must differ")
    relative = PurePosixPath(request_relative(descriptor_path.parent, root), image)
    return _confined_file(root, relative), grip, tip, layer


def request_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _point(value: object) -> PointF:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(number) not in {int, float} for number in value)
        or any(not 0.0 <= float(number) <= 1.0 for number in value)
    ):
        raise ValueError("weapon descriptor point is invalid")
    return float(value[0]), float(value[1])


def _confined_file(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("input path escapes the input directory")
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("input path escapes the input directory") from None
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        path_stat = current.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = getattr(path_stat, "st_file_attributes", 0)
        if stat.S_ISLNK(path_stat.st_mode) or bool(reparse and attributes & reparse):
            raise ValueError("input path contains a reparse point")
    if not candidate.is_file():
        raise ValueError("input path must be a file")
    return candidate


def _load_rgba(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is unreadable") from error
```

During implementation, keep `_parse_weapon_descriptor`'s path join expressed
with `PurePosixPath`; do not accept platform separators or resolve untrusted
absolute paths.

- [ ] **Step 4: Run the input tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_inputs.py' -q
```

Expected: all descriptor, alpha, and confinement tests pass.

- [ ] **Step 5: Commit input handling**

```powershell
git add game_asset_api/animation_inputs.py `
  tests/game_asset_api_test/test_animation_inputs.py
git commit -m 'Add production animation input validation'
```

### Task 3: Build Dense Phase-Aware Sword Motion

**Files:**
- Create: `game_asset_api/animation_motion.py`
- Modify: `game_asset_api/pose_sequence.py:165-197`
- Create: `tests/game_asset_api_test/test_animation_motion.py`
- Modify: `tests/game_asset_api_test/test_pose_workflow.py:158-204`

- [ ] **Step 1: Write failing motion continuity tests**

```python
from math import atan2, pi

import pytest

from game_asset_api.animation_motion import plan_sword_attack


@pytest.mark.parametrize("frame_count", [8, 12, 16])
def test_sword_attack_plan_has_dense_and_selected_frames(frame_count):
    plan = plan_sword_attack(frame_count)

    assert len(plan.dense_frames) == frame_count * 3
    assert len(plan.frames) == frame_count
    assert plan.frames[0].phase == "anticipation"
    assert plan.frames[-1].phase == "recovery"
    assert sum("hit" in frame.events for frame in plan.frames) == 1
    assert all(frame.duration > 0 for frame in plan.frames)


def test_sword_attack_plan_locks_planted_foot_through_contact():
    plan = plan_sword_attack(16)
    planted = [frame.pose[10] for frame in plan.dense_frames if frame.time <= 0.50]
    assert len(set(planted)) == 1


def test_sword_attack_weapon_angle_uses_one_continuous_arc():
    frames = plan_sword_attack(16).dense_frames
    angles = [
        atan2(frame.weapon_tip[1] - frame.weapon_grip[1], frame.weapon_tip[0] - frame.weapon_grip[0])
        for frame in frames
    ]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        while angle - unwrapped[-1] > pi:
            angle -= 2 * pi
        while angle - unwrapped[-1] < -pi:
            angle += 2 * pi
        unwrapped.append(angle)
    assert all(next_angle >= angle for angle, next_angle in zip(unwrapped, unwrapped[1:]))
    assert max(next_angle - angle for angle, next_angle in zip(unwrapped, unwrapped[1:])) < 0.35


def test_sword_attack_selected_contact_maps_to_authored_contact():
    plan = plan_sword_attack(12)
    contact = next(frame for frame in plan.frames if "hit" in frame.events)
    assert contact.phase == "contact"
    assert contact.weapon_grip == pytest.approx((382.0, 219.0), abs=2.0)
```

- [ ] **Step 2: Run the motion tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_motion.py' -q
```

Expected: collection fails because `animation_motion.py` does not exist.

- [ ] **Step 3: Expose pose rendering and implement the planner**

Rename `_render_pose` to `render_pose` in `pose_sequence.py` and update its
caller. Implement these public types and functions in `animation_motion.py`:

```python
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
    if frame_count not in {2, 8, 12, 16}:
        raise ValueError("sword attack frame count must be 2, 8, 12, or 16")
    dense = tuple(_sample_motion(index / (frame_count * 3 - 1)) for index in range(frame_count * 3))
    selected_times = (
        (0.0, PHASE_TIMES[4])
        if frame_count == 2
        else tuple(index / (frame_count - 1) for index in range(frame_count))
    )
    contact_index = min(range(frame_count), key=lambda index: abs(selected_times[index] - PHASE_TIMES[4]))
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
```

Implement `_sample_motion`, `_phase_index`, `_hermite_scalar`,
`_hermite_point`, and `_unwrapped_weapon_angles` directly below this public
surface. `_sample_motion` must:

- find the two authored phases surrounding `time`;
- compute Catmull-style tangents from adjacent authored values;
- evaluate cubic Hermite interpolation for every joint, root, grip, weapon
  length, and unwrapped weapon angle;
- force joint index `10` to `SWORD_ATTACK_POSES[0][10]` through time `0.50`;
- derive `root` from the authored ankle midpoint curve;
- set the layer to the earlier phase's layer, using `behind_character` through
  contact and `in_front_of_character` afterwards;
- assign the nearest authored phase name and a temporary duration of `1 / 12`.

Use float interpolation internally and round joint coordinates once when
constructing the final `Pose`; do not create tensors or add a numeric
dependency.

- [ ] **Step 4: Run motion and existing pose tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_motion.py' `
  'tests\game_asset_api_test\test_pose_workflow.py' -q
```

Expected: dense motion tests and existing 2/8-frame pose tests pass.

- [ ] **Step 5: Commit motion planning**

```powershell
git add game_asset_api/animation_motion.py game_asset_api/pose_sequence.py `
  tests/game_asset_api_test/test_animation_motion.py `
  tests/game_asset_api_test/test_pose_workflow.py
git commit -m 'Add dense sword attack motion planning'
```

### Task 4: Build One Batched Temporal ComfyUI Graph

**Files:**
- Create: `game_asset_api/animation_workflow.py`
- Create: `tests/game_asset_api_test/test_animation_workflow.py`
- Create: `scripts/export_production_animation_workflow.py`
- Create: `workflows/production_animation_api.json`

- [ ] **Step 1: Write failing graph-contract tests**

```python
from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_workflow import (
    OUTPUT_NODE_ID,
    build_production_animation_workflow,
)


def _request(frame_count=12):
    return parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "white-robed cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": frame_count,
            "seed": 42,
        }
    )


def test_temporal_workflow_uses_one_batch_and_current_animatediff_nodes():
    graph = build_production_animation_workflow(
        _request(),
        "job-id",
        reference_image="game_assets/job-id/production/reference.png",
        pose_images=tuple(
            f"game_assets/job-id/production/poses/{index:03d}.png"
            for index in range(12)
        ),
    )

    by_type = {}
    for node_id, node in graph.items():
        by_type.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    assert len(by_type["LoadImage"]) == 13
    assert len(by_type["ImageBatch"]) == 11
    assert by_type["EmptyLatentImage"][0][1]["batch_size"] == 12
    assert by_type["ADE_LoadAnimateDiffModel"][0][1] == {
        "model_name": "mm_sdxl_v10_beta.safetensors"
    }
    assert by_type["ADE_StandardUniformContextOptions"][0][1] == {
        "context_length": 8,
        "context_stride": 1,
        "context_overlap": 2,
    }
    assert by_type["ADE_UseEvolvedSampling"][0][1]["beta_schedule"] == "autoselect"
    assert graph[OUTPUT_NODE_ID]["class_type"] == "SaveImage"
    assert graph[OUTPUT_NODE_ID]["inputs"]["filename_prefix"] == ".animation_work/job-id/source"


def test_temporal_workflow_excludes_model_generated_weapons():
    graph = build_production_animation_workflow(
        _request(8),
        "job-id",
        reference_image="reference.png",
        pose_images=tuple(f"pose-{index}.png" for index in range(8)),
    )
    text_nodes = [node["inputs"]["text"] for node in graph.values() if node["class_type"] == "CLIPTextEncode"]
    assert any("without a weapon" in text for text in text_nodes)
    assert any("sword, weapon, scabbard" in text for text in text_nodes)


def test_temporal_workflow_rejects_pose_count_mismatch():
    with pytest.raises(ValueError, match="pose image count"):
        build_production_animation_workflow(
            _request(12), "job-id", reference_image="reference.png", pose_images=("one.png",)
        )
```

- [ ] **Step 2: Run graph tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_workflow.py' -q
```

Expected: collection fails because `animation_workflow.py` does not exist.

- [ ] **Step 3: Implement the temporal graph builder**

Create a direct graph builder that follows existing workflow dictionaries. Use
these stable node IDs so the runner can resolve output records:

```python
OUTPUT_NODE_ID = "73"


def build_production_animation_workflow(
    request: AnimationRequest,
    job_id: str,
    *,
    reference_image: str,
    pose_images: tuple[str, ...],
) -> Workflow:
    if len(pose_images) != request.frame_count:
        raise ValueError("pose image count must equal frame_count")

    graph: Workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": "pixel-art-xl.safetensors", "strength_model": 1.0, "strength_clip": 1.0}},
        "3": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "4": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"}},
        "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}},
        "6": {"class_type": "IPAdapterAdvanced", "inputs": {"model": ["2", 0], "ipadapter": ["4", 0], "image": ["3", 0], "clip_vision": ["5", 0], "weight": 0.8, "weight_type": "style transfer", "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": _positive_prompt(request), "clip": ["2", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": _negative_prompt(), "clip": ["2", 1]}},
    }

    load_ids = []
    for index, image in enumerate(pose_images):
        node_id = str(20 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        load_ids.append(node_id)
    pose_batch: list[object] = [load_ids[0], 0]
    for index, load_id in enumerate(load_ids[1:]):
        node_id = str(40 + index)
        graph[node_id] = {"class_type": "ImageBatch", "inputs": {"image1": pose_batch, "image2": [load_id, 0]}}
        pose_batch = [node_id, 0]

    graph.update(
        {
            "60": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "OpenPoseXL2.safetensors"}},
            "61": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["7", 0], "negative": ["8", 0], "control_net": ["60", 0], "image": pose_batch, "strength": 0.9, "start_percent": 0.0, "end_percent": 1.0}},
            "62": {"class_type": "ADE_LoadAnimateDiffModel", "inputs": {"model_name": "mm_sdxl_v10_beta.safetensors"}},
            "63": {"class_type": "ADE_ApplyAnimateDiffModelSimple", "inputs": {"motion_model": ["62", 0]}},
            "64": {"class_type": "ADE_StandardUniformContextOptions", "inputs": {"context_length": 8, "context_stride": 1, "context_overlap": 2}},
            "65": {"class_type": "ADE_UseEvolvedSampling", "inputs": {"model": ["6", 0], "beta_schedule": "autoselect", "m_models": ["63", 0], "context_options": ["64", 0]}},
            "66": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": request.frame_count}},
            "67": {"class_type": "KSampler", "inputs": {"model": ["65", 0], "seed": request.seed or 0, "steps": 30, "cfg": 7, "sampler_name": "dpmpp_2m", "scheduler": "karras", "positive": ["61", 0], "negative": ["61", 1], "latent_image": ["66", 0], "denoise": 1.0}},
            "68": {"class_type": "VAEDecode", "inputs": {"samples": ["67", 0], "vae": ["1", 2]}},
            "69": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "BiRefNet-general-epoch_244.safetensors"}},
            "70": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["69", 0], "image": ["68", 0]}},
            "71": {"class_type": "InvertMask", "inputs": {"mask": ["70", 0]}},
            "72": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["68", 0], "alpha": ["71", 0]}},
            OUTPUT_NODE_ID: {"class_type": "SaveImage", "inputs": {"images": ["72", 0], "filename_prefix": f".animation_work/{job_id}/source"}},
        }
    )
    return graph
```

For the internal two-frame preflight set context length to `2` and overlap to
`0`. For eight-frame requests set context length to `8` and overlap to `0`;
keep the specified `8/2` window for `12` and `16`. Add a graph test for the
two-frame context even though the HTTP request parser rejects two frames.
`_positive_prompt` must include the
character prompt, fixed side view, locked camera, consistent identity, empty
hands, and pixel-art style. `_negative_prompt` must include the existing
negative prompt plus `sword, weapon, scabbard`, camera drift, duplicate limbs,
and cropped character.

- [ ] **Step 4: Add the deterministic exporter and verify its artifact**

`scripts/export_production_animation_workflow.py` must build an eight-frame
representative graph with job ID `example-production-animation`, wrap it as
`{"prompt": graph}`, and write
`workflows/production_animation_api.json` with UTF-8, two-space indentation,
and one trailing newline.

Run:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' `
  '.\scripts\export_production_animation_workflow.py'
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_workflow.py' -q
```

Expected: graph tests pass and the exported JSON equals a graph rebuilt by the
test byte-for-byte after JSON normalization.

- [ ] **Step 5: Commit the temporal workflow**

```powershell
git add game_asset_api/animation_workflow.py `
  tests/game_asset_api_test/test_animation_workflow.py `
  scripts/export_production_animation_workflow.py `
  workflows/production_animation_api.json
git commit -m 'Add temporal production animation workflow'
```

### Task 5: Stabilize Character Frames Before Weapon Composition

**Files:**
- Create: `game_asset_api/animation_stabilization.py`
- Create: `tests/game_asset_api_test/test_animation_stabilization.py`

- [ ] **Step 1: Write failing alpha, alignment, and clipping tests**

```python
from pathlib import Path

import pytest
from PIL import Image

from game_asset_api.animation_motion import MotionFrame
from game_asset_api.animation_stabilization import stabilize_character_frames


def _motion(root=(240.0, 416.0)):
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


def _write_frame(path: Path, box=(4, 2, 12, 14)):
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            image.putpixel((x, y), (20, 120, 220, 255))
    image.save(path)


def test_stabilization_aligns_bottom_center_to_planned_root(tmp_path):
    source = tmp_path / "frame.png"
    _write_frame(source)

    result = stabilize_character_frames((source,), (_motion(),), sprite_size=64)

    assert result.translations == ((0, 0),)
    assert result.frames[0].size == (64, 64)
    assert result.frames[0].mode == "RGBA"


def test_stabilization_uses_integer_translation_for_root_jitter(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_frame(first, box=(4, 2, 12, 14))
    _write_frame(second, box=(5, 2, 13, 14))

    result = stabilize_character_frames(
        (first, second), (_motion(), _motion()), sprite_size=64
    )

    assert result.translations == ((0, 0), (-1, 0))


def test_stabilization_rejects_opaque_background_or_clipping(tmp_path):
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (16, 16), (20, 30, 40, 255)).save(opaque)
    with pytest.raises(ValueError, match="transparent background"):
        stabilize_character_frames((opaque,), (_motion(),), 64)

    clipped = tmp_path / "clipped.png"
    _write_frame(clipped, box=(0, 2, 8, 14))
    with pytest.raises(ValueError, match="clipped"):
        stabilize_character_frames((clipped,), (_motion(root=(480.0, 416.0)),), 64)
```

- [ ] **Step 2: Run stabilization tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_stabilization.py' -q
```

Expected: collection fails because `animation_stabilization.py` does not exist.

- [ ] **Step 3: Implement integer root alignment and nearest resize**

```python
"""Validate and align generated character frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from game_asset_api.animation_motion import MotionFrame


@dataclass(frozen=True, slots=True)
class StabilizedSequence:
    frames: tuple[Image.Image, ...]
    translations: tuple[tuple[int, int], ...]


def stabilize_character_frames(
    paths: tuple[Path, ...],
    motion: tuple[MotionFrame, ...],
    sprite_size: int,
) -> StabilizedSequence:
    if len(paths) != len(motion):
        raise ValueError("generated frame count must match motion frame count")
    frames = []
    translations = []
    for path, target in zip(paths, motion):
        with Image.open(path) as source:
            frame = source.convert("RGBA").copy()
        alpha = frame.getchannel("A")
        minimum, maximum = alpha.getextrema()
        if minimum != 0 or maximum == 0:
            raise ValueError("generated frame must contain foreground and transparent background")
        bounds = alpha.getbbox()
        if bounds is None:
            raise ValueError("generated frame has no foreground")
        left, top, right, bottom = bounds
        anchor = ((left + right - 1) / 2.0, bottom - 1.0)
        scale_x = frame.width / 512.0
        scale_y = frame.height / 512.0
        desired = (target.root[0] * scale_x, target.root[1] * scale_y)
        translation = (
            round(desired[0] - anchor[0]),
            round(desired[1] - anchor[1]),
        )
        shifted_bounds = (
            left + translation[0],
            top + translation[1],
            right + translation[0],
            bottom + translation[1],
        )
        if (
            shifted_bounds[0] < 0
            or shifted_bounds[1] < 0
            or shifted_bounds[2] > frame.width
            or shifted_bounds[3] > frame.height
        ):
            raise ValueError("aligned character would be clipped")
        aligned = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        aligned.alpha_composite(frame, translation)
        frames.append(
            aligned.resize((sprite_size, sprite_size), Image.Resampling.NEAREST)
        )
        translations.append(translation)
    return StabilizedSequence(tuple(frames), tuple(translations))
```

- [ ] **Step 4: Run stabilization tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_stabilization.py' -q
```

Expected: all tests pass without changing generated pixels except integer
translation and nearest-neighbor resize.

- [ ] **Step 5: Commit stabilization**

```powershell
git add game_asset_api/animation_stabilization.py `
  tests/game_asset_api_test/test_animation_stabilization.py
git commit -m 'Add production character frame stabilization'
```

### Task 6: Composite One Rigid Weapon And One Shared Palette

**Files:**
- Create: `game_asset_api/weapon_composite.py`
- Create: `tests/game_asset_api_test/test_weapon_composite.py`

- [ ] **Step 1: Write failing rigid-transform and palette tests**

```python
import pytest
from PIL import Image

from game_asset_api.animation_inputs import WeaponAsset
from game_asset_api.animation_motion import MotionFrame
from game_asset_api.weapon_composite import composite_weapons


def _character(color):
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(20, 58):
        for x in range(22, 42):
            image.putpixel((x, y), (*color, 255))
    return image


def _weapon():
    image = Image.new("RGBA", (32, 8), (0, 0, 0, 0))
    for x in range(2, 30):
        image.putpixel((x, 4), (220, 230, 240, 255))
    return WeaponAsset(image, grip=(2 / 31, 4 / 7), tip=(29 / 31, 4 / 7), default_layer="behind_character")


def _motion(grip, tip, layer="behind_character"):
    return MotionFrame(0.0, "contact", ((0, 0),) * 18, (256.0, 448.0), grip, tip, layer, 1 / 12, ("hit",))


def test_weapon_transform_maps_grip_and_tip_within_one_output_pixel():
    result = composite_weapons(
        (_character((20, 120, 220)),),
        (_motion((160.0, 240.0), (400.0, 240.0)),),
        _weapon(),
    )
    transform = result.transforms[0]
    assert transform.transformed_grip == pytest.approx((20.0, 30.0), abs=1.0)
    assert transform.transformed_tip == pytest.approx((50.0, 30.0), abs=1.0)
    assert result.frames[0].mode == "RGBA"


def test_weapon_shape_and_palette_are_shared_across_frames():
    result = composite_weapons(
        (_character((20, 120, 220)), _character((30, 125, 225))),
        (
            _motion((160.0, 240.0), (400.0, 240.0)),
            _motion((180.0, 220.0), (360.0, 360.0), "in_front_of_character"),
        ),
        _weapon(),
    )
    color_sets = [set(frame.getdata()) for frame in result.frames]
    shared = color_sets[0] | color_sets[1]
    assert all(len(colors) <= 256 for colors in color_sets)
    assert all(colors <= shared for colors in color_sets)
    assert result.transforms[0].source_digest == result.transforms[1].source_digest


def test_weapon_composite_rejects_clipped_target():
    with pytest.raises(ValueError, match="weapon would be clipped"):
        composite_weapons(
            (_character((20, 120, 220)),),
            (_motion((500.0, 240.0), (620.0, 240.0)),),
            _weapon(),
        )
```

- [ ] **Step 2: Run weapon tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_weapon_composite.py' -q
```

Expected: collection fails because `weapon_composite.py` does not exist.

- [ ] **Step 3: Implement inverse affine mapping and sequence quantization**

```python
"""Rigid weapon placement and final sequence palette."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, cos, hypot, sin

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
    digest = sha256(weapon.image.tobytes()).hexdigest()
    raw_frames = []
    transforms = []
    for character, target in zip(characters, motion):
        transformed, record = _transform_weapon(character.size, target, weapon, digest)
        if target.weapon_layer == "behind_character":
            combined = Image.alpha_composite(transformed, character)
        else:
            combined = Image.alpha_composite(character, transformed)
        raw_frames.append(combined)
        transforms.append(record)
    return CompositedSequence(_shared_palette(tuple(raw_frames)), tuple(transforms))
```

Implement `_transform_weapon` with this exact coordinate contract:

1. Convert source normalized endpoints with `(width - 1, height - 1)`.
2. Convert target 512-space endpoints with `(output_width / 512,
   output_height / 512)`.
3. Compute `scale = target_length / source_length` and
   `rotation = target_angle - source_angle`.
4. Pass the inverse coefficients below to `weapon.image.transform` with
   `Image.Transform.AFFINE` and `Image.Resampling.NEAREST`:

```python
inverse_scale = 1.0 / scale
c = cos(rotation)
s = sin(rotation)
a = c * inverse_scale
b = s * inverse_scale
d = -s * inverse_scale
e = c * inverse_scale
c0 = source_grip_x - a * target_grip_x - b * target_grip_y
f0 = source_grip_y - d * target_grip_x - e * target_grip_y
affine = (a, b, c0, d, e, f0)
```

Transform the weapon alpha bounding-box corners with the forward transform and
reject any corner outside the output canvas. Record the forward translation,
analytically transformed endpoints, rotation in radians, scale, and digest.

Implement `_shared_palette` by concatenating RGB views of all completed frames,
quantizing once to `255` colors with `Image.Quantize.MEDIANCUT` and no dithering,
then quantizing each RGB frame against that palette and restoring its original
alpha channel. Transparent pixels remain alpha zero.

- [ ] **Step 4: Run weapon and stabilization tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_weapon_composite.py' `
  'tests\game_asset_api_test\test_animation_stabilization.py' -q
```

Expected: all tests pass and no temporal image generation is involved in the
weapon test.

- [ ] **Step 5: Commit rigid composition**

```powershell
git add game_asset_api/weapon_composite.py `
  tests/game_asset_api_test/test_weapon_composite.py
git commit -m 'Add rigid weapon animation composition'
```

### Task 7: Export A Godot 4.x Animation Bundle

**Files:**
- Create: `game_asset_api/godot_export.py`
- Create: `tests/game_asset_api_test/test_godot_export.py`
- Create: `scripts/validate_godot_export.py`

- [ ] **Step 1: Write failing artifact and timing tests**

```python
import json

from PIL import Image

from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_motion import plan_sword_attack
from game_asset_api.godot_export import write_godot_bundle
from game_asset_api.weapon_composite import WeaponTransform


def test_godot_bundle_matches_frames_regions_and_timing(tmp_path):
    request = parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": 8,
            "sprite_size": 64,
        }
    )
    plan = plan_sword_attack(8)
    frames = tuple(Image.new("RGBA", (64, 64), (index, 80, 160, 255)) for index in range(8))
    transforms = tuple(
        WeaponTransform(1.0, 0.0, (0.0, 0.0), (10.0, 10.0), (40.0, 10.0), "digest")
        for _ in frames
    )

    artifacts = write_godot_bundle(tmp_path, request, frames, plan.frames, ((0, 0),) * 8, transforms)

    assert [path.name for path in artifacts.frames] == [f"{index:03d}.png" for index in range(8)]
    assert artifacts.spritesheet.name == "spritesheet.png"
    assert artifacts.sprite_frames.name == "sprite_frames.tres"
    assert artifacts.metadata.name == "animation.json"
    assert artifacts.preview.name == "preview.gif"
    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["loop"] is False
    assert metadata["events"] == [{"frame": next(index for index, frame in enumerate(plan.frames) if "hit" in frame.events), "name": "hit"}]
    assert len(metadata["frames"]) == 8
    tres = artifacts.sprite_frames.read_text(encoding="utf-8")
    assert 'type="SpriteFrames"' in tres
    assert 'path="res://game_assets/cultivator_attack/spritesheet.png"' in tres
    assert '"loop": false' in tres
    assert '"speed": 12.0' in tres
```

- [ ] **Step 2: Run exporter tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_godot_export.py' -q
```

Expected: collection fails because `godot_export.py` does not exist.

- [ ] **Step 3: Implement the bundle writer**

Use the existing `write_sprite_sheet` helper for atlas layout. Add these public
types and entry point:

```python
@dataclass(frozen=True, slots=True)
class GodotArtifacts:
    frames: tuple[Path, ...]
    spritesheet: Path
    sprite_frames: Path
    metadata: Path
    preview: Path


def write_godot_bundle(
    output: Path,
    request: AnimationRequest,
    frames: tuple[Image.Image, ...],
    motion: tuple[MotionFrame, ...],
    translations: tuple[tuple[int, int], ...],
    weapon_transforms: tuple[WeaponTransform, ...],
) -> GodotArtifacts:
    if not (len(frames) == len(motion) == len(translations) == len(weapon_transforms)):
        raise ValueError("animation artifact counts must match")
    output.mkdir(parents=True, exist_ok=False)
    frame_directory = output / "frames"
    frame_directory.mkdir()
    paths = []
    for index, frame in enumerate(frames):
        path = frame_directory / f"{index:03d}.png"
        frame.save(path, format="PNG")
        paths.append(path)
    spritesheet, columns, rows = write_sprite_sheet(list(frames), output / "spritesheet.png")
    metadata = _write_metadata(output, request, motion, translations, weapon_transforms, columns, rows)
    sprite_frames = _write_sprite_frames(output, request, motion, columns)
    preview = _write_preview(output, frames, motion)
    return GodotArtifacts(tuple(paths), spritesheet, sprite_frames, metadata, preview)
```

`_write_metadata` must serialize the approved schema with sorted keys and a
trailing newline. Include frame index, duration, phase, events, atlas region,
root, alignment translation, and the full weapon transform. Build one top-level
event entry per frame event.

`_write_sprite_frames` must emit valid Godot 4 text-resource syntax:

```python
header = f'[gd_resource type="SpriteFrames" load_steps={len(frames) + 2} format=3]'
external = (
    '[ext_resource type="Texture2D" '
    f'path="{request.godot_resource_prefix}/spritesheet.png" id="1_atlas"]'
)
subresource = (
    f'[sub_resource type="AtlasTexture" id="AtlasTexture_{index:03d}"]\n'
    'atlas = ExtResource("1_atlas")\n'
    f'region = Rect2({x}, {y}, {request.sprite_size}, {request.sprite_size})'
)
frame_entry = (
    '{"duration": '
    f'{motion[index].duration * 12.0}, '
    f'"texture": SubResource("AtlasTexture_{index:03d}")}}'
)
```

`_write_preview` must save all frames with per-frame millisecond durations,
`save_all=True`, `append_images`, `disposal=2`, and no loop field.

- [ ] **Step 4: Add real Godot headless validation**

`scripts/validate_godot_export.py` takes `--godot`, `--bundle`, and
`--resource-prefix`. `--godot` defaults from `GODOT_BIN`. The validator reads
the prefix stored in `animation.json` and rejects a mismatched command-line
prefix. It creates a temporary Godot project, copies the bundle to the matching
resource prefix, formats that path into this validation script, and invokes the configured
binary with `--headless --path $project --script $script`:

```gdscript
extends SceneTree

func _initialize():
    var frames = load("%s/sprite_frames.tres")
    if frames == null or not frames is SpriteFrames:
        quit(1)
        return
    if not frames.has_animation(&"sword_attack"):
        quit(2)
        return
    quit(0)
```

Before invoking it, run `$godot --version`, require an output beginning with
`4.`, and fail clearly for a missing or non-4.x binary. Unit-test the command
construction with a fake executable script; the live Godot call is Task 12.

- [ ] **Step 5: Run exporter tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_godot_export.py' -q
```

Expected: bundle, timing, and fake headless validation tests pass.

- [ ] **Step 6: Commit Godot export**

```powershell
git add game_asset_api/godot_export.py `
  tests/game_asset_api_test/test_godot_export.py `
  scripts/validate_godot_export.py
git commit -m 'Add Godot production animation export'
```

### Task 8: Coordinate Atomic Production Processing

**Files:**
- Create: `game_asset_api/animation_pipeline.py`
- Create: `tests/game_asset_api_test/test_animation_pipeline.py`
- Create: `scripts/run_production_animation.py`

- [ ] **Step 1: Write a failing end-to-end processor test with a fake Comfy client**

```python
import json
from pathlib import Path

import pytest
from PIL import Image

from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_pipeline import AnimationProcessor


class FakeClient:
    def __init__(self, history):
        self.history = history
        self.graphs = []

    async def submit(self, graph):
        self.graphs.append(graph)
        return "prompt-animation"

    async def wait_for_prompt(self, prompt_id, timeout_seconds=1800):
        return self.history


@pytest.mark.asyncio
async def test_animation_processor_builds_bundle_and_publishes_once(tmp_path):
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    source = tmp_path / "output" / ".animation_work" / "job-id"
    source.mkdir(parents=True)
    records = []
    for index in range(8):
        path = source / f"source_{index:05d}_.png"
        _write_rgba_character(path, x_offset=index % 2)
        records.append({"filename": path.name, "subfolder": ".animation_work/job-id", "type": "output"})
    client = FakeClient(_history("73", records))
    processor = AnimationProcessor(tmp_path, client)

    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)
    prompt_id, generated = await processor.generate(request, "job-id", prepared, plan)
    stabilized = processor.stabilize(request, plan, generated)
    composited = processor.composite(plan, stabilized, prepared)
    staged = processor.export(request, "job-id", plan, stabilized, composited)
    artifacts = processor.validate_and_publish(request, "job-id", staged)

    assert prompt_id == "prompt-animation"
    assert artifacts.metadata.is_file()
    assert artifacts.sprite_frames.is_file()
    final = tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    assert artifacts.metadata.parent == final
    assert not (final.parent / ".production_action.tmp").exists()
    assert not (tmp_path / "output" / ".animation_work" / "job-id").exists()


@pytest.mark.asyncio
async def test_animation_processor_removes_temporary_outputs_after_failure(tmp_path):
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    processor = AnimationProcessor(tmp_path, FakeClient(_history("73", [])))
    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)

    with pytest.raises(ValueError, match="generated frame count"):
        await processor.generate(request, "job-id", prepared, plan)
    processor.cleanup("job-id")

    assert not (tmp_path / "output" / "game_assets" / "job-id" / ".production_action.tmp").exists()
    assert not (tmp_path / "output" / ".animation_work" / "job-id").exists()
```

Add these helpers to the test file:

```python
def _write_valid_runtime_inputs(root, frame_count):
    (root / "main.py").write_text("", encoding="utf-8")
    character = root / "input" / "characters" / "cultivator.png"
    weapon = root / "input" / "weapons" / "sword.png"
    descriptor = weapon.with_name("sword.json")
    character.parent.mkdir(parents=True)
    weapon.parent.mkdir(parents=True)
    Image.new("RGBA", (128, 128), (20, 120, 220, 255)).save(character)
    blade = Image.new("RGBA", (64, 16), (0, 0, 0, 0))
    for x in range(4, 60):
        blade.putpixel((x, 8), (220, 230, 240, 255))
    blade.save(weapon)
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": "sword.png",
                "grip": [4 / 63, 8 / 15],
                "tip": [59 / 63, 8 / 15],
                "default_layer": "behind_character",
            }
        ),
        encoding="utf-8",
    )
    return parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": frame_count,
            "sprite_size": 64,
            "seed": 42,
        }
    )


def _write_rgba_character(path, x_offset):
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    for y in range(80, 449):
        for x in range(180 + x_offset, 333 + x_offset):
            image.putpixel((x, y), (20, 120, 220, 255))
    image.save(path)


def _history(node_id, records):
    return {
        "status": {"status_str": "success", "messages": []},
        "outputs": {node_id: {"images": records}},
    }
```

- [ ] **Step 2: Run processor tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_pipeline.py' -q
```

Expected: collection fails because `animation_pipeline.py` does not exist.

- [ ] **Step 3: Implement the state-neutral processor**

```python
class AnimationProcessor:
    def __init__(self, project_root: Path, client: ComfyPromptClient, timeout_seconds: float = 1800):
        self.project_root = Path(project_root)
        self.input_root = self.project_root / "input"
        self.output_root = self.project_root / "output"
        self.jobs_root = self.output_root / "game_assets"
        self.client = client
        self.timeout_seconds = timeout_seconds

    def validate_inputs(self, request: AnimationRequest, job_id: str) -> PreparedAnimation:
        loaded = load_animation_inputs(self.input_root, request)
        staging = self.input_root / "game_assets" / job_id / "production"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        reference = staging / "reference.png"
        loaded.character.save(reference, format="PNG")
        return PreparedAnimation(loaded, reference, staging / "poses")

    def plan_motion(self, request, job_id, prepared):
        plan = plan_sword_attack(request.frame_count)
        write_pose_images(plan, prepared.pose_directory)
        return plan

    async def generate(self, request, job_id, prepared, plan):
        graph = build_production_animation_workflow(
            request,
            job_id,
            reference_image=_input_relative(self.input_root, prepared.reference),
            pose_images=tuple(_input_relative(self.input_root, path) for path in sorted(prepared.pose_directory.glob("*.png"))),
        )
        prompt_id = await self.client.submit(graph)
        history = await self.client.wait_for_prompt(prompt_id, self.timeout_seconds)
        records = image_records(history, OUTPUT_NODE_ID)
        if len(records) != request.frame_count:
            raise ValueError("generated frame count must equal requested frame_count")
        paths = tuple(_resolve_output_image(self.output_root, record) for record in records)
        return prompt_id, paths

    def stabilize(self, request, plan, generated):
        return stabilize_character_frames(generated, plan.frames, request.sprite_size)

    def composite(self, plan, stabilized, prepared):
        return composite_weapons(stabilized.frames, plan.frames, prepared.inputs.weapon)

    def export(self, request, job_id, plan, stabilized, composited):
        job_root = self.jobs_root / job_id
        temporary = job_root / ".production_action.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        return write_godot_bundle(
            temporary,
            request,
            composited.frames,
            plan.frames,
            stabilized.translations,
            composited.transforms,
        )

    def validate_and_publish(self, request, job_id, staged):
        _validate_artifacts(staged, request.frame_count, request.sprite_size)
        job_root = self.jobs_root / job_id
        temporary = job_root / ".production_action.tmp"
        final = job_root / "production_action"
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        self._remove_generation_work(job_id)
        return _rebased_artifacts(staged, final)

    def cleanup(self, job_id: str) -> None:
        shutil.rmtree(self.jobs_root / job_id / ".production_action.tmp", ignore_errors=True)
        self._remove_generation_work(job_id)
```

Define immutable `PreparedAnimation`, `_validate_artifacts`, and a narrow
`_rebased_artifacts` helper. `_validate_artifacts` must reopen every frame,
parse `animation.json`, verify the requested count/size and all relative
filenames, and confirm the expected `.tres`, sprite sheet, and preview exist
before any rename.
Sort generated Comfy records by normalized subfolder/filename before resolving
them. Keep all state transitions out of this module.

- [ ] **Step 4: Add the explicit CLI runner**

`scripts/run_production_animation.py` accepts `--root`, `--character-image`,
`--weapon`, `--asset-name`, `--character-prompt`, `--frame-count` choices
`2/8/12/16`, `--sprite-size`, `--seed`, `--job-id`, and `--base-url`. It
constructs `AnimationProcessor` and calls the same stages as the job runner.
The CLI may construct the frozen request directly for the internal two-frame
preflight; HTTP parsing must continue to reject frame count `2`.

- [ ] **Step 5: Run processor tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_animation_pipeline.py' -q
```

Expected: successful publication and failed cleanup tests pass.

- [ ] **Step 6: Commit the processor**

```powershell
git add game_asset_api/animation_pipeline.py `
  tests/game_asset_api_test/test_animation_pipeline.py `
  scripts/run_production_animation.py
git commit -m 'Add atomic production animation processing'
```

### Task 9: Integrate Production Jobs Into The Existing Serialized Queue

**Files:**
- Modify: `game_asset_api/jobs.py:29-162,165-360`
- Modify: `game_asset_api/app.py:31-80,120-160`
- Modify: `game_asset_api/__main__.py:8-25`
- Modify: `tests/game_asset_api_test/test_jobs.py:41-72,229-378`
- Modify: `tests/game_asset_api_test/test_app.py:16-130`

- [ ] **Step 1: Write failing durable-job and API tests**

Add these behaviors to the existing test files:

```python
def test_job_store_round_trips_a_production_animation_request(tmp_path):
    store = JobStore(tmp_path / "jobs")
    request = _animation_request()

    job = store.create_animation(request)
    transitioned = store.transition(job.id, JobStatus.VALIDATING_INPUTS)

    assert transitioned.kind is JobKind.PRODUCTION_ANIMATION
    assert transitioned.request == request
    manifest = json.loads((tmp_path / "jobs" / job.id / "job.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "production_animation"


def test_job_store_reads_existing_manifests_without_a_kind_as_game_assets(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create(_request())
    manifest_path = tmp_path / "jobs" / job.id / "job.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("kind")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert store.read(job.id).kind is JobKind.GAME_ASSET


@pytest.mark.asyncio
async def test_post_animation_request_queues_typed_job(aiohttp_client, tmp_path):
    runner = _FakeRunner(tmp_path)
    client = await aiohttp_client(create_app(runner))

    response = await client.post(
        "/v1/animations",
        json={
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
        },
    )

    payload = await response.json()
    assert response.status == 202
    assert runner.store.read(payload["job_id"]).kind is JobKind.PRODUCTION_ANIMATION


@pytest.mark.asyncio
async def test_completed_animation_job_exposes_godot_bundle_urls(aiohttp_client, tmp_path):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue_animation(_animation_request())
    for status in ANIMATION_STATUS_SEQUENCE:
        job = runner.store.transition(job.id, status)
    runner.store.transition(
        job.id,
        JobStatus.COMPLETED,
        outputs={
            "frame_000": f"/assets/{job.id}/production_action/frames/000.png",
            "spritesheet": f"/assets/{job.id}/production_action/spritesheet.png",
            "metadata": f"/assets/{job.id}/production_action/animation.json",
            "sprite_frames": f"/assets/{job.id}/production_action/sprite_frames.tres",
            "preview": f"/assets/{job.id}/production_action/preview.gif",
        },
    )
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/v1/jobs/{job.id}")

    assert await response.json() == {
        "job_id": job.id,
        "status": "completed",
        "frames": [f"/assets/{job.id}/production_action/frames/000.png"],
        "spritesheet": f"/assets/{job.id}/production_action/spritesheet.png",
        "metadata": f"/assets/{job.id}/production_action/animation.json",
        "sprite_frames": f"/assets/{job.id}/production_action/sprite_frames.tres",
        "preview": f"/assets/{job.id}/production_action/preview.gif",
    }
```

Add one async `JobRunner` test with a fake `AnimationProcessor`. It must assert
the exact status order, one prompt ID named `animation`, one final output map,
serialized GPU access, and `cleanup(job_id)` after an injected failure at each
processor stage.

- [ ] **Step 2: Run job and app tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_jobs.py' `
  'tests\game_asset_api_test\test_app.py' -q
```

Expected: failures for missing `JobKind`, production statuses,
`create_animation`, `enqueue_animation`, and `/v1/animations`.

- [ ] **Step 3: Extend the durable manifest without changing old requests**

Add:

```python
class JobKind(str, Enum):
    GAME_ASSET = "game_asset"
    PRODUCTION_ANIMATION = "production_animation"


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_CHARACTER = "generating_character"
    GENERATING_ACTION = "generating_action"
    POSTPROCESSING = "postprocessing"
    VALIDATING_INPUTS = "validating_inputs"
    MOTION_PLANNING = "motion_planning"
    TEMPORAL_GENERATION = "temporal_generation"
    CHARACTER_STABILIZATION = "character_stabilization"
    WEAPON_COMPOSITE = "weapon_composite"
    GODOT_EXPORT = "godot_export"
    VALIDATING_OUTPUTS = "validating_outputs"
    COMPLETED = "completed"
    FAILED = "failed"
```

Add the production chain to `_ALLOWED_TRANSITIONS` while retaining every old
transition. Change `Job` to carry `kind: JobKind`,
`request: AssetRequest | AnimationRequest`, and
`failed_stage: str | None = None`. Preserve all three fields in every
replacement `Job` created by `transition` and `record_prompt_id`. When
transitioning to `FAILED`, set `failed_stage` to the previous status value.

Keep `JobStore.create(AssetRequest)` unchanged for callers. Add:

```python
def create_animation(self, request: AnimationRequest) -> Job:
    return self._create(JobKind.PRODUCTION_ANIMATION, request)
```

Move shared creation into `_create`. Serialize `kind` in `_job_manifest`.
In `_job_from_manifest`, default an absent kind to `game_asset`; parse the
request with `parse_asset_request` for old jobs and `parse_animation_request`
for production jobs. Serialize and validate `failed_stage`. In
`create_animation`, replace a missing request seed with `secrets.randbits(64)`
before persistence so the actual production seed is durable and appears in
metadata.

- [ ] **Step 4: Dispatch production stages from the existing one-worker queue**

Define an `AnimationProcessorProtocol` beside `ComfyPromptClient` with only the
seven methods used below. Accept
`animation_processor: AnimationProcessorProtocol | None = None` in
`JobRunner.__init__`; this avoids importing the concrete pipeline into
`jobs.py` and creating a cycle. Add `enqueue_animation`, then split the current
`_process` body into `_process_game_asset`. Dispatch from `_process` by
`job.kind` and fail clearly if a production job is queued without a processor.

In `game_asset_api/__main__.py`, construct
`AnimationProcessor(project_root, client)` and pass it to `JobRunner`. Update
the existing constructor test to assert the client and processor instances are
shared correctly.

Implement `_process_animation` in this exact owner order:

```python
async def _process_animation(self, job: Job) -> None:
    job = self.store.transition(job.id, JobStatus.VALIDATING_INPUTS)
    prepared = self.animation_processor.validate_inputs(job.request, job.id)
    job = self.store.transition(job.id, JobStatus.MOTION_PLANNING)
    plan = self.animation_processor.plan_motion(job.request, job.id, prepared)
    job = self.store.transition(job.id, JobStatus.TEMPORAL_GENERATION)
    prompt_id, generated = await self.animation_processor.generate(
        job.request, job.id, prepared, plan
    )
    job = self.store.record_prompt_id(job.id, "animation", prompt_id)
    job = self.store.transition(job.id, JobStatus.CHARACTER_STABILIZATION)
    stabilized = self.animation_processor.stabilize(job.request, plan, generated)
    job = self.store.transition(job.id, JobStatus.WEAPON_COMPOSITE)
    composited = self.animation_processor.composite(plan, stabilized, prepared)
    job = self.store.transition(job.id, JobStatus.GODOT_EXPORT)
    staged = self.animation_processor.export(
        job.request, job.id, plan, stabilized, composited
    )
    job = self.store.transition(job.id, JobStatus.VALIDATING_OUTPUTS)
    artifacts = self.animation_processor.validate_and_publish(
        job.request, job.id, staged
    )
    self.store.transition(
        job.id,
        JobStatus.COMPLETED,
        outputs=_animation_outputs(job.id, artifacts),
    )
```

Update Task 8's processor implementation so `export` writes only to
`.production_action.tmp`; `validate_and_publish` checks every artifact count,
path, and metadata field before `os.replace` publishes the directory. This
preserves the required atomic boundary.

The outer `_process` exception handler must call `animation_processor.cleanup`
only for production jobs, then persist `FAILED` with the current stage and
error. Keep the existing exception list and existing game-asset behavior.

- [ ] **Step 5: Add the HTTP route and production URLs**

In `app.py`, register `POST /v1/animations`. Parse JSON with the same malformed
JSON handling as `_create_job`, call `parse_animation_request`, and enqueue with
`runner.enqueue_animation`.

Extend `_add_completed_outputs` with only these known keys:

```python
for key in ("spritesheet", "metadata", "sprite_frames", "preview"):
    if (url := _public_asset_url(job.id, job.outputs.get(key))) is not None:
        payload[key] = url
```

Keep `character_design` exclusive to old jobs and keep frame sorting unchanged.
For failed jobs, add `stage: job.failed_stage` when present. Add an HTTP test
that injects a temporal-generation failure and expects both the sanitized error
and `"stage": "temporal_generation"`.

- [ ] **Step 6: Run job and HTTP tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_jobs.py' `
  'tests\game_asset_api_test\test_app.py' `
  'tests\game_asset_api_test\test_animation_pipeline.py' -q
```

Expected: existing game-asset tests and new production tests all pass.

- [ ] **Step 7: Commit job integration**

```powershell
git add game_asset_api/jobs.py game_asset_api/app.py game_asset_api/__main__.py `
  game_asset_api/animation_pipeline.py `
  tests/game_asset_api_test/test_jobs.py `
  tests/game_asset_api_test/test_app.py `
  tests/game_asset_api_test/test_animation_pipeline.py
git commit -m 'Add production animation API jobs'
```

### Task 10: Deploy Pinned Temporal Dependencies And Workflow Discovery

**Files:**
- Modify: `game_asset_api/model_manifest.py:15-93`
- Modify: `game_asset_api/node_manifest.py:15-37`
- Modify: `game_asset_api/deployment.py:9-48`
- Modify: `scripts/deploy.py:18-120`
- Modify: `tests/game_asset_api_test/test_model_manifest.py`
- Modify: `tests/game_asset_api_test/test_node_manifest.py`
- Modify: `tests/game_asset_api_test/test_deployment.py`

- [ ] **Step 1: Write failing pinned-source tests**

Add assertions for these exact sources:

```python
def test_manifest_contains_verified_sdxl_motion_adapter():
    spec = next(spec for spec in MODEL_SPECS if spec.filename == "mm_sdxl_v10_beta.safetensors")
    assert spec.relative_dir == "animatediff_models"
    assert spec.size == 474_328_896
    assert spec.sha256 == "24c3c5f48006ce2ce7b06188622865c620b2d33db23b1af671cc1f21716b5826"
    assert spec.url == "https://hf-mirror.com/guoyww/animatediff-motion-adapter-sdxl-beta/resolve/26c864717b4d4b002bb48ae6c9d6bb431548c6cb/diffusion_pytorch_model.fp16.safetensors"
    assert spec.fallback_urls == (
        "https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta/resolve/26c864717b4d4b002bb48ae6c9d6bb431548c6cb/diffusion_pytorch_model.fp16.safetensors",
    )


def test_node_manifest_pins_animatediff_evolved():
    spec = next(spec for spec in NODE_SPECS if spec.name == "ComfyUI-AnimateDiff-Evolved")
    assert spec.revision == "d8d163cd90b1111f6227495e3467633676fbb346"
    assert spec.archive_url.endswith(spec.revision)


def test_workflow_names_include_production_animation():
    assert "production_animation_api.json" in WORKFLOW_NAMES
```

Add a model installer test where primary `curl` raises
`subprocess.CalledProcessError`, the partial remains resumable, the fallback
call writes valid bytes, and `install` publishes only the verified fallback.
Add `/object_info` fixture entries for the four approved `ADE_` node types and
the motion model option.

- [ ] **Step 2: Run deployment tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_model_manifest.py' `
  'tests\game_asset_api_test\test_node_manifest.py' `
  'tests\game_asset_api_test\test_deployment.py' -q
```

Expected: missing temporal model, node, workflow, fallback field, and discovery
requirements fail.

- [ ] **Step 3: Add fallback-aware model installation**

Extend `ModelSpec` with:

```python
fallback_urls: tuple[str, ...] = ()
```

Append the exact motion adapter spec from Step 1. In `install`, try
`(spec.url, *spec.fallback_urls)` in order. Reuse the same `.part` file and
`--continue-at -` command for every URL. Catch only
`subprocess.CalledProcessError`; re-raise the final failure. Never publish until
`verify_file` succeeds.

- [ ] **Step 4: Add the pinned node and workflow discovery**

Append:

```python
NodeSpec(
    name="ComfyUI-AnimateDiff-Evolved",
    archive_url=(
        "https://api.github.com/repos/Kosinkadink/"
        "ComfyUI-AnimateDiff-Evolved/zipball/"
        "d8d163cd90b1111f6227495e3467633676fbb346"
    ),
    revision="d8d163cd90b1111f6227495e3467633676fbb346",
)
```

Append `production_animation_api.json` to `WORKFLOW_NAMES` and make
`_API_WORKFLOW_NAMES` an explicit frozenset of all four API workflow names
rather than relying on a tuple slice.

Add discovery inputs for:

```python
("ADE_LoadAnimateDiffModel", "model_name")
("ADE_UseEvolvedSampling", "beta_schedule")
```

The production graph must also require
`ADE_ApplyAnimateDiffModelSimple`. Existing node and model installer scripts
already iterate their manifests; do not add parallel installers.

- [ ] **Step 5: Replace the old deployment smoke with production preflight**

Keep workflow publication, node installation, model installation, and
discovery order unchanged. Add `_write_smoke_weapon(comfy_root)` in
`scripts/deploy.py`; it writes a small transparent test sword and descriptor
under `input/game_assets/deployment-smoke/` with Pillow and JSON. Then invoke
`run_production_animation.py` with frame count `2`, sprite size `64`, the
official `input/example.png`, and that descriptor.

Update deployment tests to assert this exact final subprocess and keep every
existing skip flag behavior.

- [ ] **Step 6: Run all deployment-focused tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_model_manifest.py' `
  'tests\game_asset_api_test\test_node_manifest.py' `
  'tests\game_asset_api_test\test_deployment.py' `
  'tests\game_asset_api_test\test_animation_workflow.py' -q
```

Expected: all pinned-source, installer, workflow publication, discovery, and
preflight-order tests pass.

- [ ] **Step 7: Commit deployment support**

```powershell
git add game_asset_api/model_manifest.py game_asset_api/node_manifest.py `
  game_asset_api/deployment.py scripts/deploy.py `
  tests/game_asset_api_test/test_model_manifest.py `
  tests/game_asset_api_test/test_node_manifest.py `
  tests/game_asset_api_test/test_deployment.py
git commit -m 'Deploy production animation dependencies'
```

### Task 11: Document And Verify The Complete Repository

**Files:**
- Modify: `README.md`
- Modify: `tests/game_asset_api_test/test_app.py`
- Modify: `tests/game_asset_api_test/test_repository_audit.py` only if a new
  tracked path reveals a real policy gap

- [ ] **Step 1: Write a failing README contract test**

```python
def test_readme_documents_production_animation_and_godot_export():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (
        "POST /v1/animations",
        "weapon descriptor",
        "frame_count",
        "8, 12, or 16",
        "sprite_frames.tres",
        "animation.json",
        "GODOT_BIN",
        "curl 7.71",
    ):
        assert text in readme
```

- [ ] **Step 2: Run the README test and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_app.py::test_readme_documents_production_animation_and_godot_export' -q
```

Expected: failure for undocumented production fields.

- [ ] **Step 3: Document the production workflow concisely**

Add these sections to `README.md`:

- prerequisites including `curl >= 7.71`, Godot 4.x headless, and 16GB VRAM;
- verified AnimateDiff node/model deployment sources;
- weapon PNG and descriptor schema;
- `POST /v1/animations` request and asynchronous polling example;
- output bundle layout and the required Godot `res://` copy location;
- `GODOT_BIN` and `validate_godot_export.py` command;
- no-fallback behavior and stage-specific failure messages;
- 2-frame preflight and 8/12/16 production validation commands.

Every PowerShell example must check `$LASTEXITCODE` immediately after an
external command. Do not describe unsupported actions or engines.

- [ ] **Step 4: Run the complete suite and repository checks**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\audit_repository.py'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: all tests pass, repository audit exits `0`, and `git diff --check`
prints nothing.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md tests/game_asset_api_test/test_app.py
git commit -m 'Document production animation workflow'
```

### Task 12: Deploy And Run Fresh Production Acceptance

**Files:**
- Runtime only under `E:\ComfyUI`; do not track downloaded models, nodes,
  Godot binaries, inputs, or outputs
- Update after evidence: `docs/deployment-audit.md`

- [ ] **Step 1: Download and verify the Godot 4.7.1 validation binary**

Use the official release and keep it under the existing untracked runtime
area:

```powershell
$archive = 'E:\ComfyUI\.codex-runtime\godot\Godot_v4.7.1-stable_win64.exe.zip'
$directory = Split-Path -Parent $archive
New-Item -ItemType Directory -Force -Path $directory | Out-Null
curl.exe -x http://127.0.0.1:10809 --fail --location --continue-at - `
  --retry 10 --retry-all-errors --output "$archive.part" `
  'https://github.com/godotengine/godot/releases/download/4.7.1-stable/Godot_v4.7.1-stable_win64.exe.zip'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$hash = (Get-FileHash -Algorithm SHA512 -LiteralPath "$archive.part").Hash.ToLowerInvariant()
if ($hash -ne 'a6b02c527c18ba9936e63562032701432b2dc57d98d6483ceaccb00fe14af16af5773ae8a55e7b4d614edf121c4d9e420d870f804edb1dac16362298a01ce6c4') { throw 'Godot archive hash mismatch' }
Move-Item -LiteralPath "$archive.part" -Destination $archive -Force
Expand-Archive -LiteralPath $archive -DestinationPath $directory -Force
```

Expected: the extracted binary reports a version beginning with `4.7.1`.

- [ ] **Step 2: Deploy workflow, node, model, and two-frame preflight**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\deploy.py' `
  --comfy-root 'E:\ComfyUI' `
  --base-url 'http://127.0.0.1:8188' `
  --skip-discovery --skip-smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Use the fast mirror first. If direct GitHub archive transfer resets, set only
the command-scoped proxy `http://127.0.0.1:10809`; do not persist proxy config
in either Git repository.

Expected: verified node/model installation and workflow publication. Discovery
and smoke wait until ComfyUI has loaded the new node.

- [ ] **Step 3: Restart only the affected loopback services and revalidate discovery**

Record the current listeners and command lines. Stop only the process owning
`127.0.0.1:8188`, restart official ComfyUI with its existing
`--listen 127.0.0.1 --port 8188` command, and poll `/system_stats` until it
reports `0.28.0`. Then run:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\deploy.py' `
  --comfy-root 'E:\ComfyUI' `
  --base-url 'http://127.0.0.1:8188' `
  --skip-nodes --skip-models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: `/object_info` advertises every production workflow node and
`mm_sdxl_v10_beta.safetensors`, followed by a completed two-frame 64-pixel
production bundle.

Start the branch API on unused port `8191` from this worktree with:

```powershell
$env:COMFYUI_ROOT = 'E:\ComfyUI'
$env:GAME_ASSET_API_HOST = '127.0.0.1'
$env:GAME_ASSET_API_PORT = '8191'
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m game_asset_api
```

Do not stop the existing `8190` service during branch acceptance.

- [ ] **Step 4: Generate fresh 8-, 12-, and 16-frame jobs**

Place the approved cultivator reference plus a deterministic transparent sword
and descriptor under `E:\ComfyUI\input\game_assets\production-acceptance\`.
Submit three `/v1/animations` requests to `127.0.0.1:8191`, differing only in
`frame_count`. Poll each UUID until `completed` or `failed`; a failure stops the
acceptance run and records its stage before any tuning.

Use this exact submission and polling loop:

```powershell
$bundleByCount = @{}
$jobByCount = @{}
foreach ($frameCount in 8, 12, 16) {
  $body = @{
    asset_name = "cultivator_attack_$frameCount"
    character_image = 'game_assets/production-acceptance/reference.png'
    character_prompt = 'side-view cultivation swordsman in white and cyan robes'
    weapon = 'game_assets/production-acceptance/sword.json'
    action = 'sword_attack'
    frame_count = $frameCount
    sprite_size = 128
    seed = 42
    godot_resource_prefix = "res://game_assets/cultivator_attack_$frameCount"
  } | ConvertTo-Json
  $created = Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8191/v1/animations' `
    -ContentType 'application/json' -Body $body
  do {
    Start-Sleep -Seconds 2
    $job = Invoke-RestMethod `
      -Uri "http://127.0.0.1:8191/v1/jobs/$($created.job_id)"
  } while ($job.status -notin 'completed', 'failed')
  if ($job.status -eq 'failed') {
    throw "Animation $frameCount failed at $($job.stage): $($job.error)"
  }
  $jobByCount[$frameCount] = $job
  $bundleByCount[$frameCount] = Join-Path `
    'E:\ComfyUI\output\game_assets' `
    (Join-Path $created.job_id 'production_action')
}
```

Expected for every job:

- exact requested RGBA frame count at 128x128;
- non-empty transparent background;
- one `hit` event and `loop: false`;
- matching PNG, atlas, metadata, GIF, and Godot durations;
- grip and tip mapping error at most one pixel;
- no clipped foreground or weapon.

- [ ] **Step 5: Load all three exports with real Godot 4.x**

```powershell
$godot = 'E:\ComfyUI\.codex-runtime\godot\Godot_v4.7.1-stable_win64.exe'
$bundle8 = $bundleByCount[8]
& 'E:\ComfyUI\.venv\Scripts\python.exe' `
  '.\scripts\validate_godot_export.py' `
  --godot $godot --bundle $bundle8 `
  --resource-prefix 'res://game_assets/cultivator_attack_8'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Repeat for 12 and 16 frames. Expected: all three headless loads exit `0` and
report animation `sword_attack` with the requested frame count.

- [ ] **Step 6: Perform visual and numeric continuity review**

Create a side-by-side contact sheet and preview for all three jobs. Review:

- face, hair, robe, palette, silhouette, hands, and limb topology;
- planted foot and root stability;
- anticipation, acceleration, contact hold, follow-through, and recovery;
- invariant sword silhouette and length;
- hand-to-grip attachment and front/behind occlusion.

Report any remaining character-only diffusion drift separately from weapon
geometry. Do not classify the workflow production-ready if identity, limb
topology, or hand occlusion fails visual review even when automated checks pass.

- [ ] **Step 7: Run final fresh verification and record evidence**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\audit_repository.py'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Record test count, repository commit, ComfyUI official commit, installed node
revision, model size/hash, Godot version, live job IDs, output paths, numeric
checks, and honest visual verdict in `docs/deployment-audit.md`.

- [ ] **Step 8: Commit only the evidence update**

```powershell
git add docs/deployment-audit.md
git commit -m 'Audit production animation deployment'
```
