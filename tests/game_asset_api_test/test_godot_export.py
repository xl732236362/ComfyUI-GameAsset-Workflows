import importlib.util
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_motion import plan_sword_attack
from game_asset_api.godot_export import GodotArtifacts, write_godot_bundle
from game_asset_api.weapon_composite import WeaponTransform


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_godot_export.py"


def _request():
    return parse_animation_request(
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


def _frames() -> tuple[Image.Image, ...]:
    return tuple(
        Image.new("RGBA", (64, 64), (index, 80, 160, 255))
        for index in range(8)
    )


def _transforms() -> tuple[WeaponTransform, ...]:
    return tuple(
        WeaponTransform(
            1.0,
            0.0,
            (0.0, 0.0),
            (10.0, 10.0),
            (40.0, 10.0),
            "digest",
        )
        for _ in range(8)
    )


def _write_bundle(tmp_path: Path) -> GodotArtifacts:
    return write_godot_bundle(
        tmp_path / "bundle",
        _request(),
        _frames(),
        plan_sword_attack(8).frames,
        ((0, 0),) * 8,
        _transforms(),
    )


def _load_validator():
    module_spec = importlib.util.spec_from_file_location(
        "validate_godot_export", VALIDATOR
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def test_godot_bundle_matches_frames_regions_and_timing(tmp_path):
    artifacts = _write_bundle(tmp_path)
    motion = plan_sword_attack(8).frames

    assert isinstance(artifacts, GodotArtifacts)
    assert [path.name for path in artifacts.frames] == [
        f"{index:03d}.png" for index in range(8)
    ]
    assert artifacts.spritesheet.name == "spritesheet.png"
    assert artifacts.sprite_frames.name == "sprite_frames.tres"
    assert artifacts.metadata.name == "animation.json"
    assert artifacts.preview.name == "preview.gif"

    metadata_text = artifacts.metadata.read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata_text.endswith("\n")
    assert metadata_text == json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    assert metadata["schema_version"] == 1
    assert metadata["asset"] == "cultivator_attack"
    assert metadata["action"] == "sword_attack"
    assert metadata["loop"] is False
    assert metadata["canvas"] == {"height": 64, "width": 64}
    assert metadata["frame_count"] == 8
    assert metadata["grid"] == {"columns": 3, "rows": 3}
    assert metadata["godot_resource_prefix"] == "res://game_assets/cultivator_attack"
    hit = next(index for index, frame in enumerate(motion) if "hit" in frame.events)
    assert metadata["events"] == [{"frame": hit, "name": "hit"}]
    assert len(metadata["frames"]) == 8
    first = metadata["frames"][0]
    assert first == {
        "alignment_translation": [0, 0],
        "duration": motion[0].duration,
        "events": [],
        "index": 0,
        "phase": motion[0].phase,
        "region": {"height": 64, "width": 64, "x": 0, "y": 0},
        "root": list(motion[0].root),
        "weapon": {
            "target_grip": list(motion[0].weapon_grip),
            "target_tip": list(motion[0].weapon_tip),
            "transform": {
                "rotation": 0.0,
                "scale": 1.0,
                "source_digest": "digest",
                "transformed_grip": [10.0, 10.0],
                "transformed_tip": [40.0, 10.0],
                "translation": [0.0, 0.0],
            },
        },
    }

    tres = artifacts.sprite_frames.read_text(encoding="utf-8")
    assert 'type="SpriteFrames"' in tres
    assert 'path="res://game_assets/cultivator_attack/spritesheet.png"' in tres
    assert 'region = Rect2(0, 0, 64, 64)' in tres
    assert '"loop": false' in tres
    assert '"speed": 12.0' in tres
    assert f'"duration": {motion[0].duration * 12.0}' in tres

    with Image.open(artifacts.preview) as preview:
        assert preview.n_frames == 8
        assert preview.info["duration"] == round(motion[0].duration * 100) * 10
        preview.seek(1)
        assert preview.info["duration"] == round(motion[1].duration * 100) * 10
        assert "loop" not in preview.info


def test_godot_bundle_rejects_mismatched_artifact_counts(tmp_path):
    with pytest.raises(ValueError, match="^animation artifact counts must match$"):
        write_godot_bundle(
            tmp_path / "bundle",
            _request(),
            _frames()[:-1],
            plan_sword_attack(8).frames,
            ((0, 0),) * 8,
            _transforms(),
        )


def test_godot_bundle_does_not_reuse_an_existing_output_directory(tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_godot_bundle(
            output,
            _request(),
            _frames(),
            plan_sword_attack(8).frames,
            ((0, 0),) * 8,
            _transforms(),
        )


def test_godot_artifacts_are_immutable():
    artifacts = GodotArtifacts((), Path("spritesheet.png"), Path("frames.tres"), Path("animation.json"), Path("preview.gif"))

    with pytest.raises(AttributeError):
        artifacts.preview = Path("other.gif")


def test_validator_copies_the_bundle_and_runs_godot_4_headlessly(tmp_path):
    artifacts = _write_bundle(tmp_path)
    validator = _load_validator()
    fake_godot = tmp_path / "godot.exe"
    fake_godot.write_text("fake", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "4.4.stable\n", "")
        project = Path(command[command.index("--path") + 1])
        resource = project / "game_assets" / "cultivator_attack"
        assert (resource / "sprite_frames.tres").is_file()
        assert (resource / "animation.json").is_file()
        assert (project / "validate_sprite_frames.gd").is_file()
        return subprocess.CompletedProcess(command, 0, "", "")

    validator.validate_bundle(
        fake_godot,
        artifacts.metadata.parent,
        "res://game_assets/cultivator_attack",
        runner=fake_run,
    )

    assert calls[0][0] == [str(fake_godot), "--version"]
    assert calls[1][0][0] == str(fake_godot)
    assert calls[1][0][1:] == ["--headless", "--path", calls[1][0][3], "--script", calls[1][0][5]]


def test_validator_rejects_parser_and_prefix_errors(tmp_path, monkeypatch):
    validator = _load_validator()

    with pytest.raises(SystemExit):
        validator.parse_args([])

    fake_godot = tmp_path / "godot.exe"
    fake_godot.write_text("fake", encoding="utf-8")
    metadata = tmp_path / "animation.json"
    metadata.write_text(
        json.dumps({"godot_resource_prefix": "res://game_assets/other"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resource prefix does not match"):
        validator.validate_bundle(
            fake_godot,
            tmp_path,
            "res://game_assets/cultivator_attack",
            runner=lambda *_args, **_kwargs: None,
        )

    monkeypatch.setenv("GODOT_BIN", str(fake_godot))
    parsed = validator.parse_args(
        [
            "--bundle",
            str(tmp_path),
            "--resource-prefix",
            "res://game_assets/cultivator_attack",
        ]
    )
    assert parsed.godot == fake_godot


def test_validator_rejects_an_unsafe_resource_prefix(tmp_path):
    validator = _load_validator()
    fake_godot = tmp_path / "godot.exe"
    fake_godot.write_text("fake", encoding="utf-8")
    resource_prefix = "res://game_assets\\..\\outside"
    (tmp_path / "animation.json").write_text(
        json.dumps({"godot_resource_prefix": resource_prefix}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="^resource prefix must be safe$"):
        validator.validate_bundle(
            fake_godot,
            tmp_path,
            resource_prefix,
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "4.4\n"),
        )
