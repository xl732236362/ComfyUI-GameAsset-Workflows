"""Validate a generated Godot 4 SpriteFrames bundle with a local engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from tempfile import TemporaryDirectory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", type=Path, default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--resource-prefix", required=True)
    args = parser.parse_args(argv)
    if args.godot is None:
        parser.error("--godot is required when GODOT_BIN is not set")
    return args


def validate_bundle(
    godot: Path,
    bundle: Path,
    resource_prefix: str,
    *,
    runner=subprocess.run,
) -> None:
    if not godot.is_file():
        raise ValueError(f"Godot binary does not exist: {godot}")
    if not bundle.is_dir():
        raise ValueError(f"Godot bundle does not exist: {bundle}")

    metadata_path = bundle / "animation.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Godot bundle is missing animation.json: {bundle}") from error
    except json.JSONDecodeError as error:
        raise ValueError("Godot bundle animation.json is invalid") from error
    stored_prefix = metadata.get("godot_resource_prefix")
    if stored_prefix != resource_prefix:
        raise ValueError("resource prefix does not match animation metadata")

    resource_path = _resource_path(resource_prefix)
    try:
        version = runner(
            [str(godot), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"unable to run Godot binary: {godot}") from error
    if not version.stdout.lstrip().startswith("4."):
        raise ValueError("Godot binary must be version 4.x")

    with TemporaryDirectory() as temporary:
        project = Path(temporary)
        _write_project(project)
        target = project / resource_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, target)
        try:
            runner(
                [str(godot), "--headless", "--path", str(project), "--import"],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError("Godot import failed") from error
        script = project / "validate_sprite_frames.gd"
        script.write_text(_validation_script(resource_prefix), encoding="utf-8")
        try:
            runner(
                [
                    str(godot),
                    "--headless",
                    "--path",
                    str(project),
                    "--script",
                    str(script),
                ],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError("Godot headless SpriteFrames validation failed") from error


def _resource_path(resource_prefix: str) -> Path:
    if not resource_prefix.startswith("res://"):
        raise ValueError("resource prefix must begin with res://")
    suffix = resource_prefix.removeprefix("res://")
    path = PurePosixPath(suffix)
    if (
        not suffix
        or "\\" in suffix
        or path.is_absolute()
        or any(
            not part or part in {".", ".."} or ":" in part
            for part in path.parts
        )
    ):
        raise ValueError("resource prefix must be safe")
    return Path(*path.parts)


def _write_project(project: Path) -> None:
    (project / "project.godot").write_text(
        '[application]\nconfig/name="Godot Export Validation"\n', encoding="utf-8"
    )


def _validation_script(resource_prefix: str) -> str:
    return f'''extends SceneTree

func _initialize():
    var frames = load("{resource_prefix}/sprite_frames.tres")
    if frames == null or not frames is SpriteFrames:
        quit(1)
        return
    if not frames.has_animation(&"sword_attack"):
        quit(2)
        return
    quit(0)
'''


def main() -> None:
    args = parse_args()
    validate_bundle(args.godot, args.bundle, args.resource_prefix)


if __name__ == "__main__":
    main()
