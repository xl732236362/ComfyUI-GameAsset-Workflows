"""Deploy workflow sources and validate the target ComfyUI installation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.deployment import (
    publish_workflows,
    validate_comfy_root,
    validate_object_info,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the stable deployment command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--skip-nodes", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args(argv)


def discover_object_info(base_url: str) -> dict:
    """Return ComfyUI object metadata after validating its HTTP response."""
    url = f"{base_url.rstrip('/')}/object_info"
    try:
        with urlopen(url, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"ComfyUI object_info request failed with HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        code = error.code
        reason = error.reason
        error.close()
        raise RuntimeError(
            f"ComfyUI object_info request failed with HTTP {code}: {reason}"
        ) from error
    except RuntimeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid ComfyUI object_info response: {url}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid ComfyUI object_info response: {url}")
    return payload


def deploy(arguments: argparse.Namespace) -> None:
    """Run the deployment operations in their stable production order."""
    comfy_root, python = validate_comfy_root(arguments.comfy_root)
    python_command = str(python)
    root_argument = str(comfy_root)

    publish_workflows(ROOT / "workflows", comfy_root)

    if not arguments.skip_nodes:
        subprocess.run(
            [
                python_command,
                str(ROOT / "scripts" / "install_pose_workflow_nodes.py"),
                "--root",
                root_argument,
                "--python",
                python_command,
            ],
            check=True,
        )
    if not arguments.skip_models:
        subprocess.run(
            [
                python_command,
                str(ROOT / "scripts" / "install_game_asset_models.py"),
                "--root",
                root_argument,
            ],
            check=True,
        )
    if not arguments.skip_discovery:
        object_info = discover_object_info(arguments.base_url)
        validate_object_info(object_info, ROOT / "workflows")
    if not arguments.skip_smoke:
        reference = (comfy_root / "input" / "example.png").resolve()
        if not reference.is_file():
            raise ValueError(f"smoke reference not found: {reference}")
        weapon = _write_smoke_weapon(comfy_root)
        subprocess.run(
            [
                python_command,
                str(ROOT / "scripts" / "run_production_animation.py"),
                "--root",
                root_argument,
                "--character-image",
                reference.relative_to(comfy_root / "input").as_posix(),
                "--weapon",
                weapon.relative_to(comfy_root / "input").as_posix(),
                "--asset-name",
                "deployment-smoke",
                "--job-id",
                "deployment-smoke",
                "--character-prompt",
                "pixel art knight",
                "--frame-count",
                "2",
                "--sprite-size",
                "64",
                "--base-url",
                arguments.base_url,
            ],
            check=True,
        )


def _write_smoke_weapon(comfy_root: Path) -> Path:
    directory = comfy_root / "input" / "game_assets" / "deployment-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    weapon = directory / "sword.png"
    image = Image.new("RGBA", (16, 4), (0, 0, 0, 0))
    for x in range(2, 14):
        image.putpixel((x, 2), (220, 230, 240, 255))
    image.save(weapon, format="PNG")

    descriptor = directory / "sword.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": weapon.name,
                "grip": [0.125, 0.5],
                "tip": [0.875, 0.5],
                "default_layer": "behind_character",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return descriptor


def main(argv: list[str] | None = None) -> None:
    deploy(parse_arguments(argv))


if __name__ == "__main__":
    main()
