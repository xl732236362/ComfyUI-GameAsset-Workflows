"""Run the local game asset API server."""

from __future__ import annotations

import os
from pathlib import Path

from aiohttp import web

from game_asset_api.animation_pipeline import AnimationProcessor
from game_asset_api.app import create_app
from game_asset_api.comfy_client import ComfyClient
from game_asset_api.jobs import JobRunner


def main() -> None:
    """Construct and run the local game asset API."""
    port = _port_from_environment()
    project_root = _project_root_from_environment()
    client = ComfyClient()
    animation_processor = AnimationProcessor(project_root, client)
    runner = JobRunner(project_root, client, animation_processor=animation_processor)
    app = create_app(runner, client)
    web.run_app(app, host=os.environ.get("GAME_ASSET_API_HOST", "127.0.0.1"), port=port)


def _project_root_from_environment() -> Path:
    value = os.environ.get("COMFYUI_ROOT")
    if not value:
        raise ValueError("COMFYUI_ROOT must point to the ComfyUI installation")
    root = Path(value).expanduser().resolve()
    if not (root / "main.py").is_file():
        raise ValueError("COMFYUI_ROOT must contain main.py")
    return root


def _port_from_environment() -> int:
    value = os.environ.get("GAME_ASSET_API_PORT", "8190")
    try:
        port = int(value)
    except ValueError:
        raise ValueError("GAME_ASSET_API_PORT must be an integer") from None
    if not 0 <= port <= 65535:
        raise ValueError("GAME_ASSET_API_PORT must be between 0 and 65535")
    return port


if __name__ == "__main__":
    main()
