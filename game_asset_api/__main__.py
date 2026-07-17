"""Run the local game asset API server."""

from __future__ import annotations

import os
from pathlib import Path

from aiohttp import web

from game_asset_api.app import create_app
from game_asset_api.comfy_client import ComfyClient
from game_asset_api.jobs import JobRunner


def main() -> None:
    """Construct and run the local game asset API."""
    port = _port_from_environment()
    project_root = Path(__file__).resolve().parents[1]
    client = ComfyClient()
    runner = JobRunner(project_root, client)
    app = create_app(runner, client)
    web.run_app(app, host=os.environ.get("GAME_ASSET_API_HOST", "127.0.0.1"), port=port)


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
