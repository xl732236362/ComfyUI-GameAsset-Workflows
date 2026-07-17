"""Run the production sword-attack pipeline against a local ComfyUI server."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.animation_contracts import AnimationRequest, parse_animation_request
from game_asset_api.animation_pipeline import AnimationProcessor
from game_asset_api.comfy_client import ComfyClient
from game_asset_api.godot_export import GodotArtifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse manual production or two-frame preflight arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--character-image", required=True)
    parser.add_argument("--weapon", required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--character-prompt", required=True)
    parser.add_argument("--frame-count", type=int, choices=(2, 8, 12, 16), default=12)
    parser.add_argument(
        "--sprite-size", type=int, choices=(64, 96, 128, 256), default=128
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser.parse_args(argv)


def build_request(arguments: argparse.Namespace) -> AnimationRequest:
    """Build a validated production request, allowing only local two-frame preflight."""
    data = {
        "asset_name": arguments.asset_name,
        "character_image": arguments.character_image,
        "character_prompt": arguments.character_prompt,
        "weapon": arguments.weapon,
        "action": "sword_attack",
        "frame_count": 8 if arguments.frame_count == 2 else arguments.frame_count,
        "sprite_size": arguments.sprite_size,
    }
    if arguments.seed is not None:
        data["seed"] = arguments.seed
    request = parse_animation_request(data)
    if arguments.frame_count == 2:
        return replace(request, frame_count=2)
    return request


async def run(arguments: argparse.Namespace) -> GodotArtifacts:
    """Execute every state-neutral processor stage and return the final bundle."""
    request = build_request(arguments)
    async with ComfyClient(arguments.base_url) as client:
        processor = AnimationProcessor(
            arguments.root, client, timeout_seconds=arguments.timeout_seconds
        )
        prepared = processor.validate_inputs(request, arguments.job_id)
        plan = processor.plan_motion(request, arguments.job_id, prepared)
        _, generated = await processor.generate(
            request, arguments.job_id, prepared, plan
        )
        stabilized = processor.stabilize(request, plan, generated)
        composited = processor.composite(plan, stabilized, prepared)
        staged = processor.export(
            request, arguments.job_id, plan, stabilized, composited
        )
        return processor.validate_and_publish(request, arguments.job_id, staged)


def main(argv: list[str] | None = None) -> None:
    """Run the pipeline and print the published artifact paths as JSON."""
    artifacts = asyncio.run(run(parse_args(argv)))
    sys.stdout.write(
        json.dumps(
            {
                "frames": [str(path) for path in artifacts.frames],
                "spritesheet": str(artifacts.spritesheet),
                "sprite_frames": str(artifacts.sprite_frames),
                "metadata": str(artifacts.metadata),
                "preview": str(artifacts.preview),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
