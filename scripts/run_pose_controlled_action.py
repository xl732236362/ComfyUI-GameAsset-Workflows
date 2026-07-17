"""Run a deterministic pose-controlled sword attack through local ComfyUI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.comfy_client import ComfyClient
from game_asset_api.contracts import parse_asset_request
from game_asset_api.pose_runner import PoseRunResult, run_pose_controlled_action


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the local runner command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--character-prompt", required=True)
    parser.add_argument("--action-prompt", default="forward sword slash")
    parser.add_argument(
        "--camera",
        choices=("side", "front", "top_down", "three_quarter"),
        default="side",
    )
    parser.add_argument("--frame-count", type=int, choices=(2, 8), default=8)
    parser.add_argument(
        "--sprite-size", type=int, choices=(64, 96, 128, 256), default=128
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controlnet-strength", type=float, default=0.9)
    parser.add_argument("--ipadapter-weight", type=float, default=0.65)
    parser.add_argument(
        "--ipadapter-weight-type",
        choices=(
            "linear",
            "style transfer",
            "strong style transfer",
            "composition",
        ),
        default="style transfer",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser.parse_args(argv)


async def _run(arguments: argparse.Namespace) -> PoseRunResult:
    request = parse_asset_request(
        {
            "character_prompt": arguments.character_prompt,
            "action_prompt": arguments.action_prompt,
            "camera": arguments.camera,
            "frame_count": arguments.frame_count,
            "sprite_size": arguments.sprite_size,
            "seed": arguments.seed,
        }
    )
    async with ComfyClient(arguments.base_url) as client:
        return await run_pose_controlled_action(
            arguments.root,
            client,
            request,
            arguments.job_id,
            arguments.reference,
            controlnet_strength=arguments.controlnet_strength,
            ipadapter_weight=arguments.ipadapter_weight,
            ipadapter_weight_type=arguments.ipadapter_weight_type,
            timeout_seconds=arguments.timeout_seconds,
        )


def main(argv: list[str] | None = None) -> None:
    """Run the requested action and print stable output paths as JSON."""
    result = asyncio.run(_run(parse_arguments(argv)))
    payload = json.dumps(
        {
            "frames": [str(path) for path in result.frames],
            "spritesheet": str(result.sprite_sheet),
            "prompt_ids": list(result.prompt_ids),
        },
        indent=2,
    )
    sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
