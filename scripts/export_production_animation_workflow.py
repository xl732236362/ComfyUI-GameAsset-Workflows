"""Export the representative production animation ComfyUI API prompt graph."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_workflow import build_production_animation_workflow


OUTPUT_PATH = ROOT / "workflows" / "production_animation_api.json"
JOB_ID = "example-production-animation"


def main() -> None:
    """Write the representative prompt-wrapped workflow artifact."""
    request = parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "white-robed cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": 8,
            "seed": 42,
        }
    )
    pose_images = tuple(
        f"game_assets/{JOB_ID}/poses/{index:03d}.png"
        for index in range(request.frame_count)
    )
    graph = build_production_animation_workflow(
        request,
        JOB_ID,
        reference_image=f"game_assets/{JOB_ID}/reference.png",
        pose_images=pose_images,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"prompt": graph}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
