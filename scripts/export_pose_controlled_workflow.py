"""Export a representative pose-controlled ComfyUI API prompt graph."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.contracts import parse_asset_request
from game_asset_api.pose_workflow import build_pose_controlled_workflow


OUTPUT_PATH = ROOT / "workflows" / "pose_controlled_pixel_action_api.json"


def main() -> None:
    """Write the representative prompt-wrapped workflow artifact."""
    request = parse_asset_request(
        {
            "character_prompt": "white-haired xianxia swordsman in flowing robes",
            "action_prompt": "forward sword slash",
            "camera": "side",
            "frame_count": 8,
            "seed": 42,
            "sprite_size": 128,
        }
    )
    graph = build_pose_controlled_workflow(request, "example-pose-job", frame_index=0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"prompt": graph}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
