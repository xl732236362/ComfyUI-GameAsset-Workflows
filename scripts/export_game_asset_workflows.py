"""Export representative ComfyUI API prompt workflow artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.contracts import parse_asset_request
from game_asset_api.workflows import (
    build_action_workflow,
    build_character_workflow,
    reference_input_path,
)


WORKFLOW_DIRECTORY = ROOT / "workflows"
JOB_ID = "example-job"


def main() -> None:
    """Write representative API prompt JSON for character and action generation."""
    request = parse_asset_request(
        {
            "character_prompt": "armored knight with a blue tabard",
            "action_prompt": "idle breathing animation",
            "camera": "side",
        }
    )
    character = build_character_workflow(request, JOB_ID)
    action = build_action_workflow(request, JOB_ID, reference_input_path(JOB_ID))

    WORKFLOW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_workflow("pixel_character_design_api.json", character)
    _write_workflow("pixel_character_action_api.json", action)


def _write_workflow(name: str, graph: dict[str, dict[str, object]]) -> None:
    path = WORKFLOW_DIRECTORY / name
    path.write_text(
        json.dumps({"prompt": graph}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
