import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from game_asset_api.contracts import parse_asset_request


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_MODULE = "game_asset_api.workflows"


def _workflows_module():
    assert importlib.util.find_spec(WORKFLOWS_MODULE) is not None
    return importlib.import_module(WORKFLOWS_MODULE)


def test_character_workflow_applies_pixel_lora_and_saves_rgb_and_alpha_images():
    workflows = _workflows_module()
    request = parse_asset_request(
        {"character_prompt": "knight", "action_prompt": "idle"}
    )

    graph = workflows.build_character_workflow(request, "job-1")

    assert graph["2"] == {
        "class_type": "LoraLoader",
        "inputs": {
            "model": ["1", 0],
            "clip": ["1", 1],
            "lora_name": "pixel-art-xl.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
        },
    }
    assert graph["9"]["class_type"] == "RemoveBackground"
    assert graph["13"] == {
        "class_type": "InvertMask",
        "inputs": {"mask": ["9", 0]},
    }
    assert graph["10"]["class_type"] == "JoinImageWithAlpha"
    assert graph["10"]["inputs"]["alpha"] == ["13", 0]
    assert graph["11"]["class_type"] == "SaveImage"
    assert graph["12"]["class_type"] == "SaveImage"


def test_action_workflow_uses_reference_as_the_wan_start_image():
    workflows = _workflows_module()
    request = parse_asset_request(
        {"character_prompt": "knight", "action_prompt": "idle", "frame_count": 8}
    )

    graph = workflows.build_action_workflow(
        request, "job-1", "game_assets/job-1/reference.png"
    )

    assert graph["10"]["class_type"] == "Wan22ImageToVideoLatent"
    assert graph["10"]["inputs"]["length"] == 9
    assert graph["10"]["inputs"]["start_image"] == ["9", 0]
    assert graph["17"] == {
        "class_type": "InvertMask",
        "inputs": {"mask": ["13", 0]},
    }
    assert graph["14"]["inputs"]["alpha"] == ["17", 0]


def test_reference_input_path_is_controlled_by_the_job_id():
    workflows = _workflows_module()
    reference_path = getattr(workflows, "reference_input_path", None)

    assert reference_path is not None
    assert reference_path("job-1") == "game_assets/job-1/reference.png"


def test_action_workflow_rejects_an_uncontrolled_reference_path():
    workflows = _workflows_module()
    request = parse_asset_request({"character_prompt": "knight", "action_prompt": "idle"})

    with pytest.raises(ValueError, match="expected input path"):
        workflows.build_action_workflow(request, "job-1", "../../outside.png")


@pytest.mark.parametrize("job_id", ["", "../escape", "job/path", "job space", "job.1"])
def test_character_workflow_rejects_unsafe_job_ids(job_id):
    workflows = _workflows_module()
    request = parse_asset_request({"character_prompt": "knight", "action_prompt": "idle"})

    with pytest.raises(ValueError, match="job_id"):
        workflows.build_character_workflow(request, job_id)


def test_export_script_writes_prompt_wrapped_workflow_json_artifacts():
    script = ROOT / "scripts" / "export_game_asset_workflows.py"

    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    assert not (ROOT / "user").exists()

    for name in ("pixel_character_design_api.json", "pixel_character_action_api.json"):
        path = ROOT / "workflows" / name
        with path.open(encoding="utf-8") as workflow_file:
            workflow = json.load(workflow_file)
        assert isinstance(workflow.get("prompt"), dict)

    action = json.loads(
        (ROOT / "workflows" / "pixel_character_action_api.json").read_text(
            encoding="utf-8"
        )
    )
    assert action["prompt"]["9"]["inputs"]["image"] == (
        "game_assets/example-job/reference.png"
    )
