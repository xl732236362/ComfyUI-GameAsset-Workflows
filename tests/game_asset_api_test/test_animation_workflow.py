import json
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from game_asset_api.animation_contracts import AnimationRequest, parse_animation_request
from game_asset_api.animation_workflow import (
    OUTPUT_NODE_ID,
    build_production_animation_workflow,
)
from game_asset_api.prompting import NEGATIVE


ROOT = Path(__file__).resolve().parents[2]


def _request(frame_count=8, seed=42):
    return AnimationRequest(
        asset_name="cultivator_attack",
        character_image="characters/cultivator.png",
        character_prompt="white-robed cultivator",
        weapon="weapons/sword.json",
        action="sword_attack",
        frame_count=frame_count,
        seed=seed,
    )


def _pose_images(job_id, frame_count):
    return tuple(
        f"game_assets/{job_id}/poses/{index:03d}.png"
        for index in range(frame_count)
    )


def _graph(frame_count=8, seed=42):
    job_id = "production-job"
    return build_production_animation_workflow(
        _request(frame_count, seed),
        job_id,
        reference_image=f"game_assets/{job_id}/reference.png",
        pose_images=_pose_images(job_id, frame_count),
    )


def _single_node(graph, class_type):
    nodes = [node for node in graph.values() if node["class_type"] == class_type]
    assert len(nodes) == 1
    return nodes[0]


def _normalized_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_production_animation_workflow_builds_one_batched_pose_graph():
    graph = _graph()

    assert OUTPUT_NODE_ID == "73"
    assert len(graph) == 33
    assert sum(node["class_type"] == "LoadImage" for node in graph.values()) == 9
    assert sum(node["class_type"] == "ImageBatch" for node in graph.values()) == 7
    assert _single_node(graph, "CheckpointLoaderSimple")["inputs"] == {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
    assert _single_node(graph, "LoraLoader")["inputs"]["lora_name"] == (
        "pixel-art-xl.safetensors"
    )

    ipadapter_model = _single_node(graph, "IPAdapterModelLoader")
    assert ipadapter_model["inputs"] == {
        "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
    }
    clip_vision = _single_node(graph, "CLIPVisionLoader")
    assert clip_vision["inputs"] == {
        "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    }
    ipadapter = _single_node(graph, "IPAdapterAdvanced")
    assert ipadapter["inputs"] == {
        "model": ["2", 0],
        "ipadapter": ["4", 0],
        "image": ["3", 0],
        "clip_vision": ["5", 0],
        "weight": 0.8,
        "weight_type": "style transfer",
        "combine_embeds": "concat",
        "start_at": 0.0,
        "end_at": 1.0,
        "embeds_scaling": "V only",
    }

    assert graph["3"]["inputs"]["image"] == "game_assets/production-job/reference.png"
    assert [
        graph[str(node_id)]["inputs"]["image"] for node_id in range(20, 28)
    ] == list(_pose_images("production-job", 8))
    assert graph["40"] == {
        "class_type": "ImageBatch",
        "inputs": {"image1": ["20", 0], "image2": ["21", 0]},
    }
    assert graph["46"] == {
        "class_type": "ImageBatch",
        "inputs": {"image1": ["45", 0], "image2": ["27", 0]},
    }
    assert graph["60"] == {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": "OpenPoseXL2.safetensors"},
    }
    assert graph["61"]["inputs"]["image"] == ["46", 0]
    assert graph["61"]["inputs"]["strength"] == 0.9
    assert _single_node(graph, "EmptyLatentImage")["inputs"] == {
        "width": 1024,
        "height": 1024,
        "batch_size": 8,
    }

    assert not any(node["class_type"].startswith("ADE_") for node in graph.values())

    sampler = _single_node(graph, "KSampler")
    assert sampler["inputs"] == {
        "model": ["6", 0],
        "seed": 42,
        "steps": 30,
        "cfg": 7,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "positive": ["61", 0],
        "negative": ["61", 1],
        "latent_image": ["66", 0],
        "denoise": 1.0,
    }
    assert _single_node(graph, "LoadBackgroundRemovalModel")["inputs"] == {
        "bg_removal_name": "BiRefNet-general-epoch_244.safetensors"
    }
    assert graph["71"] == {
        "class_type": "InvertMask",
        "inputs": {"mask": ["70", 0]},
    }
    assert graph["72"] == {
        "class_type": "JoinImageWithAlpha",
        "inputs": {"image": ["68", 0], "alpha": ["71", 0]},
    }
    assert graph[OUTPUT_NODE_ID] == {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["72", 0],
            "filename_prefix": ".animation_work/production-job/source",
        },
    }


def test_production_animation_workflow_prompts_for_an_unarmed_character():
    graph = _graph()

    positive = graph["7"]["inputs"]["text"]
    negative = graph["8"]["inputs"]["text"]

    for prompt_fragment in (
        "white-robed cultivator",
        "fixed side view",
        "locked camera",
        "consistent identity",
        "empty hands",
        "both hands empty",
        "single character",
        "pixel-art",
    ):
        assert prompt_fragment in positive
    assert "weapons/sword.json" not in positive
    assert "sword" not in positive.lower()
    assert NEGATIVE in negative
    for prompt_fragment in (
        "sword",
        "weapon",
        "scabbard",
        "staff",
        "cane",
        "polearm",
        "wand",
        "held object",
        "multiple characters",
        "character sheet",
        "collage",
        "grid",
        "panels",
        "camera drift",
        "duplicate limbs",
        "cropped character",
    ):
        assert prompt_fragment in negative


def test_production_animation_workflow_rejects_pose_count_mismatch():
    request = _request()

    with pytest.raises(ValueError, match="^pose_images must contain exactly 8 images$"):
        build_production_animation_workflow(
            request,
            "production-job",
            reference_image="game_assets/production-job/reference.png",
            pose_images=_pose_images("production-job", 7),
        )


def test_production_animation_workflow_uses_two_frame_preflight_batch():
    graph = _graph(frame_count=2, seed=None)

    assert not any(node["class_type"].startswith("ADE_") for node in graph.values())
    assert graph["61"]["inputs"]["image"] == ["40", 0]
    assert _single_node(graph, "EmptyLatentImage")["inputs"]["batch_size"] == 2
    assert _single_node(graph, "KSampler")["inputs"]["seed"] == 0


@pytest.mark.parametrize(
    "frame_count",
    (2, 8, 12, 16),
)
def test_production_animation_workflow_batches_each_supported_frame_count(frame_count):
    graph = _graph(frame_count=frame_count)

    assert not any(node["class_type"].startswith("ADE_") for node in graph.values())
    assert sum(node["class_type"] == "ImageBatch" for node in graph.values()) == (
        frame_count - 1
    )
    assert _single_node(graph, "EmptyLatentImage")["inputs"]["batch_size"] == frame_count


def test_animation_workflow_exporter_rebuilds_the_committed_json_artifact(tmp_path):
    temporary_root = tmp_path / "exporter_repo"
    script = temporary_root / "scripts" / "export_production_animation_workflow.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / script.name, script)
    shutil.copytree(ROOT / "game_asset_api", temporary_root / "game_asset_api")

    subprocess.run([sys.executable, str(script)], cwd=temporary_root, check=True)

    generated_path = temporary_root / "workflows" / "production_animation_api.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    request = replace(
        parse_animation_request(
            {
                "asset_name": "cultivator_attack",
                "character_image": "characters/cultivator.png",
                "character_prompt": "white-robed cultivator",
                "weapon": "weapons/sword.json",
                "action": "sword_attack",
                "frame_count": 8,
                "seed": 42,
            }
        ),
        frame_count=1,
    )
    expected_graph = build_production_animation_workflow(
        request,
        "example-production-animation",
        reference_image="game_assets/example-production-animation/reference.png",
        pose_images=_pose_images("example-production-animation", 1),
    )
    committed = json.loads(
        (ROOT / "workflows" / "production_animation_api.json").read_text(
            encoding="utf-8"
        )
    )

    assert _normalized_json(generated) == _normalized_json({"prompt": expected_graph})
    assert _normalized_json(committed) == _normalized_json(generated)
    assert generated_path.read_bytes().endswith(b"\n")
