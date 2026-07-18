"""Batched temporal ComfyUI graph for production character animations."""

from __future__ import annotations

from collections.abc import Sequence

from game_asset_api.animation_contracts import AnimationRequest
from game_asset_api.prompting import NEGATIVE


Workflow = dict[str, dict[str, object]]
OUTPUT_NODE_ID = "73"


def build_production_animation_workflow(
    request: AnimationRequest,
    job_id: str,
    *,
    reference_image: str,
    pose_images: Sequence[str],
) -> Workflow:
    """Build one pose-conditioned temporal animation prompt graph."""
    if len(pose_images) != request.frame_count:
        raise ValueError(
            f"pose_images must contain exactly {request.frame_count} images"
        )
    positive_prompt = ", ".join(
        (
            request.character_prompt,
            "fixed side view",
            "locked camera",
            "consistent identity",
            "empty hands",
            "both hands empty",
            "single character",
            "pixel-art",
        )
    )
    negative_prompt = ", ".join(
        (
            NEGATIVE,
            "sword, weapon, scabbard, staff, cane, polearm, wand, held object, multiple characters, character sheet, collage, grid, panels",
            "camera drift",
            "duplicate limbs",
            "cropped character",
        )
    )
    graph: Workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "pixel-art-xl.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image},
        },
        "4": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {
                "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
            },
        },
        "5": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            },
        },
        "6": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
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
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 1]},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 1]},
        },
    }

    pose_node_ids = []
    for index, pose_image in enumerate(pose_images):
        node_id = str(20 + index)
        graph[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image},
        }
        pose_node_ids.append(node_id)

    batched_poses = [pose_node_ids[0], 0]
    for index, pose_node_id in enumerate(pose_node_ids[1:], start=1):
        batch_node_id = str(39 + index)
        graph[batch_node_id] = {
            "class_type": "ImageBatch",
            "inputs": {
                "image1": batched_poses,
                "image2": [pose_node_id, 0],
            },
        }
        batched_poses = [batch_node_id, 0]

    graph.update(
        {
            "60": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "OpenPoseXL2.safetensors"},
            },
            "61": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "control_net": ["60", 0],
                    "image": batched_poses,
                    "strength": 0.9,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            },
            "66": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": request.frame_count,
                },
            },
            "67": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["6", 0],
                    "seed": request.seed or 0,
                    "steps": 30,
                    "cfg": 7,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "positive": ["61", 0],
                    "negative": ["61", 1],
                    "latent_image": ["66", 0],
                    "denoise": 1.0,
                },
            },
            "68": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["67", 0], "vae": ["1", 2]},
            },
            "69": {
                "class_type": "LoadBackgroundRemovalModel",
                "inputs": {
                    "bg_removal_name": "BiRefNet-general-epoch_244.safetensors"
                },
            },
            "70": {
                "class_type": "RemoveBackground",
                "inputs": {"bg_removal_model": ["69", 0], "image": ["68", 0]},
            },
            "71": {
                "class_type": "InvertMask",
                "inputs": {"mask": ["70", 0]},
            },
            "72": {
                "class_type": "JoinImageWithAlpha",
                "inputs": {"image": ["68", 0], "alpha": ["71", 0]},
            },
            OUTPUT_NODE_ID: {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["72", 0],
                    "filename_prefix": f".animation_work/{job_id}/source",
                },
            },
        }
    )
    return graph
