"""Batched temporal ComfyUI graph for production character animations."""

from __future__ import annotations

from collections.abc import Sequence

from game_asset_api.animation_contracts import AnimationRequest
from game_asset_api.prompting import NEGATIVE


Workflow = dict[str, dict[str, object]]
OUTPUT_NODE_ID = "73"
_CONTEXT_OPTIONS = {
    2: (2, 0),
    8: (8, 0),
    12: (8, 2),
    16: (8, 2),
}


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
    context_length, context_overlap = _context_options(request.frame_count)
    positive_prompt = ", ".join(
        (
            request.character_prompt,
            "fixed side view",
            "locked camera",
            "consistent identity",
            "empty hands",
            "pixel-art",
        )
    )
    negative_prompt = ", ".join(
        (
            NEGATIVE,
            "sword, weapon, scabbard",
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
        node_id = str(9 + index)
        graph[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image},
        }
        pose_node_ids.append(node_id)

    batched_poses = [pose_node_ids[0], 0]
    for index, pose_node_id in enumerate(pose_node_ids[1:], start=1):
        batch_node_id = str(24 + index)
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
            "40": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "OpenPoseXL2.safetensors"},
            },
            "41": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "control_net": ["40", 0],
                    "image": batched_poses,
                    "strength": 0.9,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            },
            "42": {
                "class_type": "ADE_LoadAnimateDiffModel",
                "inputs": {"model_name": "mm_sdxl_v10_beta.safetensors"},
            },
            "43": {
                "class_type": "ADE_StandardUniformContextOptions",
                "inputs": {
                    "context_length": context_length,
                    "context_overlap": context_overlap,
                    "context_stride": 1,
                    "context_schedule": "uniform",
                    "closed_loop": False,
                    "fuse_method": "flat",
                },
            },
            "44": {
                "class_type": "ADE_ApplyAnimateDiffModelSimple",
                "inputs": {
                    "motion_model": ["42", 0],
                    "context_options": ["43", 0],
                },
            },
            "45": {
                "class_type": "ADE_UseEvolvedSampling",
                "inputs": {
                    "model": ["6", 0],
                    "m_models": ["44", 0],
                    "beta_schedule": "autoselect",
                },
            },
            "46": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": request.frame_count,
                },
            },
            "47": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["45", 0],
                    "seed": request.seed or 0,
                    "steps": 30,
                    "cfg": 7,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "positive": ["41", 0],
                    "negative": ["41", 1],
                    "latent_image": ["46", 0],
                    "denoise": 1.0,
                },
            },
            "48": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["47", 0], "vae": ["1", 2]},
            },
            "49": {
                "class_type": "LoadBackgroundRemovalModel",
                "inputs": {
                    "bg_removal_name": "BiRefNet-general-epoch_244.safetensors"
                },
            },
            "50": {
                "class_type": "RemoveBackground",
                "inputs": {"bg_removal_model": ["49", 0], "image": ["48", 0]},
            },
            "51": {
                "class_type": "InvertMask",
                "inputs": {"mask": ["50", 0]},
            },
            "52": {
                "class_type": "JoinImageWithAlpha",
                "inputs": {"image": ["48", 0], "alpha": ["51", 0]},
            },
            OUTPUT_NODE_ID: {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["52", 0],
                    "filename_prefix": f".animation_work/{job_id}/source",
                },
            },
        }
    )
    return graph


def _context_options(frame_count: int) -> tuple[int, int]:
    try:
        return _CONTEXT_OPTIONS[frame_count]
    except KeyError:
        raise ValueError("frame_count must be one of 2, 8, 12, 16") from None
