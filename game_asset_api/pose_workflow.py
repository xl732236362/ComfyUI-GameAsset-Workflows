"""Pose-controlled ComfyUI API graph for one pixel character action frame."""

from __future__ import annotations

from game_asset_api.contracts import AssetRequest
from game_asset_api.prompting import NEGATIVE, build_action_prompt
from game_asset_api.workflows import Workflow, reference_input_path


def pose_input_path(job_id: str, frame_index: int) -> str:
    """Return the controlled ComfyUI input filename for one authored pose."""
    job_directory = reference_input_path(job_id).rsplit("/", 1)[0]
    return f"{job_directory}/poses/{frame_index:03d}.png"


def build_pose_controlled_workflow(
    request: AssetRequest,
    job_id: str,
    frame_index: int,
    *,
    controlnet_strength: float = 0.9,
    ipadapter_weight: float = 0.65,
    ipadapter_weight_type: str = "style transfer",
) -> Workflow:
    """Build the API prompt graph for one pose-authored action frame."""
    reference_image = reference_input_path(job_id)
    pose_image = pose_input_path(job_id, frame_index)
    positive_prompt = ", ".join(
        (
            request.character_prompt,
            build_action_prompt(request),
            "single character, full body, centered, plain contrasting background",
        )
    )
    negative_prompt = ", ".join(
        (
            NEGATIVE,
            "multiple characters, duplicate limbs, extra arms, extra legs",
            "missing weapon, deformed weapon, cropped character, changing camera",
        )
    )

    return {
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
                "weight": ipadapter_weight,
                "weight_type": ipadapter_weight_type,
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.85,
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
        "9": {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image},
        },
        "10": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "OpenPoseXL2.safetensors"},
        },
        "11": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "control_net": ["10", 0],
                "image": ["9", 0],
                "strength": controlnet_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "12": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "seed": request.seed or 0,
                "steps": 30,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["12", 0],
                "denoise": 1.0,
            },
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["1", 2]},
        },
        "15": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {
                "bg_removal_name": "BiRefNet-general-epoch_244.safetensors"
            },
        },
        "16": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["15", 0], "image": ["14", 0]},
        },
        "17": {
            "class_type": "InvertMask",
            "inputs": {"mask": ["16", 0]},
        },
        "18": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["14", 0], "alpha": ["17", 0]},
        },
        "19": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["18", 0],
                "upscale_method": "nearest-exact",
                "width": request.sprite_size,
                "height": request.sprite_size,
                "crop": "disabled",
            },
        },
        "20": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["19", 0],
                "filename_prefix": (
                    f"game_assets/{job_id}/pose_frames/{frame_index:03d}"
                ),
            },
        },
    }
