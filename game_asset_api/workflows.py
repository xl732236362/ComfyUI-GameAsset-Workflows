"""ComfyUI API prompt graphs for pixel character generation."""

from __future__ import annotations

import re

from game_asset_api.contracts import AssetRequest
from game_asset_api.postprocess import wan_source_frame_count
from game_asset_api.prompting import NEGATIVE, build_action_prompt, build_character_prompt


Workflow = dict[str, dict[str, object]]
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def reference_input_path(job_id: str) -> str:
    """Return the controlled ComfyUI input filename for a job reference image."""
    return f"game_assets/{_validate_job_id(job_id)}/reference.png"


def build_character_workflow(request: AssetRequest, job_id: str) -> Workflow:
    """Build the ComfyUI API prompt graph for a character reference image."""
    job_id = _validate_job_id(job_id)
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
            "class_type": "CLIPTextEncode",
            "inputs": {"text": build_character_prompt(request), "clip": ["2", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["2", 1]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": request.seed or 0,
                "steps": 30,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": "BiRefNet-general-epoch_244.safetensors"},
        },
        "9": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["8", 0], "image": ["7", 0]},
        },
        "10": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["7", 0], "alpha": ["13", 0]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": f"game_assets/{job_id}/reference_rgb",
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": f"game_assets/{job_id}/character",
            },
        },
        "13": {
            "class_type": "InvertMask",
            "inputs": {"mask": ["9", 0]},
        },
    }


def build_action_workflow(
    request: AssetRequest, job_id: str, reference_image: str
) -> Workflow:
    """Build the ComfyUI API prompt graph for a character action sequence."""
    expected_reference_image = reference_input_path(job_id)
    if reference_image != expected_reference_image:
        raise ValueError(
            f"reference_image must equal expected input path: {expected_reference_image}"
        )

    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan2.2_vae.safetensors"},
        },
        "4": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["1", 0], "shift": 8},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": build_action_prompt(request), "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["2", 0]},
        },
        "7": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": "BiRefNet-general-epoch_244.safetensors"},
        },
        "9": {
            "class_type": "LoadImage",
            "inputs": {"image": expected_reference_image},
        },
        "10": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": 512,
                "height": 512,
                "length": wan_source_frame_count(request.frame_count),
                "batch_size": 1,
                "start_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "seed": request.seed or 0,
                "steps": 20,
                "cfg": 5,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["10", 0],
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["7", 0], "image": ["12", 0]},
        },
        "14": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["12", 0], "alpha": ["17", 0]},
        },
        "15": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["14", 0],
                "upscale_method": "nearest-exact",
                "width": request.sprite_size,
                "height": request.sprite_size,
                "crop": "disabled",
            },
        },
        "16": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["15", 0],
                "filename_prefix": f"game_assets/{job_id}/wan_frames",
            },
        },
        "17": {
            "class_type": "InvertMask",
            "inputs": {"mask": ["13", 0]},
        },
    }


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("job_id must contain only ASCII letters, digits, hyphens, and underscores")
    return job_id
