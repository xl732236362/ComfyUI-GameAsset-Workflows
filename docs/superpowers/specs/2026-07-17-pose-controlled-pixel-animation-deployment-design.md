# Pose-Controlled Pixel Animation Deployment Design

## Goal

Deploy a local, repeatable ComfyUI workflow for producing 2D pixel-game
character action frames with per-frame pose control and character-reference
conditioning. It replaces Wan image-to-video as the production path for
walks, attacks, and other exact game actions.

The target machine has an NVIDIA RTX 4070 Ti SUPER with 16 GB VRAM, 541 GB
free disk space, SDXL Base 1.0, Pixel Art XL, and BiRefNet already installed.
It has no ControlNet, IP-Adapter, pose-preprocessor nodes, or clip-vision
models.

## Decision

Use SDXL Base 1.0 with the installed Pixel Art XL LoRA. For every target frame,
apply an OpenPose control image through an SDXL OpenPose ControlNet and apply
the approved character reference through IP-Adapter Plus. The image then runs
through the existing BiRefNet alpha step and nearest-neighbor scaling before
sprite-sheet assembly.

Generation is one 512 by 512 frame at a time on the 16 GB card. Eight supplied
pose images define an attack or walk cycle; the workflow does not ask a video
model to invent timing, limbs, or weapon positions. A fixed seed is used while
testing one pose sequence so changes remain attributable.

## Components

Install these node repositories from pinned source archives served by the
official GitHub API. Direct Git transport is unavailable on the current
network, while the API archive endpoint is reachable:

- `Fannovel16/comfyui_controlnet_aux` at commit
  `e8b689a513c3e6b63edc44066560ca5919c0576e` (Apache-2.0) supplies
  DWPose/OpenPose preprocessors and pose-keypoint utilities.
- `cubiq/ComfyUI_IPAdapter_plus` at commit
  `a0f451a5113cf9becb0847b92884cb10cbdec0ef` (GPL-3.0, maintenance mode)
  supplies the established ComfyUI IP-Adapter nodes and example graphs.

Download and SHA-256 verify these model files through the Hugging Face mirror
with resumable transfers:

| Destination | Source file | SHA-256 | Size |
| --- | --- | --- | --- |
| `models/controlnet/OpenPoseXL2.safetensors` | `thibaud/controlnet-openpose-sdxl-1.0/OpenPoseXL2.safetensors` | `5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92` | 4.66 GB |
| `models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` | `h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` | `3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581` | 0.79 GB |
| `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `h94/IP-Adapter/models/image_encoder/model.safetensors` | `6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030` | 2.35 GB |

The ControlNet model card is tagged `license:other`; deployment records that
source and does not alter its license terms. DWPose support assets are acquired
by the installed auxiliary node on first use with `HF_ENDPOINT` set to the same
mirror. This keeps their cache layout compatible with the node implementation.

## Workflow

1. Load the approved RGBA character reference, retaining its RGB image for
   generation conditioning.
2. Load one authored pose PNG or create it with DWPose/OpenPose from a pose
   reference image. One image represents one output frame.
3. Load SDXL Base, apply Pixel Art XL, encode the character/action prompt, and
   create a 512 square latent.
4. Apply OpenPoseXL2 through `ControlNetLoader` and `ControlNetApplyAdvanced`.
5. Apply the IP-Adapter Plus SDXL model with its ViT-H clip-vision encoder to
   preserve costume, weapon, palette, and silhouette.
6. Sample one image with a fixed seed, decode, remove the background through
   BiRefNet, invert the foreground mask before `JoinImageWithAlpha`, and scale
   to 64, 96, 128, or 256 pixels with `nearest-exact`.
7. Save numbered RGBA frames and compose them into a row-major sprite sheet.

The checked-in workflow artifact exposes the character reference, pose frame,
positive prompt, negative prompt, seed, ControlNet strength, IP-Adapter weight,
and output size as editable inputs. It does not expose the local API publicly
or replace the existing Wan workflow.

## Download And Validation

The installer records the source URL, expected filename or source commit,
byte count, SHA-256 where published, and installation timestamp in a local
deployment manifest. Model transfers use
`curl.exe --continue-at - --retry 10 --retry-all-errors`; a hash mismatch leaves
the partial file unpublished. Node archives use the official GitHub API and are
installed only after their top-level source layout is validated. The Hugging
Face mirror URL is preferred and the upstream Hugging Face URL is retained as
a documented fallback.

After dependency installation, restart ComfyUI once and validate the ControlNet,
IP-Adapter, clip-vision, and pose nodes through `/object_info`. Run a low-cost
two-pose 64-pixel smoke action, then an eight-pose 128-pixel sword attack. Each
run must return the requested number of non-empty RGBA PNGs, preserve nonzero
foreground alpha, keep the pose order, and create a correctly sized sheet.

## Scope Boundaries

This deployment does not download an unreviewed checkpoint, install FaceID or
InsightFace, add an external SaaS dependency, expose ComfyUI beyond loopback,
or attempt automatic motion interpolation. It does not modify the Wan API path;
that workflow remains available for concept-video generation.
