# Wan2.2 5B Dual-Mode Workflow Design

## Goal

Create one native ComfyUI workflow for Wan2.2-TI2V-5B that supports both text-to-video and image-to-video on an NVIDIA GeForce RTX 4070 Ti SUPER with 16 GB VRAM. The workflow must favor a balanced quality/speed profile, require no custom nodes, save MP4/H.264 output, and pass an end-to-end smoke generation.

## Scope

The deliverable is `E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json`.

The workflow will:

- Use the three already-installed official model files.
- Share one model, conditioning, sampling, decoding, and output chain between T2V and I2V.
- Default to T2V while allowing I2V by enabling the image input node.
- Use balanced production defaults of 832 x 480, 81 frames, 20 steps, and 24 fps.
- Save to `E:\ComfyUI\output\video` with the `Wan2.2_5B` filename prefix.

The workflow will not add custom nodes, LoRAs, upscaling, interpolation, audio, prompt enhancement, or multiple quality preset branches.

## Approaches Considered

### Chosen: Optimized official native graph

Adapt the official Wan2.2 5B TI2V workflow and keep one shared generation chain. The `LoadImage` node is bypassed by default for T2V and enabled for I2V. This has the smallest graph, the lowest compatibility risk, and no duplicated model memory.

### Rejected: Two branches on one canvas

Separate T2V and I2V input branches make mode differences more visible, but increase graph size and create more manual bypass state to maintain.

### Rejected: Two independent workflow files

Separate files are simple in isolation but do not meet the single unified workflow requirement.

## Architecture

The graph is arranged from left to right in five groups:

1. **Model loading**
   - `UNETLoader`: `wan2.2_ti2v_5B_fp16.safetensors`, weight dtype `default`.
   - `CLIPLoader`: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, type `wan`, device `default`.
   - `VAELoader`: `wan2.2_vae.safetensors`.

2. **Mode input**
   - `LoadImage` connects to the optional `start_image` input of `Wan22ImageToVideoLatent`.
   - The node is bypassed by default, producing T2V behavior.
   - Enabling the node and selecting an image produces I2V behavior.

3. **Prompt conditioning**
   - Separate positive and negative `CLIPTextEncode` nodes share the Wan CLIP loader.
   - The positive prompt starts with a practical cinematic example that the user can replace.
   - The negative prompt retains the official Wan2.2 quality and anatomy exclusions.

4. **Generation**
   - `Wan22ImageToVideoLatent`: width 832, height 480, length 81, batch size 1.
   - `ModelSamplingSD3`: shift 8.
   - `KSampler`: random seed, 20 steps, CFG 5, `uni_pc`, `simple`, denoise 1.0.

5. **Decode and output**
   - `VAEDecode` converts the sampled latent video to frames.
   - `CreateVideo` uses 24 fps.
   - `SaveVideo` uses prefix `video/Wan2.2_5B`, format `mp4`, and codec `h264`.

## Data Flow

The diffusion model passes through `ModelSamplingSD3` into `KSampler`. The text encoder feeds positive and negative conditioning into the same sampler. The VAE and optional start image feed `Wan22ImageToVideoLatent`, whose latent output is sampled, decoded, assembled at 24 fps, and saved as MP4/H.264.

No image is required in T2V mode. In I2V mode, the enabled `LoadImage` output supplies the initial visual condition while all downstream nodes remain unchanged.

## Constraints And Error Handling

- Width and height remain multiples of 16.
- Frame count remains in the `4n+1` form required by the Wan video latent path; 81 is valid.
- Model loader widgets reference exact installed filenames so a missing model fails before sampling.
- An enabled image node without a valid selected image must fail validation before GPU sampling.
- Every workflow node type is provided by ComfyUI core; the workflow does not depend on frontend-only note nodes or custom nodes.
- Output format and codec are explicit rather than relying on automatic selection.
- The existing official workflow is preserved; the new balanced workflow is a separate file.

## Verification

### Structural verification

- Parse the saved workflow as JSON.
- Confirm all backend node types exist in `http://127.0.0.1:8188/object_info`.
- Confirm model loader option lists contain the selected diffusion model, text encoder, and VAE.
- Confirm the workflow appears in the ComfyUI user-data workflow listing.
- Confirm the saved production widgets are 832 x 480, 81 frames, 20 steps, 24 fps, MP4, and H.264.

### End-to-end smoke generation

Submit a temporary T2V API prompt using the same model and node chain with 512 x 288, 17 frames, and 4 sampling steps. The temporary prompt must not alter the production workflow defaults.

The smoke test succeeds when:

- ComfyUI finishes the queued prompt without an execution error.
- The diffusion model, UMT5 encoder, and VAE all load during execution.
- Sampling and VAE decoding complete.
- A non-empty MP4 file is written under `E:\ComfyUI\output\video`.
- The ComfyUI service remains available at `http://127.0.0.1:8188` after execution.

## Acceptance Criteria

- One unified workflow supports T2V by default and I2V by enabling the image node.
- It uses no custom nodes and references only the installed official Wan2.2 5B assets.
- Production defaults match the approved balanced profile.
- Output is MP4/H.264 at 24 fps.
- Structural checks and the end-to-end smoke generation pass.
