# Wan2.2 5B Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the official Wan2.2-TI2V-5B native workflow and model weights into the existing ComfyUI installation.

**Architecture:** Use ComfyUI's native Wan2.2 nodes with the official split model files. Store the diffusion model, text encoder, and VAE in their standard model directories, and install the official workflow under the default user's workflows directory.

**Tech Stack:** ComfyUI native nodes, Wan2.2-TI2V-5B FP16, UMT5 XXL FP8, SafeTensors, PowerShell, curl

---

### Task 1: Install the official workflow

**Files:**
- Create: `user/default/workflows/video_wan2_2_5B_ti2v.json`

- [x] **Step 1: Create the workflow directory**

Run:

```powershell
New-Item -ItemType Directory -Force 'E:\ComfyUI\user\default\workflows'
```

Expected: `E:\ComfyUI\user\default\workflows` exists.

- [x] **Step 2: Download the official workflow**

Run:

```powershell
curl.exe --fail --location --retry 5 --output 'E:\ComfyUI\user\default\workflows\video_wan2_2_5B_ti2v.json' 'https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json'
```

Expected: the file size is exactly `14523` bytes and its SHA-256 is `50390798a133757075d750ab78b26274e3bf3e4a09f46aaa2f08d9559e62750b`.

### Task 2: Install the official model weights

**Files:**
- Create: `models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
- Create: `models/vae/wan2.2_vae.safetensors`
- Create: `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`

- [x] **Step 1: Download the diffusion model with resume support**

Run:

```powershell
curl.exe --fail --location --continue-at - --retry 20 --retry-all-errors --retry-delay 5 --connect-timeout 30 --output 'E:\ComfyUI\models\diffusion_models\wan2.2_ti2v_5B_fp16.safetensors.part' 'https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors'
```

Expected: the partial file reaches `9999658848` bytes.

- [x] **Step 2: Download the VAE with resume support**

Run:

```powershell
curl.exe --fail --location --continue-at - --retry 20 --retry-all-errors --retry-delay 5 --connect-timeout 30 --output 'E:\ComfyUI\models\vae\wan2.2_vae.safetensors.part' 'https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors'
```

Expected: the partial file reaches `1409400960` bytes.

- [x] **Step 3: Download the text encoder with resume support**

Run:

```powershell
curl.exe --fail --location --continue-at - --retry 20 --retry-all-errors --retry-delay 5 --connect-timeout 30 --output 'E:\ComfyUI\models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors.part' 'https://hf-mirror.com/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors'
```

Expected: the partial file reaches `6735906897` bytes.

- [x] **Step 4: Verify SHA-256 hashes and publish the files**

Run `Get-FileHash -Algorithm SHA256` on all three `.part` files. Expected hashes:

```text
wan2.2_ti2v_5B_fp16.safetensors.part     456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e
wan2.2_vae.safetensors.part              e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156
umt5_xxl_fp8_e4m3fn_scaled.safetensors.part c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
```

After all hashes match, rename each file by removing the `.part` suffix with `Move-Item -LiteralPath`.

### Task 3: Validate ComfyUI integration

**Files:**
- Inspect: `models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
- Inspect: `models/vae/wan2.2_vae.safetensors`
- Inspect: `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- Inspect: `user/default/workflows/video_wan2_2_5B_ti2v.json`

- [x] **Step 1: Validate the workflow JSON**

Run:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe -m json.tool 'E:\ComfyUI\user\default\workflows\video_wan2_2_5B_ti2v.json' NUL
```

Expected: exit code `0` with no JSON parsing error.

- [x] **Step 2: Restart ComfyUI**

Stop only the process currently owning `127.0.0.1:8188`, then start `E:\ComfyUI\.venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188` from `E:\ComfyUI` with output redirected to the existing startup logs.

Expected: `http://127.0.0.1:8188` returns HTTP `200`.

- [x] **Step 3: Validate model discovery through the ComfyUI API**

Query `http://127.0.0.1:8188/object_info` and confirm the three installed filenames are offered by the diffusion-model, CLIP, and VAE loader nodes.

Expected: all three filenames are discoverable and the server log contains no import traceback.
