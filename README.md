# ComfyUI Game Asset Workflows

This repository is the standalone source for ComfyUI game-asset workflows, the
local API, deployment and export scripts, pinned dependency manifests, tests,
and documentation. Keep the official ComfyUI checkout separate. For the
examples below, the two roots are:

```text
E:\ComfyUI-GameAsset-Workflows  standalone source repository
E:\ComfyUI                      ComfyUI runtime installation
```

Only workflow JSON, API and helper code, manifests, tests, and documentation
belong in this repository. Model weights, downloaded custom-node source,
ComfyUI source, `input`, `output`, caches, virtual environments, logs, partial
downloads, and secrets are runtime or local assets and must not be committed.
This repository has no `LICENSE` file and grants no license.

## Requirements

- Windows PowerShell 5.1 or later.
- Git and `curl.exe` available on `PATH`; curl 7.71 or later is required.
- Network access to GitHub and Hugging Face or `hf-mirror.com` for pinned node
  archives and model downloads.
- A separate ComfyUI root containing `main.py` and a Python 3.11 or later
  `.venv\Scripts\python.exe`. The examples use `E:\ComfyUI`.
- The project dependencies from `pyproject.toml` available to that Python
  environment, including the test dependencies when running pytest.
- A local ComfyUI HTTP server, normally `http://127.0.0.1:8188`, for discovery,
  deployment smoke tests, API jobs, and pose runs.
- Enough disk space and GPU resources for the manifest-listed models. Actual
  requirements depend on the selected model and workflow. Production animation
  requires at least 16 GB VRAM.
- Godot 4.x headless for export validation, selected with `GODOT_BIN` or the
  `--godot` option.

## Verify And Deploy

Run the full repository test suite from the standalone checkout:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### Model Coverage

The six workflows reference ten loader files. The seven entries in
`game_asset_api\model_manifest.py` are managed models: deployment downloads
missing files from their pinned mirror URLs and verifies byte size and SHA-256
before publishing them.

Production animation adds [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
at revision `d8d163cd90b1111f6227495e3467633676fbb346` and the
`guoyww/animatediff-motion-adapter-sdxl-beta` motion adapter. The adapter is
installed as `models/animatediff_models/mm_sdxl_v10_beta.safetensors`; its
primary source is `hf-mirror.com` and its explicit upstream fallback is
Hugging Face. Both sources use the same resumable partial file and the final
file is promoted only after the pinned SHA-256 verifies.

The following three Wan files are not in `MODEL_SPECS` and must already be
installed under the ComfyUI root:

| Relative path | Bytes | SHA-256 | Official source |
| --- | ---: | --- | --- |
| `models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | `9999658848` | `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) |
| `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `6735906897` | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| `models/vae/wan2.2_vae.safetensors` | `1409400960` | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) |

The [Wan2.2 deployment plan](docs/superpowers/plans/2026-07-15-wan2-2-5b-deployment.md)
records resumable mirror download and verification steps. Verify the listed
size and SHA-256 before placing each final file. Full deployment only confirms
that `/object_info` advertises these three filenames; it does not download or
hash-check them.

### Already-Provisioned Deployment

When all ten model files and pinned custom nodes are already installed, and
the running ComfyUI server has loaded those nodes, ensure
`E:\ComfyUI\input\example.png` exists and run the supported entry point:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& '.\deploy.ps1' -ComfyRoot 'E:\ComfyUI'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Deployment validates the ComfyUI root, publishes the six JSON files from
`workflows` to `E:\ComfyUI\user\default\workflows`, installs or verifies the
pinned custom nodes and seven managed models, checks every workflow node and
configured loader option against the live `/object_info` response, and runs a
two-frame, 64-pixel smoke action using `input\example.png`.

The published files are `pixel_character_design_api.json`,
`pixel_character_action_api.json`, `pose_controlled_pixel_action_api.json`,
`video_wan2_2_5B_ti2v.json`, `wan2_2_5b_dual_balanced.json`, and
`production_animation_api.json`.

### Fresh Or Newly Installed Nodes

A fresh installation is a two-stage operation because a running server cannot
discover nodes that it has not loaded. Stop ComfyUI normally first; this
repository does not provide or require a process-kill command. While ComfyUI is
stopped, install the three unmanaged Wan files listed above, then run the
installation stage from the standalone repository:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\deploy.py' `
  --comfy-root 'E:\ComfyUI' `
  --skip-discovery `
  --skip-smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

This stage publishes the workflows, installs the pinned custom nodes and their
requirements, and downloads or verifies the seven managed models. Its skipped
discovery and smoke stages mean it is not a complete deployment validation.

Start or restart ComfyUI normally, wait until `http://127.0.0.1:8188` is
healthy and startup has finished, then run the complete deployment:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& '.\deploy.ps1' -ComfyRoot 'E:\ComfyUI'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

`deploy.ps1` accepts only `-ComfyRoot` and `-BaseUrl`. The diagnostic
`--skip-nodes`, `--skip-models`, `--skip-discovery`, and `--skip-smoke`
switches belong only to the direct Python CLI and each bypasses part of full
validation. A fresh root is therefore not completely provisioned and verified
by one wrapper command.

## Run The API

Start the package from the standalone repository so Python imports
`game_asset_api` from this checkout. Do not copy or assume the package exists
inside ComfyUI:

```powershell
Set-Location E:\ComfyUI-GameAsset-Workflows
$env:COMFYUI_ROOT = 'E:\ComfyUI'
$env:GAME_ASSET_API_PORT = '8190'
$env:GAME_ASSET_API_HOST = '127.0.0.1'
& E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

`COMFYUI_ROOT` must contain `main.py`. `GAME_ASSET_API_HOST` defaults to
`127.0.0.1`, and the API submits jobs to ComfyUI at `127.0.0.1:8188`. A successful
`POST /v1/game-assets` returns `202 Accepted`; poll `GET /v1/jobs/{job_id}`.
The API has no authentication. Do not expose it to an untrusted network.

The POST contract requires only `character_prompt` and `action_prompt`; the job
generates its own character reference before the action stage, so no reference
upload or local reference path is required. Submit and poll a minimal job with:

```powershell
$body = @{
  character_prompt = 'pixel art knight in blue armor'
  action_prompt = 'walking in place'
} | ConvertTo-Json

$job = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8190/v1/game-assets' `
  -ContentType 'application/json' `
  -Body $body

$job
Invoke-RestMethod `
  -Method Get `
  -Uri ("http://127.0.0.1:8190/v1/jobs/{0}" -f $job.job_id)
```

## Run A Production Animation

`POST /v1/animations` accepts a saved character image and a weapon descriptor
below `E:\ComfyUI\input`. `character_image` and `weapon` are safe relative
paths; the first production release accepts only `sword_attack` and
`frame_count` values `8, 12, or 16`. A request without a seed receives a
durable server-assigned seed.

The weapon descriptor references a transparent PNG relative to the descriptor
and fixes its normalized grip and tip points. For
`E:\ComfyUI\input\weapons\sword.json`, use this descriptor with
`E:\ComfyUI\input\weapons\sword.png`:

```json
{
  "schema_version": 1,
  "image": "sword.png",
  "grip": [0.125, 0.5],
  "tip": [0.875, 0.5],
  "default_layer": "behind_character"
}
```

Submit an asynchronous animation job and poll its status:

```powershell
$body = @{
  asset_name = 'cultivator_attack'
  character_image = 'characters/cultivator.png'
  character_prompt = 'side-view cultivator in white and cyan robes'
  weapon = 'weapons/sword.json'
  action = 'sword_attack'
  frame_count = 8
  sprite_size = 128
  seed = 42
  godot_resource_prefix = 'res://game_assets/cultivator_attack'
} | ConvertTo-Json

$job = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8190/v1/animations' `
  -ContentType 'application/json' `
  -Body $body

do {
  Start-Sleep -Seconds 1
  $status = Invoke-RestMethod `
    -Method Get `
    -Uri ("http://127.0.0.1:8190/v1/jobs/{0}" -f $job.job_id)
} while ($status.status -in 'queued', 'validating_inputs', 'motion_planning', 'temporal_generation', 'character_stabilization', 'weapon_composite', 'godot_export', 'validating_outputs')

$status
```

Generation has no alternate-model or pose-workflow fallback. A failed job
returns its public error and the stage-specific `stage` value; correct the
reported input, discovery, generation, composition, export, or validation
failure before submitting a new job.

Use the CLI for the 2-frame preflight. It is deliberately a local-only
exception to the HTTP contract and validates the same transparent weapon input:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\run_production_animation.py' `
  --root 'E:\ComfyUI' `
  --character-image 'characters/cultivator.png' `
  --weapon 'weapons/sword.json' `
  --asset-name 'cultivator-preflight' `
  --character-prompt 'side-view cultivator in white and cyan robes' `
  --job-id 'cultivator-preflight-2f' `
  --frame-count 2 `
  --sprite-size 64 `
  --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Run production validation at eight frames, then repeat with `--frame-count 12`
and `--frame-count 16` before accepting a new action:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\run_production_animation.py' `
  --root 'E:\ComfyUI' `
  --character-image 'characters/cultivator.png' `
  --weapon 'weapons/sword.json' `
  --asset-name 'cultivator-attack' `
  --character-prompt 'side-view cultivator in white and cyan robes' `
  --job-id 'cultivator-attack-8f' `
  --frame-count 8 `
  --sprite-size 128 `
  --seed 42 `
  --base-url 'http://127.0.0.1:8188'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## Run A Pose-Controlled Action

The reference path must name a real image. These examples use the same local
file required by deployment, `E:\ComfyUI\input\example.png`.

A two-frame run is a low-cost smoke sequence containing the anticipation and
contact poses:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\run_pose_controlled_action.py' `
  --root 'E:\ComfyUI' `
  --reference 'E:\ComfyUI\input\example.png' `
  --job-id 'pose-smoke-2f' `
  --character-prompt 'pixel art knight in blue armor holding a sword' `
  --action-prompt 'forward sword slash' `
  --camera 'side' `
  --frame-count 2 `
  --sprite-size 64 `
  --seed 42 `
  --base-url 'http://127.0.0.1:8188'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

An eight-frame run renders the complete authored sequence: anticipation,
draw-back, wind-up, acceleration, contact, follow-through, overshoot, and
recovery.

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\run_pose_controlled_action.py' `
  --root 'E:\ComfyUI' `
  --reference 'E:\ComfyUI\input\example.png' `
  --job-id 'pose-attack-8f' `
  --character-prompt 'pixel art knight in blue armor holding a sword' `
  --action-prompt 'right hand swings the sword in a wide horizontal arc' `
  --camera 'side' `
  --frame-count 8 `
  --sprite-size 128 `
  --seed 20260717 `
  --base-url 'http://127.0.0.1:8188'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## Runtime Outputs

Job files are written below
`E:\ComfyUI\output\game_assets\<job-id>`. The pose runner places validated
frames and `spritesheet.png` in that job's `pose_action` directory. Deployed
workflows live in `E:\ComfyUI\user\default\workflows`.

Production animation publishes only after validation under
`E:\ComfyUI\output\game_assets\<job-id>\production_action`:

```text
production_action/
  frames/000.png ...
  spritesheet.png
  sprite_frames.tres
  animation.json
  preview.gif
```

Copy that directory into the target Godot project at the same requested
`res://game_assets/<asset-name>` resource prefix. `animation.json` records the
prefix and `sprite_frames.tres` contains the `sword_attack` animation. Validate
the exported bundle with Godot 4.x headless before committing it to the game:

```powershell
$env:GODOT_BIN = 'C:\Tools\Godot\Godot_v4.7.1-stable_win64_console.exe'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\validate_godot_export.py' `
  --bundle 'E:\ComfyUI\output\game_assets\cultivator-attack-8f\production_action' `
  --resource-prefix 'res://game_assets/cultivator-attack'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

All runtime models, custom nodes, copied inputs, generated outputs, caches, and
deployed workflow copies stay under the ComfyUI root and do not enter this
repository's Git index.

## Provenance And Licensing

`game_asset_api\model_manifest.py` pins each managed model's preferred source
URL, destination, byte size, and SHA-256. `game_asset_api\node_manifest.py` pins
each custom-node archive URL and source revision. Managed model downloads are
promoted only after size and hash verification, and custom-node installs record
their pinned revision.

Model weights and custom-node sources remain governed by their own upstream
licenses and model cards. This repository neither redistributes nor
relicenses them. Public visibility of this repository or an upstream artifact
does not by itself grant reuse rights; review every upstream license before
use or redistribution.

## Audit And Export

Regenerate the checked-in workflow artifacts, confirm the generated bytes are
unchanged, and audit the Git index:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\export_game_asset_workflows.py'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\export_pose_controlled_workflow.py'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --exit-code -- workflows
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\audit_repository.py'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

An expected export leaves `git diff --exit-code -- workflows` at exit code 0.
The audit examines Git-tracked paths in the index: permitted top-level shape,
secret-like filenames, blocked runtime/model suffixes, and index blob size. It
does not read file contents and cannot detect a token or credential stored in
an otherwise ordinary filename. Before public release, also run an independent
content-aware secret scanner and manually review the complete staged diff.

## Troubleshooting

- `invalid COMFYUI_ROOT`, missing `main.py`, or missing Python: pass the actual
  ComfyUI root and confirm both `E:\ComfyUI\main.py` and
  `E:\ComfyUI\.venv\Scripts\python.exe` exist.
- Server or `/object_info` errors: start ComfyUI at the `--base-url`, wait for
  startup to finish, and inspect its console. If nodes were just installed,
  restart ComfyUI and rerun deployment so discovery sees them.
- Smoke reference errors: create or select a real
  `E:\ComfyUI\input\example.png` before running deployment.
- Download, size, or hash failures: check network and mirror access.
  Manifest-managed model downloads use resumable `.part` files and node
  archives use `.zip.part`; invalid partials are never promoted over verified
  assets. Verify the three unmanaged Wan files separately against the table.
- Audit violations: inspect `git status --short` and `git ls-files`, then keep
  models, nodes, inputs, outputs, caches, credentials, and local archives
  outside the index.
- Reference or alpha problems: use a readable PNG with one clear, centered
  character and a clean, contrasting or transparent background. Inspect the
  resulting RGBA frames; the runner rejects frames without both transparent
  background and non-empty foreground alpha.

## Known OpenPose Weapon Limitation

OpenPose controls human joints, not sword or weapon geometry. In a fast attack,
independently sampled frames can drift in weapon shape, direction, length, or
hand continuity. IP-Adapter and the reference help stabilize character palette,
costume, and silhouette, but they are not a weapon rig and do not solve exact
action continuity.

Production work can add explicit hand and weapon keypoints, render the weapon
as a separate layered sprite, or apply manual pixel cleanup. These are
production techniques, not capabilities already implemented by this workflow.
