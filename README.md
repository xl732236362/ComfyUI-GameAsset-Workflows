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
- A separate ComfyUI root containing `main.py` and
  `.venv\Scripts\python.exe`. The examples use `E:\ComfyUI`.
- The project dependencies from `pyproject.toml` available to that Python
  environment, including the test dependencies when running pytest.
- A local ComfyUI HTTP server, normally `http://127.0.0.1:8188`, for discovery,
  deployment smoke tests, API jobs, and pose runs.
- Enough disk space and GPU resources for the manifest-listed models. Actual
  requirements depend on the selected model and workflow.

## Verify And Deploy

Run the full repository test suite from the standalone checkout:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
```

With ComfyUI running locally and `E:\ComfyUI\input\example.png` present, deploy
with the supported PowerShell entry point:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& '.\deploy.ps1' -ComfyRoot 'E:\ComfyUI'
```

Deployment validates the ComfyUI root, publishes the five JSON files from
`workflows` to `E:\ComfyUI\user\default\workflows`, installs or verifies the
pinned custom nodes and models, checks every workflow node and configured
loader option against the live `/object_info` response, and runs a two-frame,
64-pixel smoke action using `input\example.png`.

The published files are `pixel_character_design_api.json`,
`pixel_character_action_api.json`, `pose_controlled_pixel_action_api.json`,
`video_wan2_2_5B_ti2v.json`, and `wan2_2_5b_dual_balanced.json`.

Model downloads use the preferred mirror URLs fixed in the manifest. The first
deployment can take substantial time. `deploy.ps1` does not restart ComfyUI;
when it installs new nodes, restart ComfyUI so the server loads them, then run
the same deployment command again.

The skip switches belong to the Python deployment CLI, not `deploy.ps1`. They
are useful for isolating an installation, discovery, or smoke-test problem. For
example, this publishes workflows and runs discovery without reinstalling
dependencies or running the smoke action:

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\deploy.py' `
  --comfy-root 'E:\ComfyUI' `
  --base-url 'http://127.0.0.1:8188' `
  --skip-nodes `
  --skip-models `
  --skip-smoke
```

The Python CLI also supports `--skip-discovery`; use skip switches only to
diagnose a known stage, not as evidence of a complete deployment.

## Run The API

Start the package from the standalone repository so Python imports
`game_asset_api` from this checkout. Do not copy or assume the package exists
inside ComfyUI:

```powershell
Set-Location E:\ComfyUI-GameAsset-Workflows
$env:COMFYUI_ROOT = 'E:\ComfyUI'
$env:GAME_ASSET_API_PORT = '8190'
$env:GAME_ASSET_API_HOST = '127.0.0.1'
E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
```

`COMFYUI_ROOT` must contain `main.py`. `GAME_ASSET_API_HOST` defaults to
`127.0.0.1`, and the API submits jobs to ComfyUI at `127.0.0.1:8188`. A successful
`POST /v1/game-assets` returns `202 Accepted`; poll `GET /v1/jobs/{job_id}`.
The API has no authentication. Do not expose it to an untrusted network.

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
```

## Runtime Outputs

Job files are written below
`E:\ComfyUI\output\game_assets\<job-id>`. The pose runner places validated
frames and `spritesheet.png` in that job's `pose_action` directory. Deployed
workflows live in `E:\ComfyUI\user\default\workflows`.

All runtime models, custom nodes, copied inputs, generated outputs, caches, and
deployed workflow copies stay under the ComfyUI root and do not enter this
repository's Git index.

## Provenance And Licensing

`game_asset_api\model_manifest.py` pins each model's preferred source URL,
destination, byte size, and SHA-256. `game_asset_api\node_manifest.py` pins each
custom-node archive URL and source revision. Model downloads are promoted only
after size and hash verification, and custom-node installs record their pinned
revision.

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
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\export_pose_controlled_workflow.py'
git diff -- workflows
& 'E:\ComfyUI\.venv\Scripts\python.exe' '.\scripts\audit_repository.py'
```

An expected export leaves `git diff -- workflows` empty. The audit exits
nonzero if the Git index contains a forbidden runtime, secret, or oversized
artifact.

## Troubleshooting

- `invalid COMFYUI_ROOT`, missing `main.py`, or missing Python: pass the actual
  ComfyUI root and confirm both `E:\ComfyUI\main.py` and
  `E:\ComfyUI\.venv\Scripts\python.exe` exist.
- Server or `/object_info` errors: start ComfyUI at the `--base-url`, wait for
  startup to finish, and inspect its console. If nodes were just installed,
  restart ComfyUI and rerun deployment so discovery sees them.
- Smoke reference errors: create or select a real
  `E:\ComfyUI\input\example.png` before running deployment.
- Download, size, or hash failures: check network and mirror access. Model
  downloads use resumable `.part` files and node archives use `.zip.part`;
  invalid partials are never promoted over verified assets.
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
