# ComfyUI Game Asset Workflows

This repository is a standalone toolkit for creating and running ComfyUI game-asset workflows. It contains the API package, workflow definitions, helper scripts, tests, and their design and deployment documentation.

## Test

Run the copied baseline test suite with the existing ComfyUI environment:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe -m pytest -q
```

## Deploy

Deploy the toolkit into a ComfyUI installation with:

```powershell
.\deploy.ps1 -ComfyRoot E:\ComfyUI
```

`deploy.ps1` will be implemented in a later task.

## Run the API

From the standalone repository, configure the separate ComfyUI installation and API port, then start the package with the ComfyUI Python environment:

```powershell
Set-Location E:\ComfyUI-GameAsset-Workflows
$env:COMFYUI_ROOT = 'E:\ComfyUI'
$env:GAME_ASSET_API_PORT = '8190'
E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
```

The standalone repository is the workflow and API source checkout. `COMFYUI_ROOT` must point to the ComfyUI runtime installation and must contain `main.py`.

`GAME_ASSET_API_HOST` defaults to `127.0.0.1`. Set it to `0.0.0.0` only for a trusted LAN; the API has no authentication and must not be exposed to untrusted networks. ComfyUI itself must remain available at `127.0.0.1:8188`.

### POST /v1/game-assets

Submit a game-asset job as JSON. A successful request returns `202 Accepted` with a `job_id`; poll `GET /v1/jobs/{job_id}` for completion and retrieve generated files from the returned `/assets/{job_id}/...` URLs.

Models and generated assets are excluded from this repository. Required models and other runtime assets are acquired from verified manifests.

## License

No license is granted. Public visibility of this repository does not grant permission to use, copy, modify, or redistribute its contents.
