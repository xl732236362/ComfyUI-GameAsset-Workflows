# Pixel Game Asset API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local or LAN API that generates a pixel-art character design and transparent action sprite frames through ComfyUI.

**Architecture:** An `aiohttp` service validates requests, queues one job at a time, builds two ComfyUI API prompt graphs, and polls ComfyUI on loopback. Character generation uses SDXL Base 1.0 with Pixel Art XL LoRA; animation uses the installed Wan 2.2 TI2V workflow stack, then BiRefNet masks and Pillow build exact-frame RGBA sprites and metadata.

**Tech Stack:** Python 3.13, aiohttp, Pillow, pytest, ComfyUI HTTP API, SDXL Base 1.0, Pixel Art XL LoRA, BiRefNet, Wan 2.2 TI2V 5B.

---

## File Structure

- Create: `game_asset_api/contracts.py` - public request parsing and normalized immutable request model.
- Create: `game_asset_api/prompting.py` - camera vocabulary and deterministic positive/negative prompts.
- Create: `game_asset_api/workflows.py` - executable ComfyUI API graph builders and workflow JSON writer.
- Create: `game_asset_api/comfy_client.py` - typed `/prompt` and `/history` client with timeout/error handling.
- Create: `game_asset_api/postprocess.py` - Wan frame count conversion, source-frame selection, RGBA sheet composition, safe paths.
- Create: `game_asset_api/jobs.py` - durable job manifest and serialized two-stage orchestration.
- Create: `game_asset_api/app.py` - HTTP routes and application lifecycle.
- Create: `game_asset_api/__main__.py` - command-line server entry point.
- Create: `game_asset_api/model_manifest.py` - pinned model sources, bytes, SHA-256 values, and resumable install routine.
- Create: `scripts/install_game_asset_models.py` - model installer CLI.
- Create: `scripts/export_game_asset_workflows.py` - writes the two API graph artifacts to `user/default/workflows`.
- Create: `tests-unit/game_asset_api_test/test_contracts.py`
- Create: `tests-unit/game_asset_api_test/test_workflows.py`
- Create: `tests-unit/game_asset_api_test/test_postprocess.py`
- Create: `tests-unit/game_asset_api_test/test_model_manifest.py`
- Create: `tests-unit/game_asset_api_test/test_jobs.py`
- Create: `tests-unit/game_asset_api_test/test_app.py`
- Create: `user/default/workflows/pixel_character_design_api.json`
- Create: `user/default/workflows/pixel_character_action_api.json`
- Modify: `.gitignore` - ignore runtime job output and visual-companion session files.
- Modify: `README.md` - add setup, LAN binding, request, poll, and asset retrieval instructions.

### Task 1: Model Manifest And Installer

**Files:**
- Create: `game_asset_api/model_manifest.py`
- Create: `scripts/install_game_asset_models.py`
- Test: `tests-unit/game_asset_api_test/test_model_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

```python
from game_asset_api.model_manifest import MODEL_SPECS, verify_file


def test_manifest_pins_the_three_new_local_assets():
    names = {spec.filename: spec for spec in MODEL_SPECS}
    assert names["sd_xl_base_1.0.safetensors"].size == 6_938_078_334
    assert names["pixel-art-xl.safetensors"].sha256 == "4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7"
    assert names["BiRefNet-general-epoch_244.safetensors"].size == 444_473_596


def test_verify_file_rejects_a_hash_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"wrong")
    assert not verify_file(path, size=5, sha256="0" * 64)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_model_manifest.py -q`

Expected: `ModuleNotFoundError: No module named 'game_asset_api'`.

- [ ] **Step 3: Implement the pinned manifest and streaming verifier**

```python
# game_asset_api/model_manifest.py
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    relative_dir: str
    url: str
    size: int
    sha256: str

    def destination(self, root: Path) -> Path:
        return root / "models" / self.relative_dir / self.filename


MODEL_SPECS = (
    ModelSpec("sd_xl_base_1.0.safetensors", "checkpoints", "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", 6_938_078_334, "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"),
    ModelSpec("pixel-art-xl.safetensors", "loras", "https://hf-mirror.com/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors", 170_543_052, "4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7"),
    ModelSpec("BiRefNet-general-epoch_244.safetensors", "background_removal", "https://hf-mirror.com/ZhengPeng7/BiRefNet/resolve/main/model.safetensors", 444_473_596, "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"),
)


def verify_file(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def install(spec: ModelSpec, root: Path) -> Path:
    destination = spec.destination(root)
    if verify_file(destination, spec.size, spec.sha256):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(["curl.exe", "--fail", "--location", "--continue-at", "-", "--retry", "10", "--retry-all-errors", "--output", str(partial), spec.url], check=True)
    if not verify_file(partial, spec.size, spec.sha256):
        raise RuntimeError(f"verification failed for {spec.filename}")
    partial.replace(destination)
    return destination
```

Implement `scripts/install_game_asset_models.py` with `argparse`, resolve the project root as `Path(__file__).resolve().parents[1]`, and call `install()` for every item in `MODEL_SPECS`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_model_manifest.py -q`

Expected: two passing tests.

- [ ] **Step 5: Commit the installer**

```powershell
rtk git add game_asset_api/model_manifest.py scripts/install_game_asset_models.py tests-unit/game_asset_api_test/test_model_manifest.py
rtk git commit -m "feat: add verified game asset model installer"
```

### Task 2: Request Contract And Prompt Expansion

**Files:**
- Create: `game_asset_api/contracts.py`
- Create: `game_asset_api/prompting.py`
- Test: `tests-unit/game_asset_api_test/test_contracts.py`

- [ ] **Step 1: Write failing request and prompt tests**

```python
import pytest
from game_asset_api.contracts import RequestError, parse_asset_request
from game_asset_api.prompting import build_character_prompt


def test_custom_camera_requires_camera_prompt():
    with pytest.raises(RequestError, match="camera_prompt is required"):
        parse_asset_request({"character_prompt": "knight", "action_prompt": "idle", "camera": "custom"})


def test_normalized_request_has_pixel_defaults():
    request = parse_asset_request({"character_prompt": "knight", "action_prompt": "run"})
    assert (request.frame_count, request.sprite_size, request.camera) == (8, 128, None)


def test_character_prompt_carries_camera_and_pixel_constraints():
    request = parse_asset_request({"character_prompt": "knight", "action_prompt": "run", "camera": "side"})
    prompt = build_character_prompt(request)
    assert "knight" in prompt and "side view" in prompt and "pixel art" in prompt
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_contracts.py -q`

Expected: import failure for `game_asset_api.contracts`.

- [ ] **Step 3: Implement the contract and deterministic prompt helpers**

```python
# game_asset_api/contracts.py
from dataclasses import dataclass
from typing import Any

CAMERAS = {"side", "front", "top_down", "three_quarter", "custom"}
SPRITE_SIZES = {64, 96, 128, 256}


class RequestError(ValueError):
    pass


@dataclass(frozen=True)
class AssetRequest:
    character_prompt: str
    action_prompt: str
    frame_count: int = 8
    camera: str | None = None
    camera_prompt: str | None = None
    seed: int | None = None
    sprite_size: int = 128


def parse_asset_request(data: dict[str, Any]) -> AssetRequest:
    character = str(data.get("character_prompt", "")).strip()
    action = str(data.get("action_prompt", "")).strip()
    camera = data.get("camera")
    camera_prompt = str(data.get("camera_prompt", "")).strip() or None
    frame_count = int(data.get("frame_count", 8))
    sprite_size = int(data.get("sprite_size", 128))
    if not character or not action:
        raise RequestError("character_prompt and action_prompt are required")
    if camera is not None and camera not in CAMERAS:
        raise RequestError("camera is invalid")
    if camera == "custom" and not camera_prompt:
        raise RequestError("camera_prompt is required when camera is custom")
    if not 2 <= frame_count <= 16:
        raise RequestError("frame_count must be between 2 and 16")
    if sprite_size not in SPRITE_SIZES:
        raise RequestError("sprite_size must be one of 64, 96, 128, 256")
    seed = data.get("seed")
    return AssetRequest(character, action, frame_count, camera, camera_prompt, None if seed is None else int(seed), sprite_size)
```

```python
# game_asset_api/prompting.py
from .contracts import AssetRequest

CAMERA_TEXT = {"side": "fixed side view", "front": "fixed front view", "top_down": "fixed top-down view", "three_quarter": "fixed three-quarter view"}
PIXEL_STYLE = "pixel art game sprite, limited palette, crisp hard edges, isolated full body character"
NEGATIVE = "photorealistic, blurry, anti-aliased, text, watermark, UI, multiple characters, background scenery"


def camera_text(request: AssetRequest) -> str:
    return request.camera_prompt or CAMERA_TEXT.get(request.camera, "camera view inferred from the character prompt")


def build_character_prompt(request: AssetRequest) -> str:
    return ", ".join((request.character_prompt, camera_text(request), PIXEL_STYLE, "plain studio background"))


def build_action_prompt(request: AssetRequest) -> str:
    return ", ".join((request.action_prompt, camera_text(request), PIXEL_STYLE, "locked camera, consistent character identity"))
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_contracts.py -q`

Expected: three passing tests.

- [ ] **Step 5: Commit the contract module**

```powershell
rtk git add game_asset_api/contracts.py game_asset_api/prompting.py tests-unit/game_asset_api_test/test_contracts.py
rtk git commit -m "feat: validate game asset generation requests"
```

### Task 3: ComfyUI Graph Builders And Exported Workflow Artifacts

**Files:**
- Create: `game_asset_api/workflows.py`
- Create: `scripts/export_game_asset_workflows.py`
- Create: `user/default/workflows/pixel_character_design_api.json`
- Create: `user/default/workflows/pixel_character_action_api.json`
- Test: `tests-unit/game_asset_api_test/test_workflows.py`

- [ ] **Step 1: Write failing graph tests**

```python
from game_asset_api.contracts import parse_asset_request
from game_asset_api.workflows import build_action_workflow, build_character_workflow


def test_character_graph_applies_pixel_lora_and_saves_rgb_and_rgba():
    graph = build_character_workflow(parse_asset_request({"character_prompt": "knight", "action_prompt": "idle"}), "job-1")
    assert graph["2"]["class_type"] == "LoraLoader"
    assert graph["2"]["inputs"]["lora_name"] == "pixel-art-xl.safetensors"
    assert {node["class_type"] for node in graph.values()} >= {"RemoveBackground", "JoinImageWithAlpha", "SaveImage"}


def test_action_graph_uses_image_conditioning_and_wan_compatible_length():
    graph = build_action_workflow(parse_asset_request({"character_prompt": "knight", "action_prompt": "run", "frame_count": 8}), "job-1", "game_assets/job-1/reference.png")
    assert graph["10"]["class_type"] == "Wan22ImageToVideoLatent"
    assert graph["10"]["inputs"]["length"] == 9
    assert graph["10"]["inputs"]["start_image"] == ["9", 0]
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_workflows.py -q`

Expected: import failure for `game_asset_api.workflows`.

- [ ] **Step 3: Implement both API graph builders**

`build_character_workflow()` must return a graph with these numbered nodes and links: `CheckpointLoaderSimple` (1), `LoraLoader` (2), positive/negative `CLIPTextEncode` (3/4), `EmptyLatentImage` (5), `KSampler` (6), `VAEDecode` (7), `LoadBackgroundRemovalModel` (8), `RemoveBackground` (9), `JoinImageWithAlpha` (10), RGB `SaveImage` (11), and RGBA `SaveImage` (12). Configure node 1 with `sd_xl_base_1.0.safetensors`, node 2 strengths at `1.0`, KSampler at 30 steps / CFG 7 / `dpmpp_2m` / `karras`, and prefixes `game_assets/{job_id}/reference_rgb` and `game_assets/{job_id}/character`.

`build_action_workflow()` must create `UNETLoader`, `CLIPLoader`, `VAELoader`, `ModelSamplingSD3`, positive/negative encoders, `LoadBackgroundRemovalModel`, `LoadImage`, `Wan22ImageToVideoLatent`, `KSampler`, `VAEDecode`, `RemoveBackground`, `JoinImageWithAlpha`, `ImageScale`, and `SaveImage`. Use the installed Wan filenames, shift 8, `uni_pc` / `simple`, CFG 5, 20 steps, a 512 square source, `nearest-exact`, and prefix `game_assets/{job_id}/wan_frames`.

Use this link form in every graph:

```python
graph["10"] = {
    "class_type": "Wan22ImageToVideoLatent",
    "inputs": {
        "vae": ["3", 0], "width": 512, "height": 512,
        "length": wan_source_frame_count(request.frame_count), "batch_size": 1,
        "start_image": ["9", 0],
    },
}
```

`scripts/export_game_asset_workflows.py` must build each graph from a fixed representative request and write `{"prompt": graph}` as indented UTF-8 JSON to the two workflow paths.

- [ ] **Step 4: Run the focused tests and structural workflow checks**

Run:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_workflows.py -q
E:\ComfyUI\.venv\Scripts\python.exe scripts/export_game_asset_workflows.py
E:\ComfyUI\.venv\Scripts\python.exe -c "import json, urllib.request; info=json.load(urllib.request.urlopen('http://127.0.0.1:8188/object_info')); graph=json.load(open('user/default/workflows/pixel_character_action_api.json', encoding='utf-8'))['prompt']; missing={n['class_type'] for n in graph.values()}-set(info); assert not missing, missing"
```

Expected: workflow tests pass, both JSON artifacts are written, and node availability check exits with code 0.

- [ ] **Step 5: Commit workflow builders and artifacts**

```powershell
rtk git add game_asset_api/workflows.py scripts/export_game_asset_workflows.py user/default/workflows/pixel_character_design_api.json user/default/workflows/pixel_character_action_api.json tests-unit/game_asset_api_test/test_workflows.py
rtk git commit -m "feat: add pixel character ComfyUI API workflows"
```

### Task 4: Deterministic Frame Postprocessing

**Files:**
- Create: `game_asset_api/postprocess.py`
- Test: `tests-unit/game_asset_api_test/test_postprocess.py`

- [ ] **Step 1: Write failing frame and image tests**

```python
from PIL import Image
from game_asset_api.postprocess import frame_indices, wan_source_frame_count, write_sprite_sheet


def test_wan_frame_conversion_and_selection_are_exact():
    assert wan_source_frame_count(8) == 9
    assert frame_indices(9, 8) == [0, 1, 2, 3, 5, 6, 7, 8]


def test_sprite_sheet_preserves_rgba_and_row_major_order(tmp_path):
    frames = [Image.new("RGBA", (2, 2), color) for color in ((255, 0, 0, 0), (0, 255, 0, 128), (0, 0, 255, 255))]
    path, columns, rows = write_sprite_sheet(frames, tmp_path / "sheet.png")
    image = Image.open(path)
    assert (image.mode, image.size, columns, rows) == ("RGBA", (4, 4), 2, 2)
    assert image.getpixel((2, 0)) == (0, 255, 0, 128)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_postprocess.py -q`

Expected: import failure for `game_asset_api.postprocess`.

- [ ] **Step 3: Implement strict frame mapping and RGBA export**

```python
# game_asset_api/postprocess.py
from math import ceil, sqrt
from pathlib import Path
from PIL import Image


def wan_source_frame_count(frame_count: int) -> int:
    return 4 * ceil((frame_count - 1) / 4) + 1


def frame_indices(source_count: int, target_count: int) -> list[int]:
    return [round(index * (source_count - 1) / (target_count - 1)) for index in range(target_count)]


def write_sprite_sheet(frames: list[Image.Image], path: Path) -> tuple[Path, int, int]:
    columns = ceil(sqrt(len(frames)))
    rows = ceil(len(frames) / columns)
    width, height = frames[0].size
    sheet = Image.new("RGBA", (columns * width, rows * height))
    for index, frame in enumerate(frames):
        sheet.alpha_composite(frame.convert("RGBA"), ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path, columns, rows
```

Also add `copy_selected_frames(source_paths, selected_indices, destination)` that names frames with `000.png` through `NNN.png`, converts each image to RGBA, and raises `ValueError` when ComfyUI returned fewer source frames than expected.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_postprocess.py -q`

Expected: two passing tests.

- [ ] **Step 5: Commit frame postprocessing**

```powershell
rtk git add game_asset_api/postprocess.py tests-unit/game_asset_api_test/test_postprocess.py
rtk git commit -m "feat: compose transparent sprite frame outputs"
```

### Task 5: ComfyUI Client And Durable Job Orchestrator

**Files:**
- Create: `game_asset_api/comfy_client.py`
- Create: `game_asset_api/jobs.py`
- Test: `tests-unit/game_asset_api_test/test_jobs.py`

- [ ] **Step 1: Write failing job transition tests**

```python
import pytest
from game_asset_api.contracts import parse_asset_request
from game_asset_api.jobs import JobStore, JobStatus


def test_job_store_writes_ordered_durable_statuses(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(parse_asset_request({"character_prompt": "knight", "action_prompt": "run"}))
    store.transition(job.job_id, JobStatus.GENERATING_CHARACTER)
    stored = store.read(job.job_id)
    assert stored.status is JobStatus.GENERATING_CHARACTER
    assert (tmp_path / job.job_id / "job.json").is_file()


def test_terminal_job_cannot_transition_again(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(parse_asset_request({"character_prompt": "knight", "action_prompt": "run"}))
    store.transition(job.job_id, JobStatus.FAILED, error="ComfyUI timed out")
    with pytest.raises(ValueError, match="terminal"):
        store.transition(job.job_id, JobStatus.POSTPROCESSING)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_jobs.py -q`

Expected: import failure for `game_asset_api.jobs`.

- [ ] **Step 3: Implement the client and serialized pipeline**

Define `ComfyClient(base_url, session)` with `async submit(graph) -> str`, `async wait_for_prompt(prompt_id, timeout_seconds=1800) -> dict`, and `extract_images(history) -> list[Path]`. `submit` posts `{"prompt": graph}` to `/prompt`; `wait_for_prompt` polls `/history/<prompt_id>` every two seconds, raises `RuntimeError` when a status message starts with `execution_error`, and raises `TimeoutError` at deadline.

Implement `JobStatus` as a string `Enum` with `QUEUED`, `GENERATING_CHARACTER`, `GENERATING_ACTION`, `POSTPROCESSING`, `COMPLETED`, and `FAILED`. `JobStore` writes JSON atomically through a `.tmp` file and `Path.replace()`. Permit only this transition order:

```python
ALLOWED = {
    JobStatus.QUEUED: {JobStatus.GENERATING_CHARACTER, JobStatus.FAILED},
    JobStatus.GENERATING_CHARACTER: {JobStatus.GENERATING_ACTION, JobStatus.FAILED},
    JobStatus.GENERATING_ACTION: {JobStatus.POSTPROCESSING, JobStatus.FAILED},
    JobStatus.POSTPROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}
```

`JobRunner` must own one `asyncio.Queue` and one worker task. Its worker submits the character graph, copies the RGB output into `input/game_assets/{job_id}/reference.png`, submits the action graph, selects output frames, writes the sheet and metadata, and transitions the manifest at each named stage. Catch `RuntimeError`, `TimeoutError`, and `ValueError`; write only their message to the job manifest and mark it `FAILED`.

- [ ] **Step 4: Run job tests to verify the implementation passes**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_jobs.py -q`

Expected: two passing tests.

- [ ] **Step 5: Commit the job pipeline**

```powershell
rtk git add game_asset_api/comfy_client.py game_asset_api/jobs.py tests-unit/game_asset_api_test/test_jobs.py
rtk git commit -m "feat: orchestrate serialized ComfyUI asset jobs"
```

### Task 6: Local And LAN HTTP API

**Files:**
- Create: `game_asset_api/app.py`
- Create: `game_asset_api/__main__.py`
- Test: `tests-unit/game_asset_api_test/test_app.py`

- [ ] **Step 1: Write failing HTTP tests**

```python
async def test_post_creates_a_queued_job(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.post("/v1/game-assets", json={"character_prompt": "knight", "action_prompt": "run"})
    body = await response.json()
    assert response.status == 202
    assert body["status"] == "queued"


async def test_asset_route_rejects_path_traversal(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get("/assets/not-a-uuid/../../main.py")
    assert response.status == 404
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_app.py -q`

Expected: import failure for `game_asset_api.app`.

- [ ] **Step 3: Implement the application factory and entry point**

```python
# game_asset_api/app.py
from aiohttp import web
from .contracts import RequestError, parse_asset_request


async def create_job(request: web.Request) -> web.Response:
    try:
        payload = parse_asset_request(await request.json())
    except (RequestError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)
    job = request.app["runner"].enqueue(payload)
    return web.json_response({"job_id": job.job_id, "status": job.status.value}, status=202)


def create_app(runner, asset_root) -> web.Application:
    app = web.Application()
    app["runner"] = runner
    app["asset_root"] = asset_root
    app.router.add_post("/v1/game-assets", create_job)
    app.router.add_get("/v1/jobs/{job_id}", get_job)
    app.router.add_get("/assets/{job_id}/{path:.*}", get_asset)
    return app
```

Implement `get_job` with UUID parsing and `404` for unknown jobs. Implement `get_asset` by resolving `(asset_root / job_id / path)` and returning `404` unless the resolved file is inside `(asset_root / job_id).resolve()` and is a regular file. `__main__.py` must read `GAME_ASSET_API_HOST` (default `127.0.0.1`) and `GAME_ASSET_API_PORT` (default `8190`), start the runner, then call `web.run_app()`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_app.py -q`

Expected: two passing tests.

- [ ] **Step 5: Commit the HTTP API**

```powershell
rtk git add game_asset_api/app.py game_asset_api/__main__.py tests-unit/game_asset_api_test/test_app.py
rtk git commit -m "feat: expose local game asset API"
```

### Task 7: Runtime Documentation, Ignore Rules, And Full Test Suite

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Test: all `tests-unit/game_asset_api_test` files

- [ ] **Step 1: Write a failing documentation-check test**

```python
from pathlib import Path


def test_readme_documents_the_game_asset_api():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "POST /v1/game-assets" in readme
    assert "GAME_ASSET_API_HOST" in readme
```

- [ ] **Step 2: Run it to verify it fails**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test/test_app.py::test_readme_documents_the_game_asset_api -q`

Expected: assertion failure because the API section is absent.

- [ ] **Step 3: Document execution and ignore runtime state**

Append these ignore rules:

```gitignore
/.superpowers/
/output/game_assets/
/input/game_assets/
```

Add a `## Pixel Game Asset API` README section with these exact commands:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe scripts\install_game_asset_models.py
E:\ComfyUI\.venv\Scripts\python.exe scripts\export_game_asset_workflows.py
E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
```

Document the `POST /v1/game-assets` payload from the design, `GET /v1/jobs/{job_id}`, the `/assets/...` result URLs, and `GAME_ASSET_API_HOST=0.0.0.0` for a trusted LAN. State explicitly that ComfyUI must remain on `127.0.0.1:8188` and that the first release has no authentication.

- [ ] **Step 4: Run all focused tests**

Run: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit/game_asset_api_test -q`

Expected: all unit tests pass.

- [ ] **Step 5: Commit documentation and runtime ignores**

```powershell
rtk git add .gitignore README.md tests-unit/game_asset_api_test/test_app.py
rtk git commit -m "docs: document pixel game asset API"
```

### Task 8: Live ComfyUI Verification

**Files:**
- Inspect: `user/default/workflows/pixel_character_design_api.json`
- Inspect: `user/default/workflows/pixel_character_action_api.json`
- Inspect: `output/game_assets/{job_id}/`

- [ ] **Step 1: Install and prove model discovery**

Run:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe scripts\install_game_asset_models.py
$info = Invoke-RestMethod 'http://127.0.0.1:8188/object_info'
@('sd_xl_base_1.0.safetensors','pixel-art-xl.safetensors','BiRefNet-general-epoch_244.safetensors') | ForEach-Object { if (($info | ConvertTo-Json -Depth 8) -notmatch [regex]::Escape($_)) { throw "model not discovered: $_" } }
```

Expected: installer verifies all three hashes and all model names are discoverable after restarting ComfyUI once.

- [ ] **Step 2: Start the API and submit the low-cost smoke request**

Run in one PowerShell session:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
```

Run in another:

```powershell
$body = @{ character_prompt = 'blue armored knight'; action_prompt = 'take one step right'; frame_count = 2; camera = 'side'; sprite_size = 64; seed = 42 } | ConvertTo-Json
$created = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8190/v1/game-assets' -ContentType 'application/json' -Body $body
$created.job_id | Set-Content "$env:TEMP\game-asset-job-id.txt"
$created
```

Expected: `202` response with a UUID and `queued` status.

- [ ] **Step 3: Poll and verify final artifacts**

Run:

```powershell
$job = (Get-Content "$env:TEMP\game-asset-job-id.txt").Trim()
do { $state = Invoke-RestMethod "http://127.0.0.1:8190/v1/jobs/$job"; Start-Sleep -Seconds 2 } while ($state.status -in 'queued','generating_character','generating_action','postprocessing')
if ($state.status -ne 'completed') { throw ($state | ConvertTo-Json -Depth 8) }
$png = Invoke-WebRequest "http://127.0.0.1:8190$($state.frames[0])" -OutFile "$env:TEMP\game-asset-frame.png" -PassThru
if ($png.StatusCode -ne 200 -or (Get-Item "$env:TEMP\game-asset-frame.png").Length -le 0) { throw 'frame not served' }
```

Expected: `completed`, two frame URLs, non-empty character/sheet/metadata URLs, and a non-empty downloaded PNG.

- [ ] **Step 4: Run the production default verification**

Run the same request with `frame_count = 8` and `sprite_size = 128`. Open `metadata.json` and verify `source_frame_count` is `9`, eight RGBA frames are listed, and the sheet dimensions equal `384 x 384` because eight 128-pixel frames use three columns and three rows.

- [ ] **Step 5: Record the verification result without staging runtime artifacts**

Run: `rtk git status --short`

Expected: no files below `output/game_assets/` or `input/game_assets/` are staged. If the live check exposes a source defect, stop verification and add a new TDD task naming the failing test and affected source files before editing it.

## Plan Self-Review

- Spec coverage: Tasks 1 through 3 install and wire the selected local models and two ComfyUI graphs; Tasks 4 and 5 guarantee exact frame counts, alpha assets, job states, and metadata; Task 6 exposes the local/LAN contract; Tasks 7 and 8 document and prove the completed service.
- Completeness scan: all model URLs, byte counts, SHA-256 values, file paths, test commands, API values, node names, and expected outcomes are explicit.
- Type consistency: `AssetRequest`, `JobStatus`, `ComfyClient`, `JobStore`, `JobRunner`, `wan_source_frame_count`, and the status string values use the same names across the tasks.
