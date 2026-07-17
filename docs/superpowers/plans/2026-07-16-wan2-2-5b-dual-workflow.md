# Wan2.2 5B Dual-Mode Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one native ComfyUI workflow that runs Wan2.2 5B in text-to-video mode by default and image-to-video mode when its image input is enabled.

**Architecture:** Derive a new workflow from the installed official Wan2.2 5B template using a structured JSON transformation, preserving the official executable graph while removing the note node, applying the approved balanced defaults, and arranging five clear node groups. Validate the production file through ComfyUI APIs, then submit a separate low-cost API prompt for end-to-end generation without changing production values.

**Tech Stack:** ComfyUI 0.26.2 core nodes, Wan2.2-TI2V-5B FP16, UMT5 XXL FP8, Python 3.13 standard library, PowerShell, ComfyUI HTTP API

---

### Task 1: Create the production workflow

**Files:**
- Read: `E:\ComfyUI\user\default\workflows\video_wan2_2_5B_ti2v.json`
- Create: `E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json`

- [x] **Step 1: Run the missing-artifact test**

Run:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -c "from pathlib import Path; p=Path(r'E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json'); assert p.exists(), f'missing workflow: {p}'"
```

Expected: exit code `1` with `AssertionError: missing workflow`.

- [x] **Step 2: Transform the official workflow into the balanced dual-mode workflow**

Run from `E:\ComfyUI`:

```powershell
$code = @'
import json
import uuid
from pathlib import Path

source = Path(r"E:\ComfyUI\user\default\workflows\video_wan2_2_5B_ti2v.json")
target = Path(r"E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json")
workflow = json.loads(source.read_text(encoding="utf-8"))

workflow["id"] = str(uuid.uuid4())
workflow["revision"] = 0
workflow["nodes"] = [node for node in workflow["nodes"] if node["type"] != "MarkdownNote"]
workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])

nodes = {node["id"]: node for node in workflow["nodes"]}

nodes[37]["title"] = "Wan2.2 5B Diffusion Model"
nodes[37]["widgets_values"] = ["wan2.2_ti2v_5B_fp16.safetensors", "default"]
nodes[38]["title"] = "Wan Text Encoder"
nodes[38]["widgets_values"] = ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"]
nodes[39]["title"] = "Wan2.2 VAE"
nodes[39]["widgets_values"] = ["wan2.2_vae.safetensors"]

nodes[56]["title"] = "Image Input - Enable For I2V"
nodes[56]["mode"] = 4
nodes[55]["title"] = "Video Size And Length"
nodes[55]["widgets_values"] = [832, 480, 81, 1]

nodes[6]["title"] = "Positive Prompt"
nodes[6]["widgets_values"] = [
    "Cinematic wide shot of a rain-washed city street at blue hour, warm storefront lights reflecting on the pavement, pedestrians moving naturally, slow camera dolly forward, realistic motion, detailed lighting."
]
nodes[7]["title"] = "Negative Prompt"

nodes[48]["title"] = "Wan Sampling Shift"
nodes[48]["widgets_values"] = [8]
nodes[3]["title"] = "Wan2.2 Sampler"
nodes[3]["widgets_values"] = [42, "randomize", 20, 5, "uni_pc", "simple", 1]

nodes[8]["title"] = "Decode Video Frames"
nodes[57]["title"] = "Create 24 FPS Video"
nodes[57]["widgets_values"] = [24]
nodes[58]["title"] = "Save MP4 H.264"
nodes[58]["widgets_values"] = ["video/Wan2.2_5B", "mp4", "h264"]

positions = {
    37: [0, 0],
    38: [0, 130],
    39: [0, 300],
    56: [420, 0],
    55: [420, 350],
    6: [800, 0],
    7: [800, 220],
    48: [1280, 0],
    3: [1280, 120],
    8: [1660, 0],
    57: [1660, 100],
    58: [1660, 230],
}
for node_id, position in positions.items():
    nodes[node_id]["pos"] = position

group_color = "#3f789e"
workflow["groups"] = [
    {"id": 1, "title": "1 - Model Loading", "bounding": [-20, -40, 380, 460], "color": group_color, "font_size": 24, "flags": {}},
    {"id": 2, "title": "2 - T2V / I2V Input", "bounding": [400, -40, 340, 760], "color": group_color, "font_size": 24, "flags": {}},
    {"id": 3, "title": "3 - Prompts", "bounding": [780, -40, 470, 500], "color": group_color, "font_size": 24, "flags": {}},
    {"id": 4, "title": "4 - Sampling", "bounding": [1260, -40, 360, 480], "color": group_color, "font_size": 24, "flags": {}},
    {"id": 5, "title": "5 - Decode And Save", "bounding": [1640, -40, 700, 740], "color": group_color, "font_size": 24, "flags": {}},
]

workflow.setdefault("extra", {})["ds"] = {"scale": 0.55, "offset": [80, 80]}
target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)
'@
$code | & 'E:\ComfyUI\.venv\Scripts\python.exe' -
```

Expected: exit code `0` and the absolute target path is printed.

- [x] **Step 3: Run the artifact test again**

Run the Step 1 command again.

Expected: exit code `0` with no assertion output.

### Task 2: Verify the production graph and ComfyUI integration

**Files:**
- Test: `E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json`

- [x] **Step 1: Verify graph structure, model names, modes, links, groups, and production widgets**

Run:

```powershell
$code = @'
import json
from pathlib import Path

path = Path(r"E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json")
workflow = json.loads(path.read_text(encoding="utf-8"))
nodes = {node["id"]: node for node in workflow["nodes"]}
node_ids = set(nodes)

assert len(workflow["nodes"]) == 12
assert all(node["type"] != "MarkdownNote" for node in workflow["nodes"])
assert nodes[37]["widgets_values"] == ["wan2.2_ti2v_5B_fp16.safetensors", "default"]
assert nodes[38]["widgets_values"] == ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"]
assert nodes[39]["widgets_values"] == ["wan2.2_vae.safetensors"]
assert nodes[56]["mode"] == 4
assert nodes[55]["widgets_values"] == [832, 480, 81, 1]
assert nodes[48]["widgets_values"] == [8]
assert nodes[3]["widgets_values"][1:] == ["randomize", 20, 5, "uni_pc", "simple", 1]
assert nodes[57]["widgets_values"] == [24]
assert nodes[58]["widgets_values"] == ["video/Wan2.2_5B", "mp4", "h264"]
assert [group["title"] for group in workflow["groups"]] == [
    "1 - Model Loading",
    "2 - T2V / I2V Input",
    "3 - Prompts",
    "4 - Sampling",
    "5 - Decode And Save",
]
for link in workflow["links"]:
    assert link[1] in node_ids, f"missing link source: {link}"
    assert link[3] in node_ids, f"missing link target: {link}"
print("production_workflow_ok")
'@
$code | & 'E:\ComfyUI\.venv\Scripts\python.exe' -
```

Expected: `production_workflow_ok` and exit code `0`.

- [x] **Step 2: Verify all executable node types and selected models through `/object_info`**

Run:

```powershell
$code = @'
import json
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:8188"
workflow = json.loads(Path(r"E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json").read_text(encoding="utf-8"))
with urllib.request.urlopen(base + "/object_info", timeout=30) as response:
    info = json.load(response)

types = {node["type"] for node in workflow["nodes"]}
missing = sorted(types - info.keys())
assert not missing, f"missing node types: {missing}"
assert "wan2.2_ti2v_5B_fp16.safetensors" in info["UNETLoader"]["input"]["required"]["unet_name"][0]
assert "umt5_xxl_fp8_e4m3fn_scaled.safetensors" in info["CLIPLoader"]["input"]["required"]["clip_name"][0]
assert "wan2.2_vae.safetensors" in info["VAELoader"]["input"]["required"]["vae_name"][0]
print("object_info_ok")
'@
$code | & 'E:\ComfyUI\.venv\Scripts\python.exe' -
```

Expected: `object_info_ok` and exit code `0`.

- [x] **Step 3: Verify the workflow is listed by the user-data API**

Run:

```powershell
$items = Invoke-RestMethod -Uri 'http://127.0.0.1:8188/userdata?dir=workflows&recurse=true&full_info=true' -TimeoutSec 30
$workflow = $items | Where-Object { $_.path -eq 'wan2_2_5b_dual_balanced.json' }
$workflow | Format-List
if (-not $workflow -or $workflow.size -le 0) { exit 2 }
```

Expected: one item named `wan2_2_5b_dual_balanced.json` with a positive size and exit code `0`.

### Task 3: Run the end-to-end smoke generation

**Files:**
- Create: `E:\ComfyUI\output\video\Wan2.2_5B_smoke_*.mp4`
- Test: `http://127.0.0.1:8188/prompt`
- Test: `http://127.0.0.1:8188/history/{prompt_id}`

- [x] **Step 1: Submit a temporary 512 x 288, 17-frame, 4-step T2V prompt and wait for completion**

Run:

```powershell
$code = @'
import json
import time
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:8188"
output_dir = Path(r"E:\ComfyUI\output\video")
started_ns = time.time_ns()

prompt = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
    "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "Cinematic shot of a red paper lantern swaying gently in a night breeze, soft warm light, dark simple background, natural motion, locked camera."}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "static, blurry, low quality, distorted, text, watermark, camera shake"}},
    "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 512, "height": 288, "length": 17, "batch_size": 1}},
    "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": 42, "steps": 4, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "denoise": 1.0}},
    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
    "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": 24.0}},
    "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": "video/Wan2.2_5B_smoke", "format": "mp4", "codec": "h264"}},
}

body = json.dumps({"prompt": prompt, "client_id": "codex-wan22-smoke"}).encode("utf-8")
request = urllib.request.Request(base + "/prompt", data=body, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(request, timeout=60) as response:
    queued = json.load(response)
prompt_id = queued["prompt_id"]
print("prompt_id=" + prompt_id, flush=True)

deadline = time.time() + 1800
history_item = None
while time.time() < deadline:
    with urllib.request.urlopen(base + "/history/" + prompt_id, timeout=30) as response:
        history = json.load(response)
    if prompt_id in history:
        history_item = history[prompt_id]
        status = history_item.get("status", {})
        messages = status.get("messages", [])
        errors = [message for message in messages if message and message[0] == "execution_error"]
        if errors:
            raise RuntimeError(json.dumps(errors, ensure_ascii=True))
        if status.get("completed"):
            break
    time.sleep(2)
else:
    raise TimeoutError("smoke generation did not finish within 1800 seconds")

files = [
    path for path in output_dir.glob("Wan2.2_5B_smoke_*.mp4")
    if path.stat().st_mtime_ns >= started_ns and path.stat().st_size > 0
]
assert files, "no non-empty smoke MP4 was created"
latest = max(files, key=lambda path: path.stat().st_mtime_ns)
print("smoke_output=" + str(latest))
print("smoke_bytes=" + str(latest.stat().st_size))
'@
$code | & 'E:\ComfyUI\.venv\Scripts\python.exe' -
```

Expected: a `prompt_id`, a `smoke_output` path ending in `.mp4`, a positive `smoke_bytes` value, and exit code `0`.

- [x] **Step 2: Confirm ComfyUI remains healthy after generation**

Run:

```powershell
$response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8188' -TimeoutSec 10
$response.StatusCode
```

Expected: HTTP status `200`.

### Task 4: Recheck production defaults and hand off the workflow

**Files:**
- Test: `E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json`
- Inspect: `E:\ComfyUI\output\video\Wan2.2_5B_smoke_*.mp4`

- [x] **Step 1: Recheck production defaults, node availability, models, and user-data listing**

Run:

```powershell
$code = @'
import json
import urllib.parse
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:8188"
path = Path(r"E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json")
workflow = json.loads(path.read_text(encoding="utf-8"))
nodes = {node["id"]: node for node in workflow["nodes"]}

assert len(workflow["nodes"]) == 12
assert all(node["type"] != "MarkdownNote" for node in workflow["nodes"])
assert nodes[37]["widgets_values"] == ["wan2.2_ti2v_5B_fp16.safetensors", "default"]
assert nodes[38]["widgets_values"] == ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"]
assert nodes[39]["widgets_values"] == ["wan2.2_vae.safetensors"]
assert nodes[56]["mode"] == 4
assert nodes[55]["widgets_values"] == [832, 480, 81, 1]
assert nodes[48]["widgets_values"] == [8]
assert nodes[3]["widgets_values"][1:] == ["randomize", 20, 5, "uni_pc", "simple", 1]
assert nodes[57]["widgets_values"] == [24]
assert nodes[58]["widgets_values"] == ["video/Wan2.2_5B", "mp4", "h264"]

with urllib.request.urlopen(base + "/object_info", timeout=30) as response:
    info = json.load(response)
types = {node["type"] for node in workflow["nodes"]}
assert not (types - info.keys()), f"missing node types: {sorted(types - info.keys())}"
assert "wan2.2_ti2v_5B_fp16.safetensors" in info["UNETLoader"]["input"]["required"]["unet_name"][0]
assert "umt5_xxl_fp8_e4m3fn_scaled.safetensors" in info["CLIPLoader"]["input"]["required"]["clip_name"][0]
assert "wan2.2_vae.safetensors" in info["VAELoader"]["input"]["required"]["vae_name"][0]

query = urllib.parse.urlencode({"dir": "workflows", "recurse": "true", "full_info": "true"})
with urllib.request.urlopen(base + "/userdata?" + query, timeout=30) as response:
    items = json.load(response)
matches = [item for item in items if item.get("path") == path.name and item.get("size", 0) > 0]
assert len(matches) == 1, f"workflow listing mismatch: {matches}"
print("final_production_recheck_ok")
'@
$code | & 'E:\ComfyUI\.venv\Scripts\python.exe' -
```

Expected: `final_production_recheck_ok` and exit code `0`. This confirms that the API smoke prompt did not modify production values.

- [x] **Step 2: Record final runtime state**

Run:

```powershell
$listener = Get-NetTCPConnection -State Listen -LocalPort 8188 -ErrorAction Stop | Select-Object -First 1
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
[pscustomobject]@{
    URL = 'http://127.0.0.1:8188'
    PID = $listener.OwningProcess
    Process = $process.Name
    Workflow = 'E:\ComfyUI\user\default\workflows\wan2_2_5b_dual_balanced.json'
} | Format-List
```

Expected: the listener is a Python process, the URL is `http://127.0.0.1:8188`, and the workflow path is the production file.
