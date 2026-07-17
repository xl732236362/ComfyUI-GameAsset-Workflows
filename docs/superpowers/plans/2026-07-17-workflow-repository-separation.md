# Standalone ComfyUI Workflow Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete game-asset workflow toolkit into a public standalone repository while restoring `E:\ComfyUI` to an official-only Git checkout.

**Architecture:** `E:\ComfyUI-GameAsset-Workflows` is the sole source repository for workflow JSON, the Python API, dependency manifests, deployment tooling, tests, and documentation. An idempotent deployment entry point copies validated workflow artifacts and installs pinned runtime dependencies into an explicitly selected ComfyUI root; the ComfyUI checkout contains only official source plus ignored runtime assets.

**Tech Stack:** Git, PowerShell 5.1, Python 3.13, aiohttp, Pillow, pytest, Git Credential Manager, GitHub REST API, ComfyUI HTTP API.

---

## File Structure

The standalone repository at `E:\ComfyUI-GameAsset-Workflows` will contain:

- Create: `.gitignore` - exclude secrets, models, runtime data, caches, environments, and local migration backups.
- Create: `pyproject.toml` - declare the standalone package and pytest configuration.
- Create: `README.md` - document setup, deployment, API startup, and live action commands.
- Create: `deploy.ps1` - stable PowerShell entry point that selects the ComfyUI Python interpreter.
- Copy: `game_asset_api/*.py` - the existing API, workflows, runners, manifests, and post-processing code.
- Create: `game_asset_api/deployment.py` - validate a ComfyUI root and atomically publish the five workflow JSON files.
- Create: `game_asset_api/repository_audit.py` - reject disallowed tracked paths, secrets, runtime artifacts, and oversized files.
- Copy and modify: `scripts/*.py` - export, install, deploy, and run commands.
- Copy: `workflows/*.json` - the five approved workflow artifacts.
- Copy and modify: `tests/game_asset_api_test/*.py` - the existing 130-test suite under the standalone layout.
- Create: `tests/game_asset_api_test/test_deployment.py` - deployment validation, atomicity, and idempotency tests.
- Create: `tests/game_asset_api_test/test_repository_audit.py` - public-repository content policy tests.
- Copy: `docs/` - relevant game-asset, Wan, pose deployment, tuning, and repository-separation documents.

The ComfyUI repository will only change Git configuration and checked-out branch after the standalone repository is committed, deployed, and pushed successfully.

### Task 1: Create The Standalone Repository Baseline

**Files:**
- Create: `E:\ComfyUI-GameAsset-Workflows\.gitignore`
- Create: `E:\ComfyUI-GameAsset-Workflows\pyproject.toml`
- Create: `E:\ComfyUI-GameAsset-Workflows\README.md`
- Copy: `E:\ComfyUI\game_asset_api` to `E:\ComfyUI-GameAsset-Workflows\game_asset_api`
- Copy: approved scripts to `E:\ComfyUI-GameAsset-Workflows\scripts`
- Copy: `E:\ComfyUI\tests-unit\game_asset_api_test` to `E:\ComfyUI-GameAsset-Workflows\tests\game_asset_api_test`
- Copy: five workflow JSON files to `E:\ComfyUI-GameAsset-Workflows\workflows`
- Copy: relevant documents to `E:\ComfyUI-GameAsset-Workflows\docs`

- [ ] **Step 1: Create the directory and initialize an independent Git history**

Run:

```powershell
New-Item -ItemType Directory -Path 'E:\ComfyUI-GameAsset-Workflows' -ErrorAction Stop
git -C 'E:\ComfyUI-GameAsset-Workflows' init -b main
```

Expected: a new repository whose `git rev-parse --show-toplevel` is exactly `E:/ComfyUI-GameAsset-Workflows`.

- [ ] **Step 2: Add a restrictive public-repository ignore policy**

Create `.gitignore` with:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
venv/
models/
custom_nodes/
input/
output/
temp/
wheelhouse/
.local-backup/
*.part
*.log
*.safetensors
*.ckpt
*.pt
*.pth
*.pem
id_ed25519*
.env
```

- [ ] **Step 3: Add standalone Python configuration**

Create `pyproject.toml` with:

```toml
[project]
name = "comfyui-game-asset-workflows"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "aiohttp>=3.11.8",
  "Pillow>=10.0",
]

[project.optional-dependencies]
test = [
  "pytest>=8.0",
  "pytest-asyncio>=0.24",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 4: Copy the approved source set mechanically**

Copy these exact paths without copying `.git`, models, inputs, outputs, caches, or custom-node source:

```text
game_asset_api/
scripts/create_sword_attack_pose_sequence.py
scripts/export_game_asset_workflows.py
scripts/export_pose_controlled_workflow.py
scripts/install_game_asset_models.py
scripts/install_pose_workflow_nodes.py
scripts/run_pose_controlled_action.py
tests-unit/game_asset_api_test/ -> tests/game_asset_api_test/
docs/superpowers/2026-07-17-pose-controlled-pixel-animation-deployment-audit.md
docs/superpowers/plans/*game-asset*.md
docs/superpowers/plans/*wan2-2*.md
docs/superpowers/plans/*pose-controlled*.md
docs/superpowers/specs/*game-asset*.md
docs/superpowers/specs/*wan2-2*.md
docs/superpowers/specs/*pose-controlled*.md
docs/superpowers/specs/2026-07-17-workflow-repository-separation-design.md
docs/superpowers/plans/2026-07-17-workflow-repository-separation.md
```

Copy these five workflow artifacts into `workflows/`:

```text
pixel_character_design_api.json
pixel_character_action_api.json
pose_controlled_pixel_action_api.json
video_wan2_2_5B_ti2v.json
wan2_2_5b_dual_balanced.json
```

- [ ] **Step 5: Add the initial README**

Document these supported commands exactly:

```powershell
E:\ComfyUI\.venv\Scripts\python.exe -m pytest -q
.\deploy.ps1 -ComfyRoot E:\ComfyUI
$env:COMFYUI_ROOT = 'E:\ComfyUI'
$env:GAME_ASSET_API_PORT = '8190'
E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api
```

State that model files and generated assets are intentionally excluded and acquired through verified manifests.
Do not add a `LICENSE` file during migration; public visibility does not grant a
reuse license, and license selection remains a separate explicit user decision.

- [ ] **Step 6: Run the copied baseline suite**

Run:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
```

Expected: the existing suite passes before standalone path behavior changes.

- [ ] **Step 7: Commit the baseline migration**

```powershell
git add .gitignore pyproject.toml README.md game_asset_api scripts tests workflows docs
git commit -m "feat: create standalone workflow toolkit"
```

### Task 2: Make Workflow Artifacts Standalone-Owned

**Files:**
- Modify: `scripts/export_game_asset_workflows.py`
- Modify: `scripts/export_pose_controlled_workflow.py`
- Modify: `tests/game_asset_api_test/test_workflows.py`
- Modify: `tests/game_asset_api_test/test_pose_workflow.py`

- [ ] **Step 1: Change tests to require repository-owned workflow output**

Change workflow assertions from `ROOT / "user" / "default" / "workflows"` to:

```python
WORKFLOW_DIRECTORY = ROOT / "workflows"
```

Require the two export commands to produce the three generated API artifacts in that directory without creating a top-level `user` directory.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_workflows.py' `
  'tests\game_asset_api_test\test_pose_workflow.py::test_export_script_writes_prompt_wrapped_pose_workflow' -q
```

Expected: failures show the exporters still write below `user/default/workflows`.

- [ ] **Step 3: Point both exporters at `ROOT / "workflows"`**

Use:

```python
WORKFLOW_DIRECTORY = ROOT / "workflows"
```

and for the pose artifact:

```python
OUTPUT_PATH = ROOT / "workflows" / "pose_controlled_pixel_action_api.json"
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again. Expected: all selected tests pass and no `user/` directory is created.

- [ ] **Step 5: Regenerate and verify all five workflow JSON files**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\export_game_asset_workflows.py'
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\export_pose_controlled_workflow.py'
Get-ChildItem 'workflows' -Filter '*.json' | Select-Object -ExpandProperty Name
```

Expected: exactly the five approved names remain, and every file parses with `ConvertFrom-Json`.

- [ ] **Step 6: Commit**

```powershell
git add scripts/export_game_asset_workflows.py scripts/export_pose_controlled_workflow.py tests/game_asset_api_test/test_workflows.py tests/game_asset_api_test/test_pose_workflow.py workflows
git commit -m "refactor: own workflow artifacts in standalone repo"
```

### Task 3: Require An Explicit ComfyUI Runtime Root

**Files:**
- Modify: `game_asset_api/__main__.py`
- Modify: `scripts/install_game_asset_models.py`
- Modify: `tests/game_asset_api_test/test_app.py`
- Modify: `tests/game_asset_api_test/test_model_manifest.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing runtime-root tests**

Add tests requiring:

```python
def test_project_root_comes_from_comfyui_root(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFYUI_ROOT", str(tmp_path))
    assert module._project_root_from_environment() == tmp_path


def test_project_root_requires_comfyui_root(monkeypatch):
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
    with pytest.raises(ValueError, match="COMFYUI_ROOT"):
        module._project_root_from_environment()
```

Update the model-installer CLI test so omitting `--root` raises `SystemExit`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest `
  'tests\game_asset_api_test\test_app.py' `
  'tests\game_asset_api_test\test_model_manifest.py' -q
```

Expected: `_project_root_from_environment` is missing and `--root` is still optional.

- [ ] **Step 3: Implement explicit runtime-root resolution**

Add to `game_asset_api/__main__.py`:

```python
def _project_root_from_environment() -> Path:
    value = os.environ.get("COMFYUI_ROOT")
    if not value:
        raise ValueError("COMFYUI_ROOT must point to the ComfyUI installation")
    root = Path(value).expanduser()
    if not (root / "main.py").is_file():
        raise ValueError("COMFYUI_ROOT must contain main.py")
    return root
```

Use it instead of `Path(__file__).resolve().parents[1]` when constructing `JobRunner`. Make `--root` required in `scripts/install_game_asset_models.py`.

- [ ] **Step 4: Update README commands**

Document setting `COMFYUI_ROOT` before `python -m game_asset_api`; remove the old assumption that the package lives inside ComfyUI.

- [ ] **Step 5: Run focused and full tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add game_asset_api/__main__.py scripts/install_game_asset_models.py tests/game_asset_api_test/test_app.py tests/game_asset_api_test/test_model_manifest.py README.md
git commit -m "fix: separate workflow source from ComfyUI runtime"
```

### Task 4: Build Atomic, Idempotent Workflow Deployment Under TDD

**Files:**
- Create: `game_asset_api/deployment.py`
- Create: `scripts/deploy.py`
- Create: `deploy.ps1`
- Create: `tests/game_asset_api_test/test_deployment.py`

- [ ] **Step 1: Write failing deployment tests**

Cover these contracts with real temporary directories:

```python
import json
import os
from pathlib import Path

import pytest

from game_asset_api.deployment import WORKFLOW_NAMES, publish_workflows, validate_comfy_root


def _write_sources(directory: Path) -> dict[str, bytes]:
    directory.mkdir(parents=True)
    payloads = {}
    for name in WORKFLOW_NAMES:
        payload = (json.dumps({"prompt": {"name": name}}) + "\n").encode()
        (directory / name).write_bytes(payload)
        payloads[name] = payload
    return payloads


def _valid_comfy_root(root: Path) -> Path:
    (root / "main.py").write_text("# test\n", encoding="utf-8")
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test python")
    return python


def test_publish_workflows_validates_and_copies_all_five(tmp_path):
    source = tmp_path / "source"
    payloads = _write_sources(source)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)

    published = publish_workflows(source, comfy_root)

    destination = comfy_root / "user" / "default" / "workflows"
    assert tuple(path.name for path in published) == WORKFLOW_NAMES
    assert {path.name: path.read_bytes() for path in destination.glob("*.json")} == payloads


def test_publish_workflows_rejects_invalid_json_without_replacing_destination(tmp_path):
    source = tmp_path / "source"
    _write_sources(source)
    (source / WORKFLOW_NAMES[3]).write_text("not json", encoding="utf-8")
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    destination = comfy_root / "user" / "default" / "workflows"
    destination.mkdir(parents=True)
    old_payloads = {}
    for name in WORKFLOW_NAMES:
        payload = f"old:{name}".encode()
        (destination / name).write_bytes(payload)
        old_payloads[name] = payload

    with pytest.raises(ValueError, match="workflow JSON"):
        publish_workflows(source, comfy_root)

    assert {path.name: path.read_bytes() for path in destination.glob("*.json")} == old_payloads


def test_publish_workflows_preserves_mtime_when_bytes_match(tmp_path):
    source = tmp_path / "source"
    _write_sources(source)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    publish_workflows(source, comfy_root)
    target = comfy_root / "user" / "default" / "workflows" / WORKFLOW_NAMES[0]
    fixed_time = 1_700_000_000
    os.utime(target, (fixed_time, fixed_time))

    publish_workflows(source, comfy_root)

    assert target.stat().st_mtime == fixed_time


def test_validate_comfy_root_requires_main_and_python(tmp_path):
    with pytest.raises(ValueError, match="main.py"):
        validate_comfy_root(tmp_path)
    (tmp_path / "main.py").write_text("# test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="python.exe"):
        validate_comfy_root(tmp_path)
    python = _valid_comfy_root(tmp_path)
    assert validate_comfy_root(tmp_path) == (tmp_path.resolve(), python.resolve())
```

Use one valid minimal payload per source file: `{"prompt": {}}`.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test\test_deployment.py' -q
```

Expected: import failure for `game_asset_api.deployment`.

- [ ] **Step 3: Implement the deployment boundary**

Expose these APIs in `game_asset_api/deployment.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path


WORKFLOW_NAMES = (
    "pixel_character_design_api.json",
    "pixel_character_action_api.json",
    "pose_controlled_pixel_action_api.json",
    "video_wan2_2_5B_ti2v.json",
    "wan2_2_5b_dual_balanced.json",
)


def validate_comfy_root(root: Path) -> tuple[Path, Path]:
    """Return normalized root and its .venv Python after structural checks."""
    root = Path(root).expanduser().resolve()
    if not (root / "main.py").is_file():
        raise ValueError("ComfyUI root must contain main.py")
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise ValueError("ComfyUI root must contain .venv/Scripts/python.exe")
    return root, python.resolve()


def publish_workflows(source: Path, comfy_root: Path) -> tuple[Path, ...]:
    """Validate all five JSON files and atomically publish changed bytes."""
    root, _ = validate_comfy_root(comfy_root)
    source = Path(source)
    payloads = {}
    for name in WORKFLOW_NAMES:
        path = source / name
        try:
            payload = path.read_bytes()
            parsed = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid workflow JSON: {name}") from error
        if not isinstance(parsed, dict) or not isinstance(parsed.get("prompt"), dict):
            raise ValueError(f"invalid workflow JSON: {name}")
        payloads[name] = payload

    destination = root / "user" / "default" / "workflows"
    destination.mkdir(parents=True, exist_ok=True)
    published = []
    for name in WORKFLOW_NAMES:
        target = destination / name
        payload = payloads[name]
        if not target.is_file() or target.read_bytes() != payload:
            temporary = target.with_name(f"{target.name}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        published.append(target)
    return tuple(published)
```

Parse every source with `json.loads` before writing anything. Write changed files beside the destination as `<name>.tmp`, then use `os.replace`. If bytes match, do not rewrite the destination.

- [ ] **Step 4: Implement the Python deploy orchestrator**

`scripts/deploy.py` must accept:

```text
--comfy-root PATH        required
--base-url URL           default http://127.0.0.1:8188
--skip-nodes             test/diagnostic option
--skip-models            test/diagnostic option
--skip-discovery         offline option
--skip-smoke             fast validation option
```

Its production order is publish workflows, install nodes, install models, validate `/object_info`, then run the two-frame pose smoke action. Each child operation receives the explicit ComfyUI root and selected Python executable.

- [ ] **Step 5: Add the stable PowerShell wrapper**

Create `deploy.ps1`:

```powershell
param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyRoot,
    [string]$BaseUrl = 'http://127.0.0.1:8188'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ComfyRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ComfyUI Python not found: $python"
}
& $python (Join-Path $PSScriptRoot 'scripts\deploy.py') `
    --comfy-root $ComfyRoot --base-url $BaseUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

- [ ] **Step 6: Run focused and full tests**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test\test_deployment.py' -q
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add game_asset_api/deployment.py scripts/deploy.py deploy.ps1 tests/game_asset_api_test/test_deployment.py
git commit -m "feat: add idempotent ComfyUI workflow deployment"
```

### Task 5: Enforce Public Repository Content Policy

**Files:**
- Create: `game_asset_api/repository_audit.py`
- Create: `scripts/audit_repository.py`
- Create: `tests/game_asset_api_test/test_repository_audit.py`

- [ ] **Step 1: Write failing audit tests**

Require the audit to accept approved top-level paths and reject:

```text
models/model.safetensors
output/frame.png
.env
id_ed25519
asset.part
unexpected.bin larger than 10 MiB
```

The allowed top-level names are `.gitignore`, `README.md`, `deploy.ps1`,
`pyproject.toml`, `game_asset_api`, `scripts`, `tests`, `workflows`, and `docs`.

Use these complete tests:

```python
from pathlib import Path

import pytest

from game_asset_api.repository_audit import audit_paths


def _write(root: Path, relative: str, size: int = 1) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return Path(relative)


def test_audit_accepts_approved_toolkit_paths(tmp_path):
    paths = [
        _write(tmp_path, "README.md"),
        _write(tmp_path, "game_asset_api/workflows.py"),
        _write(tmp_path, "scripts/deploy.py"),
        _write(tmp_path, "tests/test_deployment.py"),
        _write(tmp_path, "workflows/example.json"),
        _write(tmp_path, "docs/design.md"),
    ]
    assert audit_paths(tmp_path, paths) == []


@pytest.mark.parametrize(
    "relative,size",
    [
        ("models/model.safetensors", 1),
        ("output/frame.png", 1),
        (".env", 1),
        ("id_ed25519", 1),
        ("asset.part", 1),
        ("docs/unexpected.bin", 10 * 1024 * 1024 + 1),
    ],
)
def test_audit_rejects_runtime_secret_and_oversized_paths(tmp_path, relative, size):
    path = _write(tmp_path, relative, size)
    violations = audit_paths(tmp_path, [path])
    assert len(violations) == 1
    assert relative.replace("\\", "/") in violations[0]
```

- [ ] **Step 2: Run and verify RED**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test\test_repository_audit.py' -q
```

Expected: import failure for `game_asset_api.repository_audit`.

- [ ] **Step 3: Implement the audit**

Expose:

```python
from __future__ import annotations

from pathlib import Path


ALLOWED_TOP_LEVEL = {
    ".gitignore",
    "README.md",
    "deploy.ps1",
    "pyproject.toml",
    "game_asset_api",
    "scripts",
    "tests",
    "workflows",
    "docs",
}
DISALLOWED_NAMES = {".env", "id_ed25519", "id_rsa"}
DISALLOWED_SUFFIXES = {".part", ".safetensors", ".ckpt", ".pt", ".pth", ".pem"}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def audit_paths(root: Path, relative_paths: list[Path]) -> list[str]:
    """Return deterministic policy violations for proposed tracked files."""
    root = Path(root)
    violations = []
    for relative in sorted(Path(path) for path in relative_paths):
        normalized = relative.as_posix()
        path = root / relative
        reason = None
        if not relative.parts or relative.parts[0] not in ALLOWED_TOP_LEVEL:
            reason = "disallowed top-level path"
        elif relative.name in DISALLOWED_NAMES or relative.name.startswith("id_ed25519"):
            reason = "secret filename"
        elif relative.suffix.lower() in DISALLOWED_SUFFIXES:
            reason = "disallowed runtime or model file"
        elif path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            reason = "file exceeds 10 MiB"
        if reason is not None:
            violations.append(f"{normalized}: {reason}")
    return violations
```

Reject disallowed top-level components, private-key names, secret files,
runtime directories, model-weight extensions, `.part` files, and regular files
larger than `10 * 1024 * 1024` bytes.

- [ ] **Step 4: Implement the Git-index CLI**

`scripts/audit_repository.py` runs `git ls-files -z`, passes those paths to
`audit_paths`, prints each violation to stderr, and exits 1 on any violation.

Implement it as:

```python
from pathlib import Path
import subprocess
import sys

from game_asset_api.repository_audit import audit_paths


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = [Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value]
    violations = audit_paths(root, paths)
    if violations:
        sys.stderr.write("\n".join(violations) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run focused tests and audit the real index**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test\test_repository_audit.py' -q
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\audit_repository.py'
```

Expected: tests pass and the real index audit exits 0.

- [ ] **Step 6: Commit**

```powershell
git add game_asset_api/repository_audit.py scripts/audit_repository.py tests/game_asset_api_test/test_repository_audit.py
git commit -m "feat: audit public workflow repository contents"
```

### Task 6: Verify The Standalone Repository Before External Changes

**Files:**
- Modify: `README.md`
- Create: `.local-backup/comfyui-status.txt` (ignored)
- Create: `.local-backup/README.patch` (ignored)

- [ ] **Step 1: Record the ComfyUI working state without staging it**

```powershell
New-Item -ItemType Directory '.local-backup' -Force | Out-Null
git -C 'E:\ComfyUI' status --porcelain=v1 -uall > '.local-backup\comfyui-status.txt'
git -C 'E:\ComfyUI' diff -- README.md > '.local-backup\README.patch'
```

- [ ] **Step 2: Run full static and unit verification**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m compileall -q game_asset_api scripts
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
git diff --check
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\audit_repository.py'
```

Expected: compile exit 0, all tests pass, no whitespace errors, and audit exit 0.

- [ ] **Step 3: Verify generated workflow artifacts are deterministic**

Run both exporters twice and compare SHA-256 values of all five files. Expected:
the second run produces exactly the same hashes.

- [ ] **Step 4: Complete README operational documentation**

Document repository purpose, excluded assets, `deploy.ps1`, API startup,
pose-runner examples, model licensing notes, troubleshooting, and the known
OpenPose weapon-geometry limitation.

- [ ] **Step 5: Commit documentation changes**

```powershell
git add README.md workflows
git commit -m "docs: document standalone workflow operations"
```

### Task 7: Deploy Twice And Run Live Validation

**Files:**
- Runtime only: `E:\ComfyUI\user\default\workflows\*.json`
- Runtime only: `E:\ComfyUI\models\...`
- Runtime only: `E:\ComfyUI\custom_nodes\...`
- Runtime only: `E:\ComfyUI\output\game_assets\...`

- [ ] **Step 1: Run the first full deployment**

```powershell
& '.\deploy.ps1' -ComfyRoot 'E:\ComfyUI'
```

Expected: all five workflows publish; existing pinned nodes and models verify
without redownload; `/object_info` discovery succeeds; two-frame smoke passes.

- [ ] **Step 2: Capture workflow hashes and run deployment again**

Capture SHA-256 and last-write times for the five deployed workflow files, run
the same deploy command, and capture them again. Expected: hashes and mtimes are
unchanged on the second run.

- [ ] **Step 3: Run the full eight-frame live action**

Use the existing approved xianxia reference and parameters:

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\run_pose_controlled_action.py' `
  --root 'E:\ComfyUI' `
  --reference 'E:\ComfyUI\input\game_assets\4b39a56a-f754-4653-b718-30d4227bba2b\reference.png' `
  --job-id 'standalone-repo-live-audit' `
  --character-prompt 'young xianxia cultivator in flowing white and azure robes, long black hair tied back, holding a slender jade sword' `
  --action-prompt 'right hand swings the slender jade sword in a wide horizontal arc to the right' `
  --camera side --frame-count 8 --sprite-size 128 --seed 20260717
```

Expected: eight RGBA frames, a `384x384` sprite sheet, and transparent corners.
Create an eight-frame GIF preview with the installed ffmpeg and verify its frame
count with ffprobe.

- [ ] **Step 4: Re-run repository audit**

Expected: runtime outputs remain outside the standalone Git index.

### Task 8: Create And Push The Public GitHub Repository

**Files:**
- Modify local Git config in `E:\ComfyUI-GameAsset-Workflows\.git\config`
- External: create `xl732236362/ComfyUI-GameAsset-Workflows`

- [ ] **Step 1: Authenticate Git Credential Manager with device flow**

```powershell
git credential-manager github login --device --username xl732236362 --no-ui
```

Expected: the user authorizes the one-time device code and
`git credential-manager github list` reports `xl732236362` without exposing a
token.

- [ ] **Step 2: Create the public non-fork repository through GitHub REST**

Use the credential helper output only in process memory. POST this JSON to
`https://api.github.com/user/repos`:

```json
{
  "name": "ComfyUI-GameAsset-Workflows",
  "description": "Reproducible ComfyUI workflows and tooling for 2D pixel game assets",
  "private": false,
  "has_issues": true,
  "has_projects": false,
  "has_wiki": false
}
```

Never print the credential-helper password/token. If GitHub returns 422 because
the repository already exists, verify its owner, visibility, and non-fork state
before continuing.

Use this PowerShell flow so the token is never written to disk or stdout:

```powershell
$credentialInput = "protocol=https`nhost=github.com`n`n"
$credentialLines = $credentialInput | git credential fill
$credential = @{}
foreach ($line in $credentialLines) {
    $separator = $line.IndexOf('=')
    if ($separator -gt 0) {
        $credential[$line.Substring(0, $separator)] = $line.Substring($separator + 1)
    }
}
if (-not $credential.ContainsKey('password')) { throw 'GitHub credential unavailable' }
$headers = @{
    Authorization = "Bearer $($credential['password'])"
    Accept = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
}
$body = @{
    name = 'ComfyUI-GameAsset-Workflows'
    description = 'Reproducible ComfyUI workflows and tooling for 2D pixel game assets'
    private = $false
    has_issues = $true
    has_projects = $false
    has_wiki = $false
} | ConvertTo-Json
try {
    $repository = Invoke-RestMethod -Method Post `
        -Uri 'https://api.github.com/user/repos' `
        -Headers $headers -ContentType 'application/json' -Body $body
} finally {
    $headers.Authorization = $null
    $credential.Clear()
    Remove-Variable credentialLines, credentialInput -ErrorAction SilentlyContinue
}
$repository | Select-Object full_name,private,fork,default_branch,html_url
```

- [ ] **Step 3: Configure and push the standalone origin over HTTPS**

```powershell
git remote add origin https://github.com/xl732236362/ComfyUI-GameAsset-Workflows.git
git push -u origin main
```

Expected: push exit 0 and `git rev-parse HEAD` equals
`git ls-remote origin refs/heads/main`.

- [ ] **Step 4: Verify the GitHub repository is public and contains no disallowed paths**

Query the repository metadata API and the `main` tree. Expected: `fork=false`,
`private=false`, default branch `main`, and only audited toolkit paths.

### Task 9: Restore ComfyUI To Official-Only Git Tracking

**Files:**
- Modify local Git config: `E:\ComfyUI\.git\config`
- Preserve ignored runtime files under `E:\ComfyUI`

- [ ] **Step 1: Confirm all migration gates before changing ComfyUI**

Require: standalone tests green, live deployment green, GitHub push verified,
and `.local-backup` inventory files present. Stop if any gate is missing.

- [ ] **Step 2: Restore the official remote configuration**

```powershell
git -C 'E:\ComfyUI' remote set-url origin https://github.com/Comfy-Org/ComfyUI.git
git -C 'E:\ComfyUI' remote remove upstream
git -C 'E:\ComfyUI' fetch origin master --tags
```

Expected: both fetch and push URLs for `origin` are official; no workflow
repository remote remains in the ComfyUI checkout.

- [ ] **Step 3: Preserve the modified README and switch to official master**

After verifying `.local-backup\README.patch`, restore only the tracked README in
the ComfyUI worktree, then run:

```powershell
git -C 'E:\ComfyUI' switch -c master --track origin/master
```

If local `master` already exists, verify it contains no unique commits, switch
to it, and fast-forward it with `git merge --ff-only origin/master`. Do not use
`reset --hard`. Keep `codex/pixel-action-continuity-tuning` as a local safety
branch.

- [ ] **Step 4: Redeploy workflows after the branch switch**

The official checkout removes the formerly tracked workflow artifacts. Restore
runtime copies from the standalone source:

```powershell
& 'E:\ComfyUI-GameAsset-Workflows\deploy.ps1' -ComfyRoot 'E:\ComfyUI'
```

Expected: all five files return under `user/default/workflows`, while
`git -C E:\ComfyUI status --short` does not list them because `/user/` is
officially ignored.

- [ ] **Step 5: Restart the game-asset API from the standalone repository**

Set `COMFYUI_ROOT=E:\ComfyUI`, start `python -m game_asset_api` from the
standalone repository, and verify `GET /v1/jobs/<unknown-uuid>` returns 404
without exposing a filesystem path.

### Task 10: Final Dual-Repository Audit

**Files:**
- Modify: `E:\ComfyUI-GameAsset-Workflows\docs\deployment-audit.md`

- [ ] **Step 1: Verify ComfyUI official tracking**

Check:

```powershell
git -C 'E:\ComfyUI' branch --show-current
git -C 'E:\ComfyUI' remote -v
git -C 'E:\ComfyUI' status --short
```

Expected: branch `master`, official origin URLs, and only the pre-existing
preserved local runtime/untracked files.

- [ ] **Step 2: Verify standalone tracking and remote parity**

Check clean status, repository audit, full pytest, origin URL, public metadata,
and equality between local `HEAD` and remote `main`.

- [ ] **Step 3: Verify live runtime state**

Confirm ComfyUI `/system_stats`, `/userdata`, `/object_info`, model hashes, node
revision markers, five deployed workflow hashes, two-frame smoke output, and
eight-frame live audit artifacts.

- [ ] **Step 4: Write and commit the deployment audit**

Record timestamps, repository URLs and commits, workflow hashes, model hashes,
node revisions, test counts, live job paths, visual verdict, and the known
weapon-continuity limitation.

```powershell
git add docs/deployment-audit.md
git commit -m "docs: audit standalone workflow deployment"
git push
```

- [ ] **Step 5: Run final verification after the audit commit**

```powershell
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest 'tests\game_asset_api_test' -q
& 'E:\ComfyUI\.venv\Scripts\python.exe' 'scripts\audit_repository.py'
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: tests and audit exit 0, standalone worktree clean, and remote `main`
matches local `HEAD`.

## Plan Self-Review

- Spec coverage: Tasks 1-3 establish the repository boundary; Tasks 4-5 build
  atomic deployment and public-content enforcement; Tasks 6-7 verify local and
  live behavior; Task 8 creates the public non-fork repository; Task 9 restores
  official-only ComfyUI tracking; Task 10 audits both repositories and runtime.
- Placeholder scan: repository names, paths, workflow names, commands, API
  payloads, environment variables, failure gates, and expected results are
  explicit; no deferred implementation markers remain.
- Type consistency: `WORKFLOW_NAMES`, `validate_comfy_root`,
  `publish_workflows`, and `audit_paths` retain the same signatures across
  their implementation and test tasks.
