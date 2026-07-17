from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from game_asset_api.deployment import (
    WORKFLOW_NAMES,
    publish_workflows,
    validate_comfy_root,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WORKFLOW_NAMES = (
    "pixel_character_design_api.json",
    "pixel_character_action_api.json",
    "pose_controlled_pixel_action_api.json",
    "video_wan2_2_5B_ti2v.json",
    "wan2_2_5b_dual_balanced.json",
)


def _write_sources(directory: Path) -> dict[str, bytes]:
    directory.mkdir(parents=True)
    payloads = {}
    for name in WORKFLOW_NAMES:
        payload = (json.dumps({"prompt": {"name": name}}) + "\n").encode()
        (directory / name).write_bytes(payload)
        payloads[name] = payload
    return payloads


def _valid_comfy_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# test\n", encoding="utf-8")
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test python")
    return python


def _load_deploy_script():
    script = ROOT / "scripts" / "deploy.py"
    spec = importlib.util.spec_from_file_location("deploy_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_names_are_complete_and_ordered():
    assert WORKFLOW_NAMES == EXPECTED_WORKFLOW_NAMES


def test_source_helper_writes_valid_prompt_objects(tmp_path):
    source = tmp_path / "source"

    payloads = _write_sources(source)

    assert tuple(payloads) == WORKFLOW_NAMES
    for name, payload in payloads.items():
        parsed = json.loads(payload.decode("utf-8"))
        assert isinstance(parsed, dict)
        assert isinstance(parsed["prompt"], dict)
        assert (source / name).read_bytes() == payload


def test_comfy_root_helper_writes_required_files(tmp_path):
    python = _valid_comfy_root(tmp_path)

    assert (tmp_path / "main.py").is_file()
    assert python == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert python.is_file()


def test_publish_workflows_validates_and_copies_all_five(tmp_path):
    source = tmp_path / "source"
    payloads = _write_sources(source)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)

    published = publish_workflows(source, comfy_root)

    destination = comfy_root / "user" / "default" / "workflows"
    assert tuple(path.name for path in published) == WORKFLOW_NAMES
    assert tuple(path.parent for path in published) == (destination,) * 5
    assert {path.name: path.read_bytes() for path in published} == payloads


def test_publish_workflows_accepts_the_repository_api_and_ui_formats(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)

    published = publish_workflows(ROOT / "workflows", comfy_root)

    assert tuple(path.name for path in published) == WORKFLOW_NAMES
    for path in published:
        assert path.read_bytes() == (ROOT / "workflows" / path.name).read_bytes()


@pytest.mark.parametrize(
    "replacement",
    [
        b"\xff",
        b"not json",
        b"[]",
        b'{"prompt": []}',
        b"{}",
        b'{"nodes": []}',
    ],
    ids=(
        "invalid-utf8",
        "invalid-json",
        "non-object",
        "non-object-prompt",
        "unrecognized-object",
        "incomplete-ui-object",
    ),
)
def test_publish_workflows_rejects_invalid_source_before_creating_destination(
    tmp_path, replacement
):
    source = tmp_path / "source"
    _write_sources(source)
    invalid_name = WORKFLOW_NAMES[3]
    (source / invalid_name).write_bytes(replacement)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    destination = comfy_root / "user" / "default" / "workflows"

    with pytest.raises(
        ValueError, match=rf"workflow JSON.*{invalid_name}"
    ):
        publish_workflows(source, comfy_root)

    assert not destination.exists()


def test_publish_workflows_rejects_missing_source_before_creating_destination(
    tmp_path,
):
    source = tmp_path / "source"
    _write_sources(source)
    missing_name = WORKFLOW_NAMES[-1]
    (source / missing_name).unlink()
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    destination = comfy_root / "user" / "default" / "workflows"

    with pytest.raises(
        ValueError, match=rf"workflow JSON.*{missing_name}"
    ):
        publish_workflows(source, comfy_root)

    assert not destination.exists()


def test_publish_workflows_rejects_invalid_json_without_replacing_destination(
    tmp_path,
):
    source = tmp_path / "source"
    _write_sources(source)
    invalid_name = WORKFLOW_NAMES[3]
    (source / invalid_name).write_text("not json", encoding="utf-8")
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    destination = comfy_root / "user" / "default" / "workflows"
    destination.mkdir(parents=True)
    old_payloads = {}
    for name in WORKFLOW_NAMES:
        payload = f"old:{name}".encode()
        (destination / name).write_bytes(payload)
        old_payloads[name] = payload

    with pytest.raises(
        ValueError, match=rf"workflow JSON.*{invalid_name}"
    ):
        publish_workflows(source, comfy_root)

    assert {
        path.name: path.read_bytes() for path in destination.glob("*.json")
    } == old_payloads


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


def test_publish_workflows_removes_temporary_file_after_replace_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    _write_sources(source)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    first_name = WORKFLOW_NAMES[0]
    temporary = (
        comfy_root
        / "user"
        / "default"
        / "workflows"
        / f"{first_name}.tmp"
    )

    def fail_replace(source_path, target_path):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        publish_workflows(source, comfy_root)

    assert not temporary.exists()


def test_validate_comfy_root_requires_main_and_python(tmp_path):
    with pytest.raises(ValueError, match="main.py"):
        validate_comfy_root(tmp_path)
    (tmp_path / "main.py").write_text("# test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="python.exe"):
        validate_comfy_root(tmp_path)
    python = _valid_comfy_root(tmp_path)
    assert validate_comfy_root(tmp_path) == (tmp_path.resolve(), python.resolve())


def test_validate_comfy_root_expands_and_resolves_user_path(tmp_path, monkeypatch):
    comfy_root = tmp_path / "ComfyUI"
    python = _valid_comfy_root(comfy_root)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    root, selected_python = validate_comfy_root(Path("~/ComfyUI"))

    assert root == comfy_root.resolve()
    assert selected_python == python.resolve()


def test_deploy_arguments_have_stable_defaults_and_skip_flags(tmp_path):
    module = _load_deploy_script()

    defaults = module.parse_arguments(["--comfy-root", str(tmp_path)])
    skipped = module.parse_arguments(
        [
            "--comfy-root",
            str(tmp_path),
            "--base-url",
            "http://localhost:9000",
            "--skip-nodes",
            "--skip-models",
            "--skip-discovery",
            "--skip-smoke",
        ]
    )

    assert defaults.comfy_root == tmp_path
    assert defaults.base_url == "http://127.0.0.1:8188"
    assert not defaults.skip_nodes
    assert not defaults.skip_models
    assert not defaults.skip_discovery
    assert not defaults.skip_smoke
    assert skipped.base_url == "http://localhost:9000"
    assert skipped.skip_nodes
    assert skipped.skip_models
    assert skipped.skip_discovery
    assert skipped.skip_smoke


def test_deploy_runs_all_operations_in_order_with_explicit_root_and_python(
    tmp_path, monkeypatch
):
    module = _load_deploy_script()
    comfy_root = tmp_path / "ComfyUI"
    python = _valid_comfy_root(comfy_root)
    reference = comfy_root / "input" / "example.png"
    reference.parent.mkdir()
    reference.write_bytes(b"reference")
    events = []

    def fake_publish(source, selected_root):
        events.append(("publish", source, selected_root))
        return ()

    def fake_run(command, check):
        events.append(("run", command, check))

    def fake_discover(base_url):
        events.append(("discover", base_url))
        return {}

    monkeypatch.setattr(module, "publish_workflows", fake_publish)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "discover_object_info", fake_discover)
    arguments = module.parse_arguments(
        ["--comfy-root", str(comfy_root), "--base-url", "http://localhost:9000/"]
    )

    module.deploy(arguments)

    root = comfy_root.resolve()
    selected_python = str(python.resolve())
    assert events[0] == ("publish", ROOT / "workflows", root)
    assert events[1] == (
        "run",
        [
            selected_python,
            str(ROOT / "scripts" / "install_pose_workflow_nodes.py"),
            "--root",
            str(root),
            "--python",
            selected_python,
        ],
        True,
    )
    assert events[2] == (
        "run",
        [
            selected_python,
            str(ROOT / "scripts" / "install_game_asset_models.py"),
            "--root",
            str(root),
        ],
        True,
    )
    assert events[3] == ("discover", "http://localhost:9000/")
    assert events[4] == (
        "run",
        [
            selected_python,
            str(ROOT / "scripts" / "run_pose_controlled_action.py"),
            "--root",
            str(root),
            "--reference",
            str(reference.resolve()),
            "--job-id",
            "deployment-smoke",
            "--character-prompt",
            "pixel art knight",
            "--frame-count",
            "2",
            "--sprite-size",
            "64",
            "--base-url",
            "http://localhost:9000/",
        ],
        True,
    )


def test_deploy_skip_flags_avoid_all_optional_side_effects(tmp_path, monkeypatch):
    module = _load_deploy_script()
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    events = []

    def fake_publish(source, selected_root):
        events.append((source, selected_root))
        return ()

    monkeypatch.setattr(module, "publish_workflows", fake_publish)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must be skipped"),
    )
    monkeypatch.setattr(
        module,
        "discover_object_info",
        lambda *args, **kwargs: pytest.fail("discovery must be skipped"),
    )
    arguments = module.parse_arguments(
        [
            "--comfy-root",
            str(comfy_root),
            "--skip-nodes",
            "--skip-models",
            "--skip-discovery",
            "--skip-smoke",
        ]
    )

    module.deploy(arguments)

    assert events == [(ROOT / "workflows", comfy_root.resolve())]


def test_deploy_requires_the_official_example_for_smoke(tmp_path, monkeypatch):
    module = _load_deploy_script()
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    monkeypatch.setattr(module, "publish_workflows", lambda source, root: ())
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("runner must not start"),
    )
    arguments = module.parse_arguments(
        [
            "--comfy-root",
            str(comfy_root),
            "--skip-nodes",
            "--skip-models",
            "--skip-discovery",
        ]
    )

    with pytest.raises(ValueError, match=r"input.*example\.png"):
        module.deploy(arguments)


class _Response:
    def __init__(self, status: int, payload: bytes):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._payload


def test_discovery_requires_successful_object_response(monkeypatch):
    module = _load_deploy_script()
    requested = []

    def fake_urlopen(url, timeout):
        requested.append((url, timeout))
        return _Response(200, b'{"LoadImage": {}}')

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    discovered = module.discover_object_info("http://localhost:9000/")

    assert discovered == {"LoadImage": {}}
    assert requested == [("http://localhost:9000/object_info", 30)]


@pytest.mark.parametrize(
    ("status", "payload"),
    [(500, b"{}"), (200, b"not json"), (200, b"[]")],
    ids=("http-failure", "invalid-json", "non-object"),
)
def test_discovery_rejects_failed_or_non_object_responses(
    monkeypatch, status, payload
):
    module = _load_deploy_script()
    monkeypatch.setattr(
        module, "urlopen", lambda url, timeout: _Response(status, payload)
    )

    with pytest.raises(RuntimeError, match="object_info"):
        module.discover_object_info("http://localhost:9000")


def test_powershell_wrapper_is_the_stable_deployment_entrypoint():
    wrapper = (ROOT / "deploy.ps1").read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in wrapper
    assert "[string]$ComfyRoot" in wrapper
    assert "[string]$BaseUrl = 'http://127.0.0.1:8188'" in wrapper
    assert "$ErrorActionPreference = 'Stop'" in wrapper
    assert "Test-Path -LiteralPath $python -PathType Leaf" in wrapper
    assert "'scripts\\deploy.py'" in wrapper
    assert "--comfy-root $ComfyRoot --base-url $BaseUrl" in wrapper
    assert "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }" in wrapper
