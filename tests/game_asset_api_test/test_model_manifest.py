"""Tests for the verified game-asset model manifest."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import importlib.util

import pytest

import game_asset_api.model_manifest as model_manifest
from game_asset_api.model_manifest import MODEL_SPECS, ModelSpec, install, verify_file


def test_manifest_contains_verified_model_specs():
    specs_by_filename = {spec.filename: spec for spec in MODEL_SPECS}

    assert specs_by_filename["sd_xl_base_1.0.safetensors"].size == 6_938_078_334
    assert (
        specs_by_filename["pixel-art-xl.safetensors"].sha256
        == "4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7"
    )
    assert (
        specs_by_filename["BiRefNet-general-epoch_244.safetensors"].size
        == 444_473_596
    )
    assert specs_by_filename["OpenPoseXL2.safetensors"] == ModelSpec(
        filename="OpenPoseXL2.safetensors",
        relative_dir="controlnet",
        url="https://hf-mirror.com/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors",
        size=5_004_167_829,
        sha256="5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92",
    )
    assert specs_by_filename[
        "ip-adapter-plus_sdxl_vit-h.safetensors"
    ] == ModelSpec(
        filename="ip-adapter-plus_sdxl_vit-h.safetensors",
        relative_dir="ipadapter",
        url="https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        size=847_517_512,
        sha256="3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581",
    )
    assert specs_by_filename[
        "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    ] == ModelSpec(
        filename="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        relative_dir="clip_vision",
        url="https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        size=2_528_373_448,
        sha256="6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
    )
    assert "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" not in specs_by_filename


def test_model_spec_is_immutable_and_builds_destination(tmp_path):
    spec = ModelSpec("model.safetensors", "checkpoints", "https://example.invalid/model", 1, "0" * 64)

    assert spec.destination(tmp_path) == tmp_path / "models" / "checkpoints" / "model.safetensors"
    with pytest.raises(FrozenInstanceError):
        spec.filename = "other.safetensors"


def test_verify_file_rejects_matching_size_with_wrong_hash(tmp_path):
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"12345")

    assert verify_file(candidate, 5, "0" * 64) is False


def test_install_returns_an_already_verified_destination(tmp_path):
    payload = b"verified model"
    spec = ModelSpec(
        "model.safetensors",
        "checkpoints",
        "https://example.invalid/model",
        len(payload),
        sha256(payload).hexdigest(),
    )
    destination = spec.destination(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    assert install(spec, tmp_path) == destination


def _spec_for_payload(payload: bytes) -> ModelSpec:
    return ModelSpec(
        "model.safetensors",
        "checkpoints",
        "https://example.invalid/model",
        len(payload),
        sha256(payload).hexdigest(),
    )


def test_install_promotes_a_verified_complete_partial_without_invoking_curl(tmp_path, monkeypatch):
    payload = b"complete partial"
    spec = _spec_for_payload(payload)
    destination = spec.destination(tmp_path)
    partial = destination.with_name(f"{destination.name}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload)

    def fail_if_curl_runs(*_args, **_kwargs):
        pytest.fail("curl must not run when the partial file is already verified")

    monkeypatch.setattr(model_manifest.subprocess, "run", fail_if_curl_runs)

    assert install(spec, tmp_path) == destination
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_install_restarts_an_invalid_complete_partial_before_curl(tmp_path, monkeypatch):
    payload = b"replacement data"
    spec = _spec_for_payload(payload)
    destination = spec.destination(tmp_path)
    partial = destination.with_name(f"{destination.name}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * len(payload))
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command == ["curl.exe", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="curl 8.0.0")

        assert not partial.exists()
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(model_manifest.subprocess, "run", fake_run)

    assert install(spec, tmp_path) == destination
    assert destination.read_bytes() == payload
    assert calls[0] == ["curl.exe", "--version"]
    assert "--continue-at" in calls[1]


@pytest.mark.parametrize(
    ("version", "expects_retry_all_errors"),
    [("7.71.0", True), ("7.70.0", False)],
)
def test_install_adds_retry_all_errors_only_for_supported_curl(tmp_path, monkeypatch, version, expects_retry_all_errors):
    payload = b"downloaded data"
    spec = _spec_for_payload(payload)
    partial = spec.destination(tmp_path).with_name(f"{spec.filename}.part")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command == ["curl.exe", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"curl {version} test")

        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(model_manifest.subprocess, "run", fake_run)

    install(spec, tmp_path)

    assert commands[0] == ["curl.exe", "--version"]
    assert ("--retry-all-errors" in commands[1]) is expects_retry_all_errors


def test_install_omits_retry_all_errors_when_curl_version_cannot_be_parsed(tmp_path, monkeypatch):
    payload = b"downloaded data"
    spec = _spec_for_payload(payload)
    partial = spec.destination(tmp_path).with_name(f"{spec.filename}.part")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command == ["curl.exe", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="unsupported curl build")

        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(model_manifest.subprocess, "run", fake_run)

    install(spec, tmp_path)

    assert commands[0] == ["curl.exe", "--version"]
    assert "--retry-all-errors" not in commands[1]


def test_installer_script_imports_manifest_when_launched_by_path():
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "install_game_asset_models.py"
    code = (
        "from pathlib import Path; import runpy, sys; "
        f"root = Path({str(project_root)!r}); script = Path({str(script)!r}); "
        "sys.path = [str(script.parent)] + [entry for entry in sys.path[1:] "
        "if entry and Path(entry).resolve() != root]; "
        "runpy.run_path(str(script), run_name='installer_test')"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_installer_accepts_an_explicit_deployment_root(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "install_game_asset_models.py"
    spec = importlib.util.spec_from_file_location("install_game_asset_models", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    monkeypatch.setattr(module, "MODEL_SPECS", ("model-spec",))
    monkeypatch.setattr(module, "install", lambda model, root: calls.append((model, root)))

    module.main(["--root", str(tmp_path)])

    assert calls == [("model-spec", tmp_path)]
