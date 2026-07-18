from __future__ import annotations

import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

import game_asset_api.deployment as deployment_module
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
    "production_animation_api.json",
)
_API_WORKFLOW_NAMES = frozenset(
    {
        "pixel_character_design_api.json",
        "pixel_character_action_api.json",
        "pose_controlled_pixel_action_api.json",
        "production_animation_api.json",
    }
)


def _write_sources(directory: Path) -> dict[str, bytes]:
    directory.mkdir(parents=True)
    payloads = {}
    for name in WORKFLOW_NAMES:
        if name in _API_WORKFLOW_NAMES:
            workflow = {"prompt": {}}
        else:
            workflow = {"nodes": [{"id": 1, "type": "KSampler"}], "links": []}
        payload = (json.dumps(workflow) + "\n").encode()
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


def _object_info_fixture() -> dict:
    node_types = set()
    for name in WORKFLOW_NAMES:
        workflow = json.loads((ROOT / "workflows" / name).read_text(encoding="utf-8"))
        if "prompt" in workflow:
            node_types.update(node["class_type"] for node in workflow["prompt"].values())
        else:
            node_types.update(
                node["type"]
                for node in workflow["nodes"]
                if node["type"] != "MarkdownNote"
            )
    object_info = {
        node_type: {"input": {"required": {}, "optional": {}}}
        for node_type in node_types
    }

    def choices(*values):
        return [list(values), {"default": values[0]}]

    required_choices = {
        "CheckpointLoaderSimple": {
            "ckpt_name": choices("sd_xl_base_1.0.safetensors")
        },
        "LoraLoader": {"lora_name": choices("pixel-art-xl.safetensors")},
        "LoadBackgroundRemovalModel": {
            "bg_removal_name": [
                "COMBO",
                {"options": ["BiRefNet-general-epoch_244.safetensors"]},
            ]
        },
        "UNETLoader": {
            "unet_name": choices("wan2.2_ti2v_5B_fp16.safetensors"),
            "weight_dtype": choices("default"),
        },
        "CLIPLoader": {
            "clip_name": choices("umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "type": choices("wan"),
        },
        "VAELoader": {"vae_name": choices("wan2.2_vae.safetensors")},
        "IPAdapterModelLoader": {
            "ipadapter_file": choices(
                "ip-adapter-plus_sdxl_vit-h.safetensors"
            )
        },
        "CLIPVisionLoader": {
            "clip_name": choices(
                "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            )
        },
        "IPAdapterAdvanced": {
            "weight_type": choices("style transfer"),
            "combine_embeds": choices("concat"),
            "embeds_scaling": choices("V only"),
        },
        "ControlNetLoader": {
            "control_net_name": choices("OpenPoseXL2.safetensors")
        },
        "ImageScale": {
            "upscale_method": choices("nearest-exact"),
            "crop": choices("disabled"),
        },
        "KSampler": {
            "sampler_name": choices("dpmpp_2m", "uni_pc"),
            "scheduler": choices("karras", "simple"),
        },
        "SaveVideo": {
            "format": choices("mp4"),
            "codec": choices("h264"),
        },
    }
    for node_type, inputs in required_choices.items():
        object_info[node_type]["input"]["required"].update(inputs)
    object_info["CLIPLoader"]["input"]["optional"]["device"] = [
        "COMBO",
        {"options": ["default"]},
    ]
    return object_info


def _validate_object_info(object_info: dict) -> None:
    validator = getattr(deployment_module, "validate_object_info", None)
    assert validator is not None, "validate_object_info must be implemented"
    validator(object_info, ROOT / "workflows")


def test_workflow_names_are_complete_and_ordered():
    assert WORKFLOW_NAMES == EXPECTED_WORKFLOW_NAMES


def test_production_discovery_does_not_require_animatediff_nodes():
    assert ("ADE_LoadAnimateDiffModel", "model_name") not in deployment_module._DISCOVERY_INPUTS
    assert ("ADE_UseEvolvedSampling", "beta_schedule") not in deployment_module._DISCOVERY_INPUTS


def test_publish_workflows_validates_and_copies_all_workflows(tmp_path):
    source = tmp_path / "source"
    payloads = _write_sources(source)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)

    published = publish_workflows(source, comfy_root)

    destination = comfy_root / "user" / "default" / "workflows"
    assert tuple(path.name for path in published) == WORKFLOW_NAMES
    assert tuple(path.parent for path in published) == (destination,) * len(WORKFLOW_NAMES)
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


@pytest.mark.parametrize(
    ("invalid_name", "replacement"),
    [
        (
            WORKFLOW_NAMES[0],
            b'{"prompt":{"1":null}}',
        ),
        (
            WORKFLOW_NAMES[3],
            b'{"nodes":[null],"links":[]}',
        ),
        (
            WORKFLOW_NAMES[0],
            b'{"nodes":[{"id":1,"type":"KSampler"}],"links":[]}',
        ),
        (
            WORKFLOW_NAMES[3],
            b'{"prompt":{}}',
        ),
        (
            WORKFLOW_NAMES[3],
            b'{"nodes":[],"links":[]}',
        ),
        (
            WORKFLOW_NAMES[0],
            b'{"prompt":{"1":{"class_type":"KSampler","inputs":{"cfg":NaN}}}}',
        ),
    ],
    ids=(
        "invalid-api-node",
        "invalid-ui-node",
        "ui-shell-for-api-name",
        "api-shell-for-ui-name",
        "empty-ui-nodes",
        "non-finite-number",
    ),
)
def test_publish_workflows_rejects_filename_or_structure_mismatches_before_writes(
    tmp_path, invalid_name, replacement
):
    source = tmp_path / "source"
    _write_sources(source)
    (source / invalid_name).write_bytes(replacement)
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    destination = comfy_root / "user" / "default" / "workflows"

    with pytest.raises(ValueError, match=rf"workflow JSON.*{invalid_name}"):
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


def test_validate_object_info_reports_every_missing_node_for_empty_discovery():
    expected_nodes = set(_object_info_fixture())

    with pytest.raises(ValueError, match="object_info") as captured:
        _validate_object_info({})

    message = str(captured.value)
    for node_type in expected_nodes:
        assert node_type in message
    assert "MarkdownNote" not in message


def test_validate_object_info_rejects_a_missing_required_node():
    object_info = _object_info_fixture()
    object_info.pop("ControlNetApplyAdvanced")

    with pytest.raises(ValueError, match="ControlNetApplyAdvanced"):
        _validate_object_info(object_info)


def test_validate_object_info_rejects_a_missing_loader_filename():
    object_info = _object_info_fixture()
    object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = [
        ["another.safetensors"]
    ]

    with pytest.raises(ValueError, match="CheckpointLoaderSimple.ckpt_name") as captured:
        _validate_object_info(object_info)

    assert "sd_xl_base_1.0.safetensors" in str(captured.value)


def test_validate_object_info_rejects_missing_ipadapter_weight_type():
    object_info = _object_info_fixture()
    object_info["IPAdapterAdvanced"]["input"]["required"]["weight_type"] = [
        ["linear"]
    ]

    with pytest.raises(ValueError, match="IPAdapterAdvanced.weight_type") as captured:
        _validate_object_info(object_info)

    assert "style transfer" in str(captured.value)


def test_validate_object_info_reports_all_missing_image_scale_options():
    object_info = _object_info_fixture()
    image_scale = object_info["ImageScale"]["input"]["required"]
    image_scale["upscale_method"] = [["bilinear"]]
    image_scale["crop"] = [["center"]]

    with pytest.raises(ValueError, match="object_info") as captured:
        _validate_object_info(object_info)

    message = str(captured.value)
    assert "ImageScale.upscale_method" in message
    assert "nearest-exact" in message
    assert "ImageScale.crop" in message
    assert "disabled" in message


@pytest.mark.parametrize(
    ("node_type", "input_name", "missing_value"),
    [
        ("KSampler", "sampler_name", "dpmpp_2m"),
        ("KSampler", "scheduler", "simple"),
        ("SaveVideo", "format", "mp4"),
        ("SaveVideo", "codec", "h264"),
    ],
)
def test_validate_object_info_rejects_missing_runtime_options(
    node_type, input_name, missing_value
):
    object_info = _object_info_fixture()
    schema = object_info[node_type]["input"]["required"][input_name]
    schema[0].remove(missing_value)

    with pytest.raises(ValueError, match=rf"{node_type}.{input_name}") as captured:
        _validate_object_info(object_info)

    assert missing_value in str(captured.value)


def test_validate_object_info_accepts_all_required_nodes_and_options():
    assert _validate_object_info(_object_info_fixture()) is None


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
    events = []

    def fake_publish(source, selected_root):
        events.append(("publish", source, selected_root))
        return ()

    def fake_run(command, check):
        events.append(("run", command, check))

    def fake_discover(base_url):
        events.append(("discover", base_url))
        return {"discovered": {}}

    def fake_validate(object_info, source):
        events.append(("validate", object_info, source))

    monkeypatch.setattr(module, "publish_workflows", fake_publish)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "discover_object_info", fake_discover)
    monkeypatch.setattr(module, "validate_object_info", fake_validate, raising=False)
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
        "validate",
        {"discovered": {}},
        ROOT / "workflows",
    )
    assert events[5] == (
        "run",
        [
            selected_python,
            str(ROOT / "scripts" / "run_production_animation.py"),
            "--root",
            str(root),
            "--character-image",
            "game_assets/deployment-smoke/character.png",
            "--weapon",
            "game_assets/deployment-smoke/sword.json",
            "--asset-name",
            "deployment-smoke",
            "--job-id",
            "deployment-smoke",
            "--character-prompt",
            "pixel art, full body unarmed white-robed cultivator, fixed side view, locked camera, consistent identity, both hands empty",
            "--frame-count",
            "2",
            "--sprite-size",
            "64",
            "--base-url",
            "http://localhost:9000/",
        ],
        True,
    )
    descriptor = root / "input" / "game_assets" / "deployment-smoke" / "sword.json"
    weapon = descriptor.with_name("sword.png")
    assert json.loads(descriptor.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "image": "sword.png",
        "grip": [0.125, 0.5],
        "tip": [0.875, 0.5],
        "default_layer": "behind_character",
    }
    with Image.open(weapon) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
    with Image.open(descriptor.with_name("character.png")) as image:
        assert image.mode == "RGB"
        assert image.size == (512, 512)
        assert image.getpixel((0, 0)) == (108, 85, 44)


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


def test_deploy_writes_a_smoke_character_without_the_official_example(tmp_path, monkeypatch):
    module = _load_deploy_script()
    comfy_root = tmp_path / "ComfyUI"
    _valid_comfy_root(comfy_root)
    monkeypatch.setattr(module, "publish_workflows", lambda source, root: ())
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    arguments = module.parse_arguments(
        [
            "--comfy-root",
            str(comfy_root),
            "--skip-nodes",
            "--skip-models",
            "--skip-discovery",
        ]
    )

    module.deploy(arguments)

    character = comfy_root / "input" / "game_assets" / "deployment-smoke" / "character.png"
    with Image.open(character) as image:
        assert image.mode == "RGB"
        assert image.size == (512, 512)


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
    [(200, b"not json"), (200, b"[]")],
    ids=("invalid-json", "non-object"),
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


def test_discovery_closes_http_error_and_preserves_status_and_reason(monkeypatch):
    module = _load_deploy_script()
    error = HTTPError(
        "http://localhost:9000/object_info",
        503,
        "Service Unavailable",
        None,
        BytesIO(b"unavailable"),
    )
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda url, timeout: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError, match="object_info") as captured:
        module.discover_object_info("http://localhost:9000")

    assert "503" in str(captured.value)
    assert "Service Unavailable" in str(captured.value)
    assert error.fp.closed


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
