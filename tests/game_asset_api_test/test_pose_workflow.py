import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image, ImageChops
import pytest

from game_asset_api.contracts import parse_asset_request


MODULE_NAME = "game_asset_api.pose_workflow"
POSE_SEQUENCE_MODULE_NAME = "game_asset_api.pose_sequence"
POSE_RUNNER_MODULE_NAME = "game_asset_api.pose_runner"
ROOT = Path(__file__).resolve().parents[2]


def _pose_workflow_module():
    assert importlib.util.find_spec(MODULE_NAME) is not None
    return importlib.import_module(MODULE_NAME)


def _pose_sequence_module():
    assert importlib.util.find_spec(POSE_SEQUENCE_MODULE_NAME) is not None
    return importlib.import_module(POSE_SEQUENCE_MODULE_NAME)


def _pose_runner_module():
    assert importlib.util.find_spec(POSE_RUNNER_MODULE_NAME) is not None
    return importlib.import_module(POSE_RUNNER_MODULE_NAME)


def _single_node(graph, class_type):
    nodes = [node for node in graph.values() if node["class_type"] == class_type]
    assert len(nodes) == 1
    return nodes[0]


def test_pose_workflow_wires_sdxl_pose_identity_and_alpha_pipeline():
    module = _pose_workflow_module()
    request = parse_asset_request(
        {
            "character_prompt": "white-haired xianxia swordsman",
            "action_prompt": "forward sword slash",
            "camera": "side",
            "seed": 42,
            "sprite_size": 128,
        }
    )

    graph = module.build_pose_controlled_workflow(
        request,
        "job-1",
        frame_index=0,
        controlnet_strength=0.9,
        ipadapter_weight=0.65,
    )

    assert _single_node(graph, "CheckpointLoaderSimple")["inputs"] == {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    }
    assert _single_node(graph, "LoraLoader")["inputs"]["lora_name"] == (
        "pixel-art-xl.safetensors"
    )
    assert _single_node(graph, "IPAdapterModelLoader")["inputs"] == {
        "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
    }
    assert _single_node(graph, "CLIPVisionLoader")["inputs"] == {
        "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    }

    ipadapter = _single_node(graph, "IPAdapterAdvanced")
    assert ipadapter["inputs"]["weight"] == 0.65
    assert ipadapter["inputs"]["weight_type"] == "style transfer"
    assert ipadapter["inputs"]["clip_vision"] == ["5", 0]
    assert ipadapter["inputs"]["image"] == ["3", 0]

    controlnet = _single_node(graph, "ControlNetLoader")
    assert controlnet["inputs"] == {"control_net_name": "OpenPoseXL2.safetensors"}
    control_apply = _single_node(graph, "ControlNetApplyAdvanced")
    assert control_apply["inputs"]["image"] == ["9", 0]
    assert control_apply["inputs"]["strength"] == 0.9

    sampler = _single_node(graph, "KSampler")
    assert sampler["inputs"]["model"] == ["6", 0]
    assert sampler["inputs"]["positive"] == ["11", 0]
    assert sampler["inputs"]["negative"] == ["11", 1]
    assert sampler["inputs"]["seed"] == 42

    assert _single_node(graph, "LoadBackgroundRemovalModel")["inputs"] == {
        "bg_removal_name": "BiRefNet-general-epoch_244.safetensors"
    }
    assert _single_node(graph, "InvertMask")["inputs"] == {"mask": ["16", 0]}
    assert _single_node(graph, "JoinImageWithAlpha")["inputs"]["alpha"] == ["17", 0]
    scale = _single_node(graph, "ImageScale")
    assert scale["inputs"]["upscale_method"] == "nearest-exact"
    assert scale["inputs"]["width"] == 128
    assert scale["inputs"]["height"] == 128
    assert _single_node(graph, "SaveImage")["inputs"]["images"] == ["19", 0]


def test_pose_workflow_exposes_ipadapter_weight_type_for_tuning():
    module = _pose_workflow_module()
    request = parse_asset_request(
        {
            "character_prompt": "xianxia swordsman",
            "action_prompt": "sword slash",
        }
    )

    graph = module.build_pose_controlled_workflow(
        request,
        "job-1",
        frame_index=0,
        ipadapter_weight_type="style transfer",
    )

    assert _single_node(graph, "IPAdapterAdvanced")["inputs"]["weight_type"] == (
        "style transfer"
    )


def test_export_script_writes_prompt_wrapped_pose_workflow(tmp_path):
    temporary_root = tmp_path / "exporter_repo"
    script = temporary_root / "scripts" / "export_pose_controlled_workflow.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / script.name, script)
    shutil.copytree(ROOT / "game_asset_api", temporary_root / "game_asset_api")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=temporary_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (temporary_root / "user").exists()
    output = temporary_root / "workflows" / "pose_controlled_pixel_action_api.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    graph = payload["prompt"]
    load_images = [
        node["inputs"]["image"]
        for node in graph.values()
        if node["class_type"] == "LoadImage"
    ]
    assert load_images == [
        "game_assets/example-pose-job/reference.png",
        "game_assets/example-pose-job/poses/000.png",
    ]


@pytest.mark.parametrize("frame_count", [2, 8])
def test_sword_attack_pose_sequence_writes_unique_chronological_pngs(
    tmp_path, frame_count
):
    module = _pose_sequence_module()

    paths = module.write_sword_attack_pose_sequence(tmp_path, frame_count)

    assert [path.name for path in paths] == [
        f"{index:03d}.png" for index in range(frame_count)
    ]
    payloads = []
    for path in paths:
        payloads.append(path.read_bytes())
        with Image.open(path) as image:
            assert image.size == (512, 512)
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (0, 0, 0)
            assert ImageChops.difference(image, Image.new("RGB", image.size)).getbbox()
    assert len(set(payloads)) == frame_count


def test_pose_sequence_script_writes_selected_frame_count(tmp_path):
    script = ROOT / "scripts" / "create_sword_attack_pose_sequence.py"
    output_directory = tmp_path / "poses"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_directory),
            "--frame-count",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert [path.name for path in sorted(output_directory.glob("*.png"))] == [
        "000.png",
        "001.png",
    ]


def test_pose_runner_script_parses_reference_and_tuning_arguments(tmp_path):
    script = ROOT / "scripts" / "run_pose_controlled_action.py"
    spec = importlib.util.spec_from_file_location("run_pose_controlled_action", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = tmp_path / "reference.png"

    arguments = module.parse_arguments(
        [
            "--root",
            str(tmp_path),
            "--reference",
            str(reference),
            "--job-id",
            "xianxia-smoke",
            "--character-prompt",
            "white-haired xianxia swordsman",
            "--frame-count",
            "2",
            "--sprite-size",
            "64",
            "--seed",
            "42",
            "--controlnet-strength",
            "0.85",
            "--ipadapter-weight",
            "0.7",
            "--ipadapter-weight-type",
            "style transfer",
        ]
    )

    assert arguments.root == tmp_path
    assert arguments.reference == reference
    assert arguments.job_id == "xianxia-smoke"
    assert arguments.frame_count == 2
    assert arguments.sprite_size == 64
    assert arguments.seed == 42
    assert arguments.controlnet_strength == 0.85
    assert arguments.ipadapter_weight == 0.7
    assert arguments.ipadapter_weight_type == "style transfer"
    assert arguments.base_url == "http://127.0.0.1:8188"


def test_pose_runner_script_defaults_to_style_transfer(tmp_path):
    script = ROOT / "scripts" / "run_pose_controlled_action.py"
    spec = importlib.util.spec_from_file_location("run_pose_controlled_action", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arguments = module.parse_arguments(
        [
            "--root",
            str(tmp_path),
            "--reference",
            str(tmp_path / "reference.png"),
            "--job-id",
            "xianxia-smoke",
            "--character-prompt",
            "white-haired xianxia swordsman",
        ]
    )

    assert arguments.ipadapter_weight_type == "style transfer"


@pytest.mark.parametrize("sprite_size", [64, 96, 128, 256])
def test_pose_runner_script_accepts_supported_sprite_sizes(tmp_path, sprite_size):
    script = ROOT / "scripts" / "run_pose_controlled_action.py"
    spec = importlib.util.spec_from_file_location("run_pose_controlled_action", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arguments = module.parse_arguments(
        [
            "--root",
            str(tmp_path),
            "--reference",
            str(tmp_path / "reference.png"),
            "--job-id",
            "xianxia-smoke",
            "--character-prompt",
            "white-haired xianxia swordsman",
            "--sprite-size",
            str(sprite_size),
        ]
    )

    assert arguments.sprite_size == sprite_size


@pytest.mark.asyncio
async def test_pose_runner_script_forwards_ipadapter_weight_type(
    tmp_path, monkeypatch
):
    script = ROOT / "scripts" / "run_pose_controlled_action.py"
    spec = importlib.util.spec_from_file_location("run_pose_controlled_action", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    class _Client:
        def __init__(self, base_url):
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def _run_pose(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "ComfyClient", _Client)
    monkeypatch.setattr(module, "run_pose_controlled_action", _run_pose)
    arguments = SimpleNamespace(
        root=tmp_path,
        reference=tmp_path / "reference.png",
        job_id="xianxia-smoke",
        character_prompt="white-haired xianxia swordsman",
        action_prompt="horizontal sword slash",
        camera="side",
        frame_count=2,
        sprite_size=64,
        seed=42,
        controlnet_strength=0.9,
        ipadapter_weight=0.65,
        ipadapter_weight_type="style transfer",
        base_url="http://127.0.0.1:8188",
        timeout_seconds=1800,
    )

    await module._run(arguments)

    assert captured["ipadapter_weight_type"] == "style transfer"


class _SequentialClient:
    def __init__(self, histories):
        self.histories = histories
        self.submitted = []
        self.active_waits = 0
        self.max_active_waits = 0

    async def submit(self, graph):
        self.submitted.append(graph)
        return f"prompt-{len(self.submitted)}"

    async def wait_for_prompt(self, prompt_id, timeout_seconds=1800):
        self.active_waits += 1
        self.max_active_waits = max(self.max_active_waits, self.active_waits)
        history = self.histories[int(prompt_id.removeprefix("prompt-")) - 1]
        self.active_waits -= 1
        return history


def _alpha_frame(path, color):
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(16, 48):
        for y in range(8, 56):
            image.putpixel((x, y), (*color, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@pytest.mark.asyncio
async def test_pose_runner_submits_frames_sequentially_and_writes_rgba_sheet(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    source_directory = tmp_path / "output" / "game_assets" / "source"
    _alpha_frame(source_directory / "frame-000.png", (255, 0, 0))
    _alpha_frame(source_directory / "frame-001.png", (0, 255, 0))
    histories = [
        {
            "outputs": {
                "20": {
                    "images": [
                        {
                            "filename": f"frame-{index:03d}.png",
                            "subfolder": "game_assets/source",
                            "type": "output",
                        }
                    ]
                }
            }
        }
        for index in range(2)
    ]
    client = _SequentialClient(histories)
    request = parse_asset_request(
        {
            "character_prompt": "white-haired xianxia swordsman",
            "action_prompt": "forward sword slash",
            "camera": "side",
            "frame_count": 2,
            "sprite_size": 64,
            "seed": 42,
        }
    )

    result = await module.run_pose_controlled_action(
        tmp_path,
        client,
        request,
        "pose-job",
        reference,
    )

    assert client.max_active_waits == 1
    assert len(client.submitted) == 2
    assert client.submitted[0]["9"]["inputs"]["image"].endswith("poses/000.png")
    assert client.submitted[1]["9"]["inputs"]["image"].endswith("poses/001.png")
    assert client.submitted[0]["6"]["inputs"]["weight_type"] == "style transfer"
    assert [path.name for path in result.frames] == ["000.png", "001.png"]
    with Image.open(result.sprite_sheet) as sheet:
        assert sheet.mode == "RGBA"
        assert sheet.size == (128, 64)
        assert sheet.getpixel((20, 20)) == (255, 0, 0, 255)
        assert sheet.getpixel((84, 20)) == (0, 255, 0, 255)


@pytest.mark.asyncio
async def test_pose_runner_removes_stale_artifacts_before_rerun(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    source_directory = tmp_path / "output" / "game_assets" / "source"
    _alpha_frame(source_directory / "frame-000.png", (255, 0, 0))
    _alpha_frame(source_directory / "frame-001.png", (0, 255, 0))
    histories = [
        {
            "outputs": {
                "20": {
                    "images": [
                        {
                            "filename": f"frame-{index:03d}.png",
                            "subfolder": "game_assets/source",
                            "type": "output",
                        }
                    ]
                }
            }
        }
        for index in range(2)
    ]
    client = _SequentialClient(histories)
    pose_directory = tmp_path / "input" / "game_assets" / "pose-job" / "poses"
    pose_directory.mkdir(parents=True)
    (pose_directory / "007.png").write_bytes(b"stale pose")
    output_directory = (
        tmp_path / "output" / "game_assets" / "pose-job" / "pose_action"
    )
    stale_frame = output_directory / "frames" / "007.png"
    _alpha_frame(stale_frame, (0, 0, 255))
    (output_directory / "spritesheet.png").write_bytes(b"stale sheet")
    request = parse_asset_request(
        {
            "character_prompt": "white-haired xianxia swordsman",
            "action_prompt": "forward sword slash",
            "frame_count": 2,
            "sprite_size": 64,
        }
    )

    await module.run_pose_controlled_action(
        tmp_path, client, request, "pose-job", reference
    )

    assert [path.name for path in sorted(pose_directory.glob("*.png"))] == [
        "000.png",
        "001.png",
    ]
    assert [
        path.name
        for path in sorted((output_directory / "frames").glob("*.png"))
    ] == ["000.png", "001.png"]
    with Image.open(output_directory / "spritesheet.png") as sheet:
        assert sheet.size == (128, 64)


@pytest.mark.asyncio
async def test_pose_runner_removes_previous_sheet_before_failed_rerun(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    output_directory = (
        tmp_path / "output" / "game_assets" / "pose-job" / "pose_action"
    )
    stale_frame = output_directory / "frames" / "007.png"
    _alpha_frame(stale_frame, (0, 0, 255))
    (output_directory / "spritesheet.png").write_bytes(b"stale sheet")
    request = parse_asset_request(
        {
            "character_prompt": "white-haired xianxia swordsman",
            "action_prompt": "forward sword slash",
            "frame_count": 2,
            "sprite_size": 64,
        }
    )

    class _FailingClient:
        async def submit(self, graph):
            return "prompt-1"

        async def wait_for_prompt(self, prompt_id, timeout_seconds=1800):
            raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        await module.run_pose_controlled_action(
            tmp_path, _FailingClient(), request, "pose-job", reference
        )

    assert not (output_directory / "spritesheet.png").exists()
    assert not stale_frame.exists()


@pytest.mark.asyncio
async def test_pose_runner_rejects_more_than_one_output_for_a_frame(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    source = tmp_path / "output" / "game_assets" / "source" / "frame.png"
    _alpha_frame(source, (255, 0, 0))
    record = {
        "filename": "frame.png",
        "subfolder": "game_assets/source",
        "type": "output",
    }
    history = {"outputs": {"20": {"images": [record, record]}}}
    client = _SequentialClient([history, history])
    request = parse_asset_request(
        {
            "character_prompt": "xianxia swordsman",
            "action_prompt": "sword slash",
            "frame_count": 2,
            "sprite_size": 64,
        }
    )

    with pytest.raises(ValueError, match="output count"):
        await module.run_pose_controlled_action(
            tmp_path, client, request, "pose-job", reference
        )


@pytest.mark.asyncio
async def test_pose_runner_rejects_frame_without_transparent_background(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    source = tmp_path / "output" / "game_assets" / "source" / "opaque.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(source)
    history = {
        "outputs": {
            "20": {
                "images": [
                    {
                        "filename": "opaque.png",
                        "subfolder": "game_assets/source",
                        "type": "output",
                    }
                ]
            }
        }
    }
    client = _SequentialClient([history, history])
    request = parse_asset_request(
        {
            "character_prompt": "xianxia swordsman",
            "action_prompt": "sword slash",
            "frame_count": 2,
            "sprite_size": 64,
        }
    )

    with pytest.raises(ValueError, match="alpha"):
        await module.run_pose_controlled_action(
            tmp_path, client, request, "pose-job", reference
        )


@pytest.mark.asyncio
async def test_pose_runner_rejects_frame_with_wrong_sprite_dimensions(tmp_path):
    module = _pose_runner_module()
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(reference)
    source = tmp_path / "output" / "game_assets" / "source" / "small.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(8, 24):
        for y in range(4, 28):
            image.putpixel((x, y), (255, 0, 0, 255))
    image.save(source)
    history = {
        "outputs": {
            "20": {
                "images": [
                    {
                        "filename": "small.png",
                        "subfolder": "game_assets/source",
                        "type": "output",
                    }
                ]
            }
        }
    }
    client = _SequentialClient([history, history])
    request = parse_asset_request(
        {
            "character_prompt": "xianxia swordsman",
            "action_prompt": "sword slash",
            "frame_count": 2,
            "sprite_size": 64,
        }
    )

    with pytest.raises(ValueError, match="dimensions"):
        await module.run_pose_controlled_action(
            tmp_path, client, request, "pose-job", reference
        )
