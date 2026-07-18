import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

from game_asset_api.animation_contracts import parse_animation_request
from game_asset_api.animation_pipeline import AnimationProcessor


class FakeClient:
    def __init__(self, history: dict[str, object]) -> None:
        self.history = history
        self.graphs: list[dict[str, object]] = []

    async def submit(self, graph: dict[str, object]) -> str:
        self.graphs.append(graph)
        return "prompt-animation"

    async def wait_for_prompt(
        self, prompt_id: str, timeout_seconds: float = 1800
    ) -> dict[str, object]:
        return self.history


@pytest.mark.asyncio
async def test_animation_processor_publishes_a_complete_eight_frame_bundle(
    tmp_path: Path,
) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    records = _write_generated_records(tmp_path, "job-id", range(8))
    client = FakeClient(_history(records))
    processor = AnimationProcessor(tmp_path, client)

    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)
    prompt_id, generated = await processor.generate(request, "job-id", prepared, plan)
    stabilized = processor.stabilize(request, plan, generated)
    composited = processor.composite(plan, stabilized, prepared)
    staged = processor.export(request, "job-id", plan, stabilized, composited)
    artifacts = processor.validate_and_publish(request, "job-id", staged)

    final = tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    assert prompt_id == "prompt-animation"
    assert len(client.graphs) == 1
    assert client.graphs[0]["3"]["inputs"]["image"] == "game_assets/job-id/production/reference.png"
    assert [path.name for path in generated] == [
        f"source_{index:05d}_.png" for index in range(8)
    ]
    assert artifacts.metadata == final / "animation.json"
    assert all(path.parent == final / "frames" for path in artifacts.frames)
    assert artifacts.metadata.is_file()
    assert artifacts.sprite_frames.is_file()
    assert not (final.parent / ".production_action.tmp").exists()
    assert not (tmp_path / "output" / ".animation_work" / "job-id").exists()


@pytest.mark.asyncio
async def test_animation_processor_publishes_when_project_root_is_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("runtime")
    request = _write_valid_runtime_inputs(root, frame_count=8)
    records = _write_generated_records(root, "job-id", range(8))
    processor = AnimationProcessor(root, FakeClient(_history(records)))

    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)
    _, generated = await processor.generate(request, "job-id", prepared, plan)
    stabilized = processor.stabilize(request, plan, generated)
    composited = processor.composite(plan, stabilized, prepared)
    staged = processor.export(request, "job-id", plan, stabilized, composited)
    artifacts = processor.validate_and_publish(request, "job-id", staged)

    final = root / "output" / "game_assets" / "job-id" / "production_action"
    assert artifacts.metadata == final / "animation.json"
    assert artifacts.metadata.is_file()


@pytest.mark.asyncio
async def test_animation_processor_sorts_comfy_output_records_before_stabilizing(
    tmp_path: Path,
) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    records = _write_generated_records(tmp_path, "job-id", reversed(range(8)))
    processor = AnimationProcessor(tmp_path, FakeClient(_history(records)))
    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)

    _, generated = await processor.generate(request, "job-id", prepared, plan)

    assert [path.name for path in generated] == [
        f"source_{index:05d}_.png" for index in range(8)
    ]


@pytest.mark.asyncio
async def test_animation_processor_rejects_bad_records_and_cleanup_keeps_final_output(
    tmp_path: Path,
) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    temporary = tmp_path / "output" / ".animation_work" / "job-id"
    temporary.mkdir(parents=True)
    processor = AnimationProcessor(
        tmp_path,
        FakeClient(_history([{"filename": "../escape.png", "subfolder": "", "type": "output"}])),
    )
    prepared = processor.validate_inputs(request, "job-id")
    plan = processor.plan_motion(request, "job-id", prepared)
    final = tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    final.mkdir(parents=True)
    (final / "published.txt").write_text("keep", encoding="utf-8")
    staged = final.parent / ".production_action.tmp"
    staged.mkdir()

    with pytest.raises(ValueError, match="image record filename must be a relative path"):
        await processor.generate(request, "job-id", prepared, plan)
    processor.cleanup("job-id")

    assert not temporary.exists()
    assert not staged.exists()
    assert (final / "published.txt").read_text(encoding="utf-8") == "keep"


def test_validate_and_publish_rejects_corrupt_frame_before_rename(tmp_path: Path) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    processor = AnimationProcessor(tmp_path, FakeClient({}))
    staged = _write_staged_bundle(processor, request, "job-id")
    staged.frames[3].write_bytes(b"not a PNG")

    with pytest.raises(ValueError, match="production frame is unreadable"):
        processor.validate_and_publish(request, "job-id", staged)

    final = tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    assert not final.exists()
    assert staged.metadata.parent.exists()


def test_validate_and_publish_rejects_non_relative_metadata_artifacts(tmp_path: Path) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    processor = AnimationProcessor(tmp_path, FakeClient({}))
    staged = _write_staged_bundle(processor, request, "job-id")
    metadata = json.loads(staged.metadata.read_text(encoding="utf-8"))
    metadata["artifacts"]["spritesheet"] = "../spritesheet.png"
    staged.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="animation artifact path must be relative"):
        processor.validate_and_publish(request, "job-id", staged)

    assert not (
        tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    ).exists()


def test_validate_and_publish_preserves_an_existing_final_bundle(tmp_path: Path) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    processor = AnimationProcessor(tmp_path, FakeClient({}))
    staged = _write_staged_bundle(processor, request, "job-id")
    final = tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    final.mkdir(parents=True)
    preserved = final / "published.txt"
    preserved.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="production action is already published"):
        processor.validate_and_publish(request, "job-id", staged)

    assert preserved.read_text(encoding="utf-8") == "keep"
    assert staged.metadata.parent.exists()


@pytest.mark.parametrize(
    ("artifact", "content", "message"),
    [
        ("spritesheet", b"not a PNG", "production spritesheet is unreadable"),
        ("preview", b"not a GIF", "production preview is unreadable"),
        ("sprite_frames", b"", "production sprite frames are unreadable"),
    ],
)
def test_validate_and_publish_rejects_corrupt_required_artifacts_before_rename(
    tmp_path: Path, artifact: str, content: bytes, message: str
) -> None:
    request = _write_valid_runtime_inputs(tmp_path, frame_count=8)
    processor = AnimationProcessor(tmp_path, FakeClient({}))
    staged = _write_staged_bundle(processor, request, "job-id")
    getattr(staged, artifact).write_bytes(content)

    with pytest.raises(ValueError, match=message):
        processor.validate_and_publish(request, "job-id", staged)

    assert not (
        tmp_path / "output" / "game_assets" / "job-id" / "production_action"
    ).exists()


def test_cli_accepts_two_frame_preflight_without_changing_http_contract() -> None:
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--root",
            "runtime",
            "--character-image",
            "characters/cultivator.png",
            "--weapon",
            "weapons/sword.json",
            "--asset-name",
            "cultivator_attack",
            "--character-prompt",
            "cultivator",
            "--frame-count",
            "2",
            "--sprite-size",
            "64",
            "--seed",
            "42",
            "--job-id",
            "preflight",
            "--base-url",
            "http://127.0.0.1:8188",
        ]
    )

    request = runner.build_request(args)

    assert request.frame_count == 2
    assert request.seed == 42
    with pytest.raises(Exception, match="frame_count must be one of 8, 12, 16"):
        parse_animation_request(
            {
                "asset_name": "cultivator_attack",
                "character_image": "characters/cultivator.png",
                "character_prompt": "cultivator",
                "weapon": "weapons/sword.json",
                "action": "sword_attack",
                "frame_count": 2,
            }
        )


@pytest.mark.asyncio
async def test_cli_cleans_up_when_a_processor_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    cleaned: list[str] = []

    class FakeComfyClient:
        def __init__(self, _base_url: str) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    class FailingProcessor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def validate_inputs(self, *args: object) -> object:
            raise RuntimeError("stage failed")

        def cleanup(self, job_id: str) -> None:
            cleaned.append(job_id)

    monkeypatch.setattr(runner, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(runner, "AnimationProcessor", FailingProcessor)
    arguments = runner.parse_args(
        [
            "--root",
            "runtime",
            "--character-image",
            "characters/cultivator.png",
            "--weapon",
            "weapons/sword.json",
            "--asset-name",
            "cultivator_attack",
            "--character-prompt",
            "cultivator",
            "--job-id",
            "failed-job",
        ]
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        await runner.run(arguments)

    assert cleaned == ["failed-job"]


def _write_valid_runtime_inputs(root: Path, frame_count: int):
    character = root / "input" / "characters" / "cultivator.png"
    weapon = root / "input" / "weapons" / "sword.png"
    descriptor = weapon.with_name("sword.json")
    character.parent.mkdir(parents=True)
    weapon.parent.mkdir(parents=True)
    Image.new("RGBA", (128, 128), (20, 120, 220, 255)).save(character)
    blade = Image.new("RGBA", (64, 16), (0, 0, 0, 0))
    for x in range(4, 60):
        blade.putpixel((x, 8), (220, 230, 240, 255))
    blade.save(weapon)
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image": "sword.png",
                "grip": [4 / 63, 8 / 15],
                "tip": [59 / 63, 8 / 15],
                "default_layer": "behind_character",
            }
        ),
        encoding="utf-8",
    )
    return parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": frame_count,
            "sprite_size": 64,
            "seed": 42,
        }
    )


def _write_generated_records(
    root: Path, job_id: str, indices: object
) -> list[dict[str, str]]:
    source = root / "output" / ".animation_work" / job_id
    source.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        path = source / f"source_{index:05d}_.png"
        _write_rgba_character(path, index % 2)
        records.append(
            {
                "filename": path.name,
                "subfolder": f".animation_work/{job_id}",
                "type": "output",
            }
        )
    return records


def _write_rgba_character(path: Path, x_offset: int) -> None:
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    for y in range(80, 417):
        for x in range(180 + x_offset, 333 + x_offset):
            image.putpixel((x, y), (20, 120, 220, 255))
    image.save(path)


def _history(records: list[dict[str, str]]) -> dict[str, object]:
    return {
        "status": {"status_str": "success", "messages": []},
        "outputs": {"73": {"images": records}},
    }


def _write_staged_bundle(
    processor: AnimationProcessor, request, job_id: str
):
    records = _write_generated_records(processor.project_root, job_id, range(8))
    generated = tuple(
        processor.output_root / record["subfolder"] / record["filename"]
        for record in records
    )
    prepared = processor.validate_inputs(request, job_id)
    plan = processor.plan_motion(request, job_id, prepared)
    stabilized = processor.stabilize(request, plan, generated)
    composited = processor.composite(plan, stabilized, prepared)
    return processor.export(request, job_id, plan, stabilized, composited)


def _load_runner():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_production_animation.py"
    spec = importlib.util.spec_from_file_location("run_production_animation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
