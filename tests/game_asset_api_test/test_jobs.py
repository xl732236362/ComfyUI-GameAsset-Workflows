import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest
from aiohttp import web
from PIL import Image

from game_asset_api.animation_contracts import AnimationRequest, parse_animation_request
from game_asset_api.comfy_client import ComfyClient, image_records
from game_asset_api.contracts import AssetRequest, parse_asset_request
from game_asset_api.jobs import JobKind, JobRunner, JobStatus, JobStore, _resolve_output_image


def _request(**overrides) -> AssetRequest:
    return parse_asset_request(
        {
            "character_prompt": "armored knight",
            "action_prompt": "walking in place",
            "frame_count": 2,
            "sprite_size": 64,
            "seed": 7,
            **overrides,
        }
    )


def _animation_request(**overrides) -> AnimationRequest:
    return parse_animation_request(
        {
            "asset_name": "cultivator_attack",
            "character_image": "characters/cultivator.png",
            "character_prompt": "cultivator",
            "weapon": "weapons/sword.json",
            "action": "sword_attack",
            "frame_count": 8,
            "sprite_size": 64,
            "seed": 7,
            **overrides,
        }
    )


def _history(images_by_node: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    return {
        "status": {"status_str": "success", "messages": []},
        "outputs": {
            node_id: {"images": images} for node_id, images in images_by_node.items()
        },
    }


def _write_png(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), color).save(path)


def test_job_store_persists_typed_job_and_allows_first_transition(tmp_path):
    store = JobStore(tmp_path / "jobs")

    job = store.create(_request())
    transitioned = store.transition(job.id, JobStatus.GENERATING_CHARACTER)

    assert transitioned.status is JobStatus.GENERATING_CHARACTER
    manifest_path = tmp_path / "jobs" / job.id / "job.json"
    assert manifest_path.is_file()
    assert not manifest_path.with_suffix(".json.tmp").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["id"] == job.id
    assert manifest["request"]["character_prompt"] == "armored knight"
    assert store.read(job.id).status is JobStatus.GENERATING_CHARACTER


def test_job_store_rejects_transition_out_of_terminal_state(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create(_request())
    store.transition(job.id, JobStatus.FAILED, error="no model")

    with pytest.raises(ValueError, match="terminal"):
        store.transition(job.id, JobStatus.POSTPROCESSING)


def test_job_store_rejects_illegal_nonterminal_transition(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create(_request())

    with pytest.raises(ValueError, match="invalid job transition"):
        store.transition(job.id, JobStatus.POSTPROCESSING)


def test_job_store_round_trips_a_production_animation_request(tmp_path):
    store = JobStore(tmp_path / "jobs")
    request = _animation_request()

    job = store.create_animation(request)
    transitioned = store.transition(job.id, JobStatus.VALIDATING_INPUTS)

    assert transitioned.kind is JobKind.PRODUCTION_ANIMATION
    assert transitioned.request == request
    manifest = json.loads((tmp_path / "jobs" / job.id / "job.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "production_animation"
    assert store.read(job.id) == transitioned


def test_job_store_assigns_a_durable_seed_to_an_animation_request(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr("game_asset_api.jobs.secrets.randbits", lambda _: 1234)

    job = store.create_animation(replace(_animation_request(), seed=None))

    assert job.request.seed == 1234
    manifest = json.loads((tmp_path / "jobs" / job.id / "job.json").read_text(encoding="utf-8"))
    assert manifest["request"]["seed"] == 1234


def test_job_store_reads_existing_manifests_without_a_kind_as_game_assets(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create(_request())
    manifest_path = tmp_path / "jobs" / job.id / "job.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("kind")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert store.read(job.id).kind is JobKind.GAME_ASSET


@dataclass(frozen=True)
class _FakeArtifacts:
    frames: tuple[Path, ...]
    spritesheet: Path
    sprite_frames: Path
    metadata: Path
    preview: Path


class _FakeAnimationProcessor:
    def __init__(self, output_root: Path, failing_stage: str | None = None) -> None:
        self.output_root = output_root
        self.failing_stage = failing_stage
        self.cleanups: list[str] = []
        self.stages: list[tuple[str, str]] = []
        self.active_generations = 0
        self.max_active_generations = 0
        self.store: JobStore | None = None
        self.current_job_id: str | None = None

    def _stage(self, name: str, job_id: str) -> None:
        assert self.store is not None
        self.current_job_id = job_id
        self.stages.append((name, self.store.read(job_id).status.value))
        if self.failing_stage == name:
            raise RuntimeError(f"{name} failed")

    def validate_inputs(self, request, job_id):
        self._stage("validate_inputs", job_id)
        return "prepared"

    def plan_motion(self, request, job_id, prepared):
        self._stage("plan_motion", job_id)
        return "plan"

    async def generate(self, request, job_id, prepared, plan, on_prompt=None):
        self._stage("generate", job_id)
        self.active_generations += 1
        self.max_active_generations = max(self.max_active_generations, self.active_generations)
        await asyncio.sleep(0)
        self.active_generations -= 1
        if on_prompt is not None:
            on_prompt(0, "animation-prompt-000")
            on_prompt(1, "animation-prompt-001")
        return "animation-prompt", "generated"

    def stabilize(self, request, plan, generated):
        assert self.current_job_id is not None
        self._stage("stabilize", self.current_job_id)
        return "stabilized"

    def composite(self, plan, stabilized, prepared):
        assert self.current_job_id is not None
        self._stage("composite", self.current_job_id)
        return "composited"

    def export(self, request, job_id, plan, stabilized, composited):
        self._stage("export", job_id)
        return "staged"

    def validate_and_publish(self, request, job_id, staged):
        self._stage("validate_and_publish", job_id)
        base = self.output_root / job_id / "production_action"
        return _FakeArtifacts(
            (base / "frames" / "000.png",),
            base / "spritesheet.png",
            base / "sprite_frames.tres",
            base / "animation.json",
            base / "preview.gif",
        )

    def cleanup(self, job_id):
        self.cleanups.append(job_id)


@pytest.mark.asyncio
async def test_job_runner_processes_animation_stages_in_order_and_serializes_generation(tmp_path):
    processor = _FakeAnimationProcessor(tmp_path / "output" / "game_assets")
    runner = JobRunner(tmp_path, _FakeComfyClient({}, {}), animation_processor=processor)
    processor.store = runner.store
    runner.start()
    first = runner.enqueue_animation(_animation_request())
    second = runner.enqueue_animation(_animation_request(asset_name="second_attack"))
    await runner.join()
    await runner.stop()

    assert runner.store.read(first.id).status is JobStatus.COMPLETED
    assert runner.store.read(second.id).status is JobStatus.COMPLETED
    assert processor.max_active_generations == 1
    assert processor.stages == [
        ("validate_inputs", "validating_inputs"),
        ("plan_motion", "motion_planning"),
        ("generate", "temporal_generation"),
        ("stabilize", "character_stabilization"),
        ("composite", "weapon_composite"),
        ("export", "godot_export"),
        ("validate_and_publish", "validating_outputs"),
    ] * 2
    completed = runner.store.read(first.id)
    assert completed.prompt_ids == {
        "animation_000": "animation-prompt-000",
        "animation_001": "animation-prompt-001",
    }
    assert completed.outputs == {
        "frame_000": f"/assets/{first.id}/production_action/frames/000.png",
        "spritesheet": f"/assets/{first.id}/production_action/spritesheet.png",
        "sprite_frames": f"/assets/{first.id}/production_action/sprite_frames.tres",
        "metadata": f"/assets/{first.id}/production_action/animation.json",
        "preview": f"/assets/{first.id}/production_action/preview.gif",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing_stage", "expected_status"),
    [
        ("validate_inputs", JobStatus.VALIDATING_INPUTS),
        ("plan_motion", JobStatus.MOTION_PLANNING),
        ("generate", JobStatus.TEMPORAL_GENERATION),
        ("stabilize", JobStatus.CHARACTER_STABILIZATION),
        ("composite", JobStatus.WEAPON_COMPOSITE),
        ("export", JobStatus.GODOT_EXPORT),
        ("validate_and_publish", JobStatus.VALIDATING_OUTPUTS),
    ],
)
async def test_job_runner_cleans_production_work_after_each_stage_failure(
    tmp_path, failing_stage, expected_status
):
    processor = _FakeAnimationProcessor(
        tmp_path / "output" / "game_assets", failing_stage=failing_stage
    )
    runner = JobRunner(tmp_path, _FakeComfyClient({}, {}), animation_processor=processor)
    processor.store = runner.store
    runner.start()
    job = runner.enqueue_animation(_animation_request())
    await runner.join()
    await runner.stop()

    failed = runner.store.read(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.failed_stage == expected_status.value
    assert failed.error == f"{failing_stage} failed"
    assert processor.cleanups == [job.id]


class _CleanupFailingProcessor(_FakeAnimationProcessor):
    def __init__(self, output_root: Path) -> None:
        super().__init__(output_root)
        self.fail_first_job = True

    def validate_inputs(self, request, job_id):
        self._stage("validate_inputs", job_id)
        if self.fail_first_job:
            self.fail_first_job = False
            raise RuntimeError("generation failed")
        return "prepared"

    def cleanup(self, job_id):
        super().cleanup(job_id)
        raise RuntimeError("cleanup failed")


@pytest.mark.asyncio
async def test_job_runner_persists_generation_failure_and_continues_after_cleanup_failure(
    tmp_path,
):
    processor = _CleanupFailingProcessor(tmp_path / "output" / "game_assets")
    runner = JobRunner(tmp_path, _FakeComfyClient({}, {}), animation_processor=processor)
    processor.store = runner.store
    runner.start()
    first = runner.enqueue_animation(_animation_request())
    second = runner.enqueue_animation(_animation_request(asset_name="second_attack"))

    try:
        await asyncio.wait_for(runner.join(), timeout=0.2)
    finally:
        worker = runner._worker
        if worker is not None and worker.done():
            worker.exception()
        elif worker is not None:
            await runner.stop()

    failed = runner.store.read(first.id)
    assert failed.status is JobStatus.FAILED
    assert failed.failed_stage == JobStatus.VALIDATING_INPUTS.value
    assert failed.error == "generation failed"
    assert runner.store.read(second.id).status is JobStatus.COMPLETED
    assert processor.cleanups == [first.id]


@pytest.mark.asyncio
async def test_comfy_client_submits_prompt_and_reads_completed_history(aiohttp_client):
    submitted: list[dict[str, object]] = []
    polls = 0

    async def submit(request):
        submitted.append(await request.json())
        return web.json_response({"prompt_id": "prompt-1"})

    async def history(request):
        nonlocal polls
        polls += 1
        return web.json_response(
            {"prompt-1": _history({"16": [{"filename": "frame.png", "subfolder": "game_assets/job", "type": "output"}]})}
        )

    app = web.Application()
    app.router.add_post("/prompt", submit)
    app.router.add_get("/history/{prompt_id}", history)
    server = await aiohttp_client(app)
    client = ComfyClient(base_url=str(server.make_url("")), session=server.session)

    prompt_id = await client.submit({"1": {"class_type": "SaveImage"}})
    history_entry = await client.wait_for_prompt(prompt_id)

    assert prompt_id == "prompt-1"
    assert submitted == [{"prompt": {"1": {"class_type": "SaveImage"}}}]
    assert polls == 1
    assert image_records(history_entry, "16") == [
        {"filename": "frame.png", "subfolder": "game_assets/job", "type": "output"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload"),
    [(500, {}), (200, {}), (200, {"prompt_id": 42})],
)
async def test_comfy_client_submit_rejects_error_or_malformed_responses(
    aiohttp_client, status, payload
):
    async def submit(request):
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_post("/prompt", submit)
    server = await aiohttp_client(app)
    client = ComfyClient(base_url=str(server.make_url("")), session=server.session)

    with pytest.raises(RuntimeError):
        await client.submit({"1": {"class_type": "SaveImage"}})


@pytest.mark.asyncio
async def test_comfy_client_raises_for_execution_error(aiohttp_client):
    async def history(request):
        return web.json_response(
            {
                "prompt-1": {
                    "status": {
                        "messages": [["execution_error", {"exception_message": "model path leaked\\ntraceback"}]]
                    },
                    "outputs": {},
                }
            }
        )

    app = web.Application()
    app.router.add_get("/history/{prompt_id}", history)
    server = await aiohttp_client(app)
    client = ComfyClient(base_url=str(server.make_url("")), session=server.session)

    with pytest.raises(RuntimeError, match="model path leaked") as error:
        await client.wait_for_prompt("prompt-1")

    assert "\n" not in str(error.value)


@pytest.mark.asyncio
async def test_comfy_client_times_out_incomplete_history_without_real_sleep(
    aiohttp_client,
):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    polls = 0

    async def history(request):
        nonlocal polls
        polls += 1
        return web.json_response({})

    async def advance_clock(delay):
        clock.now += delay

    app = web.Application()
    app.router.add_get("/history/{prompt_id}", history)
    server = await aiohttp_client(app)
    client = ComfyClient(
        base_url=str(server.make_url("")),
        session=server.session,
        poll_interval_seconds=0.5,
        clock=clock,
        sleep=advance_clock,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await client.wait_for_prompt("prompt-1", timeout_seconds=1)

    assert polls == 2


@pytest.mark.asyncio
async def test_comfy_client_times_out_a_hanging_history_request(aiohttp_client):
    started = asyncio.Event()
    release = asyncio.Event()

    async def history(request):
        started.set()
        await release.wait()
        return web.json_response({})

    app = web.Application()
    app.router.add_get("/history/{prompt_id}", history)
    server = await aiohttp_client(app)
    client = ComfyClient(base_url=str(server.make_url("")), session=server.session)

    try:
        with pytest.raises(TimeoutError, match="timed out"):
            await client.wait_for_prompt("prompt-1", timeout_seconds=0.01)
    finally:
        release.set()

    assert started.is_set()


def test_image_records_rejects_missing_or_malformed_images():
    with pytest.raises(ValueError, match="images"):
        image_records({"outputs": {"16": {}}}, "16")
    with pytest.raises(ValueError, match="images"):
        image_records({"outputs": {"16": {"images": ["not-an-object"]}}}, "16")


def test_output_image_rejects_drive_relative_records(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        _resolve_output_image(
            tmp_path,
            {"type": "output", "filename": "C:escape.png", "subfolder": ""},
        )


class _FakeComfyClient:
    def __init__(self, character_history, action_history):
        self.character_history = character_history
        self.action_history = action_history
        self.submitted: list[dict[str, dict[str, object]]] = []
        self.active_waits = 0
        self.max_active_waits = 0

    async def submit(self, graph):
        self.submitted.append(graph)
        return f"prompt-{len(self.submitted)}"

    async def wait_for_prompt(self, prompt_id, timeout_seconds=1800):
        self.active_waits += 1
        self.max_active_waits = max(self.max_active_waits, self.active_waits)
        await asyncio.sleep(0)
        self.active_waits -= 1
        if len(self.submitted) % 2:
            return self.character_history
        return self.action_history


@pytest.mark.asyncio
async def test_job_runner_uses_project_root_for_comfy_and_asset_paths(tmp_path):
    source_root = tmp_path / "output" / "game_assets" / "source"
    _write_png(source_root / "reference_rgb.png", (255, 0, 0, 255))
    _write_png(source_root / "character.png", (0, 255, 0, 128))
    _write_png(source_root / "frame-000.png", (0, 0, 255, 255))
    _write_png(source_root / "frame-001.png", (255, 255, 0, 255))
    client = _FakeComfyClient(
        _history(
            {
                "11": [{"filename": "reference_rgb.png", "subfolder": "game_assets/source", "type": "output"}],
                "12": [{"filename": "character.png", "subfolder": "game_assets/source", "type": "output"}],
            }
        ),
        _history(
            {
                "16": [
                    {"filename": "frame-001.png", "subfolder": "game_assets/source", "type": "output"},
                    {"filename": "frame-000.png", "subfolder": "game_assets/source", "type": "output"},
                ]
            }
        ),
    )
    runner = JobRunner(project_root=tmp_path, client=client)
    runner.start()
    job = runner.enqueue(_request())
    await runner.join()
    await runner.stop()

    asset_root = tmp_path / "output" / "game_assets" / job.id
    assert json.loads((asset_root / "job.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert (asset_root / "character.png").is_file()
    assert (asset_root / "spritesheet.png").is_file()
    assert (asset_root / "metadata.json").is_file()
    assert (asset_root / "frames" / "000.png").is_file()
    assert (tmp_path / "input" / "game_assets" / job.id / "reference.png").is_file()
    assert not (tmp_path / "output" / "output").exists()
    assert not (tmp_path / "output" / "input").exists()


@pytest.mark.asyncio
async def test_job_runner_serializes_jobs_and_writes_relative_assets(tmp_path):
    project_root = tmp_path
    jobs_root = project_root / "output" / "game_assets"
    output_root = tmp_path / "output" / "game_assets" / "source"
    _write_png(output_root / "reference_rgb.png", (255, 0, 0, 255))
    _write_png(output_root / "character.png", (0, 255, 0, 128))
    source_colors = [
        (255, 0, 0, 255),
        (255, 255, 0, 255),
        (0, 255, 0, 255),
        (0, 255, 255, 255),
        (0, 0, 255, 255),
    ]
    for index, color in enumerate(source_colors):
        _write_png(output_root / f"source-{index:03d}.png", color)

    character_history = _history(
        {
            "11": [{"filename": "reference_rgb.png", "subfolder": "game_assets/source", "type": "output"}],
            "12": [{"filename": "character.png", "subfolder": "game_assets/source", "type": "output"}],
        }
    )
    action_history = _history(
        {
            "16": [
                {"filename": f"source-{index:03d}.png", "subfolder": "game_assets/source", "type": "output"}
                for index in reversed(range(5))
            ]
        }
    )
    client = _FakeComfyClient(character_history, action_history)
    runner = JobRunner(project_root, client)
    runner.start()
    first = runner.enqueue(_request())
    second = runner.enqueue(_request(character_prompt="mage"))
    await runner.join()
    await runner.stop()

    completed = runner.store.read(first.id)
    assert completed.status is JobStatus.COMPLETED
    assert runner.store.read(second.id).status is JobStatus.COMPLETED
    assert client.max_active_waits == 1
    assert (tmp_path / "input" / "game_assets" / first.id / "reference.png").is_file()
    frames = sorted((jobs_root / first.id / "frames").glob("*.png"))
    assert [frame.name for frame in frames] == ["000.png", "001.png"]
    with Image.open(frames[0]) as first_frame:
        assert first_frame.getpixel((0, 0)) == source_colors[0]
    with Image.open(frames[1]) as second_frame:
        assert second_frame.getpixel((0, 0)) == source_colors[4]
    for path in (
        jobs_root / first.id / "character.png",
        jobs_root / first.id / "spritesheet.png",
    ):
        with Image.open(path) as image:
            assert image.mode == "RGBA"

    metadata = json.loads(
        (jobs_root / first.id / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_frame_count"] == 5
    assert metadata["source_frame_indices"] == [0, 4]
    assert metadata["output_frames"] == ["000.png", "001.png"]
    assert metadata["sprite_sheet"] == {"filename": "spritesheet.png", "columns": 2, "rows": 1}
    assert completed.outputs["spritesheet"] == f"/assets/{first.id}/spritesheet.png"
    assert all(value.startswith(f"/assets/{first.id}/") for value in completed.outputs.values())


@pytest.mark.asyncio
async def test_job_runner_rejects_traversal_output_records_before_copying(tmp_path):
    project_root = tmp_path
    character_history = _history(
        {
            "11": [{"filename": "../outside.png", "subfolder": "game_assets", "type": "output"}],
            "12": [{"filename": "character.png", "subfolder": "game_assets", "type": "output"}],
        }
    )
    client = _FakeComfyClient(character_history, _history({"16": []}))
    runner = JobRunner(project_root, client)
    runner.start()
    job = runner.enqueue(_request())
    await runner.join()
    await runner.stop()

    failed = runner.store.read(job.id)
    assert failed.status is JobStatus.FAILED
    assert "relative" in (failed.error or "") or "output" in (failed.error or "")
    assert not (tmp_path / "input" / "game_assets" / job.id / "reference.png").exists()
