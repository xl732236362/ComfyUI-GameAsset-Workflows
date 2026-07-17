from pathlib import Path
import runpy
import subprocess
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

import pytest

import game_asset_api.app as app_module
from game_asset_api.app import create_app
from game_asset_api.contracts import AssetRequest, parse_asset_request
from game_asset_api.jobs import JobStatus, JobStore


class _FakeRunner:
    def __init__(self, project_root: Path) -> None:
        self.jobs_root = project_root / "output" / "game_assets"
        self.store = JobStore(self.jobs_root)
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    def enqueue(self, request: AssetRequest):
        return self.store.create(request)


class _FakeClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def _request() -> AssetRequest:
    return parse_asset_request(
        {"character_prompt": "armored knight", "action_prompt": "walk"}
    )


def _create_directory_junction(link: Path, target: Path) -> None:
    junction = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if junction.returncode != 0:
        pytest.skip("directory junctions are unavailable in this environment")


@pytest.mark.asyncio
async def test_post_asset_request_queues_valid_request_and_rejects_invalid_camera(
    aiohttp_client, tmp_path
):
    runner = _FakeRunner(tmp_path)
    client = await aiohttp_client(create_app(runner))

    created = await client.post(
        "/v1/game-assets",
        json={"character_prompt": "armored knight", "action_prompt": "walk"},
    )
    rejected = await client.post(
        "/v1/game-assets",
        json={
            "character_prompt": "armored knight",
            "action_prompt": "walk",
            "camera": "custom",
        },
    )
    malformed = await client.post(
        "/v1/game-assets", data="{", headers={"Content-Type": "application/json"}
    )

    created_payload = await created.json()
    assert created.status == 202
    assert created_payload["status"] == "queued"
    assert runner.store.read(created_payload["job_id"]).status is JobStatus.QUEUED
    assert rejected.status == 400
    assert await rejected.json() == {"error": "camera_prompt is required when camera is custom"}
    assert malformed.status == 400
    assert "error" in await malformed.json()


@pytest.mark.asyncio
async def test_job_status_exposes_completed_assets_in_frame_order(aiohttp_client, tmp_path):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    for status in (
        JobStatus.GENERATING_CHARACTER,
        JobStatus.GENERATING_ACTION,
        JobStatus.POSTPROCESSING,
    ):
        runner.store.transition(job.id, status)
    runner.store.transition(
        job.id,
        JobStatus.COMPLETED,
        outputs={
            "character": f"/assets/{job.id}/character.png",
            "frame_010": f"/assets/{job.id}/frames/010.png",
            "frame_002": f"/assets/{job.id}/frames/002.png",
            "spritesheet": f"/assets/{job.id}/spritesheet.png",
            "metadata": f"/assets/{job.id}/metadata.json",
        },
    )
    client = await aiohttp_client(create_app(runner))

    completed = await client.get(f"/v1/jobs/{job.id}")
    malformed = await client.get("/v1/jobs/not-a-uuid")
    unknown = await client.get(f"/v1/jobs/{uuid4()}")

    assert completed.status == 200
    assert await completed.json() == {
        "job_id": job.id,
        "status": "completed",
        "character_design": f"/assets/{job.id}/character.png",
        "frames": [
            f"/assets/{job.id}/frames/002.png",
            f"/assets/{job.id}/frames/010.png",
        ],
        "spritesheet": f"/assets/{job.id}/spritesheet.png",
        "metadata": f"/assets/{job.id}/metadata.json",
    }
    assert malformed.status == 404
    assert unknown.status == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        r"could not open E:\private\secret.png or /srv/private/secret.png",
        r"could not open \\server\share\private.png",
        r"could not open \Windows\private.png",
        r"could not open C:\Program Files\private.png",
        "could not open /srv/private files/secret.png",
        "could not open path:/srv/private/secret.png",
        r"could not open path:\Windows\private.txt",
    ],
)
async def test_failed_job_error_with_path_marker_is_generic(aiohttp_client, tmp_path, error):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    runner.store.transition(
        job.id,
        JobStatus.FAILED,
        error=error,
    )
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/v1/jobs/{job.id}")
    payload = await response.json()

    assert response.status == 200
    assert payload["error"] == "generation failed due to an internal error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (" frame 2/8 failed ", "frame 2/8 failed"),
        (r"profile\name is invalid", r"profile\name is invalid"),
        ("https://public.example failed", "https://public.example failed"),
    ],
)
async def test_failed_job_error_without_absolute_path_marker_is_preserved(
    aiohttp_client, tmp_path, error, expected
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    runner.store.transition(job.id, JobStatus.FAILED, error=error)
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/v1/jobs/{job.id}")
    payload = await response.json()

    assert response.status == 200
    assert payload["error"] == expected


@pytest.mark.asyncio
async def test_assets_are_scoped_to_the_requested_job_directory(aiohttp_client, tmp_path):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    sibling = runner.enqueue(_request())
    asset = runner.jobs_root / job.id / "frames" / "000.png"
    asset.parent.mkdir()
    asset.write_bytes(b"PNG")
    sibling_asset = runner.jobs_root / sibling.id / "private.png"
    sibling_asset.write_bytes(b"private")
    client = await aiohttp_client(create_app(runner))

    served = await client.get(f"/assets/{job.id}/frames/000.png")
    traversal = await client.get(f"/assets/{job.id}/%2E%2E%2F{sibling.id}/private.png")
    backslash = await client.get(f"/assets/{job.id}/frames%5C000.png")
    drive_path = await client.get(f"/assets/{job.id}/C%3Asecret.png")

    assert served.status == 200
    assert await served.read() == b"PNG"
    assert traversal.status == 404
    assert backslash.status == 404
    assert drive_path.status == 404


@pytest.mark.asyncio
async def test_asset_read_rejects_a_directory_swapped_after_precheck(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    frames_directory = runner.jobs_root / job.id / "frames"
    frames_directory.mkdir()
    asset = frames_directory / "000.png"
    asset.write_bytes(b"safe")
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    (external_directory / "000.png").write_bytes(b"external")
    displaced_directory = tmp_path / "displaced-frames"
    original_component_stats = getattr(app_module, "_asset_component_stats", None)
    swapped = False

    def swap_after_precheck(root, candidate):
        nonlocal swapped
        stats = original_component_stats(root, candidate)
        if stats is not None and not swapped:
            swapped = True
            frames_directory.rename(displaced_directory)
            _create_directory_junction(frames_directory, external_directory)
        return stats

    monkeypatch.setattr(
        app_module, "_asset_component_stats", swap_after_precheck, raising=False
    )
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/frames/000.png")
    body = await response.read()

    assert swapped
    assert response.status == 404
    assert b"external" not in body


@pytest.mark.asyncio
async def test_asset_read_rejects_an_open_handle_outside_jobs_root(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    asset = runner.jobs_root / job.id / "frames" / "000.png"
    asset.parent.mkdir()
    asset.write_bytes(b"safe")
    external_path = tmp_path / "external" / "000.png"
    external_path.parent.mkdir()
    external_path.write_bytes(b"external")
    monkeypatch.setattr(
        app_module,
        "_final_handle_path",
        lambda asset_file: external_path,
        raising=False,
    )
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/frames/000.png")
    body = await response.read()

    assert response.status == 404
    assert b"external" not in body


@pytest.mark.asyncio
async def test_asset_read_rejects_a_handle_outside_the_pinned_jobs_root(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    asset = runner.jobs_root / job.id / "frames" / "000.png"
    asset.parent.mkdir()
    asset.write_bytes(b"safe")
    external_root = tmp_path / "external"
    external_root.mkdir()

    @contextmanager
    def outside_jobs_root(jobs_root):
        yield external_root

    monkeypatch.setattr(
        app_module,
        "_pinned_jobs_root_final_path",
        outside_jobs_root,
        raising=False,
    )
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/frames/000.png")

    assert response.status == 404


@pytest.mark.asyncio
async def test_asset_read_rejects_a_junctioned_jobs_root_before_component_checks(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    jobs_root = runner.jobs_root
    safe_root = tmp_path / "safe-root"
    jobs_root.rename(safe_root)
    external_root = tmp_path / "external-root"
    external_job_directory = external_root / job.id
    external_job_directory.mkdir(parents=True)
    (external_job_directory / "job.json").write_bytes(
        (safe_root / job.id / "job.json").read_bytes()
    )
    external_asset = external_job_directory / "frames" / "000.png"
    external_asset.parent.mkdir()
    external_asset.write_bytes(b"external")
    _create_directory_junction(jobs_root, external_root)
    original_component_stats = app_module._asset_component_stats

    def external_component_stats(root, candidate):
        relative_path = candidate.relative_to(root)
        return original_component_stats(external_root, external_root / relative_path)

    monkeypatch.setattr(app_module, "_asset_component_stats", external_component_stats)
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/frames/000.png")
    body = await response.read()

    assert response.status == 404
    assert b"external" not in body


@pytest.mark.asyncio
async def test_assets_reject_a_junctioned_job_directory_with_lstat_fallback(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    sibling = runner.enqueue(_request())
    job_directory = runner.jobs_root / job.id
    displaced_directory = tmp_path / "displaced" / job.id
    displaced_directory.parent.mkdir()
    job_directory.rename(displaced_directory)
    (runner.jobs_root / sibling.id / "private.png").write_bytes(b"private")
    _create_directory_junction(job_directory, runner.jobs_root / sibling.id)
    monkeypatch.delattr(Path, "is_junction", raising=False)
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/private.png")

    assert response.status == 404


@pytest.mark.asyncio
async def test_assets_reject_a_junctioned_candidate_with_lstat_fallback(
    aiohttp_client, monkeypatch, tmp_path
):
    runner = _FakeRunner(tmp_path)
    job = runner.enqueue(_request())
    target_directory = runner.jobs_root / job.id / "source"
    target_directory.mkdir()
    (target_directory / "private.png").write_bytes(b"private")
    _create_directory_junction(
        runner.jobs_root / job.id / "frames", target_directory
    )
    monkeypatch.delattr(Path, "is_junction", raising=False)
    client = await aiohttp_client(create_app(runner))

    response = await client.get(f"/assets/{job.id}/frames/private.png")

    assert response.status == 404


@pytest.mark.asyncio
async def test_app_lifecycle_starts_and_stops_runner_once(aiohttp_client, tmp_path):
    runner = _FakeRunner(tmp_path)
    owned_client = _FakeClient()
    client = await aiohttp_client(create_app(runner, owned_client))

    assert runner.start_calls == 1
    await client.close()
    assert runner.stop_calls == 1
    assert owned_client.close_calls == 1


@pytest.mark.asyncio
async def test_app_cleanup_closes_owned_client_when_runner_stop_fails(aiohttp_client, tmp_path):
    class FailingRunner(_FakeRunner):
        async def stop(self) -> None:
            await super().stop()
            if self.stop_calls == 1:
                raise RuntimeError("stop failed")

    runner = FailingRunner(tmp_path)
    owned_client = _FakeClient()
    client = await aiohttp_client(create_app(runner, owned_client))

    with pytest.raises(RuntimeError, match="stop failed"):
        await client.close()

    assert runner.stop_calls == 1
    assert owned_client.close_calls == 1


def test_main_constructs_local_api_with_environment_host_and_port(monkeypatch):
    calls: dict[str, object] = {}

    class FakeClient:
        pass

    class FakeRunner:
        def __init__(self, project_root, client) -> None:
            calls["project_root"] = project_root
            calls["runner_client"] = client

    def fake_create_app(runner, client):
        calls["app_runner"] = runner
        calls["app_client"] = client
        return "app"

    def fake_run_app(app, *, host, port):
        calls["run"] = (app, host, port)

    monkeypatch.setenv("GAME_ASSET_API_HOST", "0.0.0.0")
    monkeypatch.setenv("GAME_ASSET_API_PORT", "9001")
    with (
        patch("game_asset_api.comfy_client.ComfyClient", FakeClient),
        patch("game_asset_api.jobs.JobRunner", FakeRunner),
        patch("game_asset_api.app.create_app", fake_create_app),
        patch("aiohttp.web.run_app", fake_run_app),
    ):
        runpy.run_module("game_asset_api.__main__", run_name="__main__")

    assert calls["project_root"] == Path(__file__).resolve().parents[2]
    assert calls["runner_client"] is calls["app_client"]
    assert calls["app_runner"] is not None
    assert calls["run"] == ("app", "0.0.0.0", 9001)


def test_main_rejects_invalid_port_before_starting_server(monkeypatch):
    monkeypatch.setenv("GAME_ASSET_API_PORT", "not-a-port")

    with patch("aiohttp.web.run_app") as run_app:
        with pytest.raises(ValueError, match="GAME_ASSET_API_PORT must be an integer"):
            runpy.run_module("game_asset_api.__main__", run_name="__main__")

    run_app.assert_not_called()


def test_readme_documents_game_asset_api_runtime():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "POST /v1/game-assets" in readme
    assert "GAME_ASSET_API_HOST" in readme
    assert r"E:\ComfyUI\.venv\Scripts\python.exe -m game_asset_api" in readme
