from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest

from game_asset_api.repository_audit import (
    ALLOWED_TOP_LEVEL,
    DISALLOWED_NAMES,
    DISALLOWED_SUFFIXES,
    MAX_TRACKED_BYTES,
    audit_paths,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_repository.py"


def _load_audit_script():
    spec = importlib.util.spec_from_file_location("audit_repository_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sized_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.truncate(size)


def test_policy_constants_match_the_public_repository_contract():
    assert ALLOWED_TOP_LEVEL == {
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
    assert {".env", "id_ed25519", "id_rsa"} <= DISALLOWED_NAMES
    assert DISALLOWED_SUFFIXES == {
        ".part",
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".pem",
    }
    assert MAX_TRACKED_BYTES == 10 * 1024 * 1024


def test_audit_accepts_approved_repository_paths(tmp_path):
    approved = [
        Path("README.md"),
        Path("game_asset_api/workflows.py"),
        Path("scripts/deploy.py"),
        Path("tests/test_deployment.py"),
        Path("workflows/example.json"),
        Path("docs/design.md"),
    ]

    assert audit_paths(tmp_path, approved) == []


def test_audit_reports_one_specific_violation_per_rejected_path(tmp_path):
    oversized = Path("docs/unexpected.bin")
    _sized_file(tmp_path / oversized, MAX_TRACKED_BYTES + 1)
    rejected = [
        Path("models/model.safetensors"),
        Path("output/frame.png"),
        Path(".env"),
        Path("id_ed25519"),
        Path("asset.part"),
        oversized,
    ]

    assert audit_paths(tmp_path, rejected) == [
        ".env: secret filename",
        "asset.part: disallowed runtime or model file",
        "docs/unexpected.bin: file exceeds 10 MiB",
        "id_ed25519: secret filename",
        "models/model.safetensors: disallowed runtime or model file",
        "output/frame.png: disallowed top-level path",
    ]


@pytest.mark.parametrize(
    "name",
    ["id_ed25519.pub", "id_ed25519.backup", "id_rsa"],
)
def test_audit_rejects_secret_key_filenames(tmp_path, name):
    assert audit_paths(tmp_path, [Path("docs") / name]) == [
        f"docs/{name}: secret filename"
    ]


@pytest.mark.parametrize(
    "name",
    [
        "asset.PART",
        "model.SAFETENSORS",
        "model.CKPT",
        "model.Pt",
        "model.pTH",
        "certificate.PEM",
    ],
)
def test_audit_matches_disallowed_suffixes_case_insensitively(tmp_path, name):
    assert audit_paths(tmp_path, [Path("docs") / name]) == [
        f"docs/{name}: disallowed runtime or model file"
    ]


def test_audit_accepts_a_regular_file_at_exactly_ten_mib(tmp_path):
    relative = Path("docs/exactly-ten-mib.bin")
    _sized_file(tmp_path / relative, MAX_TRACKED_BYTES)

    assert audit_paths(tmp_path, [relative]) == []


def test_audit_rejects_an_empty_path(tmp_path):
    assert audit_paths(tmp_path, [Path()]) == [
        ".: disallowed top-level path"
    ]


def test_audit_returns_violations_in_path_order(tmp_path):
    paths = [
        Path("output/frame.png"),
        Path("docs/z.PT"),
        Path(".env"),
        Path("docs/a.CKPT"),
    ]

    assert audit_paths(tmp_path, paths) == [
        ".env: secret filename",
        "docs/a.CKPT: disallowed runtime or model file",
        "docs/z.PT: disallowed runtime or model file",
        "output/frame.png: disallowed top-level path",
    ]


def test_cli_audits_nul_delimited_git_paths_and_exits_cleanly(
    monkeypatch, capsys
):
    module = _load_audit_script()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args[0], 0, stdout=b"README.md\0docs/design notes.md\0\0"
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() is None
    assert calls == [
        (
            (["git", "ls-files", "-z"],),
            {"cwd": ROOT, "check": True, "capture_output": True},
        )
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_writes_each_violation_and_exits_one(monkeypatch, capsys):
    module = _load_audit_script()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=b".env\0asset.PT\0"
        ),
    )

    with pytest.raises(SystemExit) as captured_exit:
        module.main()

    assert captured_exit.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        ".env: secret filename\n"
        "asset.PT: disallowed runtime or model file\n"
    )


def test_cli_decodes_git_paths_as_strict_utf8(monkeypatch):
    module = _load_audit_script()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=b"docs/\xff.bin\0"
        ),
    )

    with pytest.raises(UnicodeDecodeError):
        module.main()


def test_cli_allows_git_failures_to_propagate(monkeypatch):
    module = _load_audit_script()
    failure = subprocess.CalledProcessError(
        128, ["git", "ls-files", "-z"], stderr=b"git failed"
    )

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError) as captured:
        module.main()

    assert captured.value is failure
