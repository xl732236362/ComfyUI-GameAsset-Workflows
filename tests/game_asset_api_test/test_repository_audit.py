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


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _initialize_git_repository(path: Path) -> Path:
    path.mkdir()
    _run_git(path, "init", "-q")
    return path


def _mock_index_commands(
    monkeypatch,
    module,
    *,
    tracked: bytes,
    metadata: bytes,
    objects: bytes = b"",
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["git", "ls-files", "-z"]:
            stdout = tracked
        elif command == ["git", "ls-files", "-s", "-z"]:
            stdout = metadata
        elif command == [
            "git",
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ]:
            stdout = objects
        else:
            pytest.fail(f"unexpected Git command: {command!r}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


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
    [
        ".env",
        ".env.local",
        ".ENV.BACKUP",
        "id_ed25519.pub",
        "id_ed25519.backup",
        "ID_ED25519.PUB",
        "id_rsa",
        "id_rsa.pub",
        "ID_RSA.BACKUP",
    ],
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


@pytest.mark.parametrize(
    "name",
    [
        "download.part.backup",
        "model.safetensors.old",
        "model.CKPT.BAK",
        "weights.pt.previous",
        "weights.PTH.OLD",
        "key.pem.backup",
    ],
)
def test_audit_rejects_backup_names_containing_a_disallowed_suffix(
    tmp_path, name
):
    assert audit_paths(tmp_path, [Path("docs") / name]) == [
        f"docs/{name}: disallowed runtime or model file"
    ]


def test_audit_accepts_a_regular_file_at_exactly_ten_mib(tmp_path):
    relative = Path("docs/exactly-ten-mib.bin")
    _sized_file(tmp_path / relative, MAX_TRACKED_BYTES)

    assert audit_paths(tmp_path, [relative]) == []


def test_audit_prefers_supplied_index_size_over_the_working_tree(tmp_path):
    relative = Path("docs/staged.bin")
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_bytes(b"small working tree file")

    assert audit_paths(
        tmp_path,
        [relative],
        tracked_sizes={relative: MAX_TRACKED_BYTES + 1},
    ) == ["docs/staged.bin: file exceeds 10 MiB"]


def test_audit_rejects_an_empty_path(tmp_path):
    assert audit_paths(tmp_path, [Path()]) == [
        ".: disallowed top-level path"
    ]


@pytest.mark.parametrize(
    "relative",
    [
        Path("README.md/payload.bin"),
        Path(".gitignore/secret"),
        Path("docs"),
    ],
)
def test_audit_enforces_top_level_file_and_directory_shapes(tmp_path, relative):
    assert audit_paths(tmp_path, [relative]) == [
        f"{relative.as_posix()}: disallowed top-level path"
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
    first_oid = b"1111111111111111111111111111111111111111"
    second_oid = b"2222222222222222222222222222222222222222"
    calls = _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"README.md\0docs/design notes.md\0\0",
        metadata=(
            b"100644 " + first_oid + b" 0\tREADME.md\0"
            b"100644 " + second_oid + b" 0\tdocs/design notes.md\0"
        ),
        objects=(
            first_oid + b" blob 12\n"
            + second_oid + b" blob 34\n"
        ),
    )

    assert module.main() is None
    assert calls == [
        (
            ["git", "ls-files", "-z"],
            {"cwd": ROOT, "check": True, "capture_output": True},
        ),
        (
            ["git", "ls-files", "-s", "-z"],
            {"cwd": ROOT, "check": True, "capture_output": True},
        ),
        (
            [
                "git",
                "cat-file",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            {
                "cwd": ROOT,
                "check": True,
                "capture_output": True,
                "input": first_oid + b"\n" + second_oid + b"\n",
            },
        ),
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_writes_each_violation_and_exits_one(monkeypatch, capsys):
    module = _load_audit_script()
    first_oid = b"1111111111111111111111111111111111111111"
    second_oid = b"2222222222222222222222222222222222222222"
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b".env\0asset.PT\0",
        metadata=(
            b"100644 " + first_oid + b" 0\t.env\0"
            b"100644 " + second_oid + b" 0\tasset.PT\0"
        ),
        objects=(
            first_oid + b" blob 12\n"
            + second_oid + b" blob 34\n"
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


def test_cli_forwards_git_stderr_and_allows_failure_to_propagate(
    monkeypatch, capsys
):
    module = _load_audit_script()
    failure = subprocess.CalledProcessError(
        128,
        ["git", "ls-files", "-z"],
        stderr=b"fatal: index unavailable \xff\n",
    )

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError) as captured:
        module.main()

    assert captured.value is failure
    assert capsys.readouterr().err == "fatal: index unavailable \ufffd\n"


@pytest.mark.parametrize("working_tree_state", ["smaller", "deleted"])
def test_cli_audits_the_staged_blob_when_working_tree_differs(
    tmp_path, monkeypatch, capsys, working_tree_state
):
    repository = _initialize_git_repository(tmp_path / "repository")
    relative = Path("docs/staged.bin")
    staged_file = repository / relative
    _sized_file(staged_file, MAX_TRACKED_BYTES + 1)
    _run_git(repository, "add", "--", relative.as_posix())
    if working_tree_state == "smaller":
        staged_file.write_bytes(b"small")
    else:
        staged_file.unlink()
    module = _load_audit_script()
    monkeypatch.setattr(module, "root", repository)

    with pytest.raises(SystemExit) as captured_exit:
        module.main()

    assert captured_exit.value.code == 1
    assert capsys.readouterr().err == "docs/staged.bin: file exceeds 10 MiB\n"


def test_cli_accepts_ordinary_and_exactly_ten_mib_staged_blobs(
    tmp_path, monkeypatch, capsys
):
    repository = _initialize_git_repository(tmp_path / "repository")
    ordinary = repository / "README.md"
    boundary = repository / "docs" / "boundary.bin"
    ordinary.write_bytes(b"ordinary")
    _sized_file(boundary, MAX_TRACKED_BYTES)
    _run_git(repository, "add", "--", "README.md", "docs/boundary.bin")
    ordinary.unlink()
    boundary.unlink()
    module = _load_audit_script()
    monkeypatch.setattr(module, "root", repository)

    assert module.main() is None
    assert capsys.readouterr().err == ""


def test_cli_accepts_symlink_mode_metadata_and_uses_its_blob_size(
    monkeypatch, capsys
):
    module = _load_audit_script()
    oid = b"1111111111111111111111111111111111111111"
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"docs/link.bin\0",
        metadata=b"120000 " + oid + b" 0\tdocs/link.bin\0",
        objects=oid + f" blob {MAX_TRACKED_BYTES + 1}\n".encode(),
    )

    with pytest.raises(SystemExit) as captured_exit:
        module.main()

    assert captured_exit.value.code == 1
    assert capsys.readouterr().err == "docs/link.bin: file exceeds 10 MiB\n"


def test_cli_rejects_an_unmerged_index_entry(monkeypatch):
    module = _load_audit_script()
    oid = b"1111111111111111111111111111111111111111"
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"docs/conflict.bin\0",
        metadata=b"100644 " + oid + b" 2\tdocs/conflict.bin\0",
    )

    with pytest.raises(ValueError, match="stage 0"):
        module.main()


def test_cli_rejects_a_missing_index_metadata_record(monkeypatch):
    module = _load_audit_script()
    oid = b"1111111111111111111111111111111111111111"
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"README.md\0docs/missing.bin\0",
        metadata=b"100644 " + oid + b" 0\tREADME.md\0",
        objects=oid + b" blob 12\n",
    )

    with pytest.raises(ValueError, match="metadata"):
        module.main()


def test_cli_rejects_a_non_blob_index_object(monkeypatch):
    module = _load_audit_script()
    oid = b"1111111111111111111111111111111111111111"
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"docs/submodule.bin\0",
        metadata=b"160000 " + oid + b" 0\tdocs/submodule.bin\0",
        objects=oid + b" commit 12\n",
    )

    with pytest.raises(ValueError, match="blob"):
        module.main()


@pytest.mark.parametrize(
    ("metadata", "objects"),
    [
        (b"malformed metadata\0", b""),
        (
            b"100644 1111111111111111111111111111111111111111 0\tdocs/file.bin\0",
            b"malformed object response\n",
        ),
    ],
    ids=("index-metadata", "batch-check-response"),
)
def test_cli_rejects_malformed_index_data(monkeypatch, metadata, objects):
    module = _load_audit_script()
    _mock_index_commands(
        monkeypatch,
        module,
        tracked=b"docs/file.bin\0",
        metadata=metadata,
        objects=objects,
    )

    with pytest.raises(ValueError, match="index|object"):
        module.main()
