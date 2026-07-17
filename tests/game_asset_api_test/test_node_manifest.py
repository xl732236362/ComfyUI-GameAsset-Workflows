from pathlib import Path
import importlib.util
from zipfile import ZipFile

import pytest

from game_asset_api.node_manifest import NodeSpec, install_node_archive


def _load_node_installer():
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "install_pose_workflow_nodes.py"
    module_spec = importlib.util.spec_from_file_location(
        "install_pose_workflow_nodes", script
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _forbid_installer_subprocess(module, monkeypatch):
    calls = []

    def fail_run(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("node installer subprocess must not run")

    monkeypatch.setattr(module.subprocess, "run", fail_run)
    return calls


def _write_archive(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_install_node_archive_publishes_one_valid_source_tree(tmp_path):
    spec = NodeSpec(
        name="example_node",
        archive_url="https://api.github.com/repos/example/node/zipball/revision",
        revision="a" * 40,
    )
    archive = _write_archive(
        tmp_path / "node.zip",
        {
            "owner-node-revision/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            "owner-node-revision/requirements.txt": b"Pillow\n",
        },
    )

    destination = install_node_archive(spec, tmp_path / "comfy", archive)

    assert destination == tmp_path / "comfy" / "custom_nodes" / "example_node"
    assert (destination / "__init__.py").is_file()
    assert (destination / "requirements.txt").read_text() == "Pillow\n"
    assert (destination / ".codex-source-revision").read_text() == "a" * 40 + "\n"


def test_install_node_archive_rejects_path_traversal(tmp_path):
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", "b" * 40)
    archive = _write_archive(
        tmp_path / "bad.zip",
        {
            "owner-node-revision/__init__.py": b"",
            "owner-node-revision/../../escape.py": b"bad",
        },
    )

    with pytest.raises(ValueError, match="unsafe archive member"):
        install_node_archive(spec, tmp_path / "comfy", archive)

    assert not (tmp_path / "escape.py").exists()


def test_install_node_archive_preserves_matching_existing_install(tmp_path):
    revision = "c" * 40
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", revision)
    destination = tmp_path / "comfy" / "custom_nodes" / "example_node"
    destination.mkdir(parents=True)
    (destination / ".codex-source-revision").write_text(revision + "\n")
    (destination / "keep.txt").write_text("unchanged")

    result = install_node_archive(spec, tmp_path / "comfy", tmp_path / "missing.zip")

    assert result == destination
    assert (destination / "keep.txt").read_text() == "unchanged"


def test_install_node_archive_refuses_to_overwrite_unmanaged_directory(tmp_path):
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", "d" * 40)
    destination = tmp_path / "comfy" / "custom_nodes" / "example_node"
    destination.mkdir(parents=True)
    (destination / "user-file.py").write_text("user data")

    with pytest.raises(FileExistsError, match="unmanaged custom node"):
        install_node_archive(spec, tmp_path / "comfy", tmp_path / "missing.zip")

    assert (destination / "user-file.py").read_text() == "user data"


def test_node_installer_skips_download_for_matching_existing_install(
    tmp_path, monkeypatch
):
    module = _load_node_installer()
    revision = "e" * 40
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", revision)
    root = tmp_path / "comfy"
    destination = root / "custom_nodes" / spec.name
    destination.mkdir(parents=True)
    (destination / ".codex-source-revision").write_text(revision + "\n")
    (destination / "requirements.txt").write_text("example-package\n")
    calls = _forbid_installer_subprocess(module, monkeypatch)

    result = module.install_one(spec, root, tmp_path / "python.exe")

    assert result == destination
    assert calls == []
    assert not (root / "temp").exists()
    assert (destination / "requirements.txt").read_text() == "example-package\n"


@pytest.mark.parametrize("marker_revision", [None, "f" * 40], ids=("missing", "mismatch"))
def test_node_installer_refuses_unmanaged_existing_install_before_download(
    tmp_path, monkeypatch, marker_revision
):
    module = _load_node_installer()
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", "e" * 40)
    root = tmp_path / "comfy"
    destination = root / "custom_nodes" / spec.name
    destination.mkdir(parents=True)
    user_file = destination / "user-file.py"
    user_file.write_text("user data")
    if marker_revision is not None:
        (destination / ".codex-source-revision").write_text(marker_revision + "\n")
    calls = _forbid_installer_subprocess(module, monkeypatch)

    with pytest.raises(FileExistsError, match="unmanaged custom node"):
        module.install_one(spec, root, tmp_path / "python.exe")

    assert calls == []
    assert not (root / "temp").exists()
    assert user_file.read_text() == "user data"


def test_node_installer_refuses_nondirectory_destination_before_download(
    tmp_path, monkeypatch
):
    module = _load_node_installer()
    spec = NodeSpec("example_node", "https://example.invalid/node.zip", "e" * 40)
    root = tmp_path / "comfy"
    destination = root / "custom_nodes" / spec.name
    destination.parent.mkdir(parents=True)
    destination.write_text("user data")
    calls = _forbid_installer_subprocess(module, monkeypatch)

    with pytest.raises(FileExistsError, match="unmanaged custom node"):
        module.install_one(spec, root, tmp_path / "python.exe")

    assert calls == []
    assert not (root / "temp").exists()
    assert destination.read_text() == "user data"


def test_node_installer_uses_explicit_root_and_python(tmp_path, monkeypatch):
    module = _load_node_installer()
    node_spec = NodeSpec("example_node", "https://example.invalid/node.zip", "e" * 40)
    calls = []

    monkeypatch.setattr(module, "NODE_SPECS", (node_spec,))
    monkeypatch.setattr(
        module,
        "install_one",
        lambda spec, root, python: calls.append((spec, root, python)),
    )

    module.main(
        [
            "--root",
            str(tmp_path / "comfy"),
            "--python",
            str(tmp_path / "python.exe"),
        ]
    )

    assert calls == [
        (node_spec, tmp_path / "comfy", tmp_path / "python.exe")
    ]
