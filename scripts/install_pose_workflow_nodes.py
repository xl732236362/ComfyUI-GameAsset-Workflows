"""Install pinned custom nodes used by the pose-controlled pixel workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_asset_api.node_manifest import NODE_SPECS, NodeSpec, install_node_archive


def install_one(spec: NodeSpec, root: Path, python: Path) -> Path:
    archive_dir = root / "temp" / "pose_workflow_node_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{spec.name}-{spec.revision}.zip"
    if not _valid_zip(archive):
        partial = archive.with_suffix(".zip.part")
        command = [
            "curl.exe",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-all-errors",
            "--header",
            "User-Agent: Codex-Local-Deployment",
            "--output",
            str(partial),
            spec.archive_url,
        ]
        subprocess.run(command, check=True)
        if not _valid_zip(partial):
            raise RuntimeError(f"Downloaded node archive is invalid: {spec.name}")
        os.replace(partial, archive)

    destination = install_node_archive(spec, root, archive)
    requirements = destination / "requirements.txt"
    if requirements.is_file():
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements)],
            check=True,
        )
    return destination


def _valid_zip(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    arguments = parser.parse_args(argv)
    for spec in NODE_SPECS:
        install_one(spec, arguments.root, arguments.python)


if __name__ == "__main__":
    main()
