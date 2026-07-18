"""Pinned custom-node sources for the pose-controlled pixel workflow."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    archive_url: str
    revision: str


NODE_SPECS = (
    NodeSpec(
        name="comfyui_controlnet_aux",
        archive_url=(
            "https://api.github.com/repos/Fannovel16/comfyui_controlnet_aux/zipball/"
            "e8b689a513c3e6b63edc44066560ca5919c0576e"
        ),
        revision="e8b689a513c3e6b63edc44066560ca5919c0576e",
    ),
    NodeSpec(
        name="ComfyUI_IPAdapter_plus",
        archive_url=(
            "https://api.github.com/repos/cubiq/ComfyUI_IPAdapter_plus/zipball/"
            "a0f451a5113cf9becb0847b92884cb10cbdec0ef"
        ),
        revision="a0f451a5113cf9becb0847b92884cb10cbdec0ef",
    ),
    NodeSpec(
        name="ComfyUI-AnimateDiff-Evolved",
        archive_url=(
            "https://api.github.com/repos/Kosinkadink/"
            "ComfyUI-AnimateDiff-Evolved/zipball/"
            "d8d163cd90b1111f6227495e3467633676fbb346"
        ),
        revision="d8d163cd90b1111f6227495e3467633676fbb346",
    ),
)


def existing_node_install(spec: NodeSpec, root: Path) -> Path | None:
    """Return a verified install or reject an occupied destination."""
    destination = Path(root) / "custom_nodes" / spec.name
    marker = destination / ".codex-source-revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == spec.revision:
        return destination
    if destination.exists():
        raise FileExistsError(f"unmanaged custom node directory exists: {destination}")
    return None


def install_node_archive(spec: NodeSpec, root: Path, archive_path: Path) -> Path:
    """Safely publish one pinned GitHub source archive below ``custom_nodes``."""
    custom_nodes = Path(root) / "custom_nodes"
    destination = custom_nodes / spec.name
    existing = existing_node_install(spec, root)
    if existing is not None:
        return existing

    with ZipFile(archive_path) as archive:
        top_levels = _validated_top_levels(archive)
        if len(top_levels) != 1:
            raise ValueError("node archive must contain exactly one top-level directory")
        top_level = next(iter(top_levels))
        custom_nodes.mkdir(parents=True, exist_ok=True)
        staging = custom_nodes / f".{spec.name}-{spec.revision}.installing"
        if staging.exists():
            raise FileExistsError(f"stale node installation directory exists: {staging}")
        archive.extractall(custom_nodes)

    extracted = custom_nodes / top_level
    if not extracted.is_dir():
        raise ValueError("node archive did not extract a source directory")
    (extracted / ".codex-source-revision").write_text(
        spec.revision + "\n", encoding="utf-8"
    )
    os.replace(extracted, destination)
    return destination


def _validated_top_levels(archive: ZipFile) -> set[str]:
    top_levels: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or ":" in path.parts[0]
        ):
            raise ValueError(f"unsafe archive member: {member.filename}")
        top_levels.add(path.parts[0])
    return top_levels
