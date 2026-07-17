"""Validate and publish workflow sources into an explicit ComfyUI root."""

from __future__ import annotations

import json
import os
from pathlib import Path


WORKFLOW_NAMES = (
    "pixel_character_design_api.json",
    "pixel_character_action_api.json",
    "pose_controlled_pixel_action_api.json",
    "video_wan2_2_5B_ti2v.json",
    "wan2_2_5b_dual_balanced.json",
)


def validate_comfy_root(root: Path) -> tuple[Path, Path]:
    """Return the normalized root and its virtual-environment Python."""
    root = Path(root).expanduser().resolve()
    if not (root / "main.py").is_file():
        raise ValueError("ComfyUI root must contain main.py")
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise ValueError("ComfyUI root must contain .venv/Scripts/python.exe")
    return root, python.resolve()


def publish_workflows(source: Path, comfy_root: Path) -> tuple[Path, ...]:
    """Validate all workflow JSON, then atomically publish changed bytes."""
    root, _ = validate_comfy_root(comfy_root)
    source = Path(source)
    payloads: dict[str, bytes] = {}
    for name in WORKFLOW_NAMES:
        try:
            payload = (source / name).read_bytes()
            parsed = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid workflow JSON: {name}") from error
        is_api_workflow = isinstance(parsed, dict) and isinstance(
            parsed.get("prompt"), dict
        )
        is_ui_workflow = (
            isinstance(parsed, dict)
            and isinstance(parsed.get("nodes"), list)
            and isinstance(parsed.get("links"), list)
        )
        if not is_api_workflow and not is_ui_workflow:
            raise ValueError(f"invalid workflow JSON: {name}")
        payloads[name] = payload

    destination = root / "user" / "default" / "workflows"
    destination.mkdir(parents=True, exist_ok=True)
    published = []
    for name in WORKFLOW_NAMES:
        target = destination / name
        payload = payloads[name]
        if not target.is_file() or target.read_bytes() != payload:
            temporary = target.with_name(f"{target.name}.tmp")
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        published.append(target)
    return tuple(published)
