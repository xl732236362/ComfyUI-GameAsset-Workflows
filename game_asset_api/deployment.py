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
    "production_animation_api.json",
)
_API_WORKFLOW_NAMES = frozenset(
    {
        "pixel_character_design_api.json",
        "pixel_character_action_api.json",
        "pose_controlled_pixel_action_api.json",
        "production_animation_api.json",
    }
)
_DISCOVERY_INPUTS = frozenset(
    {
        ("CheckpointLoaderSimple", "ckpt_name"),
        ("LoraLoader", "lora_name"),
        ("LoadBackgroundRemovalModel", "bg_removal_name"),
        ("UNETLoader", "unet_name"),
        ("UNETLoader", "weight_dtype"),
        ("CLIPLoader", "clip_name"),
        ("CLIPLoader", "type"),
        ("CLIPLoader", "device"),
        ("VAELoader", "vae_name"),
        ("IPAdapterModelLoader", "ipadapter_file"),
        ("CLIPVisionLoader", "clip_name"),
        ("IPAdapterAdvanced", "weight_type"),
        ("IPAdapterAdvanced", "combine_embeds"),
        ("IPAdapterAdvanced", "embeds_scaling"),
        ("ControlNetLoader", "control_net_name"),
        ("ImageScale", "upscale_method"),
        ("ImageScale", "crop"),
        ("KSampler", "sampler_name"),
        ("KSampler", "scheduler"),
    }
)
_UI_ONLY_DISCOVERY_VALUES = {
    ("SaveVideo", "format"): frozenset({"mp4"}),
    ("SaveVideo", "codec"): frozenset({"h264"}),
}


def validate_comfy_root(root: Path) -> tuple[Path, Path]:
    """Return the normalized root and its virtual-environment Python."""
    root = Path(root).expanduser().resolve()
    if not (root / "main.py").is_file():
        raise ValueError("ComfyUI root must contain main.py")
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise ValueError("ComfyUI root must contain .venv/Scripts/python.exe")
    return root, python.resolve()


def _reject_non_finite_json(constant: str) -> None:
    raise ValueError(f"non-finite JSON constant: {constant}")


def _valid_api_workflow(parsed: object) -> bool:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("prompt"), dict):
        return False
    return all(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and bool(node["class_type"].strip())
        and isinstance(node.get("inputs"), dict)
        for node in parsed["prompt"].values()
    )


def _valid_ui_workflow(parsed: object) -> bool:
    if (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("nodes"), list)
        or not parsed["nodes"]
        or not isinstance(parsed.get("links"), list)
    ):
        return False
    return all(
        isinstance(node, dict)
        and type(node.get("id")) is int
        and isinstance(node.get("type"), str)
        and bool(node["type"].strip())
        for node in parsed["nodes"]
    )


def _workflow_discovery_requirements(
    source: Path,
) -> tuple[set[str], dict[tuple[str, str], set[object]]]:
    node_types: set[str] = set()
    input_values: dict[tuple[str, str], set[object]] = {}
    for name in WORKFLOW_NAMES:
        try:
            parsed = json.loads(
                (Path(source) / name).read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json,
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"invalid workflow JSON: {name}") from error

        if name in _API_WORKFLOW_NAMES:
            if not _valid_api_workflow(parsed):
                raise ValueError(f"invalid workflow JSON: {name}")
            nodes = parsed["prompt"].values()
            for node in nodes:
                node_type = node["class_type"]
                node_types.add(node_type)
                for input_name, value in node["inputs"].items():
                    key = (node_type, input_name)
                    if key in _DISCOVERY_INPUTS:
                        input_values.setdefault(key, set()).add(value)
        else:
            if not _valid_ui_workflow(parsed):
                raise ValueError(f"invalid workflow JSON: {name}")
            node_types.update(
                node["type"]
                for node in parsed["nodes"]
                if node["type"] != "MarkdownNote"
            )

    for key, values in _UI_ONLY_DISCOVERY_VALUES.items():
        input_values.setdefault(key, set()).update(values)
    return node_types, input_values


def _advertised_options(node_info: object, input_name: str) -> set[object] | None:
    if not isinstance(node_info, dict) or not isinstance(node_info.get("input"), dict):
        return None
    schema = None
    for section_name in ("required", "optional"):
        section = node_info["input"].get(section_name)
        if isinstance(section, dict) and input_name in section:
            schema = section[input_name]
            break
    if not isinstance(schema, (list, tuple)) or not schema:
        return None
    if isinstance(schema[0], (list, tuple)):
        return set(schema[0])
    if (
        schema[0] == "COMBO"
        and len(schema) > 1
        and isinstance(schema[1], dict)
        and isinstance(schema[1].get("options"), (list, tuple))
    ):
        return set(schema[1]["options"])
    return None


def validate_object_info(object_info: dict, source: Path) -> None:
    """Require all workflow nodes and configured choices in ComfyUI discovery."""
    required_nodes, required_inputs = _workflow_discovery_requirements(source)
    issues = [
        f"missing node {node_type}"
        for node_type in sorted(required_nodes - set(object_info))
    ]
    for (node_type, input_name), values in sorted(required_inputs.items()):
        if node_type not in object_info:
            continue
        options = _advertised_options(object_info[node_type], input_name)
        missing = values if options is None else values - options
        if missing:
            rendered = ", ".join(sorted(map(str, missing)))
            issues.append(f"{node_type}.{input_name} missing options: {rendered}")
    if issues:
        raise ValueError("invalid ComfyUI object_info: " + "; ".join(issues))


def publish_workflows(source: Path, comfy_root: Path) -> tuple[Path, ...]:
    """Validate all workflow JSON, then atomically publish changed bytes."""
    root, _ = validate_comfy_root(comfy_root)
    source = Path(source)
    payloads: dict[str, bytes] = {}
    for name in WORKFLOW_NAMES:
        try:
            payload = (source / name).read_bytes()
            parsed = json.loads(
                payload.decode("utf-8"), parse_constant=_reject_non_finite_json
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"invalid workflow JSON: {name}") from error
        valid = (
            _valid_api_workflow(parsed)
            if name in _API_WORKFLOW_NAMES
            else _valid_ui_workflow(parsed)
        )
        if not valid:
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
