"""Deterministic prompt fragments for game asset generation."""

from __future__ import annotations

from game_asset_api.contracts import AssetRequest


CAMERA_TEXT = {
    "side": "fixed side view",
    "front": "fixed front view",
    "top_down": "fixed top-down view",
    "three_quarter": "fixed three-quarter view",
}
PIXEL_STYLE = "pixel art, crisp readable silhouettes, limited color palette"
NEGATIVE = "blurry, low quality, distorted, photorealistic, text, watermark"


def build_character_prompt(request: AssetRequest) -> str:
    """Build the character image prompt for *request*."""
    return ", ".join(
        (
            request.character_prompt,
            _resolved_camera_text(request),
            PIXEL_STYLE,
            "plain studio background",
        )
    )


def build_action_prompt(request: AssetRequest) -> str:
    """Build the animation prompt for *request*."""
    return ", ".join(
        (
            request.action_prompt,
            _resolved_camera_text(request),
            PIXEL_STYLE,
            "locked camera, consistent character identity",
        )
    )


def _resolved_camera_text(request: AssetRequest) -> str:
    if request.camera_prompt:
        return request.camera_prompt
    if request.camera:
        return CAMERA_TEXT[request.camera]
    return "camera view inferred from the character prompt"
