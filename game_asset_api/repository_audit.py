"""Policy checks for files tracked by the public workflow repository."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


_ALLOWED_TOP_LEVEL_FILES = {
    ".gitignore",
    "README.md",
    "deploy.ps1",
    "pyproject.toml",
}
_ALLOWED_TOP_LEVEL_DIRECTORIES = {
    "game_asset_api",
    "scripts",
    "tests",
    "workflows",
    "docs",
}
ALLOWED_TOP_LEVEL = _ALLOWED_TOP_LEVEL_FILES | _ALLOWED_TOP_LEVEL_DIRECTORIES
DISALLOWED_NAMES = {".env", "id_ed25519", "id_rsa"}
DISALLOWED_SUFFIXES = {
    ".part",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".pem",
}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def audit_paths(
    root: Path,
    relative_paths: list[Path],
    *,
    tracked_sizes: Mapping[Path, int] | None = None,
) -> list[str]:
    """Return policy violations for repository-relative paths."""
    violations = []
    for relative in sorted(relative_paths):
        reason = None
        name = relative.name.casefold()
        if (
            name in DISALLOWED_NAMES
            or name.startswith(".env.")
            or name.startswith("id_ed25519")
            or name.startswith("id_rsa")
        ):
            reason = "secret filename"
        elif any(
            name.endswith(suffix) or f"{suffix}." in name
            for suffix in DISALLOWED_SUFFIXES
        ):
            reason = "disallowed runtime or model file"
        elif not relative.parts or not (
            relative.parts[0] in _ALLOWED_TOP_LEVEL_FILES
            and len(relative.parts) == 1
            or relative.parts[0] in _ALLOWED_TOP_LEVEL_DIRECTORIES
            and len(relative.parts) > 1
        ):
            reason = "disallowed top-level path"
        else:
            if tracked_sizes is not None:
                size = tracked_sizes[relative]
            else:
                candidate = root / relative
                size = candidate.stat().st_size if candidate.is_file() else 0
            if size > MAX_TRACKED_BYTES:
                reason = "file exceeds 10 MiB"

        if reason is not None:
            violations.append(f"{relative.as_posix()}: {reason}")

    return violations
