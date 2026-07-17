"""Policy checks for files tracked by the public workflow repository."""

from __future__ import annotations

from pathlib import Path


ALLOWED_TOP_LEVEL = {
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


def audit_paths(root: Path, relative_paths: list[Path]) -> list[str]:
    """Return policy violations for repository-relative paths."""
    violations = []
    for relative in sorted(relative_paths):
        reason = None
        if (
            relative.name in DISALLOWED_NAMES
            or relative.name.startswith("id_ed25519")
        ):
            reason = "secret filename"
        elif relative.suffix.lower() in DISALLOWED_SUFFIXES:
            reason = "disallowed runtime or model file"
        elif not relative.parts or relative.parts[0] not in ALLOWED_TOP_LEVEL:
            reason = "disallowed top-level path"
        else:
            candidate = root / relative
            if candidate.is_file() and candidate.stat().st_size > MAX_TRACKED_BYTES:
                reason = "file exceeds 10 MiB"

        if reason is not None:
            violations.append(f"{relative.as_posix()}: {reason}")

    return violations
