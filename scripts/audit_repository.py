"""Audit the Git index for content unsuitable for the public repository."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from game_asset_api.repository_audit import audit_paths


def main() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = [
        Path(raw_path.decode("utf-8"))
        for raw_path in tracked.stdout.split(b"\0")
        if raw_path
    ]
    violations = audit_paths(root, relative_paths)
    if violations:
        sys.stderr.write("\n".join(violations) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
