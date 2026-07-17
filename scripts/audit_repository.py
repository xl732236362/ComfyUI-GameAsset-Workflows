"""Audit the Git index for content unsuitable for the public repository."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from game_asset_api.repository_audit import audit_paths


def _run_git(command: list[str], *, input_data: bytes | None = None):
    arguments = {
        "cwd": root,
        "check": True,
        "capture_output": True,
    }
    if input_data is not None:
        arguments["input"] = input_data
    try:
        return subprocess.run(command, **arguments)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            if isinstance(error.stderr, bytes):
                stderr = error.stderr.decode("utf-8", errors="replace")
            else:
                stderr = error.stderr
            sys.stderr.write(stderr)
        raise


def _parse_index_sizes(relative_paths: list[Path]) -> dict[Path, int]:
    metadata = _run_git(["git", "ls-files", "-s", "-z"])
    entries = []
    seen_paths = set()
    for record in metadata.stdout.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, oid, stage = header.split()
        except ValueError as error:
            raise ValueError("malformed Git index metadata") from error
        if len(mode) != 6 or any(character not in b"01234567" for character in mode):
            raise ValueError("malformed Git index mode")
        if not oid or any(character not in b"0123456789abcdefABCDEF" for character in oid):
            raise ValueError("malformed Git index object ID")
        path = Path(raw_path.decode("utf-8"))
        if stage != b"0":
            raise ValueError(f"Git index entry is not stage 0: {path.as_posix()}")
        if path in seen_paths:
            raise ValueError(f"duplicate Git index metadata: {path.as_posix()}")
        seen_paths.add(path)
        entries.append((path, oid))

    if len(relative_paths) != len(set(relative_paths)) or seen_paths != set(
        relative_paths
    ):
        raise ValueError("Git index metadata does not match tracked paths")

    object_ids = b"".join(oid + b"\n" for _, oid in entries)
    objects = _run_git(
        [
            "git",
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        input_data=object_ids,
    )
    object_records = objects.stdout.splitlines()
    if len(object_records) != len(entries):
        raise ValueError("Git object response count does not match index metadata")

    sizes = {}
    for (path, expected_oid), record in zip(entries, object_records, strict=True):
        try:
            oid, object_type, raw_size = record.split()
            size = int(raw_size)
        except (TypeError, ValueError) as error:
            raise ValueError("malformed Git object response") from error
        if oid != expected_oid:
            raise ValueError("Git object response does not match index metadata")
        if object_type != b"blob":
            raise ValueError(f"Git index object is not a blob: {path.as_posix()}")
        if size < 0:
            raise ValueError("Git object size cannot be negative")
        sizes[path] = size

    return sizes


def main() -> None:
    tracked = _run_git(["git", "ls-files", "-z"])
    relative_paths = [
        Path(raw_path.decode("utf-8"))
        for raw_path in tracked.stdout.split(b"\0")
        if raw_path
    ]
    tracked_sizes = _parse_index_sizes(relative_paths)
    violations = audit_paths(root, relative_paths, tracked_sizes=tracked_sizes)
    if violations:
        sys.stderr.write("\n".join(violations) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
