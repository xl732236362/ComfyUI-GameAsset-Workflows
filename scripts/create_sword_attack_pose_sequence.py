"""Write deterministic OpenPose control frames for a sword attack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_asset_api.pose_sequence import write_sword_attack_pose_sequence


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and write the requested pose sequence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, choices=(2, 8), default=8)
    arguments = parser.parse_args(argv)
    write_sword_attack_pose_sequence(arguments.output_dir, arguments.frame_count)


if __name__ == "__main__":
    main()
