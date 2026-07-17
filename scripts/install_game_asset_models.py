"""Install the verified model set used by the pixel game asset workflow."""

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from game_asset_api.model_manifest import MODEL_SPECS, install


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ComfyUI deployment root",
    )
    root = parser.parse_args(argv).root
    for spec in MODEL_SPECS:
        install(spec, root)


if __name__ == "__main__":
    main()
