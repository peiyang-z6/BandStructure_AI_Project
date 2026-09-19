"""Run bounded CV panel preannotation over a frozen Europe PMC candidate snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.figure_candidate_preannotator import preannotate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(preannotate_snapshot(args.manifest, args.output_dir, args.workers),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
