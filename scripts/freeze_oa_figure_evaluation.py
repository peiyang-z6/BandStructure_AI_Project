"""Freeze a complete OA figure sampling frame before independent human labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.figure_review_queue import (
    freeze_evaluation_allocation, load_figure_review_queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--preannotation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-id", required=True)
    args = parser.parse_args()
    queue = load_figure_review_queue(args.dataset_manifest, args.preannotation_manifest)
    print(json.dumps(freeze_evaluation_allocation(
        queue, args.output_dir, args.benchmark_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
