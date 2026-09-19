"""Harvest 300 CC-BY electronic-band figure candidates through Europe PMC APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.europe_pmc_figure_harvester import DEFAULT_QUERY, harvest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-articles", type=int, default=1200)
    parser.add_argument("--max-per-article", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = harvest(
        args.output_dir,
        target_count=args.target,
        query=args.query,
        page_size=args.page_size,
        max_articles=args.max_articles,
        max_per_article=args.max_per_article,
        workers=args.workers,
        request_interval=args.request_interval,
        timeout=args.timeout,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["record_count"] < args.target:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
