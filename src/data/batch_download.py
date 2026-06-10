"""
Robust Materials Project batch downloader.

This script intentionally keeps the project's data split policy untouched. It only
downloads line-mode band structures and stores metadata fields required by the
spacegroup-based splitter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.mp_adapter import MPAdapter, load_api_keys


class RobustBandDownloader:
    """Single-threaded downloader with rate limiting and checkpoint resume."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        new_api_key: Optional[str] = None,
        legacy_api_key: Optional[str] = None,
        cache_dir: str = "./data_cache",
    ):
        self.api_key = api_key or new_api_key or legacy_api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.adapter = MPAdapter(
            api_key=api_key,
            new_api_key=new_api_key,
            legacy_api_key=legacy_api_key,
            cache_dir=cache_dir,
            rate_limit_delay=0.2,
        )
        self.downloaded_ids = self._load_downloaded()
        self.stats = {
            "success": 0,
            "failed": 0,
            "skipped": len(self.downloaded_ids),
            "no_data": 0,
            "spacegroup_dist": defaultdict(int),
        }

    def close(self) -> None:
        self.adapter.close()

    def _load_downloaded(self) -> set[str]:
        h5_path = os.path.join(self.cache_dir, "mp_bands.h5")
        if not os.path.exists(h5_path):
            return set()

        with h5py.File(h5_path, "r") as f:
            return set(f.keys())

    def query_and_download(
        self,
        target_count: int = 500,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        print("=" * 60)
        print(
            "Starting robust Materials Project download | "
            f"target={target_count} | cached={len(self.downloaded_ids)}"
        )
        print("=" * 60)

        query_params = dict(query_params or {})
        print(f"[Query] parameters: {query_params}")

        metadata_list = self.adapter.fetch_metadata(query_params=query_params)
        print(f"[Query] found {len(metadata_list)} metadata records")

        candidates = [
            item
            for item in metadata_list
            if item.get("material_id") and item["material_id"] not in self.downloaded_ids
        ]

        print(
            f"[Download] scanning {len(candidates)} candidates until "
            f"{target_count} successful downloads\n"
        )

        attempted = 0
        for idx, meta in enumerate(candidates, start=1):
            if self.stats["success"] >= target_count:
                break

            attempted += 1
            material_id = meta["material_id"]
            formula = meta.get("formula_pretty") or "N/A"
            spacegroup = meta.get("spacegroup_number")

            print(
                f"[{idx}/{len(candidates)}] {material_id} "
                f"({formula}) SG:{spacegroup}",
                end=" ",
                flush=True,
            )

            try:
                bs_data = self.adapter.fetch_band_structure(material_id)
            except Exception as exc:
                self.stats["failed"] += 1
                print(f"FAILED: unexpected {type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(0.3)
                continue

            error = bs_data.get("error")

            if error:
                error_lower = str(error).lower()
                if "line-mode" in error_lower or "uniform" in error_lower or "no band" in error_lower:
                    self.stats["no_data"] += 1
                    print(f"NO_LINE_MODE: {str(error)[:90]}")
                else:
                    self.stats["failed"] += 1
                    print(f"FAILED: {str(error)[:120]}")
            else:
                bs_data["metadata"] = meta
                self.adapter.save_to_hdf5(bs_data)
                self.downloaded_ids.add(material_id)
                self.stats["success"] += 1
                if spacegroup:
                    self.stats["spacegroup_dist"][spacegroup] += 1
                print("OK")

            time.sleep(0.3)

        if self.stats["success"] < target_count:
            print(
                f"[WARN] Requested {target_count} successes, got "
                f"{self.stats['success']} after {attempted} attempts."
            )

        if metadata_list:
            self.adapter.save_metadata_to_json(metadata_list)

        self._print_summary()

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("Download summary")
        print("=" * 60)
        print(f"success: {self.stats['success']}")
        print(f"line-mode unavailable: {self.stats['no_data']}")
        print(f"failed: {self.stats['failed']}")
        print(f"already cached at start: {self.stats['skipped']}")
        print(f"new spacegroups: {len(self.stats['spacegroup_dist'])}")

        report_path = os.path.join(self.cache_dir, "download_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "time": datetime.now().isoformat(),
                    "stats": {
                        key: value
                        for key, value in self.stats.items()
                        if key != "spacegroup_dist"
                    },
                    "spacegroup_dist": dict(self.stats["spacegroup_dist"]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"report saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materials Project robust downloader")
    parser.add_argument("--target", "-t", type=int, default=200, help="download target")
    parser.add_argument("--cache", "-c", type=str, default="./data_cache", help="cache dir")
    parser.add_argument("--min-gap", type=float, default=0.1, help="minimum band gap in eV")
    parser.add_argument("--max-gap", type=float, default=5.0, help="maximum band gap in eV")
    parser.add_argument("--max-sites", type=int, default=50, help="maximum num_sites filter")
    args = parser.parse_args()

    downloader = None
    try:
        api_keys = load_api_keys()
        downloader = RobustBandDownloader(
            api_key=api_keys["default"],
            new_api_key=api_keys["new"],
            legacy_api_key=api_keys["legacy"],
            cache_dir=args.cache,
        )

        query_params = {
            "band_gap": (args.min_gap, args.max_gap),
            "num_sites": (1, args.max_sites),
            "theoretical": True,
        }
        downloader.query_and_download(target_count=args.target, query_params=query_params)
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        if downloader is not None:
            downloader.close()


if __name__ == "__main__":
    main()
