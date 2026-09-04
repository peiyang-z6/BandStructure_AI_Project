"""Build the seven-split benchmark manifest for the v7 60k tensors (P0C).

Reads the frozen v7 split manifest (samples + spacegroups) and the
backfilled metadata (prototype/species/aflowlib_date), then materializes
all seven benchmark splits with the same seed-42 determinism used
everywhere else. Outputs seven_splits_manifest.json next to the tensors.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.benchmark_splits import build_all_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seven-split manifest (P0C)")
    parser.add_argument("--manifest", required=True, help="ood_split_manifest.json path")
    parser.add_argument("--metadata", required=True, help="backfilled aflow_metadata.json path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    samples = manifest.get("samples", [])
    if not samples:
        raise RuntimeError("manifest has no samples")
    # enrich samples with backfilled metadata fields
    meta = json.load(open(args.metadata, encoding="utf-8"))
    by_id = {}
    for rec in meta:
        mid = str(rec.get("material_id") or "")
        if mid:
            by_id[mid] = rec
    enriched = []
    for sample in samples:
        mid = str(sample.get("material_id") or "")
        extra = by_id.get(mid, {})
        merged = dict(sample)
        for field in ("prototype", "species", "species_pp", "aflowlib_date", "aurl"):
            if extra.get(field) not in (None, ""):
                merged[field] = extra[field]
        enriched.append(merged)

    spacegroups = np.asarray(
        [int(s.get("spacegroup_number", 0)) for s in enriched], dtype=np.int32
    )
    material_ids = np.asarray(
        [str(s.get("material_id")) for s in enriched], dtype="U64"
    )
    splits = build_all_splits(
        enriched, spacegroups, train_size=args.train_size, random_state=args.random_state
    )
    save_splits(args.output_dir, splits, material_ids, args.random_state)

    for name, split in splits.items():
        print(
            f"  {name}: train={split['num_train']} test={split['num_test']} "
            f"({split['test_fraction']:.3f}) group_key={split['group_key']}"
        )


if __name__ == "__main__":
    main()
