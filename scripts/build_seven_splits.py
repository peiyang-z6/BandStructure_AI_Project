"""Build seven_splits_manifest.v2.json in a NEW, non-canonical directory.

Reuse the canonical IDs, enrich from bound metadata and optional reported
protocol sidecar, and expose exclusions/applicability. Never overwrite history.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.benchmark_splits import build_all_splits, save_splits
from src.utils.selection_manifest import sha256_file


def enrich_samples(samples, metadata, protocol_metadata=None):
    """Join explicit identity/chemistry fields; protocol sidecar supplies no geometry or targets."""
    enriched = [dict(sample) for sample in samples]
    full_fields = ("prototype", "species", "species_pp", "aflowlib_date", "aurl",
                   "source", "source_catalog", "dft_functional", "dft_type")
    protocol_fields = ("source", "source_catalog", "dft_functional", "dft_type")
    for rows, fields in ((metadata, full_fields), (protocol_metadata or [], protocol_fields)):
        by_id = {row["material_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("duplicate metadata IDs make source/functional evidence ambiguous")
        for sample in enriched:
            extra = by_id.get(sample["material_id"], {})
            for field in fields:
                if extra.get(field) not in (None, "", []):
                    sample[field] = extra[field]
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seven-split manifest (P0C)")
    parser.add_argument("--manifest", required=True, help="ood_split_manifest.json path")
    parser.add_argument("--metadata", required=True, help="backfilled aflow_metadata.json path")
    parser.add_argument("--protocol-metadata", help="optional reported source/functional JSON sidecar; never inferred from catalog")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    from pathlib import Path
    output_dir = Path(args.output_dir).resolve()
    canonical_dir = Path(args.manifest).resolve().parent
    if output_dir == canonical_dir or canonical_dir in output_dir.parents:
        raise ValueError("output-dir must be a new version outside the canonical directory")
    if (output_dir / "seven_splits_manifest.v2.json").exists():
        raise FileExistsError("versioned split manifest already exists; choose a new output-dir")

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    samples = manifest.get("samples", [])
    if not samples:
        raise RuntimeError("manifest has no samples")
    # enrich samples with backfilled metadata fields
    meta = json.load(open(args.metadata, encoding="utf-8"))
    protocol = json.load(open(args.protocol_metadata, encoding="utf-8")) if args.protocol_metadata else []
    enriched = enrich_samples(samples, meta, protocol)

    spacegroups = np.asarray(
        [int(s.get("spacegroup_number", 0)) for s in enriched], dtype=np.int32
    )
    material_ids = np.asarray(
        [str(s.get("material_id")) for s in enriched], dtype=str
    )
    splits = build_all_splits(
        enriched, spacegroups, train_size=args.train_size, random_state=args.random_state,
        canonical_split=manifest
    )
    from src.data import benchmark_splits
    paths = {"canonical_manifest": args.manifest, "metadata": args.metadata,
             "builder_code": __file__, "split_code": benchmark_splits.__file__}
    if args.protocol_metadata:
        paths["protocol_metadata"] = args.protocol_metadata
    provenance = {role: {"path": os.path.abspath(path), "bytes": os.path.getsize(path),
                         "sha256": sha256_file(path)} for role, path in paths.items()}
    save_splits(args.output_dir, splits, material_ids, args.random_state, provenance)

    for name, split in splits.items():
        print(
            f"  {name}: train={split['num_train']} test={split['num_test']} "
            f"({split['test_fraction']:.3f}) group_key={split['group_key']}"
        )


if __name__ == "__main__":
    main()
