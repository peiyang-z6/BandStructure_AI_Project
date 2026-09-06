"""P1 structure sidecar gap-filler.

The AFLUX bulk query fingerprint (Egap(0*,*5), natoms(1*,*50)) cannot see a
subset of the 60k band HDF5 (those merged from a different download
fingerprint). Constitution 5.0 §8 P1 requires full coverage, so this script
fills the missing material_ids by pulling each one's AFLOW REST endpoints
directly (per-AURL), reusing the same sidecar mapping, and merges them into
the sidecar JSON (non-empty precedence, constitution §4).

Usage:
    python scripts/fill_sidecar_gaps.py \
        --h5 data/raw/aflow/snapshots/aflow_60000_20260831/aflow_bands.h5 \
        --metadata data/raw/aflow/snapshots/aflow_60000_20260831/aflow_metadata.json \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --workers 8 [--with-kpoints --kpoints-workers 10]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.request
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.data.structure_sidecar import sidecar_record_from_aflow_fields

AFLOWDATA_BASE = "https://aflowlib.duke.edu/AFLOWDATA/"
UA = {"User-Agent": "BandStructure-AI/phase1 (+sidecar gap fill)"}

ENDPOINTS = (
    "geometry", "positions_fractional", "species", "dft_type",
    "spin_cell", "spin_atom", "species_pp", "kpoints_bands_path",
    "kpoints", "code",
)


def _parse_scalar(body: str):
    import ast
    text = body.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text


def fetch_one(material_id: str, aurl: str, with_kpoints: bool, retries: int = 5) -> Dict[str, Any]:
    directory = aurl.split("AFLOWDATA/", 1)[-1].strip()
    base = AFLOWDATA_BASE + directory + "/"
    fields: Dict[str, Any] = {"auid": "aflow:" + material_id.split("-", 1)[-1], "aurl": aurl}
    for ep in ENDPOINTS:
        url = base + "?" + ep
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=UA)
                resp = urllib.request.urlopen(req, timeout=30)
                with resp:
                    body = resp.read().decode("utf-8-sig")
                fields[ep] = _parse_scalar(body)
                break
            except Exception as exc:
                last = exc
                if attempt + 1 < retries:
                    time.sleep(min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0.0, 0.15))
        else:
            fields[ep] = None
    rec = sidecar_record_from_aflow_fields(fields)
    if with_kpoints:
        from build_structure_sidecar import fetch_kpoints_bits
        bits = fetch_kpoints_bits(aurl)
        if "error" not in bits:
            rec["kpoints_3d"] = bits
            if "kpoints_3d" in rec["missing_fields"]:
                rec["missing_fields"].remove("kpoints_3d")
        else:
            rec["kpoints_3d"] = None
            if "kpoints_3d" not in rec["missing_fields"]:
                rec["missing_fields"].append("kpoints_3d")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill P1 sidecar gaps (missing material_ids)")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--with-kpoints", action="store_true")
    parser.add_argument("--kpoints-workers", type=int, default=10)
    args = parser.parse_args()

    import h5py
    with h5py.File(args.h5, "r") as f:
        h5_ids = set(f.keys())
    side = json.load(open(args.sidecar, encoding="utf-8"))
    side_ids = {r["material_id"] for r in side}
    missing = sorted(h5_ids - side_ids)
    print(f"[GAP] HDF5 {len(h5_ids)}, sidecar {len(side)}, missing {len(missing)}", flush=True)

    md = json.load(open(args.metadata, encoding="utf-8"))
    md_by = {r.get("material_id"): r for r in md}
    targets = [(mid, str(md_by[mid].get("aurl") or "")) for mid in missing
               if mid in md_by and "AFLOWDATA/" in str(md_by[mid].get("aurl") or "")]
    print(f"[GAP] fetch targets: {len(targets)}", flush=True)

    new_records: List[Dict[str, Any]] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, mid, aurl, args.with_kpoints): mid
                for mid, aurl in targets}
        for future in concurrent.futures.as_completed(futs):
            mid = futs[future]
            try:
                new_records.append(future.result())
            except Exception as exc:
                print(f"[GAP] {mid} failed: {exc}", flush=True)
            done += 1
            if done % 200 == 0:
                print(f"[GAP] {done}/{len(targets)} ...", flush=True)

    # Merge (non-empty precedence): keep existing sidecar records, add new.
    schema = __import__("src.data.structure_sidecar", fromlist=["StructureSidecarSchema"]).StructureSidecarSchema()
    merged = side + [schema.serialize(r) for r in new_records]

    # Coverage re-audit
    merged_ids = {r["material_id"] for r in merged}
    report = {
        "sidecar_records": len(merged),
        "hdf5_groups": len(h5_ids),
        "h5_ids_missing_in_sidecar": len(h5_ids - merged_ids),
        "sidecar_ids_not_in_h5": len(merged_ids - h5_ids),
        "gap_fill_added": len(new_records),
    }
    print(f"[GAP] report: {json.dumps(report, ensure_ascii=False)}", flush=True)

    with open(args.sidecar, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    rp = args.sidecar.replace(".json", "_report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[GAP] wrote {len(merged)} records -> {args.sidecar}", flush=True)


if __name__ == "__main__":
    main()
