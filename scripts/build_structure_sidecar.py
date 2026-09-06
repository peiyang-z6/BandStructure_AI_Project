"""P1 structure sidecar downloader.

Constitution 5.0 §8 P1: build a read-only pairing sidecar keyed by
material_id storing lattice 3x3, species, fractional_coordinates,
magnetic/spin, DFT functional/U/pseudopotential, reciprocal lattice, 3D
fractional k-points, k-path convention, source and a structure SHA-256. The
immutable aflow_bands.h5 is never modified.

Two data paths (both verified reachable 2026-09-06):
  1. AFLUX bulk paging (SAME query fingerprint as the 60k download) returns
     geometry / positions_fractional / species / dft_type / spin_cell /
     spin_atom / species_pp / kpoints_bands_path / kpoints / code / aurl.
     This covers lattice, species, fractional coords, functional, spin,
     pseudopotential and k-path convention in ~207 pages.
  2. KPOINTS.bands per-AURL (line-mode segment 3D endpoints) fills the 3D
     fractional k-points field that AFLUX bulk does not return. Optional
     (--with-kpoints); each file is ~1 KB.

Output: aflow_structure_sidecar.json (one record per material) +
aflow_structure_sidecar_report.json (per-field coverage + missing-field
breakdown). Coverage audit compares the sidecar material_id set against the
HDF5 group set.

Usage:
    python scripts/build_structure_sidecar.py \
        --h5 data/raw/aflow/snapshots/aflow_60000_20260831/aflow_bands.h5 \
        --metadata data/raw/aflow/snapshots/aflow_60000_20260831/aflow_metadata.json \
        --out data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --min-gap 0.0 --max-gap 5.0 --max-sites 50 --page-size 500 \
        [--workers 6] [--with-kpoints --kpoints-workers 8]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.structure_sidecar import sidecar_record_from_aflow_fields

API_ROOT = "https://aflow.org/API/aflux/?"
AFLOWDATA_BASE = "https://aflowlib.duke.edu/AFLOWDATA/"
UA = {"User-Agent": "BandStructure-AI/phase1 (+structure sidecar)"}

STRUCTURE_FIELDS = (
    "geometry",
    "positions_fractional",
    "species",
    "dft_type",
    "spin_cell",
    "spin_atom",
    "species_pp",
    "kpoints_bands_path",
    "kpoints",
    "code",
    "aurl",
)


def _build_query(page: int, page_size: int) -> str:
    fields = ",".join(f"{f}()" for f in STRUCTURE_FIELDS)
    aflux = f"Egap(0*,*5),natoms(1*,*50),{fields},paging({page},{page_size})"
    return API_ROOT + urllib.parse.quote(aflux, safe="(),*")


def _request_page(page: int, page_size: int, retries: int = 8) -> Dict[str, Any]:
    url = _build_query(page, page_size)
    openers = [
        urllib.request.build_opener(urllib.request.ProxyHandler({})),
        None,
    ]
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        opener = openers[attempt % len(openers)]
        try:
            req = urllib.request.Request(url, headers=UA)
            resp = (opener.open(req, timeout=40) if opener is not None
                    else urllib.request.urlopen(req, timeout=40))
            with resp:
                payload = resp.read()
            if not payload:
                raise OSError("empty response")
            return json.loads(payload.decode("utf-8-sig"))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(15.0, 1.0 * (2 ** attempt)) + random.uniform(0.0, 0.3))
    raise OSError(f"page {page} failed after {retries} attempts: {last_error}")


def fetch_all_bulk_rows(page_size: int, workers: int) -> List[Dict[str, Any]]:
    first = _request_page(1, page_size)
    total = None
    for key in first:
        if " of " in key:
            total = int(key.split()[-1])
            break
    if total is None:
        total = 120_000
    num_pages = (total + page_size - 1) // page_size
    print(f"[P1] total rows ~{total}, pages {num_pages}", flush=True)
    rows: List[Dict[str, Any]] = []

    def fetch(page: int) -> List[Dict[str, Any]]:
        payload = _request_page(page, page_size)
        return [v for v in payload.values() if isinstance(v, dict) and v.get("auid")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, page): page for page in range(1, num_pages + 1)}
        for future in concurrent.futures.as_completed(futures):
            page = futures[future]
            try:
                rows.extend(future.result())
                if len(rows) % (page_size * 20) < page_size * workers:
                    print(f"[P1] {len(rows)} rows ...", flush=True)
            except Exception as exc:
                print(f"[P1] page {page} failed: {exc}", flush=True)
    return rows


def _aurl_to_dir(aurl: str) -> str:
    # aurl like "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/FCC/B6Co21Mn2_ICSD_613157"
    return aurl.split("AFLOWDATA/", 1)[-1].strip()


def fetch_kpoints_bits(aurl: str, retries: int = 6) -> Optional[Dict[str, Any]]:
    """Fetch KPOINTS.bands 3D segment endpoints + header for one AURL."""
    directory = _aurl_to_dir(aurl)
    url = AFLOWDATA_BASE + directory + "/KPOINTS.bands"
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            resp = urllib.request.urlopen(req, timeout=30)
            with resp:
                text = resp.read().decode("utf-8", "replace")
            return parse_kpoints_bands(text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(10.0, 0.8 * (2 ** attempt)) + random.uniform(0.0, 0.2))
    return {"error": str(last_error)}


def parse_kpoints_bands(text: str) -> Dict[str, Any]:
    """Parse VASP line-mode KPOINTS.bands into 3D segment endpoints.

    Returns {path_line, nkpts, segments: [{label_from, label_to,
    k_from: [x,y,z], k_to: [x,y,z]}]}.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return {"error": "too short"}
    header = lines[0].strip()
    nkpts = None
    # VASP line-mode: line 0 = path header, line 1 = "N  !  N grids"
    if len(lines) > 1:
        grid_line = lines[1].split("!")[0].strip().split()
        if grid_line:
            try:
                nkpts = int(grid_line[0])
            except ValueError:
                nkpts = None
    segments: List[Dict[str, Any]] = []
    i = 4  # skip header, nkpts, 'Line-mode', 'reciprocal'
    while i + 1 < len(lines):
        a_line = lines[i]
        b_line = lines[i + 1]
        def _parse_pt(ln: str):
            parts = ln.split("!")
            coord = parts[0].split()
            label = parts[1].strip() if len(parts) > 1 else None
            if len(coord) < 3:
                return None, label
            return [float(x) for x in coord[:3]], label
        a_pt, a_label = _parse_pt(a_line)
        b_pt, b_label = _parse_pt(b_line)
        if a_pt is not None and b_pt is not None:
            segments.append({
                "label_from": a_label, "label_to": b_label,
                "k_from": a_pt, "k_to": b_pt,
            })
        i += 2  # blank lines already filtered; segment endpoints are consecutive
    return {"path_line": header, "nkpts": nkpts, "segments": segments}


def build_records(
    h5_path: str, bulk_rows: List[Dict[str, Any]], with_kpoints: bool, kworkers: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map bulk rows to sidecar records, keyed to the HDF5 material_id set."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        h5_ids = set(f.keys())
    print(f"[P1] HDF5 groups: {len(h5_ids)}", flush=True)

    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in bulk_rows:
        rec = sidecar_record_from_aflow_fields(row)
        mid = rec.get("material_id")
        if not mid or mid in seen:
            continue
        if mid not in h5_ids:
            continue  # only keep IDs present in the immutable band HDF5
        seen.add(mid)
        records.append(rec)

    # Optional: fill 3D k-points from KPOINTS.bands.
    if with_kpoints:
        to_fetch = [(r["material_id"], r.get("aurl")) for r in records if r.get("aurl")]
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=kworkers) as pool:
            futs = {pool.submit(fetch_kpoints_bits, aurl): mid
                    for mid, aurl in to_fetch if aurl}
            for future in concurrent.futures.as_completed(futs):
                mid = futs[future]
                try:
                    bits = future.result()
                    for r in records:
                        if r["material_id"] == mid:
                            if "error" not in bits:
                                r["kpoints_3d"] = bits
                                if "kpoints_3d" in r["missing_fields"]:
                                    r["missing_fields"].remove("kpoints_3d")
                            break
                except Exception:
                    pass
                done += 1
                if done % 1000 == 0:
                    print(f"[P1] kpoints {done}/{len(to_fetch)} ...", flush=True)

    report = coverage_report(records, h5_ids)
    return records, report


def coverage_report(records: List[Dict[str, Any]], h5_ids: set[str]) -> Dict[str, Any]:
    fields = [
        "lattice", "species", "fractional_coordinates", "dft_functional",
        "spin_cell", "spin_atom", "pseudopotential", "reciprocal_lattice",
        "kpath_segments", "structure_sha256", "kpoints_3d",
    ]
    total = len(records)
    present = {f: sum(1 for r in records if r.get(f) is not None) for f in fields}
    missing_breakdown: Dict[str, int] = {}
    for r in records:
        for f in r.get("missing_fields", []):
            missing_breakdown[f] = missing_breakdown.get(f, 0) + 1
    return {
        "sidecar_records": total,
        "hdf5_groups": len(h5_ids),
        "h5_ids_missing_in_sidecar": len(h5_ids - {r["material_id"] for r in records}),
        "sidecar_ids_not_in_h5": len({r["material_id"] for r in records} - h5_ids),
        "field_present": present,
        "missing_field_breakdown": missing_breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P1 structure sidecar")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--max-sites", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--with-kpoints", action="store_true")
    parser.add_argument("--kpoints-workers", type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    bulk_rows = fetch_all_bulk_rows(args.page_size, args.workers)
    print(f"[P1] bulk rows fetched: {len(bulk_rows)} in {time.time() - t0:.1f}s", flush=True)

    records, report = build_records(args.h5, bulk_rows, args.with_kpoints, args.kpoints_workers)
    print(f"[P1] coverage: {json.dumps(report, ensure_ascii=False)}", flush=True)

    schema = __import__("src.data.structure_sidecar", fromlist=["StructureSidecarSchema"]).StructureSidecarSchema()
    serialized = [schema.serialize(r) for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(serialized, fh, ensure_ascii=False)
    report_path = args.out.replace(".json", "_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[P1] wrote {len(serialized)} records -> {args.out}", flush=True)
    print(f"[P1] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
