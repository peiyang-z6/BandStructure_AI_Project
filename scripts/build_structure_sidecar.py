"""P1 v2: offline read-only audit or create-only enrichment; explicit download opt-in.

Examples (run from the project root; choose NEW version paths):
  python scripts/build_structure_sidecar.py --mode audit --h5 BAND_CACHE.h5 \
      --sidecar LEGACY.json --out NEW_QUALITY.v2.json
  python scripts/build_structure_sidecar.py --mode enrich --ids-json FIXED_IDS.json \
      --sidecar LEGACY.json --conventions-json CONVENTIONS.json --out NEW_SIDECAR.v2.json

Audit opens only HDF5 root keys, never energies or other datasets. --ids-json
avoids opening HDF5 at all. Both ID and per-field quality counts are emitted.
Enrich creates a JSON record list plus <out_stem>_report.json; neither output may
already exist. Missing legacy units/basis are not inferred: without an explicit
conventions declaration, the v2 structure digest is absent and legacy integrity
is reported separately. A declaration is not independent source verification.

--mode download retains the historical fixed AFLUX Egap(0*,*5),natoms(1*,*50)
query and optional KPOINTS.bands endpoint fetch. It must be explicitly selected;
it is NOT exercised by offline audit/enrich. Endpoint syntax and segment counts
never establish dense pointwise alignment or primitive/conventional cell basis.
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
    """Compatibility entry: strict endpoint parser, never a dense-k certificate."""
    from src.data.structure_sidecar import parse_kpoints_bands as parse
    return parse(text)

def build_records(
    h5_path: str, bulk_rows: List[Dict[str, Any]], with_kpoints: bool, kworkers: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map bulk rows to sidecar records, keyed to the HDF5 material_id set."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        h5_ids = set(f.keys())
    print(f"[P1] HDF5 groups: {len(h5_ids)}", flush=True)

    records: List[Dict[str, Any]] = []
    from src.data.structure_sidecar import aflow_auid_to_material_id
    seen: set[str] = set()
    clipped = []
    for row in bulk_rows:
        mid = aflow_auid_to_material_id(row.get("auid"))
        if mid in seen:
            raise ValueError(f"duplicate incoming material_id: {mid}")
        seen.add(mid)
        if mid not in h5_ids:
            clipped.append(mid)
            continue
        records.append(sidecar_record_from_aflow_fields(row))

    # Optional: fill 3D k-points from KPOINTS.bands.
    if with_kpoints:
        to_fetch = [(r["material_id"], r.get("aurl")) for r in records if r.get("aurl")]
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=kworkers) as pool:
            futs = {pool.submit(fetch_kpoints_bits, aurl): mid
                    for mid, aurl in to_fetch if aurl}
            by_id = {r["material_id"]: r for r in records}
            for future in concurrent.futures.as_completed(futs):
                mid = futs[future]
                record = by_id[mid]
                try:
                    bits = future.result()
                    if not isinstance(bits, dict):
                        raise ValueError("empty KPOINTS response")
                    if "error" in bits and bits.get("verification_status") != "invalid":
                        record.setdefault("field_read_errors", {})["kpoints_3d"] = bits["error"]
                    else:
                        record["kpoints_3d"] = bits
                except Exception as exc:
                    record.setdefault("field_read_errors", {})["kpoints_3d"] = str(exc)
                done += 1
                if done % 1000 == 0:
                    print(f"[P1] kpoints {done}/{len(to_fetch)} ...", flush=True)

    from src.data.structure_sidecar import enrich_structure_record
    records = [enrich_structure_record(r) for r in records]
    report = coverage_report(records, h5_ids)
    report["clipped_incoming_ids"] = sorted(clipped)
    return records, report


def coverage_report(records: List[Dict[str, Any]], h5_ids: set[str]) -> Dict[str, Any]:
    from src.data.structure_sidecar import build_quality_ledger
    return build_quality_ledger(records, h5_ids)


def main(argv=None) -> None:
    from pathlib import Path
    from src.data.structure_sidecar import (
        assert_new_outputs, atomic_write_json, load_fixed_ids,
        load_sidecar_records, merge_sidecar_records, StructureSidecarSchema,
    )
    parser = argparse.ArgumentParser(description="P1 v2 read-only audit / new-version enrich / explicit download")
    parser.add_argument("--mode", choices=("audit", "enrich", "download"), default="audit")
    fixed = parser.add_mutually_exclusive_group(required=True)
    fixed.add_argument("--h5", help="read root keys ONLY")
    fixed.add_argument("--ids-json", help="JSON list of fixed material IDs")
    parser.add_argument("--sidecar", help="immutable legacy/v2 input JSON")
    parser.add_argument("--metadata", help="legacy downloader metadata path")
    parser.add_argument("--out", required=True, help="new version path; existing output always protected")
    parser.add_argument("--report", help="new coverage report path for enrich/download")
    parser.add_argument("--conventions-json", help="explicit structure units/basis declaration; never k-basis proof")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--with-kpoints", action="store_true")
    parser.add_argument("--kpoints-workers", type=int, default=8)
    args = parser.parse_args(argv)
    out = Path(args.out)
    report_path = Path(args.report) if args.report else out.with_name(out.stem + "_report.json")
    outputs = [out] if args.mode == "audit" else [out, report_path]
    try:
        assert_new_outputs(outputs, inputs=(args.sidecar, args.h5, args.ids_json, args.metadata, args.conventions_json))
        ids = load_fixed_ids(h5_path=args.h5, ids_json=args.ids_json)
        input_read_error = None
        if args.mode == "download":
            if not args.h5 or not args.metadata:
                parser.error("download requires --h5 and --metadata")
            rows = fetch_all_bulk_rows(args.page_size, args.workers)
            records, report = build_records(args.h5, rows, args.with_kpoints, args.kpoints_workers)
        else:
            if not args.sidecar:
                parser.error("audit/enrich requires --sidecar")
            try:
                records = load_sidecar_records(args.sidecar)
            except (OSError, ValueError, UnicodeError) as exc:
                if args.mode != "audit":
                    raise
                records = []
                input_read_error = str(exc)
            if args.mode == "enrich":
                conventions = None
                if args.conventions_json:
                    with open(args.conventions_json, encoding="utf-8") as stream:
                        conventions = json.load(stream)
                records, report = merge_sidecar_records(records, [], ids, conventions=conventions)
            else:
                report = coverage_report(records, set(ids))
        report["input_read_error"] = input_read_error
        report["operation"] = args.mode
        report["inputs"] = {"sidecar": args.sidecar, "h5": args.h5, "ids_json": args.ids_json,
                            "hdf5_access": "root_keys_only_no_dataset_reads" if args.h5 else "not_opened"}
        if args.mode == "audit":
            atomic_write_json(out, report)
        else:
            atomic_write_json(out, [StructureSidecarSchema().serialize(r) for r in records])
            atomic_write_json(report_path, report)
        print(json.dumps({"out": str(out), "requested": report["requested"],
                          **{s: report[s] for s in ("valid", "ambiguous", "invalid", "read_error")}}, ensure_ascii=False))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
