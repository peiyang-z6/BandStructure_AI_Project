"""P1 v2 create-only, fixed-ID enrichment (offline by default).

  python scripts/fill_sidecar_gaps.py --ids-json ORIGINAL_IDS.json \
      --sidecar LEGACY.json --incoming-json LOCAL_FIELDS.json --out FILLED.v2.json

--h5 may replace --ids-json; only root IDs are read. Incoming rows may be local
sidecar objects or AFLOW field bundles with exact AUIDs. Out-of-scope IDs are
reported and clipped; duplicate IDs fail before writes. Nonempty conflicts are
preserved, not silently overwritten. Unavailable fixed IDs remain read-error
placeholders. Full field/ID quality coverage is always regenerated into a NEW
<out_stem>_report.json, never reduced to identity counts or written over legacy.

No network is used unless --download-gaps is explicitly requested with metadata.
Unknown magnetic, spin/SOC/U and pseudopotential-version values remain unknown.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import hashlib
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
    fields: Dict[str, Any] = {"auid": "aflow:" + material_id.split("-", 1)[-1], "aurl": aurl,
                              "field_read_errors": {}, "field_provenance": {}}
    names = {"geometry": "lattice", "positions_fractional": "fractional_coordinates",
             "dft_type": "dft_functional", "species_pp": "pseudopotential",
             "kpoints_bands_path": "kpath_segments"}
    for ep in ENDPOINTS:
        url = base + "?" + ep
        field = names.get(ep, ep)
        fields["field_provenance"][field] = {"url": url, "status": "read_error"}
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=UA)
                resp = urllib.request.urlopen(req, timeout=30)
                with resp:
                    data = resp.read()
                    body = data.decode("utf-8-sig")
                fields[ep] = _parse_scalar(body)
                fields["field_provenance"][field].update({"status": "received",
                    "response_sha256": hashlib.sha256(data).hexdigest()})
                break
            except Exception as exc:
                last = exc
                if attempt + 1 < retries:
                    time.sleep(min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0.0, 0.15))
        else:
            fields[ep] = None
            fields["field_read_errors"][field] = str(last)
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
            rec.setdefault("field_read_errors", {})["kpoints_3d"] = bits["error"]
    from src.data.structure_sidecar import enrich_structure_record
    return enrich_structure_record(rec)


def main(argv=None) -> None:
    from pathlib import Path
    from src.data.structure_sidecar import (
        assert_new_outputs, atomic_write_json, load_fixed_ids, load_sidecar_records,
        merge_sidecar_records, _index_unique_records, StructureSidecarSchema,
    )
    parser = argparse.ArgumentParser(description="Offline P1 v2 fixed-ID enrichment; never rewrite input/coverage")
    fixed = parser.add_mutually_exclusive_group(required=True)
    fixed.add_argument("--h5", help="read root keys only")
    fixed.add_argument("--ids-json", help="JSON list of original fixed IDs")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--incoming-json", help="local list of sidecar records or AFLUX field bundles")
    parser.add_argument("--out", required=True, help="new version only; no overwrite option")
    parser.add_argument("--report")
    parser.add_argument("--conventions-json")
    parser.add_argument("--download-gaps", action="store_true", help="explicit opt-in to per-AURL network fetch")
    parser.add_argument("--metadata")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--with-kpoints", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    report_path = Path(args.report) if args.report else out.with_name(out.stem + "_report.json")
    try:
        assert_new_outputs([out, report_path], inputs=(args.sidecar, args.incoming_json, args.h5,
                                                     args.ids_json, args.metadata, args.conventions_json))
        ids = load_fixed_ids(h5_path=args.h5, ids_json=args.ids_json)
        existing = load_sidecar_records(args.sidecar)
        incoming = load_sidecar_records(args.incoming_json) if args.incoming_json else []
        conventions = None
        if args.conventions_json:
            with open(args.conventions_json, encoding="utf-8") as stream:
                conventions = json.load(stream)
        old_index = _index_unique_records(existing)
        _index_unique_records(incoming)
        if args.download_gaps:
            if not args.metadata or args.workers < 1:
                parser.error("--download-gaps requires --metadata and positive --workers")
            metadata = _index_unique_records(load_sidecar_records(args.metadata))
            targets = [(mid, metadata[mid].get("aurl")) for mid in sorted(set(ids) - old_index.keys())
                       if mid in metadata and metadata[mid].get("aurl")]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fetch_one, mid, aurl, args.with_kpoints): mid for mid, aurl in targets}
                for future in concurrent.futures.as_completed(futures):
                    mid = futures[future]
                    try:
                        incoming.append(future.result())
                    except Exception as exc:
                        incoming.append({"material_id": mid, "read_error": str(exc)})
        records, report = merge_sidecar_records(existing, incoming, ids, conventions=conventions)
        report["operation"] = "fill_new_version"
        report["inputs"] = {"sidecar": args.sidecar, "incoming_json": args.incoming_json,
                            "h5": args.h5, "ids_json": args.ids_json,
                            "network_requested": args.download_gaps}
        atomic_write_json(out, [StructureSidecarSchema().serialize(r) for r in records])
        atomic_write_json(report_path, report)
        print(json.dumps({"out": str(out), "requested": report["requested"],
                          "gap_fill_added": report["gap_fill_added"],
                          **{s: report[s] for s in ("valid", "ambiguous", "invalid", "read_error")}}))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
