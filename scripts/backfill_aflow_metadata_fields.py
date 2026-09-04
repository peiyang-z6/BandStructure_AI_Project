"""Backfill missing AFLOW catalog fields (prototype/species/species_pp/year/experiment).

P0B: the seven-split benchmark suite needs prototype (prototype split),
species (leave-element split) and year (temporal split), none of which the
original downloader stored. This script re-pages the SAME AFLUX query
fingerprint (gap bins [0,5), natoms (1,50], page_size 500) and merges only the
whitelisted fields into the canonical metadata JSON, with non-empty
precedence (constitution §4: empty/missing must never downgrade existing
values). No band data is downloaded.

Usage:
    python scripts/backfill_aflow_metadata_fields.py \
        --metadata data/raw/aflow/snapshots/aflow_60000_20260831/aflow_metadata.json \
        --min-gap 0.0 --max-gap 5.0 --max-sites 50 --page-size 500 \
        [--workers 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BACKFILL_FIELDS = ("prototype", "species", "species_pp", "aflowlib_date")
API_ROOT = "https://aflow.org/API/aflux/?"
UA = {"User-Agent": "BandStructure-AI/phase0 (+research cache)"}


def row_to_backfill(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map one raw AFLUX row to {material_id, whitelisted fields}."""
    if not isinstance(raw, dict):
        return None
    auid = str(raw.get("auid") or "").strip()
    if not auid:
        return None
    out = {"material_id": "aflow-" + auid.split(":", 1)[-1]}
    for field in BACKFILL_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        # aflowlib_date is a list of entry dates; keep the creation date (first).
        if field == "aflowlib_date" and isinstance(value, list) and value:
            value = value[0]
        out[field] = value
    return out


def merge_backfill_rows(
    existing: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Merge whitelisted fields with non-empty precedence. Returns (list, stats)."""
    by_id = {str(rec.get("material_id")): rec for rec in existing}
    stats = {
        "rows_incoming": len(incoming),
        "new_ids": 0,
        "fields_filled": 0,
        "fields_blocked_non_empty": 0,
    }
    for row in incoming:
        mid = row["material_id"]
        if mid in by_id:
            target = by_id[mid]
        else:
            target = {"material_id": mid}
            by_id[mid] = target
            stats["new_ids"] += 1
        for field in BACKFILL_FIELDS:
            if field not in row:
                continue
            incoming_value = row[field]
            current = target.get(field)
            is_empty = current is None or (
                isinstance(current, str) and not str(current).strip()
            )
            if is_empty:
                target[field] = incoming_value
                stats["fields_filled"] += 1
            elif current != incoming_value:
                stats["fields_blocked_non_empty"] += 1
    return list(by_id.values()), stats


def _request_page(query: str, page: int, page_size: int, retries: int = 8) -> Dict[str, Any]:
    del query
    aflux = (
        "Egap(0*,*5),natoms(1*,*50),"
        "prototype(),species(),species_pp(),aflowlib_date(),"
        f"paging({page},{page_size})"
    )
    url = API_ROOT + urllib.parse.quote(aflux, safe="(),*")
    # Direct connection first: AFLOW intermittently 404s/SSL-resets through
    # some proxy routes (observed during the 60k band download campaign).
    openers = [
        urllib.request.build_opener(urllib.request.ProxyHandler({})),
        None,  # default opener (env-configured proxy)
    ]
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        opener = openers[attempt % len(openers)]
        try:
            req = urllib.request.Request(url, headers=UA)
            if opener is None:
                resp = urllib.request.urlopen(req, timeout=40)
            else:
                resp = opener.open(req, timeout=40)
            with resp:
                payload = resp.read()
            if not payload:
                raise OSError("empty response")
            return json.loads(payload.decode("utf-8-sig"))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(15.0, 1.0 * (2**attempt)) + random.uniform(0.0, 0.3))
    raise OSError(f"page {page} failed after {retries} attempts: {last_error}")


def fetch_all_rows(min_gap: float, max_gap: float, max_sites: int, page_size: int, workers: int) -> List[Dict[str, Any]]:
    del min_gap, max_gap, max_sites
    # Peek page 1: every key is "N of TOTAL" with the row as value; the total
    # is embedded in the first key.
    first = _request_page("", 1, page_size)
    total = None
    for key in first:
        if " of " in key:
            total = int(key.split()[-1])
            break
    if total is None:
        total = 120_000
    num_pages = (total + page_size - 1) // page_size
    print(f"[P0B] total rows ~{total}, pages {num_pages}", flush=True)
    rows: List[Dict[str, Any]] = []
    pending = list(range(1, num_pages + 1))

    def fetch(page: int) -> List[Dict[str, Any]]:
        payload = _request_page("", page, page_size)
        out = []
        for value in payload.values():
            if isinstance(value, dict) and value.get("auid"):
                out.append(value)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, page): page for page in pending}
        for future in concurrent.futures.as_completed(futures):
            page = futures[future]
            try:
                page_rows = future.result()
                rows.extend(page_rows)
                if len(rows) % (page_size * 20) < page_size * workers:
                    print(f"[P0B] {len(rows)} rows ...", flush=True)
            except Exception as exc:
                print(f"[P0B] page {page} failed: {exc}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill AFLOW metadata fields (P0B)")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--max-sites", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    existing = json.load(open(args.metadata, encoding="utf-8"))
    before_counts = {
        field: sum(
            1
            for rec in existing
            if rec.get(field) not in (None, "")
        )
        for field in BACKFILL_FIELDS
    }
    print(f"[P0B] existing records: {len(existing)}, coverage before: {before_counts}", flush=True)

    raw_rows = fetch_all_rows(
        args.min_gap, args.max_gap, args.max_sites, args.page_size, args.workers
    )
    incoming = [row for row in (row_to_backfill(r) for r in raw_rows) if row]
    print(f"[P0B] raw rows {len(raw_rows)} -> backfill rows {len(incoming)}", flush=True)

    updated, stats = merge_backfill_rows(existing, incoming)
    after_counts = {
        field: sum(1 for rec in updated if rec.get(field) not in (None, ""))
        for field in BACKFILL_FIELDS
    }
    print(f"[P0B] merge stats: {stats}", flush=True)
    print(f"[P0B] coverage after: {after_counts}", flush=True)

    backup = args.metadata + ".pre_backfill.bak"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False)
    with open(args.metadata, "w", encoding="utf-8") as fh:
        json.dump(updated, fh, ensure_ascii=False)
    print(f"[P0B] wrote {len(updated)} records to {args.metadata}", flush=True)


if __name__ == "__main__":
    main()
