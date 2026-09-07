"""P2 pre-flight fix: add per-atom species to the P1 structure sidecar.

The P1 sidecar's `species` field holds AFLOW's deduplicated element list
(e.g. ['Co','Sc','Zn']) while `fractional_coordinates` is per-atom (4 rows).
56,099/59,961 records have len(species) != len(fractional_coordinates), which
makes pymatgen `Structure(lattice, species, coords)` fail — a hard blocker
for the P2 crystal-graph encoder.

AFLOW's CONTCAR.relax (VASP POSCAR format) carries the authoritative per-atom
species order (header line + per-element counts). This script pulls it for the
mismatched records and adds a `species_per_atom` list WITHOUT touching the
existing `species` field (backward compatible; constitution §4 non-empty
precedence). The immutable band HDF5 is never modified.

Usage:
    python scripts/fix_sidecar_species.py \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --workers 10 [--limit N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
from typing import List, Optional

AFLOWDATA_BASE = "https://aflowlib.duke.edu/AFLOWDATA/"
UA = {"User-Agent": "BandStructure-AI/phase2 (+species fix)"}


def parse_poscar_species(text: str, known_species: Optional[List[str]] = None) -> Optional[List[str]]:
    """Return per-atom species list from a VASP POSCAR/CONTCAR.

    Lines: title, scale, 3x lattice, then (optionally species symbols), then
    (optionally) per-element counts, then 'Direct'/'Cartesian'. If the species
    symbols line is present, expand it by counts into the per-atom list.

    VASP4 has counts only (no symbols line); then `known_species` (the sidecar
    `species` dedup list, which AFLOW orders identically) supplies the symbols.
    """
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    if len(lines) < 6:
        return None
    mode_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        # Coordinate-mode line is exactly "Direct"/"Cartesian" (CONTCAR never
        # uses the bare "K" k-points mode). Do NOT prefix-match "k": potassium
        # compounds have title lines like "K_svSeSn/..." that lowercase to
        # "k_sv..." and would be mis-detected as the mode line.
        if s.startswith(("direct", "cartesian")):
            mode_idx = i
            break
    if mode_idx is None or mode_idx < 5:
        return None
    pre = lines[5:mode_idx]
    if not pre:
        return None
    sym_line = pre[0].split()
    counts = None
    if len(pre) >= 2:
        try:
            counts = [int(x) for x in pre[1].split()]
        except ValueError:
            counts = None
    if counts is None:
        # VASP5 style: species symbols only on the line before Direct, with
        # counts on the line before that.
        if len(pre) >= 2:
            try:
                counts = [int(x) for x in pre[0].split()]
                sym_line = pre[1].split()
            except ValueError:
                counts = None
    if counts is None:
        # VASP4 style: counts only (single line), no symbols.
        try:
            counts = [int(x) for x in sym_line]
            sym_line = known_species if known_species else None
        except ValueError:
            return None
    if sym_line is None or counts is None or len(counts) != len(sym_line):
        return None
    per_atom: List[str] = []
    for sym, n in zip(sym_line, counts):
        per_atom.extend([sym] * n)
    return per_atom


def fetch_species_per_atom(aurl: str, known_species: Optional[List[str]] = None, retries: int = 5) -> Optional[List[str]]:
    directory = aurl.split("AFLOWDATA/", 1)[-1].strip()
    url = AFLOWDATA_BASE + directory + "/CONTCAR.relax"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            resp = urllib.request.urlopen(req, timeout=30)
            with resp:
                text = resp.read().decode("utf-8", "replace")
            return parse_poscar_species(text, known_species=known_species)
        except Exception:
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.6 * (2 ** attempt)) + random.uniform(0.0, 0.15))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Add per-atom species to P1 sidecar")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    side = json.load(open(args.sidecar, encoding="utf-8"))
    targets = [r for r in side
               if r.get("species") is not None and r.get("fractional_coordinates") is not None
               and len(r["species"]) != len(r["fractional_coordinates"])
               and r.get("species_per_atom") is None]
    if args.limit:
        targets = targets[: args.limit]
    print(f"[FIX] total records {len(side)}, still-missing {len(targets)}", flush=True)

    by_id = {r["material_id"]: r for r in side}
    filled = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_species_per_atom, r["aurl"], r.get("species")): r["material_id"]
                for r in targets if r.get("aurl")}
        done = 0
        for future in concurrent.futures.as_completed(futs):
            mid = futs[future]
            try:
                per_atom = future.result()
                if per_atom is not None:
                    rec = by_id[mid]
                    if len(per_atom) == len(rec["fractional_coordinates"]):
                        rec["species_per_atom"] = per_atom
                        filled += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            done += 1
            if done % 2000 == 0:
                print(f"[FIX] {done}/{len(futs)} filled={filled} failed={failed}", flush=True)

    # coverage re-audit
    n_ok = sum(1 for r in side if r.get("species_per_atom"))
    print(f"[FIX] filled {filled}, failed {failed}, records with species_per_atom {n_ok}", flush=True)

    with open(args.sidecar, "w", encoding="utf-8") as fh:
        json.dump(side, fh, ensure_ascii=False)
    print(f"[FIX] wrote -> {args.sidecar}", flush=True)


if __name__ == "__main__":
    main()
