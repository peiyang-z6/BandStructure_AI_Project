"""P3 data preparation: variable multi-band tensors + structure graphs.

Constitution 5.0 §8 P3. Reads complete multi-band energies from the raw AFLOW
HDF5 (which stores (num_bands, num_kpoints) per material — num_bands VARIABLE),
extracts Fermi-proximate bands onto a fixed k-grid with a band mask, and pairs
each with its crystal graph (from the P1 structure sidecar).

The outer OOD split is REUSED from the frozen tensor NPZ (no re-splitting):
material_ids_train / material_ids_test are read verbatim.

Output (per split NPZ):
    material_ids  (N,)
    bands         (N, max_bands, n_k) float32, zero-padded
    band_mask     (N, max_bands)      bool, True = real band
    k_axis        (n_k,)              normalized [0,1] k-grid (shared)
    n_bands       (N,)                real band count per material
    atom_features (N, max_atoms, 110)
    neighbor_list (N, max_atoms, 12)  int32, -1 padding
    neighbor_dist (N, max_atoms, 12)  float32
    n_atoms       (N,)
    valid         (N,)                bool (structure + bands both present)

Usage:
    python scripts/prepare_p3_multiband.py \
        --h5 data/raw/aflow/snapshots/aflow_60000_20260831/aflow_bands.h5 \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --split-npz data/processed/aflow/ood_tensors_v7_60000_seed42/band_tensors_ood_split.npz \
        --out-dir data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband \
        [--max-bands 16 --n-k 256 --delta-e 5.0 --max-atoms 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.crystal_graph import MAX_NEIGHBORS, build_crystal_graph
from src.data.multiband import extract_fermi_bands


def load_sidecar_structures(sidecar_path: str) -> Dict[str, Dict]:
    from pymatgen.core import Lattice, Structure

    side = json.load(open(sidecar_path, encoding="utf-8"))
    out: Dict[str, Dict] = {}
    for r in side:
        mid = r.get("material_id")
        sp = r.get("species_per_atom") or r.get("species")
        fc = r.get("fractional_coordinates")
        lat = r.get("lattice")
        if not mid or sp is None or fc is None or lat is None:
            continue
        if len(sp) != len(fc):
            continue
        try:
            Structure(Lattice(lat), sp, fc)
        except Exception:
            continue
        out[mid] = r
    return out


def _aflow_efermi(metadata: Dict) -> float:
    """AFLOW energies are already Fermi-shifted to zero; return 0.0."""
    ref = str(metadata.get("energy_reference") or "")
    source = str(metadata.get("source") or "").lower()
    if source == "aflow" or ref == "fermi_shifted_zero":
        return 0.0
    # non-AFLOW fallback: use the group efermi if finite, else 0.0
    e = metadata.get("efermi")
    return float(e) if e is not None and np.isfinite(float(e)) else 0.0


def build_one(
    grp: h5py.Group,
    structures: Dict[str, Dict],
    max_bands: int,
    n_k: int,
    delta_e: float,
    max_atoms: int,
) -> Optional[Dict]:
    material_id = grp.name.rsplit("/", 1)[-1]
    if "energies" not in grp:
        return None
    energies = np.asarray(grp["energies"], dtype=np.float32)
    k_dist = (
        np.asarray(grp["k_distances"], dtype=np.float64).reshape(-1)
        if "k_distances" in grp
        else np.linspace(0.0, 1.0, energies.shape[1])
    )
    metadata = dict(grp.attrs)
    if "metadata" in grp:
        metadata.update(dict(grp["metadata"].attrs))
    efermi = _aflow_efermi(metadata)

    try:
        bands, mask, k_axis = extract_fermi_bands(
            energies, k_dist, efermi=efermi, max_bands=max_bands,
            delta_e=delta_e, n_k=n_k,
        )
    except Exception:
        return None

    rec = structures.get(material_id)
    if rec is None:
        return None
    g = build_crystal_graph(
        rec["lattice"], rec.get("species_per_atom") or rec["species"],
        rec["fractional_coordinates"], max_atoms=max_atoms,
    )

    return {
        "bands": bands,
        "mask": mask,
        "k_axis": k_axis,
        "n_bands": int(mask.sum()),
        "graph": g,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P3 variable multi-band data")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--split-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-bands", type=int, default=16)
    parser.add_argument("--n-k", type=int, default=256)
    parser.add_argument("--delta-e", type=float, default=5.0)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="debug: only process N materials per split")
    args = parser.parse_args()

    t0 = time.time()
    structures = load_sidecar_structures(args.sidecar)
    print(f"[P3] sidecar structures: {len(structures)}", flush=True)

    splits = np.load(args.split_npz)
    id_train = splits["material_ids_train"]
    id_test = splits["material_ids_test"]
    print(f"[P3] split: train {len(id_train)}, test {len(id_test)}", flush=True)

    n_elements = 110
    report: Dict[str, object] = {"max_bands": args.max_bands, "n_k": args.n_k, "delta_e": args.delta_e}

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        for split, ids in (("train", id_train), ("test", id_test)):
            ids = list(ids)
            if args.limit:
                ids = ids[: args.limit]
            n = len(ids)
            bands_arr = np.zeros((n, args.max_bands, args.n_k), dtype=np.float32)
            mask_arr = np.zeros((n, args.max_bands), dtype=bool)
            n_bands_arr = np.zeros(n, dtype=np.int32)
            atom_features = np.zeros((n, args.max_atoms, n_elements), dtype=np.float32)
            neighbor_list = np.full((n, args.max_atoms, MAX_NEIGHBORS), -1, dtype=np.int32)
            neighbor_dist = np.zeros((n, args.max_atoms, MAX_NEIGHBORS), dtype=np.float32)
            n_atoms_arr = np.zeros(n, dtype=np.int32)
            valid = np.zeros(n, dtype=bool)

            n_band_hist: Dict[int, int] = {}
            for i, mid in enumerate(ids):
                mid = str(mid)
                if mid not in f:
                    continue
                out = build_one(f[mid], structures, args.max_bands, args.n_k, args.delta_e, args.max_atoms)
                if out is None:
                    continue
                valid[i] = True
                bands_arr[i] = out["bands"]
                mask_arr[i] = out["mask"]
                n_bands_arr[i] = out["n_bands"]
                n_band_hist[out["n_bands"]] = n_band_hist.get(out["n_bands"], 0) + 1
                g = out["graph"]
                na = min(g["n_atoms"], args.max_atoms)
                atom_features[i, :na] = g["atom_features"][:na]
                neighbor_list[i, :na] = g["neighbor_list"][:na]
                neighbor_dist[i, :na] = g["neighbor_dist"][:na]
                n_atoms_arr[i] = g["n_atoms"]

            k_axis = np.linspace(0.0, 1.0, args.n_k, dtype=np.float32)
            n_valid = int(valid.sum())
            out_path = os.path.join(args.out_dir, f"p3_{split}.npz")
            np.savez(
                out_path,
                material_ids=np.asarray(ids, dtype=object),
                bands=bands_arr,
                band_mask=mask_arr,
                k_axis=k_axis,
                n_bands=n_bands_arr,
                atom_features=atom_features,
                neighbor_list=neighbor_list,
                neighbor_dist=neighbor_dist,
                n_atoms=n_atoms_arr,
                valid=valid,
            )
            # SHA-256 of the bands array for provenance
            sha = hashlib.sha256(bands_arr.tobytes()).hexdigest()
            report[f"{split}_valid"] = n_valid
            report[f"{split}_total"] = n
            report[f"{split}_band_hist"] = dict(sorted(n_band_hist.items()))
            report[f"{split}_bands_sha256"] = sha
            print(f"[P3] {split}: {n_valid}/{n} valid, band hist {dict(sorted(n_band_hist.items()))}", flush=True)
            print(f"[P3] wrote {out_path} (sha {sha[:16]}...)", flush=True)

    with open(os.path.join(args.out_dir, "p3_prepare_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[P3] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
