"""P2 paired-data preparation: align structure sidecar with band tensors.

Constitution 5.0 §8 P2 needs (structure, band) pairs for contrastive
learning. This builds:
- a material_id -> structure record map from the P1 sidecar;
- the band tensor array (flattened (N, 128, 6) input) from the frozen NPZ;
- crystal graphs for every paired material, cached to a single .npz so the
  contrastive trainer does not rebuild graphs each run.

Only materials present in BOTH the sidecar (with a pymatgen-constructible
structure) and the tensor NPZ are kept. The outer OOD split is respected:
graphs are built for train and test material_id sets separately, keyed by the
NPZ's material_ids arrays (no re-splitting).

Usage:
    python scripts/prepare_p2_pairs.py \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --tensor-npz data/processed/aflow/ood_tensors_v7_60000_seed42/band_tensors_ood_split.npz \
        --out-dir data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs \
        [--max-atoms 128]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.crystal_graph import (
    MAX_NEIGHBORS,
    build_crystal_graph,
)


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
            Structure(Lattice(lat), sp, fc)  # validate
        except Exception:
            continue
        out[mid] = r
    return out


def build_graphs_for_ids(
    structures: Dict[str, Dict],
    material_ids: np.ndarray,
    max_atoms: int,
) -> Dict[str, np.ndarray]:
    """Build graphs for the material_ids in order, returning batched arrays."""
    n = len(material_ids)
    n_elements = 110  # 109 elements + padding
    atom_features = np.zeros((n, max_atoms, n_elements), dtype=np.float32)
    neighbor_list = np.full((n, max_atoms, MAX_NEIGHBORS), -1, dtype=np.int32)
    neighbor_dist = np.zeros((n, max_atoms, MAX_NEIGHBORS), dtype=np.float32)
    n_atoms = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)

    for i, mid in enumerate(material_ids):
        mid = str(mid)
        rec = structures.get(mid)
        if rec is None:
            valid[i] = False
            continue
        g = build_crystal_graph(
            rec["lattice"], rec.get("species_per_atom") or rec["species"],
            rec["fractional_coordinates"], max_atoms=max_atoms,
        )
        na = min(g["n_atoms"], max_atoms)
        atom_features[i, :na] = g["atom_features"][:na]
        neighbor_list[i, :na] = g["neighbor_list"][:na]
        neighbor_dist[i, :na] = g["neighbor_dist"][:na]
        n_atoms[i] = g["n_atoms"]

    return {
        "atom_features": atom_features,
        "neighbor_list": neighbor_list,
        "neighbor_dist": neighbor_dist,
        "n_atoms": n_atoms,
        "valid": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P2 structure-band pairs")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--tensor-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-atoms", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    structures = load_sidecar_structures(args.sidecar)
    print(f"[P2] sidecar structures: {len(structures)}", flush=True)

    tensors = np.load(args.tensor_npz)
    train_ids = tensors["material_ids_train"]
    test_ids = tensors["material_ids_test"]
    X_train = tensors["X_train"]  # (N, 2, 128, 3)
    X_test = tensors["X_test"]
    print(f"[P2] tensors train {X_train.shape}, test {X_test.shape}", flush=True)

    # Flatten (N, 2, 128, 3) -> (N, 128, 6) to match the band encoder input.
    def flatten(X: np.ndarray) -> np.ndarray:
        return np.transpose(X, (0, 2, 1, 3)).reshape(X.shape[0], X.shape[2], -1)

    X_train_flat = flatten(X_train)
    X_test_flat = flatten(X_test)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids, xflat in (("train", train_ids, X_train_flat), ("test", test_ids, X_test_flat)):
        graphs = build_graphs_for_ids(structures, ids, args.max_atoms)
        n_valid = int(graphs["valid"].sum())
        print(f"[P2] {name}: {len(ids)} ids, {n_valid} with graphs", flush=True)
        out_path = os.path.join(args.out_dir, f"p2_{name}.npz")
        np.savez(
            out_path,
            material_ids=ids,
            band_input=xflat,
            atom_features=graphs["atom_features"],
            neighbor_list=graphs["neighbor_list"],
            neighbor_dist=graphs["neighbor_dist"],
            n_atoms=graphs["n_atoms"],
            valid=graphs["valid"],
        )
        print(f"[P2] wrote {out_path}", flush=True)

    print(f"[P2] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
