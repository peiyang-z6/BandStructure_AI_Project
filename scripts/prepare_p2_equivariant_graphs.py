"""P2 v2 step 2: precompute equivariant graphs + descriptors (PyTorch side).

The equivariant encoder needs edge-index graphs (src/dst/relative-vector) +
global descriptors, which are expensive to rebuild every training run. This
precomputes them once from the sidecar + pair NPZ and caches to disk.

Runs in the `nequip` env (pymatgen + numpy only, no torch needed at build
time beyond tensor packing).

Usage:
    python scripts/prepare_p2_equivariant_graphs.py \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --pairs data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs \
        --out data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/equivariant_graphs \
        --max-atoms 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.data.crystal_graph as cg
from src.data.structure_descriptors import global_descriptors

NUM_ELEMENTS = 109
MAX_NEIGHBORS = 12
CUTOFF = 8.0


def build_one_graph(rec: dict, max_atoms: int):
    """Return a single graph dict (flat arrays for one structure)."""
    from pymatgen.core import Lattice, Structure

    sp = rec.get("species_per_atom") or rec.get("species")
    frac = np.asarray(rec["fractional_coordinates"], dtype=float)
    lat = np.asarray(rec["lattice"], dtype=float)
    if len(sp) != len(frac):
        return None
    struct = Structure(Lattice(lat), list(sp), frac)
    na = len(struct)
    if na > max_atoms:
        return None

    onehot = cg.element_onehot(sp)
    atom_features = np.zeros((max_atoms, NUM_ELEMENTS + 1), dtype=np.float32)
    atom_features[:na] = onehot

    sg = rec.get("spacegroup_number") or rec.get("spacegroup_relax") or 0
    desc = global_descriptors(lat, sp, int(sg))

    edge_src, edge_dst, edge_vec, edge_len = [], [], [], []
    nbrs = struct.get_all_neighbors(CUTOFF, include_index=True, numerical_tol=0.01)
    for i in range(na):
        site_nbrs = sorted(nbrs[i], key=lambda x: x.nn_distance)[:MAX_NEIGHBORS]
        for nbr in site_nbrs:
            j = nbr.index
            dfrac = np.asarray(nbr.frac_coords) - frac[i]
            dfrac -= np.round(dfrac)
            vcart = dfrac @ lat
            edge_src.append(i)
            edge_dst.append(j)
            edge_vec.append(vcart)
            edge_len.append(np.linalg.norm(vcart))

    return {
        "atom_features": atom_features,
        "desc": desc,
        "n_atoms": na,
        "edge_src": np.asarray(edge_src, dtype=np.int64),
        "edge_dst": np.asarray(edge_dst, dtype=np.int64),
        "edge_vec": np.asarray(edge_vec, dtype=np.float32) if edge_vec else np.zeros((0, 3), dtype=np.float32),
        "edge_len": np.asarray(edge_len, dtype=np.float32) if edge_len else np.zeros((0,), dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute equivariant graphs")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-atoms", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sidecar = {r["material_id"]: r for r in json.load(open(args.sidecar, encoding="utf-8"))}

    for split in ("train", "test"):
        p = np.load(os.path.join(args.pairs, f"p2_{split}.npz"))
        ids = p["material_ids"]
        n = len(ids)
        atom_features = np.zeros((n, args.max_atoms, NUM_ELEMENTS + 1), dtype=np.float32)
        desc = np.zeros((n, 22), dtype=np.float32)
        n_atoms = np.zeros(n, dtype=np.int64)
        valid = np.zeros(n, dtype=bool)
        edge_src, edge_dst, edge_vec, edge_len = [], [], [], []
        offsets = []  # per-graph edge offsets for later batching
        offset = 0
        for b, mid in enumerate(ids):
            mid = str(mid)
            rec = sidecar.get(mid)
            g = build_one_graph(rec, args.max_atoms) if rec else None
            if g is None:
                valid[b] = False
                offsets.append((offset, offset))
                continue
            valid[b] = True
            atom_features[b] = g["atom_features"]
            desc[b] = g["desc"]
            n_atoms[b] = g["n_atoms"]
            # keep edges as INTRA-graph indices (0..na-1); the training loop
            # runs one graph at a time with gradient accumulation, so edges
            # must reference atoms within a single graph, not a batch.
            edge_src.append(g["edge_src"])
            edge_dst.append(g["edge_dst"])
            edge_vec.append(g["edge_vec"])
            edge_len.append(g["edge_len"])
            offsets.append((offset, offset + len(g["edge_src"])))
            offset += len(g["edge_src"])

        out = {
            "atom_features": atom_features,
            "descriptors": desc,
            "n_atoms": n_atoms,
            "valid": valid,
            "edge_src": np.concatenate(edge_src) if edge_src else np.zeros((0,), dtype=np.int64),
            "edge_dst": np.concatenate(edge_dst) if edge_dst else np.zeros((0,), dtype=np.int64),
            "edge_vec": np.concatenate(edge_vec) if edge_vec else np.zeros((0, 3), dtype=np.float32),
            "edge_len": np.concatenate(edge_len) if edge_len else np.zeros((0,), dtype=np.float32),
            "edge_offsets": np.asarray(offsets, dtype=np.int64),
            "material_ids": ids,
        }
        path = os.path.join(args.out, f"graphs_{split}.npz")
        np.savez_compressed(path, **out)
        n_valid = int(valid.sum())
        print(f"[EQG] {split}: {n} ids, {n_valid} valid, {out['edge_src'].shape[0]} edges -> {path}", flush=True)


if __name__ == "__main__":
    main()
