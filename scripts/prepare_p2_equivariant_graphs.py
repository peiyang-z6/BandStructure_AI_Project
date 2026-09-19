"""P2 v3: precompute image-preserving equivariant graphs + descriptors.

The equivariant encoder needs edge-index graphs (src/dst/relative-vector) +
global descriptors, which are expensive to rebuild every training run. This
precomputes them once from the sidecar + pair NPZ and caches to disk.

Requires pymatgen + numpy; graph preparation does not import torch or TF.
Every run requires a NEW output directory. Legacy NPZ inputs remain read-only.
The fixed distance-sort/12-neighbour selection can split tied shells: diagnostics
record this limitation, but do not change the selection policy. Source-group and
training completion manifests belong to the caller's upstream pairing contract.

Usage:
    python scripts/prepare_p2_equivariant_graphs.py \
        --sidecar data/raw/aflow/snapshots/aflow_60000_20260831/aflow_structure_sidecar.json \
        --pairs data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs \
        --out data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/equivariant_graphs_v3_new \
        --max-atoms 50
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
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
GRAPH_SCHEMA_VERSION = "p2-periodic-images-v3"


def build_one_graph(rec: dict, max_atoms: int):
    """Return a single graph dict (flat arrays for one structure)."""
    from pymatgen.core import Lattice, Structure

    max_atoms = cg.validate_max_atoms(max_atoms)
    sp = rec.get("species_per_atom")
    if sp is None:
        sp = rec.get("species")
    frac = rec.get("fractional_coordinates")
    lat = rec.get("lattice")
    if sp is None or frac is None or lat is None:
        return None
    try:
        lat, frac = cg.validate_geometry(lat, frac)
        cg.validate_species(sp, len(frac))
    except (TypeError, ValueError):
        return None
    struct = Structure(Lattice(lat), list(sp), frac)
    na = len(struct)
    if na > max_atoms:
        return None

    onehot = cg.element_onehot(sp)
    atom_features = np.zeros((max_atoms, NUM_ELEMENTS + 1), dtype=np.float32)
    atom_features[:na] = onehot

    sg = rec.get("spacegroup_number") or rec.get("spacegroup_relax") or 0
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            desc = global_descriptors(lat, sp, int(sg))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(desc).all():
        return None

    edge_src, edge_dst, edge_vec, edge_len, edge_image = [], [], [], [], []
    neighbor_count = np.zeros(max_atoms, dtype=np.int64)
    distance_12_13 = np.zeros((max_atoms, 2), dtype=np.float32)
    boundary_tied = np.zeros(max_atoms, dtype=bool)
    nbrs = struct.get_all_neighbors(CUTOFF, include_index=True, numerical_tol=0.01)
    for i in range(na):
        ordered = sorted(nbrs[i], key=lambda x: x.nn_distance)
        neighbor_count[i] = len(ordered)
        for col, rank in enumerate((MAX_NEIGHBORS - 1, MAX_NEIGHBORS)):
            if len(ordered) > rank:
                distance_12_13[i, col] = ordered[rank].nn_distance
        if len(ordered) > MAX_NEIGHBORS:
            boundary_tied[i] = np.isclose(ordered[MAX_NEIGHBORS - 1].nn_distance,
                                          ordered[MAX_NEIGHBORS].nn_distance,
                                          rtol=1e-7, atol=1e-8)
        site_nbrs = ordered[:MAX_NEIGHBORS]  # preserve the existing selection rule

        for nbr in site_nbrs:
            j = nbr.index
            # PeriodicNeighbor.frac_coords already includes its lattice image.
            # Minimum-image wrapping here would erase distinct periodic edges.
            dfrac = np.asarray(nbr.frac_coords) - frac[i]
            vcart = dfrac @ lat
            edge_src.append(i)
            edge_dst.append(j)
            edge_vec.append(vcart)
            edge_len.append(np.linalg.norm(vcart))
            edge_image.append(nbr.image)

    return {
        "atom_features": atom_features,
        "desc": desc,
        "n_atoms": na,
        "neighbor_count": neighbor_count,
        "truncated_neighbors": np.maximum(neighbor_count - MAX_NEIGHBORS, 0),
        "distance_12_13": distance_12_13,
        "boundary_tied": boundary_tied,
        "edge_src": np.asarray(edge_src, dtype=np.int64),
        "edge_dst": np.asarray(edge_dst, dtype=np.int64),
        "edge_vec": np.asarray(edge_vec, dtype=np.float32) if edge_vec else np.zeros((0, 3), dtype=np.float32),
        "edge_len": np.asarray(edge_len, dtype=np.float32) if edge_len else np.zeros((0,), dtype=np.float32),
        "edge_image": np.asarray(edge_image, dtype=np.int64).reshape(-1, 3),
    }


def validate_pair_identity(material_ids, valid):
    """No casting, dropping, sorting or independent filtering of paired rows."""
    ids = np.asarray(material_ids)
    valid = np.asarray(valid)
    if (ids.ndim != 1 or ids.dtype.kind != "U"
            or any(not mid or mid != mid.strip() for mid in ids)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("material_ids must be unique, nonempty, one-dimensional strings")
    if valid.dtype != np.bool_ or valid.shape != ids.shape:
        raise ValueError("valid must be a boolean vector aligned with material_ids")


def _validate_cache_arrays(data):
    required = {"max_atoms", "max_neighbors", "cutoff_angstrom", "atom_features", "descriptors",
                "n_atoms", "edge_src", "edge_dst", "edge_vec", "edge_len", "edge_image",
                "edge_offsets", "valid_idx"}
    if not required.issubset(data):
        raise ValueError(f"graph cache is missing arrays: {sorted(required - data.keys())}")
    for key, expected in (("max_neighbors", MAX_NEIGHBORS), ("cutoff_angstrom", CUTOFF)):
        if data[key].shape != () or data[key].item() != expected:
            raise ValueError(f"graph cache {key} differs from the fixed contract")
    if data["max_atoms"].shape != ():
        raise ValueError("graph cache max_atoms must be scalar")
    capacity = cg.validate_max_atoms(data["max_atoms"].item())
    n, edges = len(data["valid"]), data["edge_len"].size
    shapes = {"atom_features": (n, capacity, NUM_ELEMENTS + 1), "descriptors": (n, 22),
              "n_atoms": (n,), "edge_src": (edges,), "edge_dst": (edges,),
              "edge_vec": (edges, 3), "edge_len": (edges,), "edge_image": (edges, 3),
              "edge_offsets": (n, 2), "valid_idx": (int(data["valid"].sum()),)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"graph cache {key} shape mismatch")
    for key in ("n_atoms", "edge_src", "edge_dst", "edge_image", "edge_offsets", "valid_idx"):
        if data[key].dtype.kind not in "iu":
            raise ValueError(f"graph cache {key} must be integer")
    for key in ("atom_features", "descriptors", "edge_vec", "edge_len"):
        if data[key].dtype.kind != "f" or not np.isfinite(data[key]).all():
            raise ValueError(f"graph cache {key} must be finite floating point")
    counts, valid = data["n_atoms"], data["valid"]
    if np.any(counts[valid] < 1) or np.any(counts > capacity) or np.any(counts[~valid] != 0):
        raise ValueError("graph cache n_atoms/valid mismatch")
    present = np.any(data["atom_features"] != 0, axis=-1)
    if not np.array_equal(present, np.arange(capacity)[None, :] < counts[:, None]):
        raise ValueError("graph cache real/padding atom rows mismatch")
    length = data["edge_len"]
    if (np.any(length <= 0) or np.any(length > CUTOFF + 1e-5)
            or not np.allclose(length, np.linalg.norm(data["edge_vec"], axis=-1), rtol=1e-5, atol=1e-6)):
        raise ValueError("graph cache edge lengths do not match vector norms/cutoff")
    previous = 0
    for b, (lo, hi) in enumerate(data["edge_offsets"]):
        if lo != previous or hi < lo or hi > edges or (not valid[b] and hi != lo):
            raise ValueError("graph cache edge_offsets/valid mismatch")
        src, dst = data["edge_src"][lo:hi], data["edge_dst"][lo:hi]
        if np.any(src < 0) or np.any(dst < 0) or np.any(src >= counts[b]) or np.any(dst >= counts[b]):
            raise ValueError("graph cache edge index outside real atoms")
        if np.any(np.bincount(src.astype(np.int64), minlength=int(counts[b])) > MAX_NEIGHBORS):
            raise ValueError("graph cache exceeds 12 selected neighbors")
        previous = hi
    if previous != edges:
        raise ValueError("graph cache has unassigned edges")


def load_graph_cache(path, *, expected_material_ids, expected_valid):
    """Read-only, fail-closed pairing gate for consumers of the new graph cache.

    Both expected arrays refer to the FULL raw row order, before filtering.
    A graph quarantine that differs from the embedding mask requires explicit
    re-pairing by the caller, never an independent positional filter.
    """
    with np.load(path, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    return validate_graph_cache(data, expected_material_ids=expected_material_ids, expected_valid=expected_valid)


def validate_graph_cache(data, *, expected_material_ids, expected_valid):
    """Apply the v3 gate to an already authenticated, decoded snapshot."""
    if np.asarray(data.get("schema_version")).shape != () or data.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported or missing graph schema_version; legacy NPZ is read-only")
    for key in ("builder_sha256", "crystal_graph_sha256", "descriptors_sha256"):
        value = data.get(key)
        if (value is None or np.asarray(value).shape != ()
                or not isinstance(value.item(), str) or len(value.item()) != 64
                or any(c not in "0123456789abcdef" for c in value.item())):
            raise ValueError(f"missing or malformed {key}")
    validate_pair_identity(data["material_ids"], data["valid"])
    validate_pair_identity(data["material_ids"], data["pair_valid"])
    if (np.any(data["valid"] & ~data["pair_valid"])
            or not np.array_equal(data.get("valid_idx"), np.flatnonzero(data["valid"]))
            or not np.array_equal(data.get("valid_material_ids"), data["material_ids"][data["valid"]])):
        raise ValueError("inconsistent valid row mapping in graph cache")
    _validate_cache_arrays(data)
    validate_pair_identity(expected_material_ids, expected_valid)
    if not np.array_equal(data["material_ids"], expected_material_ids):
        raise ValueError("graph material_ids differ from the expected raw row order")
    if not np.array_equal(data["valid"], expected_valid):
        raise ValueError("graph valid mask differs from the paired embedding mask")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute equivariant graphs")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-atoms", type=int, default=50)
    args = parser.parse_args()
    cg.validate_max_atoms(args.max_atoms)

    # Exclusive directory creation also rejects empty legacy locations and races.
    os.makedirs(args.out, exist_ok=False)
    with open(args.sidecar, encoding="utf-8") as fh:
        records = json.load(fh)
    sidecar = {}
    for rec in records:
        mid = rec["material_id"]
        if mid in sidecar:
            raise ValueError(f"duplicate sidecar material_id: {mid}")
        sidecar[mid] = rec

    for split in ("train", "test"):
        with np.load(os.path.join(args.pairs, f"p2_{split}.npz"), allow_pickle=False) as p:
            ids = p["material_ids"]
            pair_valid = p["valid"]
        validate_pair_identity(ids, pair_valid)
        n = len(ids)
        atom_features = np.zeros((n, args.max_atoms, NUM_ELEMENTS + 1), dtype=np.float32)
        desc = np.zeros((n, 22), dtype=np.float32)
        n_atoms = np.zeros(n, dtype=np.int64)
        valid = np.zeros(n, dtype=bool)
        ledger = {
            "neighbor_count": np.zeros((n, args.max_atoms), dtype=np.int64),
            "truncated_neighbors": np.zeros((n, args.max_atoms), dtype=np.int64),
            "distance_12_13": np.zeros((n, args.max_atoms, 2), dtype=np.float32),
            "boundary_tied": np.zeros((n, args.max_atoms), dtype=bool),
        }
        edge_src, edge_dst, edge_vec, edge_len, edge_image = [], [], [], [], []
        offsets = []  # per-graph edge offsets for later batching
        offset = 0
        for b, mid in enumerate(ids):
            mid = str(mid)
            rec = sidecar.get(mid)
            g = build_one_graph(rec, args.max_atoms) if pair_valid[b] and rec else None
            if g is None:
                valid[b] = False
                offsets.append((offset, offset))
                continue
            valid[b] = True
            atom_features[b] = g["atom_features"]
            desc[b] = g["desc"]
            n_atoms[b] = g["n_atoms"]
            for key in ledger:
                ledger[key][b] = g[key]
            # Cache INTRA-graph indices (0..na-1); batch collation, not this
            # writer, rebases them into the merged node tensor.
            edge_src.append(g["edge_src"])
            edge_dst.append(g["edge_dst"])
            edge_vec.append(g["edge_vec"])
            edge_len.append(g["edge_len"])
            edge_image.append(g["edge_image"])
            offsets.append((offset, offset + len(g["edge_src"])))
            offset += len(g["edge_src"])

        root = Path(__file__).resolve().parents[1]
        out = {
            **ledger,
            "schema_version": np.asarray(GRAPH_SCHEMA_VERSION),
            "cutoff_angstrom": np.asarray(CUTOFF),
            "max_neighbors": np.asarray(MAX_NEIGHBORS),
            "max_atoms": np.asarray(args.max_atoms),
            "builder_sha256": np.asarray(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            "crystal_graph_sha256": np.asarray(hashlib.sha256(Path(cg.__file__).read_bytes()).hexdigest()),
            "descriptors_sha256": np.asarray(hashlib.sha256(
                (root / "src/data/structure_descriptors.py").read_bytes()).hexdigest()),
            "edge_image": np.concatenate(edge_image) if edge_image else np.zeros((0, 3), dtype=np.int64),
            "atom_features": atom_features,
            "descriptors": desc,
            "n_atoms": n_atoms,
            "valid": valid,
            "pair_valid": pair_valid,
            "valid_idx": np.flatnonzero(valid),
            "valid_material_ids": ids[valid],
            "edge_src": np.concatenate(edge_src) if edge_src else np.zeros((0,), dtype=np.int64),
            "edge_dst": np.concatenate(edge_dst) if edge_dst else np.zeros((0,), dtype=np.int64),
            "edge_vec": np.concatenate(edge_vec) if edge_vec else np.zeros((0, 3), dtype=np.float32),
            "edge_len": np.concatenate(edge_len) if edge_len else np.zeros((0,), dtype=np.float32),
            "edge_offsets": np.asarray(offsets, dtype=np.int64).reshape(-1, 2),
            "material_ids": ids,
        }
        path = os.path.join(args.out, f"graphs_{split}.npz")
        with open(path, "xb") as fh:
            np.savez_compressed(fh, **out)
        n_valid = int(valid.sum())
        print(f"[EQG] {split}: {n} ids, {n_valid} valid, {out['edge_src'].shape[0]} edges -> {path}", flush=True)


if __name__ == "__main__":
    main()
