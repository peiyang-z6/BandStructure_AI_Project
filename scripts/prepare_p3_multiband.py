"""P3 data preparation: variable multi-band tensors + structure graphs.

Constitution 5.0 §8 P3. Reads complete multi-band energies from the raw AFLOW
HDF5 (which stores (num_bands, num_kpoints) per material — num_bands VARIABLE),
extracts Fermi-proximate bands onto a fixed k-grid with a band mask, and pairs
each with its crystal graph (from the P1 structure sidecar).

The outer OOD split is REUSED from the frozen tensor NPZ (no re-splitting):
material_ids_train / material_ids_test are read verbatim.

Output (per split NPZ):
    material_ids  (N,)                Unicode; never pickled in new outputs
    groups        (N,)                unchanged frozen-split group values/order
    bands         (N, max_bands, n_k) float32, zero-padded
    band_mask     (N, max_bands)      bool, True = real band
    k_axis        (n_k,)              normalized [0,1] k-grid (shared)
    segment_ids   (N, n_k)            int32; duplicate-distance boundaries, -1 invalid
    selected_band_indices (N, max_bands) int32; original flattened spin/band index, -1 padding
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
        --out-dir data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband_v2_20260908 \
        [--max-bands 16 --n-k 256 --delta-e 5.0 --max-atoms 50 --split both]

The output directory MUST be new: no overwrite or implicit resume. All source
IDs retain rows, including exclusions; --limit only builds a smoke prefix and
marks the other rows invalid (it does not reduce allocated output size).
Completion requires p3_{split}_complete.json (p3_split_completion v1) AND the
last-published p3_prepare_commit.json (p3_prepare_commit v1). The global v2
report is an audit, never read by training. --limit inputs require explicit
--smoke-limit > 0 downstream. Duplicate distances are NOT verified high-symmetry
points; no 3D reciprocal-path reconstruction or P3 physical loss is implemented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.crystal_graph import MAX_NEIGHBORS, build_crystal_graph
from src.data.multiband import extract_fermi_bands, path_segments, select_fermi_bands
from src.utils.selection_manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _atomic_write(path, writer, *, binary=False):
    """Publish only flushed complete files via a same-directory atomic rename."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        options = {} if binary else {"encoding": "utf-8"}
        with os.fdopen(fd, "wb" if binary else "w", **options) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SampleExclusion(ValueError):
    """A data-quality failure with a stable, machine-readable reason."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def load_sidecar_structures(sidecar_path: str) -> Dict[str, Dict]:
    """Keep identifiable records; validate per sample so failures are audited."""
    with open(sidecar_path, encoding="utf-8") as handle:
        side = json.load(handle)
    out: Dict[str, Dict] = {}
    for r in side:
        mid = r.get("material_id")
        if mid:
            out[mid] = r
    return out


def _aflow_efermi(metadata: Dict) -> float:
    """AFLOW energies are already Fermi-shifted to zero; return 0.0."""
    ref = str(metadata.get("energy_reference") or "")
    source = str(metadata.get("source") or "").lower()
    if source == "aflow" or ref == "fermi_shifted_zero":
        return 0.0
    # Without a known zero reference, a missing/nonfinite EF is not physical.
    e = metadata.get("efermi")
    if e is None or not np.isfinite(float(e)):
        raise ValueError("missing or nonfinite efermi for non-AFLOW source")
    return float(e)


def build_one(
    grp: h5py.Group,
    structures: Dict[str, Dict],
    max_bands: int,
    n_k: int,
    delta_e: float,
    max_atoms: int,
) -> Dict:
    material_id = grp.name.rsplit("/", 1)[-1]
    if "energies" not in grp:
        raise SampleExclusion("missing_energies", "HDF5 group has no energies dataset")
    try:
        if "k_distances" not in grp:
            raise ValueError("missing k_distances; cannot fabricate a physical axis")
        energies = np.asarray(grp["energies"], dtype=np.float32)
        k_dist = np.asarray(grp["k_distances"], dtype=np.float64)
        if (energies.ndim not in (2, 3) or energies.size == 0
                or not np.isfinite(energies).all()):
            raise ValueError("energies must be nonempty finite 2D or spin-resolved 3D")
        if k_dist.ndim != 1 or len(k_dist) != energies.shape[-1]:
            raise ValueError("k_distances must be 1D and match the band k dimension")
        source_segments, segment_ids = path_segments(k_dist, n_k)
        if np.any(np.bincount(source_segments) < 2):
            raise ValueError("low-quality segment: fewer than two source k points")
        metadata = dict(grp.attrs)
        if "metadata" in grp:
            metadata.update(dict(grp["metadata"].attrs))
        efermi = _aflow_efermi(metadata)
        bands, mask, k_axis = extract_fermi_bands(
            energies, k_dist, efermi=efermi, max_bands=max_bands,
            delta_e=delta_e, n_k=n_k,
        )
        flat = energies.reshape(-1, energies.shape[-1])
        indices = select_fermi_bands(flat, efermi, max_bands, delta_e)
        selected_band_indices = np.full(max_bands, -1, dtype=np.int32)
        selected_band_indices[:len(indices)] = indices
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise SampleExclusion("invalid_band_or_k", str(exc)) from exc

    rec = structures.get(material_id)
    if rec is None:
        raise SampleExclusion("missing_structure", "no sidecar record for material_id")
    species = rec.get("species_per_atom") or rec.get("species")
    if species is None or rec.get("lattice") is None or rec.get("fractional_coordinates") is None:
        raise SampleExclusion("missing_structure", "missing lattice, species or coordinates")
    if not isinstance(species, (list, tuple)):
        raise SampleExclusion("graph_error", "species must be a per-atom sequence")
    if len(species) > max_atoms:
        raise SampleExclusion("atom_capacity_exceeded",
                              f"{len(species)} atoms exceed max_atoms={max_atoms}; no truncation")
    try:
        lattice = np.asarray(rec["lattice"], dtype=float)
        coords = np.asarray(rec["fractional_coordinates"], dtype=float)
        if (lattice.shape != (3, 3) or not np.isfinite(lattice).all()
                or abs(np.linalg.det(lattice)) <= 1e-12):
            raise ValueError("lattice must be finite and nonsingular 3x3")
        if (not len(species) or coords.shape != (len(species), 3)
                or not np.isfinite(coords).all()):
            raise ValueError("invalid coordinates/species count")
        g = build_crystal_graph(lattice, species, coords, max_atoms=max_atoms)
        if np.any(g["atom_features"][:len(species)].sum(axis=1) == 0):
            raise ValueError("unencoded element: real atom has an all-zero onehot feature")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        raise SampleExclusion("graph_error", f"{type(exc).__name__}: {exc}") from exc

    return {
        "bands": bands,
        "mask": mask,
        "k_axis": k_axis,
        "segment_ids": segment_ids,
        "selected_band_indices": selected_band_indices,
        "source_segment_count": int(source_segments[-1]) + 1,
        "n_bands": int(mask.sum()),
        "graph": g,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P3 variable multi-band data")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--split-npz", required=True,
                        help="trusted project-built local frozen NPZ only (legacy object IDs allowed)")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-bands", type=int, default=16)
    parser.add_argument("--n-k", type=int, default=256)
    parser.add_argument("--delta-e", type=float, default=5.0)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke only: build first N per split; retain other rows as invalid")
    args = parser.parse_args()
    if (args.max_bands < 1 or args.n_k < 2 or args.max_atoms < 1 or args.limit < 0
            or not np.isfinite(args.delta_e) or args.delta_e < 0):
        parser.error("require max-bands/max-atoms >= 1, n-k >= 2, limit >= 0 and finite delta-e >= 0")

    # Exclusive directory creation also prevents concurrent builders/resume.
    # Even an empty pre-existing directory is not an authorized new snapshot.
    os.makedirs(args.out_dir, exist_ok=False)
    t0 = time.time()
    source_paths = {key: Path(getattr(args, key)) for key in ("h5", "split_npz", "sidecar")}
    sources = {key: {"path": os.path.relpath(path.resolve(), ROOT).replace(os.sep, "/"),
                     "sha256": sha256_file(path)} for key, path in source_paths.items()}
    code_hashes = {path: sha256_file(ROOT / path) for path in (
        "scripts/prepare_p3_multiband.py", "src/data/multiband.py",
        "src/data/crystal_graph.py", "src/utils/selection_manifest.py")}
    structures = load_sidecar_structures(args.sidecar)
    print(f"[P3] sidecar records: {len(structures)}", flush=True)

    # Only accept a trusted project-built local frozen split. Historical ID
    # arrays use pickle objects; never use this option for untrusted downloads.
    split_names = ("train", "test") if args.split == "both" else (args.split,)
    frozen_splits = []
    with np.load(args.split_npz, allow_pickle=True) as splits:
        for name in split_names:
            if any(key not in splits.files for key in (f"material_ids_{name}", f"groups_{name}")):
                raise ValueError(f"frozen split {name}: material_ids and groups are required")
            ids, groups = splits[f"material_ids_{name}"], splits[f"groups_{name}"]
            if (ids.ndim != 1 or groups.shape != ids.shape or groups.dtype.kind not in "iu"
                    or any(not isinstance(mid, str) or not mid for mid in ids)):
                raise ValueError(f"frozen split {name}: require 1D string IDs and aligned integer groups")
            if len(set(ids)) != len(ids):
                raise ValueError(f"frozen split {name}: duplicate material_ids")
            frozen_splits.append((name, ids.astype(str), groups))
    if args.split == "both":
        _, train_ids, train_groups = frozen_splits[0]
        _, test_ids, test_groups = frozen_splits[1]
        if set(train_ids) & set(test_ids):
            raise ValueError("Outer frozen split material ID overlap")
        if set(train_groups) & set(test_groups):
            raise ValueError("Outer frozen split group overlap")

    n_elements = 110
    report: Dict[str, object] = {"max_bands": args.max_bands, "n_k": args.n_k,
                               "delta_e": args.delta_e, "smoke_only": bool(args.limit)}
    report.update({
        "schema_version": 2,
        "parameters": {key: getattr(args, key) for key in
                       ("max_bands", "n_k", "delta_e", "max_atoms", "limit", "split")},
        "sources": sources, "code_hashes": code_hashes, "splits": {},
        "selected_band_indices": {
            "field": "selected_band_indices", "padding": -1,
            "source_indexing": "original band index; 3D energies flattened in C-order, spin-major",
        },
        "segmentation": {
            "boundary_source": "duplicate_k_distances",
            "no_boundary_policy": "single_unverified_segment",
            "physical_k_path_verified": False,
            "k_axis": "normalized cumulative distance, not 3D reciprocal coordinates",
            "high_symmetry_points_verified": False,
        },
    })

    with h5py.File(args.h5, "r") as f:
        for split, ids, groups in frozen_splits:
            ids = list(ids)
            n = len(ids)
            bands_arr = np.zeros((n, args.max_bands, args.n_k), dtype=np.float32)
            mask_arr = np.zeros((n, args.max_bands), dtype=bool)
            segment_ids = np.full((n, args.n_k), -1, dtype=np.int32)
            selected_indices = np.full((n, args.max_bands), -1, dtype=np.int32)
            n_bands_arr = np.zeros(n, dtype=np.int32)
            atom_features = np.zeros((n, args.max_atoms, n_elements), dtype=np.float32)
            neighbor_list = np.full((n, args.max_atoms, MAX_NEIGHBORS), -1, dtype=np.int32)
            neighbor_dist = np.zeros((n, args.max_atoms, MAX_NEIGHBORS), dtype=np.float32)
            n_atoms_arr = np.zeros(n, dtype=np.int32)
            valid = np.zeros(n, dtype=bool)
            source_segment_counts = np.zeros(n, dtype=np.int32)

            exclusions = []
            for i, mid in enumerate(ids):
                mid = str(mid)
                try:
                    if args.limit and i >= args.limit:
                        raise SampleExclusion("smoke_limit", "not processed: --limit smoke only")
                    if mid not in f:
                        raise SampleExclusion("missing_energies", "material_id absent from HDF5")
                    out = build_one(f[mid], structures, args.max_bands, args.n_k,
                                    args.delta_e, args.max_atoms)
                except SampleExclusion as exc:
                    exclusions.append({"material_id": mid, "split_index": i,
                                       "reason": exc.reason, "detail": exc.detail})
                    continue
                bands_arr[i] = out["bands"]
                mask_arr[i] = out["mask"]
                segment_ids[i] = out["segment_ids"]
                selected_indices[i] = out["selected_band_indices"]
                n_bands_arr[i] = out["n_bands"]
                source_segment_counts[i] = out["source_segment_count"]
                g = out["graph"]
                na = g["n_atoms"]
                atom_features[i, :na] = g["atom_features"][:na]
                neighbor_list[i, :na] = g["neighbor_list"][:na]
                neighbor_dist[i, :na] = g["neighbor_dist"][:na]
                n_atoms_arr[i] = g["n_atoms"]
                valid[i] = True

            k_axis = np.linspace(0.0, 1.0, args.n_k, dtype=np.float32)
            n_valid = int(valid.sum())
            counts, frequencies = np.unique(n_bands_arr[valid], return_counts=True)
            n_band_hist = {int(k): int(v) for k, v in zip(counts, frequencies)}
            exclusions_path = Path(args.out_dir) / f"p3_{split}_exclusions.json"
            _atomic_write(exclusions_path, lambda handle: json.dump(
                exclusions, handle, ensure_ascii=False, indent=2, allow_nan=False))
            out_path = os.path.join(args.out_dir, f"p3_{split}.npz")
            arrays = dict(
                material_ids=np.asarray(ids, dtype=str),
                groups=groups,
                segment_ids=segment_ids,
                selected_band_indices=selected_indices,
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
            _atomic_write(out_path, lambda handle: np.savez(handle, **arrays), binary=True)
            del arrays  # Do not retain the previous split's tensors while building the next.
            sha = sha256_file(out_path)
            report["splits"][split] = {
                "status": "complete" if n_valid else "zero_valid",
                "original_total": n, "total": n,
                "n_valid": n_valid, "n_excluded": len(exclusions),
                "exclusions_file": exclusions_path.name,
                "exclusions_sha256": sha256_file(exclusions_path),
                "exclusion_reason_counts": dict(Counter(e["reason"] for e in exclusions)),
                "band_hist": n_band_hist,
                "max_bands_fraction": (float(np.count_nonzero(n_bands_arr[valid] == args.max_bands))
                                       / n_valid if n_valid else None),
                "duplicate_distance_materials": int(np.count_nonzero(source_segment_counts[valid] > 1)),
                "single_unverified_segment_materials": int(np.count_nonzero(source_segment_counts[valid] == 1)),
                "output_npz": {"path": Path(out_path).name, "sha256": sha,
                               "bytes": Path(out_path).stat().st_size},
            }
            report[f"{split}_valid"] = n_valid
            report[f"{split}_total"] = n
            report[f"{split}_band_hist"] = dict(sorted(n_band_hist.items()))
            report[f"{split}_npz_sha256"] = sha
            print(f"[P3] {split}: {n_valid}/{n} valid, band hist {dict(sorted(n_band_hist.items()))}", flush=True)
            print(f"[P3] wrote {out_path} (sha {sha[:16]}...)", flush=True)

    for key, path in source_paths.items():
        if sha256_file(path) != sources[key]["sha256"]:
            raise RuntimeError(f"source {key} changed during preparation; no completion report")
    empty_splits = [name for name, stats in report["splits"].items() if not stats["n_valid"]]
    report["status"] = "failed" if empty_splits else "complete"
    if empty_splits:
        report["failure_reason"] = "zero_valid_split"
    _atomic_write(Path(args.out_dir) / "p3_prepare_report.json", lambda handle: json.dump(
        report, handle, ensure_ascii=False, indent=2, allow_nan=False))
    if empty_splits:
        raise RuntimeError(f"zero valid samples in split(s): {', '.join(empty_splits)}")
    # Split credentials expose no peer IDs, groups or sample statistics. Only
    # the final neutral commit authorizes them: orphan NPZ/credentials/reports
    # from failed or interrupted preparation are deliberately unusable.
    commit_path = Path(args.out_dir) / "p3_prepare_commit.json"
    try:
        completion_hashes = {}
        for split, stats in report["splits"].items():
            credential = {
                "schema": "p3_split_completion", "schema_version": 1,
                "status": "complete", "split": split,
                "output_npz": stats["output_npz"],
                **{key: report[key] for key in ("smoke_only", "parameters", "sources", "code_hashes")},
            }
            path = Path(args.out_dir) / f"p3_{split}_complete.json"
            _atomic_write(path, lambda handle: json.dump(
                credential, handle, ensure_ascii=False, indent=2, allow_nan=False))
            completion_hashes[split] = sha256_file(path)
        # Include mutations during report/credential publication in the gate.
        for key, path in source_paths.items():
            if sha256_file(path) != sources[key]["sha256"]:
                raise RuntimeError(f"source {key} changed during finalization; no completion commit")
        commit = {"schema": "p3_prepare_commit", "schema_version": 1,
                  "status": "complete", "completion_sha256": completion_hashes}
        _atomic_write(commit_path, lambda handle: json.dump(commit, handle, indent=2, allow_nan=False))
    except BaseException:
        # Also revoke a rename followed by a reported fsync/interruption error.
        commit_path.unlink(missing_ok=True)
        raise
    print(f"[P3] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
