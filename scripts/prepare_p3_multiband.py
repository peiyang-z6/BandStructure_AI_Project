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

NEW prepare outputs also contain p3_{split}_audit.json (p3_prepare_audit v1):
one row per frozen ID, with full-raw capacity and actual segment resampling
diagnostics for prepared-valid rows; exclusions/unprocessed rows remain explicit
and have null diagnostics. This ledger does not alter selection, QA, or any NPZ
array, and is not scientific acceptance. Its pre-publication bytes/SHA are bound
by an additive preparation_audit field in the unchanged v1 split credential;
readers must not follow provenance paths or infer physical certification. Legacy
credentials do not acquire a ledger. Bad audits abort rather than exclude data;
finalization failures revoke completion credentials/commit and the success report.

Source-only audit (no tensor/graph preparation or sidecar read):
    python scripts/prepare_p3_multiband.py --mode source-qa \
        --h5 SNAPSHOT.h5 --split-npz FROZEN_SPLIT.npz --split train --out-dir NEW_DIR
Use a separate NEW_DIR and explicit --split test for outer read-only auditing;
never combine its statistics into selection/QA-rule tuning. No --limit here:
for bounded tests supply a separately identified fixture split. The ledger is
NOT a training completion credential. All arrays retain their original energy
reference; declared legacy AFLOW False does not verify nonmagnetic spin.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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

from src.data.aflow_adapter import assess_source_quality, SourceQualityError, SOURCE_QA_RULE_VERSION
from src.data.crystal_graph import MAX_NEIGHBORS, build_crystal_graph
from src.data.multiband import band_window_audit, extract_fermi_bands, path_segments, select_fermi_bands
from src.evaluation.multiband_metrics import resampling_audit
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


def read_hdf5_source_record(grp):
    """Read one immutable group; never trust a persisted source_quality verdict."""
    record = dict(grp["metadata"].attrs) if "metadata" in grp else {}
    nested_metadata = dict(record)
    record.update(dict(grp.attrs))
    record["metadata"] = nested_metadata
    for key, value in list(record.items()):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        record[key] = value.item() if isinstance(value, np.generic) else value
    record["material_id"] = grp.name.rsplit("/", 1)[-1]
    for key in ("energies", "k_distances", "kpoints"):
        if key in grp:
            record[key] = grp[key][...]
    return record


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
        source_record = read_hdf5_source_record(grp)
        source_quality = assess_source_quality(source_record)
        if source_quality["status"] == "invalid":
            if source_record.get("source") == "aflow":
                raise SourceQualityError(source_quality)
            raise ValueError(source_quality["reason"])
        if source_quality["status"] != "verified_valid":
            raise SampleExclusion("source_ambiguous", json.dumps(source_quality, allow_nan=False))
        energies = np.asarray(source_record["energies"], dtype=np.float32)
        k_dist = np.asarray(source_record["k_distances"], dtype=np.float64)
        if (energies.ndim not in (2, 3) or energies.size == 0
                or not np.isfinite(energies).all()):
            raise ValueError("energies must be nonempty finite 2D or spin-resolved 3D")
        if k_dist.ndim != 1 or len(k_dist) != energies.shape[-1]:
            raise ValueError("k_distances must be 1D and match the band k dimension")
        source_segments, segment_ids = path_segments(k_dist, n_k)
        if np.any(np.bincount(source_segments) < 2):
            raise ValueError("low-quality segment: fewer than two source k points")
        efermi = _aflow_efermi(source_record)
        bands, mask, k_axis = extract_fermi_bands(
            energies, k_dist, efermi=efermi, max_bands=max_bands,
            delta_e=delta_e, n_k=n_k,
        )
        flat = energies.reshape(-1, energies.shape[-1])
        indices = select_fermi_bands(flat, efermi, max_bands, delta_e)
        selected_band_indices = np.full(max_bands, -1, dtype=np.int32)
        selected_band_indices[:len(indices)] = indices
    except SampleExclusion:
        raise
    except SourceQualityError as exc:
        raise SampleExclusion(exc.quality["reason"], json.dumps(exc.quality, allow_nan=False)) from exc
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

    # QA has validated equal spin blocks. Restore that grouping for diagnostics
    # only; keep the actual selector, flattened indices and output arrays intact.
    audit_energies = source_record["energies"]
    if source_quality["layout"] == "spin_major_flat":
        audit_energies = audit_energies.reshape(
            source_quality["spin_channels"], -1, audit_energies.shape[-1])
    return {
        "audit_source": {
            "hdf5_group": grp.name, "provider": source_record.get("source"),
            "declared_source_sha256": source_record.get("source_sha256"),
            "source_url": source_record.get("source_url") or source_record.get("download_url"),
            "quality": source_quality,
            "raw_energy_shape": list(source_record["energies"].shape),
            "raw_energy_dtype": str(source_record["energies"].dtype),
            "audit_energy_shape": list(audit_energies.shape),
            "efermi_used_eV": efermi,
            "source_efermi_absolute": source_record.get("source_efermi_absolute"),
        },
        "band_window_audit": band_window_audit(
            audit_energies, selected_band_indices[mask], efermi=efermi,
            max_bands=max_bands, delta_e=delta_e),
        "resampling_audit": resampling_audit(
            audit_energies, selected_band_indices[mask], bands, mask,
            source_segments, segment_ids, efermi=efermi),
        "bands": bands,
        "mask": mask,
        "k_axis": k_axis,
        "segment_ids": segment_ids,
        "source_segment_ids": source_segments,  # Internal audit seam; not a new NPZ array.
        "selected_band_indices": selected_band_indices,
        "source_segment_count": int(source_segments[-1]) + 1,
        "n_bands": int(mask.sum()),
        "graph": g,
    }


def _validate_preparation_audits(out, max_bands, n_k):
    """Fail a broken diagnostic as a build error, never as a sample exclusion.

    Check the actual array/ledger seam and denominator identities, not a second
    selection algorithm or a new source-quality/physical acceptance policy.
    """
    def count(value, upper):
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError("audit count must be a nonnegative integer within its denominator")
        return value

    try:
        window, resampled = out["band_window_audit"], out["resampling_audit"]
        json.dumps([window, resampled, out["audit_source"]], allow_nan=False)
        selected = out["selected_band_indices"][out["mask"]].tolist()
        raw_shape = out["audit_source"]["raw_energy_shape"]
        raw_count = int(np.prod(raw_shape[:-1]))
        for upper, keys in (
                (raw_count, ("raw_band_count", "window_candidates", "selection_candidates")),
                (len(selected), ("selected", "selected_in_window", "selected_outside_window")),
                (raw_count - len(selected), ("window_dropped", "selection_dropped",
                                             "straddling_intervals_dropped", "capacity_shortfall")),
                (max_bands - len(selected), ("quota_underfill",))):
            for key in keys:
                count(window[key], upper)
        if (window["scope"] != "selected_window_complete_trajectories_not_energy_clipped"
                or resampled["causal_attribution"] != "not_established"):
            raise ValueError("missing diagnostic scope or causal-attribution boundary")
        if not isinstance(window["selected_indices"], list):
            raise ValueError("selected indices must be an integer list")
        for index in window["selected_indices"]:
            count(index, raw_count - 1)
        spin_indices = window["selected_spin_band_indices"]
        audit_shape = out["audit_source"]["audit_energy_shape"]
        if len(audit_shape) == 3:
            if not isinstance(spin_indices, list) or len(spin_indices) != len(selected):
                raise ValueError("spin indices must align with selected slots")
            for pair, index in zip(spin_indices, selected):
                if (not isinstance(pair, list) or len(pair) != 2
                        or count(pair[0], audit_shape[0] - 1) * audit_shape[1]
                        + count(pair[1], audit_shape[1] - 1) != index):
                    raise ValueError("spin indices disagree with actual selection")
        elif spin_indices is not None:
            raise ValueError("single-channel diagnostic spin indices must be null")
        sides = window["omitted_frontier_sides"]
        if (not isinstance(sides, list) or any(side not in ("occupied", "empty") for side in sides)
                or len(set(sides)) != len(sides)):
            raise ValueError("omitted frontier sides must be a unique side list")
        quotas = window["side_quotas"]
        if window["selection_branch"] == "clean_split":
            if (not isinstance(quotas, dict) or set(quotas) != {"occupied", "empty"}
                    or sum(count(quotas[side], max_bands) for side in quotas) != max_bands):
                raise ValueError("clean-split side quotas must be bounded integer capacities")
        elif quotas is not None:
            raise ValueError("non-clean-split side quotas must be null")
        if (window["selection_branch"] not in ("clean_split", "window_overlap", "nearest_fallback")
                or any(type(window[key]) is not bool for key in
                       ("slots_full", "capacity_insufficient", "window_capacity_insufficient"))):
            raise ValueError("invalid selection branch or capacity flags")
        for prefix in ("window", "selection"):
            dropped = window[f"{prefix}_dropped_indices"]
            if (not isinstance(dropped, list) or len(dropped) != window[f"{prefix}_dropped"]
                    or any(type(i) is not int or not 0 <= i < raw_count for i in dropped)
                    or len(set(dropped)) != len(dropped) or set(dropped) & set(selected)):
                raise ValueError("dropped indices disagree with actual selection")
        if (window["selected_indices"] != selected or window["selected"] != len(selected)
                or window["raw_band_count"] != raw_count
                or window["window_candidates"] != window["selected_in_window"] + window["window_dropped"]
                or window["selected"] != window["selected_in_window"] + window["selected_outside_window"]
                or window["selection_candidates"] != len(selected) + window["selection_dropped"]
                or window["slots_full"] != (len(selected) == max_bands)
                or window["capacity_insufficient"] != (window["selection_candidates"] > max_bands)
                or window["window_capacity_insufficient"] != (window["window_candidates"] > max_bands)
                or window["capacity_shortfall"] != max(0, window["selection_candidates"] - max_bands)
                or window["quota_underfill"] != (min(window["selection_candidates"], max_bands) - len(selected)
                                                if window["selection_branch"] == "clean_split" else 0)):
            raise ValueError("window/selection/capacity denominator mismatch")
        for key, bands_count, k_count in (("raw_full", raw_count, raw_shape[-1]),
                                           ("raw_selected", len(selected), raw_shape[-1]),
                                           ("resampled_selected", len(selected), n_k)):
            diagnostic = resampled[key]
            if (count(diagnostic["n_valid_bands"], bands_count) != bands_count
                    or count(diagnostic["n_valid_k"], k_count) != k_count):
                raise ValueError(f"{key} array denominator mismatch")
            if (diagnostic["scope"] != "selected_slot_and_energy_rank_line_diagnostic"
                    or diagnostic["causal_attribution"] != "not_established"
                    or not isinstance(diagnostic["frontier_state"], str)
                    or not diagnostic["frontier_state"]):
                raise ValueError(f"{key} missing diagnostic scope/frontier state")
            gap = diagnostic["gap_eV"]
            if gap is not None and type(gap) not in (int, float):
                raise ValueError(f"{key} gap must be numeric or unresolved null")
            for field in ("interval_straddling_bands", "within_segment_crossing_bands"):
                count(diagnostic[field], bands_count)
            # Only structure/count gates: no energies, gap or crossing recomputation.
            rows = diagnostic["segments"]
            actual_segment_ids = out["segment_ids" if key == "resampled_selected" else "source_segment_ids"]
            expected_ids = np.unique(actual_segment_ids).tolist()
            if not isinstance(rows, list) or len(rows) != len(expected_ids):
                raise ValueError(f"{key} missing or misaligned segment records")
            previous_stop = 0
            for row, sid in zip(rows, expected_ids):
                if not isinstance(row, dict):
                    raise ValueError(f"{key} segment must be a mapping")
                points = count(row["n_points"], k_count)
                start, stop = count(row["start"], k_count), count(row["stop"], k_count)
                if (count(row["segment_id"], out["source_segment_count"] - 1) != sid
                        or start != previous_stop or not points or stop - start != points):
                    raise ValueError(f"{key} segment bounds/count mismatch")
                previous_stop = stop
                positions = np.flatnonzero(actual_segment_ids == sid)
                if (start != int(positions[0]) or stop != int(positions[-1]) + 1
                        or points != len(positions)):
                    raise ValueError(f"{key} segment bounds/count disagree with actual segment IDs")
                count(row["slot_crossing_bands"], bands_count)
                count(row["source_order_inversions"], max(0, bands_count - 1) * points)
                occupations = [row[field] for field in ("n_strictly_below_ef", "n_at_or_below_ef")]
                if any(not isinstance(values, list) or len(values) != points for values in occupations):
                    raise ValueError(f"{key} occupation arrays must match segment length")
                for below, at_or_below in zip(*occupations):
                    if count(below, bands_count) > count(at_or_below, bands_count):
                        raise ValueError(f"{key} inconsistent occupation count bounds")
                if type(row["relabeling_invariant_strict_crossing"]) is not bool:
                    raise ValueError(f"{key} crossing diagnostic must be boolean")
                for field in ("slot_dispersion_rms_eV", "rank_dispersion_rms_eV",
                              "minimum_adjacent_separation_eV"):
                    value = row[field]
                    if value is None and field == "minimum_adjacent_separation_eV":
                        continue  # No comparable adjacent bands (e.g. one per spin).
                    if type(value) not in (int, float):
                        raise ValueError(f"{key} segment metric must be numeric")
                # Signed separations and false crossing flags are legitimate diagnostics.
            if previous_stop != k_count:
                raise ValueError(f"{key} segments do not cover the array k denominator")
        count(resampled["source_segment_count"], raw_shape[-1])
        count(resampled["target_segment_count"], n_k)
        lost = resampled["unrepresented_source_segment_ids"]
        if not isinstance(lost, list):
            raise ValueError("unrepresented source segments must be an integer list")
        for sid in lost:
            count(sid, out["source_segment_count"] - 1)
        if (resampled["source_segment_count"] != out["source_segment_count"]
                or resampled["target_segment_count"] != len(np.unique(out["segment_ids"]))
                or resampled["unrepresented_source_segment_ids"] != np.setdiff1d(
                    np.arange(out["source_segment_count"]), out["segment_ids"]).tolist()):
            raise ValueError("source/target segment count mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"P3 preparation audit invalid: {exc}") from exc


def _verify_preparation_audit(path, expected):
    """Check the fixed local target against bytes hashed before publication."""
    try:
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("published bytes changed")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"P3 preparation audit integrity failure: {path.name}: {exc}") from exc


def source_qa_ledger(h5_path, split_npz, out_dir, *, split):
    """Audit exactly one frozen identity set; write a NEW audit-only directory.

    Inputs are only opened read-only. No sidecar/targets/models, no training
    credentials, no download, no cache rewrite and no train/test aggregate.
    Source SHA attrs are declared provenance; snapshot SHA and content_sha256
    are recomputed. verified_valid denotes the source contract, not DFT truth.
    """
    if split not in ("train", "test"):
        raise ValueError("source QA requires explicit isolated --split train or test")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    paths = {"h5": Path(h5_path), "split_npz": Path(split_npz)}
    snapshot = {key: {"path": str(path.resolve()), "sha256": sha256_file(path),
                      "bytes": path.stat().st_size} for key, path in paths.items()}
    with np.load(split_npz, allow_pickle=True) as archive:
        ids, groups = archive[f"material_ids_{split}"], archive[f"groups_{split}"]
        if (ids.ndim != 1 or groups.shape != ids.shape or groups.dtype.kind not in "iu"
                or any(not isinstance(mid, str) or not mid for mid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("source QA frozen split: require unique string IDs and aligned integer groups")
    records = []
    with h5py.File(h5_path, "r") as handle:
        for index, mid in enumerate(ids):
            source = {}
            try:
                source = read_hdf5_source_record(handle[str(mid)])
            except (KeyError, OSError, UnicodeError) as exc:
                quality = {"rule_version": SOURCE_QA_RULE_VERSION, "status": "read_error",
                           "content_sha256": None, "reason": "hdf5_record_read_error",
                           "reasons": ["hdf5_record_read_error"], "detail": f"{type(exc).__name__}: {exc}"}
            else:
                quality = assess_source_quality(source)
            records.append({**quality, "material_id": str(mid), "split_index": index,
                            "group": int(groups[index]),
                            "declared_source_sha256": source.get("source_sha256"),
                            "source_url": source.get("source_url") or source.get("download_url")})
    counts = {"requested": len(ids), "valid": 0, "ambiguous": 0, "invalid": 0, "read_error": 0}
    for row in records:
        counts["valid" if row["status"] == "verified_valid" else row["status"]] += 1
    if counts["requested"] != sum(counts[key] for key in ("valid", "ambiguous", "invalid", "read_error")):
        raise RuntimeError("source QA denominator mismatch")
    report = {
        "schema": "source_qa_ledger", "schema_version": 1, "status": "complete",
        "rule_version": SOURCE_QA_RULE_VERSION, "split": split,
        "purpose": "train_source_audit" if split == "train" else "outer_test_read_only_audit_not_for_selection",
        "snapshot": snapshot,
        "code_hashes": {path: sha256_file(ROOT / path) for path in
                        ("src/data/aflow_adapter.py", "scripts/prepare_p3_multiband.py")},
        "counts": counts, "records": records,
        "limitations": ["exact-content quarantine is not a generic anomaly detector",
                        "verified_valid is declared source-contract validity, not independent eigenvalue certification",
                        "unknown schema/spin/layout are not formal eligible",
                        "audit-only; not a P3 preparation completion or scientific acceptance"],
    }
    for key, source_path in paths.items():
        if sha256_file(source_path) != snapshot[key]["sha256"]:
            raise RuntimeError(f"source {key} changed during QA; no complete ledger")
    path = out_dir / f"source_qa_{split}.json"
    try:
        _atomic_write(path, lambda handle: json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False))
        for key, source_path in paths.items():
            if sha256_file(source_path) != snapshot[key]["sha256"]:
                raise RuntimeError(f"source {key} changed during publication; no complete ledger")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print(f"[source QA {split}] {json.dumps(counts)} -> {path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P3 variable multi-band data")
    parser.add_argument("--mode", choices=("prepare", "source-qa"), default="prepare")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sidecar", help="required for prepare; not read by source-qa")
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
    if args.mode == "source-qa":
        if args.split == "both" or args.limit:
            parser.error("source-qa requires explicit isolated --split train or test and no --limit")
        source_qa_ledger(args.h5, args.split_npz, args.out_dir, split=args.split)
        return
    if not args.sidecar:
        parser.error("--sidecar is required for prepare")
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
        "src/data/crystal_graph.py", "src/utils/selection_manifest.py",
        "src/data/aflow_adapter.py", "src/evaluation/multiband_metrics.py")}
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
        "source_quality_policy": {"rule_version": SOURCE_QA_RULE_VERSION,
                                  "mode": "formal", "accepted_status": "verified_valid"},
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
            audit_records = []
            for i, mid in enumerate(ids):
                mid = str(mid)
                audit_row = {"material_id": mid, "split_index": i, "group": int(groups[i])}
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
                    audit_records.append({**audit_row,
                                          "status": "not_processed" if exc.reason == "smoke_limit" else "excluded",
                                          "reason": exc.reason, "detail": exc.detail,
                                          "band_window_audit": None, "resampling_audit": None})
                    continue
                _validate_preparation_audits(out, args.max_bands, args.n_k)
                audit_records.append({**audit_row, "status": "audited",
                                      "source": out["audit_source"],
                                      "band_window_audit": out["band_window_audit"],
                                      "resampling_audit": out["resampling_audit"]})
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
            audit_path = Path(args.out_dir) / f"p3_{split}_audit.json"
            audit = {"schema": "p3_prepare_audit", "schema_version": 1,
                     "status": "diagnostic_only", "split": split,
                     "accepted": False,
                     **{key: report[key] for key in ("sources", "code_hashes", "parameters", "smoke_only",
                                                     "source_quality_policy", "segmentation")},
                     "output_npz": {"path": Path(out_path).name, "sha256": sha,
                                    "bytes": Path(out_path).stat().st_size},
                     "limitations": ["source contract QA is not independent source/spin/eigenvalue verification",
                                     "slots_full is not truncation; capacity and side-quota omissions are distinct",
                                     "scalar k and duplicate-distance segments do not certify physical 3D k binding",
                                     "no physical derivatives, causal attribution or scientific acceptance"],
                     "counts": {"requested": n, **{status: sum(row["status"] == status for row in audit_records)
                                for status in ("audited", "excluded", "not_processed")}},
                     "records": audit_records}
            payload = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            audit_file = {"path": audit_path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload)}
            _atomic_write(audit_path, lambda handle: handle.write(payload), binary=True)
            _verify_preparation_audit(audit_path, audit_file)
            report["splits"][split] = {
                "audit": audit_file,
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
    # Split credentials expose no peer IDs, groups or sample statistics. Only
    # the final neutral commit authorizes them: orphan NPZ/credentials/reports
    # from failed or interrupted preparation are deliberately unusable.
    report_path = Path(args.out_dir) / "p3_prepare_report.json"
    commit_path = Path(args.out_dir) / "p3_prepare_commit.json"
    try:
        _atomic_write(report_path, lambda handle: json.dump(
            report, handle, ensure_ascii=False, indent=2, allow_nan=False))
        if empty_splits:
            raise RuntimeError(f"zero valid samples in split(s): {', '.join(empty_splits)}")
        completion_hashes = {}
        for split, stats in report["splits"].items():
            credential = {
                "schema": "p3_split_completion", "schema_version": 1,
                "status": "complete", "split": split,
                "source_quality_policy": report["source_quality_policy"],
                "output_npz": stats["output_npz"],
                "preparation_audit": stats["audit"],
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
        for split, stats in report["splits"].items():
            _verify_preparation_audit(Path(args.out_dir) / f"p3_{split}_audit.json", stats["audit"])
        _atomic_write(commit_path, lambda handle: json.dump(commit, handle, indent=2, allow_nan=False))
        for split, stats in report["splits"].items():
            _verify_preparation_audit(Path(args.out_dir) / f"p3_{split}_audit.json", stats["audit"])
    except BaseException:
        # Also revoke a rename followed by a reported fsync/interruption error.
        commit_path.unlink(missing_ok=True)
        for split in report["splits"]:
            (Path(args.out_dir) / f"p3_{split}_complete.json").unlink(missing_ok=True)
        if report["status"] == "complete":
            report_path.unlink(missing_ok=True)
        # Preserve a published zero-valid failed report as diagnostic evidence.
        raise
    print(f"[P3] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
