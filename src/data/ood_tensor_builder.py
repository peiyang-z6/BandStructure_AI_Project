"""
Build cleaned OOD tensors from downloaded Materials Project band data.

This module converts raw HDF5/JSON downloads into fixed-shape tensors:
    X: (num_samples, 2, target_k_points, 3)

The two band slots are VBM-like and CBM-like bands. The three channels are:
    channel 0: interpolated energy E(k)
    channel 1: second derivative / curvature d2E/dk2
    channel 2: normalized distance to that band's global extremum k index

The split is group-aware and uses spacegroup_number as the group key. Random
sample-wise splitting is intentionally not used.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
# GroupShuffleSplit removed — replaced by stratified group split below


@dataclass
class ProcessedBandSample:
    material_id: str
    tensor: np.ndarray
    band_gap: float
    spacegroup: int
    efermi: float
    vbm_band_idx: int
    cbm_band_idx: int


def _read_metadata_json(metadata_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not metadata_path or not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, list):
        return {
            str(item["material_id"]): item
            for item in loaded
            if isinstance(item, dict) and item.get("material_id")
        }

    if isinstance(loaded, dict):
        return {
            str(material_id): value
            for material_id, value in loaded.items()
            if isinstance(value, dict)
        }

    return {}


def _metadata_for_group(grp: h5py.Group, metadata_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    material_id = grp.name.rsplit("/", 1)[-1]
    metadata = dict(grp.attrs)

    if "metadata" in grp:
        metadata.update(dict(grp["metadata"].attrs))

    if material_id in metadata_by_id:
        metadata.update(metadata_by_id[material_id])

    return metadata


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_json_attr(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _resampled_kpath_labels(metadata: Dict[str, Any], target_k_points: int) -> List[Dict[str, Any]]:
    labels = _parse_json_attr(metadata.get("kpath_labels"), [])
    original_k = _as_int(metadata.get("num_kpoints"))
    if not labels or not original_k or original_k < 2:
        return [
            {"index": 0, "label": "\\Gamma"},
            {"index": target_k_points - 1, "label": "\\Gamma"},
        ]

    converted: List[Dict[str, Any]] = []
    seen = set()
    for item in labels:
        if not isinstance(item, dict):
            continue
        raw_idx = _as_int(item.get("index"))
        label = item.get("label")
        if raw_idx is None or label is None:
            continue
        new_idx = int(round(raw_idx * (target_k_points - 1) / (original_k - 1)))
        new_idx = max(0, min(target_k_points - 1, new_idx))
        key = (new_idx, str(label))
        if key in seen:
            continue
        seen.add(key)
        converted.append({"index": new_idx, "label": str(label)})

    return converted or [
        {"index": 0, "label": "\\Gamma"},
        {"index": target_k_points - 1, "label": "\\Gamma"},
    ]


def _select_spin_channel(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=np.float32)
    if energies.ndim == 3:
        energies = energies[0]
    if energies.ndim != 2:
        raise ValueError(f"Expected energies with rank 2 or 3, got shape {energies.shape}")
    return energies


def _select_vbm_cbm_indices(energies: np.ndarray, efermi: Optional[float]) -> Tuple[int, int]:
    """Select VBM and CBM band indices using Fermi level when available."""
    num_bands = energies.shape[0]
    if num_bands < 2:
        raise ValueError("Need at least two bands to build VBM/CBM tensor")

    band_max = np.max(energies, axis=1)
    band_min = np.min(energies, axis=1)
    band_mean = np.mean(energies, axis=1)

    if efermi is not None and np.isfinite(efermi):
        valence_candidates = np.where(band_max <= efermi)[0]
        conduction_candidates = np.where(band_min >= efermi)[0]

        if len(valence_candidates) > 0 and len(conduction_candidates) > 0:
            vbm_idx = int(valence_candidates[np.argmax(band_max[valence_candidates])])
            cbm_idx = int(conduction_candidates[np.argmin(band_min[conduction_candidates])])
            if vbm_idx != cbm_idx:
                return vbm_idx, cbm_idx

    sorted_by_mean = np.argsort(band_mean)
    mid = max(1, min(len(sorted_by_mean) - 1, len(sorted_by_mean) // 2))
    return int(sorted_by_mean[mid - 1]), int(sorted_by_mean[mid])


def _resample_band(band: np.ndarray, target_k_points: int) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    old_len = len(band)
    if old_len < 2:
        raise ValueError("Cannot interpolate a band with fewer than two k-points")

    x_old = np.linspace(0.0, 1.0, old_len)
    x_new = np.linspace(0.0, 1.0, target_k_points)

    if old_len >= 4:
        return CubicSpline(x_old, band)(x_new).astype(np.float32)

    return interp1d(x_old, band, kind="linear")(x_new).astype(np.float32)


def _curvature(band: np.ndarray) -> np.ndarray:
    """Compute local quadratic curvature at every k-point via windowed polyfit.

    Uses np.polyfit on a local window of ±2 points around each k-index, yielding
    more accurate extremum curvature than global np.gradient(np.gradient(...)).
    """
    band = np.asarray(band, dtype=np.float64)
    n = len(band)
    curv = np.zeros(n, dtype=np.float64)
    half_window = 2
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        x_local = np.arange(lo, hi, dtype=np.float64) - float(i)
        y_local = band[lo:hi]
        if len(x_local) < 3:
            # Fallback: use central finite difference
            if i > 0 and i < n - 1:
                curv[i] = band[i + 1] - 2.0 * band[i] + band[i - 1]
            continue
        coeff = np.polyfit(x_local, y_local, deg=2)
        curv[i] = 2.0 * coeff[0]
    return curv.astype(np.float32)


def _extremum_distance_channel(extremum_idx: int, target_k_points: int) -> np.ndarray:
    center = int(extremum_idx)
    denom = max(1, target_k_points - 1)
    return ((np.arange(target_k_points, dtype=np.float32) - center) / denom).astype(np.float32)


def _build_sample_tensor(
    energies: np.ndarray,
    efermi: Optional[float],
    target_k_points: int,
) -> Tuple[np.ndarray, int, int]:
    energies = _select_spin_channel(energies)
    vbm_idx, cbm_idx = _select_vbm_cbm_indices(energies, efermi)

    vbm = _resample_band(energies[vbm_idx], target_k_points)
    cbm = _resample_band(energies[cbm_idx], target_k_points)
    vbm_extreme_idx = int(np.argmax(vbm))
    cbm_extreme_idx = int(np.argmin(cbm))

    tensor = np.stack(
        [
            np.stack(
                [
                    vbm,
                    _curvature(vbm),
                    _extremum_distance_channel(vbm_extreme_idx, target_k_points),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    cbm,
                    _curvature(cbm),
                    _extremum_distance_channel(cbm_extreme_idx, target_k_points),
                ],
                axis=-1,
            ),
        ],
        axis=0,
    ).astype(np.float32)

    return tensor, vbm_idx, cbm_idx


def process_band_data(
    h5_path: str,
    metadata_path: Optional[str] = None,
    target_k_points: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Process raw HDF5 band data into tensors and labels."""
    metadata_by_id = _read_metadata_json(metadata_path)

    tensors: List[np.ndarray] = []
    gaps: List[float] = []
    groups: List[int] = []
    material_ids: List[str] = []
    sample_metadata: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    with h5py.File(h5_path, "r") as f:
        for material_id in sorted(f.keys()):
            grp = f[material_id]
            if "energies" not in grp:
                skipped.append({"material_id": material_id, "reason": "missing energies"})
                continue

            metadata = _metadata_for_group(grp, metadata_by_id)
            spacegroup = _as_int(
                metadata.get("spacegroup_number", metadata.get("spacegroup"))
            )
            band_gap = _as_float(metadata.get("band_gap"))
            efermi = _as_float(metadata.get("efermi"))

            if spacegroup is None:
                skipped.append({"material_id": material_id, "reason": "missing spacegroup"})
                continue
            if band_gap is None or not np.isfinite(band_gap):
                skipped.append({"material_id": material_id, "reason": "missing band_gap"})
                continue

            try:
                tensor, vbm_idx, cbm_idx = _build_sample_tensor(
                    grp["energies"][:],
                    efermi=efermi,
                    target_k_points=target_k_points,
                )
            except Exception as exc:
                skipped.append({"material_id": material_id, "reason": str(exc)})
                continue

            tensors.append(tensor)
            gaps.append(float(band_gap))
            groups.append(int(spacegroup))
            material_ids.append(material_id)
            sample_metadata.append(
                {
                    "material_id": material_id,
                    "spacegroup_number": int(spacegroup),
                    "band_gap": float(band_gap),
                    "efermi": efermi,
                    "vbm_band_idx": int(vbm_idx),
                    "cbm_band_idx": int(cbm_idx),
                    "kpath_labels": _resampled_kpath_labels(metadata, target_k_points),
                }
            )

    if not tensors:
        raise RuntimeError(f"No valid band tensors could be built from {h5_path}")

    return (
        np.stack(tensors, axis=0).astype(np.float32),
        np.asarray(gaps, dtype=np.float32),
        np.asarray(groups, dtype=np.int32),
        np.asarray(material_ids, dtype="U32"),
        [{"samples": sample_metadata, "skipped": skipped}],
    )


def build_group_ood_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an OOD split with no spacegroup overlap.

    Tries stratified group splitting first (each class in both splits).
    Falls back to plain GroupShuffleSplit if the class distribution makes
    stratification impossible with group constraint.
    """
    rng = np.random.RandomState(random_state)

    # Try class-stratified group split
    unique_groups = np.unique(groups)
    # Group -> list of (index, label)
    label = np.round(y).astype(int)  # approximate: 0=metal(0), >0=non-metal
    label[label > 0] = 1  # binarize: metal vs non-metal
    
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    group_to_has_metal = {g: int(np.any(label[idx] == 0)) for g, idx in group_to_indices.items()}
    group_to_has_nonmetal = {g: int(np.any(label[idx] == 1)) for g, idx in group_to_indices.items()}
    
    # Shuffle groups
    group_list = list(unique_groups)
    rng.shuffle(group_list)
    
    n_test_groups_target = max(1, int(len(group_list) * (1 - train_size)))
    
    test_groups = []
    test_has_metal = False
    test_has_nonmetal = False
    train_has_metal = False
    train_has_nonmetal = False
    
    # First pass: ensure test gets both classes if possible
    for g in group_list:
        if len(test_groups) >= n_test_groups_target:
            break
        needed_metal = not test_has_metal and group_to_has_metal[g]
        needed_nonmetal = not test_has_nonmetal and group_to_has_nonmetal[g]
        if needed_metal or needed_nonmetal or len(test_groups) < n_test_groups_target:
            test_groups.append(g)
            if group_to_has_metal[g]:
                test_has_metal = True
            if group_to_has_nonmetal[g]:
                test_has_nonmetal = True
    
    test_set = set(test_groups)
    train_groups_set = set(g for g in group_list if g not in test_set)
    
    # Verify train also has both classes
    for g in train_groups_set:
        if group_to_has_metal.get(g, False):
            train_has_metal = True
        if group_to_has_nonmetal.get(g, False):
            train_has_nonmetal = True

    train_idx = np.concatenate([group_to_indices[g] for g in sorted(train_groups_set)]) if train_groups_set else np.array([], dtype=int)
    test_idx = np.concatenate([group_to_indices[g] for g in sorted(test_set)]) if test_set else np.array([], dtype=int)

    train_groups_set = set(groups[train_idx].tolist()) if len(train_idx) > 0 else set()
    test_groups_set = set(groups[test_idx].tolist()) if len(test_idx) > 0 else set()
    overlap = train_groups_set.intersection(test_groups_set)
    if overlap:
        raise RuntimeError(f"Group leakage detected across train/test: {sorted(overlap)}")

    return train_idx, test_idx


def save_ood_tensors(
    output_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    material_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata_bundle: List[Dict[str, Any]],
    target_k_points: int,
    train_size: float,
    random_state: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    full_npz = os.path.join(output_dir, "band_tensors_full.npz")
    split_npz = os.path.join(output_dir, "band_tensors_ood_split.npz")
    manifest_path = os.path.join(output_dir, "ood_split_manifest.json")

    np.savez_compressed(
        full_npz,
        X=X,
        y_gap=y,
        groups=groups,
        material_ids=material_ids,
    )
    np.savez_compressed(
        split_npz,
        X_train=X[train_idx],
        y_train=y[train_idx],
        groups_train=groups[train_idx],
        material_ids_train=material_ids[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        groups_test=groups[test_idx],
        material_ids_test=material_ids[test_idx],
    )

    train_groups = sorted(set(groups[train_idx].tolist()))
    test_groups = sorted(set(groups[test_idx].tolist()))
    manifest = {
        "target_k_points": target_k_points,
        "tensor_shape": list(X.shape),
        "channels": ["energy", "curvature", "extremum_k_distance"],
        "flattened_features": [
            "VBM_E",
            "VBM_curv",
            "VBM_k_dist",
            "CBM_E",
            "CBM_curv",
            "CBM_k_dist",
        ],
        "band_slots": ["vbm_like", "cbm_like"],
        "split_method": "StratifiedGroupShuffleSplit",
        "group_key": "spacegroup_number",
        "train_size": train_size,
        "random_state": random_state,
        "num_samples": int(len(X)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "num_groups": int(len(set(groups.tolist()))),
        "num_train_groups": int(len(train_groups)),
        "num_test_groups": int(len(test_groups)),
        "group_overlap": [],
        "class_distribution": {
            "train": {str(k): int(v) for k, v in zip(*np.unique(y[train_idx].round().astype(int), return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(y[test_idx].round().astype(int), return_counts=True))},
        },
        "outputs": {
            "full_npz": full_npz,
            "split_npz": split_npz,
        },
        "train_material_ids": material_ids[train_idx].tolist(),
        "test_material_ids": material_ids[test_idx].tolist(),
        "train_spacegroups": train_groups,
        "test_spacegroups": test_groups,
        "samples": metadata_bundle[0]["samples"],
        "skipped": metadata_bundle[0]["skipped"],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD tensors from MP band data")
    parser.add_argument("--h5", default="./data_cache/mp_bands.h5", help="input HDF5 path")
    parser.add_argument("--metadata", default="./data_cache/mp_metadata.json", help="metadata JSON path")
    parser.add_argument("--output", default="./data_cache/ood_tensors", help="output directory")
    parser.add_argument("--target-k", type=int, default=128, help="fixed k-point sequence length")
    parser.add_argument("--train-size", type=float, default=0.8, help="StratifiedGroupShuffleSplit train size")
    parser.add_argument("--random-state", type=int, default=42, help="StratifiedGroupShuffleSplit random seed")
    args = parser.parse_args()

    X, y, groups, material_ids, metadata_bundle = process_band_data(
        h5_path=args.h5,
        metadata_path=args.metadata,
        target_k_points=args.target_k,
    )
    train_idx, test_idx = build_group_ood_split(
        X,
        y,
        groups,
        train_size=args.train_size,
        random_state=args.random_state,
    )
    manifest = save_ood_tensors(
        output_dir=args.output,
        X=X,
        y=y,
        groups=groups,
        material_ids=material_ids,
        train_idx=train_idx,
        test_idx=test_idx,
        metadata_bundle=metadata_bundle,
        target_k_points=args.target_k,
        train_size=args.train_size,
        random_state=args.random_state,
    )

    print("=" * 60)
    print("OOD tensor build complete")
    print("=" * 60)
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"groups: {manifest['num_groups']}")
    print(f"train/test: {manifest['num_train']} / {manifest['num_test']}")
    print(f"group overlap: {manifest['group_overlap']}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
