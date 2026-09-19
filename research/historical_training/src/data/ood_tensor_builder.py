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
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from src.utils.selection_manifest import (P2_RAW_CODE, P2_RAW_SEMANTICS,
    p2_file_ref, p2_code_refs, p2_feature_schema, p2_raw_observation_sha)
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
    gap_type: int


def _read_metadata_json(metadata_path: Optional[str], *, canonical_contract: bool = False) -> Dict[str, Dict[str, Any]]:
    if not metadata_path or not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded = json.load(f, object_pairs_hook=_canonical_json_pairs if canonical_contract else None)

    if isinstance(loaded, list):
        if canonical_contract:
            result = {}
            for index, item in enumerate(loaded):
                if isinstance(item, dict) and item.get('material_id'):
                    mid = str(item['material_id'])
                    result[mid] = _merge_canonical_metadata([
                        (f'{mid}:earlier JSON rows', result.get(mid, {})),
                        (f'{mid}:JSON row {index}', item)])
            return result
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


def _metadata_for_group(grp: h5py.Group, metadata_by_id: Dict[str, Dict[str, Any]],
                        *, canonical_contract: bool = False) -> Dict[str, Any]:
    material_id = grp.name.rsplit("/", 1)[-1]
    if canonical_contract:
        return _merge_canonical_metadata([
            (f'{material_id}:attrs', dict(grp.attrs)),
            (f'{material_id}:nested', dict(grp['metadata'].attrs) if 'metadata' in grp else {}),
            (f'{material_id}:external', metadata_by_id.get(material_id, {}))])
    metadata = dict(grp.attrs)

    if "metadata" in grp:
        metadata.update(dict(grp["metadata"].attrs))

    if material_id in metadata_by_id:
        metadata.update(metadata_by_id[material_id])

    return metadata


def _canonical_metadata_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        value = _parse_json_attr(value, value.strip(), canonical_contract=True)
    if isinstance(value, dict):
        return {key: _canonical_metadata_value(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_metadata_value(v) for v in value]
    return value


def _merge_canonical_metadata(layers):
    """Reject contradictions before values can disappear in a dict.update."""
    merged, origins = {}, {}
    for origin, values in layers:
        for key, value in values.items():
            value = _canonical_metadata_value(value)
            if value is None or value == '' or value == [] or value == {}:
                continue  # absent sidecar values cannot downgrade known attrs
            if key in merged and merged[key] != value:
                raise ValueError(f'Metadata conflict for {key}: {origins[key]} vs {origin}')
            merged[key], origins[key] = value, origin
    for aliases in (('efermi', 'Efermi', 'E_F', 'fermi_energy'),
                    ('source', 'provider'), ('spacegroup_number', 'spacegroup'),
                    ('dft_type', 'protocol')):
        present = [key for key in aliases if key in merged]
        if present and any(merged[key] != merged[present[0]] for key in present[1:]):
            raise ValueError(f'Metadata alias conflict: {present}')
    return merged


def _canonical_json_pairs(pairs):
    # Preserve repeated JSON keys until conflicting declarations are rejected.
    return _merge_canonical_metadata([(f'JSON occurrence {i}', {key: value})
                                      for i, (key, value) in enumerate(pairs)])


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


def _as_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return default


def _parse_json_attr(value: Any, default: Any, *, canonical_contract: bool = False) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value, object_pairs_hook=_canonical_json_pairs if canonical_contract else None)
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
        raw_distance = _as_float(item.get("k_distance"))
        path_min = _as_float(metadata.get("k_distance_min"), 0.0) or 0.0
        path_max = _as_float(metadata.get("k_distance_max"))
        if raw_distance is not None and path_max is not None and path_max > path_min:
            fraction = (raw_distance - path_min) / (path_max - path_min)
            new_idx = int(round(fraction * (target_k_points - 1)))
        else:
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


def _canonical_path_layout(labels, axis, target_k_points):
    """Map source k[index], never index fraction; retain branch-side endpoints.

    Continuous vertices snap to the nearest target point (endpoint-clamped
    interpolation on that one half-grid sliver). Disconnected duplicate vertices
    split at the first target coordinate >= the break; no values are averaged.
    """
    axis = np.asarray(axis, dtype=np.float64)
    indices = [item['index'] for item in labels]
    if target_k_points < 2 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError('Unsupported source k-path label order/target grid')
    target = np.linspace(axis[0], axis[-1], target_k_points)
    converted = []
    for item in labels:
        i = item['index']
        if 'k_distance' in item:
            coordinate = _as_float(item['k_distance'])
            if coordinate is None or not np.isclose(coordinate, axis[i], rtol=1e-7, atol=1e-8):
                raise ValueError('Source k-path label coordinate conflicts with k[index]')
        j = int(np.rint((axis[i]-axis[0])/(axis[-1]-axis[0])*(target_k_points-1)))
        converted.append({'index': j, 'label': str(item['label']),
                          'source_index': i, 'k_distance': float(axis[i])})
    breaks = set()
    for n, (a, b) in enumerate(zip(indices, indices[1:])):
        if axis[a] == axis[b]:
            if b != a+1 or n == 0 or n == len(indices)-2:
                raise ValueError('Unsupported zero-length source segment')
            j = int(np.searchsorted(target, axis[a]))
            converted[n]['index'] = converted[n+1]['index'] = j
            breaks.add(a)
    if set(np.flatnonzero(np.diff(axis) == 0).tolist()) != breaks:
        raise ValueError('Repeated source coordinate outside a labelled branch break')
    bounds = []
    raw_ids = np.full(len(axis), -1, dtype=np.int32)
    target_ids = np.full(target_k_points, -1, dtype=np.int32)
    for n, (a, b) in enumerate(zip(indices, indices[1:])):
        if a in breaks:
            continue
        start, end = converted[n]['index'], converted[n+1]['index']
        if end <= start or np.any(np.diff(axis[a:b+1]) <= 0):
            raise ValueError('Source segment is unresolved on the target grid')
        stop = end+1 if b == len(axis)-1 else end
        segment = len(bounds)
        bounds.append((a, b, start, stop))
        raw_ids[a:b+1] = segment
        target_ids[start:stop] = segment
    if np.any(raw_ids < 0) or np.any(target_ids < 0):
        raise ValueError('Unsupported source/target segment coverage')
    return converted, raw_ids, target_ids, bounds


def _segment_ids_from_resampled_labels(
    labels: List[Dict[str, Any]],
    target_k_points: int,
) -> np.ndarray:
    """Assign each resampled k point to one high-symmetry path segment."""
    positions = sorted(
        {
            max(0, min(target_k_points - 1, int(item["index"])))
            for item in labels
            if isinstance(item, dict) and item.get("index") is not None
        }
    )
    if not positions or positions[0] != 0:
        positions.insert(0, 0)
    if positions[-1] != target_k_points - 1:
        positions.append(target_k_points - 1)
    segment_ids = np.zeros(target_k_points, dtype=np.int32)
    for segment_id, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
        segment_ids[start : end + 1] = segment_id
    return segment_ids


def _select_spin_channel(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=np.float32)
    if energies.ndim == 3:
        energies = energies.reshape(-1, energies.shape[-1])
    if energies.ndim != 2:
        raise ValueError(f"Expected energies with rank 2 or 3, got shape {energies.shape}")
    return energies


def _fermi_crossing_band_index(
    energies: np.ndarray,
    efermi: Optional[float],
    segment_ids: Optional[np.ndarray] = None,
) -> Optional[int]:
    """Infer a Fermi crossing within one continuous k-path segment only."""
    if efermi is None or not np.isfinite(efermi):
        return None
    segments = (
        np.zeros(energies.shape[-1], dtype=np.int32)
        if segment_ids is None
        else np.asarray(segment_ids, dtype=np.int32)
    )
    if len(segments) != energies.shape[-1]:
        raise ValueError("segment_ids length must match raw k-point count")
    candidates: List[Tuple[float, int]] = []
    for band_idx, band in enumerate(energies):
        for segment_id in np.unique(segments):
            values = band[segments == segment_id]
            if len(values) and float(np.min(values)) <= efermi <= float(np.max(values)):
                candidates.append((float(np.min(np.abs(values - efermi))), int(band_idx)))
                break
    if not candidates:
        return None
    return min(candidates)[1]


def _select_vbm_cbm_indices(
    energies: np.ndarray,
    efermi: Optional[float],
) -> Tuple[int, int]:
    """Select non-crossing VBM/CBM bands using E(k) and E_F only."""
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

        # Label-free fallback for crossing/semimetal-like paths: choose distinct
        # bands carrying the occupied and empty states closest to E_F.
        below = np.where(energies <= efermi, energies, -np.inf)
        above = np.where(energies >= efermi, energies, np.inf)
        occupied_edge = np.max(below, axis=1)
        empty_edge = np.min(above, axis=1)
        for vbm_idx in np.argsort(np.abs(occupied_edge - efermi)):
            if not np.isfinite(occupied_edge[vbm_idx]):
                continue
            for cbm_idx in np.argsort(np.abs(empty_edge - efermi)):
                if np.isfinite(empty_edge[cbm_idx]) and vbm_idx != cbm_idx:
                    return int(vbm_idx), int(cbm_idx)

    sorted_by_mean = np.argsort(band_mean)
    mid = max(1, min(len(sorted_by_mean) - 1, len(sorted_by_mean) // 2))
    return int(sorted_by_mean[mid - 1]), int(sorted_by_mean[mid])


def _source_k_axis(k_distances: Optional[np.ndarray], old_len: int) -> np.ndarray:
    if k_distances is None:
        return np.linspace(0.0, 1.0, old_len, dtype=np.float64)
    axis = np.asarray(k_distances, dtype=np.float64).reshape(-1)
    if len(axis) != old_len or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) < -1.0e-8):
        return np.linspace(0.0, 1.0, old_len, dtype=np.float64)
    # A positive source span is valid regardless of its unit/scale.
    if float(axis[-1] - axis[0]) <= 0.0:
        return np.linspace(0.0, 1.0, old_len, dtype=np.float64)
    return axis


def _resample_band(
    band: np.ndarray,
    target_k_points: int,
    k_distances: Optional[np.ndarray] = None,
    shape_preserving: bool = False,
    *, source_segment_bounds=None,
) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    old_len = len(band)
    if old_len < 2:
        raise ValueError("Cannot interpolate a band with fewer than two k-points")

    x_old = _source_k_axis(k_distances, old_len)
    x_new = np.linspace(float(x_old[0]), float(x_old[-1]), target_k_points)

    if source_segment_bounds is not None:
        result = np.empty(target_k_points, dtype=np.float32)
        for a, b, start, stop in source_segment_bounds:
            # PCHIP slope estimation and values stay within this source segment.
            x, y = x_old[a:b+1], band[a:b+1]
            query = np.clip(x_new[start:stop], x[0], x[-1])
            result[start:stop] = PchipInterpolator(x, y, extrapolate=False)(query)
        return result

    # Repeated path coordinates can occur at disconnected branch boundaries.
    # Average them to give interpolation a strictly increasing coordinate.
    unique_x, inverse = np.unique(x_old, return_inverse=True)
    if len(unique_x) != len(x_old):
        sums = np.bincount(inverse, weights=band.astype(np.float64))
        counts = np.bincount(inverse)
        band = (sums / np.maximum(counts, 1)).astype(np.float32)
        x_old = unique_x

    if old_len >= 4:
        interpolator = PchipInterpolator(x_old, band) if shape_preserving else CubicSpline(x_old, band)
        return interpolator(x_new).astype(np.float32)

    return interp1d(x_old, band, kind="linear")(x_new).astype(np.float32)


def _curvature(
    band: np.ndarray,
    k_axis: Optional[np.ndarray] = None,
    segment_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute local quadratic curvature without crossing k-path segments."""
    band = np.asarray(band, dtype=np.float64)
    n = len(band)
    curv = np.zeros(n, dtype=np.float64)
    half_window = 2
    x_axis = (
        np.asarray(k_axis, dtype=np.float64)
        if k_axis is not None
        else np.linspace(0.0, 1.0, n, dtype=np.float64)
    )
    segments = None if segment_ids is None else np.asarray(segment_ids, dtype=np.int32)
    if segments is not None and len(segments) != n:
        raise ValueError("segment_ids length must match band length")
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        indices = np.arange(lo, hi)
        if segments is not None:
            indices = indices[segments[indices] == segments[i]]
        x_local = x_axis[indices] - x_axis[i]
        y_local = band[indices]
        if len(x_local) < 3:
            same_segment_neighbors = (
                i > 0
                and i < n - 1
                and (
                    segments is None
                    or (segments[i - 1] == segments[i] == segments[i + 1])
                )
            )
            if same_segment_neighbors:
                dx = max(float(x_axis[i + 1] - x_axis[i]), 1.0e-12)
                curv[i] = (band[i + 1] - 2.0 * band[i] + band[i - 1]) / (dx * dx)
            continue
        coeff = np.polyfit(x_local, y_local, deg=2)
        curv[i] = 2.0 * coeff[0]
    return curv.astype(np.float32)


def _extremum_distance_channel(extremum_idx: int, target_k_points: int) -> np.ndarray:
    center = int(extremum_idx)
    denom = max(1, target_k_points - 1)
    return ((np.arange(target_k_points, dtype=np.float32) - center) / denom).astype(np.float32)


def _edge_envelopes(
    energies: np.ndarray,
    efermi: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build label-free occupied/empty edge envelopes at every k point."""
    below = np.where(energies <= efermi, energies, -np.inf)
    above = np.where(energies >= efermi, energies, np.inf)
    vbm = np.max(below, axis=0)
    cbm = np.min(above, axis=0)
    if not np.all(np.isfinite(vbm)) or not np.all(np.isfinite(cbm)):
        raise ValueError("cannot construct occupied/empty edge envelopes at all k points")
    vbm_k = int(np.argmax(vbm))
    cbm_k = int(np.argmin(cbm))
    vbm_idx = int(np.argmax(below[:, vbm_k]))
    cbm_idx = int(np.argmin(above[:, cbm_k]))
    return vbm.astype(np.float32), cbm.astype(np.float32), vbm_idx, cbm_idx


def _build_sample_tensor(
    energies: np.ndarray,
    efermi: Optional[float],
    target_k_points: int,
    k_distances: Optional[np.ndarray] = None,
    segment_ids: Optional[np.ndarray] = None,
    raw_segment_ids: Optional[np.ndarray] = None,
    *, source_segment_bounds=None,
) -> Tuple[np.ndarray, int, int, bool]:
    energies = _select_spin_channel(energies)
    crossing_idx = _fermi_crossing_band_index(
        energies, efermi, segment_ids=raw_segment_ids
    )
    metal_feature_inferred = crossing_idx is not None
    # Crossing is audit information only. The feature tensor uses the same
    # occupied/empty edge-envelope rule for every provider label.
    if efermi is None or not np.isfinite(efermi):
        raise ValueError("finite Fermi reference required for edge envelopes")
    vbm_raw, cbm_raw, vbm_idx, cbm_idx = _edge_envelopes(energies, float(efermi))
    vbm = _resample_band(vbm_raw, target_k_points, k_distances, shape_preserving=True,
                         source_segment_bounds=source_segment_bounds)
    cbm = _resample_band(cbm_raw, target_k_points, k_distances, shape_preserving=True,
                         source_segment_bounds=source_segment_bounds)
    vbm = np.minimum(vbm, float(efermi)).astype(np.float32)
    cbm = np.maximum(cbm, float(efermi)).astype(np.float32)
    source_axis = _source_k_axis(k_distances, energies.shape[-1])
    k_axis = np.linspace(
        float(source_axis[0]),
        float(source_axis[-1]),
        target_k_points,
        dtype=np.float32,
    )
    vbm_extreme_idx = int(np.argmax(vbm))
    cbm_extreme_idx = int(np.argmin(cbm))

    tensor = np.stack(
        [
            np.stack(
                [
                    vbm,
                    _curvature(vbm, k_axis, segment_ids=segment_ids),
                    _extremum_distance_channel(vbm_extreme_idx, target_k_points),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    cbm,
                    _curvature(cbm, k_axis, segment_ids=segment_ids),
                    _extremum_distance_channel(cbm_extreme_idx, target_k_points),
                ],
                axis=-1,
            ),
        ],
        axis=0,
    ).astype(np.float32)

    return tensor, vbm_idx, cbm_idx, metal_feature_inferred


def derive_line_mode_topology(
    vbm: np.ndarray, cbm: np.ndarray, crossing: bool
) -> int:
    """Derive line-mode topology (0/1/2) exclusively from the band path."""
    if crossing:
        return 0
    return 1 if int(np.argmax(vbm)) == int(np.argmin(cbm)) else 2


def derive_provider_global_type(
    is_metal: Optional[bool], is_direct: Optional[bool]
) -> int:
    """Provider global electronic type (0/1/2); -1 when undetermined."""
    if is_metal is True:
        return 0
    if is_metal is False and is_direct is True:
        return 1
    if is_metal is False and is_direct is False:
        return 2
    return -1


def derive_line_global_disagreement(line_topology: int, provider_type: int) -> int:
    """Binary: 1 iff line-mode topology and provider global type conflict."""
    if provider_type < 0:
        return -1
    return int(line_topology != provider_type)


def process_band_data(
    h5_path: str,
    metadata_path: Optional[str] = None,
    target_k_points: int = 128,
    *, canonical_contract: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Process raw HDF5 band data into tensors and labels."""
    metadata_by_id = _read_metadata_json(metadata_path, canonical_contract=canonical_contract)

    tensors: List[np.ndarray] = []
    gaps: List[float] = []
    groups: List[int] = []
    material_ids: List[str] = []
    gap_types: List[int] = []
    sample_metadata: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    with h5py.File(h5_path, "r") as f:
        for material_id in sorted(f.keys()):
            grp = f[material_id]
            if "energies" not in grp:
                skipped.append({"material_id": material_id, "reason": "missing energies"})
                continue

            metadata = _metadata_for_group(grp, metadata_by_id, canonical_contract=canonical_contract)
            if canonical_contract:
                reference = metadata.get('energy_reference')
                ef = _as_float(metadata.get('efermi'))
                if (reference not in ('absolute', 'fermi_shifted_zero')
                        or ef is None or not np.isfinite(ef)
                        or (reference == 'fermi_shifted_zero' and ef != 0.)
                        or metadata.get('energy_quantity') != 'band_energy_eV'
                        or metadata.get('source') not in ('materials_project', 'aflow')):
                    raise ValueError(f'Ambiguous energy/Fermi/source semantics: {material_id}')
                if 'k_distances' not in grp:
                    raise ValueError(f'Missing source k coordinate: {material_id}')
                n_k = grp['energies'].shape[-1]
                axis = np.asarray(grp['k_distances'][:])
                labels = _parse_json_attr(metadata.get('kpath_labels'), [])
                sg = metadata.get('spacegroup_number', metadata.get('spacegroup'))
                if (axis.shape != (n_k,) or not np.isfinite(axis).all()
                        or np.any(np.diff(axis) < 0) or axis[-1] <= axis[0]
                        or _as_int(metadata.get('num_kpoints')) != n_k
                        or not isinstance(labels, list) or len(labels) < 2
                        or any(not isinstance(item, dict) or type(item.get('index')) is not int
                               or not 0 <= item['index'] < n_k or not item.get('label') for item in labels)
                        or labels[0]['index'] != 0 or labels[-1]['index'] != n_k-1
                        or not isinstance(sg, (int, np.integer)) or isinstance(sg, (bool, np.bool_))
                        or not 1 <= int(sg) <= 230):
                    raise ValueError(f'Ambiguous source k-path or spacegroup: {material_id}')
            spacegroup = _as_int(
                metadata.get("spacegroup_number", metadata.get("spacegroup"))
            )
            band_gap = _as_float(metadata.get("band_gap"))
            source = str(metadata.get("source", "unknown")).lower()
            source_efermi_absolute = _as_float(
                metadata.get("source_efermi_absolute", metadata.get("efermi"))
            )
            energy_reference = str(metadata.get("energy_reference") or "absolute")
            if (source == "aflow" and not canonical_contract) or energy_reference == "fermi_shifted_zero":
                efermi = 0.0
                energy_reference = "fermi_shifted_zero"
            else:
                efermi = _as_float(metadata.get("efermi"))
            provider_is_metal = _as_bool(metadata.get("is_metal"))
            is_metal = provider_is_metal
            metal_label_source = "provider_is_metal"
            if is_metal is None:
                is_metal = bool((band_gap or 0.0) <= 0.01)
                metal_label_source = "fallback_band_gap"
            is_direct_value = _as_bool(metadata.get("is_direct"))

            if spacegroup is None:
                skipped.append({"material_id": material_id, "reason": "missing spacegroup"})
                continue
            if band_gap is None or not np.isfinite(band_gap):
                skipped.append({"material_id": material_id, "reason": "missing band_gap"})
                continue

            raw_energies = grp["energies"][:]
            raw_labels = _parse_json_attr(metadata.get("kpath_labels"), [])
            raw_segment_ids = _segment_ids_from_resampled_labels(
                raw_labels, raw_energies.shape[-1]
            )
            resampled_labels = _resampled_kpath_labels(metadata, target_k_points)
            segment_ids = _segment_ids_from_resampled_labels(resampled_labels, target_k_points)
            source_segment_bounds = None
            if canonical_contract:
                resampled_labels, raw_segment_ids, segment_ids, source_segment_bounds = _canonical_path_layout(
                    raw_labels, axis, target_k_points)
            try:
                tensor, vbm_idx, cbm_idx, metal_feature_inferred = _build_sample_tensor(
                    raw_energies,
                    efermi=efermi,
                    target_k_points=target_k_points,
                    k_distances=grp["k_distances"][:] if "k_distances" in grp else None,
                    segment_ids=segment_ids,
                    raw_segment_ids=raw_segment_ids,
                    source_segment_bounds=source_segment_bounds,
                )
            except Exception as exc:
                skipped.append({"material_id": material_id, "reason": str(exc)})
                continue

            # Data-quality filter: skip samples whose tensor band gap is strongly
            # negative (VBM/CBM selection pathology in some AFLOW records). A small
            # negative tolerance is kept for numerical noise; anything beyond it is
            # physically inconsistent with a non-negative DFT gap label.
            tensor_gap = float(np.min(tensor[1, :, 0]) - np.max(tensor[0, :, 0]))
            if not metal_feature_inferred and tensor_gap < -0.05:
                skipped.append({
                    "material_id": material_id,
                    "reason": f"negative tensor gap ({tensor_gap:.3f} eV)",
                })
                continue

            if canonical_contract:
                # Only new construction enters the E-E_F contract; legacy arrays
                # and legacy callers retain their original energy convention.
                tensor[:, :, 0] -= float(efermi)
            tensors.append(tensor)
            gaps.append(float(band_gap))
            groups.append(int(spacegroup))
            material_ids.append(material_id)
            if is_metal:
                gap_type = 0
                gap_type_source = "provider_is_metal"
            elif is_direct_value is not None:
                gap_type = 1 if bool(is_direct_value) else 2
                gap_type_source = "provider_is_direct"
            else:
                gap_type = 1 if int(np.argmax(tensor[0, :, 0])) == int(np.argmin(tensor[1, :, 0])) else 2
                gap_type_source = "derived_from_resampled_extrema"
            gap_types.append(gap_type)
            if "k_distances" in grp:
                raw_k = np.asarray(grp["k_distances"][:], dtype=np.float64)
                metadata["k_distance_min"] = float(np.min(raw_k))
                metadata["k_distance_max"] = float(np.max(raw_k))
            sample_metadata.append(
                {
                    "material_id": material_id,
                    "spacegroup_number": int(spacegroup),
                    "band_gap": float(band_gap),
                    "efermi": efermi,
                    "source_efermi_absolute": source_efermi_absolute,
                    "energy_reference": energy_reference,
                    "protocol": str(metadata.get('dft_type') or 'unknown'),
                    "energy_quantity": str(metadata.get('energy_quantity') or 'unknown'),
                    "vbm_band_idx": int(vbm_idx),
                    "cbm_band_idx": int(cbm_idx),
                    "gap_type": int(gap_type),
                    "gap_type_source": gap_type_source,
                    "metal_label_source": metal_label_source,
                    "metal_feature_inferred": bool(metal_feature_inferred),
                    "metal_label_matches_feature": bool(bool(is_metal) == bool(metal_feature_inferred)),
                    "source": str(metadata.get("source", "unknown")),
                    "formula_pretty": str(metadata.get("formula_pretty") or ""),
                    "pearson_symbol": str(metadata.get("pearson_symbol") or ""),
                    "num_sites": _as_int(metadata.get("num_sites")),
                    "k_coordinate": "source cumulative path coordinate",
                    "kpath_labels": resampled_labels,
                    "segment_ids": segment_ids.tolist(),
                }
            )

    if not tensors:
        raise RuntimeError(f"No valid band tensors could be built from {h5_path}")

    return (
        np.stack(tensors, axis=0).astype(np.float32),
        np.asarray(gaps, dtype=np.float32),
        np.asarray(groups, dtype=np.int32),
        np.asarray(material_ids, dtype="U64"),
        np.asarray(gap_types, dtype=np.int32),
        [{"samples": sample_metadata, "skipped": skipped}],
    )


def build_group_ood_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
    class_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an OOD split with no spacegroup overlap.

    Candidate group subsets are scored for requested sample fraction and class
    distribution. The outer contract remains an exact group-disjoint split.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")
    groups = np.asarray(groups)
    labels = np.asarray(class_labels if class_labels is not None else (np.asarray(y) > 0.01), dtype=np.int32)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct groups are required")
    rng = np.random.RandomState(random_state)
    target_test = 1.0 - train_size
    observed_classes = np.unique(labels)
    global_dist = np.asarray([np.mean(labels == cls) for cls in observed_classes])
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    group_counts = {group: int(np.sum(groups == group)) for group in unique_groups}
    target_test_count = target_test * len(groups)

    for _ in range(max(256, min(2048, len(unique_groups) * 16))):
        shuffled = rng.permutation(unique_groups)
        cumulative = np.cumsum([group_counts[group] for group in shuffled])
        center = int(np.argmin(np.abs(cumulative - target_test_count))) + 1
        candidate_cuts = {
            max(1, min(len(shuffled) - 1, center + offset))
            for offset in range(-3, 4)
        }
        for cut in candidate_cuts:
            test_set = set(shuffled[:cut].tolist())
            test_idx = np.flatnonzero(np.isin(groups, list(test_set)))
            train_idx = np.flatnonzero(~np.isin(groups, list(test_set)))
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            test_dist = np.asarray([np.mean(labels[test_idx] == cls) for cls in observed_classes])
            missing = sum(
                int(not np.any(labels[split] == cls))
                for cls in observed_classes
                for split in (train_idx, test_idx)
            )
            score = (
                abs(len(test_idx) / len(groups) - target_test)
                + 0.35 * float(np.mean(np.abs(test_dist - global_dist)))
                + 2.0 * missing
            )
            if best is None or score < best[0]:
                best = (score, train_idx, test_idx)

    if best is None:
        raise RuntimeError("Could not construct a non-empty group split")
    _, train_idx, test_idx = best

    train_groups_set = set(groups[train_idx].tolist()) if len(train_idx) > 0 else set()
    test_groups_set = set(groups[test_idx].tolist()) if len(test_idx) > 0 else set()
    overlap = train_groups_set.intersection(test_groups_set)
    if overlap:
        raise RuntimeError(f"Group leakage detected across train/test: {sorted(overlap)}")

    return train_idx, test_idx


def build_group_validation_split(
    groups: np.ndarray,
    validation_size: float = 0.15,
    random_state: int = 42,
    class_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a group-disjoint inner validation split from outer-training data."""
    dummy = np.zeros(len(groups), dtype=np.float32)
    return build_group_ood_split(
        dummy,
        dummy,
        groups,
        train_size=1.0 - validation_size,
        random_state=random_state,
        class_labels=class_labels,
    )


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _composition_key(sample: Dict[str, Any]) -> str:
    formula = str(sample.get("formula_pretty") or "")
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))
    return "-".join(elements) if elements else "unknown"


def _prototype_key(sample: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(sample.get("pearson_symbol") or "unknown"),
            f"sg{sample.get('spacegroup_number', 'unknown')}",
            f"n{sample.get('num_sites', 'unknown')}",
        ]
    )


def _count_values(samples: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        value = str(sample.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def save_ood_tensors(
    output_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    material_ids: np.ndarray,
    gap_types: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata_bundle: List[Dict[str, Any]],
    target_k_points: int,
    train_size: float,
    random_state: int,
    source_h5_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    *, _construction: Optional[dict] = None,
) -> Dict[str, Any]:
    n = len(X)
    parts = (np.asarray(train_idx), np.asarray(test_idx))
    if (any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or not len(a) for a in parts)
            or any(len(a) != n for a in (y, groups, material_ids, gap_types))
            or len(set(material_ids.tolist())) != n
            or not np.array_equal(np.sort(np.concatenate(parts)), np.arange(n))):
        raise ValueError('Split indices/IDs must partition actual rows exactly once')
    if set(groups[train_idx].tolist()) & set(groups[test_idx].tolist()):
        raise ValueError('Actual train/test groups overlap')
    os.makedirs(output_dir, exist_ok=True)

    full_npz = os.path.join(output_dir, "band_tensors_full.npz")
    split_npz = os.path.join(output_dir, "band_tensors_ood_split.npz")
    manifest_path = os.path.join(output_dir, "ood_split_manifest.json")
    samples = metadata_bundle[0]["samples"]
    if len(samples) != len(X):
        raise ValueError("metadata sample count must match tensor count")
    segment_ids = np.asarray(
        [sample.get("segment_ids", [0] * target_k_points) for sample in samples],
        dtype=np.int32,
    )
    if segment_ids.shape != (len(X), target_k_points):
        raise ValueError("segment_ids must have shape (N, target_k_points)")

    np.savez_compressed(
        full_npz,
        X=X,
        y_gap=y,
        groups=groups,
        material_ids=material_ids,
        y_type=gap_types,
        segment_ids=segment_ids,
    )
    np.savez_compressed(
        split_npz,
        X_train=X[train_idx],
        y_train=y[train_idx],
        groups_train=groups[train_idx],
        material_ids_train=material_ids[train_idx],
        y_type_train=gap_types[train_idx],
        segment_ids_train=segment_ids[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        groups_test=groups[test_idx],
        material_ids_test=material_ids[test_idx],
        y_type_test=gap_types[test_idx],
        segment_ids_test=segment_ids[test_idx],
        **(_construction or {}),
    )

    train_groups = sorted(set(groups[train_idx].tolist()))
    test_groups = sorted(set(groups[test_idx].tolist()))
    train_samples = [samples[int(index)] for index in train_idx]
    test_samples = [samples[int(index)] for index in test_idx]
    train_compositions = {_composition_key(sample) for sample in train_samples}
    test_compositions = {_composition_key(sample) for sample in test_samples}
    train_prototypes = {_prototype_key(sample) for sample in train_samples}
    test_prototypes = {_prototype_key(sample) for sample in test_samples}
    mismatches = [
        sample["material_id"]
        for sample in samples
        if not bool(sample.get("metal_label_matches_feature", True))
    ]
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
        "split_method": "optimized stratified group holdout",
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
            "train": {str(k): int(v) for k, v in zip(*np.unique(gap_types[train_idx], return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(gap_types[test_idx], return_counts=True))},
        },
        "metal_feature_audit": {
            "inferred_crossing_count": int(sum(bool(s.get("metal_feature_inferred")) for s in samples)),
            "mismatch_count": int(len(mismatches)),
            "mismatch_material_ids": mismatches,
        },
        "distribution_audit": {
            "composition_overlap_count": int(len(train_compositions & test_compositions)),
            "prototype_overlap_count": int(len(train_prototypes & test_prototypes)),
            "train_source_distribution": _count_values(train_samples, "source"),
            "test_source_distribution": _count_values(test_samples, "source"),
        },
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_h5_path": source_h5_path,
            "source_h5_sha256": _sha256_file(source_h5_path),
            "metadata_path": metadata_path,
            "metadata_sha256": _sha256_file(metadata_path),
            "builder_sha256": _sha256_file(__file__),
            "full_npz_sha256": _sha256_file(full_npz),
            "split_npz_sha256": _sha256_file(split_npz),
        },
        "outputs": {
            "full_npz": full_npz,
            "split_npz": split_npz,
        },
        "train_material_ids": material_ids[train_idx].tolist(),
        "test_material_ids": material_ids[test_idx].tolist(),
        "train_spacegroups": train_groups,
        "test_spacegroups": test_groups,
        "samples": samples,
        "skipped": metadata_bundle[0]["skipped"],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD tensors from MP band data")
    parser.add_argument("--h5", default="./data/raw/materials_project/mp_bands.h5", help="input HDF5 path")
    parser.add_argument("--metadata", default="./data/raw/materials_project/mp_metadata.json", help="metadata JSON path")
    parser.add_argument("--output", default="./data/processed/materials_project/ood_tensors", help="output directory")
    parser.add_argument("--target-k", type=int, default=128, help="fixed k-point sequence length")
    parser.add_argument("--train-size", type=float, default=0.8, help="StratifiedGroupShuffleSplit train size")
    parser.add_argument("--random-state", type=int, default=42, help="StratifiedGroupShuffleSplit random seed")
    parser.add_argument('--test-scope', action='store_true', help='synthetic engineering only, at most 32 raw rows')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Refusing nonempty tensor output directory')
    with h5py.File(args.h5, 'r') as source_file:
        raw_rows = len(source_file)
        synthetic = source_file.attrs.get('scope') == 'test'
    if args.test_scope and (not synthetic or not 0 < raw_rows <= 32):
        raise ValueError('CPU test construction requires marked synthetic HDF5 and <=32 raw rows')
    if not args.test_scope and synthetic:
        raise ValueError('Synthetic HDF5 cannot be labelled formal')
    source_before = p2_file_ref(args.h5)
    metadata_before = p2_file_ref(args.metadata) if args.metadata else None
    code_before = p2_code_refs(P2_RAW_CODE)
    construction_id = uuid.uuid4().hex
    scope = 'test' if args.test_scope else 'formal'
    embedded = {'construction_id': np.asarray(construction_id)}
    for split in ('train', 'test'):
        embedded.update({f'input_space_{split}': np.asarray('raw_canonical_6d'),
                         f'scope_{split}': np.asarray(scope),
                         f'feature_schema_{split}': np.asarray(json.dumps(p2_feature_schema(args.target_k)))})

    X, y, groups, material_ids, gap_types, metadata_bundle = process_band_data(
        h5_path=args.h5,
        metadata_path=args.metadata,
        target_k_points=args.target_k,
        canonical_contract=True,
    )
    train_idx, test_idx = build_group_ood_split(
        X,
        y,
        groups,
        train_size=args.train_size,
        random_state=args.random_state,
        class_labels=gap_types,
    )
    if (p2_file_ref(args.h5) != source_before
            or (args.metadata and p2_file_ref(args.metadata) != metadata_before)
            or p2_code_refs(P2_RAW_CODE) != code_before):
        raise ValueError('Source/metadata/code changed during tensor construction')
    source_record = {'kind': 'new_hdf5_construction', 'raw_rows': raw_rows,
                     'h5_ref': source_before, 'metadata_ref': metadata_before}
    qualifications = {}
    for split, indices in (('train', train_idx), ('test', test_idx)):
        qualifications[split] = [
            {key: metadata_bundle[0]['samples'][int(i)][key]
             for key in ('material_id', 'source', 'efermi', 'energy_reference', 'protocol', 'energy_quantity')}
            for i in indices]
        embedded[f'observation_sha256_{split}'] = np.asarray(p2_raw_observation_sha(source_record, qualifications[split]))
    manifest = save_ood_tensors(
        output_dir=args.output,
        X=X,
        y=y,
        groups=groups,
        material_ids=material_ids,
        gap_types=gap_types,
        train_idx=train_idx,
        test_idx=test_idx,
        metadata_bundle=metadata_bundle,
        target_k_points=args.target_k,
        train_size=args.train_size,
        random_state=args.random_state,
        source_h5_path=args.h5,
        metadata_path=args.metadata,
        _construction=embedded,
    )

    # Emitted only after raw HDF5 construction, never for pre-existing NPZs.
    tensor_path = Path(args.output) / 'band_tensors_ood_split.npz'
    for split, indices in (('train', train_idx), ('test', test_idx)):
        samples = [metadata_bundle[0]['samples'][int(i)] for i in indices]
        contract = {
            'schema_version': 1, 'kind': 'raw_tensor',
            'scope': 'test' if args.test_scope else 'formal', 'split': split,
            'construction_id': construction_id, 'input_space': 'raw_canonical_6d',
            'feature_schema': p2_feature_schema(args.target_k),
            'raw_semantics': P2_RAW_SEMANTICS,
            'material_ids': material_ids[indices].tolist(), 'groups': groups[indices].tolist(),
            'tensor_ref': p2_file_ref(tensor_path), 'code': code_before,
            'source': source_record,
            'source_qualifications': qualifications[split],
        }
        path = Path(args.output) / f'band_tensors_ood_split.{split}.tensor_contract.json'
        path.write_text(json.dumps(contract, indent=2, allow_nan=False), encoding='utf-8')

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
