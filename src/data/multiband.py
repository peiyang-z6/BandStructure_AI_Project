"""P3 variable multi-band extraction (Constitution 5.0 §8 P3).

Extracts Fermi-level-proximate bands from raw AFLOW HDF5 energies and
resamples them onto a fixed k-grid, supporting a VARIABLE number of bands via
a boolean band mask (unlike the fixed 2-band edge-envelope tensor).

Design:
- A clean occupied/empty split with no EF-straddling band uses a balanced
  VBM/CBM-side selection. Otherwise, window overlap prioritizes EF-straddling
  intervals before deterministic truncation. This is selection, not a global
  metal/insulator label; true topology requires segment-aware evidence.
- Keep complete selected trajectories and report capped capacity. The entire
  trajectory need not lie inside the selection window.
- Normalize the cumulative k axis, split duplicate-distance boundaries BEFORE
  independent PCHIP interpolation, and retain the same segment assignments.
  `path_segments` is shared with the preparation script; no averaging of
  disconnected-branch energies is permitted. Output energies are EF-relative.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from scipy.interpolate import PchipInterpolator


def select_fermi_bands(
    energies: np.ndarray,
    efermi: float,
    max_bands: int,
    delta_e: float,
) -> np.ndarray:
    """Return sorted band indices around the Fermi level.

    Clean split (both frontier sides present AND no EF-straddling band): select
    the `max_bands//2` valence bands nearest the VBM (highest band_max) plus the
    remaining conduction bands nearest the CBM (lowest band_min). This keeps BOTH
    sides of the gap — AFLOW anchors E_F at the VBM, so a pure distance-to-E_F
    sort would otherwise pick only valence bands.

    EF-straddling or incomplete-frontier input: keep bands whose
    [min,max] interval overlaps [E_F - delta_e, E_F + delta_e], truncating to
    the `max_bands` nearest E_F (or the two nearest when the window is empty).
    """
    energies = np.asarray(energies, dtype=np.float32)
    if energies.ndim == 3:
        energies = energies.reshape(-1, energies.shape[-1])
    if energies.ndim != 2:
        raise ValueError(f"expected 2D or spin-resolved 3D energies, got {energies.shape}")
    num_bands = energies.shape[0]
    if num_bands < 1:
        raise ValueError("need at least one band")

    band_min = np.min(energies, axis=1)
    band_max = np.max(energies, axis=1)
    ef = float(efermi)

    valence = np.where(band_max <= ef)[0]
    # An upward EF touch is conduction; a flat EF band remains valence-only.
    conduction = np.where((band_min >= ef) & (band_max > ef))[0]

    crossing = (band_min < ef) & (band_max > ef)
    if not crossing.any() and len(valence) > 0 and len(conduction) > 0:
        n_v = max_bands // 2
        n_c = max_bands - n_v
        vbm_order = valence[np.argsort(-band_max[valence])]  # VBM first
        cbm_order = conduction[np.argsort(band_min[conduction])]  # CBM first
        sel = np.concatenate([vbm_order[:n_v], cbm_order[:n_c]])
        return np.sort(sel).astype(np.int64)

    # metal/semimetal fallback: window overlap + distance-to-E_F truncation
    lo = ef - float(delta_e)
    hi = ef + float(delta_e)
    overlap = (band_min <= hi) & (band_max >= lo)
    idx = np.where(overlap)[0]

    if len(idx) == 0:
        dist = np.minimum(np.abs(band_min - ef), np.abs(band_max - ef))
        idx = np.argsort(dist)[: min(2, num_bands)]

    if len(idx) > max_bands:
        dist = np.maximum(np.maximum(band_min[idx] - ef, ef - band_max[idx]), 0.0)
        order = np.lexsort((idx, dist, ~crossing[idx]))
        idx = idx[order[:max_bands]]

    return np.sort(idx).astype(np.int64)


def path_segments(k_distances: np.ndarray, n_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Source and resampled segment IDs; duplicate distances start a new segment."""
    k = np.asarray(k_distances, dtype=np.float64).reshape(-1)
    if (len(k) < 2 or n_k < 2 or not np.isfinite(k).all()
            or np.any(np.diff(k) < 0) or k[-1] - k[0] <= 1e-12):
        raise ValueError("k_distances must be finite, monotone, with positive span")
    starts = np.r_[0, np.flatnonzero(np.diff(k) <= 1e-12) + 1]
    source = np.cumsum(np.isin(np.arange(len(k)), starts)).astype(np.int32) - 1
    axis = np.linspace(k[0], k[-1], n_k)
    target = np.searchsorted(k[starts], axis, side="right").astype(np.int32) - 1
    return source, target


def extract_fermi_bands(
    energies: np.ndarray,
    k_distances: np.ndarray,
    efermi: float,
    max_bands: int,
    delta_e: float,
    n_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (max_bands, n_k) Fermi-proximate bands + mask + k-axis.

    The k-path is normalized to [0, 1] so every material shares one decoder
    k-grid (relative k-path position), independent of its absolute path length.

    Returns:
        bands: (max_bands, n_k) float32, zero-padded for unused slots
        mask:  (max_bands,) bool, True for real bands
        k_axis: (n_k,) float32 uniform normalized k-grid in [0, 1]
    """
    energies = np.asarray(energies, dtype=np.float32)
    if energies.ndim == 3:
        energies = energies.reshape(-1, energies.shape[-1])
    if energies.ndim != 2:
        raise ValueError(f"expected 2D or spin-resolved 3D energies, got {energies.shape}")

    idx = select_fermi_bands(energies, efermi, max_bands, delta_e)
    n_sel = len(idx)

    mask = np.zeros(max_bands, dtype=bool)
    mask[:n_sel] = True
    bands = np.zeros((max_bands, n_k), dtype=np.float32)

    # Normalize the source k-path to [0, 1] for a unified decoder grid.
    k = np.asarray(k_distances, dtype=np.float64).reshape(-1)
    if len(k) != energies.shape[-1]:
        raise ValueError("k_distances length must match energies")
    source_segments, target_segments = path_segments(k, n_k)
    k_norm = (k - float(k[0])) / float(k[-1] - k[0])

    k_axis = np.linspace(0.0, 1.0, n_k, dtype=np.float32)

    # Duplicate distances delimit source path segments. Never average two
    # different branches into a fictitious energy at their shared x-coordinate.
    starts = np.r_[0, np.flatnonzero(np.diff(source_segments)) + 1]
    for sid, start in enumerate(starts):
        stop = starts[sid + 1] if sid + 1 < len(starts) else len(k_norm)
        positions = np.flatnonzero(target_segments == sid)
        if not len(positions):
            continue
        values = energies[idx, start:stop] - float(efermi)
        if stop - start == 1:
            bands[:n_sel, positions] = values
        else:
            query = np.clip(k_axis[positions], k_norm[start], k_norm[stop - 1])
            bands[:n_sel, positions] = PchipInterpolator(
                k_norm[start:stop], values, axis=1)(query)

    return bands, mask, k_axis
