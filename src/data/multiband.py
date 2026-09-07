"""P3 variable multi-band extraction (Constitution 5.0 §8 P3).

Extracts Fermi-level-proximate bands from raw AFLOW HDF5 energies and
resamples them onto a fixed k-grid, supporting a VARIABLE number of bands via
a boolean band mask (unlike the fixed 2-band edge-envelope tensor).

Design:
- `select_fermi_bands` keeps bands whose [min,max] energy interval overlaps
  [E_F - delta_e, E_F + delta_e]; if that window is empty it falls back to the
  two bands nearest E_F; if more than `max_bands` qualify it keeps the
  `max_bands` nearest to E_F. This yields a physically meaningful "bands around
  the Fermi level" slice with a variable count.
- `extract_fermi_bands` resamples each selected band onto a uniform k-grid
  (shape-preserving PCHIP, no cubic-spline overshoot near the Fermi level) and
  pads to (max_bands, n_k) with a boolean mask.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.data.ood_tensor_builder import _resample_band


def select_fermi_bands(
    energies: np.ndarray,
    efermi: float,
    max_bands: int,
    delta_e: float,
) -> np.ndarray:
    """Return sorted band indices around the Fermi level.

    Insulator/semiconductor (both valence and conduction bands present): select
    the `max_bands//2` valence bands nearest the VBM (highest band_max) plus the
    remaining conduction bands nearest the CBM (lowest band_min). This keeps BOTH
    sides of the gap — AFLOW anchors E_F at the VBM, so a pure distance-to-E_F
    sort would otherwise pick only valence bands.

    Metal/semimetal (no clean valence/conduction split): keep bands whose
    [min,max] interval overlaps [E_F - delta_e, E_F + delta_e], truncating to
    the `max_bands` nearest E_F (or the two nearest when the window is empty).
    """
    energies = np.asarray(energies, dtype=np.float32)
    if energies.ndim != 2:
        raise ValueError(f"expected 2D energies (num_bands, num_kpoints), got {energies.shape}")
    num_bands = energies.shape[0]
    if num_bands < 1:
        raise ValueError("need at least one band")

    band_min = np.min(energies, axis=1)
    band_max = np.max(energies, axis=1)
    ef = float(efermi)

    valence = np.where(band_max <= ef)[0]
    conduction = np.where(band_min > ef)[0]  # strict > so band==E_F is valence-only

    if len(valence) > 0 and len(conduction) > 0:
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
        dist = np.minimum(np.abs(band_min[idx] - ef), np.abs(band_max[idx] - ef))
        idx = idx[np.argsort(dist)[:max_bands]]

    return np.sort(idx).astype(np.int64)


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
    if energies.ndim != 2:
        raise ValueError(f"expected 2D energies, got {energies.shape}")

    idx = select_fermi_bands(energies, efermi, max_bands, delta_e)
    n_sel = len(idx)

    mask = np.zeros(max_bands, dtype=bool)
    mask[:n_sel] = True
    bands = np.zeros((max_bands, n_k), dtype=np.float32)

    # Normalize the source k-path to [0, 1] for a unified decoder grid.
    k = np.asarray(k_distances, dtype=np.float64).reshape(-1)
    if len(k) >= 2 and float(k[-1] - k[0]) > 1.0e-12:
        k_norm = (k - float(k[0])) / float(k[-1] - k[0])
    else:
        k_norm = np.linspace(0.0, 1.0, len(k), dtype=np.float64)

    k_axis = np.linspace(0.0, 1.0, n_k, dtype=np.float32)

    for slot, band_idx in enumerate(idx):
        bands[slot] = _resample_band(
            energies[band_idx], n_k, k_norm, shape_preserving=True
        )

    return bands, mask, k_axis
