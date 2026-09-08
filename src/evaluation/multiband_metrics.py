"""P3 multi-band evaluation metrics (numpy, no TF dependency).

`derive_gap` computes per-sample gaps from band energies. `band_mae` and
`gap_mae` evaluate a multi-band decoder against targets. The insulator/metal
split is decided from the TARGET bands only, never from the prediction —
otherwise a metal whose prediction spuriously produces a conduction-side band
would get a fake non-zero predicted gap and inflate gap MAE.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def derive_gap(
    bands: np.ndarray,
    mask: np.ndarray,
    efermi: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample gap and insulator mask from (B, max_bands, n_k) energies.

    Returns:
        gap: (B,) gap in eV; 0.0 for metals (no clean valence/conduction split)
        is_insulator: (B,) bool, True when BOTH a valence-side and a
            conduction-side band exist in the given `bands`
    """
    bands = np.asarray(bands, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    bmin = bands.min(axis=-1)  # (B, max_bands)
    bmax = bands.max(axis=-1)
    m = mask.astype(np.float32)

    has_val = ((bmax <= efermi) & mask).any(axis=-1)
    has_con = ((bmin >= efermi) & mask).any(axis=-1)
    is_insulator = has_val & has_con

    vbm = np.where(bmax <= efermi, bmax, -1e9) * m
    cbm = np.where(bmin >= efermi, bmin, 1e9)
    cbm = np.where(m > 0, cbm, 1e9)

    gap = np.where(is_insulator, cbm.min(axis=-1) - vbm.max(axis=-1), 0.0)
    return gap.astype(np.float32), is_insulator


def band_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Masked mean absolute error over (B, max_bands, n_k)."""
    err = np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32))
    m = np.asarray(mask, dtype=bool).astype(np.float32)
    n_k = pred.shape[-1]
    denom = m.sum() * n_k
    if denom == 0:
        return float("nan")
    return float((err * m[..., None]).sum() / denom)


def sorted_band_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Band MAE after per-k-point 1D optimal-transport (sorting) band matching.

    Measures energy discrepancy independent of band identity (crossings), so it
    is the band-matching counterpart of `band_mae` used as the P3 primary metric.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    large = 1e9
    pm = np.where(mask[..., None], pred, large)
    tm = np.where(mask[..., None], target, large)
    ps = np.sort(pm, axis=1)
    ts = np.sort(tm, axis=1)
    ms = np.sort(mask.astype(np.float32), axis=1)[:, ::-1]  # valid first
    err = np.abs(ps - ts)
    err = np.where(np.isfinite(err), err, 0.0)
    denom = ms.sum() * pred.shape[-1]
    if denom == 0:
        return float("nan")
    return float((err * ms[..., None]).sum() / denom)


def gap_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, efermi: float = 0.0) -> float:
    """Gap MAE over insulators only (insulator/metal decided from TARGET)."""
    pred_gap, _ = derive_gap(pred, mask, efermi)
    true_gap, is_insulator = derive_gap(target, mask, efermi)
    if not is_insulator.any():
        return float("nan")
    return float(np.abs(pred_gap[is_insulator] - true_gap[is_insulator]).mean())
