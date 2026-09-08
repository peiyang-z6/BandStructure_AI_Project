"""P3 multi-band evaluation metrics (numpy, no TF dependency).

Gaps are derived independently from targets and predictions. False gaps on
metals are real errors, not a reason to drop metals. `evaluate_multiband`
reports separate populations and unknown/coverage counts. The older `gap_mae`
API remains an explicitly insulator-only diagnostic, not the main metric.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def derive_gap(
    bands: np.ndarray,
    mask: np.ndarray,
    efermi: float = 0.0,
    segment_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample gap and insulator mask from (B, max_bands, n_k) energies.

    Returns:
        gap: (B,) selected-window gap in eV; zero for a witnessed within-segment
            crossing or touching valence/conduction edges, NaN for missing edges
            or unresolved between-segment changes.
        is_insulator: (B,) True only for a positive gap with a clean occupied/empty
            split and no EF-straddling band. Not a provider-global electronic label.
    """
    bands = np.asarray(bands, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    bmin = bands.min(axis=-1)  # (B, max_bands)
    bmax = bands.max(axis=-1)
    m = mask.astype(np.float32)

    has_val = ((bmax <= efermi) & mask).any(axis=-1)
    has_con = ((bmin >= efermi) & (bmax > efermi) & mask).any(axis=-1)
    straddles = ((bmin < efermi) & (bmax > efermi) & mask).any(axis=-1)
    has_crossing = straddles.copy()
    if segment_ids is not None:
        segments = np.asarray(segment_ids, dtype=np.int32)
        if segments.ndim == 1:
            segments = np.broadcast_to(segments, (len(bands), bands.shape[-1]))
        if segments.shape != (len(bands), bands.shape[-1]):
            raise ValueError("segment_ids must match sample and k dimensions")
        has_crossing[:] = False
        for i in np.flatnonzero(straddles):
            for sid in np.unique(segments[i]):
                values = bands[i][:, segments[i] == sid]
                crosses = (values.min(axis=-1) < efermi) & (values.max(axis=-1) > efermi)
                if np.any(crosses & mask[i]):
                    has_crossing[i] = True
                    break
    resolved_split = has_val & has_con & ~straddles

    vbm = np.where((bmax <= efermi) & mask, bmax, -1e9)
    cbm = np.where((bmin >= efermi) & (bmax > efermi), bmin, 1e9)
    cbm = np.where(m > 0, cbm, 1e9)

    gap = np.where(resolved_split, cbm.min(axis=-1) - vbm.max(axis=-1),
                   np.where(has_crossing, 0.0, np.nan))
    is_insulator = resolved_split & (gap > 0.0)
    return gap.astype(np.float32), is_insulator


def band_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Masked mean absolute error over (B, max_bands, n_k)."""
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        return float("nan")
    p = np.asarray(pred)[valid].astype(np.float64)
    t = np.asarray(target)[valid].astype(np.float64)
    return float(np.abs(p - t).mean())


def sorted_band_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Band MAE after per-k-point 1D optimal-transport (sorting) band matching.

    Measures energy discrepancy independent of band identity (crossings), so it
    is the band-matching counterpart of `band_mae` used as the P3 primary metric.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("valid band energies must be finite")
    large = np.inf
    pm = np.where(mask[..., None], pred, large)
    tm = np.where(mask[..., None], target, large)
    ps = np.sort(pm, axis=1)
    ts = np.sort(tm, axis=1)
    ms = np.sort(mask.astype(np.float32), axis=1)[:, ::-1]  # valid first
    ps = np.where(ms[..., None] > 0, ps, 0.)
    ts = np.where(ms[..., None] > 0, ts, 0.)
    err = np.abs(ps.astype(np.float64) - ts.astype(np.float64))
    denom = ms.sum() * pred.shape[-1]
    if denom == 0:
        return float("nan")
    return float((err * ms[..., None]).sum() / denom)


def evaluate_multiband(pred, target, mask, segment_ids=None) -> dict:
    """Window/line diagnostics, not a provider-global DFT gap benchmark.

    Report both metals and insulators; never hide unresolved targets/predictions.
    Per-k OT does not establish whole-trajectory identity or continuity.
    """
    true_gap, true_ins = derive_gap(target, mask, segment_ids=segment_ids)
    pred_gap, _ = derive_gap(pred, mask, segment_ids=segment_ids)
    true_known = np.isfinite(true_gap)
    pred_known = np.isfinite(pred_gap)
    resolved = true_known & pred_known
    metals = true_known & ~true_ins
    errors = np.abs(pred_gap - true_gap)
    def average(values, selected):
        return float(np.mean(values[selected])) if np.any(selected) else None
    return {
        "scope": "selected_window_segment_aware_line_diagnostic",
        "n_samples": int(len(true_gap)),
        "band_mae_slot": band_mae(pred, target, mask) if np.any(mask) else None,
        "band_mae_per_k_ot": sorted_band_mae(pred, target, mask) if np.any(mask) else None,
        "n_target_insulator": int(true_ins.sum()),
        "n_target_metal": int(metals.sum()),
        "n_target_unknown": int((~true_known).sum()),
        "n_prediction_unknown": int((~pred_known).sum()),
        "n_gap_pairs_resolved": int(resolved.sum()),
        "gap_prediction_coverage": float(resolved.sum() / true_known.sum()) if true_known.any() else None,
        "gap_mae_all_resolved": average(errors, resolved),
        "gap_mae_insulator_resolved": average(errors, resolved & true_ins),
        "gap_mae_metal_resolved": average(errors, resolved & metals),
        "false_gap_rate_on_resolved_metals": average((pred_gap > 1e-6).astype(float), resolved & metals),
    }


def gap_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, efermi: float = 0.0) -> float:
    """Gap MAE over insulators only (insulator/metal decided from TARGET)."""
    pred_gap, _ = derive_gap(pred, mask, efermi)
    true_gap, is_insulator = derive_gap(target, mask, efermi)
    if not is_insulator.any():
        return float("nan")
    return float(np.abs(pred_gap[is_insulator] - true_gap[is_insulator]).mean())
