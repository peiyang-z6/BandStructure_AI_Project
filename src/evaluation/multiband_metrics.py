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


def band_path_diagnostics(bands, mask, segment_ids, *, efermi=0., band_groups=None) -> dict:
    """Selected-slot and rank evidence; occupation counts are not electron counts.

    A strict relabeling-invariant crossing requires max N(E<EF) > min N(E<=EF)
    WITHIN a segment. EF touches alone do not satisfy this sufficient condition.
    Every contiguous segment run is independent, even when its label repeats;
    -1 marks k padding. Masked bands and padded k never enter arithmetic.
    ``band_groups`` identifies comparable source-order blocks (e.g. spins).
    Its default is one block, so callers with flattened spins must supply it.
    Dispersion RMS is sample-point-weighted, segment-centered energy spread,
    NOT a k derivative, physical curvature, effective mass or wavefunction
    identity. Energy ranks combine valid bands; inversions do not mix blocks.
    """
    mask = np.asarray(mask)
    bands = np.asarray(bands, dtype=np.float64)
    segments = np.asarray(segment_ids)
    if (bands.ndim != 2 or mask.shape != bands.shape[:1] or mask.dtype.kind != "b"
            or segments.shape != bands.shape[1:] or segments.dtype.kind not in "iu"
            or np.any(segments < -1) or not np.isfinite(efermi)):
        raise ValueError("require 2D bands, boolean band mask, aligned integer segments >= -1 and finite EF")
    values = bands[mask]
    groups = np.zeros(len(mask), dtype=int) if band_groups is None else np.asarray(band_groups)
    if groups.shape != mask.shape or groups.dtype.kind not in "iu":
        raise ValueError("band_groups must be aligned integer spin/order groups")
    groups = groups[mask]
    valid_k = segments >= 0
    if not np.isfinite(values[:, valid_k]).all():
        raise ValueError("valid band energies must be finite")
    result = {"scope": "selected_slot_and_energy_rank_line_diagnostic",
              "causal_attribution": "not_established", "n_valid_bands": int(len(values)),
              "n_valid_k": int(valid_k.sum()), "gap_eV": None, "segments": [],
              "interval_straddling_bands": 0, "within_segment_crossing_bands": 0}
    if not len(values) or not valid_k.any():
        result["frontier_state"] = "unresolved_no_valid_bands" if not len(values) else "unresolved_no_valid_k"
        return result
    rows = []
    crossing = np.zeros(len(values), dtype=bool)
    bounds = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(segments)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        sid = segments[start]
        if sid < 0:
            continue
        local = values[:, start:stop]
        crosses = (local.min(axis=1) < efermi) & (local.max(axis=1) > efermi)
        crossing |= crosses
        below = np.sum(local < efermi, axis=0)
        nonpositive = np.sum(local <= efermi, axis=0)
        differences = [np.diff(local[groups == group], axis=0).ravel() for group in np.unique(groups)]
        differences = np.concatenate(differences)
        ranks = np.sort(local, axis=0)
        rows.append({"segment_id": int(sid), "n_points": int(local.shape[-1]),
                     "source_order_inversions": int(np.sum(differences < 0)),
                     "minimum_adjacent_separation_eV": float(differences.min()) if differences.size else None,
                     "slot_dispersion_rms_eV": float(np.sqrt(np.mean((local - local.mean(axis=1, keepdims=True)) ** 2))),
                     "rank_dispersion_rms_eV": float(np.sqrt(np.mean((ranks - ranks.mean(axis=1, keepdims=True)) ** 2))),
                     "start": int(start), "stop": int(stop),
                     "slot_crossing_bands": int(crosses.sum()),
                     "n_strictly_below_ef": below.tolist(),
                     "n_at_or_below_ef": nonpositive.tolist(),
                     "relabeling_invariant_strict_crossing": bool(below.max() > nonpositive.min())})
    # Unique run IDs preserve the existing gap definition without joining a
    # repeated label on disconnected pieces of a path.
    runs = np.cumsum(np.r_[True, segments[1:] != segments[:-1]])
    values = values[:, valid_k]
    gap, _ = derive_gap(values[None], np.ones((1, len(values)), bool), efermi, runs[valid_k])
    low, high = values.min(axis=1), values.max(axis=1)
    straddling = (low < efermi) & (high > efermi)
    if crossing.any():
        state = "witnessed_within_segment_crossing"
    elif straddling.any():
        state = "unresolved_between_segments"
    elif np.isfinite(gap[0]):
        state = "resolved_positive_gap" if gap[0] > 0 else "resolved_touching_frontiers"
    elif np.all(values == efermi):
        state = "unresolved_ef_flat_only"
    elif not np.any(high <= efermi):
        state = "unresolved_missing_occupied_frontier"
    else:
        state = "unresolved_missing_empty_frontier"
    return {**result,
            "frontier_state": state,
            "gap_eV": float(gap[0]) if np.isfinite(gap[0]) else None,
            "interval_straddling_bands": int(straddling.sum()),
            "within_segment_crossing_bands": int(crossing.sum()),
            "causal_attribution": "not_established", "segments": rows}


def resampling_audit(raw_energies, selected_indices, bands, mask,
                     source_segment_ids, target_segment_ids, *, efermi=0.) -> dict:
    """Compare full raw, actual selected raw, and EF-relative resampled arrays.

    Source selection indices are unpadded, in the order of True mask slots.
    Rank inversions are counted within spin blocks, never across flattened spins.
    This ledger does not certify source semantics or explain model errors.
    """
    raw = np.asarray(raw_energies)
    indices = np.asarray(selected_indices)
    mask = np.asarray(mask)
    if raw.ndim not in (2, 3) or not raw.size:
        raise ValueError("raw selection energies must be nonempty 2D/3D")
    flat = raw.reshape(-1, raw.shape[-1])
    if (mask.ndim != 1 or mask.dtype.kind != "b" or indices.ndim != 1
            or (indices.size and indices.dtype.kind not in "iu")
            or len(indices) != int(mask.sum()) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(flat))):
        raise ValueError("selection indices must be unique valid integers matching real resampled slots")
    indices = indices.astype(np.int64)
    groups = np.repeat(np.arange(raw.shape[0]), raw.shape[1]) if raw.ndim == 3 else np.zeros(len(flat), int)
    selected_groups = np.zeros(len(mask), int)
    selected_groups[np.asarray(mask)] = groups[indices]
    source_ids = np.unique(source_segment_ids)
    target_ids = np.unique(target_segment_ids)
    source_ids, target_ids = source_ids[source_ids >= 0], target_ids[target_ids >= 0]
    if np.setdiff1d(target_ids, source_ids).size:
        raise ValueError("target segment IDs must reference source segments")
    return {
        "source_segment_count": int(len(source_ids)), "target_segment_count": int(len(target_ids)),
        "unrepresented_source_segment_ids": np.setdiff1d(source_ids, target_ids).tolist(),
        "raw_full": band_path_diagnostics(flat, np.ones(len(flat), bool), source_segment_ids,
                                           efermi=efermi, band_groups=groups),
        "raw_selected": band_path_diagnostics(flat[indices], np.ones(len(indices), bool), source_segment_ids,
                                               efermi=efermi, band_groups=groups[indices]),
        "resampled_selected": band_path_diagnostics(bands, mask, target_segment_ids,
                                                     band_groups=selected_groups),
        "causal_attribution": "not_established",
    }


def trajectory_matching_diagnostics(pred, target, mask, segment_ids, *, max_cost_elements=1_000_000) -> dict:
    """Single-material equal-element L1 assignments; no causal/identity claim.

    OT <= segment H <= global H <= slot for the same valid energies. A tiny
    H-minus-OT difference is not evidence that trajectory errors caused a gap
    error, or that changing the loss would repair it. Existing OT is unchanged.
    """
    from scipy.optimize import linear_sum_assignment

    if (isinstance(max_cost_elements, (bool, np.bool_))
            or not isinstance(max_cost_elements, (int, np.integer)) or max_cost_elements < 1):
        raise ValueError("max_cost_elements must be a positive integer")
    mask = np.asarray(mask)
    p, t = np.asarray(pred, dtype=np.float32), np.asarray(target, dtype=np.float32)
    diagnostic = band_path_diagnostics(t, mask, segment_ids)
    band_path_diagnostics(p, mask, segment_ids)
    valid_k = np.asarray(segment_ids) >= 0
    p, t = p[mask].astype(np.float64), t[mask].astype(np.float64)
    result = {"scope": "selected_slot_assignment_not_wavefunction_identity", "causal_attribution": "not_established",
              "per_k_ot_eV": None, "segment_hungarian_eV": None, "global_hungarian_eV": None, "slot_mae_eV": None}
    if not len(p) or not valid_k.any():
        return result
    if len(p) ** 2 * int(valid_k.sum()) > max_cost_elements:
        raise ValueError("bounded trajectory assignment exceeds max_cost_elements")

    def matched_sum(left, right):
        cost = np.abs(left[:, None, :] - right[None, :, :]).sum(axis=-1)
        row, column = linear_sum_assignment(cost)
        return float(cost[row, column].sum())

    segment_sum = sum(matched_sum(p[:, row["start"]:row["stop"]], t[:, row["start"]:row["stop"]])
                      for row in diagnostic["segments"])
    p, t = p[:, valid_k], t[:, valid_k]
    valid = np.ones((1, len(p)), bool)
    return {**result,
            "per_k_ot_eV": sorted_band_mae(p[None], t[None], valid),
            "segment_hungarian_eV": segment_sum / p.size,
            "global_hungarian_eV": matched_sum(p, t) / p.size,
            "slot_mae_eV": float(np.abs(p - t).mean())}


def physical_metric_eligibility(k_points, segment_ids, *, k_unit=None, source_evidence=None) -> dict:
    """Fail-closed capability audit, NOT a physical derivative implementation.

    Physical slopes/curvatures/masses remain disabled in legacy P3. A unit label
    cannot turn a normalized scalar into reciprocal geometry. No finite-difference
    proxy is silently reported as a physical observable.

    ``source_evidence`` is NOT trusted as a verdict. No approved pointwise
    energy/k/source-binding verifier or derivative implementation exists here;
    even a dense-looking 3D array therefore stays ineligible. This function is
    an executable refusal boundary, not a placeholder scientific validation.
    """
    k = np.asarray(k_points)
    reasons = []
    if k.ndim == 1:
        reasons.append("scalar_k_is_not_3d_reciprocal_geometry")
    segments = np.asarray(segment_ids)
    compatible = segments.ndim == 1 and k.ndim == 2 and k.shape == (segments.size, 3)
    if not compatible:
        reasons.append("k_point_count_or_dimension_mismatch")
    if k_unit != "1/angstrom":
        reasons.append("reciprocal_length_unit_missing_or_unsupported")
    if k.dtype.kind not in "fiu" or not np.isfinite(k).all():
        reasons.append("nonfinite_k_geometry")
    if segments.ndim != 1 or segments.dtype.kind not in "iu" or np.any(segments < 0):
        reasons.append("invalid_segment_ids")
    reasons.extend(["pointwise_energy_k_source_binding_not_verified", "physical_derivatives_not_implemented"])
    return {"eligible": False, "reasons": reasons,
            "geometry_shape_compatible": bool(compatible), "source_binding_verified": False,
            "metrics": {"slope": False, "curvature": False, "directional_effective_mass": False},
            "scope": "selected_slot_directional_not_full_mass_tensor"}


def p3_coverage_report(records) -> dict:
    """Recompute diagnostic coverage from arrays; never promote an experiment.

    Input rows contain material_id, bands, mask and segment_ids; optional k_points,
    k_unit, band_groups, efermi and source_status are diagnostic context only.
    Declared source statuses are counted separately, not verified from flags.
    Scientific source/3D/physics/fair-baseline/multiseed gates need independent
    evidence outside this report; supplied ``accepted`` flags are never consumed.
    """
    from collections import Counter

    rows, source_counts, seen = [], Counter(), set()
    for record in records:
        mid = record.get("material_id")
        if not isinstance(mid, str) or not mid or mid in seen:
            raise ValueError("material_id must be a nonempty unique string")
        seen.add(mid)
        source_counts[record.get("source_status", "unknown")] += 1
        row = {"material_id": record["material_id"], "diagnostic_eligible": False,
               "gap_resolved": False, "physical_metric_eligible": False, "invalid": False}
        try:
            diagnostic = band_path_diagnostics(record["bands"], record["mask"], record["segment_ids"],
                                                efermi=record.get("efermi", 0.),
                                                band_groups=record.get("band_groups"))
            physical = physical_metric_eligibility(record.get("k_points"), record["segment_ids"],
                                                   k_unit=record.get("k_unit"),
                                                   source_evidence=record.get("source_evidence"))
        except (ValueError, TypeError, KeyError) as exc:
            row.update(invalid=True, reason=str(exc))
        else:
            row.update(diagnostic_eligible=bool(diagnostic["n_valid_bands"] and diagnostic["n_valid_k"]),
                       frontier_state=diagnostic["frontier_state"], gap_resolved=diagnostic["gap_eV"] is not None,
                       physical_metric_eligible=physical["eligible"], physical_blockers=physical["reasons"])
        rows.append(row)
    resolved = sum(row["gap_resolved"] for row in rows)
    eligible = sum(row["diagnostic_eligible"] for row in rows)
    invalid = sum(row["invalid"] for row in rows)
    return {"status": "NO-GO", "accepted": False, "scope": "diagnostic_coverage_not_scientific_acceptance",
            "n_requested": len(rows), "n_diagnostic_eligible": eligible,
            "n_invalid": invalid, "n_empty": len(rows) - eligible - invalid,
            "diagnostic_coverage": eligible / len(rows) if rows else None,
            "n_gap_resolved": resolved, "n_gap_unknown": eligible - resolved,
            "gap_resolution_coverage": resolved / eligible if eligible else None,
            "n_physical_metric_eligible": sum(row["physical_metric_eligible"] for row in rows),
            "declared_source_status_counts": dict(source_counts), "records": rows,
            "unverified_gates": ["independent_source_verification", "pointwise_3d_energy_k_binding",
                                 "full_p3_physical_constraints", "same_data_split_fair_bandformer_baseline",
                                 "multiseed_scientific_comparison", "strict_numerical_reproducibility"]}


def gap_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, efermi: float = 0.0) -> float:
    """Gap MAE over insulators only (insulator/metal decided from TARGET)."""
    pred_gap, _ = derive_gap(pred, mask, efermi)
    true_gap, is_insulator = derive_gap(target, mask, efermi)
    if not is_insulator.any():
        return float("nan")
    return float(np.abs(pred_gap[is_insulator] - true_gap[is_insulator]).mean())
