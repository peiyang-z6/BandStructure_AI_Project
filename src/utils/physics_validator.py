"""
Physics validation utilities for reconstructed or predicted band tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def local_curvature(
    band: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Local quadratic curvature via ±2 windows confined to one k-path segment."""
    band = np.asarray(band, dtype=np.float64)
    n = len(band)
    if segment_ids is None:
        segments = np.zeros(n, dtype=np.int32)
    else:
        segments = np.asarray(segment_ids, dtype=np.int32)
        if segments.shape != (n,):
            raise ValueError(
                f"Expected segment_ids shape {(n,)}, got {segments.shape}"
            )
    curv = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        indices = np.arange(lo, hi, dtype=np.int32)
        indices = indices[segments[indices] == segments[i]]
        x_local = indices.astype(np.float64) - float(i)
        y_local = band[indices]
        if len(indices) >= 3:
            coeff = np.polyfit(x_local, y_local, deg=2)
            curv[i] = 2.0 * coeff[0]
        elif (
            0 < i < n - 1
            and segments[i - 1] == segments[i] == segments[i + 1]
        ):
            curv[i] = band[i + 1] - 2.0 * band[i] + band[i - 1]
    return curv.astype(np.float32)


def _local_curvature(band: np.ndarray) -> np.ndarray:
    """Backward-compatible unsegmented wrapper."""
    return local_curvature(band)


@dataclass
class PhysicsValidator:
    """Score basic physical consistency of VBM/CBM tensors."""

    violation_tolerance: float = 0.05

    def validate(
        self,
        predicted_gap: np.ndarray,
        band_tensors: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> Dict[str, float]:
        predicted_gap = np.asarray(predicted_gap, dtype=np.float32).reshape(-1)
        tensors = np.asarray(band_tensors, dtype=np.float32)

        if tensors.ndim != 4 or tensors.shape[1] != 2:
            raise ValueError(f"Expected band_tensors with shape (N, 2, K, C), got {tensors.shape}")
        if segment_ids is None:
            segments = np.zeros((len(tensors), tensors.shape[2]), dtype=np.int32)
        else:
            segments = np.asarray(segment_ids, dtype=np.int32)
            expected = (len(tensors), tensors.shape[2])
            if segments.shape != expected:
                raise ValueError(
                    f"Expected segment_ids with shape {expected}, got {segments.shape}"
                )

        vbm_energy = tensors[:, 0, :, 0]
        cbm_energy = tensors[:, 1, :, 0]
        vbm_center = np.argmax(vbm_energy, axis=1)
        cbm_center = np.argmin(cbm_energy, axis=1)

        vbm_curvature = np.asarray(
            [
                local_curvature(row, segment_ids=segments[index])
                for index, row in enumerate(vbm_energy)
            ]
        )
        cbm_curvature = np.asarray(
            [
                local_curvature(row, segment_ids=segments[index])
                for index, row in enumerate(cbm_energy)
            ]
        )
        vbm_at_extreme = np.asarray([vbm_curvature[i, vbm_center[i]] for i in range(len(tensors))])
        cbm_at_extreme = np.asarray([cbm_curvature[i, cbm_center[i]] for i in range(len(tensors))])

        tensor_gap = np.min(cbm_energy, axis=1) - np.max(vbm_energy, axis=1)
        residual = predicted_gap - tensor_gap

        violations = {
            "negative_predicted_gap_rate": float(np.mean(predicted_gap < 0.0)),
            "vbm_positive_curvature_rate": float(np.mean(vbm_at_extreme > 0.0)),
            "cbm_negative_curvature_rate": float(np.mean(cbm_at_extreme < 0.0)),
            "gap_identity_large_residual_rate": float(
                np.mean(np.abs(residual) > max(self.violation_tolerance, 1e-6))
            ),
        }
        violation_rate = float(max(violations.values()))
        return {
            **violations,
            "gap_identity_mae": float(np.mean(np.abs(residual))),
            "physics_violation_rate": violation_rate,
            "physics_score": float(max(0.0, 1.0 - violation_rate)),
        }

    def physics_score(
        self,
        predicted_gap: np.ndarray,
        band_tensors: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> float:
        return self.validate(
            predicted_gap,
            band_tensors,
            segment_ids=segment_ids,
        )["physics_score"]
