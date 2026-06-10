"""
Physics validation utilities for reconstructed or predicted band tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _local_curvature(band: np.ndarray) -> np.ndarray:
    """Local quadratic curvature via windowed polyfit (±2 pts)."""
    band = np.asarray(band, dtype=np.float64)
    n = len(band)
    curv = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        x_local = np.arange(lo, hi, dtype=np.float64) - float(i)
        y_local = band[lo:hi]
        if len(x_local) < 3:
            if i > 0 and i < n - 1:
                curv[i] = band[i + 1] - 2.0 * band[i] + band[i - 1]
            continue
        coeff = np.polyfit(x_local, y_local, deg=2)
        curv[i] = 2.0 * coeff[0]
    return curv.astype(np.float32)


@dataclass
class PhysicsValidator:
    """Score basic physical consistency of VBM/CBM tensors."""

    violation_tolerance: float = 0.05

    def validate(self, predicted_gap: np.ndarray, band_tensors: np.ndarray) -> Dict[str, float]:
        predicted_gap = np.asarray(predicted_gap, dtype=np.float32).reshape(-1)
        tensors = np.asarray(band_tensors, dtype=np.float32)

        if tensors.ndim != 4 or tensors.shape[1] != 2:
            raise ValueError(f"Expected band_tensors with shape (N, 2, K, C), got {tensors.shape}")

        vbm_energy = tensors[:, 0, :, 0]
        cbm_energy = tensors[:, 1, :, 0]
        vbm_center = np.argmax(vbm_energy, axis=1)
        cbm_center = np.argmin(cbm_energy, axis=1)

        vbm_curvature = np.asarray([_local_curvature(row) for row in vbm_energy])
        cbm_curvature = np.asarray([_local_curvature(row) for row in cbm_energy])
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

    def physics_score(self, predicted_gap: np.ndarray, band_tensors: np.ndarray) -> float:
        return self.validate(predicted_gap, band_tensors)["physics_score"]
