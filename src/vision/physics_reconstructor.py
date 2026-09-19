"""Reconstruct 6D physical tensors from parsed band-plot geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from .multi_format_parser import ParsedBandData


@dataclass
class ReconstructedTensor:
    """Model-ready tensor reconstructed from a figure."""

    raw_tensor: np.ndarray
    flat_tensor: np.ndarray
    k_axis: np.ndarray
    vbm_energy: np.ndarray
    cbm_energy: np.ndarray
    vbm_curvature: np.ndarray
    cbm_curvature: np.ndarray
    vbm_index: int
    cbm_index: int
    tensor_gap: float
    is_metallic: bool
    kpath_labels: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicsReconstructor:
    """Convert parsed `(k, E)` traces into the project's 6D tensor contract."""

    def __init__(
        self,
        target_k_points: int = 128,
        metal_threshold_ev: float = 0.03,
        conduction_guard_ev: float = 0.8,
    ) -> None:
        self.target_k_points = int(target_k_points)
        self.metal_threshold_ev = float(metal_threshold_ev)
        self.conduction_guard_ev = float(conduction_guard_ev)

    def reconstruct(self, parsed: ParsedBandData) -> ReconstructedTensor:
        k = np.asarray(parsed.k, dtype=np.float32)
        energy = np.asarray(parsed.energy, dtype=np.float32) - float(parsed.fermi_energy)
        finite = np.isfinite(k) & np.isfinite(energy)
        k = np.clip(k[finite], 0.0, 1.0)
        energy = energy[finite]
        if len(k) < 8:
            raise ValueError("Need at least 8 finite extracted points for 6D tensor reconstruction")

        vb_k, vb_e, cb_k, cb_e = self._split_valence_conduction(k, energy)
        k_axis = np.linspace(0.0, 1.0, self.target_k_points, dtype=np.float32)
        vbm_energy = self._resample_band(vb_k, vb_e, k_axis, mode="valence")
        cbm_energy = self._resample_band(cb_k, cb_e, k_axis, mode="conduction")

        vbm_idx = int(np.argmax(vbm_energy))
        cbm_idx = int(np.argmin(cbm_energy))
        tensor_gap = float(cbm_energy[cbm_idx] - vbm_energy[vbm_idx])
        is_metallic = bool(
            tensor_gap <= self.metal_threshold_ev
            or self._fermi_crosses_band(vbm_energy, cbm_energy)
        )

        vbm_curv = self._local_quadratic_curvature(k_axis, vbm_energy)
        cbm_curv = self._local_quadratic_curvature(k_axis, cbm_energy)
        vbm_k_dist = self._normalized_k_distance(vbm_idx)
        cbm_k_dist = self._normalized_k_distance(cbm_idx)

        raw_tensor = np.stack(
            [
                np.stack([vbm_energy, vbm_curv, vbm_k_dist], axis=-1),
                np.stack([cbm_energy, cbm_curv, cbm_k_dist], axis=-1),
            ],
            axis=0,
        )[None, ...].astype(np.float32)
        flat_tensor = (
            np.transpose(raw_tensor, (0, 2, 1, 3))
            .reshape(1, self.target_k_points, 6)
            .astype(np.float32)
        )

        return ReconstructedTensor(
            raw_tensor=raw_tensor,
            flat_tensor=flat_tensor,
            k_axis=k_axis,
            vbm_energy=vbm_energy.astype(np.float32),
            cbm_energy=cbm_energy.astype(np.float32),
            vbm_curvature=vbm_curv.astype(np.float32),
            cbm_curvature=cbm_curv.astype(np.float32),
            vbm_index=vbm_idx,
            cbm_index=cbm_idx,
            tensor_gap=tensor_gap,
            is_metallic=is_metallic,
            kpath_labels=parsed.kpath_labels,
            metadata={
                "source_path": parsed.source_path,
                "source_type": parsed.source_type,
                "fermi_energy": parsed.fermi_energy,
                "vbm_k": float(k_axis[vbm_idx]),
                "cbm_k": float(k_axis[cbm_idx]),
                **parsed.metadata,
            },
        )

    def reconstruct_from_manual(
        self,
        source_path: str,
        panel_bbox: List[float],
        y_calibration: List[Dict[str, float]],
        x_calibration: List[Dict[str, float]],
        vbm_pixel: Dict[str, float],
        cbm_pixel: Dict[str, float],
        valence_points: List[Dict[str, float]],
        conduction_points: List[Dict[str, float]],
        fermi_y_pixel: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> ReconstructedTensor:
        """Build the 6D tensor from human-calibrated GUI points.

        CV provides only the plot panel and skeleton. Human experts provide the
        physical axes, VBM/CBM extrema, and curve keypoints. This method is the
        trusted Plot-to-Physics path for real paper figures.
        """

        if len(y_calibration) < 2:
            raise ValueError("Need two Y-axis calibration points with eV values")
        if len(x_calibration) < 2:
            raise ValueError("Need two X-axis calibration points with normalized k values")
        if not vbm_pixel or not cbm_pixel:
            raise ValueError("Need human-selected VBM and CBM pixels")
        if len(valence_points) + 1 < 3:
            raise ValueError("Need at least two valence curve keypoints plus VBM")
        if len(conduction_points) + 1 < 3:
            raise ValueError("Need at least two conduction curve keypoints plus CBM")

        y0, y1 = y_calibration[0], y_calibration[1]
        x0, x1 = x_calibration[0], x_calibration[1]
        delta_y_pixel = float(y1["y"]) - float(y0["y"])
        delta_x_pixel = float(x1["x"]) - float(x0["x"])
        delta_y_value = float(y1["value"]) - float(y0["value"])
        delta_x_value = float(x1["value"]) - float(x0["value"])
        if abs(delta_y_pixel) < 1.0e-6 or abs(delta_y_value) < 1.0e-12:
            raise ValueError("Y-axis calibration points and energy values must be distinct")
        if abs(delta_x_pixel) < 1.0e-6 or abs(delta_x_value) < 1.0e-12:
            raise ValueError("X-axis calibration points and k values must be distinct")
        y_scale = delta_y_value / delta_y_pixel
        x_scale = delta_x_value / delta_x_pixel
        if fermi_y_pixel is None:
            fermi_y_pixel = float(y0["y"]) + (0.0 - float(y0["value"])) / y_scale
        raw_fermi_energy = float(y0["value"]) + (float(fermi_y_pixel) - float(y0["y"])) * y_scale

        def pixel_to_physics(point: Dict[str, float]) -> tuple[float, float]:
            k = float(x0["value"]) + (float(point["x"]) - float(x0["x"])) * x_scale
            raw_energy = float(y0["value"]) + (float(point["y"]) - float(y0["y"])) * y_scale
            energy = raw_energy - raw_fermi_energy
            return float(np.clip(k, 0.0, 1.0)), float(energy)

        vbm_k, vbm_e = pixel_to_physics(vbm_pixel)
        cbm_k, cbm_e = pixel_to_physics(cbm_pixel)
        vb_pairs = [pixel_to_physics(p) for p in valence_points] + [(vbm_k, vbm_e)]
        cb_pairs = [pixel_to_physics(p) for p in conduction_points] + [(cbm_k, cbm_e)]

        k_axis = np.linspace(0.0, 1.0, self.target_k_points, dtype=np.float32)
        vbm_energy = self._resample_manual_trace(vb_pairs, k_axis, mode="valence")
        cbm_energy = self._resample_manual_trace(cb_pairs, k_axis, mode="conduction")

        vbm_idx = int(np.argmin(np.abs(k_axis - vbm_k)))
        cbm_idx = int(np.argmin(np.abs(k_axis - cbm_k)))
        vbm_energy[vbm_idx] = max(float(vbm_energy[vbm_idx]), float(vbm_e))
        cbm_energy[cbm_idx] = min(float(cbm_energy[cbm_idx]), float(cbm_e))
        tensor_gap = float(cbm_energy[cbm_idx] - vbm_energy[vbm_idx])
        is_metallic = bool(tensor_gap <= self.metal_threshold_ev)

        vbm_curv = self._local_quadratic_curvature(k_axis, vbm_energy)
        cbm_curv = self._local_quadratic_curvature(k_axis, cbm_energy)
        vbm_k_dist = self._normalized_k_distance(vbm_idx)
        cbm_k_dist = self._normalized_k_distance(cbm_idx)
        raw_tensor = np.stack(
            [
                np.stack([vbm_energy, vbm_curv, vbm_k_dist], axis=-1),
                np.stack([cbm_energy, cbm_curv, cbm_k_dist], axis=-1),
            ],
            axis=0,
        )[None, ...].astype(np.float32)
        flat_tensor = (
            np.transpose(raw_tensor, (0, 2, 1, 3))
            .reshape(1, self.target_k_points, 6)
            .astype(np.float32)
        )

        meta = dict(metadata or {})
        meta.update(
            {
                "source_path": source_path,
                "source_type": "manual_gui_calibration",
                "panel_bbox": panel_bbox,
                "fermi_y_pixel": float(fermi_y_pixel),
                "raw_fermi_energy_ev": float(raw_fermi_energy),
                "energy_reference": "Fermi level shifted to 0 eV",
                "x_scale": float(x_scale),
                "y_scale_ev_per_pixel": float(y_scale),
                "vbm_k": float(k_axis[vbm_idx]),
                "cbm_k": float(k_axis[cbm_idx]),
                "human_in_the_loop": True,
            }
        )
        return ReconstructedTensor(
            raw_tensor=raw_tensor,
            flat_tensor=flat_tensor,
            k_axis=k_axis,
            vbm_energy=vbm_energy.astype(np.float32),
            cbm_energy=cbm_energy.astype(np.float32),
            vbm_curvature=vbm_curv.astype(np.float32),
            cbm_curvature=cbm_curv.astype(np.float32),
            vbm_index=vbm_idx,
            cbm_index=cbm_idx,
            tensor_gap=tensor_gap,
            is_metallic=is_metallic,
            kpath_labels=[],
            metadata=meta,
        )

    def _resample_manual_trace(
        self, pairs: List[tuple[float, float]], k_axis: np.ndarray, mode: str
    ) -> np.ndarray:
        arr = np.asarray(pairs, dtype=np.float32)
        arr = arr[np.isfinite(arr).all(axis=1)]
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        unique_k = []
        unique_e = []
        for k_val in np.unique(arr[:, 0]):
            local = arr[np.isclose(arr[:, 0], k_val)]
            unique_k.append(float(k_val))
            unique_e.append(
                float(np.max(local[:, 1]) if mode == "valence" else np.min(local[:, 1]))
            )
        k = np.asarray(unique_k, dtype=np.float32)
        e = np.asarray(unique_e, dtype=np.float32)
        if len(k) < 2:
            raise ValueError(f"Need at least two unique {mode} k-points")
        if k[0] > 0.0:
            k = np.insert(k, 0, 0.0)
            e = np.insert(e, 0, e[0])
        if k[-1] < 1.0:
            k = np.append(k, 1.0)
            e = np.append(e, e[-1])
        if len(k) >= 4:
            interpolator = PchipInterpolator(k, e, extrapolate=True)
            values = interpolator(k_axis)
        else:
            values = np.interp(k_axis, k, e)
        return np.nan_to_num(values, nan=float(np.nanmedian(e))).astype(np.float32)

    def _split_valence_conduction(
        self, k: np.ndarray, energy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vb_mask = energy <= 0.0
        cb_mask = energy >= 0.0
        guarded_cb_mask = energy >= self.conduction_guard_ev
        if guarded_cb_mask.sum() >= 4:
            cb_mask = guarded_cb_mask

        if vb_mask.sum() < 4 or cb_mask.sum() < 4:
            median = float(np.median(energy))
            vb_mask = energy <= median
            cb_mask = energy >= median

        vb_k, vb_e = self._aggregate_extreme_by_k(k[vb_mask], energy[vb_mask], mode="valence")
        cb_k, cb_e = self._aggregate_extreme_by_k(k[cb_mask], energy[cb_mask], mode="conduction")
        if len(vb_k) < 2 or len(cb_k) < 2:
            raise ValueError("Could not separate enough valence/conduction points around E_F")
        return vb_k, vb_e, cb_k, cb_e

    def _aggregate_extreme_by_k(
        self, k: np.ndarray, energy: np.ndarray, mode: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        bins = np.linspace(0.0, 1.0, min(self.target_k_points, max(16, len(k) // 3)) + 1)
        inds = np.clip(np.digitize(k, bins) - 1, 0, len(bins) - 2)
        out_k: List[float] = []
        out_e: List[float] = []
        for idx in np.unique(inds):
            mask = inds == idx
            if not np.any(mask):
                continue
            local_e = energy[mask]
            select = int(np.argmax(local_e) if mode == "valence" else np.argmin(local_e))
            local_k = k[mask]
            out_k.append(float(local_k[select]))
            out_e.append(float(local_e[select]))
        order = np.argsort(out_k)
        return np.asarray(out_k, dtype=np.float32)[order], np.asarray(out_e, dtype=np.float32)[
            order
        ]

    def _resample_band(
        self, k: np.ndarray, energy: np.ndarray, k_axis: np.ndarray, mode: str
    ) -> np.ndarray:
        k_unique, inv = np.unique(k, return_inverse=True)
        if len(k_unique) == 1:
            return np.full_like(k_axis, float(energy[0]), dtype=np.float32)

        reduced = np.zeros_like(k_unique, dtype=np.float32)
        for idx in range(len(k_unique)):
            local = energy[inv == idx]
            reduced[idx] = np.max(local) if mode == "valence" else np.min(local)

        order = np.argsort(k_unique)
        k_unique = k_unique[order]
        reduced = reduced[order]
        if len(k_unique) >= 4:
            spline = PchipInterpolator(k_unique, reduced, extrapolate=False)
            k_eval = np.clip(k_axis, float(k_unique[0]), float(k_unique[-1]))
            values = spline(k_eval)
        else:
            values = interp1d(
                k_unique,
                reduced,
                bounds_error=False,
                fill_value=(float(reduced[0]), float(reduced[-1])),
            )(k_axis)
        values = np.nan_to_num(
            values,
            nan=float(np.nanmedian(reduced)),
            posinf=float(np.nanmax(reduced)),
            neginf=float(np.nanmin(reduced)),
        )
        values = np.clip(values, -self._vision_energy_clip(), self._vision_energy_clip())
        return self._smooth_resampled_trace(values.astype(np.float32))

    def _smooth_resampled_trace(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        if len(values) < window:
            return values.astype(np.float32)
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        med = np.asarray(
            [np.median(padded[i : i + window]) for i in range(len(values))], dtype=np.float32
        )
        kernel = np.ones(window, dtype=np.float32) / float(window)
        avg = np.convolve(np.pad(med, (pad, pad), mode="edge"), kernel, mode="valid")
        return avg.astype(np.float32)

    def _vision_energy_clip(self) -> float:
        return 60.0

    def _local_quadratic_curvature(
        self, k_axis: np.ndarray, energy: np.ndarray, half_window: int = 2
    ) -> np.ndarray:
        curv = np.zeros_like(energy, dtype=np.float32)
        n = len(energy)
        for idx in range(n):
            start = max(0, idx - half_window)
            end = min(n, idx + half_window + 1)
            if end - start < 3:
                if start == 0:
                    end = min(n, 3)
                else:
                    start = max(0, n - 3)
            local_k = k_axis[start:end] - k_axis[idx]
            local_e = energy[start:end]
            if len(local_k) < 3 or np.allclose(local_k, local_k[0]):
                curv[idx] = 0.0
                continue
            coeff = np.polyfit(local_k.astype(np.float64), local_e.astype(np.float64), deg=2)
            curv[idx] = float(2.0 * coeff[0])
        return curv

    def _normalized_k_distance(self, extremum_idx: int) -> np.ndarray:
        denom = max(self.target_k_points - 1, 1)
        idx = np.arange(self.target_k_points, dtype=np.float32)
        return ((idx - float(extremum_idx)) / float(denom)).astype(np.float32)

    def _fermi_crosses_band(self, vbm_energy: np.ndarray, cbm_energy: np.ndarray) -> bool:
        guard = max(self.conduction_guard_ev, self.metal_threshold_ev)
        return bool(np.max(vbm_energy) > guard or np.min(cbm_energy) < -guard)
