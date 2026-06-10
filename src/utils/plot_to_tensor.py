"""
Plot-to-Physics utilities for extracting approximate band traces from images.

This module is a Phase 5 extension. It is intentionally decoupled from the core
download/OOD/SSL/fine-tune pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class PlotToTensorConfig:
    target_k_points: int = 128
    image_height_ev: float = 8.0
    fermi_y_fraction: float = 0.5
    threshold: int = 180


def _load_grayscale(image_path: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for plot_to_tensor.py") from exc

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    return image


def extract_skeleton_coordinates(image_path: str, config: PlotToTensorConfig | None = None) -> np.ndarray:
    """Extract a rough line skeleton from a band-structure plot image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for plot_to_tensor.py") from exc

    cfg = config or PlotToTensorConfig()
    gray = _load_grayscale(image_path)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY_INV)[1]

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skeleton = cv2.ximgproc.thinning(binary)
    else:
        skeleton = binary

    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        raise ValueError("No band-like pixels were detected")
    return np.stack([xs, ys], axis=1).astype(np.float32)


def plot_image_to_tensor(
    image_path: str,
    config: PlotToTensorConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an image skeleton into a 128-point normalized k/energy trace."""
    cfg = config or PlotToTensorConfig()
    coords = extract_skeleton_coordinates(image_path, cfg)
    xs = coords[:, 0]
    ys = coords[:, 1]

    width = max(float(xs.max() - xs.min()), 1.0)
    height = max(float(ys.max() - ys.min()), 1.0)
    k = (xs - xs.min()) / width

    y_center = ys.min() + cfg.fermi_y_fraction * height
    energy = -(ys - y_center) / height * cfg.image_height_ev

    order = np.argsort(k)
    k_sorted = k[order]
    e_sorted = energy[order]

    unique_k, inverse = np.unique(k_sorted, return_inverse=True)
    mean_e = np.zeros_like(unique_k, dtype=np.float32)
    counts = np.zeros_like(unique_k, dtype=np.float32)
    np.add.at(mean_e, inverse, e_sorted)
    np.add.at(counts, inverse, 1.0)
    mean_e = mean_e / np.maximum(counts, 1.0)

    if len(unique_k) < 2:
        raise ValueError("Need at least two distinct k positions after skeleton extraction")

    k_new = np.linspace(0.0, 1.0, cfg.target_k_points, dtype=np.float32)
    energy_new = interp1d(unique_k, mean_e, bounds_error=False, fill_value="extrapolate")(k_new)
    curvature = np.gradient(np.gradient(energy_new)).astype(np.float32)
    tensor = np.stack([energy_new.astype(np.float32), curvature], axis=-1)
    return k_new, tensor
