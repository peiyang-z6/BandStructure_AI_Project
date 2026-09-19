"""
Physics-aware dataset utilities for SSL training.

The active SSL path uses virtual strain augmentation instead of unstructured
Gaussian noise. A small random strain is applied in k-space, then each feature
sequence is re-interpolated onto the canonical k grid.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


MBM_FEATURE_ORDER = (
    "VBM_E", "VBM_curv", "VBM_k_dist", "CBM_E", "CBM_curv", "CBM_k_dist",
)


def normalize_mbm_inputs(band_inputs, norm_stats):
    """Pure raw -> MBM transform using the frozen training statistics, never fit.

    Accept raw (N,2,K,3) or canonical raw (N,K,6); return float32 (N,K,6).
    Callers must check cache input_space before calling: already normalized
    caches are NOT inputs to this function (numerical values cannot reveal that).
    """
    x = np.asarray(band_inputs)
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("band inputs must contain finite real numbers")
    if x.ndim == 4 and x.shape[1] == 2 and x.shape[3] == 3:
        x = x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], 6)
    elif x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError("expected raw (N,2,K,3) or canonical (N,K,6)")
    if x.shape[1] == 0:
        raise ValueError("empty k sequence")
    try:
        mean = np.asarray(norm_stats["mean"], dtype=np.float32)
        std = np.asarray(norm_stats["std"], dtype=np.float32)
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen mean/std are required") from exc
    if (mean.shape not in ((6,), (1, 1, 6)) or std.shape not in ((6,), (1, 1, 6))
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("mean/std must be finite six-channel statistics; std > 0")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (x.astype(np.float32) - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)
    if not np.isfinite(result).all():
        raise ValueError("normalization overflow")
    return result


class VirtualStrainAugmentation:
    """Apply a small virtual lattice strain to normalized 1D k-path sequences."""

    def __init__(self, strain_scale: float = 0.01):
        self.strain_scale = float(strain_scale)

    def __call__(self, sequence: tf.Tensor) -> tf.Tensor:
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        seq_len = tf.shape(sequence)[0]
        num_features = tf.shape(sequence)[1]

        k = tf.linspace(0.0, 1.0, seq_len)
        centered = k - 0.5

        theta = tf.random.uniform([], 0.0, 2.0 * 3.141592653589793, dtype=tf.float32)
        c = tf.cos(theta)
        s = tf.sin(theta)
        eps_x = tf.random.uniform([], -self.strain_scale, self.strain_scale, dtype=tf.float32)
        eps_y = tf.random.uniform([], -self.strain_scale, self.strain_scale, dtype=tf.float32)

        # Project a 2D orthogonally-rotated strain tensor back onto the 1D path.
        projected_scale = c * c * (1.0 + eps_x) + s * s * (1.0 + eps_y)
        strained_k = tf.clip_by_value(0.5 + centered * projected_scale, 0.0, 1.0)

        source = strained_k * tf.cast(seq_len - 1, tf.float32)
        left = tf.cast(tf.floor(source), tf.int32)
        right = tf.minimum(left + 1, seq_len - 1)
        weight = tf.reshape(source - tf.cast(left, tf.float32), [-1, 1])

        left_values = tf.gather(sequence, left)
        right_values = tf.gather(sequence, right)
        augmented = left_values * (1.0 - weight) + right_values * weight
        return tf.reshape(augmented, [seq_len, num_features])


SSLAugmentation = VirtualStrainAugmentation
