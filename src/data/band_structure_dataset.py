"""
Physics-aware dataset utilities for SSL training.

The active SSL path uses virtual strain augmentation instead of unstructured
Gaussian noise. A small random strain is applied in k-space, then each feature
sequence is re-interpolated onto the canonical k grid.
"""
from __future__ import annotations

import tensorflow as tf


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
