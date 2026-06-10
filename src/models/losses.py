"""
Physics-aware loss functions for band-structure SSL and fine-tuning.

Masked Band Modeling objective:

    L_total = L_mse + w_c * L_curvature + w_s * L_symmetry

where

    L_mse = mean_{masked k, features} ||E_hat(k) - E(k)||^2

    L_curvature =
        relu( <d2 E_vbm / dk2>_soft-vbm )
        + relu( -<d2 E_cbm / dk2>_soft-cbm )

The soft extrema are differentiable windows:

    p_vbm(k) = softmax(E_vbm(k) / tau)
    p_cbm(k) = softmax(-E_cbm(k) / tau)
    <d2E/dk2> = sum_k p(k) * d2E(k)/dk2

Thus VBM is penalized when curvature is positive and CBM is penalized when
curvature is negative. Symmetry/consistency loss checks that explicit curvature
channels agree with finite-difference curvature from energy channels.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="BandStructureAI")
class BandCurvatureLoss(keras.losses.Loss):
    """Differentiable curvature-sign loss for VBM/CBM extrema."""

    def __init__(self, temperature: float = 0.05, name: str = "band_curvature_loss"):
        super().__init__(name=name)
        self.temperature = float(temperature)

    def second_derivative(self, energy: tf.Tensor) -> tf.Tensor:
        second = energy[:, 2:] - 2.0 * energy[:, 1:-1] + energy[:, :-2]
        return tf.pad(second, [[0, 0], [1, 1]])

    def call(self, y_true, y_pred):
        del y_true
        features_per_band = tf.shape(y_pred)[-1] // 2
        vbm_energy = y_pred[:, :, 0]
        cbm_energy = y_pred[:, :, features_per_band]

        vbm_curv = self.second_derivative(vbm_energy)
        cbm_curv = self.second_derivative(cbm_energy)

        tau = tf.cast(tf.maximum(self.temperature, 1e-6), y_pred.dtype)
        vbm_weights = tf.nn.softmax(vbm_energy / tau, axis=1)
        cbm_weights = tf.nn.softmax(-cbm_energy / tau, axis=1)

        vbm_soft_curv = tf.reduce_sum(vbm_weights * vbm_curv, axis=1)
        cbm_soft_curv = tf.reduce_sum(cbm_weights * cbm_curv, axis=1)

        return tf.reduce_mean(tf.nn.relu(vbm_soft_curv)) + tf.reduce_mean(
            tf.nn.relu(-cbm_soft_curv)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


@keras.utils.register_keras_serializable(package="BandStructureAI")
class CurvatureConsistencyLoss(keras.losses.Loss):
    """Match explicit curvature channels to finite-difference energy curvature."""

    def __init__(self, name: str = "curvature_consistency_loss"):
        super().__init__(name=name)

    def second_derivative(self, energy: tf.Tensor) -> tf.Tensor:
        second = energy[:, 2:] - 2.0 * energy[:, 1:-1] + energy[:, :-2]
        return tf.pad(second, [[0, 0], [1, 1]])

    def call(self, y_true, y_pred):
        del y_true
        features_per_band = tf.shape(y_pred)[-1] // 2
        vbm_energy = y_pred[:, :, 0]
        vbm_curv_channel = y_pred[:, :, 1]
        cbm_energy = y_pred[:, :, features_per_band]
        cbm_curv_channel = y_pred[:, :, features_per_band + 1]

        return 0.5 * (
            tf.reduce_mean(tf.square(vbm_curv_channel - self.second_derivative(vbm_energy)))
            + tf.reduce_mean(tf.square(cbm_curv_channel - self.second_derivative(cbm_energy)))
        )


@keras.utils.register_keras_serializable(package="BandStructureAI")
class FocalLoss(keras.losses.Loss):
    """Multiclass softmax focal loss.

    L = sum_c alpha * class_weight_c * (1 - p_c)^gamma * [-y_c log(p_c)]
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        class_weights: list[float] | None = None,
        name: str = "focal_loss",
    ):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor = tf.pow(1.0 - y_pred, self.gamma)
        loss = self.alpha * focal_factor * cross_entropy
        if self.class_weights is not None:
            weights = tf.constant(self.class_weights, dtype=y_pred.dtype)
            loss = loss * tf.reshape(weights, [1, -1])
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "class_weights": self.class_weights,
            }
        )
        return config
