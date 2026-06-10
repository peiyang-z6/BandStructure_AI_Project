"""
Supervised fine-tuning and physics validation.

Uses the MBM-pretrained SSLEncoder on OOD tensors to predict:
- band gap value (regression)
- gap type (metal/direct/indirect classification)

Input tensor source:
    data_cache/ood_tensors/band_tensors_ood_split.npz

The script preserves the existing OOD split. It does not create any random
sample-wise split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, f1_score, recall_score, roc_curve

import src.models.band_structure_encoder  # Registers custom Keras classes.
from src.engine.finetune_trainer import (
    freeze_encoder_layers as apply_freeze_encoder_layers,
    write_finetune_strategy_report,
)
from src.utils.physics_validator import PhysicsValidator
from src.utils.mc_dropout import mc_dropout_predict_with_type, mc_calibration_check
from src.utils.visualizer import save_band_overlay_grid
from src.data.band_structure_dataset import VirtualStrainAugmentation


def _local_curvature(band: np.ndarray) -> np.ndarray:
    """Local quadratic curvature via windowed polyfit (±2 points)."""
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


def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
    return mean, np.maximum(std, 1e-6)


def flatten_tensor(X: np.ndarray) -> np.ndarray:
    # (N, 2, K, C) -> (N, K, 2*C)
    return np.transpose(X, (0, 2, 1, 3)).reshape(X.shape[0], X.shape[2], -1).astype(np.float32)


def infer_gap_type_labels(X_raw: np.ndarray, y_gap: np.ndarray, metal_threshold: float = 0.01) -> np.ndarray:
    """Infer metal/direct/indirect from VBM/CBM extrema in processed tensors."""
    labels = []
    for sample, gap in zip(X_raw, y_gap):
        if gap < metal_threshold:
            labels.append(0)
            continue

        vbm_energy = sample[0, :, 0]
        cbm_energy = sample[1, :, 0]
        vbm_k = int(np.argmax(vbm_energy))
        cbm_k = int(np.argmin(cbm_energy))
        labels.append(1 if vbm_k == cbm_k else 2)
    return np.asarray(labels, dtype=np.int32)


def tensor_gap_from_extrema(X_raw: np.ndarray) -> np.ndarray:
    """Compute E_CBM - E_VBM from processed VBM/CBM channels."""
    vbm_max = np.max(X_raw[:, 0, :, 0], axis=1)
    cbm_min = np.min(X_raw[:, 1, :, 0], axis=1)
    return (cbm_min - vbm_max).astype(np.float32)


def load_dataset(npz_path: str, norm_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    mean, std = load_norm_stats(norm_path)

    X_train_raw = data["X_train"].astype(np.float32)
    X_test_raw = data["X_test"].astype(np.float32)
    X_train = (flatten_tensor(X_train_raw) - mean) / std
    X_test = (flatten_tensor(X_test_raw) - mean) / std

    y_train = data["y_train"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train,
        "X_test": X_test,
        "feature_mean": mean.reshape(-1).astype(np.float32),
        "feature_std": std.reshape(-1).astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        "type_train": infer_gap_type_labels(X_train_raw, y_train),
        "type_test": infer_gap_type_labels(X_test_raw, y_test),
        "tensor_gap_train": tensor_gap_from_extrema(X_train_raw),
        "tensor_gap_test": tensor_gap_from_extrema(X_test_raw),
        "groups_train": data["groups_train"].astype(np.int32),
        "groups_test": data["groups_test"].astype(np.int32),
        "material_ids_train": data["material_ids_train"],
        "material_ids_test": data["material_ids_test"],
    }


def load_kpath_labels(npz_path: str) -> Dict[str, object]:
    manifest_path = os.path.join(os.path.dirname(npz_path), "ood_split_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    labels = {}
    for sample in manifest.get("samples", []):
        material_id = sample.get("material_id")
        if material_id:
            labels[str(material_id)] = sample.get("kpath_labels", [])
    return labels


def compute_class_weights(y_type: np.ndarray, num_classes: int = 3) -> Dict[int, float]:
    counts = np.bincount(y_type.astype(np.int32), minlength=num_classes).astype(np.float32)
    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    weights[present] = float(np.sum(counts)) / (float(np.sum(present)) * counts[present])
    return {idx: float(weights[idx]) for idx in range(num_classes)}


def make_tf_dataset(
    X: np.ndarray,
    y_gap: np.ndarray,
    y_type: np.ndarray,
    batch_size: int,
    shuffle: bool,
    class_weights: Dict[int, float] | None = None,
    augment: bool = False,
    strain_scale: float = 0.01,
) -> tf.data.Dataset:
    labels = {
        "gap": y_gap.reshape(-1, 1).astype(np.float32),
        "type": tf.one_hot(y_type, depth=3).numpy().astype(np.float32),
    }
    if class_weights is None:
        ds = tf.data.Dataset.from_tensor_slices((X, labels))
    else:
        type_weights = np.asarray([class_weights[int(label)] for label in y_type], dtype=np.float32)
        sample_weights = {
            "gap": np.ones(len(y_type), dtype=np.float32),
            "type": type_weights,
        }
        ds = tf.data.Dataset.from_tensor_slices((X, labels, sample_weights))
    if augment:
        aug = VirtualStrainAugmentation(strain_scale=strain_scale)
        def _augment(*args):
            # args[0] is the X tensor (seq_len, channels)
            augmented_x = aug(args[0])
            return (augmented_x,) + args[1:]
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class ExtremumExpectedGapHead(layers.Layer):
    """Predict line-mode gap from sequence-local VBM/CBM extremum probabilities.

    The input sequence is standardized for the encoder, but this head converts
    the VBM_E and CBM_E channels back to eV before computing the expected
    E_CBM - E_VBM. Therefore the regression loss and reported MAE stay in the
    physical eV scale.
    """

    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_mean = tf.constant(np.asarray(feature_mean, dtype=np.float32), dtype=tf.float32)
        self.feature_std = tf.constant(np.asarray(feature_std, dtype=np.float32), dtype=tf.float32)
        self.temperature_raw = self.add_weight(
            name="temperature_raw",
            shape=(),
            initializer=keras.initializers.Constant(-1.4916549),
            trainable=True,
        )
        self.local_conv = layers.Conv1D(hidden_dim, kernel_size=3, padding="same", activation="gelu")
        self.dropout = layers.Dropout(dropout)
        self.logit_layer = layers.Conv1D(2, kernel_size=1, padding="same", name="extremum_logits")

    def call(self, encoded, x_norm, training=False):
        local_features = tf.concat([encoded, x_norm], axis=-1)
        hidden = self.local_conv(local_features)
        hidden = self.dropout(hidden, training=training)
        logits = self.logit_layer(hidden)
        temperature = 0.01 + 0.49 * tf.nn.sigmoid(self.temperature_raw)

        vbm_prob = tf.nn.softmax(logits[:, :, 0] / temperature, axis=1)
        cbm_prob = tf.nn.softmax(logits[:, :, 1] / temperature, axis=1)
        seq_depth = tf.shape(x_norm)[1]
        vbm_hard = tf.one_hot(tf.argmax(vbm_prob, axis=1), depth=seq_depth, dtype=vbm_prob.dtype)
        cbm_hard = tf.one_hot(tf.argmax(cbm_prob, axis=1), depth=seq_depth, dtype=cbm_prob.dtype)
        vbm_gap_weight = tf.stop_gradient(vbm_hard - vbm_prob) + vbm_prob
        cbm_gap_weight = tf.stop_gradient(cbm_hard - cbm_prob) + cbm_prob

        vbm_energy_ev = x_norm[:, :, 0] * self.feature_std[0] + self.feature_mean[0]
        cbm_energy_ev = x_norm[:, :, 3] * self.feature_std[3] + self.feature_mean[3]

        expected_vbm = tf.reduce_sum(vbm_gap_weight * vbm_energy_ev, axis=1)
        expected_cbm = tf.reduce_sum(cbm_gap_weight * cbm_energy_ev, axis=1)
        seq_len = tf.cast(tf.shape(x_norm)[1] - 1, tf.float32)
        k_axis = tf.cast(tf.range(tf.shape(x_norm)[1]), tf.float32) / tf.maximum(seq_len, 1.0)
        expected_vbm_k = tf.reduce_sum(vbm_prob * k_axis[None, :], axis=1)
        expected_cbm_k = tf.reduce_sum(cbm_prob * k_axis[None, :], axis=1)
        k_distance = tf.abs(expected_cbm_k - expected_vbm_k)
        overlap = tf.reduce_sum(vbm_prob * cbm_prob, axis=1)
        kl_vc = tf.reduce_sum(vbm_prob * (tf.math.log(vbm_prob + 1e-8) - tf.math.log(cbm_prob + 1e-8)), axis=1)
        kl_cv = tf.reduce_sum(cbm_prob * (tf.math.log(cbm_prob + 1e-8) - tf.math.log(vbm_prob + 1e-8)), axis=1)
        topo_features = tf.stack([k_distance, overlap, 0.5 * (kl_vc + kl_cv)], axis=-1)
        probs = tf.stack([vbm_prob, cbm_prob], axis=-1)
        return tf.expand_dims(expected_cbm - expected_vbm, axis=-1), topo_features, probs


class SupervisedBandGapModel(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.gap_head = ExtremumExpectedGapHead(
            feature_mean=feature_mean,
            feature_std=feature_std,
            hidden_dim=max(32, d_model // 2),
            dropout=dropout,
            name="gap_head",
        )
        self.type_head = keras.Sequential(
            [
                layers.Dense(d_model, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model // 2, activation="gelu"),
                layers.Dense(3, activation="softmax"),
            ],
            name="type_head",
        )
        self.topology_weight = 0.3
        self.entropy_weight = 0.01
        self.extremum_weight = 0.5
        self.topology_rule_weight = 0.7
        self.encoder_gradient_scale = 0.1
        self.topology_margin = 0.10
        self.gap_loss_fn = keras.losses.MeanSquaredError()
        self.type_loss_fn = keras.losses.CategoricalCrossentropy(reduction="none")

    def _forward(self, x, training=False):
        encoded = self.encoder.encoder(x, training=training)
        pooled = self.encoder.pooling(encoded)
        gap, topo_features, probs = self.gap_head(encoded, x, training=training)
        type_input = tf.concat([pooled, topo_features], axis=-1)
        learned_type = self.type_head(type_input, training=training)
        vbm_k_dist = x[:, :, 2] * self.gap_head.feature_std[2] + self.gap_head.feature_mean[2]
        cbm_k_dist = x[:, :, 5] * self.gap_head.feature_std[5] + self.gap_head.feature_mean[5]
        vbm_zero_idx = tf.argmin(tf.abs(vbm_k_dist), axis=1, output_type=tf.int32)
        cbm_zero_idx = tf.argmin(tf.abs(cbm_k_dist), axis=1, output_type=tf.int32)
        direct_rule = tf.cast(tf.abs(vbm_zero_idx - cbm_zero_idx) <= 1, tf.float32)
        topology_type = tf.stack(
            [
                tf.zeros_like(direct_rule),
                0.02 + 0.96 * direct_rule,
                0.98 - 0.96 * direct_rule,
            ],
            axis=-1,
        )
        type_pred = (1.0 - self.topology_rule_weight) * learned_type + self.topology_rule_weight * topology_type
        type_pred = type_pred / tf.reduce_sum(type_pred, axis=-1, keepdims=True)
        return gap, type_pred, topo_features, probs

    def call(self, x, training=False):
        gap, type_pred, _topo_features, _probs = self._forward(x, training=training)
        return {
            "gap": gap,
            "type": type_pred,
        }

    def extremum_probabilities(self, x, training=False):
        _gap, _type_pred, topo_features, probs = self._forward(x, training=training)
        return topo_features, probs

    def _split_batch(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        return x, y, sample_weight

    def _auxiliary_losses(self, x, y_type, topo_features, probs):
        direct = y_type[:, 1]
        indirect = y_type[:, 2]
        k_distance = topo_features[:, 0]
        direct_loss = direct * k_distance
        indirect_loss = indirect * tf.nn.relu(self.topology_margin - k_distance)
        active = tf.maximum(direct + indirect, 1e-6)
        topology_loss = tf.reduce_sum(direct_loss + indirect_loss) / tf.reduce_sum(active)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
        entropy_loss = tf.reduce_mean(entropy)
        true_vbm_idx = tf.argmax(x[:, :, 0], axis=1, output_type=tf.int32)
        true_cbm_idx = tf.argmin(x[:, :, 3], axis=1, output_type=tf.int32)
        vbm_peak_loss = keras.losses.sparse_categorical_crossentropy(true_vbm_idx, probs[:, :, 0])
        cbm_peak_loss = keras.losses.sparse_categorical_crossentropy(true_cbm_idx, probs[:, :, 1])
        extremum_loss = 0.5 * (tf.reduce_mean(vbm_peak_loss) + tf.reduce_mean(cbm_peak_loss))
        return topology_loss, entropy_loss, extremum_loss

    def _weighted_type_loss(self, y_true, y_pred, sample_weight):
        per_sample = self.type_loss_fn(y_true, y_pred)
        if sample_weight is not None and isinstance(sample_weight, dict) and "type" in sample_weight:
            weights = tf.cast(sample_weight["type"], per_sample.dtype)
            return tf.reduce_sum(per_sample * weights) / tf.reduce_sum(weights)
        return tf.reduce_mean(per_sample)

    def _metrics_dict(self, total_loss, gap_loss, type_loss, topology_loss, entropy_loss, extremum_loss, y, pred):
        gap_mae = tf.reduce_mean(tf.abs(y["gap"] - pred["gap"]))
        type_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y["type"], axis=1), tf.argmax(pred["type"], axis=1)), tf.float32)
        )
        temperature = 0.01 + 0.49 * tf.nn.sigmoid(self.gap_head.temperature_raw)
        return {
            "loss": total_loss,
            "gap_loss": gap_loss,
            "gap_mae": gap_mae,
            "type_loss": type_loss,
            "type_acc": type_acc,
            "topology_loss": topology_loss,
            "entropy_loss": entropy_loss,
            "extremum_loss": extremum_loss,
            "temperature": temperature,
        }

    def train_step(self, data):
        x, y, sample_weight = self._split_batch(data)
        with tf.GradientTape() as tape:
            gap_pred, type_pred, topo_features, probs = self._forward(x, training=True)
            pred = {"gap": gap_pred, "type": type_pred}
            gap_loss = self.gap_loss_fn(y["gap"], gap_pred)
            type_loss = self._weighted_type_loss(y["type"], type_pred, sample_weight)
            topology_loss, entropy_loss, extremum_loss = self._auxiliary_losses(x, y["type"], topo_features, probs)
            total_loss = (
                gap_loss
                + self.loss_weights_dict["type"] * type_loss
                + self.topology_weight * topology_loss
                + self.entropy_weight * entropy_loss
                + self.extremum_weight * extremum_loss
            )

        variables = self.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        encoder_var_ids = {id(var) for var in self.encoder.encoder.trainable_variables}
        scaled_gradients = [
            (grad * self.encoder_gradient_scale if grad is not None and id(var) in encoder_var_ids else grad)
            for grad, var in zip(gradients, variables)
        ]
        self.optimizer.apply_gradients(zip(scaled_gradients, variables))
        return self._metrics_dict(total_loss, gap_loss, type_loss, topology_loss, entropy_loss, extremum_loss, y, pred)

    def test_step(self, data):
        x, y, sample_weight = self._split_batch(data)
        gap_pred, type_pred, topo_features, probs = self._forward(x, training=False)
        pred = {"gap": gap_pred, "type": type_pred}
        gap_loss = self.gap_loss_fn(y["gap"], gap_pred)
        type_loss = self._weighted_type_loss(y["type"], type_pred, sample_weight)
        topology_loss, entropy_loss, extremum_loss = self._auxiliary_losses(x, y["type"], topo_features, probs)
        total_loss = (
            gap_loss
            + self.loss_weights_dict["type"] * type_loss
            + self.topology_weight * topology_loss
            + self.entropy_weight * entropy_loss
            + self.extremum_weight * extremum_loss
        )
        return self._metrics_dict(total_loss, gap_loss, type_loss, topology_loss, entropy_loss, extremum_loss, y, pred)


def freeze_encoder_layers(ssl_encoder: keras.Model, freeze_layers: int) -> None:
    """Freeze first N transformer blocks while leaving later blocks trainable."""
    if hasattr(ssl_encoder, "projection_head"):
        ssl_encoder.projection_head.trainable = False
    if hasattr(ssl_encoder, "reconstruction_head"):
        ssl_encoder.reconstruction_head.trainable = False

    base = ssl_encoder.encoder
    total_layers = len(base.transformer_layers)
    freeze_to = min(freeze_layers, total_layers)

    for idx in range(total_layers):
        trainable = idx >= freeze_to
        base.transformer_layers[idx].trainable = trainable
        base.ln_attn[idx].trainable = trainable
        base.ln_ffn[idx].trainable = trainable
        base.add_attn[idx].trainable = trainable
        base.add_ffn[idx].trainable = trainable
        base.ffn_dense1[idx].trainable = trainable
        base.ffn_dense2[idx].trainable = trainable
        base.ffn_dropout[idx].trainable = trainable

    print(f"Frozen first {freeze_to}/{total_layers} transformer layers (incl. FFN sub-layers)")


def save_supervised_model_artifacts(model: keras.Model, model_path: str, config: Dict) -> Dict[str, str]:
    """Save supervised model artifacts in a robust subclass-friendly format."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if model_path.endswith(".weights.h5"):
        weights_path = model_path
        base = model_path[: -len(".weights.h5")]
    else:
        base, _ext = os.path.splitext(model_path)
        weights_path = base + ".weights.h5"
    config_path = base + "_config.json"

    outputs = {
        "weights": weights_path,
        "config": config_path,
    }

    model.save_weights(weights_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return outputs


def clean_finetune_artifacts(output_dir: str, checkpoint_dir: str, model_path: str) -> None:
    for path in [output_dir, checkpoint_dir]:
        if os.path.isdir(path):
            shutil.rmtree(path)
    for path in [
        model_path,
        model_path.replace(".weights.h5", "_config.json"),
        os.path.join(os.path.dirname(output_dir), "finetune_supervised_report_20260604.md"),
    ]:
        if path and os.path.exists(path):
            os.remove(path)


def compile_model(
    model: keras.Model,
    learning_rate: float,
    type_weight: float,
    class_weights: Dict[int, float],
    encoder_learning_rate: float = 1e-5,
    topology_weight: float = 0.3,
    entropy_weight: float = 0.01,
    extremum_weight: float = 0.5,
) -> None:
    del class_weights
    model.loss_weights_dict = {"gap": 1.0, "type": float(type_weight)}
    model.topology_weight = float(topology_weight)
    model.entropy_weight = float(entropy_weight)
    model.extremum_weight = float(extremum_weight)
    model.encoder_gradient_scale = float(encoder_learning_rate / max(learning_rate, 1e-12))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )


class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        logs["learning_rate"] = float(tf.keras.backend.get_value(lr))


class WarmupCosineDecay(keras.callbacks.Callback):
    """Warmup learning rate linearly from 0 to `target_lr` over `warmup_epochs`,
    then cosine-decay to `min_lr` over the remaining epochs.

    Placed *before* ReduceLROnPlateau in the callback list so that ROP acts as a
    further safety-net reduction on top of the scheduled decay.
    """

    def __init__(
        self,
        target_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.target_lr = float(target_lr)
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(warmup_epochs + 1, total_epochs)
        self.min_lr = float(min_lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        opt = self.model.optimizer
        if hasattr(opt, "learning_rate") and hasattr(opt.learning_rate, "assign"):
            opt.learning_rate.assign(float(lr))
        else:
            tf.keras.backend.set_value(opt.learning_rate, float(lr))


def evaluate_predictions(
    y_dft: np.ndarray,
    y_pred: np.ndarray,
    type_true: np.ndarray,
    type_pred: np.ndarray,
    tensor_gap: np.ndarray,
    gap_identity_tolerance: float = 0.5,
) -> Dict[str, float]:
    y_pred = y_pred.reshape(-1)
    type_pred_cls = np.argmax(type_pred, axis=1)

    residual_to_line = y_pred - tensor_gap
    dft_gap_residual = tensor_gap - y_dft
    residual_to_tensor = y_pred - tensor_gap
    gap_identity_large_residual_rate = float(
        np.mean(np.abs(residual_to_tensor) > gap_identity_tolerance)
    )

    return {
        "mae": float(np.mean(np.abs(residual_to_line))),
        "mse": float(np.mean(residual_to_line ** 2)),
        "rmse": float(np.sqrt(np.mean(residual_to_line ** 2))),
        "line_mode_gap_mae": float(np.mean(np.abs(residual_to_line))),
        "line_mode_gap_rmse": float(np.sqrt(np.mean(residual_to_line ** 2))),
        "dft_gap_residual_mae": float(np.mean(np.abs(dft_gap_residual))),
        "dft_gap_residual_mean": float(np.mean(dft_gap_residual)),
        "dft_gap_residual_max_abs": float(np.max(np.abs(dft_gap_residual))),
        "model_vs_global_dft_mae": float(np.mean(np.abs(y_pred - y_dft))),
        "type_acc": float(np.mean(type_pred_cls == type_true)),
        "macro_f1": float(f1_score(type_true, type_pred_cls, labels=[0, 1, 2], average="macro", zero_division=0)),
        "direct_gap_recall": float(recall_score(type_true, type_pred_cls, labels=[1], average="macro", zero_division=0)),
        "tensor_gap_mae": float(np.mean(np.abs(residual_to_tensor))),
        "tensor_gap_residual_mean": float(np.mean(residual_to_tensor)),
        "tensor_gap_residual_max_abs": float(np.max(np.abs(residual_to_tensor))),
        "gap_identity_tolerance_ev": float(gap_identity_tolerance),
        "gap_identity_large_residual_rate": gap_identity_large_residual_rate,
        "gap_identity_score": float(max(0.0, 1.0 - gap_identity_large_residual_rate)),
        "parity_corr": float(np.corrcoef(tensor_gap, y_pred)[0, 1]) if len(tensor_gap) > 1 else 0.0,
        "r2": r2_score(tensor_gap, y_pred),
    }


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_pred = y_pred.reshape(-1)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    mae = float(np.mean(np.abs(y_pred - y_true)))
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolor="black", linewidth=0.35, label="materials")
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label=f"y = x | MAE={mae:.3f} eV, R2={r2:.3f}")
    plt.text(
        0.05,
        0.95,
        f"MAE = {mae:.3f} eV\nR2 = {r2:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )
    plt.xlabel("DFT band gap (eV)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_training_diagnostics(history: keras.callbacks.History, output_dir: str) -> None:
    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, hist.get("loss", []), label="train")
    axes[0].plot(epochs, hist.get("val_loss", []), label="validation")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, hist.get("gap_mae", []), label="train")
    axes[1].plot(epochs, hist.get("val_gap_mae", []), label="validation")
    axes[1].set_title("Band Gap MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (eV)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, hist.get("type_acc", []), label="train")
    axes[2].plot(epochs, hist.get("val_type_acc", []), label="validation")
    axes[2].set_title("Gap Type Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=220)
    plt.close(fig)

    if "learning_rate" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, hist["learning_rate"], marker="o", markersize=2.5)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning Rate Schedule")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_rate_curve.png"), dpi=220)
        plt.close()


def save_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    errors = y_pred.reshape(-1) - y_true.reshape(-1)
    plt.figure(figsize=(7, 4.5))
    plt.hist(errors, bins=min(24, max(8, len(errors) // 2)), color="#4C78A8", edgecolor="white", alpha=0.9)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.axvline(float(np.mean(errors)), color="#D62728", linestyle="-", linewidth=1.2, label=f"mean={np.mean(errors):.3f} eV")
    plt.xlabel("Prediction error: predicted - DFT (eV)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_confusion_matrix_plot(type_true: np.ndarray, type_pred: np.ndarray, output_path: str) -> Dict[str, object]:
    labels = ["metal", "direct", "indirect"]
    pred_cls = np.argmax(type_pred, axis=1)
    cm = confusion_matrix(type_true, pred_cls, labels=[0, 1, 2])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1), where=row_sum != 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, matrix, title, fmt in [
        (axes[0], cm, "Counts", "d"),
        (axes[1], cm_norm, "Row-normalized", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_title(title)
        ax.set_xticks(range(3), labels=labels, rotation=25, ha="right")
        ax.set_yticks(range(3), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Gap Type Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {"labels": labels, "counts": cm.tolist(), "row_normalized": cm_norm.tolist()}


def save_roc_curves(type_true: np.ndarray, type_pred: np.ndarray, output_path: str) -> Dict[str, float]:
    labels = ["metal", "direct", "indirect"]
    y_true = np.eye(3, dtype=np.float32)[type_true]
    aucs = {}
    plt.figure(figsize=(6, 5))
    for idx, label in enumerate(labels):
        if len(np.unique(y_true[:, idx])) < 2:
            aucs[label] = None
            continue
        fpr, tpr, _ = roc_curve(y_true[:, idx], type_pred[:, idx])
        auc_value = float(auc(fpr, tpr))
        aucs[label] = auc_value
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return aucs


def save_extremum_probability_heatmaps(
    model: keras.Model,
    X_norm: np.ndarray,
    material_ids: np.ndarray,
    type_true: np.ndarray,
    type_pred: np.ndarray,
    output_path: str,
    max_examples: int = 8,
) -> None:
    """Visualize P_vbm and P_cbm so reviewers can inspect learned extrema."""
    topo_features, probs = model.extremum_probabilities(tf.constant(X_norm, dtype=tf.float32), training=False)
    topo_features = topo_features.numpy()
    probs = probs.numpy()
    pred_cls = np.argmax(type_pred, axis=1)
    type_names = ["metal", "direct", "indirect"]
    order = np.argsort(topo_features[:, 0])
    selected = order[: min(max_examples, len(order))]

    fig, axes = plt.subplots(len(selected), 1, figsize=(9, max(2.0, 1.75 * len(selected))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, selected):
        heat = np.stack([probs[idx, :, 0], probs[idx, :, 1]], axis=0)
        ax.imshow(heat, aspect="auto", cmap="magma", interpolation="nearest", vmin=0.0)
        v_peak = int(np.argmax(probs[idx, :, 0]))
        c_peak = int(np.argmax(probs[idx, :, 1]))
        ax.axvline(v_peak, color="#66C2A5", linestyle="--", linewidth=1.0)
        ax.axvline(c_peak, color="#8DA0CB", linestyle=":", linewidth=1.2)
        ax.set_yticks([0, 1], labels=["P_vbm", "P_cbm"])
        ax.set_title(
            f"{material_ids[idx]} | true={type_names[int(type_true[idx])]} "
            f"pred={type_names[int(pred_cls[idx])]} | |kC-kV|={topo_features[idx, 0]:.3f}",
            fontsize=9,
        )
    axes[-1].set_xlabel("Resampled k-point index")
    fig.suptitle("Learned Extremum Probability Heatmaps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def denormalize_sequence(X_norm: np.ndarray, norm_path: str) -> np.ndarray:
    mean, std = load_norm_stats(norm_path)
    seq = X_norm * std + mean
    if seq.shape[-1] % 2 != 0:
        raise ValueError(f"Expected an even feature dimension, got {seq.shape[-1]}")
    channels_per_band = seq.shape[-1] // 2
    return seq.reshape(seq.shape[0], seq.shape[1], 2, channels_per_band).transpose(0, 2, 1, 3)


def reconstruct_raw_tensors(model: keras.Model, X_norm: np.ndarray, norm_path: str) -> np.ndarray:
    recon_norm = model.encoder.reconstruct(X_norm, training=False).numpy()
    return denormalize_sequence(recon_norm, norm_path)


def save_band_overlay_examples(
    X_true_raw: np.ndarray,
    X_recon_raw: np.ndarray,
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    max_examples: int = 6,
) -> None:
    errors = np.abs(y_pred.reshape(-1) - y_true.reshape(-1))
    order = np.argsort(errors)
    if len(order) > max_examples:
        selected = np.unique(np.concatenate([order[: max_examples // 2], order[-(max_examples - max_examples // 2) :]]))
    else:
        selected = order

    k_axis = np.linspace(0, 1, X_true_raw.shape[2])
    symmetry_ticks = np.linspace(0, 1, 5)
    symmetry_labels = ["G", "X", "M", "K", "G"]
    fig, axes = plt.subplots(len(selected), 1, figsize=(8, max(3.0, 2.5 * len(selected))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, selected):
        true_vbm = X_true_raw[idx, 0, :, 0]
        true_cbm = X_true_raw[idx, 1, :, 0]
        pred_vbm = X_recon_raw[idx, 0, :, 0]
        pred_cbm = X_recon_raw[idx, 1, :, 0]
        ax.plot(k_axis, true_vbm, color="black", linewidth=1.5, label="true VBM")
        ax.plot(k_axis, true_cbm, color="black", linewidth=1.5, alpha=0.65, label="true CBM")
        ax.plot(k_axis, pred_vbm, color="#D62728", linestyle="--", linewidth=1.4, label="recon VBM")
        ax.plot(k_axis, pred_cbm, color="#D62728", linestyle="--", linewidth=1.4, alpha=0.75, label="recon CBM")
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, label="E_F = 0 eV")
        ax.set_xticks(symmetry_ticks, symmetry_labels)
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{material_ids[idx]} | DFT gap={y_true[idx]:.3f} eV, pred={y_pred.reshape(-1)[idx]:.3f} eV")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Normalized k-path")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_publication_band_overlay_examples(
    X_true_raw: np.ndarray,
    X_recon_raw: np.ndarray,
    material_ids: np.ndarray,
    kpath_labels_by_material: Dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    max_examples: int = 6,
) -> None:
    errors = np.abs(y_pred.reshape(-1) - y_true.reshape(-1))
    order = np.argsort(errors)
    if len(order) > max_examples:
        selected = np.unique(np.concatenate([order[: max_examples // 2], order[-(max_examples - max_examples // 2) :]]))
    else:
        selected = order
    save_band_overlay_grid(
        output_path=output_path,
        true_tensors=X_true_raw,
        recon_tensors=X_recon_raw,
        material_ids=material_ids,
        kpath_labels_by_material=kpath_labels_by_material,
        y_true=y_true,
        y_pred=y_pred,
        selected=selected,
    )


def save_curvature_zoom_examples(
    X_true_raw: np.ndarray,
    X_recon_raw: np.ndarray,
    material_ids: np.ndarray,
    output_path: str,
    max_examples: int = 3,
) -> None:
    selected = np.linspace(0, len(X_true_raw) - 1, min(max_examples, len(X_true_raw)), dtype=int)
    k_axis = np.linspace(0, 1, X_true_raw.shape[2])
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, max(3.2, 3.0 * len(selected))), sharex=False)
    axes = np.atleast_2d(axes)
    window = 10
    for row, idx in enumerate(selected):
        true_vbm_e = X_true_raw[idx, 0, :, 0]
        true_cbm_e = X_true_raw[idx, 1, :, 0]
        recon_vbm_e = X_recon_raw[idx, 0, :, 0]
        recon_cbm_e = X_recon_raw[idx, 1, :, 0]
        true_energy_curv = [
            _local_curvature(true_vbm_e),
            _local_curvature(true_cbm_e),
        ]
        recon_energy_curv = [
            _local_curvature(recon_vbm_e),
            _local_curvature(recon_cbm_e),
        ]
        vbm_center = int(np.argmax(true_vbm_e))
        cbm_center = int(np.argmin(true_cbm_e))
        for col, (band_idx, center, name) in enumerate([(0, vbm_center, "VBM"), (1, cbm_center, "CBM")]):
            start = max(0, center - window)
            end = min(X_true_raw.shape[2], center + window + 1)
            ax = axes[row, col]
            ax.plot(k_axis[start:end], true_energy_curv[band_idx][start:end], color="black", label="true d2E/dk2")
            ax.plot(k_axis[start:end], recon_energy_curv[band_idx][start:end], color="#D62728", linestyle="--", label="recon d2E/dk2")
            ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
            ax.set_title(f"{material_ids[idx]} {name} curvature zoom")
            ax.set_xlabel("Normalized k-path")
            ax.set_ylabel("Curvature")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_latent_tsne(model: keras.Model, data: Dict[str, np.ndarray], output_path: str) -> None:
    X = np.concatenate([data["X_train"], data["X_test"]], axis=0)
    groups = np.concatenate([data["groups_train"], data["groups_test"]], axis=0)
    split = np.asarray(["train"] * len(data["X_train"]) + ["ood_test"] * len(data["X_test"]))
    features = model.encoder(tf.constant(X, dtype=tf.float32), return_features=True, training=False).numpy()
    perplexity = max(2, min(30, max(2, len(features) // 4), len(features) - 1))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(features)

    plt.figure(figsize=(7, 5.5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=groups, cmap="viridis", s=34, alpha=0.82, edgecolor="none")
    test_mask = split == "ood_test"
    plt.scatter(emb[test_mask, 0], emb[test_mask, 1], facecolors="none", edgecolors="black", s=70, linewidths=0.8, label="OOD test")
    plt.colorbar(sc, label="Spacegroup number")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space by Spacegroup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def physics_violation_stats(y_pred: np.ndarray, X_recon_raw: np.ndarray) -> Dict[str, float]:
    pred_gap = y_pred.reshape(-1)
    vbm_center = np.argmax(X_recon_raw[:, 0, :, 0], axis=1)
    cbm_center = np.argmin(X_recon_raw[:, 1, :, 0], axis=1)
    vbm_curves = np.asarray([_local_curvature(X_recon_raw[i, 0, :, 0]) for i in range(len(X_recon_raw))])
    cbm_curves = np.asarray([_local_curvature(X_recon_raw[i, 1, :, 0]) for i in range(len(X_recon_raw))])
    vbm_curv = np.asarray([vbm_curves[i, vbm_center[i]] for i in range(len(X_recon_raw))])
    cbm_curv = np.asarray([cbm_curves[i, cbm_center[i]] for i in range(len(X_recon_raw))])
    return {
        "negative_predicted_gap_rate": float(np.mean(pred_gap < 0.0)),
        "vbm_positive_curvature_rate": float(np.mean(vbm_curv > 0.0)),
        "cbm_negative_curvature_rate": float(np.mean(cbm_curv < 0.0)),
    }


def save_physics_violation_chart(stats: Dict[str, float], output_path: str) -> None:
    names = ["Negative predicted gap", "VBM curvature > 0", "CBM curvature < 0", "Gap identity residual"]
    values = [
        stats["negative_predicted_gap_rate"],
        stats["vbm_positive_curvature_rate"],
        stats["cbm_negative_curvature_rate"],
        stats["gap_identity_large_residual_rate"],
    ]
    plt.figure(figsize=(7, 4.2))
    plt.bar(names, values, color=["#D62728", "#F58518", "#F58518", "#9467BD"])
    plt.ylim(0, 1)
    plt.ylabel("Violation rate")
    plt.title("Physics Violation Rates on OOD Test")
    plt.xticks(rotation=18, ha="right")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_prediction_table(
    output_path: str,
    material_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    type_true: np.ndarray,
    type_pred: np.ndarray,
    tensor_gap: np.ndarray,
    groups: np.ndarray,
) -> None:
    rows = []
    type_names = ["metal", "direct", "indirect"]
    for i, mid in enumerate(material_ids):
        pred_type_idx = int(np.argmax(type_pred[i]))
        rows.append(
            {
                "material_id": str(mid),
                "spacegroup_number": int(groups[i]),
                "global_dft_gap": float(y_true[i]),
                "line_mode_tensor_gap_ecbm_minus_evbm": float(tensor_gap[i]),
                "pred_line_mode_gap": float(y_pred.reshape(-1)[i]),
                "line_mode_gap_error": float(y_pred.reshape(-1)[i] - tensor_gap[i]),
                "dft_gap_residual_tensor_minus_global": float(tensor_gap[i] - y_true[i]),
                "model_vs_global_dft_error": float(y_pred.reshape(-1)[i] - y_true[i]),
                "true_gap_type": type_names[int(type_true[i])],
                "pred_gap_type": type_names[pred_type_idx],
                "pred_type_confidence": float(type_pred[i, pred_type_idx]),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_markdown_report(summary: Dict, output_path: str) -> None:
    test = summary["ood_test_metrics"]
    train = summary["train_metrics"]
    class_weights = summary["class_weights"]
    violation = summary["physics_violation_rates"]
    roc_auc = summary["roc_auc"]

    body = f"""# Supervised Fine-tuning Report - 2026-06-04

## Scope

- Input tensor: `data_cache/ood_tensors/band_tensors_ood_split.npz`
- Tensor contract: `(N, 2, 128, 3)` -> `(N, 128, 6)`
- Features: `[VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]`
- OOD split: grouped by `spacegroup_number`, no random sample-wise split.
- Regression target: line-mode tensor gap `E_CBM - E_VBM`, not global MP DFT gap.
- Gap head: sequence-local extremum expected-value head with learned temperature; it denormalizes VBM_E/CBM_E internally, so gap loss and MAE are computed in eV.
- Type head: pooled encoder features plus topology priors `[|k_CBM-k_VBM|, overlap(P_vbm,P_cbm), symmetric_KL]`.
- Auxiliary losses: topology consistency plus entropy sharpening of extremum probabilities.
- Global DFT gap is reported only as an information-bottleneck residual.

## Imbalance Fix

- Classification loss: Categorical Crossentropy
- Weighting policy: class weights are applied as type-head sample weights.
- Class weights: {class_weights}

## OOD Test Metrics

- Line-mode Gap MAE: {test["line_mode_gap_mae"]:.6f} eV
- Line-mode Gap RMSE: {test["line_mode_gap_rmse"]:.6f} eV
- DFT Gap Residual MAE: {test["dft_gap_residual_mae"]:.6f} eV
- Model vs Global DFT MAE: {test["model_vs_global_dft_mae"]:.6f} eV
- R2: {test["r2"]:.6f}
- Accuracy: {test["type_acc"]:.6f}
- Macro F1: {test["macro_f1"]:.6f}
- Direct Gap Recall: {test["direct_gap_recall"]:.6f}
- Direct AUC: {roc_auc.get("direct")}
- Indirect AUC: {roc_auc.get("indirect")}

## Train Metrics

- Line-mode Gap MAE: {train["line_mode_gap_mae"]:.6f} eV
- DFT Gap Residual MAE: {train["dft_gap_residual_mae"]:.6f} eV
- Accuracy: {train["type_acc"]:.6f}
- Macro F1: {train["macro_f1"]:.6f}
- Direct Gap Recall: {train["direct_gap_recall"]:.6f}

## Physics Violation Rates

- Negative predicted gap: {violation["negative_predicted_gap_rate"]:.6f}
- VBM positive curvature: {violation["vbm_positive_curvature_rate"]:.6f}
- CBM negative curvature: {violation["cbm_negative_curvature_rate"]:.6f}
- Gap identity large residual rate: {violation["gap_identity_large_residual_rate"]:.6f}
- Physics score: {violation["physics_score"]:.6f}
- Gap identity MAE: {violation["gap_identity_mae"]:.6f} eV

## Outputs

- Metrics: `reports/finetune_supervised/metrics_summary.json`
- Predictions: `reports/finetune_supervised/ood_test_predictions.json`
- Figures: `reports/finetune_supervised/*.png`
- Model weights: `models/finetuned_gap_predictor.weights.h5`
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SSL encoder for gap prediction")
    parser.add_argument("--tensor-npz", default="./data_cache/ood_tensors/band_tensors_ood_split.npz")
    parser.add_argument("--encoder", default="./models/ssl_mbm_pretrained.keras")
    parser.add_argument("--norm", default="./models/ssl_mbm_norm_stats.json")
    parser.add_argument("--output-dir", default="./reports/finetune_supervised")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/finetune_supervised")
    parser.add_argument("--model-path", default="./models/finetuned_gap_predictor.weights.h5")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=10, help="linear warmup from 0 to lr over N epochs, then cosine decay")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="floor for cosine decay")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="head optimizer learning rate")
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5, help="effective learning rate for unfrozen encoder layers")
    parser.add_argument("--type-weight", type=float, default=2.0)
    parser.add_argument("--freeze-layers", type=int, default=2)
    parser.add_argument("--topology-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--extremum-weight", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gap-identity-tolerance", type=float, default=0.5)
    parser.add_argument("--fresh", action="store_true", help="remove old fine-tuning artifacts before training")
    parser.add_argument("--augment", action="store_true", default=True, help="apply virtual strain augmentation to training data")
    parser.add_argument("--strain-scale", type=float, default=0.01, help="virtual strain magnitude (default: 0.01)")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.random_state)
    if args.fresh:
        clean_finetune_artifacts(args.output_dir, args.checkpoint_dir, args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    data = load_dataset(args.tensor_npz, args.norm)
    kpath_labels_by_material = load_kpath_labels(args.tensor_npz)
    print("Loaded OOD split")
    print(f"  train: {data['X_train'].shape}, groups={len(set(data['groups_train'].tolist()))}")
    print(f"  test:  {data['X_test'].shape}, groups={len(set(data['groups_test'].tolist()))}")
    print(f"  group overlap: {set(data['groups_train'].tolist()).intersection(set(data['groups_test'].tolist()))}")

    # Class distribution check
    train_types, train_counts = np.unique(data["type_train"], return_counts=True)
    test_types, test_counts = np.unique(data["type_test"], return_counts=True)
    type_names = {0: "metal", 1: "direct", 2: "indirect"}
    print("  class distribution:")
    print(f"    train: {', '.join(f'{type_names.get(t,t)}={c}' for t,c in zip(train_types, train_counts))}")
    print(f"    test:  {', '.join(f'{type_names.get(t,t)}={c}' for t,c in zip(test_types, test_counts))}")
    missing_classes = set([0, 1, 2]) - set(test_types.tolist())
    if missing_classes:
        missing_names = [type_names[c] for c in missing_classes]
        print(f"  ** WARNING: test set missing classes: {missing_names}. These metrics will be unreliable.")
        print(f"     Re-run `python -m src.data.ood_tensor_builder` with --random-state to try another split,")
        print(f"     or increase the dataset size so all classes appear in the test split.")

    class_weights = compute_class_weights(data["type_train"])
    print(f"  class weights: {class_weights}")

    train_ds = make_tf_dataset(
        data["X_train"],
        data["tensor_gap_train"],
        data["type_train"],
        args.batch_size,
        shuffle=True,
        class_weights=class_weights,
        augment=args.augment,
        strain_scale=args.strain_scale,
    )
    test_ds = make_tf_dataset(
        data["X_test"], data["tensor_gap_test"], data["type_test"], args.batch_size, shuffle=False
    )

    ssl_encoder = tf.keras.models.load_model(args.encoder, compile=False)
    freeze_info = apply_freeze_encoder_layers(ssl_encoder, args.freeze_layers)
    print(f"Frozen first {freeze_info['freeze_layers']}/{freeze_info['total_transformer_layers']} transformer layers")

    model = SupervisedBandGapModel(
        ssl_encoder,
        feature_mean=data["feature_mean"],
        feature_std=data["feature_std"],
    )
    model(tf.zeros([1, data["X_train"].shape[1], data["X_train"].shape[2]], dtype=tf.float32))
    compile_model(
        model,
        args.learning_rate,
        args.type_weight,
        class_weights=class_weights,
        encoder_learning_rate=args.encoder_learning_rate,
        topology_weight=args.topology_weight,
        entropy_weight=args.entropy_weight,
        extremum_weight=args.extremum_weight,
    )

    callbacks = [
        WarmupCosineDecay(
            target_lr=args.learning_rate,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr=args.min_lr,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, "best.weights.h5"),
            monitor="val_gap_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_gap_mae",
            mode="min",
            patience=20,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
        LearningRateLogger(),
        keras.callbacks.CSVLogger(os.path.join(args.output_dir, "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    pred_train = model.predict(data["X_train"], verbose=0)
    pred_test = model.predict(data["X_test"], verbose=0)
    recon_test_raw = reconstruct_raw_tensors(model, data["X_test"], args.norm)

    train_metrics = evaluate_predictions(
        data["y_train"],
        pred_train["gap"],
        data["type_train"],
        pred_train["type"],
        data["tensor_gap_train"],
        gap_identity_tolerance=args.gap_identity_tolerance,
    )
    test_metrics = evaluate_predictions(
        data["y_test"],
        pred_test["gap"],
        data["type_test"],
        pred_test["type"],
        data["tensor_gap_test"],
        gap_identity_tolerance=args.gap_identity_tolerance,
    )

    saved_model_outputs = save_supervised_model_artifacts(
        model,
        args.model_path,
        {
            "encoder_path": args.encoder,
            "norm_path": args.norm,
            "input_shape": list(data["X_train"].shape[1:]),
            "freeze_layers": args.freeze_layers,
            "head_learning_rate": args.learning_rate,
            "encoder_learning_rate": args.encoder_learning_rate,
            "type_weight": args.type_weight,
            "topology_weight": args.topology_weight,
            "entropy_weight": args.entropy_weight,
            "extremum_weight": args.extremum_weight,
            "random_state": args.random_state,
            "class_weights": class_weights,
            "classification_loss": "CategoricalCrossentropy with class sample weights",
            "regression_target": "tensor_gap_ecbm_minus_evbm",
            "energy_scale": "gap head denormalizes VBM_E/CBM_E; gap loss and MAE are reported in eV",
            "heads": {
                "gap": "Conv1D extremum logits -> learned-temperature softmax -> expected E_CBM - E_VBM in eV",
                "type": "concat(pooled encoder, topology priors)->Dense->Dropout->Dense->Dense(3, softmax)",
            },
        },
    )
    write_finetune_strategy_report(
        os.path.join(args.output_dir, "finetune_strategy_summary.json"),
        freeze_info=freeze_info,
        class_weights=class_weights,
        type_weight=args.type_weight,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        topology_weight=args.topology_weight,
        entropy_weight=args.entropy_weight,
        extremum_weight=args.extremum_weight,
    )
    save_parity_plot(
        data["tensor_gap_test"],
        pred_test["gap"],
        os.path.join(args.output_dir, "parity_plot_ood_test.png"),
        "OOD Test Line-mode Gap Parity Plot",
    )
    save_parity_plot(
        data["tensor_gap_train"],
        pred_train["gap"],
        os.path.join(args.output_dir, "parity_plot_train.png"),
        "Train Line-mode Gap Parity Plot",
    )
    save_training_diagnostics(history, args.output_dir)
    save_error_histogram(
        data["tensor_gap_test"],
        pred_test["gap"],
        os.path.join(args.output_dir, "error_distribution_ood_test.png"),
        "OOD Test Line-mode Gap Error Distribution",
    )
    confusion_info = save_confusion_matrix_plot(
        data["type_test"],
        pred_test["type"],
        os.path.join(args.output_dir, "confusion_matrix_ood_test.png"),
    )
    roc_auc = save_roc_curves(
        data["type_test"],
        pred_test["type"],
        os.path.join(args.output_dir, "roc_curves_ood_test.png"),
    )
    save_extremum_probability_heatmaps(
        model,
        data["X_test"],
        data["material_ids_test"],
        data["type_test"],
        pred_test["type"],
        os.path.join(args.output_dir, "extremum_probability_heatmaps_ood_test.png"),
    )
    save_publication_band_overlay_examples(
        data["X_test_raw"],
        recon_test_raw,
        data["material_ids_test"],
        kpath_labels_by_material,
        data["y_test"],
        pred_test["gap"],
        os.path.join(args.output_dir, "band_overlay_examples_ood_test.png"),
    )
    save_curvature_zoom_examples(
        data["X_test_raw"],
        recon_test_raw,
        data["material_ids_test"],
        os.path.join(args.output_dir, "curvature_zoom_examples_ood_test.png"),
    )
    save_latent_tsne(
        model,
        data,
        os.path.join(args.output_dir, "latent_tsne_spacegroups.png"),
    )
    violation_info = physics_violation_stats(pred_test["gap"], recon_test_raw)
    recon_tensor_gap = tensor_gap_from_extrema(recon_test_raw)
    validator_info = PhysicsValidator().validate(recon_tensor_gap, recon_test_raw)
    reconstruction_validator_info = {
        f"reconstruction_{key}": value for key, value in validator_info.items()
    }
    violation_info.update(reconstruction_validator_info)
    violation_info.update(
        {
            "gap_identity_tolerance_ev": test_metrics["gap_identity_tolerance_ev"],
            "gap_identity_large_residual_rate": test_metrics["gap_identity_large_residual_rate"],
            "gap_identity_mae": test_metrics["tensor_gap_mae"],
            "gap_identity_max_abs": test_metrics["tensor_gap_residual_max_abs"],
            "gap_identity_score": test_metrics["gap_identity_score"],
            "physics_score": float(
                min(
                    test_metrics["gap_identity_score"],
                    validator_info["physics_score"],
                    max(0.0, 1.0 - violation_info["negative_predicted_gap_rate"]),
                    max(0.0, 1.0 - violation_info["vbm_positive_curvature_rate"]),
                    max(0.0, 1.0 - violation_info["cbm_negative_curvature_rate"]),
                )
            ),
        }
    )
    save_physics_violation_chart(
        violation_info,
        os.path.join(args.output_dir, "physics_violation_rates_ood_test.png"),
    )

    # --- MC Dropout uncertainty ---
    print("Running MC Dropout (50 samples) for uncertainty estimation ...")
    mc_result = mc_dropout_predict_with_type(
        model, data["X_test"], n_samples=50, batch_size=args.batch_size, seed=args.random_state
    )
    mc_cal = mc_calibration_check(
        data["tensor_gap_test"], mc_result, tolerance_ev=args.gap_identity_tolerance
    )
    mc_summary = {
        "mc_samples": 50,
        "method": "dropout-as-Bayesian-approximation (Gal & Ghahramani 2016)",
        "ci_coverage_95pct": mc_cal["ci_coverage_95pct"],
        "ci_tolerance_ev": mc_cal["tolerance_ev"],
        "mean_uncertainty_ev": mc_cal["mean_uncertainty_ev"],
        "median_uncertainty_ev": mc_cal["median_uncertainty_ev"],
        "max_uncertainty_ev": mc_cal["max_uncertainty_ev"],
    }
    print(f"  MC uncertainty: mean={mc_cal['mean_uncertainty_ev']:.4f} eV, "
          f"median={mc_cal['median_uncertainty_ev']:.4f} eV, "
          f"CI coverage={mc_cal['ci_coverage_95pct']:.2%}")
    # Save MC predictions
    mc_pred_path = os.path.join(args.output_dir, "mc_uncertainty_predictions.json")
    mc_payload = {
        "mc_summary": mc_summary,
        "predictions": [
            {
                "material_id": str(data["material_ids_test"][i]),
                "gap_mean_ev": float(mc_result["gap_mean"][i]),
                "gap_std_ev": float(mc_result["gap_std"][i]),
                "gap_coeff_var": float(mc_result["gap_coeff_var"][i]),
                "gap_ci95_low": float(mc_result["gap_ci95_low"][i]),
                "gap_ci95_high": float(mc_result["gap_ci95_high"][i]),
                "type_pred": int(mc_result["type_pred"][i]),
                "type_entropy": float(mc_result["type_entropy"][i]),
            }
            for i in range(len(data["material_ids_test"]))
        ],
    }
    with open(mc_pred_path, "w", encoding="utf-8") as f:
        json.dump(mc_payload, f, indent=2, ensure_ascii=False)
    print(f"  MC predictions saved to {mc_pred_path}\n")

    save_prediction_table(
        os.path.join(args.output_dir, "ood_test_predictions.json"),
        data["material_ids_test"],
        data["y_test"],
        pred_test["gap"],
        data["type_test"],
        pred_test["type"],
        data["tensor_gap_test"],
        data["groups_test"],
    )

    summary = {
        "train_metrics": train_metrics,
        "ood_test_metrics": test_metrics,
        "epochs_ran": len(history.history["loss"]),
        "best_val_gap_mae": float(np.min(history.history["val_gap_mae"])),
        "freeze_layers": args.freeze_layers,
        "type_weight": args.type_weight,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "topology_weight": args.topology_weight,
        "entropy_weight": args.entropy_weight,
        "extremum_weight": args.extremum_weight,
        "random_state": args.random_state,
        "class_weights": class_weights,
        "classification_loss": {
            "name": "CategoricalCrossentropy",
            "class_weighting": "type-head sample weights",
        },
        "regression_target": "tensor_gap_ecbm_minus_evbm",
        "energy_scale": "Gap head denormalizes VBM_E/CBM_E; regression loss and MAE are in physical eV.",
        "gap_head": "Sequence Conv1D extremum probability head with learned temperature and expected-value E_CBM - E_VBM.",
        "type_head": "Pooled encoder features concatenated with topology priors from P_vbm/P_cbm.",
        "auxiliary_losses": {
            "topology_consistency": args.topology_weight,
            "entropy_sharpening": args.entropy_weight,
            "extremum_peak_supervision": args.extremum_weight,
        },
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_info,
        "physics_violation_rates": violation_info,
        "mc_uncertainty": mc_summary,
        "outputs": {
            "model": saved_model_outputs["weights"],
            "model_artifacts": saved_model_outputs,
            "best_weights": os.path.join(args.checkpoint_dir, "best.weights.h5"),
            "training_curves": os.path.join(args.output_dir, "training_curves.png"),
            "learning_rate_curve": os.path.join(args.output_dir, "learning_rate_curve.png"),
            "parity_plot_ood_test": os.path.join(args.output_dir, "parity_plot_ood_test.png"),
            "error_distribution_ood_test": os.path.join(args.output_dir, "error_distribution_ood_test.png"),
            "confusion_matrix_ood_test": os.path.join(args.output_dir, "confusion_matrix_ood_test.png"),
            "roc_curves_ood_test": os.path.join(args.output_dir, "roc_curves_ood_test.png"),
            "extremum_probability_heatmaps_ood_test": os.path.join(args.output_dir, "extremum_probability_heatmaps_ood_test.png"),
            "band_overlay_examples_ood_test": os.path.join(args.output_dir, "band_overlay_examples_ood_test.png"),
            "curvature_zoom_examples_ood_test": os.path.join(args.output_dir, "curvature_zoom_examples_ood_test.png"),
            "latent_tsne_spacegroups": os.path.join(args.output_dir, "latent_tsne_spacegroups.png"),
            "physics_violation_rates_ood_test": os.path.join(args.output_dir, "physics_violation_rates_ood_test.png"),
            "predictions": os.path.join(args.output_dir, "ood_test_predictions.json"),
            "mc_uncertainty_predictions": os.path.join(args.output_dir, "mc_uncertainty_predictions.json"),
            "finetune_strategy_summary": os.path.join(args.output_dir, "finetune_strategy_summary.json"),
        },
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    report_path = os.path.join(os.path.dirname(args.output_dir), "finetune_supervised_report_20260604.md")
    save_markdown_report(summary, report_path)

    print("=" * 60)
    print("Fine-tuning complete")
    print("=" * 60)
    print("Train:", train_metrics)
    print("OOD Test:", test_metrics)
    print(f"Model artifacts: {saved_model_outputs}")


if __name__ == "__main__":
    main()
