"""
SSL trainer for Masked Band Modeling with physics-loss monitoring.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.band_structure_encoder import SSLEncoder
from src.models.losses import BandCurvatureLoss, CurvatureConsistencyLoss


class MBMTrainer:
    """Masked Band Modeling trainer with adaptive physics-loss weights."""

    def __init__(
        self,
        model: SSLEncoder,
        learning_rate: float,
        mask_ratio: float,
        sign_weight: float,
        consistency_weight: float,
        checkpoint_dir: str,
        log_dir: str,
    ):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.mask_ratio = mask_ratio
        self.curvature_weight = tf.Variable(sign_weight, trainable=False, dtype=tf.float32)
        self.symmetry_weight = tf.Variable(consistency_weight, trainable=False, dtype=tf.float32)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.curvature_loss_fn = BandCurvatureLoss()
        self.symmetry_loss_fn = CurvatureConsistencyLoss()

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
        best_val = float("inf")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_metrics = self._run_epoch(train_dataset, training=True)
            val_metrics = self._run_epoch(val_dataset, training=False)
            self._adapt_physics_weights(val_metrics)

            with self.train_writer.as_default():
                for key, value in train_metrics.items():
                    tf.summary.scalar(key, value, step=epoch)
                tf.summary.scalar("curvature_weight", float(self.curvature_weight.numpy()), step=epoch)
                tf.summary.scalar("symmetry_weight", float(self.symmetry_weight.numpy()), step=epoch)
            with self.val_writer.as_default():
                for key, value in val_metrics.items():
                    tf.summary.scalar(key, value, step=epoch)
                tf.summary.scalar("curvature_weight", float(self.curvature_weight.numpy()), step=epoch)
                tf.summary.scalar("symmetry_weight", float(self.symmetry_weight.numpy()), step=epoch)

            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"train={train_metrics['total']:.5f} "
                f"val={val_metrics['total']:.5f} "
                f"mse={val_metrics['mse_loss']:.5f} "
                f"curv={val_metrics['curvature_loss']:.5f} "
                f"sym={val_metrics['symmetry_loss']:.5f} "
                f"cw={float(self.curvature_weight.numpy()):.3f} "
                f"sw={float(self.symmetry_weight.numpy()):.3f} "
                f"time={time.time() - start:.1f}s"
            )

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                self.save_checkpoint("best")

            if epoch % 10 == 0:
                self.save_checkpoint(str(epoch))

    def _adapt_physics_weights(self, val_metrics: Dict[str, float]) -> None:
        """Proportionally adjust physics-loss weights based on validation metrics.

        Uses log-scale proportional control: larger violation → larger weight increase,
        avoiding the dead zone between 1e-5 and 1e-3 in the previous implementation.
        Symmetry weight also decays when consistently satisfied.
        """
        curv = val_metrics["curvature_loss"]
        sym = val_metrics["symmetry_loss"]
        mse = val_metrics["mse_loss"]

        # Curvature: proportional adjustment — stronger violation → stronger push
        if curv > 1e-6:
            factor = 1.0 + 0.05 * min(1.0, np.log10(max(curv, 1e-8) / 1e-6) / 3.0)
            self.curvature_weight.assign(tf.minimum(self.curvature_weight * factor, 10.0))
        else:
            self.curvature_weight.assign(tf.maximum(self.curvature_weight * 0.995, 0.05))

        # Symmetry: grow when it dominates MSE, decay otherwise
        if sym > mse * 0.5:
            self.symmetry_weight.assign(tf.minimum(self.symmetry_weight * 1.03, 5.0))
        else:
            self.symmetry_weight.assign(tf.maximum(self.symmetry_weight * 0.997, 0.01))

    def _run_epoch(self, dataset: tf.data.Dataset, training: bool) -> Dict[str, float]:
        totals = {"total": [], "mse_loss": [], "curvature_loss": [], "symmetry_loss": []}
        for batch, _groups in dataset:
            metrics = self._train_step(batch) if training else self._val_step(batch)
            for key in totals:
                totals[key].append(float(metrics[key].numpy()))
        return {key: float(np.mean(values)) for key, values in totals.items()}

    @tf.function
    def _train_step(self, batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            metrics = self._compute_loss(batch, training=True)
        grads = tape.gradient(metrics["total"], self.model.trainable_variables)
        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(grads, self.model.trainable_variables)
            if grad is not None
        )
        return metrics

    @tf.function
    def _val_step(self, batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        return self._compute_loss(batch, training=False)

    def _compute_loss(self, batch: tf.Tensor, training: bool) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(batch)[0]
        seq_len = tf.shape(batch)[1]
        num_masked = tf.maximum(
            1,
            tf.cast(tf.round(tf.cast(seq_len, tf.float32) * self.mask_ratio), tf.int32),
        )

        random_scores = tf.random.uniform([batch_size, seq_len], dtype=tf.float32)
        _, mask_indices = tf.math.top_k(random_scores, k=num_masked, sorted=False)
        mask = tf.reduce_sum(tf.one_hot(mask_indices, depth=seq_len, dtype=tf.float32), axis=1)
        mask = tf.expand_dims(mask, axis=-1)

        masked_batch = batch * (1.0 - mask)
        reconstructed = self.model.reconstruct(masked_batch, training=training)

        mse_loss = tf.reduce_sum(tf.square(reconstructed - batch) * mask) / (
            tf.reduce_sum(mask) * tf.cast(tf.shape(batch)[-1], tf.float32) + 1e-8
        )
        curvature_loss = self.curvature_loss_fn(batch, reconstructed)
        symmetry_loss = self.symmetry_loss_fn(batch, reconstructed)
        total = mse_loss + self.curvature_weight * curvature_loss + self.symmetry_weight * symmetry_loss

        return {
            "total": total,
            "mse_loss": mse_loss,
            "curvature_loss": curvature_loss,
            "symmetry_loss": symmetry_loss,
        }

    def save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.checkpoint_dir, f"ckpt-{name}")
        self.checkpoint.write(path)
        print(f"Checkpoint saved: {path}")
