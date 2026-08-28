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


def span_mask_from_starts(
    starts: tf.Tensor,
    span_length: int | tf.Tensor,
    seq_len: int | tf.Tensor,
    segment_ids: tf.Tensor | None = None,
) -> tf.Tensor:
    """Build a (B,L,1) mask; spans never cross k-path segments."""
    starts = tf.cast(starts, tf.int32)
    seq_len = tf.cast(seq_len, tf.int32)
    span_length = tf.cast(span_length, tf.int32)
    positions = tf.range(seq_len, dtype=tf.int32)[None, None, :]
    start_grid = starts[:, :, None]
    within = tf.logical_and(positions >= start_grid, positions < start_grid + span_length)
    if segment_ids is not None:
        segment_ids = tf.cast(segment_ids, tf.int32)
        start_segments = tf.gather(segment_ids, starts, batch_dims=1)
        within = tf.logical_and(
            within,
            segment_ids[:, None, :] == start_segments[:, :, None],
        )
    mask = tf.reduce_any(within, axis=1)
    return tf.cast(mask[:, :, None], tf.float32)


def random_span_mask(
    batch_size: tf.Tensor,
    seq_len: tf.Tensor,
    mask_ratio: float,
    min_span: int,
    max_span: int,
    segment_ids: tf.Tensor | None,
) -> tf.Tensor:
    """Sample approximate-ratio contiguous spans for each sequence."""
    span_length = tf.random.uniform([], min_span, max_span + 1, dtype=tf.int32)
    num_masked = tf.maximum(
        1, tf.cast(tf.round(tf.cast(seq_len, tf.float32) * mask_ratio), tf.int32)
    )
    num_spans = tf.maximum(
        1,
        tf.cast(
            tf.math.ceil(tf.cast(num_masked, tf.float32) / tf.cast(span_length, tf.float32)),
            tf.int32,
        ),
    )
    scores = tf.random.uniform([batch_size, seq_len], dtype=tf.float32)
    _, starts = tf.math.top_k(scores, k=num_spans, sorted=False)
    return span_mask_from_starts(starts, span_length, seq_len, segment_ids)


def warmup_cosine_learning_rate(
    step: int | tf.Tensor,
    total_steps: int | tf.Tensor,
    warmup_steps: int | tf.Tensor,
    base_learning_rate: float,
    min_learning_rate: float,
) -> tf.Tensor:
    """Single warmup+cosine schedule with an exact final-step floor."""
    step = tf.cast(step, tf.float32)
    total_steps = tf.cast(total_steps, tf.float32)
    warmup_steps = tf.cast(warmup_steps, tf.float32)
    base = tf.cast(base_learning_rate, tf.float32)
    floor = tf.cast(min_learning_rate, tf.float32)
    warmup_lr = base * step / tf.maximum(warmup_steps, 1.0)
    decay_steps = tf.maximum(total_steps - warmup_steps - 1.0, 1.0)
    progress = tf.clip_by_value((step - warmup_steps) / decay_steps, 0.0, 1.0)
    cosine_lr = floor + 0.5 * (base - floor) * (1.0 + tf.cos(np.pi * progress))
    return tf.where(step < warmup_steps, warmup_lr, cosine_lr)


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
        min_span: int = 5,
        max_span: int = 15,
        min_learning_rate: float = 1.0e-6,
        warmup_epochs: int = 5,
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1.0e-5,
    ):
        self.model = model
        self.base_learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.mask_ratio = mask_ratio
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        if not 1 <= self.min_span <= self.max_span:
            raise ValueError("span bounds must satisfy 1 <= min_span <= max_span")
        self.curvature_weight = tf.Variable(sign_weight, trainable=False, dtype=tf.float32)
        self.symmetry_weight = tf.Variable(consistency_weight, trainable=False, dtype=tf.float32)
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
        self.best_val_var = tf.Variable(np.inf, trainable=False, dtype=tf.float32, name="best_val")
        self.epochs_without_improvement = tf.Variable(
            0, trainable=False, dtype=tf.int64, name="epochs_without_improvement"
        )
        self.total_steps = tf.Variable(1, trainable=False, dtype=tf.int64, name="total_steps")
        self.warmup_steps = tf.Variable(0, trainable=False, dtype=tf.int64, name="warmup_steps")
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.curvature_loss_fn = BandCurvatureLoss()
        self.symmetry_loss_fn = CurvatureConsistencyLoss()

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch_var,
            best_val=self.best_val_var,
            epochs_without_improvement=self.epochs_without_improvement,
            curvature_weight=self.curvature_weight,
            symmetry_weight=self.symmetry_weight,
        )
        self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        self.val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
        steps_per_epoch = int(tf.data.experimental.cardinality(train_dataset).numpy())
        if steps_per_epoch <= 0:
            raise ValueError("train dataset cardinality must be finite and positive")
        self.total_steps.assign(max(1, steps_per_epoch * int(epochs)))
        self.warmup_steps.assign(min(int(epochs), self.warmup_epochs) * steps_per_epoch)
        start_epoch = int(self.epoch_var.numpy()) + 1
        if start_epoch > 1:
            print(f"Resuming SSL from epoch {start_epoch}")
        for epoch in range(start_epoch, epochs + 1):
            start = time.time()
            train_metrics = self._run_epoch(train_dataset, training=True)
            val_metrics = self._run_epoch(val_dataset, training=False)
            self._adapt_physics_weights(val_metrics)
            current_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))

            with self.train_writer.as_default():
                for key, value in train_metrics.items():
                    tf.summary.scalar(key, value, step=epoch)
                tf.summary.scalar("learning_rate", current_lr, step=epoch)
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
                f"mmae={val_metrics['masked_mae']:.5f} "
                f"mask={val_metrics['mask_fraction']:.3f} "
                f"curv={val_metrics['curvature_loss']:.5f} "
                f"sym={val_metrics['symmetry_loss']:.5f} "
                f"lr={current_lr:.3e} "
                f"cw={float(self.curvature_weight.numpy()):.3f} "
                f"sw={float(self.symmetry_weight.numpy()):.3f} "
                f"time={time.time() - start:.1f}s"
            )

            improved = val_metrics["total"] < (
                float(self.best_val_var.numpy()) - self.early_stopping_min_delta
            )
            if improved:
                self.best_val_var.assign(val_metrics["total"])
                self.epochs_without_improvement.assign(0)
                self.save_checkpoint("best")
            else:
                self.epochs_without_improvement.assign_add(1)

            self.epoch_var.assign(epoch)
            self.save_checkpoint("last")
            if epoch % 10 == 0:
                self.save_checkpoint(str(epoch))
            if (
                self.early_stopping_patience > 0
                and int(self.epochs_without_improvement.numpy()) >= self.early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch}: no improvement for {self.early_stopping_patience} epochs")
                break

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
        totals = {
            "total": [],
            "mse_loss": [],
            "masked_mae": [],
            "mask_fraction": [],
            "curvature_loss": [],
            "symmetry_loss": [],
        }
        for data in dataset:
            if len(data) == 3:
                batch, _groups, segment_ids = data
            else:
                batch, _groups = data
                segment_ids = None
            metrics = (
                self._train_step(batch, segment_ids)
                if training
                else self._val_step(batch, segment_ids)
            )
            for key in totals:
                totals[key].append(float(metrics[key].numpy()))
        return {key: float(np.mean(values)) for key, values in totals.items()}

    @tf.function
    def _train_step(self, batch: tf.Tensor, segment_ids: tf.Tensor | None) -> Dict[str, tf.Tensor]:
        learning_rate = warmup_cosine_learning_rate(
            self.optimizer.iterations,
            self.total_steps,
            self.warmup_steps,
            self.base_learning_rate,
            self.min_learning_rate,
        )
        self.optimizer.learning_rate.assign(learning_rate)
        with tf.GradientTape() as tape:
            metrics = self._compute_loss(batch, training=True, segment_ids=segment_ids)
        grads = tape.gradient(metrics["total"], self.model.trainable_variables)
        pairs = [
            (grad, var)
            for grad, var in zip(grads, self.model.trainable_variables)
            if grad is not None
        ]
        if self.gradient_clip_norm > 0.0 and pairs:
            clipped, _global_norm = tf.clip_by_global_norm(
                [grad for grad, _var in pairs], self.gradient_clip_norm
            )
            pairs = list(zip(clipped, [var for _grad, var in pairs]))
        self.optimizer.apply_gradients(pairs)
        return metrics

    @tf.function
    def _val_step(self, batch: tf.Tensor, segment_ids: tf.Tensor | None) -> Dict[str, tf.Tensor]:
        return self._compute_loss(batch, training=False, segment_ids=segment_ids)

    def _compute_loss(
        self,
        batch: tf.Tensor,
        training: bool,
        segment_ids: tf.Tensor | None,
    ) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(batch)[0]
        seq_len = tf.shape(batch)[1]
        mask = random_span_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            mask_ratio=self.mask_ratio,
            min_span=self.min_span,
            max_span=self.max_span,
            segment_ids=segment_ids,
        )

        masked_batch = self.model.apply_mask_token(batch, mask)
        reconstructed = self.model.reconstruct(masked_batch, training=training)

        mse_loss = tf.reduce_sum(tf.square(reconstructed - batch) * mask) / (
            tf.reduce_sum(mask) * tf.cast(tf.shape(batch)[-1], tf.float32) + 1e-8
        )
        masked_mae = tf.reduce_sum(tf.abs(reconstructed - batch) * mask) / (
            tf.reduce_sum(mask) * tf.cast(tf.shape(batch)[-1], tf.float32) + 1e-8
        )
        mask_fraction = tf.reduce_mean(mask)
        curvature_loss = self.curvature_loss_fn.call(
            batch, reconstructed, segment_ids=segment_ids
        )
        symmetry_loss = self.symmetry_loss_fn.call(
            batch, reconstructed, segment_ids=segment_ids
        )
        total = mse_loss + self.curvature_weight * curvature_loss + self.symmetry_weight * symmetry_loss

        return {
            "total": total,
            "mse_loss": mse_loss,
            "masked_mae": masked_mae,
            "mask_fraction": mask_fraction,
            "curvature_loss": curvature_loss,
            "symmetry_loss": symmetry_loss,
        }

    def restore_checkpoint(self, name: str = "last") -> bool:
        path = os.path.join(self.checkpoint_dir, f"ckpt-{name}")
        if not os.path.exists(path + ".index"):
            print(f"No checkpoint to resume: {path}")
            return False
        self.checkpoint.restore(path).expect_partial()
        print(f"Restored checkpoint: {path} (epoch={int(self.epoch_var.numpy())})")
        return True

    def save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.checkpoint_dir, f"ckpt-{name}")
        self.checkpoint.write(path)
        print(f"Checkpoint saved: {path}")
