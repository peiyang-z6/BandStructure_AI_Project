"""
SSL trainer for Masked Band Modeling with physics-loss monitoring.
"""
from __future__ import annotations

import json
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
    seed: tf.Tensor | None = None,
) -> tf.Tensor:
    """Sample segment-safe spans with an enforced 15–30% actual ratio."""
    batch_size = tf.cast(batch_size, tf.int32)
    seq_len = tf.cast(seq_len, tf.int32)
    if seed is None:
        span_length = tf.random.uniform(
            [], min_span, max_span + 1, dtype=tf.int32
        )
        scores = tf.random.uniform([batch_size, seq_len], dtype=tf.float32)
    else:
        split_seeds = tf.random.experimental.stateless_split(
            tf.cast(seed, tf.int32),
            2,
        )
        span_length = tf.random.stateless_uniform(
            [],
            seed=split_seeds[0],
            minval=min_span,
            maxval=max_span + 1,
            dtype=tf.int32,
        )
        scores = tf.random.stateless_uniform(
            [batch_size, seq_len],
            seed=split_seeds[1],
            dtype=tf.float32,
        )

    # Every k position is a candidate span start.  The cumulative union lets us
    # choose, per sample, the closest coverage to mask_ratio without overlapping
    # spans or short terminal segments silently dropping below the contract.
    starts = tf.argsort(scores, axis=1, direction="DESCENDING")
    positions = tf.range(seq_len, dtype=tf.int32)[None, None, :]
    start_grid = starts[:, :, None]
    within = tf.logical_and(
        positions >= start_grid,
        positions < start_grid + span_length,
    )
    if segment_ids is not None:
        segment_ids = tf.cast(segment_ids, tf.int32)
        start_segments = tf.gather(segment_ids, starts, batch_dims=1)
        within = tf.logical_and(
            within,
            segment_ids[:, None, :] == start_segments[:, :, None],
        )

    cumulative = tf.cumsum(tf.cast(within, tf.int32), axis=1) > 0
    covered = tf.reduce_sum(tf.cast(cumulative, tf.int32), axis=2)
    minimum = tf.cast(
        tf.math.ceil(tf.cast(seq_len, tf.float32) * 0.15),
        tf.int32,
    )
    maximum = tf.cast(
        tf.math.floor(tf.cast(seq_len, tf.float32) * 0.30),
        tf.int32,
    )
    target = tf.cast(
        tf.round(tf.cast(seq_len, tf.float32) * float(mask_ratio)),
        tf.int32,
    )
    target = tf.clip_by_value(target, minimum, maximum)
    valid = tf.logical_and(covered >= minimum, covered <= maximum)
    candidate_index = tf.range(seq_len, dtype=tf.int32)[None, :]
    distance = tf.abs(covered - target) * (seq_len + 1) + candidate_index
    invalid_penalty = tf.fill(tf.shape(distance), seq_len * seq_len * 4)
    choice = tf.argmin(
        tf.where(valid, distance, invalid_penalty),
        axis=1,
        output_type=tf.int32,
    )
    mask = tf.gather(cumulative, choice, axis=1, batch_dims=1)
    return tf.cast(mask[:, :, None], tf.float32)


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
        validation_mask_seed: int = 42,
    ):
        self.model = model
        self.base_learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.validation_mask_seed = int(validation_mask_seed)
        self.validation_batch_index = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            name="validation_batch_index",
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.mask_ratio = mask_ratio
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        if not 1 <= self.min_span <= self.max_span:
            raise ValueError("span bounds must satisfy 1 <= min_span <= max_span")
        self.curvature_weight = tf.Variable(sign_weight, trainable=False, dtype=tf.float32)
        self.symmetry_weight = tf.Variable(consistency_weight, trainable=False, dtype=tf.float32)
        self.symmetry_adaptation_enabled = float(consistency_weight) > 0.0
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
        self.best_val_var = tf.Variable(np.inf, trainable=False, dtype=tf.float32, name="best_val")
        self.epochs_without_improvement = tf.Variable(
            0, trainable=False, dtype=tf.int64, name="epochs_without_improvement"
        )
        self.total_steps = tf.Variable(1, trainable=False, dtype=tf.int64, name="total_steps")
        self.warmup_steps = tf.Variable(0, trainable=False, dtype=tf.int64, name="warmup_steps")
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.history_path = os.path.join(log_dir, "ssl_history.json")
        self.history = {
            "schema_version": 1,
            "selection_monitor": "val_total",
            "validation_mask_seed": self.validation_mask_seed,
            "epochs": [],
        }
        if os.path.isfile(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as handle:
                existing_history = json.load(handle)
            if isinstance(existing_history, dict):
                self.history.update(existing_history)

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
            selection_weights = {
                "curvature": float(self.curvature_weight.numpy()),
                "symmetry": float(self.symmetry_weight.numpy()),
            }
            self._adapt_physics_weights(val_metrics)
            post_adaptation_weights = {
                "curvature": float(self.curvature_weight.numpy()),
                "symmetry": float(self.symmetry_weight.numpy()),
            }
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
            self.epoch_var.assign(epoch)
            if improved:
                self.best_val_var.assign(val_metrics["total"])
                self.epochs_without_improvement.assign(0)
                self.save_checkpoint("best")
            else:
                self.epochs_without_improvement.assign_add(1)

            self.save_checkpoint("last")
            if epoch % 10 == 0:
                self.save_checkpoint(str(epoch))
            self.history["epochs"] = [
                item
                for item in self.history.get("epochs", [])
                if int(item.get("epoch", -1)) < epoch
            ]
            self.history["epochs"].append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "learning_rate": current_lr,
                    "selection_weights": selection_weights,
                    "post_adaptation_weights": post_adaptation_weights,
                    "curvature_weight": post_adaptation_weights["curvature"],
                    "symmetry_weight": post_adaptation_weights["symmetry"],
                    "improved": bool(improved),
                    "duration_seconds": float(time.time() - start),
                }
            )
            self._write_history()
            if (
                self.early_stopping_patience > 0
                and int(self.epochs_without_improvement.numpy()) >= self.early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch}: no improvement for {self.early_stopping_patience} epochs")
                break

    def _write_history(self) -> None:
        temporary = self.history_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.history_path)

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

        # Magnitude consistency may be hard-disabled because standardized
        # curvature and index-space energy differences do not share units.
        if not self.symmetry_adaptation_enabled:
            self.symmetry_weight.assign(0.0)
        elif sym > mse * 0.5:
            self.symmetry_weight.assign(tf.minimum(self.symmetry_weight * 1.03, 5.0))
        else:
            self.symmetry_weight.assign(tf.maximum(self.symmetry_weight * 0.997, 0.01))

    def _run_epoch(self, dataset: tf.data.Dataset, training: bool) -> Dict[str, float]:
        keys = (
            "total",
            "mse_loss",
            "masked_mae",
            "mask_fraction",
            "curvature_loss",
            "symmetry_loss",
        )
        numerators = {key: 0.0 for key in keys}
        denominators = {key: 0.0 for key in keys}
        for batch_index, data in enumerate(dataset):
            if len(data) == 3:
                batch, _groups, segment_ids = data
            else:
                batch, _groups = data
                segment_ids = None
            if training:
                metrics = self._train_step(batch, segment_ids)
            else:
                if hasattr(self, "validation_batch_index"):
                    self.validation_batch_index.assign(batch_index)
                metrics = self._val_step(batch, segment_ids)
            sample_count = float(
                metrics.get("sample_count", tf.cast(tf.shape(batch)[0], tf.float32)).numpy()
            )
            masked_elements = float(
                metrics.get("masked_elements", tf.constant(sample_count, tf.float32)).numpy()
            )
            position_count = float(
                metrics.get("position_count", tf.constant(sample_count, tf.float32)).numpy()
            )
            weights = {
                "total": sample_count,
                "mse_loss": masked_elements,
                "masked_mae": masked_elements,
                "mask_fraction": position_count,
                "curvature_loss": sample_count,
                "symmetry_loss": sample_count,
            }
            for key in keys:
                value = float(metrics[key].numpy())
                numerators[key] += value * weights[key]
                denominators[key] += weights[key]

        results = {
            key: float(numerators[key] / max(denominators[key], 1e-12))
            for key in keys
        }
        if hasattr(self, "curvature_weight") and hasattr(self, "symmetry_weight"):
            results["total"] = float(
                results["mse_loss"]
                + float(self.curvature_weight.numpy()) * results["curvature_loss"]
                + float(self.symmetry_weight.numpy()) * results["symmetry_loss"]
            )
        return results

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
        mask_seed = None
        if not training:
            mask_seed = tf.stack(
                [
                    tf.cast(self.validation_mask_seed, tf.int32),
                    tf.cast(self.validation_batch_index, tf.int32),
                ]
            )
        mask = random_span_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            mask_ratio=self.mask_ratio,
            min_span=self.min_span,
            max_span=self.max_span,
            segment_ids=segment_ids,
            seed=mask_seed,
        )

        masked_batch = self.model.apply_mask_token(batch, mask)
        reconstructed = self.model.reconstruct(masked_batch, training=training)

        masked_positions = tf.reduce_sum(mask)
        feature_count = tf.cast(tf.shape(batch)[-1], tf.float32)
        masked_elements = masked_positions * feature_count
        mse_loss = tf.reduce_sum(tf.square(reconstructed - batch) * mask) / (
            masked_elements + 1e-8
        )
        masked_mae = tf.reduce_sum(tf.abs(reconstructed - batch) * mask) / (
            masked_elements + 1e-8
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
            "sample_count": tf.cast(batch_size, tf.float32),
            "masked_elements": masked_elements,
            "position_count": tf.cast(batch_size * seq_len, tf.float32),
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
