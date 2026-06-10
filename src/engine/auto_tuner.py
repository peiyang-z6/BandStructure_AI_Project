"""
Optuna objective helpers with physics-validator pruning.

This module is intentionally optional: the main training pipeline does not run
hyperparameter search, but any Optuna study using this objective must reject
models that pass mathematical loss while violating physical constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from src.utils.physics_validator import PhysicsValidator


def _extract_features(data: Any) -> Any:
    if hasattr(data, "take"):
        batches = []
        for batch in data:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            batches.append(x)
        if not batches:
            raise ValueError("validation_data dataset is empty")
        try:
            import tensorflow as tf

            return tf.concat(batches, axis=0)
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to extract dataset features") from exc
    if isinstance(data, (tuple, list)):
        return data[0]
    return data


def _sequence_to_band_tensor(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 4:
        return sequence
    if sequence.ndim != 3 or sequence.shape[-1] % 2 != 0:
        raise ValueError(f"Cannot convert reconstructed sequence shape to band tensor: {sequence.shape}")
    channels_per_band = sequence.shape[-1] // 2
    return sequence.reshape(sequence.shape[0], sequence.shape[1], 2, channels_per_band).transpose(0, 2, 1, 3)


def _tensor_gap_from_extrema(tensors: np.ndarray) -> np.ndarray:
    tensors = np.asarray(tensors, dtype=np.float32)
    return np.min(tensors[:, 1, :, 0], axis=1) - np.max(tensors[:, 0, :, 0], axis=1)


def _gap_identity_score(predicted_gap: np.ndarray, reference_tensors: np.ndarray, tolerance: float = 0.5) -> Dict[str, float]:
    predicted_gap = np.asarray(predicted_gap, dtype=np.float32).reshape(-1)
    tensor_gap = _tensor_gap_from_extrema(reference_tensors)
    residual = predicted_gap - tensor_gap
    violation_rate = float(np.mean(np.abs(residual) > tolerance))
    return {
        "gap_identity_large_residual_rate": violation_rate,
        "gap_identity_mae": float(np.mean(np.abs(residual))),
        "gap_identity_max_abs": float(np.max(np.abs(residual))),
        "gap_identity_score": float(max(0.0, 1.0 - violation_rate)),
    }


@dataclass
class AutoTuner:
    """Small Optuna wrapper that closes each trial with physics validation."""

    build_model_fn: Callable[[Any], Any]
    train_data: Any
    validation_data: Any
    validation_tensors: np.ndarray
    validation_gaps: np.ndarray | None = None
    epochs: int = 80
    physics_threshold: float = 0.95

    def objective(self, trial: Any) -> float:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna is required for AutoTuner.objective") from exc

        model = self.build_model_fn(trial)
        history = model.fit(
            self.train_data,
            validation_data=self.validation_data,
            epochs=self.epochs,
            verbose=0,
        )

        validation_features = _extract_features(self.validation_data)
        predictions = model.predict(self.validation_data, verbose=0)
        if isinstance(predictions, dict):
            predicted_gap = predictions.get("gap")
            predicted_bands = predictions.get("bands")
            if predicted_bands is None:
                predicted_bands = predictions.get("reconstruction")
        elif isinstance(predictions, (list, tuple)):
            predicted_gap = predictions[0]
            predicted_bands = predictions[1] if len(predictions) > 1 else None
        else:
            predicted_gap = predictions
            predicted_bands = None

        if predicted_bands is None:
            if hasattr(model, "encoder") and hasattr(model.encoder, "reconstruct"):
                predicted_bands = model.encoder.reconstruct(validation_features, training=False).numpy()
            elif hasattr(model, "reconstruct"):
                predicted_bands = model.reconstruct(validation_features, training=False).numpy()
            else:
                raise RuntimeError(
                    "Physics pruning requires predicted bands or a reconstruct-capable encoder; "
                    "refusing to validate against ground-truth tensors as fallback."
                )

        predicted_bands = _sequence_to_band_tensor(predicted_bands)

        if predicted_gap is None:
            predicted_gap = _tensor_gap_from_extrema(predicted_bands)

        validator = PhysicsValidator()
        physics_info = validator.validate(_tensor_gap_from_extrema(predicted_bands), predicted_bands)
        identity_info = _gap_identity_score(predicted_gap, self.validation_tensors)
        physics_score = min(physics_info["physics_score"], identity_info["gap_identity_score"])
        physics_info.update(identity_info)
        physics_info["physics_score"] = physics_score
        trial.set_user_attr("physics_score", physics_score)
        trial.set_user_attr("physics_validation", physics_info)

        if physics_score < self.physics_threshold:
            raise optuna.TrialPruned("Physics violation too high")

        val_gap_mae = history.history.get("val_gap_mae")
        if val_gap_mae:
            return float(np.min(val_gap_mae))

        val_loss = history.history.get("val_loss")
        if not val_loss:
            raise RuntimeError("No validation metric found in trial history")
        return float(np.min(val_loss))


def objective(
    trial: Any,
    build_model_fn: Callable[[Any], Any],
    train_data: Any,
    validation_data: Any,
    validation_tensors: np.ndarray,
    epochs: int = 80,
    physics_threshold: float = 0.95,
) -> float:
    tuner = AutoTuner(
        build_model_fn=build_model_fn,
        train_data=train_data,
        validation_data=validation_data,
        validation_tensors=validation_tensors,
        epochs=epochs,
        physics_threshold=physics_threshold,
    )
    return tuner.objective(trial)
