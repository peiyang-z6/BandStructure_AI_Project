"""
Monte-Carlo Dropout uncertainty quantification for band gap prediction.

MC Dropout performs N forward passes with dropout enabled at inference time.
The variance across passes estimates epistemic (model) uncertainty — how much
the model itself is unsure after seeing the training data.

Key outputs per sample:
  - gap_mean:         mean predicted gap (eV) across MC samples
  - gap_std:          standard deviation (eV) — the uncertainty
  - gap_coeff_var:    normalized uncertainty (std / |mean|)
  - gap_ci95_low/high: 95% confidence interval

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras


def mc_dropout_predict(
    model: keras.Model,
    X: np.ndarray,
    n_samples: int = 50,
    batch_size: int = 16,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run MC Dropout to estimate prediction uncertainty.

    Parameters
    ----------
    model : keras.Model
        Trained model with dropout layers. Must have dropout enabled at inference.
    X : np.ndarray
        Input tensor(s), shape (n, seq, channels).
    n_samples : int
        Number of MC forward passes (50 is a good default for confidence intervals).
    batch_size : int
        Batch size for prediction.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        gap_samples   : (n_samples, n) raw predictions
        gap_mean      : (n,) mean predicted gap [eV]
        gap_std       : (n,) epistemic standard deviation [eV]
        gap_coeff_var : (n,) coefficient of variation
        gap_ci95_low  : (n,) lower 95% CI
        gap_ci95_high : (n,) upper 95% CI
    """
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]
    all_gaps = np.zeros((n_samples, n), dtype=np.float32)

    for i in range(n_samples):
        # training=True keeps dropout active at inference time
        pred = model(X, training=True)
        all_gaps[i] = np.asarray(pred["gap"] if isinstance(pred, dict) else pred).ravel()

    gap_mean = np.mean(all_gaps, axis=0)
    gap_std = np.std(all_gaps, axis=0, ddof=1)
    gap_coeff_var = np.divide(
        gap_std,
        np.maximum(np.abs(gap_mean), 1e-8),
        out=np.full_like(gap_std, np.nan),
        where=np.abs(gap_mean) > 1e-8,
    )

    # 95% CI: 1.96 * std_error (for large n_samples)
    ci_half = 1.96 * gap_std
    gap_ci95_low = gap_mean - ci_half
    gap_ci95_high = gap_mean + ci_half

    return {
        "gap_samples": all_gaps,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_coeff_var": gap_coeff_var,
        "gap_ci95_low": gap_ci95_low,
        "gap_ci95_high": gap_ci95_high,
    }


def mc_dropout_predict_with_type(
    model: keras.Model,
    X: np.ndarray,
    n_samples: int = 50,
    batch_size: int = 16,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Same as mc_dropout_predict but also returns type classification entropy."""
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]
    all_gaps = np.zeros((n_samples, n), dtype=np.float32)
    all_type_probs = np.zeros((n_samples, n, 3), dtype=np.float32)

    for i in range(n_samples):
        pred = model(X, training=True)
        all_gaps[i] = np.asarray(pred["gap"]).ravel()
        all_type_probs[i] = np.asarray(pred["type"])

    gap_mean = np.mean(all_gaps, axis=0)
    gap_std = np.std(all_gaps, axis=0, ddof=1)
    gap_coeff_var = np.divide(
        gap_std,
        np.maximum(np.abs(gap_mean), 1e-8),
        out=np.full_like(gap_std, np.nan),
        where=np.abs(gap_mean) > 1e-8,
    )
    ci_half = 1.96 * gap_std

    type_mean = np.mean(all_type_probs, axis=0)
    type_entropy = -np.sum(type_mean * np.log(np.maximum(type_mean, 1e-8)), axis=1)
    type_pred = np.argmax(type_mean, axis=1)

    return {
        "gap_samples": all_gaps,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_coeff_var": gap_coeff_var,
        "gap_ci95_low": gap_mean - ci_half,
        "gap_ci95_high": gap_mean + ci_half,
        "type_probs_mean": type_mean,
        "type_entropy": type_entropy,
        "type_pred": type_pred,
    }


def mc_calibration_check(
    y_true: np.ndarray,
    mc_result: dict[str, np.ndarray],
    tolerance_ev: float = 0.1,
) -> dict:
    """Check if true values fall within the MC 95% CI at a given tolerance."""
    ci_low = mc_result["gap_ci95_low"] - tolerance_ev
    ci_high = mc_result["gap_ci95_high"] + tolerance_ev
    in_ci = (y_true >= ci_low) & (y_true <= ci_high)
    coverage = np.mean(in_ci)

    return {
        "mc_samples": mc_result["gap_samples"].shape[0],
        "ci_coverage_95pct": float(coverage),
        "tolerance_ev": tolerance_ev,
        "mean_uncertainty_ev": float(np.mean(mc_result["gap_std"])),
        "median_uncertainty_ev": float(np.median(mc_result["gap_std"])),
        "max_uncertainty_ev": float(np.max(mc_result["gap_std"])),
    }
