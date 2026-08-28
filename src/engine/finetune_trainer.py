"""
Fine-tuning strategy helpers for auditability.

The active script freezes early SSL transformer blocks and fine-tunes the later
encoder blocks plus predictor heads. The report emitted here is intended for
micro-course defense and third-party review.
"""
from __future__ import annotations

import json
import os
from typing import Dict

from tensorflow import keras


def freeze_encoder_layers(ssl_encoder: keras.Model, freeze_layers: int) -> Dict[str, object]:
    if hasattr(ssl_encoder, "projection_head"):
        ssl_encoder.projection_head.trainable = False
    if hasattr(ssl_encoder, "reconstruction_head"):
        ssl_encoder.reconstruction_head.trainable = False

    base = ssl_encoder.encoder
    total_layers = len(base.transformer_layers)
    freeze_to = min(freeze_layers, total_layers)
    frozen = []
    trainable = []

    for idx in range(total_layers):
        is_trainable = idx >= freeze_to
        base.transformer_layers[idx].trainable = is_trainable
        base.ln_attn[idx].trainable = is_trainable
        base.ln_ffn[idx].trainable = is_trainable
        base.add_attn[idx].trainable = is_trainable
        base.add_ffn[idx].trainable = is_trainable
        base.ffn_dense1[idx].trainable = is_trainable
        base.ffn_dense2[idx].trainable = is_trainable
        base.ffn_dropout[idx].trainable = is_trainable
        (trainable if is_trainable else frozen).append(f"transformer_block_{idx}")

    return {
        "total_transformer_layers": total_layers,
        "freeze_layers": freeze_to,
        "frozen_layers": frozen,
        "trainable_encoder_layers": trainable,
        "frozen_heads": ["projection_head", "reconstruction_head"],
        "trainable_heads": ["extremum_expected_gap_head", "type_head"],
        "learning_rate_policy": {
            "encoder_backbone": "low effective LR / frozen early layers",
            "predictor_heads": "main optimizer LR with warmup + cosine decay",
        },
    }


def write_finetune_strategy_report(
    output_path: str,
    freeze_info: Dict[str, object],
    class_weights: Dict[int, float],
    type_weight: float,
    learning_rate: float,
    encoder_learning_rate: float | None = None,
    topology_weight: float | None = None,
    entropy_weight: float | None = None,
    extremum_weight: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "freeze_strategy": freeze_info,
        "optimizer_strategy": {
            "head_learning_rate": learning_rate,
            "encoder_effective_learning_rate": encoder_learning_rate,
            "schedule": "linear warmup followed by cosine decay",
            "note": (
                "Early SSL blocks are frozen. Later encoder blocks receive gradient-scaled "
                "updates while randomly initialized predictor heads use the larger optimizer LR."
            ),
        },
        "gap_head": {
            "name": "extremum expected-value head",
            "logic": (
                "The encoder keeps full sequence features. A Conv1D head predicts VBM/CBM "
                "k-point probability distributions with a learned temperature, denormalizes "
                "the energy channels to eV, and returns expected(E_CBM) - expected(E_VBM)."
            ),
        },
        "topology_injection": {
            "type_head_priors": ["abs_expected_k_distance", "probability_overlap", "symmetric_kl"],
            "topology_loss_weight": topology_weight,
            "entropy_sharpening_weight": entropy_weight,
            "extremum_peak_supervision_weight": extremum_weight,
        },
        "classification_loss": {
            "name": "categorical crossentropy with class sample weights",
            "formula": "sum_c class_weight_c * [-y_c log(p_c)]",
            "class_weights": class_weights,
            "type_loss_weight": type_weight,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
