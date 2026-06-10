"""
SSL Training Main Script
========================

Default path: Masked Band Modeling (MBM) on the cleaned OOD tensor dataset.

Input tensor convention from data_cache/ood_tensors/band_tensors_ood_split.npz:
    X_*: (N, 2, seq_len, 3)
        band slot 0: VBM-like band
        band slot 1: CBM-like band
        channel 0: energy
        channel 1: curvature
        channel 2: normalized distance to the band extremum

The model sees this as a sequence:
    (N, seq_len, 6) = [vbm_E, vbm_curv, vbm_k_dist, cbm_E, cbm_curv, cbm_k_dist]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.band_structure_dataset import VirtualStrainAugmentation
from src.engine.ssl_trainer import MBMTrainer as PhysicsMBMTrainer
from src.models.band_structure_encoder import SSLEncoder


def configure_tensorflow_runtime(require_gpu: bool = False) -> None:
    """Print runtime diagnostics and enable memory growth when GPUs exist."""
    print("TensorFlow:", tf.__version__)
    print("CUDA build:", tf.test.is_built_with_cuda())
    print("Build info:", dict(tf.sysconfig.get_build_info()))

    gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"[WARN] Could not set memory growth for {gpu}: {exc}")

    if require_gpu and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU device is available. On native Windows, "
            "TensorFlow >= 2.11 does not support NVIDIA CUDA. Use WSL2/Linux "
            "with a CUDA-enabled TensorFlow build, or a DirectML-specific setup."
        )


def load_ood_tensor_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_test"].astype(np.float32)
    groups_train = data["groups_train"].astype(np.int32)
    groups_val = data["groups_test"].astype(np.int32)

    # (N, 2, K, C) -> (N, K, 2*C)
    X_train = np.transpose(X_train, (0, 2, 1, 3)).reshape(X_train.shape[0], X_train.shape[2], -1)
    X_val = np.transpose(X_val, (0, 2, 1, 3)).reshape(X_val.shape[0], X_val.shape[2], -1)

    return X_train, X_val, groups_train, groups_val


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def make_tensor_dataset(
    X: np.ndarray,
    groups: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    strain_scale: float = 0.01,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, groups))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    if augment:
        strain = VirtualStrainAugmentation(strain_scale=strain_scale)

        def apply_strain(sequence, group):
            return strain(sequence), group

        dataset = dataset.map(apply_strain, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def run_tensor_mbm(args: argparse.Namespace) -> None:
    X_train, X_val, groups_train, groups_val = load_ood_tensor_data(args.tensor_npz)
    X_train, X_val, norm_stats = standardize_features(X_train, X_val)

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]

    print("Loaded OOD tensors:")
    print(f"  train: {X_train.shape}, groups={len(set(groups_train.tolist()))}")
    print(f"  val:   {X_val.shape}, groups={len(set(groups_val.tolist()))}")
    print(f"  group overlap: {set(groups_train.tolist()).intersection(set(groups_val.tolist()))}")

    train_ds = make_tensor_dataset(
        X_train,
        groups_train,
        args.batch_size,
        shuffle=True,
        augment=not args.disable_strain_augmentation,
        strain_scale=args.strain_scale,
    )
    val_ds = make_tensor_dataset(X_val, groups_val, args.batch_size, shuffle=False)

    model = SSLEncoder(
        num_features=num_features,
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dff=args.dff,
        projection_dim=args.projection_dim,
        dropout_rate=args.dropout,
    )
    model(tf.zeros([1, seq_len, num_features], dtype=tf.float32), training=False)

    trainer = PhysicsMBMTrainer(
        model=model,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        sign_weight=args.sign_weight,
        consistency_weight=args.consistency_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trainer.train(train_ds, val_ds, epochs=args.epochs)

    os.makedirs(args.model_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_dir, f"ssl_mbm_final_epoch{args.epochs}.keras")
    model.save(final_model_path)

    best_checkpoint = os.path.join(args.checkpoint_dir, "ckpt-best")
    if os.path.exists(best_checkpoint + ".index"):
        trainer.checkpoint.restore(best_checkpoint).expect_partial()
        print(f"Restored best checkpoint: {best_checkpoint}")

    model_path = os.path.join(args.model_dir, "ssl_mbm_pretrained.keras")
    model.save(model_path)
    with open(os.path.join(args.model_dir, "ssl_mbm_norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Final model saved to {final_model_path}")
    print(f"Model saved to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Band structure SSL pre-training")
    parser.add_argument("--tensor-npz", default="./data_cache/ood_tensors/band_tensors_ood_split.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--sign-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dff", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--strain-scale", type=float, default=0.01)
    parser.add_argument("--disable-strain-augmentation", action="store_true")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/ssl_mbm")
    parser.add_argument("--log-dir", default="./logs/ssl_mbm")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    print("=" * 60)
    print("Band Structure SSL Pre-training")
    print("=" * 60)

    args = parse_args()
    if not (0.15 <= args.mask_ratio <= 0.30):
        raise ValueError("--mask-ratio must be between 0.15 and 0.30 for MBM")

    configure_tensorflow_runtime(require_gpu=args.require_gpu)

    if not os.path.exists(args.tensor_npz):
        raise FileNotFoundError(
            f"Tensor dataset not found: {args.tensor_npz}. "
            "Run scripts/build_ood_tensors.py first."
        )
    run_tensor_mbm(args)


if __name__ == "__main__":
    main()
