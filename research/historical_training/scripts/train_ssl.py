"""
SSL Training Main Script
========================

Default path: Masked Band Modeling (MBM) on the cleaned OOD tensor dataset.

Input tensor convention from data/processed/materials_project/ood_tensors/band_tensors_ood_split.npz:
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
import shutil
import uuid
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.band_structure_dataset import VirtualStrainAugmentation
from src.engine.ssl_trainer import MBMTrainer as PhysicsMBMTrainer
from src.models.band_structure_encoder import SSLEncoder
from src.data.ood_tensor_builder import build_group_validation_split
from src.utils import assert_tensor_on_gpu
from src.utils.selection_manifest import (validate_raw_tensor_contract, p2_file_ref,
    p2_code_refs, P2_SSL_CODE, validate_p2_mask_configuration, validate_p2_ssl_history)


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


def load_ood_tensor_data(
    npz_path: str,
    validation_size: float = 0.15,
    random_state: int = 42,
    *, return_allocation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X_outer_train = data["X_train"].astype(np.float32)
    groups_outer_train = data["groups_train"].astype(np.int32)
    segment_outer_train = (
        data["segment_ids_train"].astype(np.int32)
        if "segment_ids_train" in data
        else np.zeros((len(X_outer_train), X_outer_train.shape[2]), dtype=np.int32)
    )
    type_labels = data["y_type_train"].astype(np.int32) if "y_type_train" in data else None
    fit_idx, val_idx = build_group_validation_split(
        groups_outer_train,
        validation_size=validation_size,
        random_state=random_state,
        class_labels=type_labels,
    )
    X_train = X_outer_train[fit_idx]
    X_val = X_outer_train[val_idx]
    groups_train = groups_outer_train[fit_idx]
    groups_val = groups_outer_train[val_idx]
    segments_train = segment_outer_train[fit_idx]
    segments_val = segment_outer_train[val_idx]

    # (N, 2, K, C) -> (N, K, 2*C)
    X_train = np.transpose(X_train, (0, 2, 1, 3)).reshape(X_train.shape[0], X_train.shape[2], -1)
    X_val = np.transpose(X_val, (0, 2, 1, 3)).reshape(X_val.shape[0], X_val.shape[2], -1)

    result = (X_train, X_val, groups_train, groups_val, segments_train, segments_val)
    if return_allocation:
        ids = data['material_ids_train'].tolist()
        allocation = {'material_ids': ids, 'groups': groups_outer_train.tolist(),
                      'fit_material_ids': [ids[i] for i in fit_idx],
                      'fit_groups': groups_train.tolist(),
                      'selection_material_ids': [ids[i] for i in val_idx],
                      'selection_groups': groups_val.tolist()}
        data.close()
        return (*result, allocation)
    data.close()
    return result


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
    segment_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    strain_scale: float = 0.01,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, groups, segment_ids))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    if augment:
        strain = VirtualStrainAugmentation(strain_scale=strain_scale)

        def apply_strain(sequence, group, segments):
            return strain(sequence), group, segments

        dataset = dataset.map(apply_strain, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _observed_model_state_sha(model):
    digest = hashlib.sha256()
    for variable in model.weights:
        value = np.ascontiguousarray(variable.numpy())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_tensor_mbm(args: argparse.Namespace) -> None:
    if args.resume:
        raise ValueError('New anchor production requires a fresh run; historical resume cannot be certified')
    if not args.tensor_contract:
        raise ValueError('Explicit --tensor-contract is required; legacy inputs remain blocked')
    if type(args.epochs) is not int or args.epochs < 1 or (args.test_scope and args.epochs > 2):
        raise ValueError('Test scope is bounded to 1-2 epochs')
    directories = [Path(p).resolve() for p in (args.model_dir, args.checkpoint_dir, args.log_dir)]
    for directory in directories:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError('Refusing nonempty SSL output directory')
    args.require_gpu = not args.test_scope or args.require_gpu
    configure_tensorflow_runtime(require_gpu=args.require_gpu)
    raw_contract = validate_raw_tensor_contract(
        args.tensor_contract, tensor_path=args.tensor_npz, split='train', allow_test_scope=args.test_scope,
        required_scope='test' if args.test_scope else None)
    if args.test_scope and raw_contract['scope'] != 'test':
        raise ValueError('CPU test scope requires a bounded synthetic tensor contract')
    validate_p2_mask_configuration(raw_contract['feature_schema']['seq_len'],
                                   args.min_span, args.max_span, args.mask_ratio)
    run_id = uuid.uuid4().hex
    code_before = p2_code_refs(P2_SSL_CODE)
    X_train, X_val, groups_train, groups_val, segments_train, segments_val, allocation = load_ood_tensor_data(
        args.tensor_npz,
        validation_size=args.validation_size,
        random_state=args.random_state,
        return_allocation=True,
    )
    validate_raw_tensor_contract(args.tensor_contract, tensor_path=args.tensor_npz,
                                 split='train', allow_test_scope=args.test_scope,
                                 expected_ids=allocation['material_ids'], expected_groups=allocation['groups'])
    X_train, X_val, norm_stats = standardize_features(X_train, X_val)

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]

    print("Loaded outer-training tensors (outer OOD test remains untouched):")
    print(f"  inner train: {X_train.shape}, groups={len(set(groups_train.tolist()))}")
    print(f"  inner val:   {X_val.shape}, groups={len(set(groups_val.tolist()))}")
    print(f"  group overlap: {set(groups_train.tolist()).intersection(set(groups_val.tolist()))}")

    train_ds = make_tensor_dataset(
        X_train,
        groups_train,
        segments_train,
        args.batch_size,
        shuffle=True,
        augment=not args.disable_strain_augmentation,
        strain_scale=args.strain_scale,
    )
    val_ds = make_tensor_dataset(
        X_val, groups_val, segments_val, args.batch_size, shuffle=False
    )

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
    probe_input = tf.zeros([1, seq_len, num_features], dtype=tf.float32)
    probe_embedding = model(probe_input, training=False)
    probe_reconstruction = model.reconstruct(probe_input, training=False)
    assert_tensor_on_gpu(
        probe_embedding,
        "SSL encoder forward",
        require_gpu=args.require_gpu,
    )
    assert_tensor_on_gpu(
        probe_reconstruction,
        "SSL reconstruction forward",
        require_gpu=args.require_gpu,
    )

    trainer = PhysicsMBMTrainer(
        model=model,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        sign_weight=args.sign_weight,
        consistency_weight=args.consistency_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        min_span=args.min_span,
        max_span=args.max_span,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        validation_mask_seed=args.random_state,
    )
    with tf.GradientTape() as tape:
        probe_metrics = trainer._compute_loss(
            tf.convert_to_tensor(X_train[:1]), training=False,
            segment_ids=tf.convert_to_tensor(segments_train[:1]))
    gradients = [g for g in tape.gradient(probe_metrics['total'], model.trainable_variables) if g is not None]
    if not gradients:
        raise ValueError('No actual MBM backward gradients')
    backward_devices = []
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, 'MBM backward probe is not finite')
        backward_devices.append(assert_tensor_on_gpu(gradient, 'SSL backward', require_gpu=args.require_gpu))
    initial_state_sha = _observed_model_state_sha(model)
    trainer.train(train_ds, val_ds, epochs=args.epochs)
    # Fail before model, origin or anchor publication; retain real failed history.
    validate_p2_ssl_history(json.loads(Path(trainer.history_path).read_text(encoding='utf-8')),
                            expected_consistency_weight=args.consistency_weight)
    last_state_sha = _observed_model_state_sha(model)
    if (p2_file_ref(args.tensor_npz) != raw_contract['tensor_ref']
            or p2_file_ref(args.tensor_contract) != raw_contract['contract_ref']
            or p2_code_refs(P2_SSL_CODE) != code_before):
        raise ValueError('Source/contract/code changed during SSL training')

    os.makedirs(args.model_dir, exist_ok=True)
    completed_epochs = int(trainer.epoch_var.numpy())
    optimizer_iterations = int(trainer.optimizer.iterations.numpy())
    final_model_path = os.path.join(args.model_dir, f"ssl_mbm_final_epoch{completed_epochs}.keras")
    model.save(final_model_path)

    best_checkpoint = os.path.join(args.checkpoint_dir, 'ckpt-best')
    if not os.path.isfile(best_checkpoint + '.index'):
        raise ValueError('No actual best checkpoint; anchor remains blocked')
    trainer.checkpoint.restore(best_checkpoint).expect_partial()
    best_epoch = int(trainer.epoch_var.numpy())
    best_state_sha = _observed_model_state_sha(model)
    best_model_path = os.path.join(args.model_dir, 'ssl_mbm_best.keras')
    model.save(best_model_path)
    model_path = os.path.join(args.model_dir, 'ssl_mbm_pretrained.keras')
    shutil.copyfile(best_model_path, model_path)
    encoder_ref = p2_file_ref(model_path)
    norm_stats.update(source='inner_fit', run_id=run_id, encoder_ref=encoder_ref,
                      fit_material_ids=allocation['fit_material_ids'],
                      fit_groups=allocation['fit_groups'],
                      source_tensor_ref=raw_contract['tensor_ref'],
                      source_tensor_contract_ref=raw_contract['contract_ref'])
    norm_path = Path(args.model_dir) / 'ssl_mbm_norm_stats.json'
    norm_path.write_text(json.dumps(norm_stats, indent=2, allow_nan=False), encoding='utf-8')
    # Copy the trainer's real atomic history and TF checkpoints without editing
    # their contents; no history construction from caller-supplied metrics.
    history_path = Path(args.model_dir) / 'ssl_history.json'
    shutil.copyfile(trainer.history_path, history_path)
    history = json.loads(history_path.read_text(encoding='utf-8'))
    checkpoint_refs = {}
    for state in ('best', 'last'):
        refs = []
        for src in sorted(Path(args.checkpoint_dir).glob(f'ckpt-{state}.*')):
            dst = Path(args.model_dir) / src.name
            shutil.copyfile(src, dst)
            refs.append(p2_file_ref(dst))
        checkpoint_refs[state] = refs
    config_path = Path(args.model_dir) / 'ssl_run_config.json'
    config_path.write_text(json.dumps(vars(args), indent=2, allow_nan=False), encoding='utf-8')
    contract = dict(allocation, schema_version=1, kind='ssl_anchor', run_id=run_id,
                    scope='test' if args.test_scope else 'formal',
                    feature_schema=raw_contract['feature_schema'],
                    encoder_ref=encoder_ref, norm_ref=p2_file_ref(norm_path),
                    source_tensor_ref=raw_contract['tensor_ref'],
                    source_tensor_contract_ref=raw_contract['contract_ref'],
                    history=history, history_ref=p2_file_ref(history_path),
                    best_epoch=best_epoch, completed_epochs=completed_epochs,
                    optimizer_iterations=optimizer_iterations, checkpoints=checkpoint_refs,
                    outer_test_accessed=False, selection_monitor='val_total',
                    accepted_source='restored_best_exact_copy', config_ref=p2_file_ref(config_path),
                    states={'best': p2_file_ref(best_model_path), 'last': p2_file_ref(final_model_path),
                            'accepted': encoder_ref}, code=code_before,
                    runtime={'tensorflow': tf.__version__, 'embedding_device': probe_embedding.device,
                             'reconstruction_device': probe_reconstruction.device,
                             'required_gpu': bool(args.require_gpu), 'backward_devices': backward_devices})
    # The actual producer binds trained weight bytes inside the Keras archive.
    # This is an integrity receipt, not a digital signature/authenticated origin.
    origin_keys = ('scope', 'run_id', 'feature_schema', 'material_ids', 'groups',
                   'fit_material_ids', 'fit_groups', 'selection_material_ids', 'selection_groups',
                   'source_tensor_ref', 'source_tensor_contract_ref', 'history_ref', 'config_ref',
                   'optimizer_iterations', 'checkpoints', 'code', 'runtime')
    for label, state_sha in (('best', best_state_sha), ('last', last_state_sha)):
        origin = {key: contract[key] for key in origin_keys}
        origin.update(kind='observed_ssl_training', schema_version=1, state=label,
                      initial_state_sha256=initial_state_sha, state_sha256=state_sha,
                      normalization={key: norm_stats[key] for key in ('mean', 'std')})
        with zipfile.ZipFile(contract['states'][label]['path'], 'a') as archive:
            origin['keras_weights_sha256'] = hashlib.sha256(archive.read('model.weights.h5')).hexdigest()
            archive.writestr('assets/ssl_origin.json', json.dumps(origin, allow_nan=False))
        contract['states'][label] = p2_file_ref(contract['states'][label]['path'])
    shutil.copyfile(best_model_path, model_path)
    contract['encoder_ref'] = contract['states']['accepted'] = p2_file_ref(model_path)
    norm_stats['encoder_ref'] = contract['encoder_ref']
    norm_path.write_text(json.dumps(norm_stats, indent=2, allow_nan=False), encoding='utf-8')
    contract['norm_ref'] = p2_file_ref(norm_path)
    (Path(args.model_dir) / 'ssl_anchor_contract.json').write_text(
        json.dumps(contract, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Final model saved to {final_model_path}")
    print(f"Model saved to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Band structure SSL pre-training")
    parser.add_argument("--tensor-npz", default="./data/processed/materials_project/ood_tensors/band_tensors_ood_split.npz")
    parser.add_argument("--tensor-contract", help="explicit new train tensor contract (legacy inputs blocked)")
    parser.add_argument("--test-scope", action="store_true", help="synthetic CPU engineering only, <=32 raw rows, <=2 epochs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--min-span", type=int, default=5)
    parser.add_argument("--max-span", type=int, default=15)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sign-weight", type=float, default=0.5)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="curvature-magnitude consistency weight; keep 0 until physical k coordinates are available",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dff", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--strain-scale", type=float, default=0.01)
    parser.add_argument(
        "--disable-strain-augmentation",
        dest="disable_strain_augmentation",
        action="store_true",
        help="disable the experimental whole-path k-warp augmentation (default)",
    )
    parser.add_argument(
        "--enable-strain-augmentation",
        dest="disable_strain_augmentation",
        action="store_false",
        help="explicitly enable the experimental, non-physical whole-path k-warp",
    )
    parser.set_defaults(disable_strain_augmentation=True)
    parser.add_argument("--checkpoint-dir", default="./artifacts/checkpoints/mp/ssl_mbm")
    parser.add_argument("--log-dir", default="./artifacts/logs/mp/ssl_mbm")
    parser.add_argument("--model-dir", default="./artifacts/models/mp")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.require_gpu = args.require_gpu or not args.test_scope
    return args


def main() -> None:
    print("=" * 60)
    print("Band Structure SSL Pre-training")
    print("=" * 60)

    args = parse_args()
    if not (0.15 <= args.mask_ratio <= 0.30):
        raise ValueError("--mask-ratio must be between 0.15 and 0.30 for MBM")

    if not os.path.exists(args.tensor_npz):
        raise FileNotFoundError(
            f"Tensor dataset not found: {args.tensor_npz}. "
            "Run scripts/build_ood_tensors.py first."
        )
    run_tensor_mbm(args)


if __name__ == "__main__":
    main()
