"""P3 structure -> E_n(k), with blind inner selection and frozen evaluation.

Train reads ONLY p3_train.npz and its fixed local completion credential plus
the neutral prepare commit (never the global report or test credential).
Legacy/uncommitted NPZs are rejected; prepare --limit inputs require explicit
--smoke-limit > 0, including evaluation. Whole datasets stay in host NumPy memory;
compiled steps transfer only mini-batches. The loss is per-k energy-spectrum
1D OT (sorting), NOT trajectory Hungarian matching or the legacy gap metric.

Formal runs require GPU. --allow-cpu-smoke requires --smoke-limit > 0 and
labels all resulting artifacts as smoke, never scientific accuracy evidence.
Weights use config + real graph/k build + load_weights, not Model.save(.keras).
An optimizer/model/epoch tf.train.Checkpoint is saved each epoch and its
restore is exercised before selection is frozen. CLI resume is not provided:
existing runs must never be overwritten, including interrupted runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.benchmark_splits import build_spacegroup_split
from src.models.multiband_decoder import MultiBandDecoder, sorted_masked_mae
from src.utils.selection_manifest import sha256_file, validate_inner_selection_manifest

GRAPH_KEYS = ("atom_features", "neighbor_list", "neighbor_dist", "segment_ids")
BATCH_DTYPES = {
    "atom_features": tf.float32, "neighbor_list": tf.int32,
    "neighbor_dist": tf.float32, "segment_ids": tf.int32,
    "bands": tf.float32, "band_mask": tf.bool,
}


def verify_split_completion(npz_path):
    """Fail closed using fixed local credentials, never the global/outer report.

    Hashes attest completion/integrity, not authenticity against a malicious
    writer able to replace the NPZ, credential and commit together. Provenance
    paths are descriptive only: they never authorize additional file reads.
    """
    path = Path(npz_path)
    try:
        if path.name not in ("p3_train.npz", "p3_test.npz"):
            raise ValueError("expected canonical split filename")
        split = path.stem.removeprefix("p3_")
        commit = json.loads((path.parent / "p3_prepare_commit.json").read_text(encoding="utf-8"))
        raw = (path.parent / f"p3_{split}_complete.json").read_bytes()
        credential = json.loads(raw)
        for record, schema in ((commit, "p3_prepare_commit"), (credential, "p3_split_completion")):
            if (record["schema"] != schema or type(record["schema_version"]) is not int
                    or record["schema_version"] != 1 or record["status"] != "complete"):
                raise ValueError("unsupported schema or incomplete status")
        if commit["completion_sha256"][split] != hashlib.sha256(raw).hexdigest():
            raise ValueError("credential hash mismatch")
        if credential["split"] != split:
            raise ValueError("wrong split credential")
        limit = credential["parameters"]["limit"]
        if (type(credential["smoke_only"]) is not bool or type(limit) is not int or limit < 0
                or credential["smoke_only"] != (limit > 0)):
            raise ValueError("inconsistent smoke_only/prepare limit")
        for key in ("h5", "split_npz", "sidecar"):
            source = credential["sources"][key]
            if not isinstance(source["path"], str) or not source["path"]:
                raise ValueError("missing provenance path")
            if len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]):
                raise ValueError("invalid source hash")
        for name in ("scripts/prepare_p3_multiband.py", "src/data/multiband.py",
                     "src/data/crystal_graph.py", "src/utils/selection_manifest.py"):
            digest = credential["code_hashes"][name]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid builder hash")
        output = credential["output_npz"]
        if (output["path"] != path.name or type(output["bytes"]) is not int
                or output["bytes"] != path.stat().st_size or output["sha256"] != sha256_file(path)):
            raise ValueError("NPZ hash/bytes/name mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"P3 completion authorization failed for {path.name}: {exc}") from exc
    return credential


def load_split(npz_path, smoke_limit=0, *, return_completion=False):
    """Verify completion before loading the named NPZ into host NumPy memory."""
    completion = verify_split_completion(npz_path)
    if completion["smoke_only"] and smoke_limit <= 0:
        raise ValueError("Prepared smoke input requires explicit --smoke-limit > 0; formal use is forbidden")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {*BATCH_DTYPES, "material_ids", "groups", "valid", "k_axis"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing P3 fields: {sorted(missing)}")
        valid = archive["valid"]
        if valid.ndim != 1 or not np.isin(valid, [0, 1]).all():
            raise ValueError("valid must be a binary sample vector")
        idx = np.flatnonzero(valid.astype(bool))
        if smoke_limit:
            idx = idx[:smoke_limit]
        data = {}
        for key in (*BATCH_DTYPES, "material_ids", "groups"):
            values = archive[key]
            if values.ndim < 1 or values.shape[0] != len(valid):
                raise ValueError(f"{key} raw sample axis must match valid length {len(valid)}; got {values.shape}")
            data[key] = values[idx]
        data["source_indices"] = idx
        k_axis = archive["k_axis"].copy()
    validate_data(data, k_axis)
    data["band_mask"] = data["band_mask"].astype(bool)
    return (data, k_axis, completion) if return_completion else (data, k_axis)


def validate_data(data, k_axis):
    n = len(data["source_indices"])
    bands, atoms = data["bands"], data["atom_features"]
    if not n or bands.ndim != 3 or min(bands.shape[1:]) < 1:
        raise ValueError("Expected non-empty bands (N, B, K)")
    if atoms.ndim != 3 or atoms.shape[1] < 1 or atoms.shape[2] != 110:
        raise ValueError("Expected atom_features (N, A, 110)")
    b, k, a = bands.shape[1], bands.shape[2], atoms.shape[1]
    shapes = {"bands": (n, b, k), "atom_features": (n, a, 110),
              "band_mask": (n, b), "neighbor_list": (n, a, 12), "neighbor_dist": (n, a, 12),
              "groups": (n,), "material_ids": (n,), "segment_ids": (n, k)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"Invalid {key} shape: {data[key].shape}, expected {shape}")
    for key in BATCH_DTYPES:
        if data[key].dtype.kind not in "biuf" or not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite/non-numeric {key}")
    mask = data["band_mask"]
    if not np.isin(mask, [0, 1]).all() or not np.any(mask, axis=1).all():
        raise ValueError("Every valid sample needs a non-empty binary band_mask")
    for key in ("groups", "segment_ids", "neighbor_list"):
        if data[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must contain integers")
    if not ((data["groups"] >= 1) & (data["groups"] <= 230)).all():
        raise ValueError("groups must be real space-group numbers 1..230")
    if (data["segment_ids"] < 0).any():
        raise ValueError("segment_ids must be non-negative")
    if (data["neighbor_list"] < -1).any() or (data["neighbor_list"] >= a).any():
        raise ValueError("neighbor_list contains invalid atom indices")
    if (data["neighbor_dist"] < 0).any():
        raise ValueError("neighbor_dist cannot be negative")
    ids = data["material_ids"]
    if ids.dtype.kind != "U" or len(np.unique(ids)) != n or np.any(np.char.str_len(ids) == 0):
        raise ValueError("material_ids must be unique non-empty Unicode identifiers")
    if k_axis.shape != (k,) or not np.isfinite(k_axis).all() or not (np.diff(k_axis) > 0).all():
        raise ValueError("k_axis must be a finite, increasing shared grid")


def inner_split(data, seed):
    """Deterministic 85/15 inner split of the outer-train pool only."""
    fit, val = build_spacegroup_split(
        data["groups"], train_size=0.85, rng=np.random.RandomState(seed)
    )
    overlap = np.intersect1d(data["groups"][fit], data["groups"][val]).size
    if overlap:
        raise RuntimeError("Inner space-group overlap")
    manifest = {"seed": seed, "train_size": 0.85, "group_overlap": int(overlap)}
    for label, indices in (("fit", fit), ("validation", val)):
        manifest[label] = {
            "indices": data["source_indices"][indices].tolist(),
            "material_ids": data["material_ids"][indices].tolist(),
            "groups": data["groups"][indices].tolist(),
        }
    return fit, val, manifest


class LossAccumulator:
    """Element-weighted masked loss, never a mean of batch means."""

    def __init__(self):
        self.total = 0.0
        self.elements = 0

    def add(self, loss, mask, n_k):
        elements = int(np.count_nonzero(mask)) * n_k
        self.total += float(loss) * elements
        self.elements += elements

    def result(self):
        return self.total / self.elements


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Blind-inner-selection P3 training/evaluation")
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attention-layers", type=int, choices=(0, 2), default=0)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "d_model", "attention_heads", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.d_model % 2 or (args.attention_layers and args.d_model % args.attention_heads):
        raise ValueError("d_model must be even and divisible by attention_heads when attention is enabled")
    if args.smoke_limit < 0 or args.warmup_epochs < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid smoke-limit, warmup-epochs or seed")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError("lr must be finite positive; min_delta must be finite non-negative")
    return args


def select_device(args):
    """Formal runs never silently downgrade; CPU requires bounded smoke."""
    if args.allow_cpu_smoke:
        if args.smoke_limit <= 0:
            raise ValueError("--allow-cpu-smoke requires --smoke-limit > 0")
        return "/CPU:0"
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU required for a formal P3 run; use explicit bounded CPU smoke")
    tf.config.set_soft_device_placement(False)
    return "/GPU:0"


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def atomic_weights(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.weights.h5")
    try:
        model.save_weights(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def batch_tensors(data, indices, device):
    """The only host -> device transfer boundary: never pass the full pool."""
    with tf.device(device):
        return {key: tf.convert_to_tensor(data[key][indices], dtype=dtype)
                for key, dtype in BATCH_DTYPES.items()}


def model_graph(batch):
    return {key: batch[key] for key in GRAPH_KEYS}


def build_model(config, held_batch, device):
    with tf.device(device):
        model = MultiBandDecoder(**config["model"])
        model(model_graph(held_batch), tf.constant(config["k_axis"], tf.float32), training=False)
    return model


def epoch_learning_rate(epoch, epochs, base_lr, warmup_epochs):
    """Zero-based epoch: linear warmup followed by a single cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def require_gpu_tensor(tensor, label):
    if tf.DeviceSpec.from_string(tensor.device).device_type != "GPU":
        raise RuntimeError(f"GPU required for {label}, found {tensor.device}")


def check_training_tensors(pred, loss, grads, require_gpu):
    """Eager preflight checks real forward/loss/backward devices and finiteness."""
    if not grads or any(grad is None for grad in grads):
        raise RuntimeError("Disconnected trainable gradient")
    dense_grads = [grad.values if isinstance(grad, tf.IndexedSlices) else grad for grad in grads]
    for label, tensor in [("prediction", pred), ("loss", loss), *[("gradient", grad) for grad in dense_grads]]:
        tf.debugging.assert_all_finite(tensor, f"Non-finite {label}")
        if require_gpu:
            require_gpu_tensor(tensor, label)
    return {"prediction_device": pred.device, "loss_device": loss.device,
            "gradient_devices": sorted({grad.device for grad in dense_grads}),
            "gradient_count": len(dense_grads), "finite": True}


def make_steps(model, optimizer, k_axis, device):
    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(batch):
        with tf.device(device):
            with tf.GradientTape() as tape:
                pred = model(model_graph(batch), k_axis, training=True)
                loss = sorted_masked_mae(pred, batch["bands"], batch["band_mask"])
            grads = tape.gradient(loss, model.trainable_variables)
            checks = [tf.debugging.assert_all_finite(pred, "Non-finite prediction"),
                      tf.debugging.assert_all_finite(loss, "Non-finite loss")]
            for grad in grads:
                if grad is None:
                    raise RuntimeError("Disconnected trainable gradient")
                values = grad.values if isinstance(grad, tf.IndexedSlices) else grad
                checks.append(tf.debugging.assert_all_finite(values, "Non-finite gradient"))
            # Without these dependencies Adam's counter/slots can mutate before
            # a failing CheckNumerics op executes, even though the call raises.
            with tf.control_dependencies(checks):
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Return the actual critical tensors, not GPU identities of CPU
            # results. The caller verifies their realized devices each batch.
            return loss, pred, grads

    @tf.function(reduce_retracing=True, jit_compile=False)
    def val_step(batch):
        with tf.device(device):
            pred = model(model_graph(batch), k_axis, training=False)
            loss = sorted_masked_mae(pred, batch["bands"], batch["band_mask"])
            tf.debugging.assert_all_finite(pred, "Non-finite prediction")
            tf.debugging.assert_all_finite(loss, "Non-finite validation loss")
            return loss, pred, ()
    return train_step, val_step


def verify_checkpoint(config, held, device, prefix, model, optimizer, epoch):
    """Exercise real last-state restore, including Adam slots and step/epoch."""
    fresh = build_model(config, held, device)
    with tf.device(device):
        restored_optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["lr"], clipnorm=1.0)
        restored_optimizer.build(fresh.trainable_variables)
        restored_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        checkpoint = tf.train.Checkpoint(model=fresh, optimizer=restored_optimizer, epoch=restored_epoch)
        checkpoint.restore(prefix).assert_consumed()
        for before, after in zip(optimizer.variables, restored_optimizer.variables, strict=True):
            np.testing.assert_array_equal(before.numpy(), after.numpy())
        for before, after in zip(model.weights, fresh.weights, strict=True):
            # Keras Dropout seed generators need not be checkpoint trackables;
            # inference predictions and every learned weight must still agree.
            np.testing.assert_array_equal(before.numpy(), after.numpy())
    if int(restored_epoch.numpy()) != epoch:
        raise RuntimeError("Checkpoint epoch did not restore")
    return {"optimizer_restored": True, "epoch": epoch,
            "iterations": int(restored_optimizer.iterations.numpy()), "prefix": prefix}


def train(args, device):
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Non-empty run directory; resume/overwrite is disabled: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive claim also closes the concurrent empty-directory race.
    with (out / ".run.claim").open("x", encoding="utf-8") as handle:
        handle.write("P3 run directory claimed; CLI resume is not supported\n")
    source = Path(args.data).resolve() / "p3_train.npz"
    source_record = file_record(source)
    data, k_axis, completion = load_split(source, args.smoke_limit, return_completion=True)
    fit, val, split_manifest = inner_split(data, args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    mode = "smoke" if args.smoke_limit > 0 else "formal"
    config = {
        "schema_version": 1, "experimental": True, "mode": mode, "data": source_record,
        "preparation": completion,
        "training": {key: value for key, value in vars(args).items() if key not in ("stage", "data", "out")},
        "model": {
            "max_bands": int(data["bands"].shape[1]), "n_k": len(k_axis),
            "d_model": args.d_model, "hidden_dim": 256, "num_elements": 109,
            "conv_layers": 3, "rbf_bins": 40, "max_neighbors": 12, "dropout_rate": 0.1,
            "attention_layers": args.attention_layers, "attention_heads": args.attention_heads,
        },
        "k_axis": k_axis.tolist(),
        "code": {name.as_posix(): file_record(ROOT / name) for name in (
            Path("scripts") / "train_p3_decoder.py", Path("src") / "models" / "multiband_decoder.py",
            Path("src") / "models" / "crystal_graph_encoder.py", Path("src") / "data" / "benchmark_splits.py",
            Path("src") / "utils" / "selection_manifest.py",
        )},
    }
    atomic_json(out / "model_config.json", config)
    atomic_json(out / "inner_split.json", split_manifest)
    held = batch_tensors(data, val[:args.batch_size], device)
    model = build_model(config, held, device)
    with tf.device(device):
        k_tensor = tf.constant(k_axis, tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        checkpoint_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=checkpoint_epoch)
        probe = batch_tensors(data, fit[:args.batch_size], device)
        with tf.GradientTape() as tape:
            probe_pred = model(model_graph(probe), k_tensor, training=True)
            probe_loss = sorted_masked_mae(probe_pred, probe["bands"], probe["band_mask"])
        preflight = check_training_tensors(probe_pred, probe_loss,
                                           tape.gradient(probe_loss, model.trainable_variables),
                                           require_gpu=device == "/GPU:0")
    manager = tf.train.CheckpointManager(checkpoint, str(out / "training_checkpoint"), max_to_keep=1)
    train_step, val_step = make_steps(model, optimizer, k_tensor, device)
    history, best_loss, patience_loss, wait, best_epoch = [], math.inf, math.inf, 0, 0
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    for epoch in range(args.epochs):
        optimizer.learning_rate.assign(epoch_learning_rate(epoch, args.epochs, args.lr, args.warmup_epochs))
        aggregates = []
        for positions, step in ((rng.permutation(fit), train_step), (val, val_step)):
            aggregate = LossAccumulator()
            for offset in range(0, len(positions), args.batch_size):
                indices = positions[offset:offset + args.batch_size]
                with tf.device(device):
                    loss, pred, grads = step(batch_tensors(data, indices, device))
                grad_values = [grad.values if isinstance(grad, tf.IndexedSlices) else grad for grad in grads]
                if device == "/GPU:0":
                    for tensor in (loss, pred, *grad_values):
                        require_gpu_tensor(tensor, "compiled step loss/prediction/gradient")
                if step is train_step:
                    compiled_train = {"loss_device": loss.device, "prediction_device": pred.device,
                                      "gradient_devices": sorted({grad.device for grad in grad_values}),
                                      "gradient_count": len(grad_values), "finite_checked_in_graph": True}
                aggregate.add(float(loss.numpy()), data["band_mask"][indices], len(k_axis))
            aggregates.append(aggregate)
        loss, val_loss = (aggregate.result() for aggregate in aggregates)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            atomic_weights(model, out / "best.weights.h5")
        if val_loss < patience_loss - args.min_delta:
            patience_loss, wait = val_loss, 0
        else:
            wait += 1
        atomic_weights(model, out / "last.weights.h5")
        checkpoint_epoch.assign(epoch + 1)
        prefix = manager.save(checkpoint_number=epoch + 1)
        history.append({
            "epoch": epoch + 1, "loss": loss, "val_loss": val_loss,
            "train_elements": aggregates[0].elements, "val_elements": aggregates[1].elements,
            "learning_rate": float(optimizer.learning_rate.numpy()),
            "seconds": time.monotonic() - started,
        })
        atomic_json(out / "history.json", history)
        print(f"[P3 {mode}] epoch={epoch + 1} loss={loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if wait >= args.patience:
            break
    checkpoint_evidence = verify_checkpoint(config, held, device, prefix, model, optimizer, len(history))
    model.load_weights(str(out / "best.weights.h5"))
    atomic_weights(model, out / "accepted.weights.h5")
    reference = model(model_graph(held), k_tensor, training=False).numpy()
    fresh = build_model(config, held, device)
    fresh.load_weights(str(out / "accepted.weights.h5"))
    reloaded = fresh(model_graph(held), k_tensor, training=False).numpy()
    np.testing.assert_allclose(reference, reloaded, rtol=1e-6, atol=1e-6)
    if file_record(source) != source_record:
        raise RuntimeError("Training data changed before selection freeze")
    if verify_split_completion(source) != completion:
        raise RuntimeError("Training preparation changed before selection freeze")
    manifest = {
        "schema_version": 1, "experimental": True, "mode": mode, "monitor": "val_loss", "outer_test_accessed": False,
        "best_epoch": best_epoch, "best_val_loss": best_loss, "last_epoch": len(history),
        "stopped_early": len(history) < args.epochs,
        "states": {label: file_record(out / f"{label}.weights.h5") for label in ("best", "last", "accepted")},
        "model_config": file_record(out / "model_config.json"),
        "inner_split": file_record(out / "inner_split.json"), "history": file_record(out / "history.json"),
        "training_checkpoint": [file_record(path) for path in sorted((out / "training_checkpoint").iterdir()) if path.is_file()],
        "checkpoint_verification": checkpoint_evidence,
        "reload_verification": {"predictions_match": True, "max_abs_difference": float(np.max(np.abs(reference - reloaded))),
                                "held_material_ids": data["material_ids"][val[:args.batch_size]].tolist()},
        "runtime": {"device": device, "data_residency": "host_numpy", "max_batch_size": args.batch_size,
                    "tensorflow": tf.__version__, "optimizer_clipnorm": 1.0, "preflight": preflight,
                    "compiled_train": compiled_train},
        "resume_policy": "Checkpoint state saved and restore verified; CLI resume/overwrite is not supported",
    }
    atomic_json(out / "inner_selection_manifest.json", manifest)
    validate_inner_selection_manifest(str(out))
    return manifest


def verify_file_record(record, path):
    actual = file_record(path)
    if any(actual[key] != record.get(key) for key in ("sha256", "bytes")):
        raise RuntimeError(f"Frozen hash/bytes mismatch: {path}")


def validate_frozen_run(out, data_dir):
    manifest = validate_inner_selection_manifest(str(out))
    for key, filename in (("model_config", "model_config.json"), ("inner_split", "inner_split.json"), ("history", "history.json")):
        verify_file_record(manifest[key], out / filename)
    config = json.loads((out / "model_config.json").read_text(encoding="utf-8"))
    verify_file_record(config["data"], Path(data_dir).resolve() / "p3_train.npz")
    if verify_split_completion(Path(data_dir).resolve() / "p3_train.npz") != config["preparation"]:
        raise RuntimeError("Frozen training preparation mismatch")
    for name, record in config["code"].items():
        verify_file_record(record, ROOT / name)
    return manifest, config


def audit_outer_disjoint(source, inner_path):
    """Evaluation-only audit against frozen fit + validation, including every
    outer ID/group row (invalid rows and rows beyond a smoke prefix included).
    The extra NPZ open is lazy: only the two identity axes are read, not bands.
    """
    inner = json.loads(Path(inner_path).read_text(encoding="utf-8"))
    audit = {"scope": "frozen_inner_pool_vs_all_outer_rows"}
    with np.load(source, allow_pickle=False) as outer:
        for field, metric in (("material_ids", "id_overlap"), ("groups", "group_overlap")):
            pool = set(inner["fit"][field]) | set(inner["validation"][field])
            audit[metric] = len(pool & set(outer[field].tolist()))
    if audit["id_overlap"] or audit["group_overlap"]:
        raise ValueError(f"Outer overlap with frozen training pool: {audit}")
    return audit


def evaluate(args, device):
    """Only frozen selection may authorize outer NPZ access."""
    out = Path(args.out).resolve()
    manifest, config = validate_frozen_run(out, args.data)
    selection_record = file_record(out / "inner_selection_manifest.json")
    # Intentionally lazy: no evaluator (especially no legacy gap) in train.
    from src.evaluation.multiband_metrics import evaluate_multiband

    source = Path(args.data).resolve() / "p3_test.npz"
    source_record = file_record(source)
    data, k_axis, completion = load_split(source, args.smoke_limit, return_completion=True)
    outer_audit = audit_outer_disjoint(source, out / "inner_split.json")
    if not np.array_equal(k_axis, np.asarray(config["k_axis"], dtype=k_axis.dtype)):
        raise ValueError("Outer k_axis differs from the frozen training grid")
    if data["bands"].shape[1:] != (config["model"]["max_bands"], config["model"]["n_k"]):
        raise ValueError("Outer band dimensions differ from frozen model_config")
    indices = np.arange(len(data["bands"]))
    held = batch_tensors(data, indices[:args.batch_size], device)
    model = build_model(config, held, device)
    model.load_weights(manifest["states"]["accepted"]["path"])
    model.compile(jit_compile=False)
    with tf.device(device):
        k_tensor = tf.constant(config["k_axis"], tf.float32)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def predict_step(batch):
        with tf.device(device):
            pred = model(model_graph(batch), k_tensor, training=False)
            tf.debugging.assert_all_finite(pred, "Non-finite outer prediction")
            return pred

    predictions = []
    for start in range(0, len(indices), args.batch_size):
        batch = batch_tensors(data, indices[start:start + args.batch_size], device)
        with tf.device(device):
            prediction = predict_step(batch)
        if device == "/GPU:0":
            require_gpu_tensor(prediction, "compiled outer prediction")
        predictions.append(prediction.numpy())
    pred = np.concatenate(predictions, axis=0)
    metrics = evaluate_multiband(pred, data["bands"], data["band_mask"], segment_ids=data["segment_ids"])
    if file_record(source) != source_record:
        raise RuntimeError("Outer data changed during evaluation")
    if verify_split_completion(source) != completion:
        raise RuntimeError("Outer preparation changed during evaluation")
    if validate_frozen_run(out, args.data) != (manifest, config) or file_record(out / "inner_selection_manifest.json") != selection_record:
        raise RuntimeError("Frozen selection changed during evaluation")
    evaluation_dir = out / "evaluations" / uuid.uuid4().hex
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    temporary = evaluation_dir / "predictions.tmp.npz"
    np.savez_compressed(temporary, pred=pred, material_ids=data["material_ids"], mask=data["band_mask"],
                        target=data["bands"], k_axis=k_axis, segment_ids=data["segment_ids"],
                        groups=data["groups"], source_indices=data["source_indices"])
    os.replace(temporary, evaluation_dir / "predictions.npz")
    report = {
        "schema_version": 1, "experimental": True, "stage": "evaluate", "scope": "outer_p3_test_only",
        "mode": "smoke" if config["mode"] == "smoke" or completion["smoke_only"] or args.smoke_limit else "formal",
        "evaluation_dir": str(evaluation_dir), "selection_manifest": selection_record,
        "accepted": manifest["states"]["accepted"], "outer_data": source_record,
        "outer_preparation": completion,
        "outer_split_audit": outer_audit,
        "metrics": metrics, "predictions": file_record(evaluation_dir / "predictions.npz"),
        "runtime": {"device": device, "max_batch_size": args.batch_size, "smoke_limit": args.smoke_limit,
                    "prediction_device": prediction.device},
        "evaluator_code": file_record(ROOT / "src/evaluation/multiband_metrics.py"),
    }
    atomic_json(evaluation_dir / "evaluation.json", report)
    print(f"[P3 {report['mode']}] evaluation -> {evaluation_dir / 'evaluation.json'}", flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args)
    if args.stage == "train":
        return train(args, device)
    return evaluate(args, device)


if __name__ == "__main__":
    main()
