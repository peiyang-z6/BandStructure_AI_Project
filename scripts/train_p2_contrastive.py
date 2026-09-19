"""P2 CGCNN: isolated inner selection and frozen evaluation.

The architecture (3 layers, hidden/embedding 128) and one-way InfoNCE at 0.07
are unchanged. Training accepts only credentialed raw pairs plus frozen MBM
normalization. No outer path is passed into train_only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_p2_pairs as contracts
from scripts.extract_band_embeddings import (configure_runtime, load_anchor, encode_bands,
                                              tf_state_sha, require_gpu_tensor, inspect_anchor)
from src.models.crystal_graph_encoder import CGCNNEncoder


def info_nce_loss(struct_emb, band_emb, temperature=.07):
    s = tf.math.l2_normalize(struct_emb, axis=-1)
    b = tf.math.l2_normalize(band_emb, axis=-1)
    logits = tf.matmul(s, b, transpose_b=True) / temperature
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        tf.range(tf.shape(logits)[0]), logits, from_logits=True))


def graph_batch(pairs, indices):
    return {k:tf.constant(pairs[k][indices]) for k in ("atom_features", "neighbor_list", "neighbor_dist")}


def encode_structures(model, pairs, indices, batch_size, formal=False):
    before = tf_state_sha(model)
    result = []
    for start in range(0, len(indices), batch_size):
        value = model(graph_batch(pairs, indices[start:start+batch_size]), training=False)
        require_gpu_tensor(value, formal)
        result.append(value.numpy())
    if tf_state_sha(model) != before:
        raise ValueError("frozen structure state changed during inference")
    return np.concatenate(result)


def encode_pairs(model, encoder_path, norm_path, pairs, indices, batch_size=64, test_scope=False, expected_anchor=None, *, anchor_contract=None):
    device = configure_runtime(test_scope, len(pairs["material_ids"]))
    with tf.device(device):
        band, stats, anchor = load_anchor(encoder_path, norm_path, expected=expected_anchor,
                                          anchor_contract=anchor_contract, test_scope=test_scope)
        b = encode_bands(band, pairs["band_input"][indices], stats, batch_size, not test_scope)
        s = encode_structures(model, pairs, indices, batch_size, not test_scope)
    contracts.verify_sources(contracts.source_refs(anchor))
    return s, b


def full_validation_loss(s, b, batch_size, formal=False):
    """Complete inner gallery, not the final minibatch or a subsampled gallery."""
    s, b = tf.math.l2_normalize(tf.constant(s), -1), tf.math.l2_normalize(tf.constant(b), -1)
    total = tf.constant(0., tf.float32)
    for start in range(0, len(s), batch_size):
        logits = tf.matmul(s[start:start+batch_size], b, transpose_b=True) / .07
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.range(start, min(start+batch_size, len(s))), logits, from_logits=True)
        require_gpu_tensor(loss, formal)
        total += tf.reduce_sum(loss)
    return float(total / len(s))


def train_only(train, band_encoder, norm, output_dir, *, epochs=40, batch_size=64,
               validation_size=.2, seed=42, learning_rate=1e-3, test_scope=False, anchor_contract=None):
    device = configure_runtime(test_scope, epochs=epochs)
    out = contracts.fresh_directory(output_dir)
    pairs, parent = contracts.load_pairs(train, "train", test_scope)
    configure_runtime(test_scope, len(pairs["material_ids"]), epochs)
    with tf.device(device):
        band, stats, anchor = load_anchor(band_encoder, norm, anchor_contract=anchor_contract, test_scope=test_scope)
    fit, val, inner = contracts.inner_partition(pairs["material_ids"], pairs["groups"],
        pairs["valid_idx"], validation_size, seed, anchor=anchor, raw_contract=parent["raw_contract"])
    config = {"epochs":epochs, "batch_size":batch_size, "validation_size":validation_size,
              "seed":seed, "learning_rate":learning_rate, "temperature":.07,
              "hidden_dim":128, "embedding_dim":128, "conv_layers":3,
              "num_elements":109, "max_atoms":int(pairs["atom_features"].shape[1]),
              "loss":"one_way_structure_to_band_InfoNCE", "selection":"complete_inner_gallery_InfoNCE"}
    code = contracts.code_refs(Path(__file__), ROOT/"scripts/extract_band_embeddings.py",
                               ROOT/"scripts/prepare_p2_pairs.py", ROOT/"src/data/band_structure_dataset.py",
                               ROOT/"src/models/crystal_graph_encoder.py", ROOT/"src/models/band_structure_encoder.py")
    tf.keras.utils.set_random_seed(seed)
    history, checkpoints = [], {}
    best = float("inf")
    with tf.device(device):
        band_before = tf_state_sha(band)
        # Frozen anchor is evaluated once on train rows only, using frozen stats.
        band_values = np.zeros((len(pairs["material_ids"]), 128), np.float32)
        encoded = encode_bands(band, pairs["band_input"][pairs["valid_idx"]], stats, batch_size, not test_scope)
        if encoded.shape[1] != 128:
            raise ValueError("pooled MBM dimension must be 128; do not silently change features or architecture")
        band_values[pairs["valid_idx"]] = encoded
        model = CGCNNEncoder(num_elements=109, conv_layers=3, hidden_dim=128, embedding_dim=128)
        model(graph_batch(pairs, fit[:2]), training=False)
        initial_state = tf_state_sha(model)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs+1):
            started = time.monotonic()
            total = tf.constant(0., tf.float32)
            for idx in contracts.batches(rng.permutation(fit), batch_size):
                with tf.GradientTape() as tape:
                    s = model(graph_batch(pairs, idx), training=True)
                    loss = info_nce_loss(s, tf.constant(band_values[idx]), .07)
                    require_gpu_tensor(s, not test_scope)
                    require_gpu_tensor(loss, not test_scope)
                grads = tape.gradient(loss, model.trainable_variables)
                for grad in grads:
                    if grad is None:
                        raise ValueError("missing structure gradient")
                    require_gpu_tensor(grad, not test_scope)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                total += tf.stop_gradient(loss) * len(idx)
            s_val = encode_structures(model, pairs, val, batch_size, not test_scope)
            val_loss = full_validation_loss(s_val, band_values[val], batch_size, not test_scope)
            row = {"epoch":epoch, "loss":float(total/len(fit)), "val_loss":val_loss,
                   "fit_n":len(fit), "validation_n":len(val), "seconds":time.monotonic()-started}
            history.append(row)
            contracts.write_json(out/"history.json", history)
            print(row, flush=True)
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                model.save_weights(out/"best.weights.h5")
                checkpoints["best"] = {"file":"best.weights.h5", "artifact":contracts.file_ref(out/"best.weights.h5"),
                                       "state_sha256":tf_state_sha(model)}
        model.save_weights(out/"last.weights.h5")
        checkpoints["last"] = {"file":"last.weights.h5", "artifact":contracts.file_ref(out/"last.weights.h5"),
                               "state_sha256":tf_state_sha(model)}
        model.load_weights(out/"best.weights.h5")
        if tf_state_sha(model) != checkpoints["best"]["state_sha256"] or tf_state_sha(band) != band_before:
            raise ValueError("restored best / frozen band state mismatch")
        shutil.copyfile(out/"best.weights.h5", out/"accepted.weights.h5")
        checkpoints["accepted"] = {**checkpoints["best"], "file":"accepted.weights.h5"}
    if contracts.file_ref(band_encoder) != anchor["encoder"] or contracts.file_ref(norm) != anchor["norm"]:
        raise ValueError("anchor source changed")
    if contracts.file_ref(train) != parent["artifact"]:
        raise ValueError("training data changed")
    manifest = {"framework":"tensorflow", "test_scope":bool(test_scope), "config":config,
                "run_id":contracts.uuid.uuid4().hex, "initial_state_sha256":initial_state,
                "history_ref":contracts.file_ref(out/"history.json"),
                "anchor":anchor, "data":{"pairs":parent}, "code":code, "inner":inner,
                "history":history, "best_epoch":best_epoch, "checkpoints":checkpoints,
                "runtime":{"python":platform.python_version(), "tensorflow":tf.__version__,
                           "numpy":np.__version__, "device":device, "scope":"cpu_test" if test_scope else "formal_gpu"}}
    return contracts.freeze_selection(out, manifest, sources=contracts.source_refs(parent, anchor))


def load_frozen(run_dir, test_scope=False, *, band_encoder=None, norm=None, anchor_contract=None):
    m = contracts.verify_selection(run_dir, "tensorflow")
    if m["test_scope"] and not test_scope:
        raise ValueError("test-scope checkpoint cannot authorize formal evaluation")
    device = configure_runtime(test_scope)
    _, anchor = inspect_anchor(band_encoder,norm,m["anchor"],anchor_contract=anchor_contract,test_scope=test_scope)
    m.snapshots.update(contracts.source_refs(anchor))
    with tf.device(device):
        model = CGCNNEncoder(num_elements=109, conv_layers=3, hidden_dim=128, embedding_dim=128)
        n = m["config"]["max_atoms"]
        model({"atom_features":tf.zeros((1,n,110)), "neighbor_list":tf.fill((1,n,12),-1),
               "neighbor_dist":tf.zeros((1,n,12))}, training=False)
        item = m["checkpoints"]["accepted"]
        with contracts.snapshot(Path(run_dir)/item["file"], item["artifact"]) as (copy, _):
            model.load_weights(copy)
        model.trainable = False
    if tf_state_sha(model) != m["checkpoints"]["accepted"]["state_sha256"]:
        raise ValueError("reloaded full model state mismatch")
    return model, m


def evaluation_only(test, band_encoder, norm, run_dir, output_dir, *, test_scope=False, batch_size=64, anchor_contract=None):
    # Frozen hashes are verified before any outer array or manifest is opened.
    m = contracts.verify_selection(run_dir, "tensorflow")
    sources = contracts.source_refs(m)
    if anchor_contract is None:
        raise ValueError("explicit upstream anchor contract required")
    _, anchor = inspect_anchor(band_encoder,norm,m["anchor"],anchor_contract=anchor_contract,test_scope=test_scope)
    sources.update(contracts.source_refs(anchor))
    model, reloaded = load_frozen(run_dir, test_scope, band_encoder=band_encoder, norm=norm, anchor_contract=anchor_contract)
    contracts.verify_sources(sources)
    if reloaded != m:
        raise ValueError("frozen selection changed during model reload")
    out = contracts.fresh_directory(output_dir)
    pairs, parent = contracts.load_pairs(test, "test", test_scope)
    sources.update(contracts.source_refs(parent))
    contracts.reject_outer_overlap(pairs["material_ids"], pairs["groups"], m)
    before = tf_state_sha(model)
    s, b = encode_pairs(model, band_encoder, norm, pairs, pairs["valid_idx"], batch_size, test_scope,
                        m["anchor"], anchor_contract=anchor_contract)
    after = tf_state_sha(model)
    if before != after:
        raise ValueError("evaluation changed frozen model state")
    contracts.verify_selection(run_dir, "tensorflow")
    ids = pairs["valid_material_ids"]
    report = {"schema":contracts.SCHEMA, "kind":"evaluation", "test_scope":bool(test_scope),
              "selection":sources[Path(run_dir)/"selection.json"], "data":{"pairs":parent},
              "state_before":before, "state_after":after, "anchor":m["anchor"], "config":m["config"],
              "runtime":{"python":platform.python_version(), "tensorflow":tf.__version__,
                         "device":configure_runtime(test_scope), "batch_size":batch_size},
              "retrieval":contracts.retrieval_evidence(s, b, ids)}
    return contracts.publish_evaluation(out, report, s, b, ids, sources=sources)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluation-only", action="store_true")
    p.add_argument("--train")
    p.add_argument("--test")
    p.add_argument("--run-dir")
    p.add_argument("--band-encoder", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--anchor-contract", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--validation-size", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--test-scope", action="store_true")
    a = p.parse_args(argv)
    if a.train_only:
        if not a.train or a.test or a.run_dir:
            p.error("train-only accepts --train, never --test/--run-dir")
        train_only(a.train, a.band_encoder, a.norm, a.output_dir, epochs=a.epochs,
                   batch_size=a.batch_size, validation_size=a.validation_size, seed=a.seed,
                   learning_rate=a.learning_rate, test_scope=a.test_scope, anchor_contract=a.anchor_contract)
    else:
        if not a.test or not a.run_dir or a.train:
            p.error("evaluation-only requires --test --run-dir and forbids --train")
        evaluation_only(a.test, a.band_encoder, a.norm, a.run_dir, a.output_dir,
                        batch_size=a.batch_size, test_scope=a.test_scope, anchor_contract=a.anchor_contract)


if __name__ == "__main__":
    main()
