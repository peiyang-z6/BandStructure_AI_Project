"""P3 decoder training: structure -> Fermi-proximate multi-band E_n(k).

Constitution 5.0 §8 P3. Runs on the server V100 (TF). Loads the P3 NPZ
(produced by prepare_p3_multiband.py), trains MultiBandDecoder with masked MAE,
and reports band MAE (masked) + gap MAE on the outer OOD test split.

The training step is `@tf.function`-compiled (the eager per-batch loop was
~0.18s/batch -> ~36 min/epoch at batch 4, untenable), so data tensors are
built once on GPU and each step runs as a fused graph op.

Usage:
    python scripts/train_p3_decoder.py \
        --data data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband \
        --out artifacts/models/aflow_noleak_v7_60k_seed42/p3_decoder \
        [--epochs 60 --batch-size 32 --lr 1e-3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.multiband_decoder import MultiBandDecoder, sorted_masked_mae
from src.evaluation.multiband_metrics import band_mae, gap_mae, sorted_band_mae


def load_split(npz_path):
    d = np.load(npz_path)
    valid = d["valid"].astype(bool)
    idx = np.where(valid)[0]
    return {
        "atom_features": d["atom_features"][idx],
        "neighbor_list": d["neighbor_list"][idx],
        "neighbor_dist": d["neighbor_dist"][idx],
        "bands": d["bands"][idx],
        "band_mask": d["band_mask"][idx],
    }, d["k_axis"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P3 multi-band decoder")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train, k_axis = load_split(os.path.join(args.data, "p3_train.npz"))
    test, _ = load_split(os.path.join(args.data, "p3_test.npz"))
    print(f"[P3] train {len(train['bands'])} valid, test {len(test['bands'])} valid", flush=True)

    n_k = int(train["bands"].shape[-1])
    max_bands = int(train["bands"].shape[1])
    k_axis_t = tf.constant(k_axis, dtype=tf.float32)

    model = MultiBandDecoder(max_bands=max_bands, n_k=n_k, d_model=args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    def to_gpu(split):
        return {
            "atom_features": tf.constant(split["atom_features"], dtype=tf.float32),
            "neighbor_list": tf.constant(split["neighbor_list"], dtype=tf.int32),
            "neighbor_dist": tf.constant(split["neighbor_dist"], dtype=tf.float32),
            "bands": tf.constant(split["bands"], dtype=tf.float32),
            "band_mask": tf.constant(split["band_mask"], dtype=tf.bool),
        }

    g_train = to_gpu(train)
    g_test = to_gpu(test)

    @tf.function
    def train_step(bidx):
        graph = {k: tf.gather(g_train[k], bidx) for k in ("atom_features", "neighbor_list", "neighbor_dist")}
        bands = tf.gather(g_train["bands"], bidx)
        mask = tf.gather(g_train["band_mask"], bidx)
        with tf.GradientTape() as tape:
            pred = model(graph, k_axis_t, training=True)
            loss = sorted_masked_mae(pred, bands, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def predict_batch(bidx, split_g):
        graph = {k: tf.gather(split_g[k], bidx) for k in ("atom_features", "neighbor_list", "neighbor_dist")}
        return model(graph, k_axis_t, training=False)

    n = g_train["bands"].shape[0]
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = tf.random.shuffle(tf.range(n))
        epoch_loss = tf.constant(0.0)
        nb = 0
        for start in range(0, n, args.batch_size):
            bidx = perm[start : start + args.batch_size]
            epoch_loss += train_step(bidx)
            nb += 1
        avg = float(epoch_loss.numpy()) / max(nb, 1)
        history.append({"epoch": epoch + 1, "loss": avg})
        print(f"[P3] epoch {epoch+1}/{args.epochs} loss {avg:.4f} ({time.time()-t0:.1f}s)", flush=True)

    def evaluate(split_g):
        preds = []
        n = split_g["bands"].shape[0]
        for start in range(0, n, args.batch_size):
            bidx = tf.range(start, min(start + args.batch_size, n))
            preds.append(predict_batch(bidx, split_g).numpy())
        pred = np.concatenate(preds, axis=0)
        bands = split_g["bands"].numpy()
        mask = split_g["band_mask"].numpy()
        return {
            "band_mae": band_mae(pred, bands, mask),
            "sorted_band_mae": sorted_band_mae(pred, bands, mask),
            "gap_mae": gap_mae(pred, bands, mask, 0.0),
        }

    report = {
        "train": evaluate(g_train),
        "test": evaluate(g_test),
        "history": history,
    }
    print(f"[P3] train: {json.dumps(report['train'])}", flush=True)
    print(f"[P3] test: {json.dumps(report['test'])}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "multiband_decoder.keras"))
    with open(os.path.join(args.out, "p3_decoder_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[P3] report -> {os.path.join(args.out, 'p3_decoder_report.json')}", flush=True)


if __name__ == "__main__":
    main()
