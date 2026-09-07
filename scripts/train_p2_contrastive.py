"""P2 cross-modal contrastive training: align structure and band embeddings.

Constitution 5.0 §8 P2. Freezes the v7 MBM band encoder (return_features)
and trains a CGCNN crystal graph encoder so that structure embeddings and
band embeddings for the SAME material are pulled together (InfoNCE) in a
shared 128-dim space.

The band encoder stays frozen (its SSL representation is the anchor); only
the CGCNN encoder + optional projection are trained. Input pairs come from
`prepare_p2_pairs.py` output (p2_train.npz / p2_test.npz), which already
respects the outer OOD split — the test set is never used for training or
checkpoint selection.

Usage:
    python scripts/train_p2_contrastive.py \
        --train data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/p2_train.npz \
        --test data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/p2_test.npz \
        --band-encoder artifacts/models/aflow_noleak_v7_60k_seed42/ssl_mbm_pretrained.keras \
        --output-dir artifacts/models/aflow_noleak_v7_60k_seed42/p2_contrastive \
        [--epochs 40 --batch-size 64 --temperature 0.07 --embedding-dim 128]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import load_ssl_encoder
from src.models.crystal_graph_encoder import CGCNNEncoder
from src.evaluation.retrieval import bidirectional_metrics, cosine_similarity_matrix


def configure_tf(require_gpu: bool = False):
    if require_gpu and not tf.config.list_physical_devices("GPU"):
        raise SystemExit("GPU required but not visible; aborting formal run")
    # Let GPU memory grow on demand instead of reserving the whole device.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(42)


def make_dataset(pairs: dict, batch_size: int, shuffle: bool):
    """Build a tf.data.Dataset from the pre-padded numpy arrays.

    The pair arrays are already padded to (N, max_atoms, ...) and
    (N, 128, 6), so from_tensor_slices is used directly (much faster than a
    python generator). Valid masks drop materials without graphs.
    """
    valid = pairs["valid"].astype(bool)
    idx = np.where(valid)[0]
    if shuffle:
        np.random.default_rng(42).shuffle(idx)

    graph = {
        "atom_features": tf.constant(pairs["atom_features"][idx]),
        "neighbor_list": tf.constant(pairs["neighbor_list"][idx]),
        "neighbor_dist": tf.constant(pairs["neighbor_dist"][idx]),
    }
    band = tf.constant(pairs["band_input"][idx])

    ds = tf.data.Dataset.from_tensor_slices((graph, band))
    ds = ds.shuffle(len(idx), seed=42) if shuffle else ds
    ds = ds.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)
    return ds


def info_nce_loss(struct_emb, band_emb, temperature):
    struct_emb = tf.math.l2_normalize(struct_emb, axis=-1)
    band_emb = tf.math.l2_normalize(band_emb, axis=-1)
    logits = tf.matmul(struct_emb, band_emb, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )


@tf.function
def train_step(struct_encoder, band_encoder, graph, band, temperature, optimizer):
    with tf.GradientTape() as tape:
        struct_emb = struct_encoder(graph, training=True)
        band_emb = band_encoder(band, return_features=True, training=False)
        loss = info_nce_loss(struct_emb, band_emb, temperature)
    grads = tape.gradient(loss, struct_encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, struct_encoder.trainable_variables))
    return loss


def evaluate_retrieval(struct_encoder, band_encoder, pairs, batch_size=64, max_samples=None):
    """Compute structure<->band retrieval metrics over valid pairs.

    max_samples limits the number of pairs evaluated (subset for smoke runs;
    full set on the GPU server). Small batch avoids laptop-GPU OOM.
    """
    valid = pairs["valid"].astype(bool)
    idx = np.where(valid)[0]
    if max_samples is not None and len(idx) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx = np.sort(idx)
    struct_embs = []
    band_embs = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        graph = {
            "atom_features": tf.constant(pairs["atom_features"][chunk]),
            "neighbor_list": tf.constant(pairs["neighbor_list"][chunk]),
            "neighbor_dist": tf.constant(pairs["neighbor_dist"][chunk]),
        }
        band = tf.constant(pairs["band_input"][chunk])
        struct_embs.append(struct_encoder(graph, training=False).numpy())
        band_embs.append(band_encoder(band, return_features=True, training=False).numpy())
    struct_embs = np.concatenate(struct_embs, axis=0)
    band_embs = np.concatenate(band_embs, axis=0)
    return bidirectional_metrics(struct_embs, band_embs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2 cross-modal contrastive model")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--band-encoder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--conv-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    configure_tf(require_gpu=args.require_gpu)

    train = np.load(args.train)
    test = np.load(args.test)
    print(f"[P2] train valid {train['valid'].sum()}/{len(train['valid'])}, "
          f"test valid {test['valid'].sum()}/{len(test['valid'])}", flush=True)

    band_encoder = load_ssl_encoder(args.band_encoder, compile=False)
    # Freeze the band encoder entirely (anchor representation).
    for layer in band_encoder.layers:
        layer.trainable = False
    band_encoder.trainable = False

    struct_encoder = CGCNNEncoder(
        num_elements=109,
        conv_layers=args.conv_layers,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
    )

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_ds = make_dataset(train, args.batch_size, shuffle=True)

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for graph, band in train_ds:
            loss = train_step(struct_encoder, band_encoder, graph, band,
                              args.temperature, optimizer)
            epoch_losses.append(float(loss))
        avg_loss = float(np.mean(epoch_losses))
        print(f"[P2] epoch {epoch+1}/{args.epochs} loss {avg_loss:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    # Evaluate retrieval on train (fit check, subset) and OOD test (generalization).
    # Run embedding extraction on CPU: the laptop GPU/WSL graphics stack is
    # unstable under full-batch evaluation forwards (crashes the whole WSL VM),
    # and evaluation is forward-only so CPU cost is acceptable for a smoke run.
    # The formal run evaluates on the V100 server (GPU).
    import os as _os
    with tf.device("/CPU:0"):
        train_metrics = evaluate_retrieval(struct_encoder, band_encoder, train,
                                           max_samples=2000)
        test_metrics = evaluate_retrieval(struct_encoder, band_encoder, test,
                                          max_samples=2000)
    print(f"[P2] train retrieval: {json.dumps(train_metrics, indent=2)}", flush=True)
    print(f"[P2] test retrieval: {json.dumps(test_metrics, indent=2)}", flush=True)

    struct_encoder.save(os.path.join(args.output_dir, "structure_encoder.keras"))
    with open(os.path.join(args.output_dir, "p2_report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "train_retrieval": train_metrics,
            "test_retrieval": test_metrics,
            "history": history,
            "config": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "temperature": args.temperature, "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim, "conv_layers": args.conv_layers,
            },
        }, fh, ensure_ascii=False, indent=2)
    print(f"[P2] report -> {os.path.join(args.output_dir, 'p2_report.json')}", flush=True)


if __name__ == "__main__":
    main()
