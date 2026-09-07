"""P2 v2 step 1: precompute frozen band-encoder embeddings (TF side).

The v1 contrastive run showed the band encoder should stay frozen as the
anchor; this script extracts its (N, 256) pooled `return_features` embeddings
for every valid material in the P2 pair data and stores them as a small .npz.
The PyTorch equivariant encoder then trains against these as InfoNCE targets
(cross-framework bridge at the embedding level only).

Usage:
    python scripts/extract_band_embeddings.py \
        --pairs data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs \
        --band-encoder artifacts/models/aflow_noleak_v7_60k_seed42/ssl_mbm_pretrained.keras \
        --out data/processed/aflow/ood_tensors_v7_60000_seed42/p2_pairs/band_embeddings.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import load_ssl_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen band embeddings")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--band-encoder", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    band = load_ssl_encoder(args.band_encoder, compile=False)
    band.trainable = False

    out = {}
    for split in ("train", "test"):
        p = np.load(os.path.join(args.pairs, f"p2_{split}.npz"))
        valid = p["valid"].astype(bool)
        idx = np.where(valid)[0]
        X = p["band_input"][idx]  # (n, 128, 6)
        embs = []
        for start in range(0, len(X), 128):
            chunk = X[start : start + 128]
            e = band(tf.constant(chunk), return_features=True, training=False).numpy()
            embs.append(e)
        embs = np.concatenate(embs, axis=0)
        out[f"band_emb_{split}"] = embs
        out[f"valid_idx_{split}"] = idx
        print(f"[BANDEMB] {split}: {len(idx)} embeddings, dim {embs.shape[1]}", flush=True)

    np.savez(args.out, **out)
    print(f"[BANDEMB] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
