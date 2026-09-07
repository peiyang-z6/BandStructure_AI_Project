"""Contrastive alignment + retrieval evaluation for P2 cross-modal search.

Constitution 5.0 §8 P2: structure–band contrastive learning aligns a shared
physical embedding on the 60k paired data; an ANN/brute-force index then
supports "band image -> most similar materials/structures/confidence".

Retrieval semantics: cross-modal retrieval is asymmetric. Query embeddings Q
(structure) are matched against a gallery G (band) where the ground-truth
pair is the diagonal (material i's structure <-> material i's band). Metrics
here take a similarity matrix S = Q @ G^T and treat row i's true match as
column i (no self-exclusion, since Q and G are different modalities).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an (N, D) embedding matrix."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N_a, D) x (N_b, D) -> (N_a, N_b) cosine similarity."""
    a = normalize_embeddings(a)
    b = normalize_embeddings(b)
    return a @ b.T


def recall_at_k(sim: np.ndarray, k: int) -> float:
    """Recall@K: row i's true match is column i (diagonal ground truth)."""
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    hits = sum(int(i in order[i, :k]) for i in range(n))
    return hits / n


def mean_average_precision(sim: np.ndarray) -> float:
    """mAP for the diagonal-pair retrieval task."""
    n = sim.shape[0]
    ap_sum = 0.0
    for i in range(n):
        order = np.argsort(-sim[i])
        true_rank = int(np.where(order == i)[0][0])
        ap_sum += 1.0 / (true_rank + 1)
    return ap_sum / n


def median_rank(sim: np.ndarray) -> float:
    """Median rank (1-indexed) of the true match across queries."""
    n = sim.shape[0]
    ranks = [int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1 for i in range(n)]
    return float(np.median(ranks)) if ranks else float("nan")


def retrieval_metrics(sim: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    return {
        **{f"recall@{k}": recall_at_k(sim, k) for k in ks},
        "map": mean_average_precision(sim),
        "median_rank": median_rank(sim),
    }


def info_nce_temperature(emb_a: np.ndarray, emb_b: np.ndarray, temperature: float = 0.07) -> float:
    """InfoNCE loss for a batch of paired embeddings (numpy, for testing)."""
    a = normalize_embeddings(emb_a)
    b = normalize_embeddings(emb_b)
    logits = a @ b.T / temperature  # (N, N)
    logits = logits - np.max(logits, axis=1, keepdims=True)  # stability
    labels = np.arange(logits.shape[0])
    loss = -np.mean(logits[np.arange(logits.shape[0]), labels]
                    - np.log(np.sum(np.exp(logits), axis=1)))
    return float(loss)


def bidirectional_metrics(struct_emb: np.ndarray, band_emb: np.ndarray, ks=(1, 5, 10)) -> Dict[str, Dict[str, float]]:
    """Both directions: structure->band (row i = structure, col i = band) and
    band->structure (row i = band, col i = structure)."""
    s2b = cosine_similarity_matrix(struct_emb, band_emb)
    b2s = cosine_similarity_matrix(band_emb, struct_emb)
    return {
        "structure_to_band": retrieval_metrics(s2b, ks),
        "band_to_structure": retrieval_metrics(b2s, ks),
    }
