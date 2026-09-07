"""Regression tests for P2 retrieval metrics and InfoNCE loss."""
import numpy as np
import pytest

from src.evaluation.retrieval import (
    cosine_similarity_matrix,
    info_nce_temperature,
    mean_average_precision,
    median_rank,
    normalize_embeddings,
    recall_at_k,
    retrieval_metrics,
)


def test_normalize_embeddings_unit_norm():
    emb = np.array([[3.0, 4.0], [0.0, 5.0]])
    normed = normalize_embeddings(emb)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_cosine_similarity_perfect_diagonal():
    emb = normalize_embeddings(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    sim = cosine_similarity_matrix(emb, emb)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim.shape == (3, 3)


def test_recall_at_k_perfect_identity():
    # identity embeddings -> each row's true match (col i) ranks first
    sim = cosine_similarity_matrix(np.eye(4), np.eye(4))
    assert recall_at_k(sim, 1) == 1.0
    assert recall_at_k(sim, 5) == 1.0


def test_recall_at_k_random_is_low():
    # cross-modal: query and gallery are DIFFERENT random matrices, so there
    # is no trivial self-match (diagonal is not forced to 1).
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200, 32))
    g = rng.normal(size=(200, 32))
    sim = cosine_similarity_matrix(q, g)
    assert recall_at_k(sim, 1) < 0.1


def test_map_perfect_is_one():
    sim = cosine_similarity_matrix(np.eye(5), np.eye(5))
    assert mean_average_precision(sim) == pytest.approx(1.0)


def test_median_rank_perfect_is_one():
    sim = cosine_similarity_matrix(np.eye(5), np.eye(5))
    assert median_rank(sim) == 1.0


def test_retrieval_metrics_keys():
    sim = cosine_similarity_matrix(np.eye(8), np.eye(8))
    m = retrieval_metrics(sim)
    assert "recall@1" in m and "recall@5" in m and "recall@10" in m
    assert "map" in m and "median_rank" in m


def test_info_nce_decreases_with_alignment():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(64, 16))
    b = rng.normal(size=(64, 16))
    loss_random = info_nce_temperature(a, b)
    loss_aligned = info_nce_temperature(a, a)
    assert loss_aligned < loss_random


def test_info_nce_perfect_alignment_is_floor():
    a = normalize_embeddings(np.eye(4))
    loss = info_nce_temperature(a, a, temperature=1.0)
    # perfect one-hot alignment: logits diag=1, off-diag=0 ->
    # loss = log(1 + (N-1) * exp(-1))
    expected = np.log(1.0 + 3.0 * np.exp(-1.0))
    assert loss == pytest.approx(expected, abs=0.01)
