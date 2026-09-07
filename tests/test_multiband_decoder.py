"""Regression tests for the P3 variable multi-band decoder."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.multiband_decoder import (
    MultiBandDecoder,
    k_positional_encoding,
    masked_mae,
)


def _graph(batch=2, n_atoms=5):
    return {
        "atom_features": np.random.rand(batch, n_atoms, 110).astype(np.float32),
        "neighbor_list": np.random.randint(0, n_atoms, size=(batch, n_atoms, 12)).astype(np.int32),
        "neighbor_dist": np.random.rand(batch, n_atoms, 12).astype(np.float32),
    }


def test_k_positional_encoding_shape():
    k_axis = tf.linspace(0.0, 1.0, 256)
    pe = k_positional_encoding(k_axis, 128)
    assert pe.shape == (256, 128)


def test_decoder_forward_shape():
    model = MultiBandDecoder(max_bands=16, n_k=256, d_model=32, hidden_dim=64)
    g = {k: tf.constant(v) for k, v in _graph().items()}
    k_axis = tf.linspace(0.0, 1.0, 256)
    out = model(g, k_axis)
    assert out.shape == (2, 16, 256)


def test_masked_mae_ignores_padding():
    # (B=1, max_bands=2, n_k=2); band 1 masked out
    pred = tf.constant([[[1.0, 1.0], [2.0, 2.0]]])
    target = tf.constant([[[1.0, 1.0], [9.0, 9.0]]])
    mask = tf.constant([[True, False]])
    # only band 0 counted: |1-1|+|1-1|=0 -> 0
    assert float(masked_mae(pred, target, mask).numpy()) == pytest.approx(0.0, abs=1e-6)


def test_masked_mae_counts_only_masked():
    pred = tf.constant([[[0.0, 0.0], [0.0, 0.0]]])
    target = tf.constant([[[2.0, 2.0], [2.0, 2.0]]])
    mask = tf.constant([[True, False]])
    # one band (2 positions) with |0-2|=2 each -> mean = 2.0
    assert float(masked_mae(pred, target, mask).numpy()) == pytest.approx(2.0, abs=1e-5)
