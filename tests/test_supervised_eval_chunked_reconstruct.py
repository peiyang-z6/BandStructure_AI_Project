"""Regression tests: evaluation-only tensor reconstruction must be chunked.

v7 OOM incident: reconstruct_raw_tensors fed the whole outer test (11,987
samples) to encoder.reconstruct in one forward pass; the 4-head attention
softmax needed a (11987, 4, 128, 128) tensor (~3.1 GB/layer) and blew the
16 GB V100. The fix mirrors extract_encoder_features_batched: bounded chunks
with concatenated outputs. These tests pin that contract.
"""
import numpy as np
import pytest


def _fake_model(factor: float = 2.0):
    import tensorflow as tf

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def reconstruct(self, x, training=False):
            self.calls.append(int(x.shape[0]))
            return tf.constant(x * factor)

    return type("FakeModel", (), {"encoder": FakeEncoder()})()


def test_reconstruct_chunks_never_exceed_batch_limit():
    from scripts.finetune_supervised import reconstruct_encoder_chunks

    model = _fake_model()
    x = np.random.RandomState(0).rand(11_987, 128, 6).astype(np.float32)
    out = reconstruct_encoder_chunks(model, x, chunk_size=1024)
    batch_sizes = model.encoder.calls
    assert sum(batch_sizes) == len(x)
    assert max(batch_sizes) <= 1024
    assert len(batch_sizes) == 12  # ceil(11987 / 1024)
    assert out.shape == x.shape


def test_reconstruct_chunks_equal_full_batch_result():
    from scripts.finetune_supervised import reconstruct_encoder_chunks

    model = _fake_model(factor=1.5)
    x = np.random.RandomState(1).rand(500, 128, 6).astype(np.float32)
    chunked = reconstruct_encoder_chunks(model, x, chunk_size=128)
    one_shot = model.encoder.reconstruct(x, training=False).numpy()
    np.testing.assert_allclose(chunked, one_shot, rtol=0, atol=0)


def test_reconstruct_chunks_small_input_single_chunk():
    from scripts.finetune_supervised import reconstruct_encoder_chunks

    model = _fake_model()
    x = np.random.RandomState(2).rand(64, 128, 6).astype(np.float32)
    out = reconstruct_encoder_chunks(model, x, chunk_size=1024)
    assert model.encoder.calls == [64]
    assert out.shape == x.shape


def test_reconstruct_chunks_rejects_nonpositive_chunk_size():
    from scripts.finetune_supervised import reconstruct_encoder_chunks

    model = _fake_model()
    x = np.zeros((4, 128, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        reconstruct_encoder_chunks(model, x, chunk_size=0)
    with pytest.raises(ValueError):
        reconstruct_encoder_chunks(model, x, chunk_size=-1)


def test_reconstruct_raw_tensors_uses_chunked_path():
    from scripts.finetune_supervised import reconstruct_raw_tensors

    model = _fake_model()
    x = np.random.RandomState(3).rand(2300, 128, 6).astype(np.float32)
    # normalization with identity stats
    stats = {"mean": [0.0] * 6, "std": [1.0] * 6}
    import json
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        norm_path = os.path.join(tmp, "norm.json")
        with open(norm_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh)
        recon = reconstruct_raw_tensors(model, x, norm_path, chunk_size=1024)
        assert recon.shape == (2300, 2, 128, 3)
        assert max(model.encoder.calls) <= 1024
        # identity stats: denormalized output equals raw input reshaped
        expected = x.reshape(2300, 128, 2, 3).transpose(0, 2, 1, 3)
        np.testing.assert_allclose(recon, expected * 2.0, rtol=0, atol=1e-6)
