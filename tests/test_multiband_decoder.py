"""Regression tests for the P3 variable multi-band decoder."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.multiband_decoder import (
    MultiBandDecoder,
    k_positional_encoding,
    masked_mae,
    sorted_masked_mae,
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


def test_k_attention_mixes_only_k_points_within_the_same_segment():
    import inspect
    assert "attention_layers" in inspect.signature(MultiBandDecoder).parameters
    tf.keras.utils.set_random_seed(42)
    model = MultiBandDecoder(max_bands=3, n_k=12, d_model=16, hidden_dim=32,
                             attention_layers=2, attention_heads=4, dropout_rate=0.)
    graph = {k: tf.constant(v) for k, v in _graph().items()}
    graph["segment_ids"] = tf.constant([[0]*6+[1]*6]*2)
    k_axis = tf.Variable(tf.linspace(0., 1., 12))
    with tf.GradientTape() as tape:
        out = model(graph, k_axis, training=False)
        selected_output = tf.reduce_sum(out[:, :, 0])
    derivative = tape.gradient(selected_output, k_axis).numpy()
    assert out.shape == (2, 3, 12)
    assert np.linalg.norm(derivative[1:6]) > 1e-7
    assert np.allclose(derivative[6:], 0., atol=1e-7)


def test_decoder_forward_shape():
    model = MultiBandDecoder(max_bands=16, n_k=256, d_model=32, hidden_dim=64)
    g = {k: tf.constant(v) for k, v in _graph().items()}
    k_axis = tf.linspace(0.0, 1.0, 256)
    out = model(g, k_axis)
    assert out.shape == (2, 16, 256)


@pytest.mark.parametrize("attention_layers", [0, 2])
def test_decoder_keras_roundtrip_restores_real_predictions(tmp_path, attention_layers):
    tf.keras.utils.set_random_seed(7)
    model = MultiBandDecoder(max_bands=3, n_k=12, d_model=16, hidden_dim=32,
                             attention_layers=attention_layers, dropout_rate=0.)
    graph = {k: tf.constant(v) for k, v in _graph().items()}
    graph["segment_ids"] = tf.constant([[0]*6+[1]*6]*2)
    axis = tf.linspace(0., 1., 12)
    expected = model(graph, axis, training=False).numpy()
    path = tmp_path / "decoder.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    actual = restored(graph, axis, training=False).numpy()
    assert restored.attention_layers == attention_layers
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


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


def test_sorted_masked_mae_invariant_to_band_order():
    # pred band order swapped vs target (crossing) -> sorted MAE should be 0
    pred = tf.constant([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    target = tf.constant([[[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]])
    mask = tf.constant([[True, True, True]])
    assert float(sorted_masked_mae(pred, target, mask).numpy()) == pytest.approx(0.0, abs=1e-5)


def test_sorted_masked_mae_ignores_padding():
    # valid bands {1.0, 5.0} on both sides; padding band (slot 1) differs but is ignored
    pred = tf.constant([[[1.0, 1.0], [9.0, 9.0], [5.0, 5.0]]])
    target = tf.constant([[[1.0, 1.0], [0.0, 0.0], [5.0, 5.0]]])
    mask = tf.constant([[True, False, True]])
    assert float(sorted_masked_mae(pred, target, mask).numpy()) == pytest.approx(0.0, abs=1e-5)


def test_sorted_loss_fails_on_nonfinite_valid_prediction():
    with pytest.raises(tf.errors.InvalidArgumentError):
        sorted_masked_mae(tf.constant([[[float("nan")]]]), tf.zeros((1, 1, 1)),
                          tf.constant([[True]])).numpy()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "tf_function"])
def test_sorted_loss_rejects_overflow_instead_of_reporting_zero(compiled):
    loss_fn = tf.function(sorted_masked_mae) if compiled else sorted_masked_mae
    with pytest.raises(tf.errors.InvalidArgumentError):
        loss_fn(tf.constant([[[3e38]]]), tf.constant([[[-3e38]]]),
                tf.constant([[True]])).numpy()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "tf_function"])
def test_sorted_loss_rejects_final_reduction_overflow(compiled):
    pred = tf.constant([[[3e38, 3e38]]])
    target = tf.zeros_like(pred)
    assert np.isfinite(tf.abs(pred - target).numpy()).all()
    loss_fn = tf.function(sorted_masked_mae) if compiled else sorted_masked_mae
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonfinite OT loss"):
        loss_fn(pred, target, tf.constant([[True]])).numpy()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "tf_function"])
def test_sorted_loss_preserves_normal_gradients_with_nonfinite_padding(compiled):
    pred = tf.Variable([[[1.0, 3.0], [np.nan, np.inf]]])
    target = tf.constant([[[0.0, 0.0], [np.inf, np.nan]]])
    loss_fn = tf.function(sorted_masked_mae) if compiled else sorted_masked_mae
    with tf.GradientTape() as tape:
        loss = loss_fn(pred, target, tf.constant([[True, False]]))
    gradient = tape.gradient(loss, pred)
    assert float(loss.numpy()) == pytest.approx(np.mean([1.0, 3.0]))
    assert gradient is not None
    np.testing.assert_allclose(gradient.numpy(), [[[0.5, 0.5], [0.0, 0.0]]])


def test_sorted_masked_mae_measures_energy_error():
    pred = tf.constant([[[0.0], [1.0]]])
    target = tf.constant([[[0.5], [1.5]]])
    mask = tf.constant([[True, True]])
    # sorted: |0-0.5| + |1-1.5| = 0.5 + 0.5 = 1.0 over 2 bands -> 0.5
    assert float(sorted_masked_mae(pred, target, mask).numpy()) == pytest.approx(0.5, abs=1e-5)
