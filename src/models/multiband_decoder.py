"""P3 variable multi-band decoder (Constitution 5.0 §8 P3).

Predicts Fermi-proximate bands E_n(k) from a crystal graph, with a VARIABLE
band count handled by a boolean band mask (unlike Bandformer's fixed count).

Starting-point architecture (deliberately simple, to be extended):
- `CGCNNEncoder` (P2, reused) -> structure embedding z (B, d_model);
- each (band_slot, k_index) position is conditioned on z + a learned band
  embedding + a sine/cosine positional encoding of the normalized k-axis;
- a small MLP head maps the conditioned vector to one energy value.

The band mask is applied only in the loss (masked MAE); band-set matching
(Hungarian / optimal transport) and physics-constraint losses are added in a
later step of 3c, once this baseline trains.

Usage: see scripts/train_p3_decoder.py.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.crystal_graph_encoder import CGCNNEncoder


def k_positional_encoding(k_axis: tf.Tensor, d_model: int) -> tf.Tensor:
    """(n_k,) normalized k-axis -> (n_k, d_model) sine/cosine encoding."""
    k_axis = tf.cast(k_axis, tf.float32)  # (n_k,)
    n_half = d_model // 2
    freqs = tf.range(n_half, dtype=tf.float32)  # (n_half,)
    freqs = 2.0 ** (freqs / tf.maximum(tf.cast(n_half - 1, tf.float32), 1.0))
    angles = tf.matmul(tf.expand_dims(k_axis, -1), tf.expand_dims(freqs, 0)) * 3.14159265  # (n_k, n_half)
    pe = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # (n_k, d_model)
    return pe


@keras.utils.register_keras_serializable(package="BandStructureAI")
class MultiBandDecoder(keras.Model):
    """Structure -> (max_bands, n_k) band energies with a band mask."""

    def __init__(
        self,
        max_bands: int = 16,
        n_k: int = 256,
        d_model: int = 128,
        hidden_dim: int = 256,
        num_elements: int = 109,
        conv_layers: int = 3,
        rbf_bins: int = 40,
        max_neighbors: int = 12,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_bands = max_bands
        self.n_k = n_k
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_elements = num_elements
        self.conv_layers = conv_layers
        self.rbf_bins = rbf_bins
        self.max_neighbors = max_neighbors
        self.dropout_rate = dropout_rate

        self.encoder = CGCNNEncoder(
            num_elements=num_elements,
            conv_layers=conv_layers,
            hidden_dim=d_model,
            rbf_bins=rbf_bins,
            max_neighbors=max_neighbors,
            embedding_dim=d_model,
            dropout_rate=dropout_rate,
        )
        self.band_embedding = self.add_weight(
            name="band_embedding",
            shape=(max_bands, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.head = keras.Sequential([
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dense(1, activation=None),
        ])

    def call(self, graph, k_axis, training=False):
        z = self.encoder(graph, training=training)  # (B, d_model)
        k_pe = k_positional_encoding(k_axis, self.d_model)  # (n_k, d_model)
        # (B, max_bands, n_k, d_model) = z + band_emb + k_pe
        h = (
            z[:, None, None, :]
            + self.band_embedding[None, :, None, :]
            + k_pe[None, None, :, :]
        )
        out = self.head(h, training=training)  # (B, max_bands, n_k, 1)
        return tf.squeeze(out, axis=-1)  # (B, max_bands, n_k)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_bands": self.max_bands,
            "n_k": self.n_k,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "num_elements": self.num_elements,
            "conv_layers": self.conv_layers,
            "rbf_bins": self.rbf_bins,
            "max_neighbors": self.max_neighbors,
            "dropout_rate": self.dropout_rate,
        })
        return config


def masked_mae(pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masked mean absolute error over (B, max_bands, n_k), mask is (B, max_bands)."""
    err = tf.abs(pred - target)  # (B, max_bands, n_k)
    masked = err * tf.cast(mask, tf.float32)[..., None]
    denom = tf.reduce_sum(tf.cast(mask, tf.float32)) * tf.cast(tf.shape(pred)[-1], tf.float32) + 1e-8
    return tf.reduce_sum(masked) / denom


def sorted_masked_mae(pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masked MAE with per-k-point 1D optimal transport (sorting) band matching.

    Band identity swaps at high-symmetry crossings; a fixed band-slot MAE
    wrongly penalizes a correct prediction whose bands exchange order at a
    crossing. Sorting each k-point's band energies independently is the exact
    solution of the 1D optimal-transport problem, so it aligns bands by energy
    (not by slot) and only measures the energy discrepancy.

    Padding bands (mask False) are pushed to +inf so they sort to the end and
    are excluded from the loss via the sorted mask.
    """
    m = tf.cast(mask, tf.float32)  # (B, max_bands)
    large = 1e9
    pred_m = tf.where(mask[..., None], pred, tf.fill(tf.shape(pred), large))
    tgt_m = tf.where(mask[..., None], target, tf.fill(tf.shape(target), large))
    pred_s = tf.sort(pred_m, axis=1)
    tgt_s = tf.sort(tgt_m, axis=1)
    mask_s = tf.sort(m, axis=1, direction="DESCENDING")  # valid bands first

    err = tf.abs(pred_s - tgt_s)  # (B, max_bands, n_k); padding pairs -> inf-inf = nan
    err = tf.where(tf.math.is_finite(err), err, 0.0)
    masked = err * mask_s[..., None]
    denom = tf.reduce_sum(mask_s) * tf.cast(tf.shape(pred)[-1], tf.float32) + 1e-8
    return tf.reduce_sum(masked) / denom
