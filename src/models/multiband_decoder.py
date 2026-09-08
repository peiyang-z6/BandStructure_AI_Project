"""P3 variable multi-band decoder (Constitution 5.0 §8 P3).

Predicts selected-window energies up to max_bands. A supplied target band mask
handles variable supervised cardinality; the model does not infer band count.

Starting-point architecture (deliberately simple, to be extended):
- `CGCNNEncoder` (P2, reused) -> structure embedding z (B, d_model);
- each (band_slot, k_index) position is conditioned on z + a learned band
  embedding + a sine/cosine positional encoding of the normalized k-axis;
- a small MLP head maps the conditioned vector to one energy value.

Optional pre-normalized self-attention operates on k tokens, segment-masked when
IDs are supplied. Training uses per-k spectral OT. Whole-trajectory matching,
full reciprocal-space conditioning and the full P3 physics constraints remain
outstanding; this is not a Bandformer reproduction or a P3 acceptance claim.

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
        attention_layers: int = 0,
        attention_heads: int = 4,
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
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.k_attention = [layers.MultiHeadAttention(
            num_heads=attention_heads, key_dim=d_model // attention_heads,
            dropout=dropout_rate, name=f"k_attention_{i}") for i in range(attention_layers)]
        self.k_norm1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(attention_layers)]
        self.k_norm2 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(attention_layers)]
        self.k_ffn = [keras.Sequential([
            layers.Dense(hidden_dim, activation="gelu"), layers.Dense(d_model)
        ]) for _ in range(attention_layers)]
        self.k_dropout = layers.Dropout(dropout_rate)

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
        # Attention operates on (B,K,D), not the much larger (B*bands,K,D).
        # A zero-layer model remains the original MLP architecture.
        k_tokens = z[:, None, :] + k_pe[None, :, :]
        segments = graph.get("segment_ids")
        attention_mask = None
        if segments is not None:
            segments = tf.convert_to_tensor(segments)
            if segments.shape.rank == 1:
                segments = tf.broadcast_to(segments[None, :], tf.shape(k_tokens)[:2])
            attention_mask = tf.equal(segments[:, :, None], segments[:, None, :])
        for attention, norm1, norm2, ffn in zip(
                self.k_attention, self.k_norm1, self.k_norm2, self.k_ffn):
            normalized = norm1(k_tokens)
            attended = attention(normalized, normalized, attention_mask=attention_mask, training=training)
            k_tokens = k_tokens + self.k_dropout(attended, training=training)
            k_tokens = k_tokens + self.k_dropout(ffn(norm2(k_tokens), training=training), training=training)
        h = k_tokens[:, None, :, :] + self.band_embedding[None, :, None, :]
        out = self.head(h, training=training)  # (B, max_bands, n_k, 1)
        return tf.squeeze(out, axis=-1)  # (B, max_bands, n_k)

    def get_build_config(self):
        # Weight shapes do not depend on the number of atoms in a graph.
        return {"graph_atoms": 1}

    def build_from_config(self, config):
        # Multi-input subclassed models must create *all* layer variables
        # before Keras loads saved weights; an unbuilt reload can look valid
        # yet yield freshly initialized predictions on its first real call.
        graph = {
            "atom_features": tf.zeros((1, 1, self.num_elements + 1)),
            "neighbor_list": tf.fill((1, 1, self.max_neighbors), -1),
            "neighbor_dist": tf.zeros((1, 1, self.max_neighbors)),
        }
        self(graph, tf.linspace(0., 1., self.n_k), training=False)

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
            "attention_layers": self.attention_layers,
            "attention_heads": self.attention_heads,
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

    Sorting gives the equal-cardinality 1D spectral OT solution at each k.
    It is invariant to independent per-k permutations, not a globally consistent
    assignment of whole band trajectories. It cannot certify continuity or
    wavefunction identity through crossings.

    Padding bands (mask False) are pushed to +inf so they sort to the end and
    are excluded from the loss via the sorted mask.
    """
    pred = tf.debugging.check_numerics(
        tf.where(mask[..., None], pred, tf.zeros_like(pred)), "nonfinite valid prediction")
    target = tf.debugging.check_numerics(
        tf.where(mask[..., None], target, tf.zeros_like(target)), "nonfinite valid target")
    m = tf.cast(mask, tf.float32)  # (B, max_bands)
    large = float("inf")
    pred_m = tf.where(mask[..., None], pred, tf.fill(tf.shape(pred), large))
    tgt_m = tf.where(mask[..., None], target, tf.fill(tf.shape(target), large))
    pred_s = tf.sort(pred_m, axis=1)
    tgt_s = tf.sort(tgt_m, axis=1)
    mask_s = tf.sort(m, axis=1, direction="DESCENDING")  # valid bands first

    # Remove padding *before* subtraction; invalid real errors must fail,
    # never be converted to an apparently perfect zero loss.
    pred_s = tf.where(mask_s[..., None] > 0, pred_s, tf.zeros_like(pred_s))
    tgt_s = tf.where(mask_s[..., None] > 0, tgt_s, tf.zeros_like(tgt_s))
    masked = tf.debugging.check_numerics(tf.abs(pred_s - tgt_s), "nonfinite OT error")
    denom = tf.reduce_sum(mask_s) * tf.cast(tf.shape(pred)[-1], tf.float32) + 1e-8
    return tf.debugging.check_numerics(
        tf.reduce_sum(masked) / denom, "nonfinite OT loss")
