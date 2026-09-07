"""CGCNN-style crystal graph encoder (pure TensorFlow/Keras).

Constitution 5.0 §8 P2: reuse a mature GNN rather than inventing one. This
implements the CGCNN message-passing recipe (Xie & Grossman 2018) on top of
the existing Keras stack so it trains in the same framework as the band
encoder (no torch/e3nn dependency).

Design (node dimension held constant at `hidden_dim`):
- atom one-hot (num_elements+1) -> Dense(hidden_dim) node features;
- per atom: gather the 12 nearest-neighbour atom features, gate them by an
  RBF-expanded inter-atomic distance (Dense(hidden_dim, sigmoid)), sum ->
  neighbour message;
- per conv layer: concat [node, message] (2*hidden_dim) -> Dense(hidden_dim)
  + LayerNorm + Dropout, then residual skip `node = node + updated`;
- mean-pool over real atoms (padding rows masked) -> Dense(embedding_dim).

`call` takes a dict {atom_features, neighbor_list, neighbor_dist} and returns
a (batch, embedding_dim) vector.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="BandStructureAI")
class CGCNNEncoder(keras.Model):
    """CGCNN-style crystal graph encoder producing a fixed graph embedding."""

    def __init__(
        self,
        num_elements: int = 109,
        conv_layers: int = 3,
        hidden_dim: int = 128,
        rbf_bins: int = 40,
        max_neighbors: int = 12,
        embedding_dim: int = 128,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_elements = num_elements
        self.conv_layers = conv_layers
        self.hidden_dim = hidden_dim
        self.rbf_bins = rbf_bins
        self.max_neighbors = max_neighbors
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.atom_embedding = layers.Dense(hidden_dim, activation="gelu")
        # RBF expansion of raw inter-atomic distances -> gate input.
        self.rbf_centers = tf.linspace(0.0, 8.0, rbf_bins)
        self.rbf_width = 8.0 / rbf_bins
        self.distance_gate = layers.Dense(hidden_dim, activation="sigmoid")
        self.convs = [
            layers.Dense(hidden_dim, activation="gelu", name=f"cgcnn_conv_{i}")
            for i in range(conv_layers)
        ]
        self.norms = [
            layers.LayerNormalization(epsilon=1e-6, name=f"cgcnn_norm_{i}")
            for i in range(conv_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)
        self.pool_projection = layers.Dense(embedding_dim, activation=None)

    def _rbf_expand(self, distances):
        """(...,) raw distances -> (..., rbf_bins) Gaussian RBF."""
        d = tf.expand_dims(distances, -1)  # (..., 1)
        diff = d - self.rbf_centers  # (..., bins)
        return tf.exp(-(diff ** 2) / (2.0 * self.rbf_width ** 2))

    def call(self, inputs, training=False):
        atom_features = inputs["atom_features"]
        neighbor_list = inputs["neighbor_list"]
        neighbor_dist = inputs["neighbor_dist"]

        node = self.atom_embedding(atom_features)  # (B, N, hidden_dim)
        dist_gate = self.distance_gate(self._rbf_expand(neighbor_dist))  # (B, N, K, hidden_dim)
        neighbor_mask = tf.cast(tf.greater_equal(neighbor_list, 0), tf.float32)

        for conv, norm in zip(self.convs, self.norms):
            neighbor_node = tf.gather(node, tf.maximum(neighbor_list, 0), batch_dims=1)
            neighbor_node = neighbor_node * neighbor_mask[..., None]
            message = tf.reduce_sum(neighbor_node * dist_gate, axis=2)  # (B, N, hidden_dim)
            combined = tf.concat([node, message], axis=-1)  # (B, N, 2*hidden_dim)
            updated = norm(conv(combined))
            updated = self.dropout(updated, training=training)
            node = node + updated  # residual

        atom_present = tf.cast(tf.reduce_sum(atom_features, axis=-1) > 0.0, tf.float32)
        denom = tf.reduce_sum(atom_present, axis=1, keepdims=True) + 1e-8
        pooled = tf.reduce_sum(node * atom_present[..., None], axis=1) / denom
        return self.pool_projection(pooled)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_elements": self.num_elements,
            "conv_layers": self.conv_layers,
            "hidden_dim": self.hidden_dim,
            "rbf_bins": self.rbf_bins,
            "max_neighbors": self.max_neighbors,
            "embedding_dim": self.embedding_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
