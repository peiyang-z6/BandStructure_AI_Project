"""
Band Structure Encoder
======================
Transformer-based encoder for 1D band structure sequences.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="BandStructureAI")
class PositionalEncoding(layers.Layer):
    """
    Positional encoding for 1D sequences.
    """

    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": int(self.pe.shape[1])})
        return config


@keras.utils.register_keras_serializable(package="BandStructureAI")
class BandStructureEncoder(keras.Model):
    """
    Transformer-based encoder for band structure sequences.

    Architecture:
    - Input: band-derived features at k-points, shape [batch, seq_len, num_features]
    - Output: Encoded representations [batch, seq_len, d_model]
    """

    def __init__(
        self,
        num_features=64,
        seq_len=1024,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dff=512,
        dropout_rate=0.1,
        **kwargs
    ):
        super(BandStructureEncoder, self).__init__(**kwargs)

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.band_projection = layers.Dense(d_model, activation='gelu')
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout = layers.Dropout(dropout_rate)

        self.transformer_layers = [
            self._create_transformer_layer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.ln_attn = [
            layers.LayerNormalization(epsilon=1e-6, name=f'ln_attn_{i}')
            for i in range(num_layers)
        ]
        self.ln_ffn = [
            layers.LayerNormalization(epsilon=1e-6, name=f'ln_ffn_{i}')
            for i in range(num_layers)
        ]
        self.add_attn = [
            layers.Add(name=f'add_attn_{i}')
            for i in range(num_layers)
        ]
        self.add_ffn = [
            layers.Add(name=f'add_ffn_{i}')
            for i in range(num_layers)
        ]

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        # Per-layer FFN (each transformer block gets its own FFN parameters)
        self.ffn_dense1 = [
            layers.Dense(dff, activation='gelu', name=f'ffn_dense1_{i}')
            for i in range(num_layers)
        ]
        self.ffn_dense2 = [
            layers.Dense(d_model, name=f'ffn_dense2_{i}')
            for i in range(num_layers)
        ]
        self.ffn_dropout = [
            layers.Dropout(dropout_rate, name=f'ffn_dropout_{i}')
            for i in range(num_layers)
        ]

    def _create_transformer_layer(self, d_model, num_heads, dff, dropout_rate):
        return layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

    def call(self, x, training=False, mask=None):
        x = self.band_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i, transformer_layer in enumerate(self.transformer_layers):
            attn_output = transformer_layer(x, x, attention_mask=mask)
            x = self.add_attn[i]([x, attn_output])
            x = self.ln_attn[i](x)

            ff_output = self._feed_forward_network(x, i, training)
            x = self.add_ffn[i]([x, ff_output])
            x = self.ln_ffn[i](x)

        return x

    def _feed_forward_network(self, x, layer_idx, training):
        inner_layer = self.ffn_dense1[layer_idx](x)
        output = self.ffn_dense2[layer_idx](inner_layer)
        output = self.ffn_dropout[layer_idx](output, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


@keras.utils.register_keras_serializable(package="BandStructureAI")
class MeanPooling(keras.Model):
    """
    Mean pooling layer for sequence aggregation.
    """

    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            mask = tf.cast(mask, tf.float32)
            sum_masked = tf.reduce_sum(x * mask, axis=1)
            sum_mask = tf.reduce_sum(mask, axis=1)
            return sum_masked / (sum_mask + 1e-8)
        return tf.reduce_mean(x, axis=1)


@keras.utils.register_keras_serializable(package="BandStructureAI")
class SSLEncoder(keras.Model):
    """
    SSL Pre-trained encoder with projection head and reconstruction head.

    Combines:
    - BandStructureEncoder for sequence encoding
    - Mean pooling for sequence aggregation
    - Projection head for contrastive learning
    - Reconstruction head for MAE reconstruction (applied to full sequence)
    """

    def __init__(
        self,
        num_features=64,
        seq_len=1024,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dff=512,
        projection_dim=128,
        dropout_rate=0.1,
        **kwargs
    ):
        super(SSLEncoder, self).__init__(**kwargs)

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate

        self.encoder = BandStructureEncoder(
            num_features=num_features,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dff=dff,
            dropout_rate=dropout_rate
        )

        self.pooling = MeanPooling()

        self.projection_head = keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dense(d_model),
            layers.Dense(projection_dim)
        ])

        self.reconstruction_head = keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dense(d_model, activation='gelu'),
            layers.Dense(num_features)
        ])

    def call(self, x, training=False, mask=None, return_features=False):
        encoded = self.encoder(x, training=training, mask=mask)
        pooled = self.pooling(encoded, mask=mask)

        if return_features:
            return pooled

        projected = self.projection_head(pooled, training=training)
        return projected

    def reconstruct(self, x, training=False):
        encoded = self.encoder(x, training=training)
        reconstructed = self.reconstruction_head(encoded)
        return reconstructed

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate
        })
        return config
