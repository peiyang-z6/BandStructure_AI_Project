"""
Phase 5 inverse generator for Band-to-Structure style experiments.

The generator is deliberately lightweight and decoupled from the core 4-step
pipeline. It maps target band properties to coarse material descriptors.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="BandStructureAI")
class InverseDescriptorGenerator(keras.Model):
    """MLP/CVAE-like generator from target properties to material descriptors."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_spacegroups: int = 231,
        num_element_bins: int = 96,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_spacegroups = num_spacegroups
        self.num_element_bins = num_element_bins

        self.condition_encoder = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="gelu"),
                layers.Dense(hidden_dim, activation="gelu"),
            ],
            name="condition_encoder",
        )
        self.mu = layers.Dense(latent_dim, name="latent_mu")
        self.logvar = layers.Dense(latent_dim, name="latent_logvar")
        self.decoder = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="gelu"),
                layers.Dense(hidden_dim, activation="gelu"),
            ],
            name="descriptor_decoder",
        )
        self.spacegroup_head = layers.Dense(num_spacegroups, activation="softmax", name="spacegroup")
        self.element_head = layers.Dense(num_element_bins, activation="sigmoid", name="element_probability")

    def sample_latent(self, mu, logvar, training=False):
        if not training:
            return mu
        eps = tf.random.normal(tf.shape(mu), dtype=mu.dtype)
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, targets, training=False):
        features = self.condition_encoder(targets, training=training)
        mu = self.mu(features)
        logvar = self.logvar(features)
        z = self.sample_latent(mu, logvar, training=training)
        decoded = self.decoder(z, training=training)
        return {
            "spacegroup": self.spacegroup_head(decoded),
            "element_probability": self.element_head(decoded),
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "num_spacegroups": self.num_spacegroups,
                "num_element_bins": self.num_element_bins,
            }
        )
        return config


def build_inverse_generator(
    target_dim: int = 4,
    latent_dim: int = 64,
    hidden_dim: int = 128,
    num_spacegroups: int = 231,
    num_element_bins: int = 96,
) -> keras.Model:
    inputs = keras.Input(shape=(target_dim,), name="target_properties")
    generator = InverseDescriptorGenerator(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_spacegroups=num_spacegroups,
        num_element_bins=num_element_bins,
    )
    outputs = generator(inputs)
    return keras.Model(inputs=inputs, outputs=outputs, name="inverse_descriptor_generator")
