"""Model definitions and serialization-safe loading helpers."""

from tensorflow import keras

from .band_structure_encoder import BandStructureEncoder, MeanPooling, PositionalEncoding, SSLEncoder


def load_ssl_encoder(path, compile=False):
    """Load a project SSL encoder with explicit custom-object registration."""
    custom_objects = {
        "BandStructureAI>PositionalEncoding": PositionalEncoding,
        "BandStructureAI>BandStructureEncoder": BandStructureEncoder,
        "BandStructureAI>MeanPooling": MeanPooling,
        "BandStructureAI>SSLEncoder": SSLEncoder,
    }
    return keras.models.load_model(path, compile=compile, custom_objects=custom_objects)


__all__ = ["BandStructureEncoder", "SSLEncoder", "load_ssl_encoder"]
