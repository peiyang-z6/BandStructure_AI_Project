"""Model definitions used by the current MBM/fine-tuning pipeline."""

from .band_structure_encoder import BandStructureEncoder, SSLEncoder

__all__ = ["BandStructureEncoder", "SSLEncoder"]
