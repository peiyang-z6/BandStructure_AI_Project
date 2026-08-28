"""Phase 5 multimodal Plot-to-Physics extension.

Submodules are lazy so image parsing/reconstruction can run without importing
the TensorFlow inference stack.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "ApplicationRecommender",
    "BrainPrediction",
    "PhysicsBrainInvoker",
    "MultiFormatParser",
    "ParsedBandData",
    "PhysicsReconstructor",
    "ReconstructedTensor",
    "SyntheticBandPlotGenerator",
    "SyntheticConfig",
]


def __getattr__(name: str) -> Any:
    routes = {
        "ApplicationRecommender": ("brain_invoker", "ApplicationRecommender"),
        "BrainPrediction": ("brain_invoker", "BrainPrediction"),
        "PhysicsBrainInvoker": ("brain_invoker", "PhysicsBrainInvoker"),
        "MultiFormatParser": ("multi_format_parser", "MultiFormatParser"),
        "ParsedBandData": ("multi_format_parser", "ParsedBandData"),
        "PhysicsReconstructor": ("physics_reconstructor", "PhysicsReconstructor"),
        "ReconstructedTensor": ("physics_reconstructor", "ReconstructedTensor"),
        "SyntheticBandPlotGenerator": ("synthetic_data_generator", "SyntheticBandPlotGenerator"),
        "SyntheticConfig": ("synthetic_data_generator", "SyntheticConfig"),
    }
    if name not in routes:
        raise AttributeError(name)
    module_name, attribute = routes[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[attribute])
    return getattr(module, attribute)
