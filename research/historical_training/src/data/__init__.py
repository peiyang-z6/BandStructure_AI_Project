"""Data download and tensor processing modules.

Heavy provider clients are imported lazily so tensor-only tools and tests do not
require pymatgen/mp-api to be installed.
"""
from __future__ import annotations

from typing import Any

__all__ = ["MPAdapter", "AFLOWAdapter", "load_api_key", "load_api_keys"]


def __getattr__(name: str) -> Any:
    if name in {"MPAdapter", "load_api_key", "load_api_keys"}:
        from . import mp_adapter

        return getattr(mp_adapter, name)
    if name == "AFLOWAdapter":
        from .aflow_adapter import AFLOWAdapter

        return AFLOWAdapter
    raise AttributeError(name)
