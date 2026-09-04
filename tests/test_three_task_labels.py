"""Regression tests for the three-task label reframing (P0A).

Contract:
- line_mode_topology: derived exclusively from the line-mode path:
  0 = metal (Fermi crossing on the path),
  1 = direct (VBM/CBM extrema at the same k index),
  2 = indirect (different k indices).
- provider_global_electronic_type: 0/1/2 from the provider metadata
  (is_metal / is_direct), never from the path.
- line_global_disagreement: binary, 1 iff the two disagree.
- Label derivation must be a pure function of resampled edge envelopes,
  crossing evidence and provider labels — no hidden state.
"""
import numpy as np

from src.data.ood_tensor_builder import (
    derive_line_mode_topology,
    derive_provider_global_type,
    derive_line_global_disagreement,
)


def _envelopes(direct: bool, crossing: bool = False, nk: int = 128):
    x = np.arange(nk, dtype=np.float64)
    center = nk // 2
    vbm = -1.0 + 0.9 * np.exp(-((x - center) ** 2) / (2 * 18.0 ** 2))
    cbm = 1.0 - 0.9 * np.exp(-((x - center) ** 2) / (2 * 18.0 ** 2))
    if not direct:
        cbm = np.roll(cbm, 23)  # CBM minimum moves to a different k index
    if crossing:
        vbm[center] = 0.5  # occupied edge pokes above E_F mid-path
        cbm[center] = -0.5
    return vbm, cbm


def test_line_mode_topology_direct():
    vbm, cbm = _envelopes(direct=True)
    assert derive_line_mode_topology(vbm, cbm, crossing=False) == 1


def test_line_mode_topology_indirect():
    vbm, cbm = _envelopes(direct=False)
    assert derive_line_mode_topology(vbm, cbm, crossing=False) == 2


def test_line_mode_topology_metal_from_crossing_only():
    vbm, cbm = _envelopes(direct=True, crossing=True)
    assert derive_line_mode_topology(vbm, cbm, crossing=True) == 0


def test_provider_global_type_is_provider_only():
    assert derive_provider_global_type(is_metal=True, is_direct=None) == 0
    assert derive_provider_global_type(is_metal=False, is_direct=True) == 1
    assert derive_provider_global_type(is_metal=False, is_direct=False) == 2
    assert derive_provider_global_type(is_metal=False, is_direct=None) == -1


def test_line_global_disagreement_is_xor():
    assert derive_line_global_disagreement(0, 0) == 0
    assert derive_line_global_disagreement(0, 1) == 1
    assert derive_line_global_disagreement(1, 1) == 0
    assert derive_line_global_disagreement(2, 1) == 1
