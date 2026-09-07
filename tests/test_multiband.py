"""Regression tests for P3 variable multi-band extraction (Fermi-proximate band
select + shape-preserving resample + band mask)."""
import numpy as np
import pytest

from src.data.multiband import (
    extract_fermi_bands,
    select_fermi_bands,
)


def _mk_bands(rows):
    """Build an (num_bands, num_kpoints) energies array from row-wise bands."""
    return np.asarray(rows, dtype=np.float32)


def test_select_insulator_keeps_both_sides_of_gap():
    # AFLOW anchors E_F at VBM; valence band at -1, conduction at +0.5/+2/+5.
    # max_bands=4 -> valence side + conduction side must BOTH be represented.
    energies = _mk_bands([[-1.0, -1.0], [0.5, 0.5], [2.0, 2.0], [5.0, 5.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=4, delta_e=5.0)
    assert 0 in idx  # valence band (VBM side)
    assert 1 in idx  # nearest conduction band (CBM side)
    assert 5.0 not in energies[idx].min(axis=1) or 2 in idx


def test_select_insulator_prefers_vbm_and_cbm():
    # many valence bands (0,-0.5,-1.0) and conduction (2,3,4); max_bands=4
    # -> 2 valence nearest VBM + 2 conduction nearest CBM
    rows = [[0.0], [-0.5], [-1.0], [2.0], [3.0], [4.0]]
    energies = _mk_bands(rows)
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=4, delta_e=5.0)
    assert len(idx) == 4
    selected = energies[idx].reshape(-1)
    # VBM side: 0.0 and -0.5 (nearest valence); CBM side: 2.0 and 3.0
    assert 0.0 in selected and -0.5 in selected
    assert 2.0 in selected and 3.0 in selected


def test_select_metal_window_with_crossing_band():
    # no valence band (band_max<=0 empty) -> metal path, window overlap
    energies = _mk_bands([[-1.0, 1.0], [2.0, 3.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=2.0)
    assert 0 in idx  # crossing band selected


def test_select_metal_truncates_to_nearest_bands():
    # all bands cross E_F (metal): keep the 3 nearest to E_F
    centers = np.array([-0.1, -0.2, 0.05, 0.3, 1.0], dtype=np.float32)
    energies = np.stack([np.full(4, c) for c in centers])
    # make them all cross E_F by adding a +/-0.5 sweep
    energies = energies + np.array([[0.5, -0.5, 0.5, -0.5]], dtype=np.float32)
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=3, delta_e=10.0)
    assert len(idx) <= 3
    assert len(idx) >= 1


def test_select_fallback_when_window_empty():
    # metal path with empty window -> fallback selects bands nearest E_F
    energies = _mk_bands([[1.0, 1.0], [9.0, 9.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=0.1)
    assert len(idx) >= 1
    assert 0 in idx  # band at 1.0 is nearest to E_F=0


def test_extract_shape_and_mask():
    # 2 bands selected, max_bands=4 -> padded (4, n_k), mask [T,T,F,F]
    energies = _mk_bands([[-2.0, -2.0, -2.0, -2.0], [1.0, 1.0, 1.0, 1.0]])
    k_dist = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    bands, mask, k_axis = extract_fermi_bands(
        energies, k_dist, efermi=0.0, max_bands=4, delta_e=5.0, n_k=8
    )
    assert bands.shape == (4, 8)
    assert mask.tolist() == [True, True, False, False]
    assert k_axis.shape == (8,)
    assert np.allclose(k_axis[0], 0.0) and np.allclose(k_axis[-1], 1.0)
    # constant bands stay constant after resampling
    assert np.allclose(bands[0], -2.0, atol=1e-4)
    assert np.allclose(bands[1], 1.0, atol=1e-4)
    # padded rows are zero
    assert np.allclose(bands[2], 0.0)
    assert np.allclose(bands[3], 0.0)


def test_extract_linear_band_preserved():
    # a linear band resampled to a different grid keeps its endpoints
    energies = _mk_bands([np.linspace(0.0, 1.0, 5)])
    k_dist = np.linspace(0.0, 1.0, 5)
    bands, mask, k_axis = extract_fermi_bands(
        energies, k_dist, efermi=0.0, max_bands=1, delta_e=5.0, n_k=11
    )
    assert mask[0]
    assert bands[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert bands[0, -1] == pytest.approx(1.0, abs=1e-4)
    assert np.all(np.diff(bands[0]) >= -1e-5)
