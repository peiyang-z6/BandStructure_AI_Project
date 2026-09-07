"""Regression tests for P3 variable multi-band extraction (fermi-window select +
shape-preserving resample + band mask)."""
import numpy as np
import pytest

from src.data.multiband import (
    extract_fermi_bands,
    select_fermi_bands,
)


def _mk_bands(rows):
    """Build an (num_bands, num_kpoints) energies array from row-wise bands."""
    return np.asarray(rows, dtype=np.float32)


def test_select_window_overlap():
    # band0 min=max=-3 (outside [-2,2]); band1 =0 (inside); band2=3 (outside)
    energies = _mk_bands([[-3.0, -3.0, -3.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=2.0)
    assert list(idx) == [1]


def test_select_window_includes_crossing_band():
    # a band crossing E_F spans the window on both sides
    energies = _mk_bands([[-1.0, 0.0, 1.0], [2.0, 2.0, 2.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=2.0)
    assert 0 in idx  # crossing band selected


def test_select_truncates_to_nearest_bands():
    # 5 bands all inside a wide window; max_bands=3 keeps the 3 nearest E_F
    centers = np.array([-0.1, -0.2, 0.05, 0.3, 1.0], dtype=np.float32)
    energies = np.stack([np.full(4, c) for c in centers])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=3, delta_e=10.0)
    assert len(idx) == 3
    kept_centers = [float(c) for c in sorted(centers[idx])]
    # nearest three to 0.0 are -0.1, 0.05, -0.2
    assert kept_centers == pytest.approx([-0.2, -0.1, 0.05])


def test_select_fallback_when_window_empty():
    # window too narrow -> fallback selects the two bands nearest E_F
    energies = _mk_bands([[-5.0, -5.0], [1.0, 1.0], [9.0, 9.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=0.1)
    assert len(idx) >= 2
    assert set(idx.tolist()) <= {0, 1, 2}
    assert 1 in idx  # band at 1.0 is nearest to E_F=0


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
    # monotonic increasing
    assert np.all(np.diff(bands[0]) >= -1e-5)
