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


@pytest.mark.parametrize("efermi", [0.0, 5.0])
@pytest.mark.parametrize("valence", [[-1.0, -1.0], [-1.0, 0.0], [0.0, 0.0]])
def test_select_keeps_upward_fermi_touching_conduction(efermi, valence):
    energies = _mk_bands([valence, [0.0, 1.0], [2.0, 2.0]]) + efermi
    idx = select_fermi_bands(energies, efermi=efermi, max_bands=2, delta_e=5.0)
    assert idx.tolist() == [0, 1]


def test_crossing_survives_when_deep_valence_and_high_conduction_exist():
    energies = _mk_bands([[-2.0, -2.0], [-1.0, 1.0], [3.0, 3.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=2, delta_e=5.0)
    assert 1 in idx


def test_select_metal_window_with_crossing_band():
    # no valence band (band_max<=0 empty) -> metal path, window overlap
    energies = _mk_bands([[-1.0, 1.0], [2.0, 3.0]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=8, delta_e=2.0)
    assert 0 in idx  # crossing band selected


def test_wide_crossing_has_zero_interval_distance_and_is_retained():
    energies = _mk_bands([[-50.0, 50.0], [-0.1, -0.1], [0.1, 0.1]])
    idx = select_fermi_bands(energies, efermi=0.0, max_bands=1, delta_e=5.0)
    assert idx.tolist() == [0]


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


def test_path_segments_match_resampling_at_discontinuous_boundary():
    from src.data import multiband as module
    assert callable(getattr(module, "path_segments", None))
    source, target = module.path_segments(np.array([0., .5, .5, 1.]), 7)
    assert source.tolist() == [0, 0, 1, 1]
    assert target.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_spin_resolved_selection_indices_reference_all_channels():
    energies = np.array([[[-2., -2.], [1., 1.]], [[-1., -1.], [2., 2.]]])
    assert select_fermi_bands(energies, 0., 4, 5.).tolist() == [0, 1, 2, 3]


def test_extract_includes_both_spin_channels():
    energies = np.array([[[-2., -2.], [1., 1.]], [[-1., -1.], [2., 2.]]])
    bands, mask, _ = extract_fermi_bands(energies, np.array([0., 1.]), 0., 4, 5., 3)
    assert sorted(bands[mask, 0].tolist()) == [-2., -1., 1., 2.]


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


def test_extract_does_not_bridge_duplicate_distance_discontinuity():
    energies = _mk_bands([[-2.0, -2.0, 3.0, 3.0]])
    bands, mask, axis = extract_fermi_bands(
        energies, np.array([0.0, 0.5, 0.5, 1.0]), 0.0, 1, 5.0, 7)
    assert mask[0]
    assert np.all(np.isin(bands[0], [-2.0, 3.0]))


def test_extract_returns_fermi_relative_energies():
    bands, mask, _ = extract_fermi_bands(
        _mk_bands([[4.0, 4.0], [7.0, 7.0]]), np.array([0.0, 1.0]),
        efermi=5.0, max_bands=2, delta_e=5.0, n_k=3)
    assert np.allclose(bands[mask], [[-1.0]*3, [2.0]*3])


@pytest.mark.parametrize("k", [[0., 1., .5], [0., np.nan, 1.], [0., 0., 0.]])
def test_invalid_physical_k_axis_is_rejected(k):
    with pytest.raises(ValueError, match="k_distances"):
        extract_fermi_bands(_mk_bands([[-1., -1., -1.]]), np.array(k), 0., 1, 5., 4)


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


def test_window_audit_distinguishes_full_slots_from_real_capacity_loss():
    from src.data import multiband as module
    assert callable(getattr(module, "band_window_audit", None))
    full = np.tile([-1., 1.], (16, 1))
    result = module.band_window_audit(full, np.arange(16), 0., 16, 5.)
    assert result["window_candidates"] == result["selected"] == 16
    assert result["window_dropped"] == 0
    assert result["slots_full"] is True
    assert result["capacity_insufficient"] is False
    extra = np.vstack([full, [-2., 2.]])
    result = module.band_window_audit(extra, np.arange(16), 0., 16, 5.)
    assert result["window_candidates"] == 17
    assert result["window_dropped_indices"] == [16]
    assert result["capacity_insufficient"] is True
    assert result["capacity_shortfall"] == 1


def test_window_audit_separates_side_quota_loss_from_capacity_and_window():
    from src.data.multiband import band_window_audit
    # Ten occupied plus one empty: 9 selected; spare slots do not undo 8/8 quotas.
    energies = np.array([[float(-i)] * 2 for i in range(1, 11)] + [[20., 20.]])
    selected = select_fermi_bands(energies, 0., 16, 5.)
    before = energies.copy()
    report = band_window_audit(energies, selected, 0., 16, 5.)
    assert report["selection_branch"] == "clean_split"
    assert report["selection_candidates"] == 11
    assert report["selection_dropped"] == 2
    assert report["selected"] == 9 and not report["slots_full"]
    assert not report["capacity_insufficient"]
    assert report["quota_underfill"] == 2
    assert report["side_quotas"] == {"occupied": 8, "empty": 8}
    assert report["window_candidates"] == 5
    assert report["selected_in_window"] == 5
    assert report["selected_outside_window"] == 4
    assert report["scope"] == "selected_window_complete_trajectories_not_energy_clipped"
    np.testing.assert_array_equal(energies, before)


def test_window_audit_tracks_actual_flat_spin_indices_and_omitted_frontiers():
    from src.data.multiband import band_window_audit
    energies = np.array([[[-2., -2.], [1., 1.]], [[-1., -1.], [2., 2.]]])
    # Audit the actual selection, not what the selector would have picked.
    report = band_window_audit(energies, [0, 3], 0., 2, 5.)
    assert report["raw_band_count"] == 4
    assert report["selected_indices"] == [0, 3]
    assert report["selected_spin_band_indices"] == [[0, 0], [1, 1]]
    assert report["window_dropped_indices"] == [1, 2]
    assert report["omitted_frontier_sides"] == ["occupied", "empty"]
    assert report["straddling_intervals_dropped"] == 0


def test_window_audit_accounts_for_empty_window_nearest_fallback():
    from src.data.multiband import band_window_audit
    report = band_window_audit([[9., 9.], [10., 10.], [11., 11.]], [0], 0., 1, 5.)
    assert report["selection_branch"] == "nearest_fallback"
    assert report["selection_candidates"] == 2
    assert report["selection_dropped_indices"] == [1]
    assert report["window_candidates"] == report["window_dropped"] == 0
    assert report["capacity_insufficient"] and not report["window_capacity_insufficient"]


@pytest.mark.parametrize("updates", [
    {"energies": [[np.nan, 1.]]}, {"energies": [1., 2.]},
    {"selected_indices": [0, 0]}, {"selected_indices": [-1]},
    {"selected_indices": [3]}, {"selected_indices": [0.5]},
    {"max_bands": 0}, {"max_bands": 1.5},
    {"efermi": np.inf}, {"delta_e": -1.}, {"delta_e": np.nan},
    {"selected_indices": [0, 1], "max_bands": 1},
])
def test_window_audit_rejects_invalid_ledger_inputs(updates):
    from src.data.multiband import band_window_audit
    inputs = dict(energies=[[-1., 1.], [2., 2.]], selected_indices=[0],
                  efermi=0., max_bands=16, delta_e=5.)
    inputs.update(updates)
    with pytest.raises(ValueError):
        band_window_audit(**inputs)


def test_window_audit_returns_json_native_counts_for_numpy_capacity():
    import json
    from src.data.multiband import band_window_audit
    result = band_window_audit([[-1., -1.], [1., 1.]], np.array([0, 1]),
                               np.float32(0.), np.int64(16), np.float32(5.))
    assert json.loads(json.dumps(result, allow_nan=False))["side_quotas"] == {"occupied": 8, "empty": 8}
