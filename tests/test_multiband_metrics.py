"""Regression tests for P3 multi-band evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.multiband_metrics import band_mae, derive_gap, gap_mae


def _insulator(batch=1):
    # VBM ~ -0.5, CBM ~ 2.5 -> gap 3.0
    bands = np.zeros((batch, 4, 8), dtype=np.float32)
    bands[:, 0] = -0.5  # valence band (band_max <= 0)
    bands[:, 1] = -1.0
    bands[:, 2] = 2.5   # conduction band (band_min >= 0)
    bands[:, 3] = 4.0
    mask = np.ones((batch, 4), dtype=bool)
    return bands, mask


def test_derive_gap_padding_cannot_raise_vbm_to_zero():
    bands = np.array([[[-0.5, -0.5], [2.5, 2.5], [0.0, 0.0]]])
    gap, is_ins = derive_gap(bands, np.array([[True, True, False]]))
    assert is_ins[0]
    assert gap[0] == pytest.approx(3.0)


def test_missing_edge_is_unknown_not_proof_of_metal():
    bands = np.array([[[-2.0, -1.0], [-3.0, -2.0]]])
    gap, is_ins = derive_gap(bands, np.ones((1, 2), dtype=bool))
    assert not is_ins[0]
    assert np.isnan(gap[0])


@pytest.mark.parametrize("efermi", [0.0, 5.0])
def test_upward_fermi_touching_conduction_sets_gap(efermi):
    bands = np.array([[[-1.0, -1.0], [0.0, 1.0], [2.0, 2.0]]]) + efermi
    gap, is_ins = derive_gap(bands, np.ones((1, 3), dtype=bool), efermi=efermi)
    assert gap[0] == pytest.approx(1.0)
    assert is_ins[0]


def test_fermi_equal_band_is_not_both_valence_and_conduction():
    bands = np.array([[[0.0, 0.0], [2.0, 2.0]]])
    gap, is_ins = derive_gap(bands, np.ones((1, 2), dtype=bool))
    assert is_ins[0]
    assert gap[0] == pytest.approx(2.0)


@pytest.mark.parametrize("efermi", [0.0, 5.0])
@pytest.mark.parametrize("valence", [[-1.0, 0.0], [0.0, 0.0]])
def test_frontiers_touching_fermi_are_resolved_zero_gap(efermi, valence):
    bands = np.array([[valence, [0.0, 1.0], [2.0, 2.0]]]) + efermi
    gap, is_ins = derive_gap(bands, np.ones((1, 3), dtype=bool), efermi=efermi)
    assert gap[0] == 0.0
    assert not is_ins[0]


@pytest.mark.parametrize("efermi", [0.0, 5.0])
def test_lone_flat_fermi_band_is_unknown(efermi):
    bands = np.array([[[0.0, 0.0]]]) + efermi
    gap, is_ins = derive_gap(bands, np.ones((1, 1), dtype=bool), efermi=efermi)
    assert np.isnan(gap[0])
    assert not is_ins[0]


def test_gap_does_not_invent_crossing_across_disconnected_segments():
    import inspect
    assert "segment_ids" in inspect.signature(derive_gap).parameters
    bands = np.array([[[-1., -1., 1., 1.], [-3., -3., -3., -3.], [3., 3., 3., 3.]]])
    gap, ins = derive_gap(bands, np.ones((1, 3), bool),
                          segment_ids=np.array([[0, 0, 1, 1]]))
    assert np.isnan(gap[0])
    assert not ins[0]


def test_derive_gap_insulator():
    bands, mask = _insulator()
    gap, is_ins = derive_gap(bands, mask)
    assert is_ins.all()
    assert gap[0] == pytest.approx(3.0, abs=1e-4)  # 2.5 - (-0.5)


def test_crossing_band_overrides_deep_valence_and_high_conduction():
    bands = np.array([[[-2.0, -2.0], [-1.0, 1.0], [3.0, 3.0]]])
    gap, is_ins = derive_gap(bands, np.ones((1, 3), dtype=bool))
    assert not is_ins[0]
    assert gap[0] == 0.0


def test_derive_gap_metal_is_zero():
    # band crossing E_F -> no clean valence/conduction split -> metal
    bands = np.zeros((1, 2, 8), dtype=np.float32)
    bands[0, 0] = -1.0  # crosses? make band0 span -1..+1
    bands[0, 0, 4:] = 1.0
    bands[0, 1] = 2.0
    mask = np.ones((1, 2), dtype=bool)
    gap, is_ins = derive_gap(bands, mask)
    assert not is_ins[0]
    assert gap[0] == 0.0


def test_gap_mae_uses_target_for_insulator_split():
    # target is metal (no conduction band), prediction spuriously has one
    target = np.zeros((1, 2, 8), dtype=np.float32)
    target[0, 0] = -1.0
    target[0, 0, 4:] = 1.0  # crossing band -> metal
    target[0, 1] = 3.0       # this IS a conduction band (band_min 3>=0), so target is insulator?
    # make target clearly metal: both bands cross
    target[0, 1] = -1.0
    target[0, 1, 4:] = 1.0
    pred = np.zeros_like(target)
    pred[0, 0] = -1.0
    pred[0, 1] = 5.0  # prediction has a conduction band -> fake gap
    mask = np.ones((1, 2), dtype=bool)
    # target has no valence-only AND conduction-only pair -> metal -> gap excluded
    assert np.isnan(gap_mae(pred, target, mask))


def test_evaluation_counts_false_metal_gaps_and_exposes_unknown_coverage():
    import json
    from src.evaluation import multiband_metrics as module
    assert callable(getattr(module, "evaluate_multiband", None))
    target = np.array([[[-1., 1.], [-2., 2.]], [[-2., -2.], [-1., -1.]]])
    pred = np.array([[[-1., -1.], [2., 2.]], [[-2., -2.], [-1., -1.]]])
    result = module.evaluate_multiband(pred, target, np.ones((2, 2), bool))
    assert result["n_target_metal"] == 1
    assert result["n_target_unknown"] == 1
    assert result["n_gap_pairs_resolved"] == 1
    assert result["gap_mae_all_resolved"] == pytest.approx(3.0)
    assert result["gap_mae_insulator_resolved"] is None
    assert result["false_gap_rate_on_resolved_metals"] == 1.0
    json.dumps(result, allow_nan=False)


def test_band_mae_ignores_nan_padding_but_not_valid_errors():
    pred = np.array([[[1., 1.], [np.nan, np.nan]]])
    target = np.array([[[2., 2.], [np.nan, np.nan]]])
    assert band_mae(pred, target, np.array([[True, False]])) == pytest.approx(1.)


def test_band_mae_masks_padding():
    pred = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    target = np.array([[[1.0, 1.0], [9.0, 9.0]]], dtype=np.float32)
    mask = np.array([[True, False]])
    assert band_mae(pred, target, mask) == pytest.approx(0.0, abs=1e-6)


def test_sorted_metric_rejects_nonfinite_valid_prediction():
    from src.evaluation.multiband_metrics import sorted_band_mae
    with pytest.raises(ValueError, match="finite"):
        sorted_band_mae(np.array([[[np.nan]]]), np.zeros((1, 1, 1)), np.ones((1, 1), bool))


def test_evaluation_empty_mask_uses_null_metrics_not_nan_json():
    import json
    from src.evaluation.multiband_metrics import evaluate_multiband
    result = evaluate_multiband(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.zeros((1, 2), bool))
    assert result["band_mae_per_k_ot"] is None
    assert result["band_mae_slot"] is None
    assert result["n_target_unknown"] == 1
    json.dumps(result, allow_nan=False)


def test_padding_sentinel_cannot_rank_before_finite_real_energies():
    from src.evaluation.multiband_metrics import sorted_band_mae
    pred = np.array([[[1e10], [0.]]], dtype=np.float32)
    target = np.array([[[1e10 + 1e4], [0.]]], dtype=np.float32)
    mask = np.array([[True, False]])
    assert sorted_band_mae(pred, target, mask) == pytest.approx(band_mae(pred, target, mask))


def test_sorted_band_mae_invariant_to_band_order():
    from src.evaluation.multiband_metrics import sorted_band_mae
    pred = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)
    target = np.array([[[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    assert sorted_band_mae(pred, target, mask) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("row, guaranteed", [([-1., 1.], True), ([-1., 0.], False)])
@pytest.mark.parametrize("ef", [0., 5.])
def test_path_diagnostics_use_both_occupation_counts_at_fermi_touch(row, guaranteed, ef):
    from src.evaluation import multiband_metrics as module
    assert callable(getattr(module, "band_path_diagnostics", None))
    result = module.band_path_diagnostics(np.array([row]) + ef, [True], [0, 0], efermi=ef)
    segment = result["segments"][0]
    assert segment["n_strictly_below_ef"] == [1, 0]
    assert segment["n_at_or_below_ef"] == ([1, 0] if guaranteed else [1, 1])
    assert segment["relabeling_invariant_strict_crossing"] is guaranteed
    assert result["scope"] == "selected_slot_and_energy_rank_line_diagnostic"
    assert result["causal_attribution"] == "not_established"


def test_path_diagnostics_never_join_disconnected_runs_with_reused_labels():
    from src.evaluation.multiband_metrics import band_path_diagnostics
    bands = [[-1., -1., 2., 2., 1., 1.]]
    result = band_path_diagnostics(bands, [True], [0, 0, 1, 1, 0, 0])
    assert len(result["segments"]) == 3
    assert not any(s["relabeling_invariant_strict_crossing"] for s in result["segments"])
    assert result["interval_straddling_bands"] == 1
    assert result["within_segment_crossing_bands"] == 0
    assert all(s["slot_crossing_bands"] == 0 for s in result["segments"])


@pytest.mark.parametrize("bands, segments, expected", [
    ([[-1., -1.]], [0, 0], "unresolved_missing_empty_frontier"),
    ([[1., 1.]], [0, 0], "unresolved_missing_occupied_frontier"),
    ([[0., 0.]], [0, 0], "unresolved_ef_flat_only"),
    ([[-1., 1.]], [0, 1], "unresolved_between_segments"),
    ([[-1., 1.]], [0, 0], "witnessed_within_segment_crossing"),
    ([[-1., 0.], [0., 1.]], [0, 0], "resolved_touching_frontiers"),
    ([[-1., -1.], [0., 1.]], [0, 0], "resolved_positive_gap"),
])
def test_path_diagnostics_classify_unresolved_frontiers_without_changing_gap(bands, segments, expected):
    from src.evaluation.multiband_metrics import band_path_diagnostics
    mask = np.ones(len(bands), bool)
    result = band_path_diagnostics(bands, mask, segments)
    assert result["frontier_state"] == expected
    gap, _ = derive_gap(np.array([bands]), mask[None], segment_ids=segments)
    assert result["gap_eV"] == (float(gap[0]) if np.isfinite(gap[0]) else None)


def test_path_diagnostics_separate_slot_motion_from_rank_motion_with_spin_groups():
    from src.evaluation.multiband_metrics import band_path_diagnostics
    import inspect
    assert "band_groups" in inspect.signature(band_path_diagnostics).parameters
    result = band_path_diagnostics([[-1., 1.], [1., -1.]], [True, True], [0, 0],
                                   band_groups=[0, 0])
    segment = result["segments"][0]
    assert segment["source_order_inversions"] == 1
    assert segment["minimum_adjacent_separation_eV"] == -2.
    assert segment["slot_dispersion_rms_eV"] == 1.
    assert segment["rank_dispersion_rms_eV"] == 0.
    # Opposite spin blocks are not presumed globally energy ordered.
    separate = band_path_diagnostics([[1., 1.], [-1., -1.]], [True, True], [0, 0],
                                     band_groups=[0, 1])
    assert separate["segments"][0]["source_order_inversions"] == 0
    assert separate["segments"][0]["minimum_adjacent_separation_eV"] is None


def test_path_diagnostics_exclude_band_and_k_padding_before_arithmetic():
    import json
    from src.evaluation.multiband_metrics import band_path_diagnostics
    result = band_path_diagnostics([[-1., -1., np.nan, 1., 1.], [np.nan] * 5],
                                   [True, False], [0, 0, -1, 0, 0])
    assert len(result["segments"]) == 2
    assert result["frontier_state"] == "unresolved_between_segments"
    assert result["n_valid_bands"] == 1 and result["n_valid_k"] == 4
    assert all(s["slot_dispersion_rms_eV"] == 0. for s in result["segments"])
    empty = band_path_diagnostics([[np.nan, np.nan]], [False], [0, 0])
    assert empty["frontier_state"] == "unresolved_no_valid_bands"
    assert empty["gap_eV"] is None
    no_k = band_path_diagnostics([[np.nan]], [True], [-1])
    assert no_k["frontier_state"] == "unresolved_no_valid_k"
    json.dumps([result, empty, no_k], allow_nan=False)


@pytest.mark.parametrize("updates", [
    {"bands": [[np.nan, 1.]]}, {"efermi": np.inf},
    {"segment_ids": [-2, -2]}, {"segment_ids": [0.5, 0.5]},
    {"mask": [1]}, {"band_groups": [0.5]},
])
def test_path_diagnostics_reject_nonphysical_valid_inputs(updates):
    from src.evaluation.multiband_metrics import band_path_diagnostics
    inputs = dict(bands=[[-1., 1.]], mask=[True], segment_ids=[0, 0])
    inputs.update(updates)
    with pytest.raises(ValueError):
        band_path_diagnostics(**inputs)


def test_resampling_audit_reports_new_rank_inversions_without_sorting_targets():
    from src.evaluation import multiband_metrics as module
    from src.data.multiband import extract_fermi_bands, path_segments, select_fermi_bands
    assert callable(getattr(module, "resampling_audit", None))
    # Synthetic source-order counterexample, not evidence of a real-model cause.
    raw = np.array([[-.588051, -2.444938, -.770119, -.742856],
                    [.860268, -1.908510, -.583672, 1.590730]], np.float32)
    k = np.array([0., .1, .8, 1.])
    selected = select_fermi_bands(raw, 0., 16, 5.)
    bands, mask, _ = extract_fermi_bands(raw, k, 0., 16, 5., 256)
    source_segments, target_segments = path_segments(k, 256)
    saved = bands.copy()
    result = module.resampling_audit(raw, selected, bands, mask, source_segments, target_segments)
    assert result["raw_full"]["segments"][0]["source_order_inversions"] == 0
    assert result["raw_selected"]["segments"][0]["source_order_inversions"] == 0
    assert result["resampled_selected"]["segments"][0]["source_order_inversions"] > 0
    assert result["causal_attribution"] == "not_established"
    np.testing.assert_array_equal(bands, saved)


def test_resampling_audit_exposes_lost_segments_with_spin_and_energy_reference():
    from src.evaluation.multiband_metrics import resampling_audit
    raw = np.array([[[4., 4., 6., 6.]], [[2., 2., 8., 8.]]])
    result = resampling_audit(raw, [0, 1], [[1., 1.], [3., 3.]], [True, True],
                              [0, 0, 1, 1], [1, 1], efermi=5.)
    assert result["unrepresented_source_segment_ids"] == [0]
    assert result["source_segment_count"] == 2 and result["target_segment_count"] == 1
    assert result["raw_selected"]["frontier_state"] == "unresolved_between_segments"
    assert result["resampled_selected"]["frontier_state"] == "unresolved_missing_occupied_frontier"
    assert all(s["source_order_inversions"] == 0 for s in result["raw_full"]["segments"])


@pytest.mark.parametrize("indices, target_segments", [([-1, 1], [0, 0]),
    ([0, 0], [0, 0]), ([0], [0, 0]), ([0, 1], [1, 1])])
def test_resampling_audit_rejects_misaligned_selection_ledger(indices, target_segments):
    from src.evaluation.multiband_metrics import resampling_audit
    with pytest.raises(ValueError, match="selection|segment"):
        resampling_audit([[-1., -1.], [1., 1.]], indices, [[-1., -1.], [1., 1.]],
                         [True, True], [0, 0], target_segments)


def test_matching_diagnostic_distinguishes_segment_and_global_assignment():
    from src.evaluation import multiband_metrics as module
    assert callable(getattr(module, "trajectory_matching_diagnostics", None))
    target = np.array([[-1., -1.], [1., 1.]])
    pred = np.array([[-1., 1.], [1., -1.]])
    result = module.trajectory_matching_diagnostics(pred, target, [True, True], [0, 1])
    assert result["per_k_ot_eV"] == result["segment_hungarian_eV"] == 0.
    assert result["global_hungarian_eV"] == result["slot_mae_eV"] == 1.
    connected = module.trajectory_matching_diagnostics(pred, target, [True, True], [0, 0])
    assert connected["segment_hungarian_eV"] == 1.
    assert result["causal_attribution"] == "not_established"
    assert result["scope"] == "selected_slot_assignment_not_wavefunction_identity"


def test_matching_diagnostic_returns_null_for_empty_valid_population():
    import json
    from src.evaluation.multiband_metrics import trajectory_matching_diagnostics
    for mask, segments in [([False], [0, 0]), ([True], [-1, -1])]:
        result = trajectory_matching_diagnostics([[np.nan, np.nan]], [[np.nan, np.nan]], mask, segments)
        for name in ["per_k_ot_eV", "segment_hungarian_eV", "global_hungarian_eV", "slot_mae_eV"]:
            assert result[name] is None
        json.dumps(result, allow_nan=False)


def test_matching_diagnostic_bounds_pairwise_cost_allocation():
    from src.evaluation.multiband_metrics import trajectory_matching_diagnostics
    import inspect
    assert "max_cost_elements" in inspect.signature(trajectory_matching_diagnostics).parameters
    with pytest.raises(ValueError, match="bounded"):
        trajectory_matching_diagnostics(np.zeros((2, 2)), np.zeros((2, 2)),
                                        [True, True], [0, 0], max_cost_elements=7)


def test_matching_feasible_set_order_and_padding_are_descriptive_properties():
    from src.evaluation.multiband_metrics import trajectory_matching_diagnostics
    rng = np.random.default_rng(42)
    for _ in range(8):
        pred, target = rng.normal(size=(2, 3, 7)).astype(np.float32)
        pred[2] = np.nan
        target[2] = np.inf
        pred[:, 3] = target[:, 3] = np.nan
        result = trajectory_matching_diagnostics(pred, target, [True, True, False], [0, 0, 1, -1, 0, 0, 0])
        assert result["per_k_ot_eV"] <= result["segment_hungarian_eV"] + 1e-7
        assert result["segment_hungarian_eV"] <= result["global_hungarian_eV"] + 1e-7
        assert result["global_hungarian_eV"] <= result["slot_mae_eV"] + 1e-7
    target = np.array([[-2., -2.], [-1., 1.], [3., 3.]], np.float32)
    result = trajectory_matching_diagnostics(target + .125, target, [True] * 3, [0, 0])
    assert result["global_hungarian_eV"] == result["per_k_ot_eV"] == .125
    with pytest.raises(ValueError, match="finite"):
        trajectory_matching_diagnostics([[np.nan, 1.]], [[0., 1.]], [True], [0, 0])


@pytest.mark.parametrize("budget", [np.nan, np.inf, 8.5])
def test_matching_budget_cannot_disable_bound_with_noninteger_values(budget):
    from src.evaluation.multiband_metrics import trajectory_matching_diagnostics
    with pytest.raises(ValueError, match="positive integer"):
        trajectory_matching_diagnostics(np.zeros((2, 2)), np.zeros((2, 2)),
                                        [True, True], [0, 0], max_cost_elements=budget)
