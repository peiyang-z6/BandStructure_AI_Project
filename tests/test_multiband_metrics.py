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
