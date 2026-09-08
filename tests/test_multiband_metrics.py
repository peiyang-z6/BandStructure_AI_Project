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


def test_derive_gap_insulator():
    bands, mask = _insulator()
    gap, is_ins = derive_gap(bands, mask)
    assert is_ins.all()
    assert gap[0] == pytest.approx(3.0, abs=1e-4)  # 2.5 - (-0.5)


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


def test_band_mae_masks_padding():
    pred = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    target = np.array([[[1.0, 1.0], [9.0, 9.0]]], dtype=np.float32)
    mask = np.array([[True, False]])
    assert band_mae(pred, target, mask) == pytest.approx(0.0, abs=1e-6)


def test_sorted_band_mae_invariant_to_band_order():
    from src.evaluation.multiband_metrics import sorted_band_mae
    pred = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)
    target = np.array([[[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    assert sorted_band_mae(pred, target, mask) == pytest.approx(0.0, abs=1e-5)
