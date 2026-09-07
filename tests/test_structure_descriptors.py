"""Regression tests for enhanced structure descriptors (P2 v2)."""
import numpy as np
import pytest

from src.data.structure_descriptors import (
    composition_stats,
    global_descriptors,
    lattice_params,
)


def test_lattice_params_cubic():
    lp = lattice_params([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    assert lp.shape == (7,)
    assert np.allclose(lp[0:3], [4, 4, 4])
    assert np.allclose(lp[3:6], [90, 90, 90])
    assert np.isclose(lp[6], 64.0)  # volume 4^3


def test_lattice_params_fcc_primitive_rhombohedral():
    # FCC primitive cell: a=b=c, alpha=beta=gamma=60 deg
    a = 2.8665
    lp = lattice_params([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    assert np.isclose(lp[3], 60.0, atol=1e-3)  # alpha
    assert np.isclose(lp[5], 60.0, atol=1e-3)  # gamma
    assert lp[6] > 0  # non-zero volume


def test_composition_stats_shape_and_values():
    cs = composition_stats(["Na", "Cl"])
    assert cs.shape == (7,)
    assert cs[4] == 2  # two unique elements
    # Na (0.93) and Cl (3.16) -> mean = 2.045
    assert np.isclose(cs[0], (0.93 + 3.16) / 2, atol=1e-3)
    assert cs[2] == pytest.approx(0.93, abs=1e-3)  # min EN


def test_composition_stats_empty():
    cs = composition_stats([])
    assert cs.shape == (7,)
    assert np.all(cs == 0)


def test_global_descriptors_length():
    gd = global_descriptors([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], 227)
    assert gd.shape == (7 + 7 + 8,)
    # Si diamond is spacegroup 227 -> bucket log2(227)=7
    assert gd[-8:].argmax() == 7


def test_global_descriptors_spacegroup_bucketing():
    gd1 = global_descriptors([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], 1)
    gd2 = global_descriptors([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], 2)
    gd3 = global_descriptors([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Si"], 230)
    # sg 1 -> bucket 0, sg 2 -> bucket 1, sg 230 -> bucket 7
    assert gd1[-8:].argmax() == 0
    assert gd2[-8:].argmax() == 1
    assert gd3[-8:].argmax() == 7
