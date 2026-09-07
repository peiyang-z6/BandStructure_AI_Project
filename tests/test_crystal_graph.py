"""Regression tests for crystal graph construction (P2)."""
import numpy as np
import pytest

from src.data.crystal_graph import (
    ELEMENTS,
    ELEMENT_INDEX,
    build_crystal_graph,
    build_crystal_graph_batch,
    element_onehot,
    rbf_expand,
)


def test_element_onehot_shape_and_encoding():
    oh = element_onehot(["Co", "Sc", "Sc", "Zn"])
    assert oh.shape == (4, len(ELEMENTS) + 1)
    assert oh[0, ELEMENT_INDEX["Co"]] == 1.0
    assert oh[1, ELEMENT_INDEX["Sc"]] == 1.0
    assert oh[2, ELEMENT_INDEX["Sc"]] == 1.0
    assert oh[3, ELEMENT_INDEX["Zn"]] == 1.0
    # each row is one-hot
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_element_onehot_unknown_symbol_pads_zero_row():
    oh = element_onehot(["Xx"])
    assert oh.shape == (1, len(ELEMENTS) + 1)
    assert np.all(oh == 0.0)  # unknown -> all-zero row (no crash)


def test_rbf_expand_peak_at_distance():
    rbf = rbf_expand(np.asarray([4.0], dtype=np.float32))
    assert rbf.shape == (1, 40)
    assert rbf.max() == pytest.approx(1.0, abs=1e-6)
    # peak should be near the center at distance 4.0
    assert np.argmax(rbf[0]) == 20  # 4.0 / (8.0/40) = 20


def test_build_crystal_graph_bcc_fe():
    lattice = [[2.8665, 0, 0], [0, 2.8665, 0], [0, 0, 2.8665]]
    species = ["Fe", "Fe"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    g = build_crystal_graph(lattice, species, coords)
    assert g["n_atoms"] == 2
    assert g["atom_features"].shape == (2, len(ELEMENTS) + 1)
    assert g["neighbor_list"].shape == (2, 12)
    # BCC Fe: each atom's nearest neighbour is the other atom (index 1),
    # ~2.48 A away (sqrt(3)/2 * a). The 2-atom cell yields the other atom
    # (and its periodic images) as the 8 nearest neighbours.
    assert g["neighbor_list"][0, 0] == 1  # atom 1 is the nearest neighbour of atom 0
    assert g["neighbor_dist"][0].sum() > 0  # raw distances present
    assert g["neighbor_dist"].shape == (2, 12)


def test_build_crystal_graph_batch_pads_to_max_atoms():
    small = {"lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]], "species": ["Si"], "fractional_coordinates": [[0, 0, 0]]}
    large = {"lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
             "species": ["Si", "Ge", "Ge", "Ge"],
             "fractional_coordinates": [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]}
    batch = build_crystal_graph_batch([small, large], max_atoms=8)
    assert batch["atom_features"].shape == (2, 8, len(ELEMENTS) + 1)
    assert batch["neighbor_list"].shape == (2, 8, 12)
    assert list(batch["n_atoms"]) == [1, 4]
    # padding atom rows are all-zero
    assert np.all(batch["atom_features"][0, 1:] == 0.0)
