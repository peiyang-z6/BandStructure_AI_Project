"""Regression tests for the CGCNN crystal graph encoder (P2)."""
import numpy as np
import tensorflow as tf

from src.models.crystal_graph_encoder import CGCNNEncoder


def _dummy_graph(batch=2, n_atoms=4, n_elements=109, max_neighbors=12):
    rng = np.random.default_rng(0)
    atom_features = np.zeros((batch, n_atoms, n_elements + 1), dtype=np.float32)
    for b in range(batch):
        for a in range(n_atoms):
            atom_features[b, a, rng.integers(1, n_elements + 1)] = 1.0
    neighbor_list = np.full((batch, n_atoms, max_neighbors), -1, dtype=np.int32)
    neighbor_dist = np.zeros((batch, n_atoms, max_neighbors), dtype=np.float32)
    for b in range(batch):
        for a in range(n_atoms):
            for k in range(min(3, max_neighbors)):
                nb = (a + k + 1) % n_atoms
                neighbor_list[b, a, k] = nb
                neighbor_dist[b, a, k] = 1.0 + k  # raw distance in Angstrom
    return {"atom_features": atom_features, "neighbor_list": neighbor_list, "neighbor_dist": neighbor_dist}


def test_encoder_output_shape():
    enc = CGCNNEncoder(num_elements=109, conv_layers=3, hidden_dim=128, embedding_dim=128)
    g = _dummy_graph()
    out = enc(g, training=False)
    assert out.shape == (2, 128)


def test_encoder_deterministic():
    tf.keras.utils.set_random_seed(42)
    enc = CGCNNEncoder(num_elements=109, embedding_dim=64)
    g = _dummy_graph(batch=1)
    o1 = enc(g, training=False).numpy()
    o2 = enc(g, training=False).numpy()
    assert np.allclose(o1, o2)


def test_encoder_padding_atoms_contribute_zero():
    # A graph with 2 real atoms + 2 padding atoms must equal a graph with
    # only the 2 real atoms (mean pooling ignores padding).
    enc = CGCNNEncoder(num_elements=109, conv_layers=2, hidden_dim=64, embedding_dim=32)
    g2 = _dummy_graph(batch=1, n_atoms=2)
    g4 = _dummy_graph(batch=1, n_atoms=4)
    # zero out atoms 2,3 in g4 to simulate padding
    g4["atom_features"][0, 2:] = 0.0
    g4["neighbor_list"][0, 2:] = -1
    g4["neighbor_dist"][0, 2:] = 0.0
    # also remove references to atoms 2,3 from atom 0,1's neighbour lists
    g4["neighbor_list"][0, :2] = g2["neighbor_list"][0]
    g4["neighbor_dist"][0, :2] = g2["neighbor_dist"][0]
    o2 = enc(g2, training=False).numpy()
    o4 = enc(g4, training=False).numpy()
    assert np.allclose(o2, o4, atol=1e-5)


def test_encoder_serialization_roundtrip():
    enc = CGCNNEncoder(num_elements=109, embedding_dim=64)
    g = _dummy_graph(batch=1)
    enc(g, training=False)
    cfg = enc.get_config()
    enc2 = CGCNNEncoder.from_config(cfg)
    enc2(g, training=False)
    assert enc2.embedding_dim == 64
