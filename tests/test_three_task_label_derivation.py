"""Regression tests for three-task label derivation script (P0A)."""
import numpy as np

from scripts.derive_three_task_labels import derive_labels


def _toy_X(n=8, nk=32):
    rng = np.random.RandomState(0)
    return rng.uniform(-2, 2, (n, 2, nk, 3)).astype(np.float32)


def test_derive_labels_shapes_and_coverage():
    n = 8
    X = _toy_X(n)
    ids = np.asarray([f"aflow-t-{i}" for i in range(n)], dtype="U32")
    manifest = {
        "samples": [
            {
                "material_id": f"aflow-t-{i}",
                "metal_feature_inferred": bool(i % 4 == 0),
                "gap_type": 0 if i % 4 == 0 else (1 if i % 2 else 2),
                "gap_type_source": "provider_is_direct" if i % 4 else "provider_is_metal",
            }
            for i in range(n)
        ]
    }
    metadata = {
        f"aflow-t-{i}": {"is_metal": bool(i % 4 == 0), "is_direct": bool(i % 2 == 0)}
        for i in range(n)
    }
    labels = derive_labels(X, ids, manifest, metadata)
    for key in ("line_mode_topology", "provider_global_electronic_type", "line_global_disagreement"):
        assert labels[key].shape == (n,)
        assert np.all(labels[key] >= -1)
    assert np.all(labels["provider_global_electronic_type"] >= 0)


def test_derive_labels_uses_provider_metadata_when_available():
    n = 4
    X = _toy_X(n)
    ids = np.asarray([f"aflow-t-{i}" for i in range(n)], dtype="U32")
    manifest = {"samples": [{"material_id": f"aflow-t-{i}"} for i in range(n)]}
    metadata = {
        "aflow-t-0": {"is_metal": True, "is_direct": None},
        "aflow-t-1": {"is_metal": False, "is_direct": True},
        "aflow-t-2": {"is_metal": False, "is_direct": False},
    }
    labels = derive_labels(X, ids, manifest, metadata)
    assert labels["provider_global_electronic_type"][0] == 0
    assert labels["provider_global_electronic_type"][1] == 1
    assert labels["provider_global_electronic_type"][2] == 2
    # disagreement defined only where provider type known
    assert labels["line_global_disagreement"][3] == -1


def test_derive_labels_disagreement_is_consistent():
    n = 4
    X = _toy_X(n)
    ids = np.asarray([f"aflow-t-{i}" for i in range(n)], dtype="U32")
    manifest = {"samples": [{"material_id": f"aflow-t-{i}"} for i in range(n)]}
    metadata = {
        f"aflow-t-{i}": {"is_metal": False, "is_direct": True} for i in range(n)
    }
    labels = derive_labels(X, ids, manifest, metadata)
    topo = labels["line_mode_topology"]
    prov = labels["provider_global_electronic_type"]
    dis = labels["line_global_disagreement"]
    for i in range(n):
        assert dis[i] == int(topo[i] != prov[i])
