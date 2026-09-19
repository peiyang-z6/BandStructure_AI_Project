"""Small CPU contracts; synthetic structures are not scientific training evidence."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eqg = _load("scripts/prepare_p2_equivariant_graphs.py", "p2_graph_contracts")


def _cubic():
    return {"lattice": np.eye(3) * 3.0, "species": ["Si"],
            "fractional_coordinates": [[0.0, 0.0, 0.0]]}


def test_single_atom_periodic_neighbors_keep_nonzero_image_lengths():
    rec = _cubic()
    graph = eqg.build_one_graph(rec, 50)
    struct = Structure(Lattice(rec["lattice"]), rec["species"], rec["fractional_coordinates"])
    expected = sorted(n.nn_distance for n in struct.get_all_neighbors(
        8.0, include_index=True, numerical_tol=0.01)[0])[:12]
    assert graph["edge_len"].shape == (12,)
    np.testing.assert_allclose(graph["edge_len"], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(graph["edge_vec"], axis=1), expected, rtol=1e-6)


def test_periodic_image_is_stored_and_reconstructs_each_selected_edge():
    rec = _cubic()
    graph = eqg.build_one_graph(rec, 50)
    assert "edge_image" in graph, "periodic images must remain auditable, not just distances"
    images = graph["edge_image"]
    assert images.shape == (12, 3) and np.issubdtype(images.dtype, np.integer)
    assert len(np.unique(images, axis=0)) == 12
    frac = np.asarray(rec["fractional_coordinates"])
    expected = (frac[graph["edge_dst"]] + images - frac[graph["edge_src"]]) @ rec["lattice"]
    np.testing.assert_allclose(graph["edge_vec"], expected, atol=1e-6)


@pytest.mark.parametrize("capacity", [0, -1, 51, 1.5, True])
def test_graph_capacity_configuration_is_bounded_integer(capacity):
    rec = _cubic()
    for builder in (
        lambda: eqg.build_one_graph(rec, capacity),
        lambda: eqg.cg.build_crystal_graph(rec["lattice"], rec["species"],
                                          rec["fractional_coordinates"], max_atoms=capacity),
    ):
        with pytest.raises(ValueError, match="max_atoms"):
            builder()


@pytest.mark.parametrize("field", ["lattice", "fractional_coordinates"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_geometry_is_rejected_before_native_neighbor_search(field, bad, monkeypatch):
    # pymatgen's native neighbour routine is unsafe on non-finite input.
    # This barrier must never be reached; valid graph tests use the real routine.
    def unsafe(*args, **kwargs):
        raise AssertionError("nonfinite geometry reached native neighbor search")
    monkeypatch.setattr(Structure, "get_all_neighbors", unsafe)
    rec = _cubic()
    rec[field] = np.asarray(rec[field]).copy()
    rec[field][0, 0] = bad
    assert eqg.build_one_graph(rec, 50) is None
    with pytest.raises(ValueError, match="finite"):
        eqg.cg.build_crystal_graph(rec["lattice"], rec["species"], rec["fractional_coordinates"])


@pytest.mark.parametrize("change", [
    {"species": [], "fractional_coordinates": []},
    {"species": ["Og"]},
    {"species": ["Si", "Ge"]},
    {"fractional_coordinates": [[0, 0]]},
    {"lattice": [[3, 0, 0], [0, 3, 0]]},
    {"lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 0]]},
])
def test_malformed_geometry_or_species_cannot_be_valid_graph(change, monkeypatch):
    def unsafe(*args, **kwargs):
        raise AssertionError("invalid structure reached native neighbor search")
    monkeypatch.setattr(Structure, "get_all_neighbors", unsafe)
    rec = dict(_cubic(), **change)
    assert eqg.build_one_graph(rec, 50) is None
    with pytest.raises(ValueError):
        eqg.cg.build_crystal_graph(rec["lattice"], rec["species"], rec["fractional_coordinates"])


def _cache_inputs(tmp_path, monkeypatch, valid=(True, False, True)):
    import json
    import sys
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    records = []
    for split in ("train", "test"):
        ids = np.array([f"toy-{split}-{i}" for i in range(len(valid))])
        np.savez(pairs / f"p2_{split}.npz", material_ids=ids, valid=np.array(valid, dtype=bool))
        for mid in ids[:2]:
            rec = _cubic()
            rec["lattice"] = rec["lattice"].tolist()
            records.append(dict(rec, material_id=str(mid)))
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(records), encoding="utf-8")
    out = tmp_path / "new-graphs-v3"
    monkeypatch.setattr(sys, "argv", ["prepare_p2_equivariant_graphs.py", "--sidecar", str(sidecar),
                                    "--pairs", str(pairs), "--out", str(out)])
    return pairs, sidecar, out


def test_cache_validity_intersects_pairs_without_losing_id_positions(tmp_path, monkeypatch):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    with np.load(out / "graphs_train.npz", allow_pickle=False) as d:
        np.testing.assert_array_equal(d["valid"], [True, False, False])
        np.testing.assert_array_equal(d["material_ids"], ["toy-train-0", "toy-train-1", "toy-train-2"])
        np.testing.assert_array_equal(d["pair_valid"], [True, False, True])
        np.testing.assert_array_equal(d["valid_idx"], [0])
        np.testing.assert_array_equal(d["valid_material_ids"], ["toy-train-0"])
        assert np.all(d["atom_features"][1:] == 0)
        assert np.all(d["n_atoms"][1:] == 0)
        assert np.all(d["edge_offsets"][1:, 0] == d["edge_offsets"][1:, 1])


@pytest.mark.parametrize("ids,valid", [
    (["toy-train-0", "toy-train-0"], [True, True]),
    (["toy-train-0", ""], [True, True]),
    ([1, 2], [True, True]),
    (["toy-train-0"], [True, False]),
    (["toy-train-0"], [1]),
    ([["toy-train-0"]], [True]),
])
def test_cache_rejects_malformed_pair_identity_contract(tmp_path, monkeypatch, ids, valid):
    pairs, _, out = _cache_inputs(tmp_path, monkeypatch)
    np.savez(pairs / "p2_train.npz", material_ids=np.array(ids), valid=np.array(valid))
    with pytest.raises(ValueError, match="material_ids|valid"):
        eqg.main()
    assert not (out / "graphs_train.npz").exists()


def test_duplicate_sidecar_ids_fail_closed_instead_of_last_record_wins(tmp_path, monkeypatch):
    import json
    _, sidecar, out = _cache_inputs(tmp_path, monkeypatch)
    records = json.loads(sidecar.read_text())
    records.append(dict(records[0], species=["Ge"]))
    sidecar.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="duplicate.*material_id"):
        eqg.main()
    assert not (out / "graphs_train.npz").exists()


@pytest.mark.parametrize("kind", ["old-npz", "empty-directory", "pairs-directory"])
def test_cache_requires_new_output_directory_and_preserves_old_bytes(tmp_path, monkeypatch, kind):
    import hashlib
    import sys
    pairs, _, out = _cache_inputs(tmp_path, monkeypatch)
    if kind == "pairs-directory":
        sys.argv[-1] = str(pairs)
        out = pairs
    else:
        out.mkdir()
    if kind == "old-npz":
        np.savez(out / "graphs_train.npz", legacy=np.array([42]))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    with pytest.raises(FileExistsError, match="new|exist"):
        eqg.main()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    assert after == before


def test_new_graph_cache_records_schema_code_hashes_and_geometry_contract(tmp_path, monkeypatch):
    import hashlib
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    with np.load(out / "graphs_train.npz", allow_pickle=False) as d:
        assert "schema_version" in d, "unversioned geometry cannot replace legacy caches"
        assert d["schema_version"].item() == "p2-periodic-images-v3"
        assert d["cutoff_angstrom"].item() == 8.0
        assert d["max_neighbors"].item() == 12
        assert d["max_atoms"].item() == 50
        for key, path in (
            ("builder_sha256", "scripts/prepare_p2_equivariant_graphs.py"),
            ("crystal_graph_sha256", "src/data/crystal_graph.py"),
            ("descriptors_sha256", "src/data/structure_descriptors.py"),
        ):
            assert d[key].item() == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert d["edge_image"].shape == (12, 3)
        np.testing.assert_array_equal(d["edge_image"], eqg.build_one_graph(_cubic(), 50)["edge_image"])
        assert d["edge_offsets"].shape == (3, 2)


def _skew():
    return {"lattice": np.array([[4.7, .2, .4], [.6, 5.1, .3], [.2, .5, 5.9]]),
            "species": ["Si", "Ge", "O", "Fe", "C"],
            "fractional_coordinates": np.random.default_rng(4).random((5, 3))}


def _rotation():
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec([.31, -.71, .49]).as_matrix()


@pytest.mark.parametrize("change", ["rotation", "translation", "individual_images"])
def test_built_graph_geometry_is_rigid_motion_and_cell_image_consistent(change):
    rec = _skew()
    before = eqg.build_one_graph(rec, 50)
    # A strict 12-neighbour cap can split tied shells. This fixture deliberately
    # has a positive 12/13 gap; tied-shell behaviour is audited separately.
    struct = Structure(Lattice(rec["lattice"]), rec["species"], rec["fractional_coordinates"])
    for ns in struct.get_all_neighbors(8, include_index=True, numerical_tol=.01):
        distances = sorted(n.nn_distance for n in ns)
        assert distances[12] - distances[11] > .01
    new = dict(rec)
    expected_vec = before["edge_vec"]
    if change == "rotation":
        new["lattice"] = rec["lattice"] @ _rotation().T
        expected_vec = expected_vec @ _rotation().T
    elif change == "translation":
        new["fractional_coordinates"] = rec["fractional_coordinates"] + [.21, -.43, 1.27]
    else:
        new["fractional_coordinates"] = rec["fractional_coordinates"] + [[2, -1, 0], [0, 3, -2], [-1, 0, 4], [1, 1, 0], [0, 0, -2]]
    after = eqg.build_one_graph(new, 50)
    # Compare multisets: distance ties may legitimately reorder equal-distance edges.
    def rows(g, vec):
        values = np.column_stack((g["edge_src"], g["edge_dst"], vec))
        return np.array(sorted(map(tuple, np.round(values, 5))))
    np.testing.assert_allclose(rows(before, expected_vec), rows(after, after["edge_vec"]), atol=2e-5)
    np.testing.assert_allclose(np.sort(before["edge_len"]), np.sort(after["edge_len"]), atol=2e-6)
    np.testing.assert_allclose(before["desc"], after["desc"], atol=2e-5)
    frac = new["fractional_coordinates"]
    reconstructed = (frac[after["edge_dst"]] + after["edge_image"] - frac[after["edge_src"]]) @ new["lattice"]
    np.testing.assert_allclose(after["edge_vec"], reconstructed, atol=2e-6)


def test_truncated_shell_ledger_exposes_tie_without_changing_selection():
    graph = eqg.build_one_graph(_cubic(), 50)
    assert "neighbor_count" in graph, "fixed cap can split shells; record it, do not change it"
    struct = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
    ns = sorted(struct.get_all_neighbors(8, include_index=True, numerical_tol=.01)[0],
                key=lambda n: n.nn_distance)
    assert graph["neighbor_count"][0] == len(ns)
    assert graph["truncated_neighbors"][0] == len(ns) - 12
    assert graph["boundary_tied"][0]
    np.testing.assert_allclose(graph["distance_12_13"][0], [ns[11].nn_distance, ns[12].nn_distance])
    np.testing.assert_array_equal(graph["edge_image"], np.array([n.image for n in ns[:12]], dtype=int))
    assert graph["edge_len"].shape == (12,)


def test_cache_persists_shell_diagnostics_for_valid_and_invalid_rows(tmp_path, monkeypatch):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    graph = eqg.build_one_graph(_cubic(), 50)
    with np.load(out / "graphs_train.npz", allow_pickle=False) as d:
        for key in ("neighbor_count", "truncated_neighbors", "distance_12_13", "boundary_tied"):
            assert key in d, f"missing cached shell diagnostic: {key}"
            np.testing.assert_array_equal(d[key][0], graph[key])
            assert not d[key][1:].any()


@pytest.mark.parametrize("change", [
    {"spacegroup_number": np.nan}, {"spacegroup_number": np.inf},
    {"lattice": np.eye(3) * 1e14},
])
def test_nonfinite_or_unrepresentable_descriptors_invalidate_graph(change):
    with np.errstate(over="ignore", invalid="ignore"):
        assert eqg.build_one_graph(dict(_cubic(), **change), 50) is None


@pytest.mark.parametrize("n", [1, 49, 50, 51])
def test_fixed_atom_capacity_no_truncation_in_either_builder(n):
    rec = {"lattice": np.eye(3) * 50, "species": ["Si"] * n,
           "fractional_coordinates": np.random.default_rng(2).random((n, 3))}
    eq = eqg.build_one_graph(rec, 50)
    if n == 51:
        assert eq is None
        with pytest.raises(ValueError, match="capacity"):
            eqg.cg.build_crystal_graph_batch([rec], max_atoms=50)
    else:
        cg = eqg.cg.build_crystal_graph_batch([rec], max_atoms=50)
        assert eq["n_atoms"] == cg["n_atoms"][0] == n
        assert eq["atom_features"].shape == (50, 110)
        assert cg["neighbor_list"].shape == (1, 50, 12)
        assert np.count_nonzero(eq["atom_features"].sum(1)) == n
        assert np.all(eq["edge_len"] > 0)
        assert np.all(eq["edge_len"] <= 8 + 1e-5)
        assert np.all(eq["edge_src"] < n) and np.all(eq["edge_dst"] < n)


def test_neighbor_lengths_match_pymatgen_and_cgcnn_for_each_selected_neighbor():
    rec = _skew()
    eq = eqg.build_one_graph(rec, 50)
    cg = eqg.cg.build_crystal_graph(rec["lattice"], rec["species"], rec["fractional_coordinates"], max_atoms=50)
    for i in range(eq["n_atoms"]):
        mask = eq["edge_src"] == i
        np.testing.assert_allclose(eq["edge_len"][mask], cg["neighbor_dist"][i], atol=1e-6)
        np.testing.assert_array_equal(eq["edge_dst"][mask], cg["neighbor_list"][i])
        assert mask.sum() == 12
    np.testing.assert_allclose(np.linalg.norm(eq["edge_vec"], axis=1), eq["edge_len"], atol=1e-6)


def test_cache_loader_matches_full_raw_id_and_valid_order(tmp_path, monkeypatch):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    loader = getattr(eqg, "load_graph_cache", None)
    assert callable(loader), "cache consumers need an ID/valid-checked read-only loader"
    ids = np.array(["toy-train-0", "toy-train-1", "toy-train-2"])
    valid = np.array([True, False, False])
    path = out / "graphs_train.npz"
    loaded = loader(path, expected_material_ids=ids, expected_valid=valid)
    np.testing.assert_array_equal(loaded["valid_idx"], [0])
    with pytest.raises(ValueError, match="material_ids"):
        loader(path, expected_material_ids=ids[::-1], expected_valid=valid)
    # A band embedding prepared under the old pair-only validity mask must not
    # independently filter into a different order or a different row count.
    with pytest.raises(ValueError, match="valid"):
        loader(path, expected_material_ids=ids, expected_valid=np.array([True, False, True]))


@pytest.mark.parametrize("field,value", [("schema_version", "legacy"), ("builder_sha256", "missing")])
def test_cache_loader_rejects_unversioned_or_untraceable_old_npz_read_only(tmp_path, monkeypatch, field, value):
    import hashlib
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    with np.load(out / "graphs_train.npz") as d:
        old = {k: d[k] for k in d.files}
    old[field] = np.array(value)
    path = tmp_path / "old-cache.npz"
    np.savez(path, **old)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="schema|sha256"):
        eqg.load_graph_cache(path, expected_material_ids=old["material_ids"], expected_valid=old["valid"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.fixture
def torch_api():
    torch = pytest.importorskip("torch", reason="run model contracts in the isolated CPU torch/e3nn env")
    pytest.importorskip("e3nn")
    module = _load("src/models/equivariant_structure_encoder.py", "p2_actual_e3nn_encoder")
    return torch, module


def _torch_graph(torch, records=None):
    records = [_skew()] if records is None else records
    graphs = [eqg.build_one_graph(rec, 50) for rec in records]
    result = {
        "atom_features": torch.tensor(np.concatenate([g["atom_features"] for g in graphs]), dtype=torch.float64),
        "n_atoms": torch.tensor([g["n_atoms"] for g in graphs]),
        "graph_offsets": torch.arange(len(graphs), dtype=torch.long) * 50,
        "edge_src": torch.tensor(np.concatenate([g["edge_src"] + i * 50 for i, g in enumerate(graphs)])),
        "edge_dst": torch.tensor(np.concatenate([g["edge_dst"] + i * 50 for i, g in enumerate(graphs)])),
        "edge_vec": torch.tensor(np.concatenate([g["edge_vec"] for g in graphs]), dtype=torch.float64),
        "edge_len": torch.tensor(np.concatenate([g["edge_len"] for g in graphs]), dtype=torch.float64),
    }
    desc = torch.tensor(np.stack([g["desc"] for g in graphs]), dtype=torch.float64)
    return result, desc


def test_actual_encoder_eval_rotation_invariance(torch_api):
    torch, ese = torch_api
    torch.manual_seed(42)
    model = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=2, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    rotated = dict(graph, edge_vec=graph["edge_vec"] @ torch.tensor(_rotation(), dtype=torch.float64).T)
    with torch.no_grad():
        before, after = model(graph, desc), model(rotated, desc)
    assert before.shape == (1, 8) and torch.isfinite(before).all()
    print("actual_encoder_rotation_max_abs_error", (before - after).abs().max().item())
    torch.testing.assert_close(before, after, rtol=1e-8, atol=1e-8)


def test_unversioned_weights_cannot_masquerade_as_new_pooling(torch_api):
    torch, ese = torch_api
    torch.manual_seed(7)
    model = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=1, embedding_dim=8)
    state = model.state_dict()
    unversioned = {k: v for k, v in state.items() if k != "_extra_state"}
    with pytest.raises(RuntimeError, match="unversioned|legacy|pooling"):
        model.load_state_dict(unversioned, strict=False)
    assert state["_extra_state"]["pooling_version"] == "irreps-vector-norm-v3"


def test_legacy_state_layout_requires_explicit_legacy_pooling_mode(torch_api):
    torch, ese = torch_api
    torch.manual_seed(7)
    new = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=1, embedding_dim=8).double().eval()
    synthetic_old_layout = {k: v for k, v in new.state_dict().items() if k != "_extra_state"}
    legacy = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=1, embedding_dim=8,
                                             pooling_version="legacy-component-slice-v2").double().eval()
    result = legacy.load_state_dict(synthetic_old_layout, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert legacy.head[0].in_features == new.head[0].in_features == 27
    assert legacy.state_dict()["_extra_state"]["pooling_version"] == "legacy-component-slice-v2"
    graph, desc = _torch_graph(torch)
    with torch.no_grad():
        assert torch.isfinite(legacy(graph, desc)).all()
    with pytest.raises(RuntimeError, match="pooling"):
        new.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize("field", ["atom_features", "edge_vec", "edge_len", "descriptors"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_actual_encoder_rejects_nonfinite_inputs_before_message_passing(torch_api, field, bad):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    target = desc if field == "descriptors" else graph[field]
    target.reshape(-1)[0] = bad
    with pytest.raises(ValueError, match="finite"):
        model(graph, desc)


@pytest.mark.parametrize("bad", [0.0, -1.0, 9.0, 1.0])
def test_actual_encoder_rejects_zero_negative_or_inconsistent_edge_length(torch_api, bad):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    graph["edge_len"][0] = bad
    with pytest.raises(ValueError, match="edge.*len|edge.*norm"):
        model(graph, desc)


@pytest.mark.parametrize("n_atoms", [0, -1, 6, 51])
def test_actual_encoder_rejects_atom_count_not_matching_real_rows(torch_api, n_atoms):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    graph["n_atoms"][0] = n_atoms
    with pytest.raises(ValueError, match="n_atoms|atom.*count|capacity"):
        model(graph, desc)


@pytest.mark.parametrize("kind", ["edge_vec", "edge_len", "edge_src_dtype", "n_atoms_dtype", "graph_offsets", "descriptors", "atom_features", "empty_batch"])
def test_actual_encoder_rejects_malformed_tensor_contract(torch_api, kind):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    if kind == "edge_vec":
        graph[kind] = graph[kind][:, :2]
    elif kind == "edge_len":
        graph[kind] = graph[kind][:, None]
    elif kind.endswith("_dtype"):
        graph[kind[:-6]] = graph[kind[:-6]].double()
    elif kind == "descriptors":
        desc = desc[:, :-1]
    elif kind == "atom_features":
        graph[kind] = graph[kind][:, :-1]
    elif kind == "empty_batch":
        graph["n_atoms"] = graph["n_atoms"][:0]
        graph["graph_offsets"] = graph["graph_offsets"][:0]
        desc = desc[:0]
    else:
        graph[kind] = graph[kind].reshape(1, 1)
    with pytest.raises(ValueError, match="shape|integer|batch"):
        model(graph, desc)


@pytest.mark.parametrize("kind", ["cross_graph", "negative_src", "negative_dst", "out_of_range", "padding_edge", "overlap", "negative_offset", "offset_range", "offset_real_rows"])
def test_actual_encoder_rejects_invalid_graph_ids_and_ranges(torch_api, kind):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch, [_skew(), _cubic()])
    if kind == "cross_graph":
        graph["edge_dst"][0] = 50
    elif kind == "negative_src":
        graph["edge_src"][0] = -1
    elif kind == "negative_dst":
        graph["edge_dst"][0] = -1
    elif kind == "out_of_range":
        graph["edge_src"][0] = 100
    elif kind == "padding_edge":
        graph["edge_src"][0] = 6
    elif kind == "overlap":
        graph["graph_offsets"][1] = 0
    elif kind == "negative_offset":
        graph["graph_offsets"][0] = -1
    elif kind == "offset_range":
        graph["graph_offsets"][1] = 100
    else:
        graph["graph_offsets"][0] = 1
    with pytest.raises(ValueError, match="graph|edge|offset|padding"):
        model(graph, desc)


def test_actual_encoder_rejects_more_than_twelve_selected_neighbors(torch_api):
    torch, ese = torch_api
    model = ese.EquivariantStructureEncoder(multiplicity=2, num_layers=1, embedding_dim=8).double().eval()
    graph, desc = _torch_graph(torch)
    for key in ("edge_src", "edge_dst", "edge_vec", "edge_len"):
        graph[key] = torch.cat([graph[key], graph[key][:1]])
    with pytest.raises(ValueError, match="12|twelve|neighbor"):
        model(graph, desc)


def test_actual_cpu_state_dict_roundtrip_is_exact_in_eval(torch_api, tmp_path):
    torch, ese = torch_api
    torch.manual_seed(2024)
    options = dict(multiplicity=4, num_layers=2, embedding_dim=8)
    model = ese.EquivariantStructureEncoder(**options).double()
    graph, desc = _torch_graph(torch, [_skew(), _cubic()])
    model.train()
    model(graph, desc)  # exercise nondefault BatchNorm buffers; no optimizer/training run
    model.eval()
    with torch.no_grad():
        expected = model(graph, desc)
    path = tmp_path / "synthetic-cpu-state.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    clone = ese.EquivariantStructureEncoder(**options).double().eval()
    clone.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = clone(graph, desc)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("cpu_state_dict_reload_max_abs_error", (actual - expected).abs().max().item())


@pytest.mark.parametrize("empty_edges", [False, True])
def test_actual_encoder_backward_has_finite_parameter_and_input_gradients(torch_api, empty_edges):
    torch, ese = torch_api
    torch.manual_seed(7)
    model = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=2, embedding_dim=8).double().train()
    records = [_skew(), _cubic()]
    if empty_edges:
        records = [dict(_cubic(), lattice=np.eye(3) * 20)]
    graph, desc = _torch_graph(torch, records)
    graph["atom_features"].requires_grad_()
    graph["edge_vec"].requires_grad_()
    desc.requires_grad_()
    pred = model(graph, desc)
    loss = pred.square().mean()
    loss.backward()
    assert torch.isfinite(pred).all() and torch.isfinite(loss)
    gradients = {name: p.grad for name, p in model.named_parameters()}
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert torch.isfinite(graph["atom_features"].grad).all()
    assert torch.isfinite(graph["edge_vec"].grad).all()
    assert torch.isfinite(desc.grad).all()
    print("finite_parameter_gradients", len(gradients), "empty_edges", empty_edges,
          "loss", loss.item(), "max_abs_grad", max(g.abs().max().item() for g in gradients.values()))


def test_actual_default_encoder_uses_full_96_vector_components_invariantly(torch_api):
    torch, ese = torch_api
    torch.manual_seed(42)
    model = ese.EquivariantStructureEncoder().double().eval()
    graph, desc = _torch_graph(torch, [_skew(), _cubic()])
    vector_slice = next(sl for (_, ir), sl in zip(model.node_irreps, model.node_irreps.slices()) if ir.l == 1)
    assert vector_slice.stop - vector_slice.start == 96
    hidden = []
    hook = model.conv_layers[-1].register_forward_hook(lambda module, args, output: hidden.append(output.detach()))
    with torch.no_grad():
        expected = model(graph, desc)
        assert hidden[-1][:, vector_slice].abs().sum() > 0
        for parity in (1, -1):
            matrix = torch.tensor(_rotation() * parity, dtype=torch.float64)
            moved = dict(graph, edge_vec=graph["edge_vec"] @ matrix.T)
            actual = model(moved, desc)
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
            print("default_m32_rotation_error", parity, (actual - expected).abs().max().item())
    hook.remove()


def test_actual_encoder_eval_preserves_bn_state_across_batch_partition_and_order(torch_api):
    import copy
    torch, ese = torch_api
    torch.manual_seed(42)
    model = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=2, embedding_dim=8).double().train()
    graph, desc = _torch_graph(torch, [_skew(), _cubic()])
    model(graph, desc)
    model.eval()
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        merged = model(graph, desc)
        separate = torch.cat([model(*_torch_graph(torch, [rec])) for rec in [_skew(), _cubic()]])
        reversed_result = model(*_torch_graph(torch, [_cubic(), _skew()]))
    torch.testing.assert_close(merged, separate, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(merged, reversed_result.flip(0), rtol=1e-10, atol=1e-10)
    for key, old in before.items():
        if torch.is_tensor(old):
            torch.testing.assert_close(model.state_dict()[key], old, rtol=0, atol=0)
        else:
            assert model.state_dict()[key] == old
    print("eval_partition_error", (merged - separate).abs().max().item(), "state_unchanged", True)


@pytest.mark.parametrize("change", ["translation", "individual_images", "rotation"])
def test_actual_encoder_on_rebuilt_tie_free_graphs_respects_geometry(torch_api, change):
    torch, ese = torch_api
    torch.manual_seed(42)
    model = ese.EquivariantStructureEncoder(multiplicity=4, num_layers=2, embedding_dim=8).double().eval()
    rec = _skew()
    moved = dict(rec)
    if change == "translation":
        moved["fractional_coordinates"] = rec["fractional_coordinates"] + [.27, -.33, 1.56]
    elif change == "individual_images":
        moved["fractional_coordinates"] = rec["fractional_coordinates"] + np.arange(15).reshape(5, 3)
    else:
        moved["lattice"] = rec["lattice"] @ _rotation().T
    with torch.no_grad():
        before = model(*_torch_graph(torch, [rec]))
        after = model(*_torch_graph(torch, [moved]))
    # Graph caches store float32 geometry; this includes quantization on rebuild.
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)
    print("rebuilt_graph_encoder_error", change, (before - after).abs().max().item())


@pytest.mark.parametrize("field,value", [
    ("valid_idx", np.array([2])),
    ("valid_material_ids", np.array(["toy-train-2"])),
    ("pair_valid", np.array([False, False, True])),
])
def test_cache_loader_rejects_inconsistent_valid_row_mapping(tmp_path, monkeypatch, field, value):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    with np.load(out / "graphs_train.npz") as d:
        data = {key: d[key] for key in d.files}
    data[field] = value
    path = tmp_path / "corrupted-mapping.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="valid|mapping"):
        eqg.load_graph_cache(path, expected_material_ids=data["material_ids"], expected_valid=data["valid"])


@pytest.mark.parametrize("kind", ["nonfinite_desc", "wrong_offset", "edge_oob", "invalid_atom_count", "bad_length", "wrong_max_neighbors", "wrong_feature_shape", "bad_valid_idx_dtype", "missing_image"])
def test_cache_loader_checks_array_schema_before_model_consumption(tmp_path, monkeypatch, kind):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    eqg.main()
    with np.load(out / "graphs_train.npz") as d:
        data = {key: d[key] for key in d.files}
    if kind == "nonfinite_desc":
        data["descriptors"][0, 0] = np.nan
    elif kind == "wrong_offset":
        data["edge_offsets"][1, 0] = 0
    elif kind == "edge_oob":
        data["edge_dst"][0] = 1
    elif kind == "invalid_atom_count":
        data["n_atoms"][1] = 2
    elif kind == "bad_length":
        data["edge_len"][0] = 1
    elif kind == "wrong_max_neighbors":
        data["max_neighbors"] = np.array(13)
    elif kind == "wrong_feature_shape":
        data["atom_features"] = data["atom_features"][:, :, :-1]
    elif kind == "bad_valid_idx_dtype":
        data["valid_idx"] = data["valid_idx"].astype(float)
    else:
        del data["edge_image"]
    path = tmp_path / "invalid-array-schema.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="cache|graph|edge|finite|valid"):
        eqg.load_graph_cache(path, expected_material_ids=data["material_ids"], expected_valid=data["valid"])


def test_new_directory_does_not_allow_later_npz_collision_overwrite(tmp_path, monkeypatch):
    _, _, out = _cache_inputs(tmp_path, monkeypatch)
    original = eqg.build_one_graph
    target = out / "graphs_train.npz"
    protected_bytes = b"concurrent existing artifact"
    def collide_then_build(rec, max_atoms):
        if not target.exists():
            target.write_bytes(protected_bytes)
        return original(rec, max_atoms)
    monkeypatch.setattr(eqg, "build_one_graph", collide_then_build)
    with pytest.raises(FileExistsError):
        eqg.main()
    assert target.read_bytes() == protected_bytes


def test_explicit_bad_per_atom_species_does_not_fall_back_to_deduplicated_species():
    assert eqg.build_one_graph(dict(_cubic(), species_per_atom=[]), 50) is None


def test_finite_input_that_overflows_cartesian_coordinates_is_rejected(monkeypatch):
    def unsafe(*args, **kwargs):
        raise AssertionError("Cartesian overflow reached native neighbor search")
    monkeypatch.setattr(Structure, "get_all_neighbors", unsafe)
    rec = dict(_cubic(), fractional_coordinates=[[1e308, 0, 0]])
    assert eqg.build_one_graph(rec, 50) is None
    with pytest.raises(ValueError, match="finite"):
        eqg.cg.build_crystal_graph(rec["lattice"], rec["species"], rec["fractional_coordinates"])


def test_empty_cache_keeps_schema_shapes_without_inventing_graphs(tmp_path, monkeypatch):
    pairs, _, out = _cache_inputs(tmp_path, monkeypatch)
    ids, valid = np.array([], dtype="U1"), np.array([], dtype=bool)
    for split in ("train", "test"):
        np.savez(pairs / f"p2_{split}.npz", material_ids=ids, valid=valid)
    eqg.main()
    data = eqg.load_graph_cache(out / "graphs_train.npz", expected_material_ids=ids, expected_valid=valid)
    assert data["edge_offsets"].shape == (0, 2)
    assert data["edge_vec"].shape == (0, 3)
    assert data["atom_features"].shape == (0, 50, 110)


def test_cli_capacity_cannot_bypass_bound_when_all_pairs_are_invalid(tmp_path, monkeypatch):
    import sys
    _, _, out = _cache_inputs(tmp_path, monkeypatch, valid=(False,))
    sys.argv += ["--max-atoms", "51"]
    with pytest.raises(ValueError, match="max_atoms"):
        eqg.main()
    assert not (out / "graphs_train.npz").exists()


def test_real_graph_cli_roundtrip_is_read_only_on_sources_and_refuses_rerun(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    import sys
    pairs, sidecar, out = _cache_inputs(tmp_path, monkeypatch)
    sources = [sidecar, pairs / "p2_train.npz", pairs / "p2_test.npz"]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    command = [sys.executable, "-B", str(ROOT / "scripts/prepare_p2_equivariant_graphs.py"),
               "--sidecar", str(sidecar), "--pairs", str(pairs), "--out", str(out)]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stdout + first.stderr
    data = eqg.load_graph_cache(out / "graphs_train.npz",
        expected_material_ids=np.array(["toy-train-0", "toy-train-1", "toy-train-2"]),
        expected_valid=np.array([True, False, False]))
    assert data["edge_len"].shape == (12,)
    output_sha = hashlib.sha256((out / "graphs_train.npz").read_bytes()).hexdigest()
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert second.returncode != 0 and "FileExistsError" in second.stderr
    assert hashlib.sha256((out / "graphs_train.npz").read_bytes()).hexdigest() == output_sha
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == hashes
