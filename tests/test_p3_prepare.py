"""Synthetic HDF5/crystal-graph checks, not scientific accuracy evidence."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p3_prepare_under_test", ROOT / "scripts/prepare_p3_multiband.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
from src.utils.selection_manifest import sha256_file


@pytest.fixture
def tiny_sources(tmp_path):
    structure = Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]])
    record = {
        "material_id": "spin-Si",
        "lattice": structure.lattice.matrix.tolist(),
        "species_per_atom": [str(site.specie) for site in structure],
        "fractional_coordinates": structure.frac_coords.tolist(),
    }
    h5_path = tmp_path / "bands.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("spin-Si")
        # spin-major, then original band index; EF is deliberately nonzero.
        group["energies"] = np.array([
            [[-3, -3, -3, -3], [-1, -1, 1, 1]],
            [[20, 20, 20, 20], [1, 1, 1, 1]],
        ], dtype=np.float32) + 2.0
        group["k_distances"] = [0, 0.5, 0.5, 1]
        group.attrs["source"] = "mp"
        group.attrs["efermi"] = 2.0
        group.attrs["source_schema"] = "mp_pymatgen_v1"
        group.attrs["band_layout"] = "spin_band_k"
        group.attrs["spin_channels"] = 2
        group.attrs["source_semantics_evidence"] = "synthetic-pymatgen-contract"
        handle.copy(group, "test-Si")
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps([record, {**record, "material_id": "test-Si"}]),
                       encoding="utf-8")
    split_npz = tmp_path / "frozen_split.npz"
    np.savez(split_npz, material_ids_train=np.array(["spin-Si"]),
             groups_train=np.array([221]), material_ids_test=np.array(["test-Si"]),
             groups_test=np.array([225]))
    return {"h5": h5_path, "sidecar": sidecar, "split_npz": split_npz,
            "record": record, "out_dir": tmp_path / "p3_multiband_v2_20260908"}


def test_build_one_preserves_spin_selection_and_discontinuous_segments(tiny_sources):
    with h5py.File(tiny_sources["h5"], "r") as handle:
        result = prepare.build_one(handle["spin-Si"],
                                   {"spin-Si": tiny_sources["record"]},
                                   max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert result is not None
    np.testing.assert_array_equal(result["segment_ids"], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result["selected_band_indices"], [0, 1, 3, -1, -1])
    np.testing.assert_array_equal(result["mask"], [True, True, True, False, False])
    np.testing.assert_allclose(result["bands"][1], [-1, -1, 1, 1, 1])
    np.testing.assert_allclose(result["bands"][0], -3)
    np.testing.assert_allclose(result["bands"][2], 1)
    np.testing.assert_array_equal(result["bands"][3:], 0)
    assert result["n_bands"] == 3
    assert result["source_segment_count"] == 2
    assert result["graph"]["n_atoms"] == 1
    assert result["graph"]["atom_features"].shape == (2, 110)
    assert np.all(result["graph"]["neighbor_dist"][0] > 0)


def cli_args(sources, *extra):
    return [str(ROOT / "scripts/prepare_p3_multiband.py"),
            "--h5", str(sources["h5"]), "--sidecar", str(sources["sidecar"]),
            "--split-npz", str(sources["split_npz"]),
            "--out-dir", str(sources["out_dir"]),
            "--max-bands", "5", "--n-k", "5", "--max-atoms", "2", *extra]


def run_prepare(monkeypatch, sources, *extra):
    monkeypatch.setattr(sys, "argv", cli_args(sources, *extra))
    return prepare.main()


@pytest.mark.parametrize("existing", [None, "p3_train.npz", "p3_test.npz",
                                     "p3_prepare_report.json", "p3_train_exclusions.json"])
def test_output_directory_must_be_new_without_overwrite(monkeypatch, tiny_sources, existing):
    out_dir = tiny_sources["out_dir"]
    out_dir.mkdir()
    if existing:
        (out_dir / existing).write_bytes(b"immutable sentinel")
    before = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    with pytest.raises(FileExistsError, match="exist|new|overwrite"):
        run_prepare(monkeypatch, tiny_sources)
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == before


def test_npz_preserves_frozen_id_group_order_without_pickle(monkeypatch, tiny_sources):
    ids = ["missing-h5", "spin-Si", "缺失结构"]
    groups = np.array([13, 221, 14], dtype=np.int64)
    # Legacy object IDs are trusted, project-built local sources only.
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids, dtype=object),
             groups_train=groups, material_ids_test=np.array(["test-Si"], dtype=object),
             groups_test=np.array([225]), y_type_train=np.array([999, 999, 999]))
    run_prepare(monkeypatch, tiny_sources)
    with np.load(tiny_sources["out_dir"] / "p3_train.npz", allow_pickle=False) as data:
        assert data["material_ids"].dtype.kind == "U"
        np.testing.assert_array_equal(data["material_ids"], ids)
        np.testing.assert_array_equal(data["groups"], groups)
        np.testing.assert_array_equal(data["valid"], [False, True, False])
        np.testing.assert_array_equal(data["segment_ids"][1], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(data["segment_ids"][[0, 2]], -1)
        np.testing.assert_array_equal(data["selected_band_indices"][1], [0, 1, 3, -1, -1])
        np.testing.assert_array_equal(data["selected_band_indices"][[0, 2]], -1)
        for name in ("segment_ids", "selected_band_indices"):
            assert data[name].dtype.kind == "i"
        for name in data.files:
            assert not data[name].dtype.hasobject, name
    with np.load(tiny_sources["out_dir"] / "p3_test.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["material_ids"], ["test-Si"])
        np.testing.assert_array_equal(data["groups"], [225])


@pytest.mark.parametrize("problem", ["missing_k", "empty_k", "nonfinite_k", "k_length",
                                     "decreasing_k", "matrix_k", "singleton_segment",
                                     "empty_bands", "nonfinite_bands", "nonfinite_ef",
                                     "missing_ef"])
def test_build_one_rejects_invalid_bands_or_k_with_reason(tiny_sources, problem):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        k_values = {
            "empty_k": [], "nonfinite_k": [0, 0.5, np.nan, 1],
            "k_length": [0, 0.5, 1], "decreasing_k": [0, 0.5, 0.25, 1],
            "matrix_k": [[0, 0.5], [0.5, 1]], "singleton_segment": [0, 0, 0.5, 1],
        }
        if problem == "missing_k" or problem in k_values:
            del group["k_distances"]
            if problem in k_values:
                group["k_distances"] = k_values[problem]
        elif problem == "empty_bands":
            del group["energies"]
            group["energies"] = np.empty((0, 4))
        elif problem == "nonfinite_bands":
            group["energies"][0, 0, 0] = np.inf
        elif problem == "nonfinite_ef":
            group.attrs["efermi"] = np.nan
        else:
            del group.attrs["efermi"]
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(ValueError, match="invalid_band_or_k") as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert caught.value.reason == "invalid_band_or_k"


def test_every_excluded_id_has_a_reason_and_retains_its_row(monkeypatch, tiny_sources):
    reasons = {"missing-h5": "missing_energies", "no-energy": "missing_energies",
               "no-structure": "missing_structure", "bad-k": "invalid_band_or_k",
               "bad-graph": "graph_error", "too-large": "atom_capacity_exceeded"}
    ids = ["spin-Si", *reasons]
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        for mid in list(reasons)[1:]:
            handle.copy(handle["spin-Si"], mid)
        del handle["no-energy/energies"]
        del handle["bad-k/k_distances"]
    records = json.loads(tiny_sources["sidecar"].read_text(encoding="utf-8"))
    records.extend([{**tiny_sources["record"], "material_id": "bad-graph",
                     "lattice": [[0, 0, 0]] * 3},
                    {**tiny_sources["record"], "material_id": "too-large",
                     "species_per_atom": ["Si"] * 3,
                     "fractional_coordinates": [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]}])
    tiny_sources["sidecar"].write_text(json.dumps(records), encoding="utf-8")
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids),
             groups_train=np.arange(len(ids)), material_ids_test=np.array(["test-Si"]),
             groups_test=np.array([225]))
    run_prepare(monkeypatch, tiny_sources)
    excluded = json.loads((tiny_sources["out_dir"] / "p3_train_exclusions.json").read_text())
    assert {entry["material_id"]: entry["reason"] for entry in excluded} == reasons
    assert [entry["split_index"] for entry in excluded] == list(range(1, len(ids)))
    assert all(entry["detail"] for entry in excluded)
    with np.load(tiny_sources["out_dir"] / "p3_train.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["material_ids"], ids)
        np.testing.assert_array_equal(data["valid"], [True] + [False] * len(reasons))
        assert int(data["valid"].sum()) + len(excluded) == len(ids)
        np.testing.assert_array_equal(data["band_mask"][~data["valid"]], False)
    assert json.loads((tiny_sources["out_dir"] / "p3_test_exclusions.json").read_text()) == []


@pytest.mark.parametrize("split", ["train", "test"])
def test_cli_selected_split_limit_is_explicit_smoke_without_hidden_ids(tiny_sources, split):
    np.savez(tiny_sources["split_npz"], **{
        f"material_ids_{split}": np.array(["spin-Si", "not-processed"]),
        f"groups_{split}": np.array([221, 14]),
    })
    completed = subprocess.run([sys.executable, *cli_args(tiny_sources, "--split", split,
                                                        "--limit", "1")],
                               cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    other = "test" if split == "train" else "train"
    assert not (tiny_sources["out_dir"] / f"p3_{other}.npz").exists()
    with np.load(tiny_sources["out_dir"] / f"p3_{split}.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["material_ids"], ["spin-Si", "not-processed"])
        np.testing.assert_array_equal(data["groups"], [221, 14])
        np.testing.assert_array_equal(data["valid"], [True, False])
    excluded = json.loads((tiny_sources["out_dir"] / f"p3_{split}_exclusions.json").read_text())
    assert excluded[0]["material_id"] == "not-processed"
    assert excluded[0]["reason"] == "smoke_limit"
    report = json.loads((tiny_sources["out_dir"] / "p3_prepare_report.json").read_text())
    assert report["smoke_only"] is True
    assert report[f"{split}_total"] == 2
    assert report[f"{split}_valid"] == 1
    reader = training_reader()
    with pytest.raises(ValueError, match="smoke.*--smoke-limit"):
        reader.load_split(tiny_sources["out_dir"] / f"p3_{split}.npz")
    data, _ = reader.load_split(tiny_sources["out_dir"] / f"p3_{split}.npz", smoke_limit=1)
    assert len(data["bands"]) == 1


def test_report_binds_complete_outputs_to_sources_code_and_segmentation(monkeypatch, tiny_sources):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        handle["test-Si/k_distances"][:] = [0, 0.25, 0.75, 1]
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(["spin-Si", "absent"]),
             groups_train=np.array([221, 14]), material_ids_test=np.array(["test-Si"]),
             groups_test=np.array([225]))
    source_hashes = {key: sha256_file(tiny_sources[key]) for key in ("h5", "split_npz", "sidecar")}
    run_prepare(monkeypatch, tiny_sources, "--max-bands", "3")
    report = json.loads((tiny_sources["out_dir"] / "p3_prepare_report.json").read_text())
    assert report["schema_version"] == 2
    assert report["status"] == "complete"
    assert report["smoke_only"] is False
    assert report["parameters"] == {"max_bands": 3, "n_k": 5, "delta_e": 5.0,
                                     "max_atoms": 2, "limit": 0, "split": "both"}
    for key, digest in source_hashes.items():
        assert report["sources"][key]["sha256"] == digest == sha256_file(tiny_sources[key])
    expected_code = ["scripts/prepare_p3_multiband.py", "src/data/multiband.py",
                     "src/data/crystal_graph.py", "src/utils/selection_manifest.py",
                     "src/data/aflow_adapter.py", "src/evaluation/multiband_metrics.py"]
    assert set(report["code_hashes"]) == set(expected_code)
    for path in expected_code:
        assert report["code_hashes"][path] == sha256_file(ROOT / path)
    assert report["selected_band_indices"]["field"] == "selected_band_indices"
    assert report["selected_band_indices"]["padding"] == -1
    assert "spin" in report["selected_band_indices"]["source_indexing"]
    assert report["segmentation"]["boundary_source"] == "duplicate_k_distances"
    assert report["segmentation"]["no_boundary_policy"] == "single_unverified_segment"
    assert report["segmentation"]["physical_k_path_verified"] is False
    assert set(report["splits"]) == {"train", "test"}
    for split, expected_total in (("train", 2), ("test", 1)):
        stats = report["splits"][split]
        assert stats["status"] == "complete"
        assert stats["original_total"] == expected_total
        assert stats["n_valid"] == 1
        assert stats["n_excluded"] == expected_total - 1
        assert stats["band_hist"] == {"3": 1}
        assert stats["max_bands_fraction"] == 1.0
        assert stats["output_npz"]["sha256"] == sha256_file(tiny_sources["out_dir"] / f"p3_{split}.npz")
        assert stats["exclusions_sha256"] == sha256_file(tiny_sources["out_dir"] / f"p3_{split}_exclusions.json")
    assert report["splits"]["train"]["duplicate_distance_materials"] == 1
    assert report["splits"]["test"]["single_unverified_segment_materials"] == 1


@pytest.mark.parametrize("failure", ["npz", "exclusions", "report"])
def test_interrupted_output_never_publishes_a_partial_target(monkeypatch, tiny_sources, failure):
    real_dump = prepare.json.dump

    def interrupted_json(data, handle, **kwargs):
        should_interrupt = ((failure == "exclusions" and isinstance(data, list))
                            or (failure == "report" and isinstance(data, dict)
                                and data.get("schema_version") == 2))
        if should_interrupt:
            handle.write("{partial")
            raise KeyboardInterrupt("simulated interrupted write")
        return real_dump(data, handle, **kwargs)

    def interrupted_npz(target, **arrays):
        if hasattr(target, "write"):
            target.write(b"partial zip")
        else:
            Path(target).write_bytes(b"partial zip")
        raise KeyboardInterrupt("simulated interrupted write")

    monkeypatch.setattr(prepare.json, "dump", interrupted_json)
    if failure == "npz":
        monkeypatch.setattr(prepare.np, "savez", interrupted_npz)
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        run_prepare(monkeypatch, tiny_sources)
    target = {"npz": "p3_train.npz", "exclusions": "p3_train_exclusions.json",
              "report": "p3_prepare_report.json"}[failure]
    assert not (tiny_sources["out_dir"] / target).exists()
    assert not (tiny_sources["out_dir"] / "p3_prepare_report.json").exists()
    assert not list(tiny_sources["out_dir"].glob("*.tmp"))


@pytest.mark.parametrize("ids", [[], ["absent"]])
def test_cli_zero_valid_is_nonzero_with_failed_audit_not_success(tiny_sources, ids):
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids, dtype=str),
             groups_train=np.full(len(ids), 14), material_ids_test=np.array(["test-Si"]),
             groups_test=np.array([225]))
    completed = subprocess.run([sys.executable, *cli_args(tiny_sources)], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert "zero valid" in completed.stderr.lower()
    report = json.loads((tiny_sources["out_dir"] / "p3_prepare_report.json").read_text())
    assert report["schema_version"] == 2
    assert report["status"] == "failed"
    assert report["failure_reason"] == "zero_valid_split"
    assert report["splits"]["train"]["n_valid"] == 0
    assert report["splits"]["train"]["n_excluded"] == len(ids)
    assert report["splits"]["train"]["status"] == "zero_valid"
    assert report["splits"]["test"]["n_valid"] == 1
    with np.load(tiny_sources["out_dir"] / "p3_train.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["material_ids"], ids)
        assert not data["valid"].any()
    reader = training_reader()
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="P3 completion"):
            reader.load_split(tiny_sources["out_dir"] / f"p3_{split}.npz")


@pytest.mark.parametrize("option,value", [("--max-bands", "0"), ("--n-k", "1"),
                                           ("--delta-e", "nan"), ("--delta-e", "-1"),
                                           ("--max-atoms", "0"), ("--limit", "-1")])
def test_invalid_cli_parameters_fail_before_creating_output(monkeypatch, tiny_sources, option, value):
    with pytest.raises(SystemExit) as caught:
        run_prepare(monkeypatch, tiny_sources, option, value)
    assert caught.value.code == 2
    assert not tiny_sources["out_dir"].exists()


@pytest.mark.parametrize("problem", ["short_groups", "matrix_ids", "duplicate_ids",
                                     "object_groups", "numeric_id", "missing_groups"])
def test_malformed_frozen_split_fails_closed_without_deriving_groups(monkeypatch, tiny_sources, problem):
    arrays = {"material_ids_train": np.array(["spin-Si"]), "groups_train": np.array([221])}
    if problem == "short_groups":
        arrays["groups_train"] = np.array([], dtype=int)
    elif problem == "matrix_ids":
        arrays["material_ids_train"] = np.array([["spin-Si"]])
    elif problem == "duplicate_ids":
        arrays["material_ids_train"] = np.array(["spin-Si", "spin-Si"])
        arrays["groups_train"] = np.array([221, 221])
    elif problem == "object_groups":
        arrays["groups_train"] = np.array([221], dtype=object)
    elif problem == "numeric_id":
        arrays["material_ids_train"] = np.array([123], dtype=object)
    else:
        del arrays["groups_train"]
    np.savez(tiny_sources["split_npz"], **arrays)
    with pytest.raises(ValueError, match="frozen split"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert not list(tiny_sources["out_dir"].glob("*.npz"))
    assert not (tiny_sources["out_dir"] / "p3_prepare_report.json").exists()


def test_source_change_during_build_cannot_publish_a_success_report(monkeypatch, tiny_sources):
    real_build = prepare.build_one
    changed = False

    def build_then_mutate_fixture(*args, **kwargs):
        nonlocal changed
        result = real_build(*args, **kwargs)
        if not changed:
            with tiny_sources["sidecar"].open("a", encoding="utf-8") as handle:
                handle.write(" ")
            changed = True
        return result

    monkeypatch.setattr(prepare, "build_one", build_then_mutate_fixture)
    with pytest.raises(RuntimeError, match="source.*changed"):
        run_prepare(monkeypatch, tiny_sources)
    assert not (tiny_sources["out_dir"] / "p3_prepare_report.json").exists()


def test_three_atom_structure_exceeds_capacity_without_valid_truncation(tiny_sources):
    struct = Structure(Lattice.cubic(3), ["Si"] * 3,
                       [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    record = {**tiny_sources["record"], "species_per_atom": [str(s.specie) for s in struct],
              "fractional_coordinates": struct.frac_coords.tolist()}
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(ValueError, match="atom_capacity_exceeded") as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": record},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert caught.value.reason == "atom_capacity_exceeded"
    assert "3" in caught.value.detail and "2" in caught.value.detail


def test_unencoded_real_element_is_graph_error_not_valid_padding(tiny_sources):
    # Og is real and accepted by pymatgen, but absent from the graph vocabulary.
    struct = Structure(Lattice.cubic(3), ["Og"], [[0, 0, 0]])
    record = {**tiny_sources["record"], "species_per_atom": [str(s.specie) for s in struct]}
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(ValueError, match="graph_error") as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": record},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert caught.value.reason == "graph_error"
    assert "element" in caught.value.detail


def test_malformed_species_is_an_auditable_graph_error(tiny_sources):
    record = {**tiny_sources["record"], "species_per_atom": 42}
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(ValueError, match="graph_error") as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": record},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert caught.value.reason == "graph_error"


@pytest.mark.parametrize("overlap", ["material_ids", "groups"])
def test_both_splits_reject_frozen_overlap_even_outside_smoke_prefix(monkeypatch, tiny_sources, overlap):
    np.savez(tiny_sources["split_npz"],
             material_ids_train=np.array(["spin-Si", "shared"]), groups_train=np.array([221, 14]),
             material_ids_test=np.array(["test-Si", "shared" if overlap == "material_ids" else "other"]),
             groups_test=np.array([225, 14 if overlap == "groups" else 15]))
    with pytest.raises(ValueError, match="[Oo]uter.*overlap"):
        run_prepare(monkeypatch, tiny_sources, "--limit", "1")
    assert not list(tiny_sources["out_dir"].glob("*.npz"))
    assert not (tiny_sources["out_dir"] / "p3_prepare_commit.json").exists()


def test_prepare_train_only_never_parses_outer_arrays(monkeypatch, tiny_sources):
    real_getitem = np.lib.npyio.NpzFile.__getitem__

    def train_only(archive, key):
        assert not key.endswith("_test"), "prepare --split train parsed outer arrays"
        return real_getitem(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", train_only)
    run_prepare(monkeypatch, tiny_sources, "--split", "train")
    commit = json.loads((tiny_sources["out_dir"] / "p3_prepare_commit.json").read_text())
    assert set(commit["completion_sha256"]) == {"train"}


def test_zero_valid_outer_cannot_authorize_otherwise_valid_train(monkeypatch, tiny_sources):
    np.savez(tiny_sources["split_npz"],
             material_ids_train=np.array(["spin-Si"]), groups_train=np.array([221]),
             material_ids_test=np.array(["absent"]), groups_test=np.array([225]))
    with pytest.raises(RuntimeError, match="zero valid"):
        run_prepare(monkeypatch, tiny_sources)
    with pytest.raises(ValueError, match="P3 completion"):
        training_reader().load_split(tiny_sources["out_dir"] / "p3_train.npz")


def test_cli_prepare_train_evaluate_with_native_smoke_credentials(tiny_sources, tmp_path):
    """Independent CPU processes on synthetic sources; no accuracy claim."""
    ids = {"train": [f"synthetic-train-{i}" for i in range(14)],
           "test": [f"synthetic-outer-{i}" for i in range(6)]}
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        for mid in [*ids["train"], *ids["test"]]:
            handle.copy(handle["spin-Si"], mid)
    tiny_sources["sidecar"].write_text(json.dumps([
        {**tiny_sources["record"], "material_id": mid} for mid in [*ids["train"], *ids["test"]]
    ]), encoding="utf-8")
    np.savez(tiny_sources["split_npz"],
             material_ids_train=np.array(ids["train"]), groups_train=np.repeat(np.arange(1, 8), 2),
             material_ids_test=np.array(ids["test"]), groups_test=np.repeat(np.arange(101, 104), 2))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    prepared = subprocess.run([sys.executable, *cli_args(tiny_sources, "--limit", "8")],
                              cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    out = tmp_path / "native-cli-run"
    command = [sys.executable, str(ROOT / "scripts/train_p3_decoder.py"),
               "--data", str(tiny_sources["out_dir"]), "--out", str(out),
               "--allow-cpu-smoke", "--smoke-limit", "6", "--epochs", "1",
               "--batch-size", "2", "--d-model", "8"]
    trained = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    config = json.loads((out / "model_config.json").read_text())
    assert config["mode"] == "smoke" and config["preparation"]["smoke_only"] is True
    certificate = tiny_sources["out_dir"] / "p3_test_complete.json"
    saved = certificate.read_bytes()
    certificate.unlink()
    denied = subprocess.run([*command, "--stage", "evaluate"], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=120)
    assert denied.returncode != 0 and "P3 completion" in denied.stderr
    assert not (out / "evaluations").exists()
    certificate.write_bytes(saved)  # Restore only this test's synthetic credential.
    evaluated = subprocess.run([*command, "--stage", "evaluate"], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=120)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    paths = list((out / "evaluations").glob("*/evaluation.json"))
    assert len(paths) == 1
    report = json.loads(paths[0].read_text())
    assert report["mode"] == "smoke" and report["outer_preparation"]["smoke_only"] is True
    assert report["outer_split_audit"]["id_overlap"] == report["outer_split_audit"]["group_overlap"] == 0
    assert report["metrics"]["n_samples"] == 6
    manifest = training_reader().validate_inner_selection_manifest(str(out))
    assert manifest["reload_verification"]["predictions_match"] is True
    print(prepared.stdout.strip())
    print(trained.stdout.strip())
    print(evaluated.stdout.strip())
    print(json.dumps({"synthetic_only": True, "prepare_exit": prepared.returncode,
                      "train_exit": trained.returncode, "missing_test_credential_exit": denied.returncode,
                      "evaluate_exit": evaluated.returncode, "outer_split_audit": report["outer_split_audit"],
                      "reload_max_abs_difference": manifest["reload_verification"]["max_abs_difference"]}))


def test_hdf5_prepare_quarantines_the_bound_source_content(tiny_sources):
    from tests.test_aflow_source_quality import bound_payload, BAD_ID
    raw = bound_payload(BAD_ID)
    table = np.asarray(raw["bands_data"], np.float32)
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        del group["energies"], group["k_distances"]
        group["energies"] = table[:, 1:].T
        group["k_distances"] = table[:, 0]
        group.attrs["source"] = "aflow"
        group.attrs["efermi"] = 0.0
        group.attrs["source_efermi_absolute"] = raw["Efermi"]
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(prepare.SampleExclusion, match="source_quarantine_published_response_v1") as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=256, delta_e=5.0, max_atoms=2)
    assert caught.value.reason == "source_quarantine_published_response_v1"

def test_hdf5_formal_rejects_legacy_false_spin_as_unknown(tiny_sources):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        group.attrs["source"] = "aflow"
        group.attrs["efermi"] = 0.0
        group.attrs["is_spin_polarized"] = False
        for key in ("source_schema", "band_layout", "spin_channels", "source_semantics_evidence"):
            del group.attrs[key]
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(prepare.SampleExclusion, match="source_ambiguous"):
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)

def test_unknown_hdf5_provider_schema_cannot_be_formal_valid(tiny_sources):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        handle["spin-Si"].attrs["source_schema"] = "future_schema_unknown"
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(prepare.SampleExclusion, match="source_ambiguous"):
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)

def test_source_qa_cli_writes_audit_only_ledger_without_sidecar(tiny_sources, tmp_path):
    out = tmp_path / "source-ledger"
    command = [sys.executable, str(ROOT / "scripts/prepare_p3_multiband.py"),
               "--mode", "source-qa", "--h5", str(tiny_sources["h5"]),
               "--split-npz", str(tiny_sources["split_npz"]), "--split", "train",
               "--out-dir", str(out)]
    before = {key: sha256_file(tiny_sources[key]) for key in ("h5", "split_npz", "sidecar")}
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    path = out / "source_qa_train.json"
    report = json.loads(path.read_text())
    assert report["schema"] == "source_qa_ledger"
    assert report["status"] == "complete"
    assert report["purpose"] == "train_source_audit"
    assert report["counts"] == {"requested": 1, "valid": 1, "ambiguous": 0, "invalid": 0, "read_error": 0}
    assert [row["material_id"] for row in report["records"]] == ["spin-Si"]
    assert report["records"][0]["status"] == "verified_valid"
    assert report["records"][0]["content_sha256"]
    assert report["snapshot"]["h5"]["sha256"] == before["h5"]
    assert report["snapshot"]["split_npz"]["sha256"] == before["split_npz"]
    assert report["rule_version"] == "aflow-source-qa-v1"
    assert report["code_hashes"]["src/data/aflow_adapter.py"] == sha256_file(ROOT / "src/data/aflow_adapter.py")
    assert all(sha256_file(tiny_sources[key]) == digest for key, digest in before.items())
    assert sorted(p.name for p in out.iterdir()) == ["source_qa_train.json"]
    denied = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert denied.returncode != 0
    assert json.loads(path.read_text()) == report

def test_source_ledger_keeps_all_four_outcomes_and_exact_id_denominator(tiny_sources, tmp_path):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        handle.copy(handle["spin-Si"], "ambiguous")
        del handle["ambiguous"].attrs["source_semantics_evidence"]
        handle.copy(handle["spin-Si"], "invalid")
        handle["invalid"].attrs["num_bands"] = 99
    ids = ["invalid", "absent", "spin-Si", "ambiguous"]
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids), groups_train=np.arange(4))
    out = tmp_path / "four-outcome-ledger"
    report = prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], out, split="train")
    assert report["counts"] == {"requested": 4, "valid": 1, "ambiguous": 1, "invalid": 1, "read_error": 1}
    assert [r["material_id"] for r in report["records"]] == ids
    assert [r["status"] for r in report["records"]] == ["invalid", "read_error", "verified_valid", "ambiguous"]
    assert all(r["reason"] for r in report["records"])
    assert report == json.loads((out / "source_qa_train.json").read_text())

def test_source_ledger_input_mutation_cannot_publish_complete(monkeypatch, tiny_sources, tmp_path):
    original = prepare.assess_source_quality
    def mutate_after_read(record):
        result = original(record)
        with tiny_sources["split_npz"].open("ab") as handle:
            handle.write(b"changed-source-fixture")
        return result
    monkeypatch.setattr(prepare, "assess_source_quality", mutate_after_read)
    out = tmp_path / "changed-ledger"
    with pytest.raises(RuntimeError, match="source.*changed"):
        prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], out, split="train")
    assert not (out / "source_qa_train.json").exists()

@pytest.mark.parametrize("failure", ["partial", "post_rename", "source_after_rename"])
def test_source_ledger_interruption_never_leaves_completed_target(monkeypatch, tiny_sources, tmp_path, failure):
    out = tmp_path / "interrupted-ledger"
    original = prepare._atomic_write
    def interrupt(path, writer, **kwargs):
        if failure == "partial":
            def partial(handle):
                handle.write("{partial")
                raise KeyboardInterrupt("synthetic partial ledger")
            return original(path, partial, **kwargs)
        original(path, writer, **kwargs)
        if failure == "post_rename":
            raise KeyboardInterrupt("synthetic after rename")
        with tiny_sources["split_npz"].open("ab") as handle:
            handle.write(b"changed-after-publication")
    monkeypatch.setattr(prepare, "_atomic_write", interrupt)
    expected = RuntimeError if failure == "source_after_rename" else KeyboardInterrupt
    with pytest.raises(expected):
        prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], out, split="train")
    assert not (out / "source_qa_train.json").exists()
    assert not list(out.glob("*.tmp"))

@pytest.mark.parametrize("split", ["train", "test"])
def test_source_ledger_reads_only_the_explicit_split(monkeypatch, tiny_sources, tmp_path, split):
    real_get = np.lib.npyio.NpzFile.__getitem__
    seen = []
    def selected_only(archive, key):
        assert key in (f"material_ids_{split}", f"groups_{split}")
        seen.append(key)
        return real_get(archive, key)
    def no_graph(*args, **kwargs):
        pytest.fail("source ledger must not build graphs or tensors")
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    monkeypatch.setattr(prepare, "build_one", no_graph)
    out = tmp_path / ("isolated-" + split)
    report = prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], out, split=split)
    assert report["split"] == split
    assert seen == [f"material_ids_{split}", f"groups_{split}"]
    assert report["purpose"] == ("train_source_audit" if split == "train" else "outer_test_read_only_audit_not_for_selection")
    assert len(report["records"]) == 1


@pytest.mark.parametrize("split", ["both", "", None])
def test_source_ledger_requires_isolated_split_before_creating_output(tiny_sources, tmp_path, split):
    out = tmp_path / "denied-ledger"
    with pytest.raises(ValueError, match="isolated"):
        prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], out, split=split)
    assert not out.exists()

def test_hdf5_metadata_cannot_change_the_qa_checked_energy_reference(tiny_sources):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        handle["spin-Si"].create_group("metadata").attrs["efermi"] = 1.0
    with h5py.File(tiny_sources["h5"], "r") as handle:
        with pytest.raises(prepare.SampleExclusion, match="conflicting_source_metadata"):
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)

@pytest.mark.parametrize("key,value", [("num_bands", 99), ("num_kpoints", 99),
                                      ("efermi", np.nan), ("source_efermi_absolute", np.inf)])
def test_aflow_hdf5_consumes_same_numeric_qa(tiny_sources, key, value):
    from src.data.aflow_adapter import assess_source_quality
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        group.attrs["source"] = "aflow"
        group.attrs["source_schema"] = "aflow_bandsdata_v1"
        group.attrs["efermi"] = 0.0
        group.attrs[key] = value
    with h5py.File(tiny_sources["h5"], "r") as handle:
        record = prepare.read_hdf5_source_record(handle["spin-Si"])
        qa = assess_source_quality(record)
        assert qa["status"] == "invalid"
        with pytest.raises(prepare.SampleExclusion) as caught:
            prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                              max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
        assert caught.value.reason == qa["reason"]


def test_formal_prepare_records_qa_policy_in_report_and_credentials(monkeypatch, tiny_sources):
    run_prepare(monkeypatch, tiny_sources, "--split", "train")
    directory = tiny_sources["out_dir"]
    report = json.loads((directory / "p3_prepare_report.json").read_text())
    credential = json.loads((directory / "p3_train_complete.json").read_text())
    policy = {"rule_version": "aflow-source-qa-v1", "mode": "formal", "accepted_status": "verified_valid"}
    assert report.get("source_quality_policy") == policy
    assert credential.get("source_quality_policy") == policy


def test_real_bound_hdf5_exclusion_preserves_ids_and_completion(monkeypatch, tiny_sources, tmp_path):
    from tests.test_aflow_source_quality import bound_payload, BAD_ID, SHA
    raw = bound_payload(BAD_ID)
    table = np.asarray(raw["bands_data"], np.float32)
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle.create_group(BAD_ID)
        group["energies"] = table[:, 1:].T
        group["k_distances"] = table[:, 0]
        group.attrs.update({"source": "aflow", "efermi": 0.0,
                            "source_efermi_absolute": raw["Efermi"], "source_sha256": SHA[BAD_ID]})
        handle.copy(handle["spin-Si"], "unknown-source")
        del handle["unknown-source"].attrs["spin_channels"]
    ids = [BAD_ID, "spin-Si", "unknown-source"]
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids), groups_train=np.array([1, 2, 3]))
    before = sha256_file(tiny_sources["h5"])
    run_prepare(monkeypatch, tiny_sources, "--split", "train")
    directory = tiny_sources["out_dir"]
    with np.load(directory / "p3_train.npz") as data:
        np.testing.assert_array_equal(data["material_ids"], ids)
        np.testing.assert_array_equal(data["valid"], [False, True, False])
    excluded = json.loads((directory / "p3_train_exclusions.json").read_text())
    assert [(r["material_id"], r["reason"]) for r in excluded] == [
        (BAD_ID, "source_quarantine_published_response_v1"), ("unknown-source", "source_ambiguous")]
    assert json.loads((directory / "p3_prepare_commit.json").read_text())["status"] == "complete"
    assert sha256_file(tiny_sources["h5"]) == before
    ledger = prepare.source_qa_ledger(tiny_sources["h5"], tiny_sources["split_npz"], tmp_path / "bound-ledger", split="train")
    assert [r["material_id"] for r in ledger["records"]] == ids
    assert ledger["counts"] == {"requested": 3, "valid": 1, "ambiguous": 1, "invalid": 1, "read_error": 0}
    assert ledger["records"][0]["quarantine_evidence"]["compressed_sha256"] == SHA[BAD_ID]




def training_reader():
    spec = importlib.util.spec_from_file_location("p3_completion_reader", ROOT / "scripts/train_p3_decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_publishes_split_credentials_then_neutral_commit(monkeypatch, tiny_sources):
    writes = []
    real_write = prepare._atomic_write

    def record_write(path, *args, **kwargs):
        real_write(path, *args, **kwargs)
        writes.append(Path(path).name)

    monkeypatch.setattr(prepare, "_atomic_write", record_write)
    run_prepare(monkeypatch, tiny_sources)
    assert writes[-1] == "p3_prepare_commit.json"
    directory = tiny_sources["out_dir"]
    commit = json.loads((directory / writes[-1]).read_text())
    assert set(commit) == {"schema", "schema_version", "status", "completion_sha256"}
    assert commit["schema"] == "p3_prepare_commit"
    assert commit["schema_version"] == 1
    assert commit["status"] == "complete"
    assert set(commit["completion_sha256"]) == {"train", "test"}
    report = json.loads((directory / "p3_prepare_report.json").read_text())
    for split in ("train", "test"):
        path = directory / f"p3_{split}_complete.json"
        credential = json.loads(path.read_text())
        assert commit["completion_sha256"][split] == sha256_file(path)
        assert credential["schema"] == "p3_split_completion"
        assert credential["schema_version"] == 1
        assert credential["status"] == "complete"
        assert credential["split"] == split
        assert credential["smoke_only"] is False
        assert credential["output_npz"] == report["splits"][split]["output_npz"]
        assert credential["sources"] == report["sources"]
        assert credential["code_hashes"] == report["code_hashes"]
        assert credential["parameters"] == report["parameters"]
        assert "splits" not in credential and "material_ids" not in credential and "groups" not in credential
        data, _ = training_reader().load_split(directory / f"p3_{split}.npz")
        assert len(data["bands"]) == 1


@pytest.mark.parametrize("failure", ["report", "train_credential", "test_credential",
                                     "commit", "commit_after_replace", "source_after_report"])
def test_finalization_interruption_cannot_authorize_any_split(monkeypatch, tiny_sources, failure):
    real_write = prepare._atomic_write
    target = {"report": "p3_prepare_report.json", "train_credential": "p3_train_complete.json",
              "test_credential": "p3_test_complete.json", "commit": "p3_prepare_commit.json",
              "commit_after_replace": "p3_prepare_commit.json"}.get(failure)

    def interrupted(path, writer, **kwargs):
        if Path(path).name == target:
            if failure == "commit_after_replace":
                real_write(path, writer, **kwargs)
                raise KeyboardInterrupt("synthetic post-rename interruption")

            def partial(handle):
                handle.write("{partial")
                raise KeyboardInterrupt("synthetic finalization interruption")

            return real_write(path, partial, **kwargs)
        real_write(path, writer, **kwargs)
        if failure == "source_after_report" and Path(path).name == "p3_prepare_report.json":
            with tiny_sources["sidecar"].open("a") as handle:
                handle.write(" ")

    monkeypatch.setattr(prepare, "_atomic_write", interrupted)
    expected = RuntimeError if failure == "source_after_report" else KeyboardInterrupt
    with pytest.raises(expected, match="changed|synthetic"):
        run_prepare(monkeypatch, tiny_sources)
    directory = tiny_sources["out_dir"]
    assert not (directory / "p3_prepare_commit.json").exists()
    reader = training_reader()
    for split in ("train", "test"):
        # NPZ bytes may survive for diagnostics; they carry no completion authority.
        with pytest.raises(ValueError, match="P3 completion"):
            reader.load_split(directory / f"p3_{split}.npz")
    assert not list(directory.glob("*.tmp"))


def test_build_one_audits_full_raw_window_with_actual_compact_spin_indices(tiny_sources):
    from src.data.multiband import band_window_audit
    with h5py.File(tiny_sources["h5"], "r") as handle:
        raw = handle["spin-Si/energies"][...]
        out = prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                                max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert "band_window_audit" in out, "prepare does not wire the full-raw capacity ledger"
    indices = out["selected_band_indices"][out["mask"]]
    audit = out["band_window_audit"]
    assert audit == band_window_audit(raw, indices, efermi=2., max_bands=5, delta_e=5.)
    assert audit["raw_band_count"] == 4
    assert audit["selected_indices"] == [0, 1, 3]
    assert audit["selected_spin_band_indices"] == [[0, 0], [0, 1], [1, 1]]
    assert audit["window_candidates"] == audit["selected"] == 3
    assert audit["selection_branch"] == "window_overlap"


@pytest.mark.parametrize("missing_segment", [False, True])
def test_build_one_resampling_audit_uses_real_source_target_segments(tiny_sources, missing_segment):
    from src.evaluation.multiband_metrics import resampling_audit
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        if missing_segment:
            del group["energies"], group["k_distances"]
            group["energies"] = np.array([
                [[-2, -1, -1, 0, -2, -1], [1, 2, 2, 3, 1, 2]],
                [[-5, -4, -4, -3, -5, -4], [2, 3, 3, 4, 2, 3]],
            ], np.float32) + 2.
            group["k_distances"] = [0, .01, .01, .02, .02, 1]
        raw, k = group["energies"][...], group["k_distances"][...]
        out = prepare.build_one(group, {"spin-Si": tiny_sources["record"]},
                                max_bands=5, n_k=5, delta_e=5.0, max_atoms=2)
    assert "resampling_audit" in out, "prepare does not wire resampling diagnostics"
    source, target = prepare.path_segments(k, 5)
    audit = out["resampling_audit"]
    assert audit == resampling_audit(raw, out["selected_band_indices"][out["mask"]],
                                    out["bands"], out["mask"], source, target, efermi=2.)
    assert audit["source_segment_count"] == (3 if missing_segment else 2)
    assert audit["target_segment_count"] == 2
    assert audit["unrepresented_source_segment_ids"] == ([1] if missing_segment else [])
    if not missing_segment:
        assert audit["raw_full"]["frontier_state"] == "unresolved_between_segments"
        assert audit["resampled_selected"]["within_segment_crossing_bands"] == 0
    else:
        assert all(r["source_order_inversions"] == 0 for r in audit["raw_full"]["segments"])


def test_audits_preserve_declared_flat_spin_blocks_without_changing_selection(tiny_sources):
    from src.data.multiband import band_window_audit
    from src.evaluation.multiband_metrics import resampling_audit
    raw = np.repeat(np.array([[-2., -1.], [-5., 2.]])[..., None], 4, axis=2)
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        del group["energies"]
        group["energies"] = raw.reshape(4, 4) + 2.
        group.attrs["band_layout"] = "spin_major_flat"
        out = prepare.build_one(group, {"spin-Si": tiny_sources["record"]},
                                max_bands=5, n_k=5, delta_e=5., max_atoms=2)
    indices = out["selected_band_indices"][out["mask"]]
    assert out["band_window_audit"] == band_window_audit(raw + 2., indices, 2., 5, 5.)
    assert out["resampling_audit"] == resampling_audit(
        raw + 2., indices, out["bands"], out["mask"], np.array([0, 0, 1, 1]),
        out["segment_ids"], efermi=2.)
    np.testing.assert_array_equal(indices, [0, 1, 3])
    expected, mask, axis = prepare.extract_fermi_bands(raw.reshape(4, 4) + 2.,
                                                     [0, .5, .5, 1], 2., 5, 5., 5)
    np.testing.assert_array_equal(out["bands"], expected)
    np.testing.assert_array_equal(out["mask"], mask)
    np.testing.assert_array_equal(out["k_axis"], axis)


def test_prepare_publishes_complete_per_material_audit_denominator(monkeypatch, tiny_sources):
    ids = ["absent", "spin-Si", "not-processed"]
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids),
             groups_train=np.array([14, 221, 15]))
    run_prepare(monkeypatch, tiny_sources, "--split", "train", "--limit", "2")
    path = tiny_sources["out_dir"] / "p3_train_audit.json"
    assert path.exists(), "NEW prepare lacks its per-material audit ledger"
    ledger = json.loads(path.read_text())
    assert ledger["schema"] == "p3_prepare_audit" and ledger["schema_version"] == 1
    assert ledger["status"] == "diagnostic_only" and ledger["split"] == "train"
    assert ledger["counts"] == {"requested": 3, "audited": 1, "excluded": 1, "not_processed": 1}
    assert [row["material_id"] for row in ledger["records"]] == ids
    assert [row["split_index"] for row in ledger["records"]] == [0, 1, 2]
    assert [row["group"] for row in ledger["records"]] == [14, 221, 15]
    assert [row["status"] for row in ledger["records"]] == ["excluded", "audited", "not_processed"]
    for row in (ledger["records"][0], ledger["records"][2]):
        assert row["reason"] in ("missing_energies", "smoke_limit")
        assert row["band_window_audit"] is None and row["resampling_audit"] is None
    with h5py.File(tiny_sources["h5"], "r") as handle:
        expected = prepare.build_one(handle["spin-Si"], {"spin-Si": tiny_sources["record"]},
                                     max_bands=5, n_k=5, delta_e=5., max_atoms=2)
    for key in ("band_window_audit", "resampling_audit"):
        assert ledger["records"][1][key] == expected[key]
    report = json.loads((tiny_sources["out_dir"] / "p3_prepare_report.json").read_text())
    assert report["splits"]["train"]["audit"] == {
        "path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


@pytest.mark.parametrize("provider", ["aflow", "mp"])
def test_prepare_audit_binds_provenance_without_certifying_physical_k(monkeypatch, tiny_sources, provider):
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        group.attrs.update(source=provider, efermi=0. if provider == "aflow" else 2.,
                           source_efermi_absolute=6.75, source_sha256="d" * 64,
                           source_schema="aflow_bandsdata_v1" if provider == "aflow" else "mp_pymatgen_v1",
                           source_url="https://example.invalid/synthetic-source")
        expected_quality = prepare.assess_source_quality(prepare.read_hdf5_source_record(group))
    before = {key: sha256_file(tiny_sources[key]) for key in ("h5", "sidecar", "split_npz")}
    run_prepare(monkeypatch, tiny_sources, "--split", "train")
    directory = tiny_sources["out_dir"]
    ledger = json.loads((directory / "p3_train_audit.json").read_text())
    assert "source" in ledger["records"][0], "material ledger is not bound to source provenance"
    source = ledger["records"][0]["source"]
    assert source == {"hdf5_group": "/spin-Si", "provider": provider,
                      "declared_source_sha256": "d" * 64,
                      "source_url": "https://example.invalid/synthetic-source",
                      "quality": expected_quality, "raw_energy_shape": [2, 2, 4],
                      "raw_energy_dtype": "float32", "audit_energy_shape": [2, 2, 4],
                      "efermi_used_eV": 0. if provider == "aflow" else 2.,
                      "source_efermi_absolute": 6.75}
    report = json.loads((directory / "p3_prepare_report.json").read_text())
    for key in ("sources", "code_hashes", "parameters", "smoke_only", "source_quality_policy", "segmentation"):
        assert ledger[key] == report[key]
    assert ledger["output_npz"] == report["splits"]["train"]["output_npz"]
    assert ledger["code_hashes"]["src/evaluation/multiband_metrics.py"] == sha256_file(ROOT / "src/evaluation/multiband_metrics.py")
    assert ledger["segmentation"]["physical_k_path_verified"] is False
    assert ledger["segmentation"]["high_symmetry_points_verified"] is False
    assert ledger["accepted"] is False and "independent" in " ".join(ledger["limitations"])
    assert all(sha256_file(tiny_sources[key]) == digest for key, digest in before.items())


@pytest.mark.parametrize("corruption", ["window_count", "actual_indices", "capacity_flag", "quota",
                                        "raw_count", "target_count", "source_segments", "nan", "missing"])
def test_bad_prepare_audit_aborts_instead_of_completing_or_excluding(monkeypatch, tiny_sources, corruption):
    key = "band_window_audit" if corruption in ("window_count", "actual_indices", "capacity_flag", "quota", "missing") else "resampling_audit"
    original = getattr(prepare, key)
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "window_count":
            result["window_dropped"] += 1
        elif corruption == "actual_indices":
            result["selected_indices"] = [0, 1, 2]
        elif corruption == "capacity_flag":
            result["capacity_insufficient"] = True
        elif corruption == "quota":
            result["quota_underfill"] = 1
        elif corruption == "raw_count":
            result["raw_full"]["n_valid_bands"] -= 1
        elif corruption == "target_count":
            result["resampled_selected"]["n_valid_k"] -= 1
        elif corruption == "source_segments":
            result["source_segment_count"] += 1
        elif corruption == "nan":
            result["raw_full"]["segments"][0]["slot_dispersion_rms_eV"] = float("nan")
        else:
            del result["selected_indices"]
        return result
    monkeypatch.setattr(prepare, key, corrupt)
    with pytest.raises(RuntimeError, match="preparation audit"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    directory = tiny_sources["out_dir"]
    assert not (directory / "p3_prepare_commit.json").exists()
    assert not list(directory.glob("p3_*_complete.json"))
    assert not (directory / "p3_prepare_report.json").exists()


@pytest.mark.parametrize("during", ["p3_train_audit.json", "p3_prepare_report.json",
                                    "p3_train_complete.json", "p3_prepare_commit.json"])
def test_audit_publication_mutation_revokes_all_completion(monkeypatch, tiny_sources, during):
    original = prepare._atomic_write
    directory = tiny_sources["out_dir"]
    def mutate(path, writer, **kwargs):
        original(path, writer, **kwargs)
        if Path(path).name == during:
            with (directory / "p3_train_audit.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")  # Valid JSON, but no longer the checked diagnostic bytes.
    monkeypatch.setattr(prepare, "_atomic_write", mutate)
    with pytest.raises(RuntimeError, match="preparation audit"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert not (directory / "p3_prepare_commit.json").exists()
    assert not list(directory.glob("p3_*_complete.json"))
    assert not (directory / "p3_prepare_report.json").exists()
    assert not list(directory.glob("*.tmp"))


def test_split_completion_retains_v1_with_split_local_diagnostic_binding(monkeypatch, tiny_sources):
    run_prepare(monkeypatch, tiny_sources, "--split", "train")
    directory = tiny_sources["out_dir"]
    credential_path = directory / "p3_train_complete.json"
    credential = json.loads(credential_path.read_text())
    report = json.loads((directory / "p3_prepare_report.json").read_text())
    assert "preparation_audit" in credential, "commit does not bind the NEW diagnostic ledger"
    assert credential["preparation_audit"] == report["splits"]["train"]["audit"]
    assert credential["schema"] == "p3_split_completion" and credential["schema_version"] == 1
    assert report["schema_version"] == 2
    commit = json.loads((directory / "p3_prepare_commit.json").read_text())
    assert commit == {"schema": "p3_prepare_commit", "schema_version": 1, "status": "complete",
                      "completion_sha256": {"train": sha256_file(credential_path)}}
    assert not ({"records", "groups", "material_ids", "splits", "accepted"} & credential.keys())
    assert credential["preparation_audit"]["path"] == "p3_train_audit.json"


@pytest.fixture
def capacity_sources(tiny_sources):
    """Eleven synthetic rows; never use a real raw snapshot or outer sample."""
    constants = lambda values: np.repeat(np.asarray(values, np.float32)[:, None], 4, axis=1)
    samples = {
        "exact16": constants([*range(-8, 0), *range(1, 9)]),
        "quota13": constants([-1, *range(1, 13)]),
        "capped18": np.tile([-2., 2., -2., 2.], (18, 1)),
        "empty-window": constants([-30, -20]),
        "lost-segment": np.array([[-2.] * 6, [2.] * 6]),
    }
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        for mid, energies in samples.items():
            group = handle.create_group(mid)
            group["energies"] = energies
            group["k_distances"] = [0, .01, .01, .02, .02, 1] if mid == "lost-segment" else [0, .5, .5, 1]
            group.attrs.update(source="aflow", efermi=0., source_efermi_absolute=6.75,
                               source_schema="aflow_bandsdata_v1", band_layout="bands_k",
                               spin_channels=1, source_semantics_evidence="synthetic-contract-only")
        handle.copy(handle["spin-Si"], "flat-spin")
        flat = handle["flat-spin/energies"][...].reshape(4, 4)
        del handle["flat-spin/energies"]
        handle["flat-spin/energies"] = flat
        handle["flat-spin"].attrs["band_layout"] = "spin_major_flat"
        handle.copy(handle["spin-Si"], "ambiguous")
        del handle["ambiguous"].attrs["source_semantics_evidence"]
        handle.copy(handle["spin-Si"], "no-structure")
    valid_ids = [*samples, "spin-Si", "flat-spin"]
    ids = [*valid_ids, "absent", "ambiguous", "no-structure", "not-processed"]
    tiny_sources["sidecar"].write_text(json.dumps([
        {**tiny_sources["record"], "material_id": mid} for mid in valid_ids]), encoding="utf-8")
    np.savez(tiny_sources["split_npz"], material_ids_train=np.array(ids, dtype=object),
             groups_train=np.arange(1, len(ids) + 1),
             material_ids_test=np.array(["outer-sentinel"]), groups_test=np.array([225]))
    return {**tiny_sources, "valid_ids": valid_ids, "ids": ids}


def test_native_prepare_capacity_witness_preserves_arrays_and_denominators(capacity_sources):
    """Existing algorithm/property witness; not a claim of new selector TDD."""
    sources = capacity_sources
    command = [sys.executable, *cli_args(sources, "--split", "train", "--max-bands", "16", "--limit", "10")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    ledger = json.loads((sources["out_dir"] / "p3_train_audit.json").read_text())
    rows = {row["material_id"]: row for row in ledger["records"]}
    assert len(rows) == len(sources["ids"]) <= 32
    assert ledger["counts"] == {"requested": 11, "audited": 7, "excluded": 3, "not_processed": 1}
    with np.load(sources["out_dir"] / "p3_train.npz", allow_pickle=False) as data, h5py.File(sources["h5"], "r") as raw:
        np.testing.assert_array_equal(data["material_ids"], sources["ids"])
        assert len(sources["ids"]) == int(data["valid"].sum()) + ledger["counts"]["excluded"] + ledger["counts"]["not_processed"]
        for i, mid in enumerate(sources["valid_ids"]):
            record = prepare.read_hdf5_source_record(raw[mid])
            expected, mask, axis = prepare.extract_fermi_bands(record["energies"], record["k_distances"],
                                                             prepare._aflow_efermi(record), 16, 5., 5)
            np.testing.assert_array_equal(data["bands"][i], expected)
            np.testing.assert_array_equal(data["band_mask"][i], mask)
            np.testing.assert_array_equal(data["k_axis"], axis)
            audit = rows[mid]["band_window_audit"]
            assert audit["selected_indices"] == data["selected_band_indices"][i][mask].tolist()
            assert audit["window_candidates"] == audit["selected_in_window"] + audit["window_dropped"]
            assert audit["selection_candidates"] == audit["selected"] + audit["selection_dropped"]
        np.testing.assert_array_equal(data["band_mask"][~data["valid"]], False)
        np.testing.assert_array_equal(data["bands"][~data["valid"]], 0.)
    exact, quota, capped = [rows[mid]["band_window_audit"] for mid in ("exact16", "quota13", "capped18")]
    assert exact["slots_full"] and not exact["capacity_insufficient"] and exact["selection_dropped"] == 0
    assert exact["selected_outside_window"] == 6
    assert not quota["slots_full"] and not quota["capacity_insufficient"]
    assert quota["quota_underfill"] == quota["selection_dropped"] == 4
    assert capped["slots_full"] and capped["capacity_insufficient"] and capped["capacity_shortfall"] == 2
    assert rows["empty-window"]["band_window_audit"]["selection_branch"] == "nearest_fallback"
    assert rows["lost-segment"]["resampling_audit"]["unrepresented_source_segment_ids"] == [1]
    assert rows["ambiguous"]["reason"] == "source_ambiguous"
    assert rows["no-structure"]["reason"] == "missing_structure"
    assert rows["not-processed"]["status"] == "not_processed"
    assert not (sources["out_dir"] / "p3_test_audit.json").exists()


def test_prepare_audits_do_not_parse_outer_or_smoke_skipped_groups(monkeypatch, capacity_sources):
    real_npz = np.lib.npyio.NpzFile.__getitem__
    real_h5 = h5py.Group.__getitem__
    seen = []
    def train_only(archive, key):
        assert not key.endswith("_test")
        seen.append(key)
        return real_npz(archive, key)
    def allowed_groups(group, key):
        assert str(key) not in ("test-Si", "outer-sentinel", "not-processed")
        return real_h5(group, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", train_only)
    monkeypatch.setattr(h5py.Group, "__getitem__", allowed_groups)
    run_prepare(monkeypatch, capacity_sources, "--split", "train", "--max-bands", "16", "--limit", "10")
    assert seen == ["material_ids_train", "groups_train"]


@pytest.mark.parametrize("audit_function", ["band_window_audit", "resampling_audit"])
def test_audit_exception_is_not_a_source_qa_exclusion(monkeypatch, tiny_sources, audit_function):
    def fail(*args, **kwargs):
        raise ValueError("synthetic diagnostic failure")
    monkeypatch.setattr(prepare, audit_function, fail)
    with pytest.raises(ValueError, match="synthetic diagnostic failure") as caught:
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert not isinstance(caught.value, prepare.SampleExclusion)
    assert not (tiny_sources["out_dir"] / "p3_prepare_commit.json").exists()
    assert not list(tiny_sources["out_dir"].glob("p3_*_complete.json"))


@pytest.mark.parametrize("after_rename", [False, True])
def test_interrupted_audit_write_cannot_publish_completion(monkeypatch, tiny_sources, after_rename):
    original = prepare._atomic_write
    def interrupt(path, writer, **kwargs):
        if Path(path).name == "p3_train_audit.json":
            if after_rename:
                original(path, writer, **kwargs)
            else:
                def partial(handle):
                    handle.write(b"{partial")
                    raise KeyboardInterrupt("synthetic audit write")
                return original(path, partial, **kwargs)
            raise KeyboardInterrupt("synthetic audit rename")
        return original(path, writer, **kwargs)
    monkeypatch.setattr(prepare, "_atomic_write", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic audit"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert not (tiny_sources["out_dir"] / "p3_prepare_commit.json").exists()
    assert not list(tiny_sources["out_dir"].glob("p3_*_complete.json"))
    assert not list(tiny_sources["out_dir"].glob("*.tmp"))


@pytest.mark.parametrize("corruption", ["dropped_selected", "integer_flag", "unknown_branch", "lost_segments"])
def test_prepare_rejects_malformed_audit_index_and_flag_contracts(monkeypatch, tiny_sources, corruption):
    key = "resampling_audit" if corruption == "lost_segments" else "band_window_audit"
    original = getattr(prepare, key)
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "dropped_selected":
            result["window_dropped_indices"] = [1]
        elif corruption == "integer_flag":
            result["capacity_insufficient"] = int(result["capacity_insufficient"])
        elif corruption == "unknown_branch":
            result["selection_branch"] = "unknown"
        else:
            result["unrepresented_source_segment_ids"] = [0]
        return result
    monkeypatch.setattr(prepare, key, corrupt)
    with pytest.raises(RuntimeError, match="preparation audit"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train", "--max-bands", "2")
    assert not (tiny_sources["out_dir"] / "p3_prepare_commit.json").exists()
    assert not list(tiny_sources["out_dir"].glob("p3_*_complete.json"))


def _assert_bad_audit_aborts(monkeypatch, sources, function, mutate):
    """Inject only the pure diagnostic result; exercise native prepare/publish."""
    original = getattr(prepare, function)
    calls = []
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        mutate(result)
        calls.append(result)
        return result
    monkeypatch.setattr(prepare, function, corrupt)
    caught = None
    try:
        run_prepare(monkeypatch, sources, "--split", "train")
    except RuntimeError as exc:
        caught = exc
    assert calls, "pure diagnostic seam was not reached"
    assert caught is not None, "malformed diagnostic was published instead of aborting"
    assert "preparation audit" in str(caught)
    directory = sources["out_dir"]
    assert not (directory / "p3_prepare_commit.json").exists()
    assert not list(directory.glob("p3_*_complete.json"))
    assert not (directory / "p3_prepare_report.json").exists()
    assert not (directory / "p3_train_audit.json").exists()


@pytest.mark.parametrize("corruption", [
    "negative_identity", "bool_zero", "straddling_over_raw", "straddling_over_unselected",
    *["float_" + key for key in (
        "raw_band_count", "selected", "window_candidates", "selected_in_window",
        "selected_outside_window", "window_dropped", "selection_candidates",
        "selection_dropped", "capacity_shortfall", "quota_underfill", "straddling_intervals_dropped")],
])
def test_window_audit_counts_are_bounded_integers(monkeypatch, tiny_sources, corruption):
    def mutate(result):
        if corruption == "negative_identity":
            result.update(window_candidates=-1, selected_in_window=-1,
                          selected_outside_window=result["selected"] + 1)
        elif corruption == "bool_zero":
            assert result["window_dropped"] == 0
            result["window_dropped"] = False
        elif corruption == "straddling_over_raw":
            result["straddling_intervals_dropped"] = result["raw_band_count"] + 1
        elif corruption == "straddling_over_unselected":
            result["straddling_intervals_dropped"] = result["raw_band_count"] - result["selected"] + 1
        else:
            key = corruption.removeprefix("float_")
            result[key] = float(result[key])  # Numerically equal is not a count type.
    _assert_bad_audit_aborts(monkeypatch, tiny_sources, "band_window_audit", mutate)


@pytest.mark.parametrize("path", ["raw_full", "raw_selected", "resampled_selected"])
@pytest.mark.parametrize("field,value", [
    ("bare_denominators", None),
    ("scope", None), ("causal_attribution", None), ("frontier_state", None),
    ("gap_eV", "missing"), ("gap_eV", False), ("gap_eV", "0.0"),
    ("n_valid_bands", "float"), ("n_valid_k", "float"),
    ("interval_straddling_bands", -1), ("interval_straddling_bands", "over_bands"),
    ("within_segment_crossing_bands", False), ("within_segment_crossing_bands", 0.0),
    ("within_segment_crossing_bands", "over_bands"),
])
def test_three_path_audits_require_typed_diagnostic_records(monkeypatch, tiny_sources, path, field, value):
    def mutate(result):
        diagnostic = result[path]
        if field == "bare_denominators":
            result[path] = {key: diagnostic[key] for key in ("n_valid_bands", "n_valid_k")}
        elif value == "missing":
            del diagnostic[field]
        elif value == "float":
            diagnostic[field] = float(diagnostic[field])
        elif value == "over_bands":
            diagnostic[field] = diagnostic["n_valid_bands"] + 1
        else:
            diagnostic[field] = value
    _assert_bad_audit_aborts(monkeypatch, tiny_sources, "resampling_audit", mutate)


@pytest.mark.parametrize("path", ["raw_full", "raw_selected", "resampled_selected"])
@pytest.mark.parametrize("corruption", [
    "missing_segments", "empty_segments", "mapping_segments", "non_mapping_row",
    *["missing_" + key for key in (
        "segment_id", "n_points", "source_order_inversions", "minimum_adjacent_separation_eV",
        "slot_dispersion_rms_eV", "rank_dispersion_rms_eV", "start", "stop", "slot_crossing_bands",
        "n_strictly_below_ef", "n_at_or_below_ef", "relabeling_invariant_strict_crossing")],
    "float_start", "bool_segment", "float_points", "negative_crossing", "crossing_over_bands",
    "inversions_over_pairs", "float_inversions", "gap_between_rows", "stop_over_k",
    "wrong_points", "wrong_segment", "occupation_short", "occupation_mapping",
    "occupation_bool", "occupation_float", "occupation_negative", "occupation_over_bands",
    "occupation_reversed_bounds", "integer_crossing_flag", "string_dispersion", "null_dispersion",
    "bool_separation",
])
def test_segment_audit_requires_bounded_rows_and_occupation_arrays(monkeypatch, tiny_sources, path, corruption):
    def mutate(result):
        diagnostic = result[path]
        row = diagnostic["segments"][0]
        if corruption == "missing_segments":
            del diagnostic["segments"]
        elif corruption == "empty_segments":
            diagnostic["segments"] = []
        elif corruption == "mapping_segments":
            diagnostic["segments"] = {"0": row}
        elif corruption == "non_mapping_row":
            diagnostic["segments"][0] = None
        elif corruption.startswith("missing_"):
            del row[corruption.removeprefix("missing_")]
        elif corruption == "float_start":
            row["start"] = float(row["start"])
        elif corruption == "bool_segment":
            row["segment_id"] = False
        elif corruption == "float_points":
            row["n_points"] = float(row["n_points"])
        elif corruption == "negative_crossing":
            row["slot_crossing_bands"] = -1
        elif corruption == "crossing_over_bands":
            row["slot_crossing_bands"] = diagnostic["n_valid_bands"] + 1
        elif corruption == "inversions_over_pairs":
            row["source_order_inversions"] = diagnostic["n_valid_bands"] * row["n_points"]
        elif corruption == "float_inversions":
            row["source_order_inversions"] = float(row["source_order_inversions"])
        elif corruption == "gap_between_rows":
            diagnostic["segments"][1]["start"] += 1
        elif corruption == "stop_over_k":
            row["stop"] = diagnostic["n_valid_k"] + 1
        elif corruption == "wrong_points":
            row["n_points"] += 1
        elif corruption == "wrong_segment":
            row["segment_id"] = result["source_segment_count"]
        elif corruption == "occupation_short":
            row["n_strictly_below_ef"].pop()
        elif corruption == "occupation_mapping":
            row["n_at_or_below_ef"] = {"0": 1, "1": 1}
        elif corruption.startswith("occupation_"):
            row["n_strictly_below_ef"][0] = {
                "occupation_bool": True, "occupation_float": 1.0, "occupation_negative": -1,
                "occupation_over_bands": diagnostic["n_valid_bands"] + 1,
                "occupation_reversed_bounds": row["n_at_or_below_ef"][0] + 1,
            }[corruption]
        elif corruption == "integer_crossing_flag":
            row["relabeling_invariant_strict_crossing"] = 0
        elif corruption == "string_dispersion":
            row["slot_dispersion_rms_eV"] = "0.0"
        elif corruption == "null_dispersion":
            row["rank_dispersion_rms_eV"] = None
        else:
            assert corruption == "bool_separation"
            row["minimum_adjacent_separation_eV"] = False
    _assert_bad_audit_aborts(monkeypatch, tiny_sources, "resampling_audit", mutate)


@pytest.mark.parametrize("corruption", [
    "window_scope", "selected_bool", "selected_float", "spin_indices", "omitted_sides",
    "quota_missing", "quota_bool", "quota_float", "quota_negative", "quota_over_capacity",
    "source_count_float", "target_count_float", "lost_ids_float", "resampling_causal",
])
def test_audit_structural_metadata_cannot_use_numeric_aliases(monkeypatch, tiny_sources, corruption):
    # A clean split with an unrepresented middle source segment.
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        del group["energies"], group["k_distances"]
        group["energies"] = np.array([[1.] * 6, [3.] * 6], np.float32)
        group["k_distances"] = [0, .01, .01, .02, .02, 1]
        group.attrs.update(band_layout="bands_k", spin_channels=1)
    function = "resampling_audit" if corruption in (
        "source_count_float", "target_count_float", "lost_ids_float", "resampling_causal") else "band_window_audit"
    def mutate(result):
        if corruption == "window_scope":
            del result["scope"]
        elif corruption in ("selected_bool", "selected_float"):
            result["selected_indices"][0] = False if corruption == "selected_bool" else 0.0
        elif corruption == "spin_indices":
            result["selected_spin_band_indices"] = [[0, 0]]
        elif corruption == "omitted_sides":
            result["omitted_frontier_sides"] = "occupied"
        elif corruption == "quota_missing":
            result["side_quotas"] = None
        elif corruption.startswith("quota_"):
            result["side_quotas"]["occupied"] = {
                "quota_bool": True, "quota_float": 2.0, "quota_negative": -1,
                "quota_over_capacity": 6,
            }[corruption]
        elif corruption in ("source_count_float", "target_count_float"):
            key = "source_segment_count" if corruption == "source_count_float" else "target_segment_count"
            result[key] = float(result[key])
        elif corruption == "lost_ids_float":
            assert result["unrepresented_source_segment_ids"] == [1]
            result["unrepresented_source_segment_ids"] = [1.0]
        else:
            result["causal_attribution"] = None
    _assert_bad_audit_aborts(monkeypatch, tiny_sources, function, mutate)


@pytest.mark.parametrize("failure", ["post_replace_interrupt", "directory_fsync"])
def test_report_publication_failure_revokes_success_report(monkeypatch, tiny_sources, failure):
    """Both injections follow an actual os.replace of complete report bytes."""
    import stat
    original_replace, original_fsync = prepare.os.replace, prepare.os.fsync
    published = []
    injected = []
    def replace(source, destination):
        original_replace(source, destination)
        if Path(destination).name == "p3_prepare_report.json":
            published.append(json.loads(Path(destination).read_text()))
            if failure == "post_replace_interrupt":
                injected.append("after_actual_replace")
                raise KeyboardInterrupt("synthetic post-report-replace interruption")
    def fsync(fd):
        if failure == "directory_fsync" and published and stat.S_ISDIR(os.fstat(fd).st_mode):
            injected.append("actual_directory_fd_after_replace")
            raise OSError("synthetic report directory fsync failure")
        return original_fsync(fd)
    monkeypatch.setattr(prepare.os, "replace", replace)
    monkeypatch.setattr(prepare.os, "fsync", fsync)
    expected = KeyboardInterrupt if failure == "post_replace_interrupt" else OSError
    with pytest.raises(expected, match="synthetic"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert len(published) == len(injected) == 1
    assert published[0]["status"] == "complete", "fault must follow successful report rename"
    directory = tiny_sources["out_dir"]
    assert not (directory / "p3_prepare_commit.json").exists()
    assert not list(directory.glob("p3_*_complete.json"))
    report = directory / "p3_prepare_report.json"
    assert not report.exists() or json.loads(report.read_text())["status"] != "complete", \
        "report publication exception left a complete success report"
    assert not list(directory.glob("*.tmp"))
    with pytest.raises(ValueError, match="P3 completion"):
        training_reader().load_split(directory / "p3_train.npz", smoke_limit=32)


@pytest.mark.parametrize("case", ["occupied_only", "empty_only", "ef_flat", "disconnected",
                                   "crossing", "touching", "inverted_order", "one_per_spin"])
def test_gate_preserves_native_unknown_false_null_diagnostics(monkeypatch, tiny_sources, case):
    """Positive property witness; no requirement to resolve uncertain physics."""
    rows, state = {
        "occupied_only": ([[-1.] * 4], "unresolved_missing_empty_frontier"),
        "empty_only": ([[1.] * 4], "unresolved_missing_occupied_frontier"),
        "ef_flat": ([[0.] * 4], "unresolved_ef_flat_only"),
        "disconnected": ([[-1., -1., 1., 1.]], "unresolved_between_segments"),
        "crossing": ([[-1., 1., -1., 1.]], "witnessed_within_segment_crossing"),
        "touching": ([[-1., 0., -1., 0.], [0., 1., 0., 1.]], "resolved_touching_frontiers"),
        "inverted_order": ([[1.] * 4, [-1.] * 4], "resolved_positive_gap"),
        "one_per_spin": ([[[-1.] * 4], [[1.] * 4]], "resolved_positive_gap"),
    }[case]
    with h5py.File(tiny_sources["h5"], "r+") as handle:
        group = handle["spin-Si"]
        del group["energies"]
        group["energies"] = np.asarray(rows, np.float32) + 2.
        group.attrs.update(band_layout="spin_band_k" if case == "one_per_spin" else "bands_k",
                           spin_channels=2 if case == "one_per_spin" else 1)
        expected = prepare.build_one(group, {"spin-Si": tiny_sources["record"]}, 5, 5, 5., 2)
    before = {key: sha256_file(tiny_sources[key]) for key in ("h5", "sidecar", "split_npz")}
    run_prepare(monkeypatch, tiny_sources, "--split", "train", "--limit", "1")
    ledger = json.loads((tiny_sources["out_dir"] / "p3_train_audit.json").read_text())
    row = ledger["records"][0]
    assert row["status"] == "audited" and ledger["accepted"] is False
    for key in ("band_window_audit", "resampling_audit"):
        assert row[key] == expected[key]
    diagnostic = row["resampling_audit"]["raw_full"]
    assert diagnostic["frontier_state"] == state
    if state.startswith("unresolved"):
        assert diagnostic["gap_eV"] is None
        assert all(s["relabeling_invariant_strict_crossing"] is False for s in diagnostic["segments"])
    if case == "one_per_spin":
        assert all(s["minimum_adjacent_separation_eV"] is None for s in diagnostic["segments"])
    if case == "inverted_order":
        assert all(s["minimum_adjacent_separation_eV"] < 0 for s in diagnostic["segments"])
    assert all(sha256_file(tiny_sources[key]) == digest for key, digest in before.items())


@pytest.fixture
def partition_sources(tmp_path):
    """One explicit synthetic material: raw K=8, target K=5, two branches."""
    mid = "synthetic-partition-only"
    h5_path = tmp_path / "partition.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group(mid)
        group["energies"] = np.array([[-1.] * 8, [1.] * 8], np.float32)
        group["k_distances"] = [0., .25, .5, .75, .75, .85, .95, 1.]
        group.attrs.update(source="aflow", efermi=0., source_efermi_absolute=6.,
                           source_schema="aflow_bandsdata_v1", band_layout="bands_k",
                           spin_channels=1, source_semantics_evidence="synthetic contract only")
    sidecar = tmp_path / "partition_sidecar.json"
    sidecar.write_text(json.dumps([{"material_id": mid, "lattice": (np.eye(3) * 4).tolist(),
                                  "species_per_atom": ["Si"],
                                  "fractional_coordinates": [[0., 0., 0.]]}]), encoding="utf-8")
    split = tmp_path / "partition_split.npz"
    np.savez(split, material_ids_train=np.array([mid]), groups_train=np.array([1]))
    return {"h5": h5_path, "sidecar": sidecar, "split_npz": split,
            "out_dir": tmp_path / "partition_bad"}


def _assert_shifted_partition_aborts(monkeypatch, sources, paths):
    """Real CLI control before corrupting only the native diagnostic return."""
    import copy
    before = {key: sha256_file(sources[key]) for key in ("h5", "sidecar", "split_npz")}
    control = {**sources, "out_dir": sources["out_dir"].with_name("partition_control")}
    result = subprocess.run([sys.executable, *cli_args(control, "--split", "train", "--limit", "1")],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    (sources["h5"].parent / "partition_control.log").write_text(result.stdout + result.stderr,
                                                               encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((control["out_dir"] / "p3_prepare_report.json").read_text())
    credential = json.loads((control["out_dir"] / "p3_train_complete.json").read_text())
    commit = json.loads((control["out_dir"] / "p3_prepare_commit.json").read_text())
    ledger = json.loads((control["out_dir"] / "p3_train_audit.json").read_text())
    assert report["status"] == credential["status"] == commit["status"] == "complete"
    assert commit["completion_sha256"]["train"] == sha256_file(control["out_dir"] / "p3_train_complete.json")
    assert credential["preparation_audit"]["sha256"] == sha256_file(control["out_dir"] / "p3_train_audit.json")
    assert ledger["counts"] == {"requested": 1, "audited": 1, "excluded": 0, "not_processed": 0}
    for key, points in (("raw_full", [4, 4]), ("raw_selected", [4, 4]),
                        ("resampled_selected", [3, 2])):
        assert [row["n_points"] for row in ledger["records"][0]["resampling_audit"][key]["segments"]] == points
    with np.load(control["out_dir"] / "p3_train.npz", allow_pickle=False) as archive:
        control_arrays = {key: archive[key].copy() for key in archive.files}
    np.testing.assert_array_equal(control_arrays["segment_ids"], [[0, 0, 0, 1, 1]])
    original = prepare.resampling_audit
    seen = []
    def shift(*args, **kwargs):
        diagnostic = original(*args, **kwargs)
        np.testing.assert_array_equal(args[4], [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(args[5], control_arrays["segment_ids"][0])
        np.testing.assert_array_equal(args[2], control_arrays["bands"][0])
        np.testing.assert_array_equal(args[3], control_arrays["band_mask"][0])
        record = {"source_ids": args[4].tolist(), "target_ids": args[5].tolist(),
                  "before": {key: copy.deepcopy(diagnostic[key]["segments"]) for key in paths}}
        for key in paths:
            left, right = diagnostic[key]["segments"]
            left["stop"] += 1
            left["n_points"] += 1
            right["start"] += 1
            right["n_points"] -= 1
            for field in ("n_strictly_below_ef", "n_at_or_below_ef"):
                left[field].append(right[field].pop(0))
        record["after"] = {key: copy.deepcopy(diagnostic[key]["segments"]) for key in paths}
        seen.append(record)
        return diagnostic
    monkeypatch.setattr(prepare, "resampling_audit", shift)
    caught = None
    try:
        run_prepare(monkeypatch, sources, "--split", "train", "--limit", "1")
    except RuntimeError as exc:
        caught = exc
    directory = sources["out_dir"]
    witness = {"synthetic_only": True, "raw_K": 8, "target_K": 5, "materials": 1,
               "paths": list(paths), "control_exit": result.returncode, "seam_calls": seen,
               "source_sha256": before, "code_hashes": credential["code_hashes"],
               "caught": str(caught) if caught is not None else None,
               "report_exists": (directory / "p3_prepare_report.json").exists(),
               "commit_exists": (directory / "p3_prepare_commit.json").exists(),
               "credential_exists": (directory / "p3_train_complete.json").exists()}
    # Retain the observation even when the intended negative assertion goes RED.
    (sources["h5"].parent / "partition_witness.json").write_text(
        json.dumps(witness, indent=2, allow_nan=False), encoding="utf-8")
    assert all(sha256_file(sources[key]) == digest for key, digest in before.items())
    assert len(seen) == 1, "actual native diagnostic seam was not reached exactly once"
    assert caught is not None, "self-consistent shifted partition was published as complete"
    assert "preparation audit" in str(caught) and "segment" in str(caught)
    assert not isinstance(caught, prepare.SampleExclusion)
    assert not any(witness[key] for key in ("report_exists", "commit_exists", "credential_exists"))
    assert not (directory / "p3_train_audit.json").exists()
    assert not (directory / "p3_train.npz").exists()


def test_resampled_partition_binds_actual_segment_ids(monkeypatch, partition_sources):
    _assert_shifted_partition_aborts(monkeypatch, partition_sources, ("resampled_selected",))


def test_raw_selected_partition_binds_actual_source_segment_ids(monkeypatch, partition_sources):
    _assert_shifted_partition_aborts(monkeypatch, partition_sources, ("raw_selected",))


@pytest.mark.parametrize("paths", [("raw_full",), ("raw_full", "raw_selected")])
def test_raw_partition_variants_bind_actual_source_segment_ids(monkeypatch, partition_sources, paths):
    """Shared raw-ID fix regression, not a separately claimed RED/GREEN cycle."""
    _assert_shifted_partition_aborts(monkeypatch, partition_sources, paths)


@pytest.mark.parametrize("target", ["p3_train_complete.json", "p3_prepare_commit.json"])
@pytest.mark.parametrize("failure", ["post_replace_interrupt", "directory_fsync"])
def test_credential_commit_publication_failures_revoke_success(monkeypatch, tiny_sources, target, failure):
    """Retain L2 coverage at real rename/fsync seams; no new publication code."""
    import stat
    original_replace, original_fsync = prepare.os.replace, prepare.os.fsync
    published, injected = [], []
    def replace(source, destination):
        original_replace(source, destination)
        if Path(destination).name == target:
            published.append(json.loads(Path(destination).read_text()))
            if failure == "post_replace_interrupt":
                injected.append("after_actual_replace")
                raise KeyboardInterrupt("synthetic post-publication interruption")
    def fsync(fd):
        if failure == "directory_fsync" and published and stat.S_ISDIR(os.fstat(fd).st_mode):
            injected.append("actual_directory_fd_after_replace")
            raise OSError("synthetic publication directory fsync failure")
        return original_fsync(fd)
    monkeypatch.setattr(prepare.os, "replace", replace)
    monkeypatch.setattr(prepare.os, "fsync", fsync)
    expected = KeyboardInterrupt if failure == "post_replace_interrupt" else OSError
    with pytest.raises(expected, match="synthetic"):
        run_prepare(monkeypatch, tiny_sources, "--split", "train")
    assert len(published) == len(injected) == 1
    assert published[0]["status"] == "complete"
    directory = tiny_sources["out_dir"]
    assert not (directory / "p3_prepare_commit.json").exists()
    assert not list(directory.glob("p3_*_complete.json"))
    assert not (directory / "p3_prepare_report.json").exists()
    assert not list(directory.glob("*.tmp"))
    with pytest.raises(ValueError, match="P3 completion"):
        training_reader().load_split(directory / "p3_train.npz", smoke_limit=32)
