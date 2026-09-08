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
                     "src/data/crystal_graph.py", "src/utils/selection_manifest.py"]
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
