"""CPU-only, synthetic-data tests for the auditable P3 entry point."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p3_training", ROOT / "scripts/train_p3_decoder.py")
p3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p3)


def seal_split(path, *, smoke_only=False):
    """Issue test-only synthetic credentials, never certify production inputs."""
    split = path.stem.removeprefix("p3_")
    credential = {
        "schema": "p3_split_completion", "schema_version": 1,
        "status": "complete", "split": split, "smoke_only": smoke_only,
        "output_npz": {"path": path.name, "sha256": p3.sha256_file(path),
                       "bytes": path.stat().st_size},
        "parameters": {"max_bands": 3, "n_k": 8, "max_atoms": 3, "delta_e": 5.0,
                       "limit": 10 if smoke_only else 0, "split": split},
        "sources": {key: {"path": "synthetic-fixture-only", "sha256": p3.sha256_file(path)}
                    for key in ("h5", "split_npz", "sidecar")},
        "code_hashes": {name: p3.sha256_file(ROOT / name) for name in (
            "scripts/prepare_p3_multiband.py", "src/data/multiband.py",
            "src/data/crystal_graph.py", "src/utils/selection_manifest.py")},
    }
    certificate = path.with_name(f"p3_{split}_complete.json")
    certificate.write_text(json.dumps(credential), encoding="utf-8")
    commit_path = path.with_name("p3_prepare_commit.json")
    commit = json.loads(commit_path.read_text()) if commit_path.exists() else {
        "schema": "p3_prepare_commit", "schema_version": 1, "status": "complete",
        "completion_sha256": {},
    }
    commit["completion_sha256"][split] = p3.sha256_file(certificate)
    commit_path.write_text(json.dumps(commit), encoding="utf-8")


def write_split(path, n=14, id_prefix="synthetic", *, smoke_only=False):
    """Small graphs/k grids; never use production data as a test fixture."""
    rng = np.random.default_rng(12)
    mask = np.ones((n, 3), dtype=bool)
    mask[::2, -1] = False
    valid = np.ones(n, dtype=bool)
    valid[1] = False
    np.savez_compressed(
        path,
        atom_features=rng.uniform(0.1, 1, (n, 3, 110)).astype(np.float32),
        neighbor_list=np.zeros((n, 3, 12), dtype=np.int32),
        neighbor_dist=np.ones((n, 3, 12), dtype=np.float32),
        bands=rng.normal(size=(n, 3, 8)).astype(np.float32),
        band_mask=mask,
        valid=valid,
        k_axis=np.linspace(0, 1, 8, dtype=np.float32),
        material_ids=np.array([f"{id_prefix}-{i}" for i in range(n)]),
        # Synthetic outer spacegroups are disjoint from the synthetic train pool.
        groups=np.repeat(np.arange(1, n // 2 + 1), 2) + (100 if path.name == "p3_test.npz" else 0),
        segment_ids=np.tile(np.repeat([0, 1], 4), (n, 1)).astype(np.int32),
    )
    seal_split(path, smoke_only=smoke_only)


@pytest.mark.parametrize("problem", [
    "missing", "partial", "failed", "schema", "wrong_split", "tampered",
    "npz_hash", "npz_bytes", "npz_changed", "redirect", "sources", "builder",
    "missing_commit", "partial_commit", "failed_commit", "schema_commit",
])
def test_completion_gate_rejects_invalid_authorization_before_npz_load(tmp_path, monkeypatch, problem):
    path = tmp_path / "p3_train.npz"
    write_split(path)
    certificate = tmp_path / "p3_train_complete.json"
    commit_path = tmp_path / "p3_prepare_commit.json"
    credential = json.loads(certificate.read_text())
    commit = json.loads(commit_path.read_text())
    if problem == "missing":
        certificate.unlink()
    elif problem == "partial":
        certificate.write_text("{partial")
    elif problem == "tampered":
        certificate.write_text(certificate.read_text() + " ")
    elif problem == "npz_changed":
        with path.open("ab") as handle:
            handle.write(b"still a readable zip, no longer certified")
    elif problem.endswith("commit"):
        if problem == "missing_commit":
            commit_path.unlink()
        elif problem == "partial_commit":
            commit_path.write_text("{partial")
        else:
            commit["status" if problem == "failed_commit" else "schema_version"] = "failed"
            commit_path.write_text(json.dumps(commit))
    else:
        if problem == "failed":
            credential["status"] = "failed"
        elif problem == "schema":
            credential["schema_version"] = 99
        elif problem == "wrong_split":
            credential["split"] = "test"
        elif problem == "npz_hash":
            credential["output_npz"]["sha256"] = "0" * 64
        elif problem == "npz_bytes":
            credential["output_npz"]["bytes"] += 1
        elif problem == "redirect":
            credential["output_npz"]["path"] = "../outer/p3_train.npz"
        else:
            del credential["sources" if problem == "sources" else "code_hashes"]
        certificate.write_text(json.dumps(credential))
        # Even internally hash-consistent but failed/malformed credentials fail.
        commit["completion_sha256"]["train"] = p3.sha256_file(certificate)
        commit_path.write_text(json.dumps(commit))

    def forbidden(*args, **kwargs):
        pytest.fail("NPZ loaded before completion authorization")

    monkeypatch.setattr(p3.np, "load", forbidden)
    with pytest.raises(ValueError, match="P3 completion"):
        p3.load_split(path)


def test_inner_split_is_deterministic_group_disjoint_and_keeps_source_ids(tmp_path):
    path = tmp_path / "p3_train.npz"
    write_split(path)
    data, k_axis = p3.load_split(path)
    assert all(isinstance(value, np.ndarray) for value in data.values())
    assert k_axis.shape == (8,)
    fit, val, manifest = p3.inner_split(data, seed=42)
    fit_again, val_again, same_manifest = p3.inner_split(data, seed=42)
    np.testing.assert_array_equal(fit, fit_again)
    np.testing.assert_array_equal(val, val_again)
    assert manifest == same_manifest
    assert not set(data["groups"][fit]) & set(data["groups"][val])
    assert manifest["group_overlap"] == 0
    assert manifest["train_size"] == 0.85
    assert set(np.concatenate([fit, val])) == set(range(13))
    for label, positions in (("fit", fit), ("validation", val)):
        assert manifest[label]["material_ids"] == data["material_ids"][positions].tolist()
        assert manifest[label]["indices"] == data["source_indices"][positions].tolist()
        assert 1 not in manifest[label]["indices"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_smoke_preparation_requires_explicit_smoke_limit_before_npz_load(tmp_path, monkeypatch, split):
    path = tmp_path / f"p3_{split}.npz"
    write_split(path, smoke_only=True)
    data, _ = p3.load_split(path, smoke_limit=4)
    assert len(data["bands"]) == 4

    def forbidden(*args, **kwargs):
        pytest.fail("smoke input reached formal NPZ load")

    monkeypatch.setattr(p3.np, "load", forbidden)
    with pytest.raises(ValueError, match="smoke.*--smoke-limit"):
        p3.load_split(path, smoke_limit=0)


@pytest.mark.parametrize("problem", ["missing_smoke", "string_smoke", "false_smoke",
                                     "true_smoke", "missing_parameters", "negative_limit"])
def test_completion_rejects_smoke_mislabel_even_with_consistent_hash(tmp_path, monkeypatch, problem):
    path = tmp_path / "p3_train.npz"
    write_split(path, smoke_only=True)
    certificate = tmp_path / "p3_train_complete.json"
    credential = json.loads(certificate.read_text())
    if problem == "missing_smoke":
        del credential["smoke_only"]
    elif problem == "string_smoke":
        credential["smoke_only"] = "false"
    elif problem == "false_smoke":
        credential["smoke_only"] = False  # --limit remains positive
    elif problem == "true_smoke":
        credential["parameters"]["limit"] = 0
    elif problem == "missing_parameters":
        del credential["parameters"]
    else:
        credential["parameters"]["limit"] = -1
    certificate.write_text(json.dumps(credential))
    commit_path = tmp_path / "p3_prepare_commit.json"
    commit = json.loads(commit_path.read_text())
    commit["completion_sha256"]["train"] = p3.sha256_file(certificate)
    commit_path.write_text(json.dumps(commit))

    def forbidden(*args, **kwargs):
        pytest.fail("mislabelled preparation reached NPZ load")

    monkeypatch.setattr(p3.np, "load", forbidden)
    with pytest.raises(ValueError, match="P3 completion"):
        p3.load_split(path, smoke_limit=4)


def test_loss_aggregation_weights_all_valid_band_k_elements():
    pred = np.zeros((3, 3, 2), dtype=np.float32)
    target = np.array([[[1, 1]] * 3, [[3, 3]] * 3, [[10, 10]] * 3], dtype=np.float32)
    mask = np.array([[True, False, True], [True, True, True], [True, False, False]])
    aggregate = p3.LossAccumulator()
    for indices in (slice(0, 2), slice(2, 3)):
        loss = float(p3.sorted_masked_mae(pred[indices], target[indices], mask[indices]))
        aggregate.add(loss, mask[indices], n_k=2)
    expected = float(p3.sorted_masked_mae(pred, target, mask))
    assert aggregate.result() == pytest.approx(expected)
    assert aggregate.elements == int(mask.sum()) * 2
    assert aggregate.result() != pytest.approx(np.mean([2.2, 10.0]))


@pytest.mark.parametrize("extra, error", [([], RuntimeError), (["--allow-cpu-smoke"], ValueError)])
def test_cpu_is_rejected_without_explicit_bounded_smoke(extra, error):
    args = p3.parse_args(["--data", "unused", "--out", "unused", *extra])
    with pytest.raises(error, match="GPU|smoke-limit"):
        p3.select_device(args)


def test_explicit_bounded_smoke_uses_cpu_and_formal_defaults_are_fixed():
    args = p3.parse_args([
        "--data", "unused", "--out", "unused", "--allow-cpu-smoke", "--smoke-limit", "6"
    ])
    assert p3.select_device(args) == "/CPU:0"
    assert (args.stage, args.epochs, args.batch_size, args.d_model) == ("train", 180, 32, 128)
    assert (args.lr, args.attention_layers, args.attention_heads, args.seed) == (1e-3, 0, 4, 42)
    assert (args.patience, args.warmup_epochs, args.min_delta) == (20, 5, 1e-4)


def smoke_cli(data, out):
    return [
        "--data", str(data), "--out", str(out), "--allow-cpu-smoke", "--smoke-limit", "10",
        "--epochs", "3", "--batch-size", "2", "--d-model", "8",
        "--warmup-epochs", "1", "--patience", "1", "--min-delta", "1000",
    ]


def test_train_only_smoke_freezes_selection_and_reloads_real_weights(tmp_path, monkeypatch):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    train_path = data_dir / "p3_train.npz"
    write_split(train_path)
    forbidden_metadata = {"p3_prepare_report.json", "p3_test_complete.json", "p3_test.npz"}
    for name in forbidden_metadata:
        (data_dir / name).write_bytes(b"outer/global poison; train must not read")
    real_open = Path.open

    def private_open(path, *args, **kwargs):
        assert path.name not in forbidden_metadata, "outer/global metadata read during training"
        return real_open(path, *args, **kwargs)
    real_load = np.load
    opened = []

    def train_only_load(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        assert Path(path).resolve() == train_path.resolve(), "outer access during training"
        return real_load(path, *args, **kwargs)

    with monkeypatch.context() as isolated:
        isolated.setattr(p3.np, "load", train_only_load)
        isolated.setattr(Path, "open", private_open)
        result = p3.main(smoke_cli(data_dir, out))
    assert opened == [train_path.resolve()]
    assert result["mode"] == "smoke"
    manifest = p3.validate_inner_selection_manifest(str(out))
    assert manifest["outer_test_accessed"] is False
    assert manifest["monitor"] == "val_loss"
    config = json.loads((out / "model_config.json").read_text())
    assert config["mode"] == "smoke"
    assert config["data"]["sha256"] == p3.sha256_file(train_path)
    assert config["preparation"] == json.loads((data_dir / "p3_train_complete.json").read_text())
    assert config["model"]["attention_layers"] == 0
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2  # patience=1 + deliberately unattainable min_delta
    assert manifest["stopped_early"] is True
    assert manifest["best_val_loss"] == min(row["val_loss"] for row in history)
    assert history[0]["learning_rate"] == pytest.approx(1e-3)
    split = json.loads((out / "inner_split.json").read_text())
    data, k_axis = p3.load_split(train_path, smoke_limit=10)
    fit, val, expected_split = p3.inner_split(data, seed=42)
    assert split == expected_split
    for row in history:
        assert row["train_elements"] == int(data["band_mask"][fit].sum()) * len(k_axis)
        assert row["val_elements"] == int(data["band_mask"][val].sum()) * len(k_axis)
    assert manifest["reload_verification"]["predictions_match"] is True
    assert manifest["reload_verification"]["max_abs_difference"] <= 1e-6
    assert manifest["checkpoint_verification"]["optimizer_restored"] is True
    assert manifest["checkpoint_verification"]["epoch"] == len(history)
    assert manifest["checkpoint_verification"]["iterations"] > 0
    assert manifest["runtime"]["device"] == "/CPU:0"
    assert manifest["runtime"]["data_residency"] == "host_numpy"
    assert manifest["runtime"]["preflight"]["finite"] is True
    assert manifest["runtime"]["preflight"]["gradient_count"] > 0
    assert manifest["runtime"]["preflight"]["loss_device"].endswith("/device:CPU:0")
    compiled = manifest["runtime"]["compiled_train"]
    assert compiled["gradient_count"] > 0
    assert all(device.endswith("/device:CPU:0") for device in compiled["gradient_devices"])
    assert compiled["prediction_device"].endswith("/device:CPU:0")
    assert compiled["loss_device"].endswith("/device:CPU:0")
    assert manifest["runtime"]["max_batch_size"] <= 2
    assert not list(out.glob("*.keras"))
    assert not list(out.glob("*.tmp*"))
    graph = p3.batch_tensors(data, val, "/CPU:0")
    model = p3.build_model(config, graph, "/CPU:0")
    model.load_weights(str(out / "accepted.weights.h5"))
    prediction = model(p3.model_graph(graph), p3.tf.constant(k_axis), training=False)
    val_loss = float(p3.sorted_masked_mae(prediction, graph["bands"], graph["band_mask"]))
    assert val_loss == pytest.approx(manifest["best_val_loss"], abs=1e-6)


@pytest.fixture(scope="module")
def native_p3_selection(tmp_path_factory):
    """Real train/freeze/reload, synthetic train only; no prepare dependency."""
    directory = tmp_path_factory.mktemp("native-p3-format")
    data_dir, out = directory / "data", directory / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz", smoke_only=True)
    p3.main([*smoke_cli(data_dir, out), "--epochs", "1"])
    return out


def test_native_p3_v1_passes_bridge_but_not_shared_formal(native_p3_selection):
    from src.utils.selection_manifest import validate_inner_selection_manifest
    manifest = p3.validate_inner_selection_manifest(str(native_p3_selection))
    assert manifest["schema_version"] == 1
    assert manifest["mode"] == "smoke"
    assert manifest["reload_verification"]["predictions_match"] is True
    # P3's explicit legacy bridge must never change the shared default.
    with pytest.raises(RuntimeError, match="legacy"):
        validate_inner_selection_manifest(str(native_p3_selection))


@pytest.fixture
def p3_selection_copy(tmp_path, native_p3_selection):
    path = tmp_path / "inner_selection_manifest.json"
    path.write_bytes((native_p3_selection / path.name).read_bytes())
    # A genuine accepted baseline, not a hand-written P3-looking envelope.
    manifest = p3.validate_inner_selection_manifest(str(tmp_path))
    return manifest, path


@pytest.mark.parametrize("mode", ["formal", "smoke"])
@pytest.mark.parametrize("empty_records", [False, True])
def test_p3_rejects_p0_v1_with_valid_p3_tags(tmp_path, mode, empty_records):
    fixture_path = Path(__file__).with_name("test_p0_supervision_contracts.py")
    spec = importlib.util.spec_from_file_location("p0_tagged_fixture", fixture_path)
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    manifest = fixture_module.manifest_fixture(tmp_path, schema=1)
    manifest.update(experimental=True, mode=mode)
    if empty_records:
        manifest.update(model_config={}, inner_split={}, history={}, runtime={})
    path = tmp_path / "inner_selection_manifest.json"
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="P3"):
        p3.validate_inner_selection_manifest(str(tmp_path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["model_config", "inner_split", "history", "runtime"])
@pytest.mark.parametrize("problem", ["missing", "empty", "null", "list", "string", "bool", "number"])
def test_p3_requires_nonempty_own_records(p3_selection_copy, field, problem):
    manifest, path = p3_selection_copy
    if problem == "missing":
        del manifest[field]
    else:
        manifest[field] = {"empty": {}, "null": None, "list": ["not-a-record"],
                           "string": "not-a-record", "bool": True, "number": 1}[problem]
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match=rf"P3.*{field}"):
        p3.validate_inner_selection_manifest(str(path.parent))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["model_config", "inner_split", "history"])
@pytest.mark.parametrize("member,value", [
    ("path", None), ("path", ""), ("path", "   "), ("path", 7), ("path", []),
    ("sha256", None), ("sha256", ""), ("sha256", 7), ("sha256", []),
    ("sha256", "g" * 64), ("sha256", "a" * 63),
    ("bytes", None), ("bytes", "7"), ("bytes", True), ("bytes", 7.0),
    ("bytes", 0), ("bytes", -1),
])
def test_p3_requires_typed_artifact_records(p3_selection_copy, field, member, value):
    manifest, path = p3_selection_copy
    if value is None:
        del manifest[field][member]
    else:
        manifest[field][member] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=rf"P3.*{field}"):
        p3.validate_inner_selection_manifest(str(path.parent))


@pytest.mark.parametrize("member,value", [
    *[(key, None) for key in ("device", "data_residency", "max_batch_size", "tensorflow",
                              "optimizer_clipnorm", "preflight", "compiled_train")],
    ("device", []), ("device", ""), ("device", 1), ("device", "/TPU:0"),
    ("data_residency", []), ("data_residency", "device"),
    ("max_batch_size", True), ("max_batch_size", "2"), ("max_batch_size", 2.0),
    ("max_batch_size", 0), ("max_batch_size", -1),
    ("tensorflow", []), ("tensorflow", ""), ("tensorflow", 1),
    ("optimizer_clipnorm", True), ("optimizer_clipnorm", "1"),
    ("optimizer_clipnorm", 0), ("optimizer_clipnorm", float("nan")),
    ("optimizer_clipnorm", float("inf")),
    ("preflight", {}), ("preflight", []), ("compiled_train", {}), ("compiled_train", []),
])
def test_p3_requires_typed_runtime_record(p3_selection_copy, member, value):
    manifest, path = p3_selection_copy
    if value is None:
        del manifest["runtime"][member]
    else:
        manifest["runtime"][member] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="P3.*runtime"):
        p3.validate_inner_selection_manifest(str(path.parent))


@pytest.mark.parametrize("stage", ["preflight", "compiled_train"])
@pytest.mark.parametrize("member,value", [
    *[(key, None) for key in ("prediction_device", "loss_device", "gradient_devices", "gradient_count", "finite")],
    ("prediction_device", []), ("prediction_device", ""), ("loss_device", 1),
    ("gradient_devices", []), ("gradient_devices", "CPU"), ("gradient_devices", [1]),
    ("gradient_devices", [""]), ("gradient_count", True), ("gradient_count", "1"),
    ("gradient_count", 0), ("gradient_count", 1.0), ("finite", False), ("finite", 1),
])
def test_p3_requires_typed_runtime_step_records(p3_selection_copy, stage, member, value):
    manifest, path = p3_selection_copy
    if stage == "compiled_train" and member == "finite":
        member = "finite_checked_in_graph"
    if value is None:
        del manifest["runtime"][stage][member]
    else:
        manifest["runtime"][stage][member] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=rf"P3.*runtime.*{stage}"):
        p3.validate_inner_selection_manifest(str(path.parent))


@pytest.mark.parametrize("value", [[], {}])
def test_p3_rejects_container_mode_as_a_format_error(p3_selection_copy, value):
    manifest, path = p3_selection_copy
    manifest["mode"] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="P3"):
        p3.validate_inner_selection_manifest(str(path.parent))


@pytest.mark.parametrize("case,schema", [
    ("classifier", 1), ("wrong_schema", 2), ("bool_schema", True), ("wrong_mode", 1),
])
def test_p3_validator_rejects_classifier_manifests(tmp_path, case, schema):
    # Reuse the genuine P0 gate fixture; these are not trained model artifacts.
    import importlib.util
    fixture_path = Path(__file__).with_name("test_p0_supervision_contracts.py")
    spec = importlib.util.spec_from_file_location("p0_selection_fixture", fixture_path)
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    manifest = fixture_module.manifest_fixture(tmp_path, schema=schema)
    if case != "classifier":
        # A classifier schema cannot become P3 merely by adding P3-looking keys.
        mode = "unrecognized" if case == "wrong_mode" else "formal"
        manifest.update(experimental=True, mode=mode, model_config={},
                        inner_split={}, history={}, runtime={})
        (tmp_path / "inner_selection_manifest.json").write_text(json.dumps(manifest))
    before = (tmp_path / "inner_selection_manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="P3"):
        p3.validate_inner_selection_manifest(str(tmp_path))
    assert (tmp_path / "inner_selection_manifest.json").read_bytes() == before


def test_revoked_completion_during_training_cannot_freeze_selection(tmp_path, monkeypatch):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    real_weights = p3.atomic_weights

    def revoke_after_weights(model, path):
        real_weights(model, path)
        if Path(path).name == "accepted.weights.h5":
            (data_dir / "p3_prepare_commit.json").unlink()

    monkeypatch.setattr(p3, "atomic_weights", revoke_after_weights)
    with pytest.raises(ValueError, match="P3 completion"):
        p3.main([*smoke_cli(data_dir, out), "--epochs", "1"])
    assert not (out / "inner_selection_manifest.json").exists()


def test_nonempty_output_fails_before_any_data_access(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="non-empty|Non-empty"):
        p3.main(smoke_cli(tmp_path / "no-data", out))
    assert sentinel.read_text() == "do not overwrite"
    assert list(out.iterdir()) == [sentinel]


def test_gpu_missing_fails_before_creating_run_or_reading_data(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="GPU"):
        p3.main(["--data", str(tmp_path / "no-data"), "--out", str(out)])
    assert not out.exists()


def test_evaluation_before_selection_fails_without_touching_outer(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("outer opened before frozen selection")

    monkeypatch.setattr(p3.np, "load", forbidden)
    with pytest.raises(FileNotFoundError, match="inner selection manifest"):
        p3.main([*smoke_cli(tmp_path, tmp_path / "missing-run"), "--stage", "evaluate"])


def test_evaluate_uses_frozen_config_real_predictions_and_unique_outputs(tmp_path, monkeypatch):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    p3.main(smoke_cli(data_dir, out))
    # The outer file does not even exist until selection is complete.
    write_split(data_dir / "p3_test.npz", n=6, id_prefix="outer")
    frozen_paths = [out / f"{name}.weights.h5" for name in ("best", "last", "accepted")]
    frozen_paths += [out / "inner_selection_manifest.json", out / "model_config.json", out / "history.json"]
    before = {str(path): p3.sha256_file(path) for path in frozen_paths}
    validated = []
    real_validate, real_load = p3.validate_inner_selection_manifest, np.load

    def validate(path):
        result = real_validate(path)
        validated.append(True)
        return result

    def outer_only(path, *args, **kwargs):
        assert validated, "selection must be verified before reading outer"
        assert Path(path).name == "p3_test.npz"
        return real_load(path, *args, **kwargs)

    with monkeypatch.context() as isolated:
        isolated.setattr(p3, "validate_inner_selection_manifest", validate)
        isolated.setattr(p3.np, "load", outer_only)
        first = p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate", "--d-model", "64"])
        second = p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate"])
    assert first["evaluation_dir"] != second["evaluation_dir"]
    assert first["mode"] == "smoke"
    assert first["outer_preparation"] == json.loads((data_dir / "p3_test_complete.json").read_text())
    assert first["outer_split_audit"] == {
        "scope": "frozen_inner_pool_vs_all_outer_rows", "id_overlap": 0, "group_overlap": 0,
    }
    assert first["runtime"]["prediction_device"].endswith("/device:CPU:0")
    assert first["metrics"]["n_samples"] == 5
    assert "band_mae_per_k_ot" in first["metrics"]
    assert "gap_mae" not in first["metrics"]
    evaluation_dir = Path(first["evaluation_dir"])
    assert json.loads((evaluation_dir / "evaluation.json").read_text()) == first
    with np.load(evaluation_dir / "predictions.npz", allow_pickle=False) as archive:
        pred, ids, mask = archive["pred"], archive["material_ids"], archive["mask"]
        assert pred.shape == (5, 3, 8)
        assert ids.tolist() == [f"outer-{i}" for i in [0, 2, 3, 4, 5]]
        assert mask.shape == (5, 3)
        assert np.isfinite(pred).all()
        assert not np.all(pred == 0)
    outer, _ = p3.load_split(data_dir / "p3_test.npz")
    from src.evaluation.multiband_metrics import evaluate_multiband
    assert first["metrics"] == evaluate_multiband(pred, outer["bands"], mask, segment_ids=outer["segment_ids"])
    assert {str(path): p3.sha256_file(path) for path in frozen_paths} == before


@pytest.mark.parametrize("part,field,row", [
    ("fit", "material_ids", 0), ("validation", "material_ids", 0),
    ("fit", "groups", 0), ("validation", "groups", 0),
    ("fit", "material_ids", 5), ("validation", "groups", 1),
])
def test_evaluation_rejects_overlap_with_entire_frozen_pool_before_inference(tmp_path, monkeypatch, part, field, row):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    p3.main([*smoke_cli(data_dir, out), "--epochs", "1"])
    inner = json.loads((out / "inner_split.json").read_text())
    path = data_dir / "p3_test.npz"
    write_split(path, n=6, id_prefix="outer")
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["material_ids"] = arrays["material_ids"].astype("U64")
    arrays[field][row] = inner[part][field][0]
    np.savez_compressed(path, **arrays)
    seal_split(path)

    def forbidden(*args, **kwargs):
        pytest.fail("model inference reached before outer overlap rejection")

    monkeypatch.setattr(p3, "build_model", forbidden)
    # Row 5 is outside the smoke prefix; row 1 is invalid. Neither hides overlap.
    with pytest.raises(ValueError, match="[Oo]uter.*overlap"):
        p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate", "--smoke-limit", "2"])
    assert not (out / "evaluations").exists()


def test_revoked_outer_completion_during_evaluation_cannot_publish(tmp_path, monkeypatch):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    p3.main([*smoke_cli(data_dir, out), "--epochs", "1"])
    write_split(data_dir / "p3_test.npz", n=6, id_prefix="outer")
    from src.evaluation import multiband_metrics
    real_metrics = multiband_metrics.evaluate_multiband

    def revoke_after_metrics(*args, **kwargs):
        result = real_metrics(*args, **kwargs)
        (data_dir / "p3_test_complete.json").unlink()
        return result

    monkeypatch.setattr(multiband_metrics, "evaluate_multiband", revoke_after_metrics)
    with pytest.raises(ValueError, match="P3 completion"):
        p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate"])
    assert not (out / "evaluations").exists()


@pytest.mark.parametrize("kind", ["cpu_fallback", "nan_loss", "nan_gradient", "missing_gradient"])
def test_critical_forward_backward_guard_rejects_cpu_or_invalid_gradients(kind):
    tf = p3.tf
    with tf.device("/CPU:0"):
        variable = tf.Variable(2.0)
        with tf.GradientTape() as tape:
            pred = variable * tf.ones((1, 2, 3))
            loss = tf.reduce_mean(pred)
        grads = [tape.gradient(loss, variable)]
        if kind == "nan_loss":
            loss = tf.constant(float("nan"))
        elif kind == "nan_gradient":
            grads = [tf.constant(float("nan"))]
        elif kind == "missing_gradient":
            grads = [None]
    error = tf.errors.InvalidArgumentError if kind.startswith("nan") else RuntimeError
    with pytest.raises(error, match="GPU|gradient|loss"):
        p3.check_training_tensors(pred, loss, grads, require_gpu=kind == "cpu_fallback")


@pytest.mark.parametrize("target", ["model_config.json", "inner_split.json", "history.json", "p3_train.npz", "p3_train_complete.json", "accepted.weights.h5"])
def test_evaluate_rejects_changed_frozen_inputs_before_outer_access(tmp_path, monkeypatch, target):
    data_dir, out = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    p3.main([*smoke_cli(data_dir, out), "--epochs", "1"])
    write_split(data_dir / "p3_test.npz", n=6, id_prefix="outer")
    changed = (data_dir if target.startswith("p3_train") else out) / target
    with changed.open("ab") as handle:
        handle.write(b"\n")  # still valid JSON/NPZ, but no longer frozen bytes

    def forbidden(*args, **kwargs):
        pytest.fail("outer data reached after a frozen-input hash mismatch")

    monkeypatch.setattr(p3, "load_split", forbidden)
    with pytest.raises((RuntimeError, ValueError), match="hash|bytes|size"):
        p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate"])
    assert not (out / "evaluations").exists()


@pytest.mark.parametrize("problem", ["nan_target", "missing_segment_ids", "all_masked", "duplicate_ids", "bad_groups", "bad_neighbors", "bad_shape"])
def test_loader_rejects_untrustworthy_training_records(tmp_path, problem):
    path = tmp_path / "p3_train.npz"
    write_split(path)
    with np.load(path, allow_pickle=False) as archive:
        values = dict(archive)
    if problem == "nan_target":
        values["bands"][0, 0, 0] = np.nan
    elif problem == "missing_segment_ids":
        del values["segment_ids"]
    elif problem == "all_masked":
        values["band_mask"][0] = False
    elif problem == "duplicate_ids":
        values["material_ids"][0] = values["material_ids"][2]
    elif problem == "bad_groups":
        values["groups"][0] = -1
    elif problem == "bad_neighbors":
        values["neighbor_list"][0, 0, 0] = 50
    elif problem == "bad_shape":
        values["segment_ids"] = values["segment_ids"][:, :-1]
    np.savez_compressed(path, **values)
    seal_split(path)
    with pytest.raises(ValueError):
        p3.load_split(path)


@pytest.mark.parametrize("field", [*p3.BATCH_DTYPES, "material_ids", "groups"])
@pytest.mark.parametrize("shape", ["long", "short", "scalar"])
def test_raw_sample_axis_matches_valid_before_filtering(tmp_path, field, shape):
    path = tmp_path / "p3_train.npz"
    write_split(path, n=4)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: value if key == "k_axis" else value[:2] for key, value in dict(archive).items()}
    # valid=[True,False] makes the selected index legal even for a short axis.
    if shape == "long":
        values[field] = np.concatenate([values[field], values[field]], axis=0)
    elif shape == "short":
        values[field] = values[field][:1]
    else:
        values[field] = np.array(0)
    np.savez_compressed(path, **values)
    seal_split(path)
    with pytest.raises(ValueError, match=rf"{field}.*sample axis"):
        p3.load_split(path, smoke_limit=1)


@pytest.mark.parametrize("extra", [
    ["--epochs", "0"], ["--lr", "nan"], ["--smoke-limit", "-1"],
    ["--batch-size", "0"], ["--attention-heads", "0"], ["--d-model", "7"],
    ["--warmup-epochs", "-1"], ["--min-delta", "nan"], ["--seed", "-1"],
])
def test_invalid_cli_values_fail_before_any_output_or_data_access(tmp_path, extra):
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        p3.main([*smoke_cli(tmp_path / "missing-data", out), *extra])
    assert not out.exists()


def test_two_layer_attention_smoke_is_explicitly_experimental(tmp_path):
    data_dir, out = tmp_path / "data", tmp_path / "attention-run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    result = p3.main([*smoke_cli(data_dir, out), "--epochs", "1", "--attention-layers", "2"])
    config = json.loads((out / "model_config.json").read_text())
    assert config["model"]["attention_layers"] == 2
    assert config["model"]["attention_heads"] == 4
    assert config["experimental"] is True
    assert result["experimental"] is True
    assert result["mode"] == "smoke"
    assert result["reload_verification"]["predictions_match"] is True
    assert result["checkpoint_verification"]["optimizer_restored"] is True
    write_split(data_dir / "p3_test.npz", n=6, id_prefix="outer")
    report = p3.main([*smoke_cli(data_dir, out), "--stage", "evaluate"])
    assert report["experimental"] is True
    assert report["mode"] == "smoke"


def test_compiled_nan_gradient_does_not_advance_optimizer():
    tf = p3.tf

    @tf.custom_gradient
    def broken_backward(value):
        return value, lambda upstream: tf.fill(tf.shape(upstream), tf.constant(float("nan")))

    class ScalarDecoder(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.scale = self.add_weight(shape=(), initializer="ones")

        def call(self, graph, k_axis, training=False):
            return tf.ones((1, 2, 3)) * broken_backward(tf.convert_to_tensor(self.scale))

    with tf.device("/CPU:0"):
        batch = {
            "atom_features": tf.ones((1, 1, 110)), "neighbor_list": tf.zeros((1, 1, 12), tf.int32),
            "neighbor_dist": tf.ones((1, 1, 12)), "segment_ids": tf.zeros((1, 3), tf.int32),
            "bands": tf.zeros((1, 2, 3)), "band_mask": tf.ones((1, 2), tf.bool),
        }
        model = ScalarDecoder()
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        step, _ = p3.make_steps(model, optimizer, tf.linspace(0.0, 1.0, 3), "/CPU:0")
        with pytest.raises(tf.errors.InvalidArgumentError, match="gradient"):
            step(batch)
    assert int(optimizer.iterations.numpy()) == 0
    assert float(model.scale.numpy()) == 1.0


def test_warmup_cosine_endpoints_and_atomic_history(tmp_path):
    rates = [p3.epoch_learning_rate(epoch, 5, 1.0, 2) for epoch in range(5)]
    expected = np.concatenate([np.arange(1, 3) / 2, (1 + np.cos(np.linspace(0, np.pi, 3))) / 2])
    np.testing.assert_allclose(rates, expected)
    path = tmp_path / "history.json"
    p3.atomic_json(path, [{"epoch": 1, "loss": 1.0}])
    before = path.read_bytes()
    with pytest.raises(ValueError):
        p3.atomic_json(path, [{"epoch": 2, "loss": float("nan")}])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_train_evaluate_in_separate_cpu_processes_outside_repo(tmp_path):
    data_dir, out = tmp_path / "data", tmp_path / "cli-run"
    data_dir.mkdir()
    write_split(data_dir / "p3_train.npz")
    command = [sys.executable, str(ROOT / "scripts/train_p3_decoder.py"),
               *smoke_cli(data_dir, out), "--epochs", "1", "--smoke-limit", "6"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    trained = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    manifest = json.loads((out / "inner_selection_manifest.json").read_text())
    write_split(data_dir / "p3_test.npz", n=6, id_prefix="outer")
    evaluated = subprocess.run([*command, "--stage", "evaluate"], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=120)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    reports = list((out / "evaluations").glob("*/evaluation.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["metrics"]["n_samples"] == 5
    assert report["mode"] == "smoke"
    assert manifest["reload_verification"]["predictions_match"] is True
    print(trained.stdout.strip())
    print(evaluated.stdout.strip())
    print(json.dumps({"train_exit": trained.returncode, "evaluate_exit": evaluated.returncode,
                      "fresh_reload_max_abs_difference": manifest["reload_verification"]["max_abs_difference"],
                      "outer_smoke_samples": report["metrics"]["n_samples"]}))
