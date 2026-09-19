"""P0 acceptance regressions; fixtures are synthetic, not scientific evidence."""
import inspect
import json

import numpy as np
import pytest

from src.data import benchmark_splits as bs


def samples_fixture():
    return [dict(material_id=f"m{i}", spacegroup_number=i + 1,
                 species=[e], formula_pretty=e, prototype=f"p{i}",
                 source="AFLOW", dft_functional="PBE",
                 aurl=f"host:AFLOWDATA/LIB{i % 2 + 1}_WEB/x",
                 aflowlib_date=f"202001{i + 1:02d}_00:00:00_GMT+0")
            for i, e in enumerate(["Si", "O", "Na", "Cl", "Fe", "C"])]


def canonical_fixture(samples):
    return {"train_material_ids": [s["material_id"] for s in samples[2:]][::-1],
            "test_material_ids": [samples[1]["material_id"], samples[0]["material_id"]]}


def test_canonical_split_reuses_exact_ids_in_exact_order():
    assert "canonical_split" in inspect.signature(bs.build_all_splits).parameters, "missing canonical ID reuse API"
    samples = samples_fixture()
    canonical = canonical_fixture(samples)
    splits = bs.build_all_splits(samples, np.arange(1, 7), canonical_split=canonical)
    ids = np.asarray([s["material_id"] for s in samples])
    for part in ("train", "test"):
        assert ids[splits["space_group"][f"{part}_idx"]].tolist() == canonical[f"{part}_material_ids"]
    assert splits["space_group"]["canonical_reused"] is True


@pytest.mark.parametrize("fault", ["duplicate", "missing", "group_overlap"])
def test_canonical_partition_integrity_is_validated(fault):
    samples = samples_fixture()
    canonical = canonical_fixture(samples)
    groups = np.arange(1, 7)
    if fault == "duplicate":
        canonical["train_material_ids"].append("m2")
    elif fault == "missing":
        canonical["train_material_ids"].remove("m2")
    else:
        groups[2] = groups[0]
    with pytest.raises(ValueError, match="canonical"):
        bs.build_all_splits(samples, groups, canonical_split=canonical)


def test_leave_element_holds_out_all_occurrences_and_quarantines_unknown():
    samples = [{"species": x} for x in (["Na", "Cl"], ["Na", "O"], ["Cl", "O"], None)]
    train, test = bs.build_leave_element_split(samples, 0.8, np.random.RandomState(42))
    train_elements = set().union(*(set(samples[i]["species"] or []) for i in train))
    assert all(set(samples[i]["species"] or []) - train_elements for i in test), "test has no unseen element"
    assert 3 not in set(train) | set(test), "unknown species cannot certify elemental isolation"
    assert set(train) | set(test) == {0, 1, 2}


def test_composition_distinguishes_stoichiometry_not_just_elements():
    assert bs._composition_key("FeO") != bs._composition_key("Fe2O3")
    assert bs._composition_key("Fe2O3") == bs._composition_key("Fe4O6")
    assert bs._composition_key("ClNa") == bs._composition_key("NaCl")


def test_protocol_uses_real_source_and_functional_never_directory_guess():
    sample = {"source": "MP", "dft_functional": "SCAN", "source_catalog": "release-1",
              "aurl": "host:AFLOWDATA/LIB1_WEB/x"}
    assert bs._source_protocol_key(sample) == "MP|release-1|SCAN"
    assert bs._parse_aurl(sample["aurl"]) == ("LIB1_WEB", "unknown")
    assert bs._source_protocol_key({"aurl": sample["aurl"]}) == "unknown|LIB1_WEB|unknown"
    assert bs._source_protocol_key({"source": "AFLOW", "dft_type": ["PAW", "PBE"]}) == "AFLOW|unknown|PAW,PBE"


def test_temporal_excludes_undated_from_chronological_claim():
    samples = samples_fixture()
    samples[0]["aflowlib_date"] = None
    train, test = bs.build_temporal_split(samples, 0.8, np.random.RandomState(42))
    assert 0 not in set(train) | set(test), "undated entries cannot be chronological training evidence"
    assert max(samples[i]["aflowlib_date"] for i in train) < min(samples[i]["aflowlib_date"] for i in test)


def test_split_applicability_reports_actual_exclusions_and_proxy_scope():
    samples = samples_fixture()
    samples[0].pop("dft_functional")
    samples[0]["species"] = None
    samples[0]["aflowlib_date"] = None
    samples[0]["prototype"] = None
    splits = bs.build_all_splits(samples, np.arange(1, 7), canonical_split=canonical_fixture(samples))
    for name in ("source_protocol", "leave_element", "temporal"):
        split = splits[name]
        assert split.get("excluded_idx", np.array([])).tolist() == [0]
        assert split["num_train"] + split["num_test"] + split["num_excluded"] == len(samples)
    assert splits["prototype"]["scope"]["proxy_count"] == 1
    assert splits["temporal"]["scope"]["claim"] == "catalog_entry_time_not_blind_dft"
    assert splits["source_protocol"]["scope"]["source_count"] == 1
    for sample in samples:
        sample.pop("dft_functional", None)
    unavailable = bs.build_all_splits(samples, np.arange(1, 7), canonical_split=canonical_fixture(samples))["source_protocol"]
    assert unavailable["status"] == "unavailable"
    assert unavailable["num_excluded"] == len(samples)
    assert unavailable["num_train"] == unavailable["num_test"] == 0


def test_builder_cli_writes_new_version_with_canonical_ids_and_real_metadata(tmp_path, monkeypatch):
    from scripts import build_seven_splits as cli
    import sys
    samples = samples_fixture()
    metadata = [{"material_id": s["material_id"], "source": f"source-{i}", "dft_type": "SCAN"}
                for i, s in enumerate(samples)]
    for sample in samples:
        sample.pop("dft_functional")
        sample.pop("source")
    canonical = dict(canonical_fixture(samples), samples=samples)
    manifest_path = tmp_path / "canonical" / "ood_split_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(canonical))
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    output = tmp_path / "new-version"
    output.mkdir()
    historical = output / "seven_splits_manifest.json"
    historical.write_text("historical immutable")
    monkeypatch.setattr(sys, "argv", ["build", "--manifest", str(manifest_path), "--metadata", str(meta_path), "--output-dir", str(output)])
    cli.main()
    path = output / "seven_splits_manifest.v2.json"
    assert path.is_file(), "new-version manifest missing"
    result = json.loads(path.read_text())
    assert result["schema_version"] == 2
    assert result["space_group"]["test_material_ids"] == canonical["test_material_ids"]
    assert result["source_protocol"]["scope"]["source_count"] == 6
    assert result["provenance"]["canonical_manifest"]["sha256"]
    assert historical.read_text() == "historical immutable"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_bytes() == before


def test_builder_refuses_writes_into_canonical_directory(tmp_path, monkeypatch):
    from scripts import build_seven_splits as cli
    import sys
    samples = samples_fixture()
    manifest = tmp_path / "ood_split_manifest.json"
    manifest.write_text(json.dumps(dict(canonical_fixture(samples), samples=samples)))
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    monkeypatch.setattr(sys, "argv", ["build", "--manifest", str(manifest), "--metadata", str(metadata), "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="canonical"):
        cli.main()
    assert not (tmp_path / "seven_splits_manifest.v2.json").exists()


def test_group_macro_is_equal_weight_per_spacegroup_not_per_class():
    from src.evaluation.bootstrap_stratify import summarize_task
    result = summarize_task(np.zeros(10, dtype=int), np.array([0] * 9 + [1]),
                            np.array([1] * 9 + [2]), "t", 2, n_boot=20)
    assert result.get("group_macro_accuracy") == 0.5
    assert result["accuracy"] == 0.9
    assert result["group_macro_bootstrap"]["point_estimate"] == 0.5
    assert result["group_macro_bootstrap"]["n_groups"] == 2


def test_every_task_cross_stratifies_agree_conflict_with_actual_counts():
    from src.evaluation.bootstrap_stratify import build_error_stratification
    samples = samples_fixture()[:3]
    for sample, d in zip(samples, [0, 1, -1]):
        sample["line_global_disagreement"] = d
        sample["provider_global_electronic_type"] = 2
    result = build_error_stratification(np.zeros(3), np.array([0, 1, 0]), "line_mode_topology", samples)
    assert "agreement" in result
    rows = {r["key"]: r for r in result["agreement"]["rows"]}
    assert rows["conflict"]["n"] == 1 and rows["conflict"]["accuracy"] == 0
    assert rows["unknown"]["n"] == 1
    for axis in ("provider_type", "spacegroup_band", "num_sites_band", "source_catalog"):
        table = result["agreement_cross"][axis]
        assert sum(r["n"] for r in table["rows"]) == 3
        assert any(r["key"].startswith("conflict|") for r in table["rows"])
    assert result["provider_type"]["rows"][0]["key"] == "2"


def test_formal_entrypoint_blocks_missing_selection_before_model_or_outer_reads(tmp_path, monkeypatch):
    from scripts import evaluate_seven_splits as cli
    import builtins
    import sys
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name != "tensorflow" and name != "scripts.finetune_supervised", "model import happened before gate"
        return original_import(name, *args, **kwargs)
    def forbidden_load(*args, **kwargs):
        pytest.fail("outer NPZ accessed before selection gate")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(np, "load", forbidden_load)
    argv = ["evaluate"]
    for flag in ("npz", "manifest", "splits", "labels", "model", "encoder", "norm", "output-dir"):
        argv.extend(["--" + flag, str(tmp_path / flag)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="selection"):
        cli.main()


def evaluation_fixture(tmp_path):
    from types import SimpleNamespace
    from src.utils.selection_manifest import sha256_file
    from scripts.evaluate_seven_splits import TASK_KEYS
    def binding(path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))
    samples = samples_fixture()
    canonical = dict(canonical_fixture(samples), samples=samples)
    manifest_path = tmp_path / "canonical.json"
    manifest_path.write_text(json.dumps(canonical))
    splits = bs.build_all_splits(samples, np.arange(1, 7), canonical_split=canonical)
    bs.save_splits(str(tmp_path / "splits"), splits, np.array([s["material_id"] for s in samples]), 42,
                   {"canonical_manifest": binding(manifest_path)})
    npz = tmp_path / "full.npz"
    np.savez(npz, X=np.zeros((6, 2, 4, 3), dtype=np.float32), material_ids=np.array([f"m{i}" for i in range(6)]), groups=np.arange(1, 7))
    labels = tmp_path / "labels.npz"
    np.savez(labels, material_ids=np.array([f"m{i}" for i in range(6)]),
             **{task: np.zeros(6, dtype=np.int32) for task in TASK_KEYS})
    source = tmp_path / "source.npz"
    np.savez(source, material_ids_train=np.array(["m2", "m3", "m4", "m5"]), groups_train=np.arange(3, 7))
    train_labels = tmp_path / "train_labels.npz"
    np.savez(train_labels, material_ids=np.array(["m2", "m3", "m4", "m5"]),
             **{task: np.zeros(4, dtype=np.int32) for task in TASK_KEYS})
    encoder = tmp_path / "encoder.keras"
    encoder.write_bytes(b"synthetic encoder gate fixture")
    norm = tmp_path / "normalization.json"
    norm.write_text(json.dumps(dict(source="inner_fit", fit_material_ids=["m2", "m3", "m4"],
                                    fit_groups=[3, 4, 5], mean=[0.] * 6, std=[1.] * 6)))
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    trainer = Path(cli.__file__).with_name("finetune_supervised.py")
    roles = dict(source_tensor=source, train_labels=train_labels, encoder=encoder, normalization=norm, trainer_code=trainer)
    contract = dict(fit_material_ids=["m2", "m3", "m4"], selection_material_ids=["m5"],
                    fit_groups=[3, 4, 5], selection_groups=[6], task_names=list(TASK_KEYS),
                    task_dimensions={k: v["n_classes"] for k, v in TASK_KEYS.items()},
                    input_shape=[4, 6], normalization_source="inner_fit",
                    normalization_fit_material_ids=["m2", "m3", "m4"],
                    artifacts={role: binding(path) for role, path in roles.items()})
    states = {}
    for state in ("best", "last", "accepted"):
        path = tmp_path / f"{state}.weights.h5"
        path.write_bytes(b"synthetic frozen state fixture")
        states[state] = binding(path)
    states["accepted"]["source"] = "restored_best"
    selection = dict(schema_version=2, experiment_id="synthetic_fixture_seed42", seed=42,
                     best_epoch_one_based=1, epochs_recorded=1, outer_test_accessed=False,
                     monitor="val_loss", states=states, training_contract=contract)
    selection_path = tmp_path / "inner_selection_manifest.json"
    selection_path.write_text(json.dumps(selection))
    args = SimpleNamespace(selection_manifest=str(selection_path), manifest=str(manifest_path),
                           splits=str(tmp_path / "splits" / "seven_splits_manifest.v2.json"),
                           npz=str(npz), labels=str(labels), model=states["accepted"]["path"],
                           encoder=str(encoder), norm=str(norm), metadata=None, mode="formal", split=["space_group"],
                           output_dir=str(tmp_path / "eval"), n_boot=20, random_state=42, batch_size=2)
    return args, selection


@pytest.mark.parametrize("fault", ["legacy", "duplicate_fit", "fit_selection_overlap", "missing_groups", "wrong_groups", "seen_test"])
def test_formal_gate_rejects_unproven_actual_fit_selection_isolation(tmp_path, fault):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    args, selection = evaluation_fixture(tmp_path)
    c = selection["training_contract"]
    if fault == "legacy":
        selection.pop("schema_version")
    elif fault == "duplicate_fit":
        c["fit_material_ids"][1] = c["fit_material_ids"][0]
    elif fault == "fit_selection_overlap":
        c["selection_material_ids"] = ["m2"]
        c["selection_groups"] = [3]
    elif fault == "missing_groups":
        c.pop("selection_groups")
    elif fault == "wrong_groups":
        c["selection_groups"] = [3]
    else:
        c["selection_material_ids"] = ["m0"]
        c["selection_groups"] = [1]
    Path(args.selection_manifest).write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="schema|fit|selection|group|seen"):
        cli.preflight_evaluation(args)


@pytest.mark.parametrize("fault", ["normalization_bytes", "missing_role", "model_swap", "source_identity", "missing_task"])
def test_formal_gate_binds_frozen_states_and_actual_training_artifacts(tmp_path, fault):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    from src.utils.selection_manifest import sha256_file
    args, selection = evaluation_fixture(tmp_path)
    c = selection["training_contract"]
    if fault == "normalization_bytes":
        Path(args.norm).write_text("changed")
    elif fault == "missing_role":
        c["artifacts"].pop("train_labels")
    elif fault == "model_swap":
        path = tmp_path / "wrong.weights.h5"
        path.write_bytes(b"not accepted weights")
        args.model = str(path)
    elif fault == "source_identity":
        path = Path(c["artifacts"]["source_tensor"]["path"])
        np.savez(path, material_ids_train=np.array(["m0", "m3", "m4", "m5"]), groups_train=np.array([1, 4, 5, 6]))
        c["artifacts"]["source_tensor"].update(bytes=path.stat().st_size, sha256=sha256_file(path))
    else:
        c["task_names"].pop()
    Path(args.selection_manifest).write_text(json.dumps(selection))
    with pytest.raises((ValueError, RuntimeError), match="(?i)artifact|hash|accepted|source|task"):
        cli.preflight_evaluation(args)


def test_manifest_carries_label_free_identity_evidence_for_preflight():
    samples = samples_fixture()
    samples[0].update(gap_type=2, y_gap=4.5, line_global_disagreement=1)
    splits = bs.build_all_splits(samples, np.arange(1, 7), canonical_split=canonical_fixture(samples))
    evidence = splits["space_group"].get("identity_metadata")
    assert isinstance(evidence, dict), "missing target-free group evidence"
    assert set(evidence) == {s["material_id"] for s in samples}
    assert evidence["m0"]["species"] == ["Si"]
    assert evidence["m0"]["spacegroup_number"] == 1
    assert not ({"gap_type", "y_gap", "line_global_disagreement"} & set(evidence["m0"]))


@pytest.mark.parametrize("fault", ["seen_group", "unknown_group", "legacy_split", "false_canonical"])
def test_formal_gate_checks_actual_target_group_identity_without_reading_targets(tmp_path, monkeypatch, fault):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    args, _ = evaluation_fixture(tmp_path)
    if fault in ("seen_group", "unknown_group"):
        np.savez(args.npz, material_ids=np.array([f"m{i}" for i in range(6)]),
                 groups=np.array([3 if fault == "seen_group" else 0, 2, 3, 4, 5, 6]))
    else:
        splits = json.loads(Path(args.splits).read_text())
        if fault == "legacy_split":
            splits.pop("schema_version")
        else:
            splits["space_group"]["train_material_ids"].pop()
        Path(args.splits).write_text(json.dumps(splits))
    original_load = np.load
    class IdentityOnly:
        def __init__(self, path, **kwargs): self.data = original_load(path, **kwargs)
        @property
        def files(self): return self.data.files
        def __enter__(self): return self
        def __exit__(self, *args): self.data.close()
        def __getitem__(self, key):
            assert key in ("material_ids", "groups", "material_ids_train", "groups_train"), "target/feature read before gate"
            return self.data[key]
    monkeypatch.setattr(np, "load", IdentityOnly)
    with pytest.raises(ValueError, match="group|schema|canonical"):
        cli.preflight_evaluation(args)


@pytest.mark.parametrize("axis", ["composition", "prototype", "leave_element", "source_protocol", "temporal"])
def test_formal_axis_semantics_are_rechecked_against_actual_seen_pool(tmp_path, axis):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    args, _ = evaluation_fixture(tmp_path)
    args.split = [axis]
    splits = json.loads(Path(args.splits).read_text())
    canonical = splits["space_group"]
    splits[axis].update(train_material_ids=canonical["train_material_ids"],
                        test_material_ids=canonical["test_material_ids"],
                        num_train=4, num_test=2, excluded_material_ids=[], num_excluded=0)
    identity = canonical["identity_metadata"]
    field = {"composition": "formula_pretty", "prototype": "prototype", "leave_element": "species",
             "source_protocol": "aurl", "temporal": "aflowlib_date"}[axis]
    identity["m0"][field] = identity["m2"][field]
    Path(args.splits).write_text(json.dumps(splits))
    with pytest.raises(ValueError, match=axis):
        cli.preflight_evaluation(args)


@pytest.mark.parametrize("n", [0, 3])
def test_group_intervals_do_not_certify_empty_or_single_group(n):
    from src.evaluation.bootstrap_stratify import group_bootstrap_ci, summarize_task
    y = np.zeros(n, dtype=int)
    groups = np.ones(n, dtype=int)
    ci = group_bootstrap_ci(y, y, groups, n_boot=20)
    assert ci.get("ci_low") is None and ci.get("ci_high") is None
    assert ci["status"] == "insufficient_groups"
    summary = summarize_task(y, y, groups, "task", 3, n_boot=20)
    assert summary["accuracy"] == (1.0 if n else None)
    assert summary["macro_accuracy"] == (1.0 if n else None)
    json.dumps(summary, allow_nan=False)


def test_evaluation_saves_real_probabilities_ids_and_task_denominators():
    import tensorflow as tf
    from scripts import evaluate_seven_splits as cli
    class ThreeHead(tf.keras.Model):
        def call(self, x, training=False):
            assert training is False
            n = tf.shape(x)[0]
            return {"gap": tf.zeros((n, 1)), "topology": tf.tile([[0.2, 0.3, 0.5]], [n, 1]),
                    "type": tf.tile([[0.6, 0.3, 0.1]], [n, 1]),
                    "disagreement": tf.tile([[0.25, 0.75]], [n, 1])}
    samples = samples_fixture()[:3]
    labels = dict(line_mode_topology=np.array([2, -1, 0]), provider_global_electronic_type=np.array([0, 1, 0]),
                  line_global_disagreement=np.array([1, -1, 0]))
    result = cli.evaluate_frozen_model(ThreeHead(), np.zeros((3, 4, 6), dtype=np.float32), np.array(["m0", "m1", "m2"]),
                                     labels, {s["material_id"]: s for s in samples}, "space_group", n_boot=20, batch_size=2)
    assert result.get("material_ids") == ["m0", "m1", "m2"]
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["probabilities"]["line_global_disagreement"] == [0.25, 0.75]
    task = result["tasks"]["line_mode_topology"]
    assert task["valid_material_ids"] == ["m0", "m2"]
    assert task["excluded_material_ids"] == ["m1"]
    assert task["n_samples"] == 2 and task["n_requested"] == 3
    assert task["group_macro_accuracy"] == 0.5
    assert "agreement_cross" in task["error_stratification"]
    json.dumps(result, allow_nan=False)


def test_label_alignment_rejects_duplicate_identity_instead_of_last_row_wins():
    from scripts import evaluate_seven_splits as cli
    with pytest.raises(ValueError, match="unique|duplicate"):
        cli.align_labels_to_ids({"task": np.array([0, 1])}, np.array(["m0", "m0"]), np.array(["m0"]))


@pytest.mark.parametrize("fault", ["label_fraction", "probability_logits", "duplicate_target"])
def test_prediction_ledger_rejects_invalid_labels_or_probabilities(tmp_path, fault):
    import tensorflow as tf
    from scripts import evaluate_seven_splits as cli
    class InvalidOutput(tf.keras.Model):
        def call(self, x, training=False):
            n = tf.shape(x)[0]
            p = [1., 2., 3.] if fault == "probability_logits" else [0.2, 0.3, 0.5]
            return {"gap": tf.zeros((n, 1)), "topology": tf.tile([p], [n, 1]),
                    "type": tf.tile([[0.6, 0.3, 0.1]], [n, 1]), "disagreement": tf.tile([[0.25, 0.75]], [n, 1])}
    ids = np.array(["m0", "m0"] if fault == "duplicate_target" else ["m0", "m1"])
    labels = {key: np.zeros(2, dtype=int) for key in cli.TASK_KEYS}
    if fault == "label_fraction": labels["line_mode_topology"] = np.array([0.5, 0.])
    with pytest.raises(ValueError, match="label|probab|unique|ID"):
        cli.evaluate_frozen_model(InvalidOutput(), np.zeros((2, 4, 6), dtype=np.float32), ids, labels,
                                  {s["material_id"]: s for s in samples_fixture()}, "space_group", n_boot=20)


@pytest.mark.parametrize("label_scope", ["full", "requested_only"])
def test_actual_serialized_model_evaluation_writes_bound_replayable_report(tmp_path, label_scope):
    from scripts import evaluate_seven_splits as cli
    assert callable(getattr(cli, "run_evaluation", None)), "missing gated execution API"
    import tensorflow as tf
    from src.models import SSLEncoder
    from scripts.finetune_supervised import SupervisedBandGapModel
    from src.utils.selection_manifest import sha256_file
    from pathlib import Path
    args, selection = evaluation_fixture(tmp_path)
    tf.keras.utils.set_random_seed(42)
    encoder = SSLEncoder(num_features=6, seq_len=4, d_model=8, num_heads=2,
                         num_layers=1, dff=16, projection_dim=4)
    x = tf.zeros((2, 4, 6))
    encoder(x, training=False)
    encoder.reconstruct(x, training=False)
    encoder.save(args.encoder)
    model = SupervisedBandGapModel(encoder, np.zeros(6), np.ones(6), d_model=8, dropout=0.)
    expected = model(x, training=False)
    for record in selection["states"].values():
        model.save_weights(record["path"])
        record.update(bytes=Path(record["path"]).stat().st_size, sha256=sha256_file(record["path"]))
    artifact = selection["training_contract"]["artifacts"]["encoder"]
    artifact.update(bytes=Path(args.encoder).stat().st_size, sha256=sha256_file(args.encoder))
    config = tmp_path / "model_config.json"
    config.write_text(json.dumps(dict(d_model=8, dropout=0.)))
    selection["training_contract"]["artifacts"]["model_config"] = dict(path=str(config), bytes=config.stat().st_size, sha256=sha256_file(config))
    Path(args.selection_manifest).write_text(json.dumps(selection))
    if label_scope == "requested_only":
        np.savez(args.labels, material_ids=np.array(["m1", "m0"]),
                 **{key: np.zeros(2, dtype=np.int32) for key in cli.TASK_KEYS})
    before = sha256_file(args.model)
    result = cli.run_evaluation(args)
    path = Path(args.output_dir) / "seven_split_evaluation.v2.json"
    assert path.is_file()
    assert result == json.loads(path.read_text())
    assert result["mode"] == "formal"
    assert set(result["splits"]) == {"space_group"}
    assert result["splits"]["space_group"]["n_test"] == 2
    assert result["splits"]["space_group"]["material_ids"] == ["m1", "m0"]
    probabilities = result["splits"]["space_group"]["predictions"][0]["probabilities"]["line_mode_topology"]
    np.testing.assert_allclose(probabilities, expected["topology"].numpy()[0], rtol=1e-6, atol=1e-6)
    for role in ("model", "data", "labels", "normalization", "encoder", "selection_manifest", "evaluator_code", "trainer_code", "splits"):
        record = result["bindings"][role]
        assert record["sha256"] == sha256_file(record["path"])
    assert sha256_file(args.model) == before
    with pytest.raises(FileExistsError):
        cli.run_evaluation(args)


@pytest.mark.parametrize("fault", ["count", "status"])
def test_formal_gate_rejects_manifest_denominator_or_applicability_mismatch(tmp_path, fault):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    args, _ = evaluation_fixture(tmp_path)
    splits = json.loads(Path(args.splits).read_text())
    if fault == "count": splits["space_group"]["num_test"] = 999
    else: splits["space_group"]["status"] = "unavailable"
    Path(args.splits).write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="count|unavailable|denominator"):
        cli.preflight_evaluation(args)


def test_cli_preflight_only_checks_requested_split_without_loading_model(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_seven_splits as cli
    import sys
    assert callable(getattr(cli, "parse_args", None)), "missing targeted preflight CLI API"
    args, _ = evaluation_fixture(tmp_path)
    argv = []
    for flag in ("npz", "manifest", "splits", "labels", "model", "encoder", "norm", "output_dir", "selection_manifest"):
        argv.extend(["--" + flag.replace("_", "-"), getattr(args, flag)])
    argv.extend(["--split", "space_group", "--preflight-only"])
    parsed = cli.parse_args(argv)
    assert parsed.split == ["space_group"] and parsed.preflight_only
    monkeypatch.setattr(sys, "argv", ["evaluate"] + argv)
    cli.main()  # encoder/model fixture is intentionally NOT loadable
    assert "preflight_passed" in capsys.readouterr().out
    from pathlib import Path
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("fault", ["nonpositive_std", "wrong_fit_ids"])
def test_normalization_content_must_be_finite_and_fitted_only_on_recorded_fit_ids(tmp_path, fault):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    from src.utils.selection_manifest import sha256_file
    args, selection = evaluation_fixture(tmp_path)
    norm = dict(source="inner_fit", fit_material_ids=["m2", "m3", "m4"], fit_groups=[3, 4, 5], mean=[0.] * 6, std=[1.] * 6)
    if fault == "nonpositive_std": norm["std"][0] = 0.
    else: norm["fit_material_ids"] = ["m5", "m3", "m4"]
    Path(args.norm).write_text(json.dumps(norm))
    record = selection["training_contract"]["artifacts"]["normalization"]
    record.update(bytes=Path(args.norm).stat().st_size, sha256=sha256_file(args.norm))
    Path(args.selection_manifest).write_text(json.dumps(selection))
    with pytest.raises((ValueError, RuntimeError), match="normalization"):
        cli.preflight_evaluation(args)


def test_invalid_species_tokens_are_unknown_not_novel_elements():
    samples = [{"species": ["Si"]}, {"species": ["O"]}, {"species": ["unknown"]}]
    train, test = bs.build_leave_element_split(samples, .8, np.random.RandomState(42))
    assert 2 not in set(train) | set(test), "placeholder is not a chemical element"


def test_temporal_orders_absolute_timestamps_and_excludes_malformed_dates():
    samples = [{"aflowlib_date": value} for value in [
        "20200101_00:00:00_GMT+0", "20200101_02:00:00_GMT+3", "unknown"]]
    train, test = bs.build_temporal_split(samples, .5, np.random.RandomState(42))
    assert train.tolist() == [1] and test.tolist() == [0]


def test_strata_preserve_unknown_spacegroups_and_reported_catalog():
    from src.evaluation.bootstrap_stratify import spacegroup_band, source_catalog
    assert spacegroup_band(0) == "unknown"
    assert spacegroup_band(None) == "unknown"
    assert source_catalog({"source_catalog": "MP-release", "aurl": "x:AFLOWDATA/LIB1_WEB/y"}) == "MP-release"


def test_builder_can_consume_bound_reported_protocol_sidecar_without_imputation(tmp_path, monkeypatch):
    from scripts import build_seven_splits as cli
    import sys
    assert callable(getattr(cli, "enrich_samples", None)), "missing explicit protocol-sidecar enrichment API"
    samples = samples_fixture()
    for sample in samples: sample.pop("dft_functional")
    protocol = [{"material_id": s["material_id"], "dft_functional": "PAW_PBE"} for s in samples[1:]]
    enriched = cli.enrich_samples(samples, [], protocol)
    assert enriched[0].get("dft_functional") is None
    assert enriched[1]["dft_functional"] == "PAW_PBE"
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    manifest = canonical_dir / "manifest.json"
    manifest.write_text(json.dumps(dict(canonical_fixture(samples), samples=samples)))
    meta = tmp_path / "meta.json"
    meta.write_text("[]")
    sidecar = tmp_path / "reported_protocol.json"
    sidecar.write_text(json.dumps(protocol))
    output = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["build", "--manifest", str(manifest), "--metadata", str(meta),
                                     "--protocol-metadata", str(sidecar), "--output-dir", str(output)])
    cli.main()
    result = json.loads((output / "seven_splits_manifest.v2.json").read_text())
    assert result["source_protocol"]["num_excluded"] == 1
    assert result["provenance"]["protocol_metadata"]["path"] == str(sidecar)


def test_frozen_prediction_refuses_inference_state_mutation():
    import tensorflow as tf
    from scripts import evaluate_seven_splits as cli
    class MutatingModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.counter = self.add_weight(shape=(), initializer="zeros", trainable=False)
        def call(self, x, training=False):
            self.counter.assign_add(1.)
            n = tf.shape(x)[0]
            return {"gap": tf.zeros((n, 1)), "type": tf.tile([[1., 0., 0.]], [n, 1]),
                    "topology": tf.tile([[1., 0., 0.]], [n, 1]), "disagreement": tf.tile([[1., 0.]], [n, 1])}
    with pytest.raises(RuntimeError, match="state|mutat"):
        cli._predict_chunked(MutatingModel(), np.zeros((2, 4, 6), dtype=np.float32), batch_size=1)


def test_builder_rejects_ambiguous_duplicate_metadata_ids():
    from scripts import build_seven_splits as cli
    with pytest.raises(ValueError, match="duplicate"):
        cli.enrich_samples(samples_fixture(), [{"material_id": "m0", "dft_type": "PBE"},
                                               {"material_id": "m0", "dft_type": "LDA"}])


def test_formal_preflight_rechecks_split_builder_input_provenance(tmp_path):
    from scripts import evaluate_seven_splits as cli
    from src.utils.selection_manifest import sha256_file
    from pathlib import Path
    args, _ = evaluation_fixture(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    splits = json.loads(Path(args.splits).read_text())
    splits["provenance"]["metadata"] = dict(path=str(metadata), bytes=metadata.stat().st_size, sha256=sha256_file(metadata))
    Path(args.splits).write_text(json.dumps(splits))
    metadata.write_text("[{}]")
    with pytest.raises(ValueError, match="artifact|metadata"):
        cli.preflight_evaluation(args)


def test_eval_does_not_silently_accept_an_unbound_metadata_override(tmp_path):
    from scripts import evaluate_seven_splits as cli
    from pathlib import Path
    args, _ = evaluation_fixture(tmp_path)
    extra = tmp_path / "unbound_metadata.json"
    extra.write_text("[]")
    args.metadata = str(extra)
    with pytest.raises(ValueError, match="metadata"):
        cli.preflight_evaluation(args)


def test_group_balancing_scales_to_many_distinct_compositions():
    import time
    groups = np.arange(10000)
    start = time.monotonic()
    train, test = bs._group_split(groups, .8, np.random.RandomState(42))
    elapsed = time.monotonic() - start
    assert len(train) == 8000 and len(test) == 2000
    assert elapsed < 5., f"group balancing took {elapsed:.3f}s; repeated full candidate materialization is not scalable"
