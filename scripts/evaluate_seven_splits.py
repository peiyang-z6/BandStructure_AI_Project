"""Formal frozen-model evaluation with fail-closed schema-2 evidence.

Preflight uses only identities/group metadata and artifact hashes, before model
imports or outer feature/target arrays. Seen fit/selection IDs or groups never
receive formal holdout scores; no automatic filtering or diagnostic fallback.
Outputs a new seven_split_evaluation.v2.json with replayable per-ID probabilities,
actual task denominators, sample/group-macro metrics, and artifact/code bindings.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.selection_manifest import sha256_file, validate_inner_selection_manifest

import numpy as np

from src.evaluation.bootstrap_stratify import (
    build_error_stratification,
    summarize_task,
)

TASK_KEYS = {
    "line_mode_topology": {"n_classes": 3, "label_key": "line_mode_topology"},
    "provider_global_electronic_type": {
        "n_classes": 3,
        "label_key": "provider_global_electronic_type",
    },
    "line_global_disagreement": {
        "n_classes": 2,
        "label_key": "line_global_disagreement",
    },
}


def _predict_chunked(model, X, batch_size=256):
    """Chunked forward pass (OOM guard, mirrors v7 chunked-eval fix)."""
    n = len(X)
    frozen = [weight.numpy().copy() for weight in model.weights]
    outs = {"gap": [], "type": [], "topology": [], "disagreement": []}
    for start in range(0, n, batch_size):
        chunk = X[start : start + batch_size]
        preds = model(chunk, training=False)
        for key in outs:
            outs[key].append(preds[key].numpy())
    if len(frozen) != len(model.weights) or any(not np.array_equal(before, weight.numpy())
                                               for before, weight in zip(frozen, model.weights)):
        raise RuntimeError("frozen model state mutated during inference; refusing metrics")
    return {key: np.concatenate(vals, axis=0) for key, vals in outs.items()}


def evaluate_frozen_model(
    model,
    X_test: np.ndarray,
    material_ids_test: np.ndarray,
    labels: dict,
    samples_by_id: dict,
    split_name: str,
    n_boot: int = 500,
    random_state: int = 42,
    batch_size: int = 256,
) -> dict:
    material_ids_test = np.asarray(material_ids_test)
    _unique_ids(material_ids_test.tolist(), "evaluation IDs")
    if len(X_test) != len(material_ids_test):
        raise ValueError("feature/ID length mismatch")
    for key, spec in TASK_KEYS.items():
        values = np.asarray(labels.get(key))
        if (values.shape != (len(material_ids_test),) or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < -1) or np.any(values >= spec["n_classes"])):
            raise ValueError(f"{key}: labels must be aligned integer classes or explicit -1 unknown")
    preds = _predict_chunked(model, X_test, batch_size=batch_size)
    heads = dict(zip(TASK_KEYS, ("topology", "type", "disagreement")))
    for key, head in heads.items():
        probability = np.asarray(preds[head])
        if (probability.shape != (len(material_ids_test), TASK_KEYS[key]["n_classes"])
                or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)
                or not np.allclose(probability.sum(axis=1), 1., rtol=0, atol=1e-5)):
            raise ValueError(f"{key}: invalid probability tensor (not a finite normalized distribution)")
    samples = [{**samples_by_id[str(mid)],
                **{key: int(np.asarray(labels[key])[i]) for key in TASK_KEYS}}
               for i, mid in enumerate(material_ids_test)]
    tasks = {}
    for task_name, spec in TASK_KEYS.items():
        y_true = np.asarray(labels[spec["label_key"]], dtype=np.int32)
        y_pred = np.argmax(preds[heads[task_name]], axis=1).astype(np.int32)
        valid = y_true >= 0
        ids_v = material_ids_test[valid]
        samples_v = [samples[i] for i in np.flatnonzero(valid)]
        groups = np.asarray([s.get("spacegroup_number", -1) for s in samples_v], dtype=np.int32)
        summary = summarize_task(y_true[valid], y_pred[valid], groups, task_name,
                                 spec["n_classes"], n_boot=n_boot, random_state=random_state)
        summary.update(n_requested=len(material_ids_test), n_excluded=int((~valid).sum()),
                       valid_material_ids=ids_v.tolist(), excluded_material_ids=material_ids_test[~valid].tolist())
        summary["error_stratification"] = build_error_stratification(
            y_true[valid], y_pred[valid], task_name, samples_v)
        tasks[task_name] = summary
    records = [{"material_id": str(mid), "spacegroup_number": samples[i].get("spacegroup_number"),
                "labels": {key: int(np.asarray(labels[key])[i]) for key in TASK_KEYS},
                "probabilities": {key: preds[head][i].tolist() for key, head in heads.items()}}
               for i, mid in enumerate(material_ids_test)]
    return {"split": split_name, "evaluation_role": "array_diagnostic", "n_test": int(len(X_test)),
            "material_ids": material_ids_test.tolist(), "predictions": records, "tasks": tasks}


def load_inputs(args, gate=None):
    gate = preflight_evaluation(args) if gate is None else gate
    with np.load(args.labels, allow_pickle=False) as labels_npz:
        labels = {key: labels_npz[key] for key in TASK_KEYS}
        label_ids = labels_npz["material_ids"].astype(str)
    with np.load(args.npz, allow_pickle=False) as full_npz:
        X = full_npz["X"]
        material_ids = full_npz["material_ids"].astype(str)
    return gate["identity_metadata"], gate["splits"], labels, label_ids, X, material_ids


def align_labels_to_ids(labels, label_ids, material_ids):
    _unique_ids(np.asarray(label_ids).tolist(), "label IDs")
    _unique_ids(np.asarray(material_ids).tolist(), "target IDs")
    idx = {str(mid): i for i, mid in enumerate(label_ids)}
    order = [idx[str(mid)] for mid in material_ids]
    return {key: np.asarray(arr)[order] for key, arr in labels.items()}


def _unique_ids(ids, label):
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(mid, str) or not mid for mid in ids) or len(set(ids)) != len(ids)):
        raise ValueError(f"{label}: nonempty unique string IDs required")
    return ids


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_binding(record, role):
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError(f"missing artifact role {role}")
    actual = _binding(record["path"])
    if actual["sha256"] != record.get("sha256") or actual["bytes"] != record.get("bytes"):
        raise ValueError(f"artifact hash/bytes mismatch: {role}")
    return actual


def _validate_axis(name, seen, test):
    from src.data.benchmark_splits import _composition_key, _element_keys, _source_protocol_key, _temporal_timestamp

    if name == "leave_element":
        seen_elements = [set(_element_keys(s.get("species"))) for s in seen]
        test_elements = [set(_element_keys(s.get("species"))) for s in test]
        if not all(seen_elements + test_elements):
            raise ValueError("leave_element has unknown species in the actual seen/test pool")
        union = set().union(*seen_elements)
        if any(not elements - union for elements in test_elements):
            raise ValueError("leave_element: a test row lacks an element unseen during fit/selection")
    elif name == "temporal":
        dates_seen = [_temporal_timestamp(s.get("aflowlib_date")) for s in seen]
        dates_test = [_temporal_timestamp(s.get("aflowlib_date")) for s in test]
        if not all(dates_seen + dates_test) or max(dates_seen) >= min(dates_test):
            raise ValueError("temporal: actual seen pool is undated or not strictly earlier than test")
    elif name in ("composition", "prototype", "source_protocol"):
        key_fn = {"composition": lambda s: _composition_key(s.get("formula_pretty")),
                  "prototype": lambda s: s.get("prototype") or "unknown",
                  "source_protocol": _source_protocol_key}[name]
        a, b = [key_fn(s) for s in seen], [key_fn(s) for s in test]
        unknown = any(k == "unknown" for k in a + b)
        if name == "source_protocol":
            unknown = any(k.split("|")[0] == "unknown" or k.split("|")[-1] == "unknown" for k in a + b)
        if unknown or set(a) & set(b):
            raise ValueError(f"{name}: unknown/proxy evidence or actual fit/selection key overlap")


def preflight_evaluation(args):
    """Validate evidence before model imports or any outer arrays/targets."""
    if not args.selection_manifest:
        raise ValueError("a frozen selection manifest is required before evaluation")
    with open(args.selection_manifest, encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("schema_version") != 2:
        raise ValueError("formal evaluation requires selection schema_version=2")
    contract = selection.get("training_contract", {})
    fit_ids = _unique_ids(contract.get("fit_material_ids"), "fit")
    selection_ids = _unique_ids(contract.get("selection_material_ids"), "selection")
    if set(fit_ids) & set(selection_ids):
        raise ValueError("fit/selection ID overlap")
    for part, ids in (("fit", fit_ids), ("selection", selection_ids)):
        groups = contract.get(f"{part}_groups")
        if (not isinstance(groups, list) or len(groups) != len(ids)
                or any(not isinstance(g, int) or not 1 <= g <= 230 for g in groups)):
            raise ValueError(f"{part} groups must be valid spacegroups aligned one-to-one to IDs")
    if set(contract["fit_groups"]) & set(contract["selection_groups"]):
        raise ValueError("fit/selection group overlap")
    if set(contract.get("task_names", [])) != set(TASK_KEYS) or len(contract["task_names"]) != len(TASK_KEYS):
        raise ValueError("training task_names must specify exactly the three P0 tasks")
    if Path(args.selection_manifest).name != "inner_selection_manifest.json":
        raise ValueError("selection manifest must use its canonical evidence filename")
    validate_inner_selection_manifest(str(Path(args.selection_manifest).parent))
    artifacts = {role: _verify_binding(contract.get("artifacts", {}).get(role), role)
                 for role in ("source_tensor", "train_labels", "encoder", "normalization", "trainer_code")}
    if _binding(args.model)["sha256"] != selection["states"]["accepted"]["sha256"]:
        raise ValueError("model is not the frozen accepted state")
    for role, path in (("encoder", args.encoder), ("normalization", args.norm),
                       ("trainer_code", Path(__file__).with_name("finetune_supervised.py"))):
        if _binding(path)["sha256"] != artifacts[role]["sha256"]:
            raise ValueError(f"runtime artifact does not match training: {role}")
    with np.load(artifacts["source_tensor"]["path"], allow_pickle=False) as source:
        ids_key = "material_ids_train" if "material_ids_train" in source.files else "material_ids"
        groups_key = "groups_train" if "groups_train" in source.files else "groups"
        source_ids = _unique_ids(source[ids_key].tolist(), "source tensor")
        source_groups = source[groups_key].tolist()
    if len(source_ids) != len(source_groups) or set(source_ids) != set(fit_ids + selection_ids):
        raise ValueError("source tensor IDs do not equal actual fit+selection IDs")
    by_id = dict(zip(source_ids, source_groups))
    for part, ids in (("fit", fit_ids), ("selection", selection_ids)):
        if [by_id[mid] for mid in ids] != contract[f"{part}_groups"]:
            raise ValueError(f"source tensor group evidence does not match {part}")
    with np.load(artifacts["train_labels"]["path"], allow_pickle=False) as train_labels:
        label_ids = _unique_ids(train_labels["material_ids"].tolist(), "training labels")
    if set(label_ids) != set(source_ids):
        raise ValueError("train_labels artifact IDs do not match source tensor training pool")
    with open(args.splits, encoding="utf-8") as handle:
        splits = json.load(handle)
    from src.data.benchmark_splits import SPLIT_NAMES
    names = getattr(args, "split", None) or list(SPLIT_NAMES)
    if splits.get("schema_version") != 2:
        raise ValueError("formal evaluation requires split schema_version=2")
    seen_ids = set(fit_ids) | set(selection_ids)
    canonical = splits.get("space_group", {})
    if canonical.get("canonical_reused") is not True or set(canonical.get("train_material_ids", [])) != seen_ids:
        raise ValueError("canonical training partition does not match actual fit+selection IDs")
    _verify_binding(splits.get("provenance", {}).get("canonical_manifest"), "canonical_manifest")
    for role, record in splits["provenance"].items():
        _verify_binding(record, f"split_{role}")
    if getattr(args, "metadata", None):
        expected = splits["provenance"].get("metadata", {})
        if _binding(args.metadata)["sha256"] != expected.get("sha256"):
            raise ValueError("metadata override is not the frozen split-builder metadata")
    if _binding(args.manifest)["sha256"] != splits["provenance"]["canonical_manifest"]["sha256"]:
        raise ValueError("canonical manifest binding mismatch")
    with np.load(args.npz, allow_pickle=False) as full:
        full_ids = _unique_ids(full["material_ids"].tolist(), "evaluation tensor")
        full_groups = full["groups"].tolist()
    if (len(full_ids) != len(full_groups)
            or any(type(g) is not int or not 1 <= g <= 230 for g in full_groups)):
        raise ValueError("evaluation tensor has missing/unknown group evidence")
    actual_groups = dict(zip(full_ids, full_groups))
    identity = canonical.get("identity_metadata", {})
    if set(identity) != set(full_ids) or any(identity[mid].get("spacegroup_number") != actual_groups[mid] for mid in full_ids):
        raise ValueError("split group evidence does not match actual tensor identity")
    if set(canonical.get("test_material_ids", [])) != set(full_ids) - seen_ids:
        raise ValueError("canonical test must be the exact unseen complement of training IDs")
    seen_groups = set(contract["fit_groups"] + contract["selection_groups"])
    for name in names:
        split = splits[name]
        if split.get("status") != "available":
            raise ValueError(f"{name}: unavailable split cannot produce formal metrics")
        for part in ("train", "test", "excluded"):
            if split.get(f"num_{part}") != len(split.get(f"{part}_material_ids", [])):
                raise ValueError(f"{name}: manifest {part} count/denominator mismatch")
        test_ids = _unique_ids(split.get("test_material_ids"), f"{name} test")
        overlap = seen_ids & set(test_ids)
        if overlap:
            raise ValueError(f"{name}: test contains {len(overlap)} actual fit/selection seen IDs")
        if not set(test_ids) <= set(full_ids):
            raise ValueError(f"{name}: test IDs absent from actual tensor")
        if {actual_groups[mid] for mid in test_ids} & seen_groups:
            raise ValueError(f"{name}: test has actual fit/selection group overlap")
        _validate_axis(name, [identity[mid] for mid in fit_ids + selection_ids], [identity[mid] for mid in test_ids])
    return {"mode": "formal", "split_names": names, "selection": selection, "splits": splits,
            "identity_metadata": identity}


def run_evaluation(args):
    """One frozen, read-back-verifiable execution; no auto diagnostic fallback."""
    gate = preflight_evaluation(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError("evaluation output-dir must be a fresh version directory")
    from src.evaluation import bootstrap_stratify
    from src.data import benchmark_splits
    paths = {"model": args.model, "data": args.npz, "labels": args.labels,
             "normalization": args.norm, "encoder": args.encoder,
             "selection_manifest": args.selection_manifest, "splits": args.splits,
             "canonical_manifest": args.manifest, "evaluator_code": __file__,
             "bootstrap_code": bootstrap_stratify.__file__, "split_code": benchmark_splits.__file__}
    bindings = {role: _binding(path) for role, path in paths.items()}
    bindings.update({role: _verify_binding(record, role)
                     for role, record in gate["selection"]["training_contract"]["artifacts"].items()})
    import tensorflow as tf
    from scripts.finetune_supervised import SupervisedBandGapModel, load_norm_stats, flatten_tensor
    from src.models import load_ssl_encoder

    tf.keras.utils.set_random_seed(args.random_state)
    samples, splits, labels, label_ids, X, material_ids = load_inputs(args, gate)
    mean, std = load_norm_stats(args.norm)
    X_flat = (flatten_tensor(X) - mean) / std
    config = {}
    if "model_config" in bindings:
        with open(bindings["model_config"]["path"], encoding="utf-8") as handle:
            config = json.load(handle)
    encoder = load_ssl_encoder(args.encoder, compile=False)
    model = SupervisedBandGapModel(encoder, feature_mean=mean.reshape(-1), feature_std=std.reshape(-1),
                                  d_model=config.get("d_model", 128), dropout=config.get("dropout", .1))
    model(tf.zeros([1, X_flat.shape[1], X_flat.shape[2]], dtype=tf.float32), training=False)
    model.load_weights(args.model)
    model.compile(jit_compile=False)
    results = {"schema_version": 2, "mode": gate["mode"], "random_state": args.random_state,
               "bindings": bindings, "splits": {}}
    id_pos = {mid: i for i, mid in enumerate(material_ids.tolist())}
    for name in gate["split_names"]:
        split = splits[name]
        positions = np.asarray([id_pos[mid] for mid in split["test_material_ids"]], dtype=int)
        labels_for_test = align_labels_to_ids(labels, label_ids, material_ids[positions])
        result = evaluate_frozen_model(model, X_flat[positions], material_ids[positions],
                                      labels_for_test,
                                      samples, name, n_boot=args.n_boot,
                                      random_state=args.random_state, batch_size=args.batch_size)
        result["evaluation_role"] = "formal_unseen_holdout"
        result["scope"] = split["scope"]
        results["splits"][name] = result
    # Do not publish if any frozen input/code artifact changed during execution.
    for role, binding in bindings.items():
        _verify_binding(binding, role)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "seven_split_evaluation.v2.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"wrote {path}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fail-closed frozen three-task evaluation (formal schema 2 only)")
    parser.add_argument("--npz", required=True, help="band_tensors_full.npz")
    parser.add_argument("--manifest", required=True, help="ood_split_manifest.json")
    parser.add_argument("--splits", required=True, help="seven_splits_manifest.json")
    parser.add_argument("--labels", required=True, help="three_task_labels.npz")
    parser.add_argument("--metadata", default=None, help="optional verification of the already-bound split-builder metadata; no evaluation-time enrichment")
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument("--model", required=True, help="frozen model weights .h5")
    parser.add_argument("--encoder", required=True, help="ssl encoder .keras")
    parser.add_argument("--norm", required=True, help="norm stats json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    from src.data.benchmark_splits import SPLIT_NAMES
    parser.add_argument("--split", action="append", choices=SPLIT_NAMES,
                        help="Evaluate only these explicit partitions; default all, with no automatic filtering")
    parser.add_argument("--preflight-only", action="store_true", help="Validate identities/artifacts only; no model or target loading")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        gate = preflight_evaluation(args)
        print(json.dumps({"status": "preflight_passed", "mode": "formal", "scores_computed": False,
                          "test_counts": {name: len(gate["splits"][name]["test_material_ids"])
                                          for name in gate["split_names"]}}, sort_keys=True))
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
