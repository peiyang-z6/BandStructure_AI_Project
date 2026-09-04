"""P0D evaluation: three-task metrics over the seven benchmark splits.

Uses a frozen trained model to predict the three P0 tasks on each split's
test partition, then produces:
  - per-task accuracy / macro-accuracy / confusion matrix
  - group bootstrap 95% CI (spacegroup-level resampling)
  - error stratification tables (provider type, spacegroup band, n_sites,
    source catalog)
Output: seven_split_evaluation.json in the output dir.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    outs = {"gap": [], "type": [], "topology": [], "disagreement": []}
    for start in range(0, n, batch_size):
        chunk = X[start : start + batch_size]
        preds = model(chunk, training=False)
        for key in outs:
            outs[key].append(preds[key].numpy())
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
    preds = _predict_chunked(model, X_test, batch_size=batch_size)
    tasks = {}
    for task_name, spec in TASK_KEYS.items():
        key = spec["label_key"]
        y_true = np.asarray(labels[key], dtype=np.int32)
        if task_name == "line_global_disagreement":
            y_pred = np.argmax(preds["disagreement"], axis=1).astype(np.int32)
        elif task_name == "line_mode_topology":
            y_pred = np.argmax(preds["topology"], axis=1).astype(np.int32)
        else:
            y_pred = np.argmax(preds["type"], axis=1).astype(np.int32)
        valid = y_true >= 0
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]
        ids_v = material_ids_test[valid]
        samples_v = [samples_by_id[str(mid)] for mid in ids_v]
        summary = summarize_task(
            y_true_v,
            y_pred_v,
            None,  # group bootstrap below, spacegroup-aligned
            task_name,
            spec["n_classes"],
            n_boot=n_boot,
            random_state=random_state,
        )
        sgs = np.asarray(
            [s.get("spacegroup_number", -1) for s in samples_v], dtype=np.int32
        )
        has_sg = sgs >= 0
        summary["group_bootstrap"] = _bootstrap_with_groups(
            y_true_v, y_pred_v, sgs, n_boot, random_state
        )
        summary["error_stratification"] = build_error_stratification(
            y_true_v, y_pred_v, task_name, samples_v
        )
        tasks[task_name] = summary
    return {"split": split_name, "n_test": int(len(X_test)), "tasks": tasks}


def _bootstrap_with_groups(y_true, y_pred, groups, n_boot, random_state):
    from src.evaluation.bootstrap_stratify import group_bootstrap_ci

    if len(np.unique(groups)) < 2:
        return None
    return group_bootstrap_ci(
        y_true, y_pred, groups, n_boot=n_boot, random_state=random_state
    )


def load_inputs(args):
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    samples = manifest.get("samples", [])
    samples_by_id = {str(s["material_id"]): s for s in samples}
    splits = json.load(open(args.splits, encoding="utf-8"))
    labels_npz = np.load(args.labels)
    labels = {
        key: labels_npz[key] for key in labels_npz.files if key != "material_ids"
    }
    label_ids = labels_npz["material_ids"].astype(str)
    full_npz = np.load(args.npz)
    X = full_npz["X"]
    material_ids = full_npz["material_ids"].astype(str)
    return samples_by_id, splits, labels, label_ids, X, material_ids


def align_labels_to_ids(labels, label_ids, material_ids):
    idx = {str(mid): i for i, mid in enumerate(label_ids)}
    order = [idx[str(mid)] for mid in material_ids]
    return {key: np.asarray(arr)[order] for key, arr in labels.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seven-split three-task evaluation")
    parser.add_argument("--npz", required=True, help="band_tensors_full.npz")
    parser.add_argument("--manifest", required=True, help="ood_split_manifest.json")
    parser.add_argument("--splits", required=True, help="seven_splits_manifest.json")
    parser.add_argument("--labels", required=True, help="three_task_labels.npz")
    parser.add_argument("--model", required=True, help="frozen model weights .h5")
    parser.add_argument("--encoder", required=True, help="ssl encoder .keras")
    parser.add_argument("--norm", required=True, help="norm stats json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.random_state)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.finetune_supervised import SupervisedBandGapModel, load_norm_stats, flatten_tensor
    from src.models import load_ssl_encoder

    samples_by_id, splits, labels, label_ids, X, material_ids = load_inputs(args)
    aligned = align_labels_to_ids(labels, label_ids, material_ids)
    mean, std = load_norm_stats(args.norm)
    X_flat = (flatten_tensor(X) - mean) / std

    encoder = load_ssl_encoder(args.encoder, compile=False)
    model = SupervisedBandGapModel(
        encoder,
        feature_mean=mean.reshape(-1).astype(np.float32),
        feature_std=std.reshape(-1).astype(np.float32),
    )
    model(tf.zeros([1, X_flat.shape[1], X_flat.shape[2]], dtype=tf.float32))
    model.load_weights(args.model)

    results = {"random_state": args.random_state, "splits": {}}
    for split_name, split in splits.items():
        if split_name == "random_state":
            continue
        test_ids = [str(mid) for mid in split["test_material_ids"]]
        id_pos = {str(mid): i for i, mid in enumerate(material_ids)}
        positions = [id_pos[mid] for mid in test_ids]
        positions = np.asarray(positions, dtype=np.int32)
        X_test = X_flat[positions]
        ids_test = material_ids[positions]
        labels_test = {k: v[positions] for k, v in aligned.items()}
        split_result = evaluate_frozen_model(
            model,
            X_test,
            ids_test,
            labels_test,
            samples_by_id,
            split_name,
            n_boot=args.n_boot,
            random_state=args.random_state,
        )
        results["splits"][split_name] = split_result
        print(f"  {split_name}: n={split_result['n_test']} done")

    out = os.path.join(args.output_dir, "seven_split_evaluation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
