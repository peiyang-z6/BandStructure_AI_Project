"""
End-to-end training pipeline for the current physics-aware model brain.

The default behavior is conservative:
- keep raw Materials Project data when enough samples already exist
- rebuild only missing downstream artifacts
- never change the spacegroup-based OOD split contract

Use --fresh to rebuild processed tensors, SSL, fine-tuning, checkpoints, and
reports. Use --clean-raw only when raw downloaded data should also be removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


PROTECTED_RELATIVE_PATHS = [
    "PROJECT_BRAIN",
    "PROJECT_BRAIN/CONSTITUTION.md",
    "configs/api_keys.env",
]


def rel(path: str) -> Path:
    return ROOT / path


def run_command(command: List[str], stage: str) -> None:
    print("\n" + "=" * 80)
    print(f"[Stage] {stage}")
    print("=" * 80)
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def build_finetune_command(
    args: argparse.Namespace,
    paths: Dict[str, str],
    ood_split: str,
    mode: str | None = None,
) -> List[str]:
    """Build the supervised command and preserve top-level GPU requirements."""
    command = [
        sys.executable,
        "scripts/finetune_supervised.py",
        "--tensor-npz",
        ood_split,
        "--encoder",
        paths["ssl_encoder"],
        "--norm",
        paths["ssl_norm"],
        "--epochs",
        str(args.finetune_epochs),
        "--batch-size",
        str(args.finetune_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--encoder-learning-rate",
        str(args.encoder_learning_rate),
        "--type-weight",
        str(args.type_weight),
        "--freeze-layers",
        str(args.freeze_layers),
        "--topology-weight",
        str(args.topology_weight),
        "--entropy-weight",
        str(args.entropy_weight),
        "--extremum-weight",
        str(args.extremum_weight),
        "--random-state",
        str(args.random_state),
        "--output-dir",
        paths["finetune_report_dir"],
        "--checkpoint-dir",
        paths["finetune_checkpoint_dir"],
        "--model-path",
        paths["finetuned_weights"],
    ]
    for flag, value in (
        ("--experiment-id", getattr(args, "experiment_id", None)),
        ("--source", getattr(args, "source", None)),
        ("--report-date", getattr(args, "report_date", None)),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if args.require_gpu:
        command.append("--require-gpu")
    if mode == "train":
        command.append("--train-only")
    elif mode == "evaluate":
        command.append("--evaluation-only")
    elif mode is not None:
        raise ValueError(f"Unknown supervised pipeline mode: {mode}")
    return command


def run_supervised_stages(
    args: argparse.Namespace,
    paths: Dict[str, str],
    ood_split: str,
) -> None:
    """Freeze inner-selected states before starting outer OOD evaluation."""
    run_command(
        build_finetune_command(args, paths, ood_split, mode="train"),
        "Supervised inner-only training and checkpoint selection",
    )
    run_command(
        build_finetune_command(args, paths, ood_split, mode="evaluate"),
        "Frozen-checkpoint outer OOD evaluation",
    )


def build_ssl_command(
    args: argparse.Namespace,
    paths: Dict[str, str],
    ood_split: str,
) -> List[str]:
    """Build a fully explicit, auditable SSL command."""
    command = [
        sys.executable,
        "scripts/train_ssl.py",
        "--tensor-npz",
        ood_split,
        "--epochs",
        str(args.ssl_epochs),
        "--batch-size",
        str(args.ssl_batch_size),
        "--mask-ratio",
        str(args.mask_ratio),
        "--random-state",
        str(args.random_state),
        "--sign-weight",
        str(args.sign_weight),
        "--consistency-weight",
        str(args.consistency_weight),
        "--d-model",
        str(args.d_model),
        "--num-heads",
        str(args.num_heads),
        "--num-layers",
        str(args.num_layers),
        "--dff",
        str(args.dff),
        "--projection-dim",
        str(args.projection_dim),
        "--strain-scale",
        str(args.strain_scale),
        "--checkpoint-dir",
        paths["ssl_checkpoint_dir"],
        "--log-dir",
        paths["ssl_log_dir"],
        "--model-dir",
        paths["model_dir"],
    ]
    command.append(
        "--disable-strain-augmentation"
        if args.disable_strain_augmentation
        else "--enable-strain-augmentation"
    )
    if args.require_gpu:
        command.append("--require-gpu")
    return command


def remove_path(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove outside project root: {resolved}")
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def count_h5_samples(h5_path: Path) -> int:
    if not h5_path.exists():
        return 0
    import h5py

    with h5py.File(h5_path, "r") as f:
        return len(f.keys())


def validate_download_gate(
    paths: Dict[str, str],
    target: int,
    *,
    require_report: bool,
) -> Dict[str, object] | None:
    """Reject downstream stages unless the canonical cache reached the target."""
    persisted = count_h5_samples(rel(paths["raw_h5"]))
    if persisted < target:
        raise RuntimeError(
            "Download completion gate failed: persisted "
            f"{persisted} < requested target {target}"
        )
    if not require_report:
        return None

    report_path = rel(paths["download_report"])
    if not report_path.exists():
        raise RuntimeError(
            f"Download completion gate failed: missing report {report_path}"
        )
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise RuntimeError("Download completion gate failed: report is not an object")
    if report.get("target_reached") is not True:
        raise RuntimeError(
            "Download completion gate failed: target_reached=false; "
            f"termination_reason={report.get('termination_reason')}"
        )
    if int(report.get("requested_target", -1)) != int(target):
        raise RuntimeError(
            "Download completion gate failed: report requested_target "
            f"{report.get('requested_target')} != {target}"
        )
    if int(report.get("total_cached", -1)) != persisted:
        raise RuntimeError(
            "Download completion gate failed: report total_cached "
            f"{report.get('total_cached')} != persisted {persisted}"
        )
    return report


def _validated_input_path(
    path: str,
    *,
    expected_prefix: tuple[str, ...],
    label: str,
    required_suffix: str | None = None,
) -> str:
    normalized = PurePosixPath(str(path).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe explicit {label}: {path!r}")
    if tuple(normalized.parts[: len(expected_prefix)]) != expected_prefix:
        raise ValueError(
            f"Explicit {label} must stay under {'/'.join(expected_prefix)}/: {path!r}"
        )
    if required_suffix and normalized.suffix.lower() != required_suffix:
        raise ValueError(
            f"Explicit {label} must end with {required_suffix}: {path!r}"
        )
    return normalized.as_posix()


def pipeline_layout(
    source: str = "mp",
    *,
    experiment_id: str | None = None,
    raw_h5: str | None = None,
    ood_dir: str | None = None,
) -> Dict[str, str]:
    """Return source-isolated paths with optional immutable versioned inputs."""
    source_key = source.lower()
    if source_key not in {"mp", "aflow"}:
        raise ValueError(f"Unsupported source: {source!r}")
    source_dir = "materials_project" if source_key == "mp" else source_key
    explicit_experiment = experiment_id is not None
    explicit_raw_input = raw_h5 is not None
    explicit_ood_input = ood_dir is not None
    if experiment_id is None:
        experiment_id = "aflow_noleak_v4_seed42" if source_key == "aflow" else source_key
    if (
        not experiment_id
        or not experiment_id[0].isalnum()
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in experiment_id
        )
    ):
        raise ValueError(f"Unsafe experiment ID: {experiment_id!r}")
    if raw_h5 is not None:
        raw_h5 = _validated_input_path(
            raw_h5,
            expected_prefix=("data", "raw", source_dir),
            label="raw HDF5",
            required_suffix=".h5",
        )
    if ood_dir is not None:
        ood_dir = _validated_input_path(
            ood_dir,
            expected_prefix=("data", "processed", source_dir),
            label="OOD directory",
        )
    raw_dir = f"data/raw/{source_dir}"
    processed_dir = ood_dir or f"data/processed/{source_dir}/ood_tensors"
    model_dir = f"artifacts/models/{experiment_id}"
    report_dir = f"artifacts/reports/{experiment_id}"
    checkpoint_dir = f"artifacts/checkpoints/{experiment_id}"
    log_dir = f"artifacts/logs/{experiment_id}"
    is_formal_aflow = source_key == "aflow"
    raw_h5_path = raw_h5 or f"{raw_dir}/{source_key}_bands.h5"
    metadata_path = (
        PurePosixPath(raw_h5_path).with_name(f"{source_key}_metadata.json").as_posix()
        if explicit_raw_input
        else f"{raw_dir}/{source_key}_metadata.json"
    )
    raw_snapshot_manifest = (
        PurePosixPath(raw_h5_path).with_name("raw_snapshot_manifest.json").as_posix()
        if explicit_raw_input
        else None
    )
    tensor_snapshot_audit = (
        f"{processed_dir}/tensor_snapshot_audit.json"
        if explicit_ood_input
        else None
    )
    return {
        "experiment_id": experiment_id,
        "raw_h5": raw_h5_path,
        "metadata": metadata_path,
        "raw_snapshot_manifest": raw_snapshot_manifest,
        "tensor_snapshot_audit": tensor_snapshot_audit,
        "download_report": f"{raw_dir}/{source_key}_download_report.json",
        "json_cache": f"{raw_dir}/json_cache",
        "ood_dir": processed_dir,
        "ssl_encoder": f"{model_dir}/ssl_mbm_pretrained.keras",
        "ssl_norm": f"{model_dir}/ssl_mbm_norm_stats.json",
        "model_dir": model_dir,
        "finetuned_weights": f"{model_dir}/{'finetuned.weights.h5' if is_formal_aflow else 'finetuned_gap_predictor.weights.h5'}",
        "finetune_report_dir": report_dir if is_formal_aflow else f"{report_dir}/finetune_supervised",
        "validation_report": (
            f"{report_dir}/latest_training_report.md"
            if is_formal_aflow and explicit_experiment
            else f"{report_dir}/latest_training_report_20260824.md"
            if is_formal_aflow
            else f"{report_dir}/finetune_supervised_report.md"
        ),
        "ssl_checkpoint_dir": (
            f"{checkpoint_dir}/ssl"
            if is_formal_aflow and explicit_experiment
            else checkpoint_dir
            if is_formal_aflow
            else f"{checkpoint_dir}/ssl_mbm"
        ),
        "finetune_checkpoint_dir": (
            f"{checkpoint_dir}/supervised"
            if is_formal_aflow and explicit_experiment
            else f"{checkpoint_dir}/ft"
            if is_formal_aflow
            else f"{checkpoint_dir}/finetune_supervised"
        ),
        "ssl_log_dir": (
            f"{log_dir}/ssl"
            if is_formal_aflow and explicit_experiment
            else f"{log_dir}/tensorboard"
            if is_formal_aflow
            else f"{log_dir}/ssl_mbm"
        ),
        "vision_synthetic_dir": (
            f"data/processed/{source_dir}/vision_synthetic_train/{experiment_id}"
            if explicit_experiment
            else f"data/processed/{source_dir}/vision_synthetic_train"
        ),
        "vision_model_dir": f"{model_dir}/vision_detector",
        "vision_runs_dir": f"{log_dir}/vision_detector",
        "manifest": f"{model_dir}/physics_model_brain_manifest.json",
    }


def required_artifacts(
    source: str = "mp",
    *,
    experiment_id: str | None = None,
    raw_h5: str | None = None,
    ood_dir: str | None = None,
) -> Dict[str, str]:
    paths = pipeline_layout(
        source,
        experiment_id=experiment_id,
        raw_h5=raw_h5,
        ood_dir=ood_dir,
    )
    artifacts = {
        "raw_h5": paths["raw_h5"],
        "metadata": paths["metadata"],
        "ood_split": f'{paths["ood_dir"]}/band_tensors_ood_split.npz',
        "ood_manifest": f'{paths["ood_dir"]}/ood_split_manifest.json',
        "ssl_encoder": paths["ssl_encoder"],
        "ssl_norm": paths["ssl_norm"],
        "finetuned_weights": paths["finetuned_weights"],
        "finetuned_config": paths["finetuned_weights"].replace(".weights.h5", "_config.json"),
        "metrics": f'{paths["finetune_report_dir"]}/metrics_summary.json',
        "validation_report": paths["validation_report"],
        "vision_detector": f'{paths["vision_model_dir"]}/band_plot_yolov8_pose_best.pt',
        "vision_detector_summary": f'{paths["vision_model_dir"]}/vision_detector_training_summary.json',
    }
    if raw_h5 is not None:
        artifacts["raw_snapshot_manifest"] = str(paths["raw_snapshot_manifest"])
    if ood_dir is not None:
        artifacts["tensor_snapshot_audit"] = str(paths["tensor_snapshot_audit"])
    if experiment_id is not None:
        artifacts.update(
            {
                "supervised_best": f'{paths["finetune_checkpoint_dir"]}/best.weights.h5',
                "supervised_last": f'{paths["finetune_checkpoint_dir"]}/last.weights.h5',
                "inner_selection_manifest": f'{paths["finetune_report_dir"]}/inner_selection_manifest.json',
                "ssl_best_index": f'{paths["ssl_checkpoint_dir"]}/ckpt-best.index',
                "ssl_best_data": f'{paths["ssl_checkpoint_dir"]}/ckpt-best.data-00000-of-00001',
                "ssl_last_index": f'{paths["ssl_checkpoint_dir"]}/ckpt-last.index',
                "ssl_last_data": f'{paths["ssl_checkpoint_dir"]}/ckpt-last.data-00000-of-00001',
                "ssl_history": f'{paths["ssl_log_dir"]}/ssl_history.json',
            }
        )
    return artifacts


def artifact_file_status(
    full_path: Path,
    *,
    display_path: str | None = None,
) -> Dict[str, object]:
    is_file = full_path.is_file()
    size = full_path.stat().st_size if is_file else None
    return {
        "path": display_path if display_path is not None else str(full_path),
        "exists": bool(is_file and size is not None and size > 0),
        "bytes": size,
    }


def validate_ssl_history_schema(
    history: Dict[str, object],
    *,
    expected_consistency_weight: float,
) -> Dict[str, int]:
    """Validate the machine-readable SSL selection history before acceptance."""
    if history.get("selection_monitor") != "val_total":
        raise RuntimeError("SSL history selection monitor is not val_total")
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise RuntimeError("SSL history has no completed epochs")
    epoch_numbers = [int(item.get("epoch", -1)) for item in epochs]
    if epoch_numbers != sorted(set(epoch_numbers)) or epoch_numbers[0] < 1:
        raise RuntimeError("SSL history epoch sequence is invalid")
    improved_epochs = []
    for item in epochs:
        val = item.get("val")
        if not isinstance(val, dict):
            raise RuntimeError("SSL history epoch is missing validation metrics")
        mask_fraction = float(val.get("mask_fraction", -1.0))
        if not 0.15 <= mask_fraction <= 0.30:
            raise RuntimeError(
                f"SSL validation mask fraction outside 15–30%: {mask_fraction}"
            )
        selection_weights = item.get("selection_weights")
        post_weights = item.get("post_adaptation_weights")
        if not isinstance(selection_weights, dict) or not isinstance(post_weights, dict):
            raise RuntimeError("SSL history is missing pre/post adaptation weights")
        if expected_consistency_weight == 0.0:
            for label, weights in (
                ("selection", selection_weights),
                ("post-adaptation", post_weights),
            ):
                if abs(float(weights.get("symmetry", 1.0))) > 1e-12:
                    raise RuntimeError(
                        f"SSL {label} consistency weight was revived from zero"
                    )
        if item.get("improved") is True:
            improved_epochs.append(int(item["epoch"]))
    if not improved_epochs:
        raise RuntimeError("SSL history has no best/improved epoch")
    return {
        "best_epoch": improved_epochs[-1],
        "last_epoch": epoch_numbers[-1],
    }


def artifact_status(
    source: str = "mp",
    *,
    experiment_id: str | None = None,
    raw_h5: str | None = None,
    ood_dir: str | None = None,
) -> Dict[str, Dict[str, object]]:
    status: Dict[str, Dict[str, object]] = {}
    artifacts = required_artifacts(
        source,
        experiment_id=experiment_id,
        raw_h5=raw_h5,
        ood_dir=ood_dir,
    )
    for name, path in artifacts.items():
        status[name] = artifact_file_status(
            rel(path),
            display_path=path,
        )
    return status


def validate_pipeline_artifact_gate(
    source: str,
    status: Dict[str, Dict[str, object]],
    *,
    skip_vision: bool,
    experiment_id: str | None = None,
    raw_h5: str | None = None,
    ood_dir: str | None = None,
) -> None:
    required_names = list(
        required_artifacts(
            source,
            experiment_id=experiment_id,
            raw_h5=raw_h5,
            ood_dir=ood_dir,
        )
    )
    if skip_vision:
        required_names = [
            name
            for name in required_names
            if name not in {"vision_detector", "vision_detector_summary"}
        ]
    missing = [
        name
        for name in required_names
        if not bool(status.get(name, {}).get("exists"))
    ]
    if missing:
        raise RuntimeError(
            "Pipeline artifact gate failed: missing required artifacts: "
            f"{missing}"
        )


def read_checkpoint_epoch(prefix: Path) -> int:
    """Read the completed epoch embedded in a TensorFlow checkpoint."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(str(prefix))
    return int(reader.get_tensor("epoch/.ATTRIBUTES/VARIABLE_VALUE"))


def validate_versioned_artifact_content(
    args: argparse.Namespace,
    paths: Dict[str, str],
) -> Dict[str, object]:
    """Validate selection hashes and SSL history/checkpoint consistency."""
    from scripts import finetune_supervised

    report_dir = rel(paths["finetune_report_dir"])
    selection = finetune_supervised.validate_inner_selection_manifest(
        str(report_dir)
    )
    history_path = rel(f'{paths["ssl_log_dir"]}/ssl_history.json')
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    history_info = validate_ssl_history_schema(
        history,
        expected_consistency_weight=float(args.consistency_weight),
    )
    best_epoch = read_checkpoint_epoch(
        rel(f'{paths["ssl_checkpoint_dir"]}/ckpt-best')
    )
    last_epoch = read_checkpoint_epoch(
        rel(f'{paths["ssl_checkpoint_dir"]}/ckpt-last')
    )
    if best_epoch != history_info["best_epoch"]:
        raise RuntimeError(
            f"SSL best checkpoint/history epoch mismatch: {best_epoch} != {history_info['best_epoch']}"
        )
    if last_epoch != history_info["last_epoch"]:
        raise RuntimeError(
            f"SSL last checkpoint/history epoch mismatch: {last_epoch} != {history_info['last_epoch']}"
        )
    return {
        "inner_selection": selection,
        "ssl_history": history_info,
    }


def write_model_brain_manifest(args: argparse.Namespace, status: Dict[str, Dict[str, object]]) -> Path:
    paths = pipeline_layout(
        args.source,
        experiment_id=getattr(args, "experiment_id", None),
        raw_h5=getattr(args, "raw_h5", None),
        ood_dir=getattr(args, "ood_dir", None),
    )
    metrics_path = rel(f'{paths["finetune_report_dir"]}/metrics_summary.json')
    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "description": "Physics-aware band-structure model brain",
        "source": args.source,
        "pipeline": [
            f"{args.source.upper()} download",
            "OOD tensor construction grouped by spacegroup_number",
            "Masked Band Modeling SSL pretraining",
            "Supervised fine-tuning for band gap and gap type",
            "Physics/statistical validation figures",
            "Phase-5 vision detector training for Plot-to-Physics panel and extremum localization",
        ],
        "ood_contract": {
            "group_key": "spacegroup_number",
            "train_size": args.train_size,
            "random_state": args.random_state,
            "target_k_points": args.target_k,
            "sample_wise_random_split": False,
        },
        "training_contract": {
            "ssl_mask_ratio": args.mask_ratio,
            "ssl_sign_weight": args.sign_weight,
            "ssl_consistency_weight": args.consistency_weight,
            "ssl_virtual_strain_scale": args.strain_scale,
            "freeze_layers": args.freeze_layers,
            "type_weight": args.type_weight,
            "head_learning_rate": args.learning_rate,
            "encoder_learning_rate": args.encoder_learning_rate,
            "topology_weight": args.topology_weight,
            "entropy_weight": args.entropy_weight,
            "extremum_weight": args.extremum_weight,
            "tensor_features": [
                "VBM_E",
                "VBM_curv",
                "VBM_k_dist",
                "CBM_E",
                "CBM_curv",
                "CBM_k_dist",
            ],
            "classification_loss": "categorical crossentropy with class sample weights",
            "regression_target": "tensor_gap_ecbm_minus_evbm",
            "gap_head": "sequence Conv1D extremum probability head with learned temperature and eV expected-value gap",
            "type_head": "pooled encoder features plus P_vbm/P_cbm topology priors",
            "energy_scale": "gap loss and MAE are computed after denormalizing VBM_E/CBM_E to eV",
        },
        "vision_training_contract": {
            "enabled": not args.skip_vision,
            "synthetic_count": args.vision_count,
            "epochs": args.vision_epochs,
            "batch_size": args.vision_batch,
            "image_size": args.vision_imgsz,
            "device": args.vision_device,
            "base_model": args.vision_base_model,
            "final_model": f'{paths["vision_model_dir"]}/band_plot_yolov8_pose_best.pt',
            "role": "vision front-end localizes band-plot panels and VBM/CBM anchors; the physics brain performs decisions",
        },
        "artifacts": status,
        "metrics": metrics,
        "protected_assets": PROTECTED_RELATIVE_PATHS,
    }

    output_path = rel(paths["manifest"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return output_path


def clean_for_fresh_run(
    source: str,
    clean_raw: bool,
    *,
    paths: Dict[str, str] | None = None,
    preserve_ood_input: bool = False,
    preserve_raw_input: bool = False,
) -> None:
    paths = paths or pipeline_layout(source)
    cleanup_paths = [
        rel(paths["ssl_encoder"]),
        rel(paths["ssl_norm"]),
        rel(paths["finetuned_weights"]),
        rel(paths["finetuned_weights"].replace(".weights.h5", "_config.json")),
        rel(paths["manifest"]),
        rel(paths["vision_model_dir"]),
        rel(paths["ssl_checkpoint_dir"]),
        rel(paths["finetune_checkpoint_dir"]),
        rel(paths["ssl_log_dir"]),
        rel(paths["finetune_report_dir"]),
        rel(paths["validation_report"]),
        rel(paths["vision_synthetic_dir"]),
        rel(paths["vision_runs_dir"]),
    ]
    if not preserve_ood_input:
        cleanup_paths.insert(0, rel(paths["ood_dir"]))
    for path in cleanup_paths:
        remove_path(path)
    for final_epoch_model in rel(paths["model_dir"]).glob("ssl_mbm_final_epoch*.keras"):
        remove_path(final_epoch_model)

    if clean_raw and not preserve_raw_input:
        for path in [
            rel(paths["raw_h5"]),
            rel(paths["metadata"]),
            rel(paths["download_report"]),
        ]:
            remove_path(path)
        if source == "mp":
            # MP JSON files historically live at the cache root. Keep AFLOW's
            # nested cache intact when the MP source is rebuilt.
            cache_root = rel(paths["json_cache"])
            if cache_root.exists():
                for cache_file in cache_root.glob("*.json"):
                    remove_path(cache_file)
        else:
            remove_path(rel(paths["json_cache"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full BandStructure AI training pipeline")
    parser.add_argument("--source", choices=("mp", "aflow"), default="mp")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="versioned experiment ID; required for new formal runs",
    )
    parser.add_argument(
        "--raw-h5",
        default=None,
        help="explicit immutable raw HDF5 input path",
    )
    parser.add_argument(
        "--ood-dir",
        default=None,
        help="explicit immutable processed tensor directory",
    )
    parser.add_argument(
        "--report-date",
        default=None,
        help="report date token in YYYYMMDD form",
    )
    parser.add_argument("--target", type=int, default=200, help="minimum total cached sample count")
    parser.add_argument("--target-k", type=int, default=128)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--max-sites", type=int, default=50)
    parser.add_argument("--download-workers", type=int, default=1)

    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ssl-batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--sign-weight", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dff", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--strain-scale", type=float, default=0.01)
    parser.add_argument(
        "--disable-strain-augmentation",
        dest="disable_strain_augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--enable-strain-augmentation",
        dest="disable_strain_augmentation",
        action="store_false",
    )
    parser.set_defaults(disable_strain_augmentation=True)

    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="head optimizer learning rate")
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--type-weight", type=float, default=2.0)
    parser.add_argument("--freeze-layers", type=int, default=2)
    parser.add_argument("--topology-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--extremum-weight", type=float, default=1.0)

    parser.add_argument("--fresh", action="store_true", help="rebuild processed/training artifacts")
    parser.add_argument("--clean-raw", action="store_true", help="with --fresh, also remove raw downloaded data")
    parser.add_argument("--force-download", action="store_true", help="run downloader even when enough raw samples exist")
    parser.add_argument("--force-ssl", action="store_true", help="rerun SSL even when SSL artifacts exist")
    parser.add_argument("--force-finetune", action="store_true", help="rerun fine-tuning even when final artifacts exist")
    parser.add_argument("--force-vision", action="store_true", help="rerun vision-detector training even when final artifacts exist")
    parser.add_argument("--skip-vision", action="store_true", help="skip Phase-5 vision-detector training")
    parser.add_argument("--status-only", action="store_true", help="only print artifact status and write manifest")
    parser.add_argument("--require-gpu", action="store_true")

    parser.add_argument("--vision-count", type=int, default=600)
    parser.add_argument("--vision-epochs", type=int, default=8)
    parser.add_argument("--vision-batch", type=int, default=4)
    parser.add_argument("--vision-imgsz", type=int, default=512)
    parser.add_argument("--vision-val-fraction", type=float, default=0.2)
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--vision-base-model", default="yolov8n-pose.pt")
    parser.add_argument("--vision-min-box-map50", type=float, default=0.75)
    parser.add_argument("--vision-min-pose-map50", type=float, default=0.50)
    return parser.parse_args()


def should_run_download(
    *,
    force_download: bool,
    raw_count: int,
    target: int,
    explicit_raw_h5: bool,
) -> bool:
    """Protect caller-declared immutable raw snapshots from downloader writes."""
    if explicit_raw_h5:
        if force_download or raw_count < target:
            raise RuntimeError(
                "Explicit immutable raw snapshot cannot be downloaded into or extended"
            )
        return False
    return bool(force_download or raw_count < target)


def should_build_ood_tensors(
    *,
    fresh: bool,
    ood_split_exists: bool,
    explicit_ood_dir: bool,
) -> bool:
    """Never rebuild or create a caller-declared immutable OOD input."""
    if explicit_ood_dir:
        if not ood_split_exists:
            raise FileNotFoundError(
                "Explicit immutable OOD split is missing; refusing to rebuild it"
            )
        return False
    return bool(fresh or not ood_split_exists)


def main() -> None:
    args = parse_args()

    if not (0.15 <= args.mask_ratio <= 0.30):
        raise ValueError("--mask-ratio must stay between 0.15 and 0.30")

    experiment_id = getattr(args, "experiment_id", None)
    raw_h5_override = getattr(args, "raw_h5", None)
    ood_dir_override = getattr(args, "ood_dir", None)
    paths = pipeline_layout(
        args.source,
        experiment_id=experiment_id,
        raw_h5=raw_h5_override,
        ood_dir=ood_dir_override,
    )
    if args.fresh:
        clean_for_fresh_run(
            source=args.source,
            clean_raw=args.clean_raw,
            paths=paths,
            preserve_ood_input=ood_dir_override is not None,
            preserve_raw_input=raw_h5_override is not None,
        )

    status = artifact_status(
        args.source,
        experiment_id=experiment_id,
        raw_h5=raw_h5_override,
        ood_dir=ood_dir_override,
    )
    if args.status_only:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    raw_h5 = paths["raw_h5"]
    raw_metadata = paths["metadata"]
    ood_split = f'{paths["ood_dir"]}/band_tensors_ood_split.npz'
    raw_count = count_h5_samples(rel(raw_h5))
    download_was_run = False
    if should_run_download(
        force_download=args.force_download,
        raw_count=raw_count,
        target=args.target,
        explicit_raw_h5=raw_h5_override is not None,
    ):
        run_command(
            [
                sys.executable,
                "-m",
                "src.data.batch_download",
                "--source",
                args.source,
                "--target",
                str(args.target),
                "--min-gap",
                str(args.min_gap),
                "--max-gap",
                str(args.max_gap),
                "--max-sites",
                str(args.max_sites),
                "--workers",
                str(args.download_workers),
            ],
            f"Download {args.source.upper()} band structures",
        )
        download_was_run = True
    else:
        print(f"[Skip] raw download: {raw_count} samples already available")

    validate_download_gate(
        paths,
        args.target,
        require_report=download_was_run,
    )

    if should_build_ood_tensors(
        fresh=args.fresh,
        ood_split_exists=rel(ood_split).exists(),
        explicit_ood_dir=ood_dir_override is not None,
    ):
        run_command(
            [
                sys.executable,
                "scripts/build_ood_tensors.py",
                "--h5",
                raw_h5,
                "--metadata",
                raw_metadata,
                "--output",
                paths["ood_dir"],
                "--target-k",
                str(args.target_k),
                "--train-size",
                str(args.train_size),
                "--random-state",
                str(args.random_state),
            ],
            "Build cleaned OOD tensors",
        )
    else:
        print("[Skip] OOD tensor build: artifact already exists")

    ssl_ready = rel(paths["ssl_encoder"]).exists() and rel(paths["ssl_norm"]).exists()
    if args.force_ssl or args.fresh or not ssl_ready:
        run_command(
            build_ssl_command(args, paths, ood_split),
            "SSL Masked Band Modeling pretraining",
        )
    else:
        print("[Skip] SSL pretraining: artifacts already exist")

    finetune_ready = (
        rel(paths["finetuned_weights"]).exists()
        and rel(f'{paths["finetune_report_dir"]}/metrics_summary.json').exists()
    )
    if args.force_finetune or args.fresh or not finetune_ready:
        run_supervised_stages(args, paths, ood_split)
    else:
        print("[Skip] fine-tuning: artifacts already exist")

    vision_ready = (
        rel(f'{paths["vision_model_dir"]}/band_plot_yolov8_pose_best.pt').exists()
        and rel(f'{paths["vision_model_dir"]}/vision_detector_training_summary.json').exists()
    )
    if args.skip_vision:
        print("[Skip] vision detector training: disabled by --skip-vision")
    elif args.force_vision or args.fresh or not vision_ready:
        command = [
            sys.executable,
            "scripts/train_vision_detector.py",
            "--h5",
            raw_h5,
            "--synthetic-dir",
            paths["vision_synthetic_dir"],
            "--model-dir",
            paths["vision_model_dir"],
            "--runs-dir",
            paths["vision_runs_dir"],
            "--count",
            str(args.vision_count),
            "--epochs",
            str(args.vision_epochs),
            "--batch",
            str(args.vision_batch),
            "--imgsz",
            str(args.vision_imgsz),
            "--val-fraction",
            str(args.vision_val_fraction),
            "--random-state",
            str(args.random_state),
            "--base-model",
            args.vision_base_model,
            "--device",
            args.vision_device,
            "--min-box-map50",
            str(args.vision_min_box_map50),
            "--min-pose-map50",
            str(args.vision_min_pose_map50),
        ]
        if args.force_vision or args.fresh:
            command.append("--fresh")
        run_command(command, "Phase-5 vision detector training")
    else:
        print("[Skip] vision detector training: artifacts already exist")

    status = artifact_status(
        args.source,
        experiment_id=experiment_id,
        raw_h5=raw_h5_override,
        ood_dir=ood_dir_override,
    )
    validate_pipeline_artifact_gate(
        args.source,
        status,
        skip_vision=args.skip_vision,
        experiment_id=experiment_id,
        raw_h5=raw_h5_override,
        ood_dir=ood_dir_override,
    )
    if experiment_id is not None:
        validate_versioned_artifact_content(args, paths)
    manifest_path = write_model_brain_manifest(args, status)

    print("\n" + "=" * 80)
    print("Pipeline complete")
    print("=" * 80)
    print(f"Model brain manifest: {manifest_path}")
    print("All required model-brain artifacts are present.")


if __name__ == "__main__":
    main()
