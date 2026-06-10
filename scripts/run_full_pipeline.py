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
from pathlib import Path
from typing import Dict, List

import h5py


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


def remove_path(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if not str(resolved).startswith(str(root)):
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
    with h5py.File(h5_path, "r") as f:
        return len(f.keys())


def required_artifacts() -> Dict[str, str]:
    return {
        "raw_h5": "data_cache/mp_bands.h5",
        "metadata": "data_cache/mp_metadata.json",
        "ood_split": "data_cache/ood_tensors/band_tensors_ood_split.npz",
        "ood_manifest": "data_cache/ood_tensors/ood_split_manifest.json",
        "ssl_encoder": "models/ssl_mbm_pretrained.keras",
        "ssl_norm": "models/ssl_mbm_norm_stats.json",
        "finetuned_weights": "models/finetuned_gap_predictor.weights.h5",
        "finetuned_config": "models/finetuned_gap_predictor_config.json",
        "metrics": "reports/finetune_supervised/metrics_summary.json",
        "validation_report": "reports/finetune_supervised_report_20260604.md",
        "vision_detector": "models/vision_detector/band_plot_yolov8_pose_best.pt",
        "vision_detector_summary": "models/vision_detector/vision_detector_training_summary.json",
    }


def artifact_status() -> Dict[str, Dict[str, object]]:
    status: Dict[str, Dict[str, object]] = {}
    for name, path in required_artifacts().items():
        full = rel(path)
        status[name] = {
            "path": path,
            "exists": full.exists(),
            "bytes": full.stat().st_size if full.exists() and full.is_file() else None,
        }
    return status


def write_model_brain_manifest(args: argparse.Namespace, status: Dict[str, Dict[str, object]]) -> Path:
    metrics_path = rel("reports/finetune_supervised/metrics_summary.json")
    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "description": "Physics-aware band-structure model brain",
        "pipeline": [
            "Materials Project download",
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
            "final_model": "models/vision_detector/band_plot_yolov8_pose_best.pt",
            "role": "vision front-end localizes band-plot panels and VBM/CBM anchors; the physics brain performs decisions",
        },
        "artifacts": status,
        "metrics": metrics,
        "protected_assets": PROTECTED_RELATIVE_PATHS,
    }

    output_path = rel("models/physics_model_brain_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return output_path


def clean_for_fresh_run(clean_raw: bool) -> None:
    for path in [
        rel("data_cache/ood_tensors"),
        rel("models/ssl_mbm_pretrained.keras"),
        rel("models/ssl_mbm_final_epoch100.keras"),
        rel("models/ssl_mbm_norm_stats.json"),
        rel("models/finetuned_gap_predictor.weights.h5"),
        rel("models/finetuned_gap_predictor_config.json"),
        rel("models/physics_model_brain_manifest.json"),
        rel("checkpoints/ssl_mbm"),
        rel("checkpoints/finetune_supervised"),
        rel("logs/ssl_mbm"),
        rel("reports/finetune_supervised"),
        rel("reports/finetune_supervised_report_20260604.md"),
    ]:
        remove_path(path)

    if clean_raw:
        for path in [
            rel("data_cache/mp_bands.h5"),
            rel("data_cache/mp_metadata.json"),
            rel("data_cache/download_report.json"),
            rel("data_cache/json_cache"),
        ]:
            remove_path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full BandStructure AI training pipeline")
    parser.add_argument("--target", type=int, default=200, help="minimum raw downloaded sample count")
    parser.add_argument("--target-k", type=int, default=128)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-gap", type=float, default=0.1)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--max-sites", type=int, default=50)

    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ssl-batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--sign-weight", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dff", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--strain-scale", type=float, default=0.01)
    parser.add_argument("--disable-strain-augmentation", action="store_true")

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


def main() -> None:
    args = parse_args()

    if not (0.15 <= args.mask_ratio <= 0.30):
        raise ValueError("--mask-ratio must stay between 0.15 and 0.30")

    if args.fresh:
        clean_for_fresh_run(clean_raw=args.clean_raw)

    status = artifact_status()
    if args.status_only:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        manifest_path = write_model_brain_manifest(args, status)
        print(f"Model brain manifest: {manifest_path}")
        return

    raw_count = count_h5_samples(rel("data_cache/mp_bands.h5"))
    if args.force_download or raw_count < args.target:
        run_command(
            [
                sys.executable,
                "-m",
                "src.data.batch_download",
                "--target",
                str(args.target),
                "--min-gap",
                str(args.min_gap),
                "--max-gap",
                str(args.max_gap),
                "--max-sites",
                str(args.max_sites),
            ],
            "Download Materials Project band structures",
        )
    else:
        print(f"[Skip] raw download: {raw_count} samples already available")

    if args.fresh or not rel("data_cache/ood_tensors/band_tensors_ood_split.npz").exists():
        run_command(
            [
                sys.executable,
                "scripts/build_ood_tensors.py",
                "--h5",
                "data_cache/mp_bands.h5",
                "--metadata",
                "data_cache/mp_metadata.json",
                "--output",
                "data_cache/ood_tensors",
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

    ssl_ready = rel("models/ssl_mbm_pretrained.keras").exists() and rel("models/ssl_mbm_norm_stats.json").exists()
    if args.force_ssl or args.fresh or not ssl_ready:
        command = [
            sys.executable,
            "scripts/train_ssl.py",
            "--epochs",
            str(args.ssl_epochs),
            "--batch-size",
            str(args.ssl_batch_size),
            "--mask-ratio",
            str(args.mask_ratio),
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
            "checkpoints/ssl_mbm",
            "--log-dir",
            "logs/ssl_mbm",
            "--model-dir",
            "models",
        ]
        if args.require_gpu:
            command.append("--require-gpu")
        if args.disable_strain_augmentation:
            command.append("--disable-strain-augmentation")
        run_command(command, "SSL Masked Band Modeling pretraining")
    else:
        print("[Skip] SSL pretraining: artifacts already exist")

    finetune_ready = (
        rel("models/finetuned_gap_predictor.weights.h5").exists()
        and rel("reports/finetune_supervised/metrics_summary.json").exists()
    )
    if args.force_finetune or args.fresh or not finetune_ready:
        run_command(
            [
                sys.executable,
                "scripts/finetune_supervised.py",
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
                "reports/finetune_supervised",
                "--checkpoint-dir",
                "checkpoints/finetune_supervised",
                "--model-path",
                "models/finetuned_gap_predictor.weights.h5",
            ],
            "Supervised fine-tuning and validation",
        )
    else:
        print("[Skip] fine-tuning: artifacts already exist")

    vision_ready = (
        rel("models/vision_detector/band_plot_yolov8_pose_best.pt").exists()
        and rel("models/vision_detector/vision_detector_training_summary.json").exists()
    )
    if args.skip_vision:
        print("[Skip] vision detector training: disabled by --skip-vision")
    elif args.force_vision or args.fresh or not vision_ready:
        command = [
            sys.executable,
            "scripts/train_vision_detector.py",
            "--h5",
            "data_cache/mp_bands.h5",
            "--synthetic-dir",
            "data_cache/vision_synthetic_train",
            "--model-dir",
            "models/vision_detector",
            "--runs-dir",
            "runs/vision_detector",
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

    status = artifact_status()
    missing = [name for name, item in status.items() if not item["exists"]]
    manifest_path = write_model_brain_manifest(args, status)

    print("\n" + "=" * 80)
    print("Pipeline complete")
    print("=" * 80)
    print(f"Model brain manifest: {manifest_path}")
    if missing:
        print(f"[WARN] Missing artifacts: {missing}")
    else:
        print("All required model-brain artifacts are present.")


if __name__ == "__main__":
    main()
