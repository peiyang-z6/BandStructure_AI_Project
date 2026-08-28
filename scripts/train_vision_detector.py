"""Train and validate the Phase-5 vision detector on synthetic Sim2Real data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.synthetic_data_generator import SyntheticBandPlotGenerator, SyntheticConfig


def clean_previous(output_dir: Path, model_dir: Path, runs_dir: Path) -> None:
    for path in (output_dir, model_dir, runs_dir):
        if path.exists():
            shutil.rmtree(path)
    model_dir.mkdir(parents=True, exist_ok=True)


def train_and_validate(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install YOLO dependency first: pip install ultralytics") from exc

    output_dir = Path(args.synthetic_dir)
    model_dir = Path(args.model_dir)
    runs_dir = Path(args.runs_dir)
    if args.fresh:
        clean_previous(output_dir, model_dir, runs_dir)

    summary = SyntheticBandPlotGenerator(
        SyntheticConfig(
            h5_path=args.h5,
            output_dir=str(output_dir),
            count=args.count,
            image_size=(args.imgsz, args.imgsz),
            random_state=args.random_state,
            val_fraction=args.val_fraction,
        )
    ).generate()

    model = YOLO(args.base_model)
    train_results = model.train(
        data=str(output_dir / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name="band_plot_pose",
        exist_ok=True,
        patience=max(args.epochs // 3, 3),
        workers=0,
        verbose=True,
    )
    best_path = Path(train_results.save_dir) / "weights" / "best.pt"
    last_path = Path(train_results.save_dir) / "weights" / "last.pt"

    trained = YOLO(str(best_path if best_path.exists() else last_path))
    metrics = trained.val(data=str(output_dir / "dataset.yaml"), imgsz=args.imgsz, batch=args.batch, device=args.device)
    box_map50 = float(getattr(metrics.box, "map50", 0.0))
    pose_map50 = float(getattr(metrics.pose, "map50", 0.0)) if hasattr(metrics, "pose") else 0.0
    passed = box_map50 >= args.min_box_map50 and pose_map50 >= args.min_pose_map50

    final_model = model_dir / "band_plot_yolov8_pose_best.pt"
    shutil.copy2(best_path if best_path.exists() else last_path, final_model)
    report = {
        "synthetic": summary,
        "final_model": str(final_model),
        "box_map50": box_map50,
        "pose_map50": pose_map50,
        "passed": passed,
        "thresholds": {"box_map50": args.min_box_map50, "pose_map50": args.min_pose_map50},
        "train_save_dir": str(train_results.save_dir),
    }
    (model_dir / "vision_detector_training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(
            f"Vision detector did not meet thresholds: box_mAP50={box_map50:.3f}, pose_mAP50={pose_map50:.3f}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO pose detector for band plot panels and extrema")
    parser.add_argument("--h5", default="data/raw/materials_project/mp_bands.h5")
    parser.add_argument("--synthetic-dir", default="data/processed/vision/synthetic_train")
    parser.add_argument("--model-dir", default="artifacts/models/vision_detector")
    parser.add_argument("--runs-dir", default="artifacts/logs/vision_detector")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--base-model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-box-map50", type=float, default=0.75)
    parser.add_argument("--min-pose-map50", type=float, default=0.50)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train_and_validate(parse_args()), indent=2))
