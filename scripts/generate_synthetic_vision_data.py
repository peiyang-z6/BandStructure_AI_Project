"""Generate Stage-1 synthetic Sim2Real vision data from `mp_bands.h5`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.synthetic_data_generator import SyntheticBandPlotGenerator, SyntheticConfig, train_yolov8_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate degraded synthetic band-plot data with YOLO/COCO labels")
    parser.add_argument("--h5", default="data/raw/materials_project/mp_bands.h5")
    parser.add_argument("--output-dir", default="data/processed/vision/synthetic")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-yolo", action="store_true", help="Optionally run YOLOv8 pose training if ultralytics is installed")
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = SyntheticBandPlotGenerator(
        SyntheticConfig(
            h5_path=args.h5,
            output_dir=args.output_dir,
            count=args.count,
            random_state=args.random_state,
        )
    ).generate()
    print(json.dumps(summary, indent=2))
    if args.train_yolo:
        train_yolov8_pose(str(Path(args.output_dir) / "dataset.yaml"), epochs=args.epochs)


if __name__ == "__main__":
    main()
