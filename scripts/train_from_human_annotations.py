#!/usr/bin/env python3
"""
Human-Annotation-to-YOLO Training Pipeline (Plan A)
===================================================

Converts human-drawn GUI annotations into YOLOv8 pose training data,
then fine-tunes the existing band-plot detector on real human labels.

This script CLOSES THE TRAINING LOOP:
  GUI Training Mode → JSON annotations → YOLO dataset → Fine-tuned model

Workflow:
  1. Read annotation JSONs from data_cache/human_annotations/
  2. For each annotation with a valid source image + canvas metadata:
     a. Resize source image to match canvas dimensions (annotations are in canvas coords)
     b. Normalize panel bbox and VBM/CBM keypoints to YOLO [0,1] format
     c. Create YOLO label file
  3. Write data.yaml config
  4. Fine-tune band_plot_yolov8_pose_best.pt on human labels
  5. Save the new model checkpoint

Usage:
  python scripts/train_from_human_annotations.py
    --epochs 50 --batch 4 --lr 0.0001

  # Dry-run: convert annotations only, skip training
  python scripts/train_from_human_annotations.py --dry-run

  # Full training with custom params
  python scripts/train_from_human_annotations.py --epochs 100 --freeze 10 --augment
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────
ANNOTATIONS_DIR = ROOT / "data_cache" / "human_annotations"
YOLO_DATASET_DIR = ROOT / "data_cache" / "yolo_human_train"
BASE_MODEL = ROOT / "models" / "vision_detector" / "band_plot_yolov8_pose_best.pt"
OUTPUT_MODEL_DIR = ROOT / "models" / "vision_detector"

# YOLO class: band_plot panel
CLASS_ID = 0
CLASS_NAME = "band_plot"
KPT_SHAPE = [2, 3]  # 2 keypoints (VBM, CBM), each with (x, y, visibility)


# ── Annotation loading ─────────────────────────────────────────────────────

def load_annotations(annotations_dir: Path) -> List[Dict[str, Any]]:
    """Load all valid training annotations with source image info."""
    records: List[Dict[str, Any]] = []
    manifest = annotations_dir / "training_manifest.jsonl"
    sources = [manifest] if manifest.exists() else []

    # Also scan individual JSON files
    for json_file in sorted(annotations_dir.glob("*.json")):
        if json_file.name == "training_manifest.jsonl":
            continue
        sources.append(json_file)

    seen_ids: set = set()
    for source in sources:
        try:
            if source.suffix == ".jsonl":
                for line in source.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record_id = rec.get("record_id", "")
                    if record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)
                    records.append(rec)
            else:
                rec = json.loads(source.read_text(encoding="utf-8"))
                record_id = rec.get("record_id", source.stem)
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                records.append(rec)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Skipping {source.name}: {exc}")

    # Filter: must have source_image_path and annotations
    valid: List[Dict[str, Any]] = []
    for rec in records:
        src = rec.get("source_image_path", "")
        if not src or not Path(src).exists():
            print(f"  [SKIP] {rec.get('record_id', '?')}: source image not found ({src})")
            continue
        ann = rec.get("annotations", {})
        if not ann.get("panel"):
            print(f"  [SKIP] {rec.get('record_id', '?')}: missing panel bbox")
            continue
        if not ann.get("vbm") or not ann.get("cbm"):
            print(f"  [SKIP] {rec.get('record_id', '?')}: missing VBM/CBM keypoints")
            continue
        cw = rec.get("canvas_width", 0)
        ch = rec.get("canvas_height", 0)
        if cw <= 0 or ch <= 0:
            print(f"  [SKIP] {rec.get('record_id', '?')}: invalid canvas dimensions ({cw}x{ch})")
            continue
        valid.append(rec)

    return valid


# ── Coordinate conversion ──────────────────────────────────────────────────

def _norm(value: float, dim: float) -> float:
    """Normalize a coordinate to [0, 1]."""
    return max(0.0, min(1.0, value / max(dim, 1.0)))


def annotation_to_yolo_label(rec: Dict[str, Any]) -> Optional[str]:
    """Convert a single annotation record to a YOLOv8 pose label line."""
    ann = rec["annotations"]
    cw = float(rec["canvas_width"])
    ch = float(rec["canvas_height"])

    # Panel bbox → (cx, cy, w, h) normalized
    panel = ann["panel"]
    px, py = float(panel["x"]), float(panel["y"])
    pw, ph = float(panel["w"]), float(panel["h"])

    cx = _norm(px + pw / 2.0, cw)
    cy = _norm(py + ph / 2.0, ch)
    nw = _norm(pw, cw)
    nh = _norm(ph, ch)

    # VBM keypoint (visibility=2 → visible and labeled)
    vbm = ann["vbm"]
    vbm_x = _norm(float(vbm["x"]), cw)
    vbm_y = _norm(float(vbm["y"]), ch)

    # CBM keypoint
    cbm = ann["cbm"]
    cbm_x = _norm(float(cbm["x"]), cw)
    cbm_y = _norm(float(cbm["y"]), ch)

    # YOLO pose label: class cx cy w h kp1_x kp1_y vis kp2_x kp2_y vis
    label = (
        f"{CLASS_ID} "
        f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} "
        f"{vbm_x:.6f} {vbm_y:.6f} 2 "
        f"{cbm_x:.6f} {cbm_y:.6f} 2"
    )
    return label


# ── Dataset builder ─────────────────────────────────────────────────────────

def build_yolo_dataset(
    records: List[Dict[str, Any]],
    output_dir: Path,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[int, int]:
    """Convert annotation records into a YOLOv8 pose dataset on disk."""
    # Clean and recreate
    if output_dir.exists():
        shutil.rmtree(output_dir)

    images_train = output_dir / "images" / "train"
    labels_train = output_dir / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    # If we have enough records, create a val split
    rng = np.random.RandomState(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(records) * val_split))
    val_idx = set(indices[:n_val])

    has_val = len(records) >= 5
    if has_val:
        images_val = output_dir / "images" / "val"
        labels_val = output_dir / "labels" / "val"
        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

    train_count, val_count = 0, 0
    for i, rec in enumerate(records):
        record_id = rec.get("record_id", f"sample_{i}")
        src_path = Path(rec["source_image_path"])
        cw = int(rec["canvas_width"])
        ch = int(rec["canvas_height"])

        # Resize source image to match canvas dimensions
        try:
            img = Image.open(src_path).convert("RGB")
            if img.size != (cw, ch):
                img = img.resize((cw, ch), Image.LANCZOS)
        except Exception as exc:
            print(f"  [WARN] Cannot open {src_path}: {exc}")
            continue

        # Generate YOLO label
        label = annotation_to_yolo_label(rec)
        if label is None:
            continue

        is_val = (i in val_idx) and has_val
        if is_val:
            img_dst = images_val
            lbl_dst = labels_val
            val_count += 1
        else:
            img_dst = images_train
            lbl_dst = labels_train
            train_count += 1

        # Use record_id as filename stem (sanitize)
        safe_stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in record_id)[:80]
        img_out = img_dst / f"{safe_stem}.jpg"
        lbl_out = lbl_dst / f"{safe_stem}.txt"

        img.save(img_out, quality=95)
        lbl_out.write_text(label + "\n", encoding="utf-8")

    # Write data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        f"train: images/train",
    ]
    if has_val:
        yaml_lines.append(f"val: images/val")
    else:
        yaml_lines.append(f"val: images/train")  # fallback
    yaml_lines += [
        "",
        f"kpt_shape: {KPT_SHAPE}",
        "names:",
        f"  0: {CLASS_NAME}",
        "",
        "# Generated by train_from_human_annotations.py",
        f"# Records: {len(records)}, Train: {train_count}, Val: {val_count}",
    ]
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")

    return train_count, val_count


# ── Training ────────────────────────────────────────────────────────────────

def train_yolo(
    data_yaml: Path,
    base_model: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 4,
    image_size: int = 640,
    learning_rate: float = 1e-4,
    freeze_layers: int = 0,
    augment: bool = False,
    workers: int = 2,
    device: str = "",
) -> Path:
    """Fine-tune YOLOv8 pose model on human-labeled data."""
    from ultralytics import YOLO

    print(f"\n  Loading base model: {base_model}")
    model = YOLO(str(base_model))

    # Training args
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": image_size,
        "lr0": learning_rate,
        "lrf": learning_rate * 0.1,
        "freeze": freeze_layers,
        "workers": workers,
        "device": device,
        "project": str(output_dir),
        "name": "human_finetune",
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "seed": 42,
    }

    if augment:
        train_args.update({
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.3,
            "shear": 2.0,
            "perspective": 0.0005,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,   # off for small datasets
            "mixup": 0.0,    # off for small datasets
        })

    print(f"\n  Starting training ({epochs} epochs, batch={batch_size}, lr={learning_rate})")
    results = model.train(**train_args)

    # Copy best model to output dir
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    output_model = output_dir / "band_plot_yolov8_human_finetuned.pt"
    if best_pt.exists():
        shutil.copy(best_pt, output_model)
        print(f"\n  Best model saved: {output_model}")

    return output_model


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 band-plot detector from GUI human annotations"
    )
    parser.add_argument(
        "--annotations", type=str,
        default=str(ANNOTATIONS_DIR),
        help="Directory containing annotation JSON files"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(YOLO_DATASET_DIR),
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--base-model", type=str,
        default=str(BASE_MODEL),
        help="Base YOLOv8 pose model to fine-tune"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Batch size (default: 4; reduce for small GPU)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 0.0001, low for fine-tuning)"
    )
    parser.add_argument(
        "--freeze", type=int, default=0,
        help="Freeze first N backbone layers (default: 0)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size (default: 640)"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable light data augmentation (recommended for >=20 samples)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Validation split fraction (default: 0.15)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Convert annotations only; skip training"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader workers (default: 2)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: 'cpu', '0', 'cuda:0', etc. (default: auto)"
    )

    args = parser.parse_args()

    annotations_dir = Path(args.annotations)
    output_dir = Path(args.output)
    base_model = Path(args.base_model)

    # ── Step 1: Load annotations ────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading human annotations")
    print("=" * 60)
    records = load_annotations(annotations_dir)
    print(f"  Loaded {len(records)} valid annotation records.\n")

    if len(records) == 0:
        print(
            "  No valid annotations found.\n\n"
            "  To create training data:\n"
            "    1. Run: python scripts/gui_workbench.py\n"
            "    2. Upload a band-structure image\n"
            "    3. Switch to 'Training Annotation Mode' tab\n"
            "    4. Draw all annotations (panel, Fermi, VB/CB, VBM/CBM, axes)\n"
            "    5. Fill in Material ID and click 'Save Training Annotation'\n"
            "    6. Repeat for 10-30 images, then re-run this script\n"
        )
        return

    # ── Step 2: Build YOLO dataset ──────────────────────────────────────
    print("=" * 60)
    print("Step 2: Building YOLO dataset")
    print("=" * 60)
    train_n, val_n = build_yolo_dataset(
        records, output_dir, val_split=args.val_split
    )
    print(f"  Train: {train_n} images, Val: {val_n} images")
    print(f"  Dataset: {output_dir}\n")

    if args.dry_run:
        print("  Dry-run complete. Skipping training.")
        print(f"\n  To train manually:\n    yolo pose train data={output_dir}/data.yaml model={base_model} epochs={args.epochs}")
        return

    # ── Step 3: Check base model exists ─────────────────────────────────
    if not base_model.exists():
        print(f"\n  [ERROR] Base model not found: {base_model}")
        print("  Train on synthetic data first, or provide a different --base-model.")
        sys.exit(1)

    # ── Step 4: Train ───────────────────────────────────────────────────
    print("=" * 60)
    print("Step 3: Fine-tuning YOLOv8 pose model")
    print("=" * 60)

    data_yaml = output_dir / "data.yaml"
    final_model = train_yolo(
        data_yaml=data_yaml,
        base_model=base_model,
        output_dir=OUTPUT_MODEL_DIR,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        learning_rate=args.lr,
        freeze_layers=args.freeze,
        augment=args.augment,
        workers=args.workers,
        device=args.device,
    )

    # ── Step 5: Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Annotations used : {len(records)}")
    print(f"  Training images  : {train_n}")
    print(f"  Validation images: {val_n}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Base model      : {base_model}")
    print(f"  Output model    : {final_model}")
    print(f"  Dataset         : {output_dir}")
    print()
    print("  Next steps:")
    print(f"    1. Copy {final_model.name} to models/vision_detector/")
    print(f"    2. Update DETECTOR_PATH in gui_workbench.py if needed")
    print(f"    3. Re-run the GUI to use the human-fine-tuned detector")


if __name__ == "__main__":
    main()
