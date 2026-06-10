"""CV locator-only demo for Phase 5.

This script deliberately does not infer Fermi level, VBM/CBM, band gap, or
direct/indirect type. It only locates the band-plot panel and saves the raw
skeleton overlay. Use `scripts/gui_workbench.py` for human calibration and
Physics Brain inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.vision import MultiFormatParser


def normalize_dragged_path(path_text: str) -> str:
    text = str(path_text).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    if text.lower().startswith("file:///"):
        text = text[8:]
    return text.strip()


def save_overlay(parsed: Any, output_dir: Path) -> str | None:
    if parsed.overlay_image is None:
        return None
    try:
        import cv2
        import numpy as np

        out = output_dir / "cv_locator_skeleton_overlay.png"
        ok, encoded = cv2.imencode(".png", parsed.overlay_image)
        if ok:
            encoded.tofile(str(out))
            return str(out)
    except Exception:
        return None
    return None


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(normalize_dragged_path(args.input)).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = MultiFormatParser(
        target_k_points=args.target_k_points,
        detector_path=args.detector,
        detector_conf=args.detector_conf,
    )
    parsed = parser.parse(str(input_path))
    overlay = save_overlay(parsed, output_dir)
    report = output_dir / "cv_locator_report.md"
    report.write_text(
        "\n".join(
            [
                "# CV Locator Report",
                "",
                "CV inference is limited to plot-panel localization and raw skeleton extraction.",
                "Human GUI calibration is required before Physics Brain inference.",
                "",
                f"- Source: `{input_path}`",
                f"- Panel detection: `{parsed.metadata.get('panel_detection')}`",
                f"- Panel bbox: `{parsed.metadata.get('panel_bbox')}`",
                f"- Skeleton pixels: `{parsed.metadata.get('skeleton_pixels')}`",
                f"- Mask mode: `{parsed.metadata.get('mask_mode')}`",
                f"- Overlay: `{overlay}`",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "report": str(report),
        "figures": {"skeleton_overlay": overlay},
        "vision": {
            "panel_detection": parsed.metadata.get("panel_detection"),
            "detector_path": parsed.metadata.get("vision_detector_path"),
            "detector_confidence": parsed.metadata.get("vision_detector_confidence"),
            "panel_bbox": parsed.metadata.get("panel_bbox"),
            "skeleton_pixels": parsed.metadata.get("skeleton_pixels"),
            "mask_mode": parsed.metadata.get("mask_mode"),
            "requires_human_calibration": True,
            "cv_physics_inference": "disabled",
        },
        "prediction": None,
        "reconstructed": None,
        "next_step": "Run scripts/gui_workbench.py and perform human calibration.",
    }
    (output_dir / "cv_locator_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 CV locator demo")
    parser.add_argument("input", nargs="?", help="Local PDF/PNG/JPG/MP4/GIF path. Drag-and-drop is supported.")
    parser.add_argument("--output-dir", default="reports/vision_pipeline")
    parser.add_argument("--target-k-points", type=int, default=128)
    parser.add_argument("--detector", default="models/vision_detector/band_plot_yolov8_pose_best.pt")
    parser.add_argument("--detector-conf", type=float, default=0.05)
    args = parser.parse_args()
    if not args.input:
        try:
            args.input = input("Drag a band-plot file here and press Enter: ")
        except EOFError as exc:
            raise SystemExit("No input path received.") from exc
    return args


def print_terminal_summary(result: Dict[str, Any]) -> None:
    vision = result.get("vision", {})
    print("\n=== CV Locator Result ===")
    print(f"Panel source: {vision.get('panel_detection') or 'unknown'}")
    if vision.get("detector_confidence") is not None:
        print(f"Detector confidence: {vision['detector_confidence']:.4f}")
    print(f"Panel bbox: {vision.get('panel_bbox')}")
    print(f"Skeleton pixels: {vision.get('skeleton_pixels')}")
    print("Physics inference: disabled until human GUI calibration.")
    print(f"Report: {result['report']}")


if __name__ == "__main__":
    result = run_demo(parse_args())
    print_terminal_summary(result)
    print("\nFull JSON summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
