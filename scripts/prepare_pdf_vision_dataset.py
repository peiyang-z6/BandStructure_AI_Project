"""Prepare a real-paper PDF fine-tuning dataset for the vision detector.

The script renders PDF pages, pseudo-labels band-plot panels with the current
YOLO pose detector, filters candidates with the parser's band-geometry score,
and writes YOLO pose labels. Pages with no accepted band panel are retained as
hard negatives using empty label files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.multi_format_parser import MultiFormatParser


def render_pdf_pages(pdf_paths: Iterable[Path], output_dir: Path, dpi: int) -> List[Path]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF rendering requires PyMuPDF. Run: pip install PyMuPDF") from exc

    import cv2

    page_dir = output_dir / "rendered_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for pdf in pdf_paths:
        doc = fitz.open(str(pdf))
        safe_stem = "".join(ch if ch.isalnum() else "_" for ch in pdf.stem)
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = arr[:, :, :3]
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out = page_dir / f"{safe_stem}_p{page_idx + 1:03d}.jpg"
            cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rendered.append(out)
        doc.close()
    return rendered


def detector_candidates(image: np.ndarray, model: Any, parser: MultiFormatParser, conf: float) -> List[Dict[str, Any]]:
    results = model.predict(image, conf=conf, verbose=False)
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    scores = boxes.conf.cpu().numpy().astype(float) if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
    keypoints_xy = None
    keypoints_conf = None
    if getattr(result, "keypoints", None) is not None and getattr(result.keypoints, "xy", None) is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        if getattr(result.keypoints, "conf", None) is not None:
            keypoints_conf = result.keypoints.conf.cpu().numpy()

    h, w = image.shape[:2]
    candidates: List[Dict[str, Any]] = []
    for idx, (box, score) in enumerate(zip(xyxy, scores)):
        x0, y0, x1, y1 = box
        x0i, y0i = max(int(x0), 0), max(int(y0), 0)
        x1i, y1i = min(int(x1), w - 1), min(int(y1), h - 1)
        if x1i <= x0i or y1i <= y0i:
            continue
        bw, bh = x1i - x0i, y1i - y0i
        if bw < 70 or bh < 55:
            continue
        if bw * bh > 0.85 * w * h:
            continue
        panel = image[y0i:y1i, x0i:x1i].copy()
        panel, offset, frame_meta = parser._crop_to_plot_frame(panel)
        px0, py0 = x0i + offset[0], y0i + offset[1]
        px1, py1 = px0 + panel.shape[1], py0 + panel.shape[0]
        geom_score = parser._score_detected_panel(panel, float(score), frame_meta)
        kps = []
        if keypoints_xy is not None and idx < len(keypoints_xy) and keypoints_xy[idx].shape[0] >= 2:
            for kp_idx in range(2):
                kx, ky = keypoints_xy[idx][kp_idx]
                kc = 1.0
                if keypoints_conf is not None and idx < len(keypoints_conf) and kp_idx < len(keypoints_conf[idx]):
                    kc = float(keypoints_conf[idx][kp_idx])
                inside = px0 <= kx <= px1 and py0 <= ky <= py1
                if not inside:
                    kx = px0 + 0.5 * (px1 - px0)
                    ky = py0 + 0.5 * (py1 - py0)
                    kc = 0.0
                kps.append((float(kx), float(ky), kc))
        if len(kps) < 2:
            kps = [
                (px0 + 0.35 * (px1 - px0), py0 + 0.55 * (py1 - py0), 0.0),
                (px0 + 0.65 * (px1 - px0), py0 + 0.45 * (py1 - py0), 0.0),
            ]
        candidates.append(
            {
                "bbox_xyxy": [float(px0), float(py0), float(px1), float(py1)],
                "confidence": float(score),
                "candidate_score": float(geom_score),
                "keypoints": kps,
                "frame_meta": frame_meta,
            }
        )
    return non_max_suppression(candidates)


def non_max_suppression(candidates: List[Dict[str, Any]], iou_threshold: float = 0.55) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: item["candidate_score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        if all(iou(cand["bbox_xyxy"], old["bbox_xyxy"]) < iou_threshold for old in kept):
            kept.append(cand)
    return kept


def iou(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def write_yolo_label(label_path: Path, candidates: List[Dict[str, Any]], image_shape: tuple[int, int]) -> None:
    h, w = image_shape
    lines = []
    for cand in candidates:
        x0, y0, x1, y1 = cand["bbox_xyxy"]
        xc = ((x0 + x1) * 0.5) / w
        yc = ((y0 + y1) * 0.5) / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        values: List[float | int] = [0, xc, yc, bw, bh]
        for kx, ky, kc in cand["keypoints"]:
            values.extend([kx / w, ky / h, 2 if kc >= 0.05 else 0])
        lines.append(" ".join(f"{v:.8f}" if isinstance(v, float) else str(v) for v in values))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    import cv2
    from ultralytics import YOLO  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [Path(p) for p in args.pdf]
    pages = render_pdf_pages(pdfs, output_dir, args.dpi)
    model = YOLO(args.detector)
    parser = MultiFormatParser(detector_path=args.detector, detector_conf=args.detector_conf)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for idx, page_path in enumerate(pages):
        image = cv2.imread(str(page_path))
        if image is None:
            continue
        candidates = [
            item
            for item in detector_candidates(image, model, parser, args.detector_conf)
            if item["candidate_score"] >= args.min_candidate_score
        ]
        split = "val" if idx % max(args.val_stride, 2) == 0 else "train"
        dst_image = output_dir / "images" / split / page_path.name
        dst_label = output_dir / "labels" / split / f"{page_path.stem}.txt"
        cv2.imwrite(str(dst_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        write_yolo_label(dst_label, candidates, image.shape[:2])
        manifest.append(
            {
                "image": str(dst_image),
                "label": str(dst_label),
                "split": split,
                "positive_boxes": len(candidates),
                "candidates": candidates,
            }
        )

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: band_plot
kpt_shape: [2, 3]
""",
        encoding="utf-8",
    )
    report = {
        "output_dir": str(output_dir),
        "pdf_count": len(pdfs),
        "rendered_pages": len(pages),
        "positive_pages": sum(1 for item in manifest if item["positive_boxes"] > 0),
        "hard_negative_pages": sum(1 for item in manifest if item["positive_boxes"] == 0),
        "positive_boxes": sum(item["positive_boxes"] for item in manifest),
        "dataset_yaml": str(dataset_yaml),
        "manifest": manifest,
    }
    (output_dir / "pdf_dataset_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rendered-PDF YOLO pose fine-tuning dataset")
    parser.add_argument("--pdf", action="append", required=True, help="PDF file path. Repeat for multiple PDFs.")
    parser.add_argument("--output-dir", default="data/processed/vision/pdf_finetune")
    parser.add_argument("--detector", default="artifacts/models/vision_detector/band_plot_yolov8_pose_best.pt")
    parser.add_argument("--detector-conf", type=float, default=0.06)
    parser.add_argument("--min-candidate-score", type=float, default=0.72)
    parser.add_argument("--dpi", type=int, default=135)
    parser.add_argument("--val-stride", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(parse_args()), indent=2, ensure_ascii=False))
