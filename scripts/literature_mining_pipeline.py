"""Stage-3 literature mining pipeline for experimental band plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import MultiFormatParser, PhysicsBrainInvoker, PhysicsReconstructor
from src.vision.brain_invoker import prediction_to_jsonable


def extract_pdf_images(pdf_dir: Path, output_dir: Path) -> List[Path]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF image extraction requires PyMuPDF. Run: pip install PyMuPDF") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(str(pdf))
        for page_index in range(doc.page_count):
            page = doc[page_index]
            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out = output_dir / f"{pdf.stem}_p{page_index + 1:03d}_img{image_index + 1:03d}.png"
                pix.save(str(out))
                extracted.append(out)
        doc.close()
    return extracted


def physics_confidence(reconstructed, prediction) -> Dict[str, Any]:
    vbm_curv = float(reconstructed.vbm_curvature[reconstructed.vbm_index])
    cbm_curv = float(reconstructed.cbm_curvature[reconstructed.cbm_index])
    vbm_prob = np.asarray(prediction.vbm_probability, dtype=np.float32)
    cbm_prob = np.asarray(prediction.cbm_probability, dtype=np.float32)
    entropy = float(
        -np.sum(vbm_prob * np.log(vbm_prob + 1e-8)) / np.log(len(vbm_prob))
        -np.sum(cbm_prob * np.log(cbm_prob + 1e-8)) / np.log(len(cbm_prob))
    ) / 2.0
    curvature_ok = bool(vbm_curv <= 0.0 and cbm_curv >= 0.0)
    gap_ok = bool(prediction.line_mode_gap_ev >= -0.05 and not reconstructed.is_metallic)
    probability_ok = bool(entropy < 0.65)
    high_confidence = curvature_ok and gap_ok and probability_ok
    return {
        "curvature_ok": curvature_ok,
        "gap_ok": gap_ok,
        "probability_ok": probability_ok,
        "probability_entropy": entropy,
        "high_confidence": high_confidence,
        "reason": "physics_self_consistent" if high_confidence else "needs_human_review",
    }


def store_experimental_band(h5_path: Path, image_path: Path, reconstructed, prediction, confidence: Dict[str, Any]) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    key = image_path.stem.replace("-", "_").replace(".", "_")
    with h5py.File(h5_path, "a") as h5:
        if key in h5:
            del h5[key]
        grp = h5.create_group(key)
        grp.create_dataset("raw_tensor", data=reconstructed.raw_tensor, compression="gzip")
        grp.create_dataset("flat_tensor", data=reconstructed.flat_tensor, compression="gzip")
        grp.create_dataset("vbm_probability", data=prediction.vbm_probability, compression="gzip")
        grp.create_dataset("cbm_probability", data=prediction.cbm_probability, compression="gzip")
        grp.attrs["source_image"] = str(image_path)
        grp.attrs["line_mode_gap_ev"] = float(prediction.line_mode_gap_ev)
        grp.attrs["predicted_type"] = prediction.predicted_type
        grp.attrs["confidence"] = json.dumps(confidence)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "extracted_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = extract_pdf_images(pdf_dir, image_dir)
    parser = MultiFormatParser(
        image_height_ev=args.image_height_ev,
        detector_path=args.detector,
        detector_conf=args.detector_conf,
    )
    reconstructor = PhysicsReconstructor()
    invoker = PhysicsBrainInvoker()

    records: List[Dict[str, Any]] = []
    h5_path = Path(args.experimental_h5)
    for image_path in images:
        try:
            parsed = parser.parse(str(image_path))
            reconstructed = reconstructor.reconstruct(parsed)
            prediction = invoker.predict(reconstructed)
            confidence = physics_confidence(reconstructed, prediction)
            if confidence["high_confidence"]:
                store_experimental_band(h5_path, image_path, reconstructed, prediction, confidence)
            records.append(
                {
                    "image": str(image_path),
                    "status": "stored" if confidence["high_confidence"] else "low_confidence",
                    "confidence": confidence,
                    "vision_metadata": reconstructed.metadata,
                    "prediction": prediction_to_jsonable(prediction),
                    "tensor_gap_ev": reconstructed.tensor_gap,
                }
            )
        except Exception as exc:
            records.append({"image": str(image_path), "status": "failed", "error": str(exc)})

    summary = {
        "pdf_dir": str(pdf_dir),
        "images_extracted": len(images),
        "high_confidence": sum(1 for r in records if r.get("status") == "stored"),
        "low_confidence": sum(1 for r in records if r.get("status") == "low_confidence"),
        "failed": sum(1 for r in records if r.get("status") == "failed"),
        "experimental_h5": str(h5_path),
        "records": records,
    }
    (output_dir / "literature_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "literature_vs_dft_report.md")
    return summary


def write_report(summary: Dict[str, Any], path: Path) -> None:
    rows = []
    for rec in summary["records"]:
        if "prediction" in rec:
            pred = rec["prediction"]
            rows.append(
                f"| {Path(rec['image']).name} | {rec['status']} | {pred['line_mode_gap_ev']:.4f} | {pred['predicted_type']} | {rec['confidence']['reason']} |"
            )
        else:
            rows.append(f"| {Path(rec['image']).name} | {rec['status']} | - | - | {rec.get('error', '')} |")

    text = f"""# Literature Experimental Data vs DFT Theory Report

This report mines experimental/theoretical band figures from PDF images, converts
valid figures into the project 6D tensor contract, and compares extracted
line-mode physics against the trained DFT-derived model brain.

## Summary

- PDF folder: `{summary['pdf_dir']}`
- Extracted images: `{summary['images_extracted']}`
- High-confidence stored bands: `{summary['high_confidence']}`
- Low-confidence review queue: `{summary['low_confidence']}`
- Failed images: `{summary['failed']}`
- Experimental HDF5: `{summary['experimental_h5']}`

## Records

| Image | Status | Brain gap (eV) | Brain type | Physics check |
|---|---:|---:|---|---|
{chr(10).join(rows)}
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine PDF literature figures into experimental band tensors")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/reports/literature_mining")
    parser.add_argument("--experimental-h5", default="data/raw/experimental/experimental_bands.h5")
    parser.add_argument("--image-height-ev", type=float, default=8.0)
    parser.add_argument("--detector", default="artifacts/models/vision_detector/band_plot_yolov8_pose_best.pt")
    parser.add_argument("--detector-conf", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_pipeline(parse_args()), indent=2))
