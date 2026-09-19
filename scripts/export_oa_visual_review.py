"""Validate all Computer Use labels and export AI-labelled COCO, CSV and diagnostics."""

from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcp_server.figure_review_queue import load_figure_review_queue
from mcp_server.visual_batch_review import finish_visual_review
from src.utils.csv_safety import spreadsheet_row


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def export_review(dataset, preannotations, review_dir, output):
    queue = load_figure_review_queue(dataset, preannotations)
    review = finish_visual_review(queue, review_dir, write_manifest=False)
    # All bytes are rechecked before producing anything; exports never edit frozen inputs.
    if output.exists():
        raise FileExistsError(str(output))
    by_id = {r["candidate_id"]: r for r in queue["records"]}
    counts, rows, images, annotations = Counter(), [], [], []
    statuses = {}
    for label in review["records"]:
        source = by_id[label["candidate_id"]]
        bbox = source["preannotation"].get("panel_bbox_xyxy")
        boxes = label["panel_boxes_pixels_xyxy"]
        best_iou = max((iou(bbox, b) for b in boxes), default=0.0) if bbox else None
        flags = []
        if label["target"] == "uncertain":
            flags.append("target_decision_uncertain")
        elif label["target"] == "no":
            flags.append("visual_non_target")
            if bbox:
                counts["cv_box_on_visual_non_target"] += 1
        else:
            if not bbox:
                flags.append("positive_missing_cv_box")
                counts["positive_missing_cv_box"] += 1
            elif best_iou < 0.5:
                flags.append("cv_best_iou_below_0.5")
                counts["positive_cv_best_iou_below_0.5"] += 1
            else:
                counts["positive_cv_best_iou_at_least_0.5"] += 1
            if len(boxes) > 1:
                flags.append("multiple_visual_panels")
                counts["positive_multi_panel_figures"] += 1
            counts["positive_visual_panels"] += len(boxes)
        key = source["preannotation"]["status"]
        statuses.setdefault(key, Counter())[label["target"]] += 1
        rows.append(
            {
                "index": label["index"],
                "candidate_id": label["candidate_id"],
                "pmcid": label["group_id"],
                "doi": source["source"].get("doi"),
                "figure": source["source"].get("figure_label"),
                "target_ai": label["target"],
                "panel_count": len(boxes),
                "boxes_pixels_xyxy": json.dumps(boxes),
                "cv_box_xyxy": json.dumps(bbox),
                "best_cv_vs_ai_iou": best_iou,
                "review_flags": ";".join(flags),
                "observation": label["observation"],
                "image_sha256": label["image_sha256"],
                "image_path": source["image_path"],
                "label_origin": label["label_origin"],
                "human_audited": False,
                "formal_approval": None,
            }
        )
        images.append(
            {
                "id": label["index"],
                "file_name": source["image"]["path"],
                "width": source["image"]["width"],
                "height": source["image"]["height"],
                "pmcid": label["group_id"],
                "candidate_id": label["candidate_id"],
                "sha256": label["image_sha256"],
                "target_ai": label["target"],
                "exclude_from_scoring": label["target"] == "uncertain",
            }
        )
        for box in boxes:
            x0, y0, x1, y1 = box
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": label["index"],
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                    "ignore": int(label["target"] == "uncertain"),
                    "label_origin": label["label_origin"],
                    "human_audited": False,
                    "box_precision": label["bbox_precision"],
                }
            )
    report = {
        "status": "ai_visual_label_audit_complete",
        "record_count": review["record_count"],
        "target_counts": review["target_counts"],
        "document_group_count": review["document_group_count"],
        "panel_count": review["panel_count"],
        "cv_comparison_counts": dict(counts),
        "cv_status_vs_visual_target": {k: dict(v) for k, v in statuses.items()},
        "comparison_scope": "CV outputs compared with approximate AI visual labels; not human-grounded accuracy",
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "curve_annotations_complete": False,
        "source_dataset_manifest_sha256": queue["dataset_manifest_sha256"],
        "source_preannotation_manifest_sha256": queue["preannotation_manifest_sha256"],
        "visual_review_manifest_sha256": hashlib.sha256(
            (review_dir / "visual_review_manifest.json").read_bytes()
        ).hexdigest(),
        "unresolved_target_indices": [
            r["index"] for r in review["records"] if r["target"] == "uncertain"
        ],
    }
    output.mkdir(parents=True)
    with (output / "figure_review.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(spreadsheet_row(row) for row in rows)
    with (output / "figure_review.json").open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
    coco = {
        "info": {
            "description": "AI visual panel labels; screen-estimated boxes; not human ground truth",
            "image_root": str(dataset.parent.resolve()),
            "human_audited": False,
            "uncertain_policy": "exclude images with exclude_from_scoring=true before COCO scoring; ignore alone may not be honoured",
        },
        "categories": [{"id": 1, "name": "electronic_band_panel"}],
        "images": images,
        "annotations": annotations,
    }
    for name, payload in [("panels_coco_ai.json", coco), ("diagnostics.json", report)]:
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    card = f"""# OA{review["record_count"]}: Computer Use visual review

All {review["record_count"]} figures from {review["document_group_count"]} PMCIDs were visually inspected
in {len(review["files"])} batches of up to six figures. Observations and multi-panel boxes were entered through the Windows UI.
Inherited batches, when present, retain their original review times and source-batch hashes.
The immutable source figures and evaluation allocation were not changed.

- Electronic band figures (AI judgement): {review["target_counts"]["yes"]}
- Non-target figures (AI judgement): {review["target_counts"]["no"]}
- Ambiguous figures: {review["target_counts"]["uncertain"]} (indices {report["unresolved_target_indices"]})
- Panel boxes: {review["panel_count"]}; {counts["positive_visual_panels"]} on positive figures
- Box precision: approximate screen estimates in full-image percentages, converted to pixels
- Independent human approval: 0

## Files

- figure_review.csv: all {review["record_count"]} figures, image paths, observations, boxes and CV comparison flags.
- panels_coco_ai.json: {review["record_count"]} image entries and all panel candidates. Image paths are relative to info.image_root.
  Remove uncertain images using exclude_from_scoring before exploratory scoring; some COCO tools ignore the custom ignore flag.
- diagnostics.json: validated counts and diagnostic comparison with the original CV preannotations.

## Scope and unresolved work

Targets include calculated or experimental electronic E(k) cuts; scalar DOS, phonon frequency,
energy-level alignment, workflow diagrams and unscaled conceptual cones are excluded.
Local E(k) enlargements are separate panels when separately framed. Internal k-path breaks stay in a parent panel.
Approximate boxes locate the plot interior, usually excluding labels; every box still requires precision review.
Source-bound AI labels permit exploratory development/diagnostics with explicit AI provenance.
They do not establish independent scientific accuracy or fulfil the existing P1 human-approval gate.
No continuous curves, axis calibration, band gaps or effective masses were newly labelled here.

Raw collection metadata reports article-level CC-BY. Third-party figure rights, re-use of identical
figures across papers, pretraining exposure, and original experimental provenance have not been independently certified.
The keyword-selected corpus is not a random sample of the literature; all figures remain evaluation-reserved.
CV-vs-AI overlap is a disagreement diagnostic, not evidence of performance against human ground truth.
"""
    (output / "DATA_CARD.md").write_text(card, encoding="utf-8")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    with (output / "export_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "files_sha256": hashes,
                "visual_review_manifest_sha256": report["visual_review_manifest_sha256"],
            },
            handle,
            indent=2,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--preannotation-manifest", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_review(
                args.dataset_manifest, args.preannotation_manifest, args.review_dir, args.output_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
