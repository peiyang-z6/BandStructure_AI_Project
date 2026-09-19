"""Source-bound, multi-panel AI visual labels collected through the review UI."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def make_visual_batch(queue: dict, start: int, submissions: list, batch_size: int = 6) -> dict:
    records = queue["records"][start : start + batch_size]
    if not records or not isinstance(submissions, list) or len(submissions) != len(records):
        raise ValueError("Supply exactly one explicit observation for each displayed figure.")
    labels = []
    for offset, (record, item) in enumerate(zip(records, submissions)):
        if (
            not isinstance(item, dict)
            or type(item.get("i")) is not int
            or item["i"] != start + offset + 1
        ):
            raise ValueError("Observation order/index does not match the displayed batch.")
        if set(item) != {"i", "target", "boxes", "note"}:
            raise ValueError("Required fields: i, target, boxes, note.")
        if item["target"] not in ("yes", "no", "uncertain"):
            raise ValueError("Target must be yes/no/uncertain.")
        boxes = item["boxes"]
        if not isinstance(boxes, list) or len(boxes) > 128:
            raise ValueError("At most 128 panel boxes per figure.")
        if (item["target"] == "yes" and not boxes) or (item["target"] == "no" and boxes):
            raise ValueError("Positive figures require boxes; negative figures require none.")
        note = item["note"]
        if not isinstance(note, str) or not 8 <= len(note.strip()) <= 3000:
            raise ValueError("An image-specific visual observation is required.")
        width, height = record["image"]["width"], record["image"]["height"]
        pixel_boxes = []
        for box in boxes:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            ):
                raise ValueError("Box must contain four finite numbers.")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100):
                raise ValueError("Boxes use full-image percentages in [0,100].")
            pixels = [
                round(x0 * width / 100),
                round(y0 * height / 100),
                round(x1 * width / 100),
                round(y1 * height / 100),
            ]
            if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
                raise ValueError("Box collapses at source pixel resolution.")
            pixel_boxes.append(pixels)
        labels.append(
            {
                "index": item["i"],
                "candidate_id": record["candidate_id"],
                "group_id": record["group_id"],
                "image_sha256": record["image"]["sha256"],
                "preannotation_sha256": record["preannotation_sha256"],
                "target": item["target"],
                "panel_boxes_pct_xyxy": boxes,
                "panel_boxes_pixels_xyxy": pixel_boxes,
                "observation": note.strip(),
                "bbox_precision": "approximate_visual_screen_estimate",
                "label_origin": "ai_visual_review_via_computer_use",
                "human_audited": False,
                "formal_approval": None,
                "eligible_for_scientific_acceptance": False,
            }
        )
    return {
        "schema_version": 1,
        "status": "ai_visual_review_complete_for_batch",
        "dataset_manifest_sha256": queue["dataset_manifest_sha256"],
        "preannotation_manifest_sha256": queue["preannotation_manifest_sha256"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "start_index": start,
        "record_count": len(labels),
        "records": labels,
        "formal_approved_count": 0,
        "human_audited_count": 0,
        "eligible_for_scientific_acceptance": False,
    }


def save_visual_batch(
    queue: dict, start: int, submissions: list, output_dir: Path, batch_size: int = 6
) -> Path:
    packet = make_visual_batch(queue, start, submissions, batch_size)
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("Output symlinks are not supported.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"batch_{start + 1:03d}_{start + packet['record_count']:03d}.json"
    with path.open("xb") as handle:
        handle.write(
            json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
    return path


def finish_visual_review(queue: dict, output_dir: Path, *, write_manifest: bool = True) -> dict:
    """Require coverage of all frozen figures, preserving uncertain and negative labels."""
    output = Path(output_dir)
    by_index, files = {}, []
    for path in sorted(output.glob("batch_*.json")):
        raw = path.read_bytes()
        packet = json.loads(raw)
        if packet["dataset_manifest_sha256"] != queue["dataset_manifest_sha256"]:
            raise ValueError("Dataset binding mismatch.")
        start = packet["start_index"]
        submissions = [
            {
                "i": r["index"],
                "target": r["target"],
                "boxes": r["panel_boxes_pct_xyxy"],
                "note": r["observation"],
            }
            for r in packet["records"]
        ]
        expected = make_visual_batch(queue, start, submissions, packet["record_count"])
        if expected["records"] != packet["records"] or any(
            packet.get(k) != expected[k]
            for k in (
                "preannotation_manifest_sha256",
                "formal_approved_count",
                "human_audited_count",
                "eligible_for_scientific_acceptance",
            )
        ):
            raise ValueError("Visual label binding or status was modified.")
        for record in packet["records"]:
            if record["index"] in by_index:
                raise ValueError("Duplicate visual review index.")
            by_index[record["index"]] = record
        files.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    if set(by_index) != set(range(1, queue["record_count"] + 1)):
        raise ValueError(f"Incomplete visual coverage: {len(by_index)}/{queue['record_count']}.")
    records = [by_index[i] for i in sorted(by_index)]
    summary = {
        "schema_version": 1,
        "status": "complete_ai_visual_review",
        "dataset_manifest_sha256": queue["dataset_manifest_sha256"],
        "preannotation_manifest_sha256": queue["preannotation_manifest_sha256"],
        "record_count": len(records),
        "document_group_count": len({r["group_id"] for r in records}),
        "target_counts": {
            t: sum(r["target"] == t for r in records) for t in ("yes", "no", "uncertain")
        },
        "panel_count": sum(len(r["panel_boxes_pixels_xyxy"]) for r in records),
        "box_precision": "approximate_visual_screen_estimate",
        "label_origin": "ai_visual_review_via_computer_use",
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "files": files,
        "records": records,
    }
    if write_manifest:
        with (output / "visual_review_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    else:
        saved = json.loads((output / "visual_review_manifest.json").read_text(encoding="utf-8"))
        if saved != summary:
            raise ValueError("Summary does not match source-bound batch records.")
    return summary
