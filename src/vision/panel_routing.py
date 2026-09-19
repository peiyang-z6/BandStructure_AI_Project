"""Source-bound host ROI and conservative, nonsemantic frame candidates."""

import csv
import hashlib
import io
import math
import numpy as np

PANEL_KINDS = {"electronic_band", "phonon", "transport", "dos", "unknown"}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_selection(selection, source_sha, size, *, page_number=1, unit="pixel"):
    required = {"source_sha256", "page_number", "coordinate_unit", "panel_bbox_xyxy", "panel_kind"}
    if not isinstance(selection, dict) or set(selection) != required:
        raise ValueError("ROI requires source SHA, page, coordinate unit, bbox and kind")
    if (
        selection["source_sha256"] != source_sha
        or type(selection["page_number"]) is not int
        or selection["page_number"] != page_number
    ):
        raise ValueError("ROI source/page binding mismatch")
    if (
        selection["coordinate_unit"] != unit
        or not isinstance(selection["panel_kind"], str)
        or selection["panel_kind"] not in PANEL_KINDS
    ):
        raise ValueError("Invalid ROI coordinate unit or panel kind")
    box = selection["panel_bbox_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box):
        raise ValueError("ROI must contain four finite numbers")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]) or min(x1 - x0, y1 - y0) < 4:
        raise ValueError("ROI lies outside the source or is too small")
    if unit == "pixel" and any(int(v) != v for v in box):
        raise ValueError("Pixel ROI must use integer coordinates")
    return list(box)


def frame_candidates(image):
    import cv2

    a = np.asarray(image.convert("RGB"))
    h, w = a.shape[:2]
    # Dark neutral lines only, not spectral colour intensity regions.
    dark = ((a.max(2) < 150) & (a.max(2).astype(int) - a.min(2) < 45)).astype("uint8") * 255
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(15, w // 18)), np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(15, h // 10), 1), np.uint8))
    grid = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 40 or bh < 40 or bw * bh < 0.012 * w * h:
            continue
        # Each of the four boundary strips must contain an actual long side.
        local = grid[y : y + bh, x : x + bw] > 0
        scores = [
            local[:4].any(0).mean(),
            local[-4:].any(0).mean(),
            local[:, :4].any(1).mean(),
            local[:, -4:].any(1).mean(),
        ]
        if min(scores) < 0.7:
            continue
        boxes.append([max(0, x + 1), max(0, y + 1), min(w, x + bw - 1), min(h, y + bh - 1)])
    boxes.sort(key=lambda b: (b[1] // 15, b[0]))
    return [
        {
            "candidate_id": f"panel-{i + 1}",
            "panel_bbox_xyxy": b,
            "coordinate_unit": "pixel",
            "panel_kind": "unknown",
            "requires_semantic_review": True,
        }
        for i, b in enumerate(boxes)
    ]


def visible_ink(image, box):
    """Export contiguous pixel runs, never connect or identify physical bands."""
    x0, y0, x1, y1 = map(int, box)
    a = np.asarray(image.convert("RGB"))[y0:y1, x0:x1].astype(int)
    spread = a.max(2) - a.min(2)
    coloured = (spread > 45) & (a.min(2) < 180)
    neutral = (spread <= 45) & (a.max(2) < 160)
    # Suppress long neutral rules only; a coloured legend must not erase black bands.
    neutral[:, neutral.mean(0) > 0.75] = False
    neutral[neutral.mean(1) > 0.75, :] = False
    mask = coloured | neutral
    mode = "mixed_coloured_and_neutral_pixels_unverified"
    mask[:2] = False
    mask[-2:] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    count = int(mask.sum())
    base = {
        "point_count": count,
        "coordinate_unit": "source_pixel",
        "band_identity_resolved": False,
        "energy_calibrated": False,
        "selection_mode": mode,
        "coloured_pixel_count": int((mask & coloured).sum()),
        "neutral_pixel_count": int((mask & neutral).sum()),
        "limitations": [
            "Visible ink only; no inferred band IDs or energies.",
            "Neutral text and annotations may remain; this is ink, not semantic curve tracing.",
            "Grid/text occlusion, colour thresholds and source cropping leave missing evidence.",
        ],
    }
    if count > 200000:
        return {
            **base,
            "status": "requires_smaller_roi",
            "export_complete": False,
            "svg": None,
            "csv_text": None,
        }
    runs = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)))
        for start, end in zip(edges[::2], edges[1::2]):
            color = np.median(a[y, start:end], axis=0).astype(int)
            runs.append((x0 + int(start), x0 + int(end), y0 + y, *map(int, color)))
    if len(runs) > 8000:
        return {
            **base,
            "status": "requires_smaller_roi",
            "export_complete": False,
            "svg": None,
            "csv_text": None,
        }
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["x0", "x1_exclusive", "y", "r", "g", "b"])
    w.writerows(runs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {x1 - x0} {y1 - y0}">',
        "<title>Uncalibrated visible source-ink runs; not complete bands</title>",
    ]
    for left, right, y, r, g, b in runs:
        parts.append(
            f'<path fill="#{r:02x}{g:02x}{b:02x}" d="M{left} {y}h{right - left}v1h-{right - left}z"/>'
        )
    parts.append("</svg>")
    return {
        **base,
        "status": "visible_ink_unverified",
        "export_complete": True,
        "run_count": len(runs),
        "svg": "".join(parts),
        "csv_text": buf.getvalue(),
    }


def selected_raster(payload, image, selection):
    sha = hashlib.sha256(payload).hexdigest()
    box = validate_selection(selection, sha, image.size)
    kind = selection["panel_kind"]
    base = {
        "source_sha256": sha,
        "panel_selection": selection,
        "panel_bbox": box,
        "image_size": list(image.size),
        "confidence": None,
        "ood_flag": None,
        "human_audited": False,
        "band_data": None,
    }
    if kind != "electronic_band":
        return {
            **base,
            "status": "requires_panel_selection" if kind == "unknown" else "non_electronic_panel",
            "message": "Only a host-reviewed electronic-band ROI enters electronic analysis.",
        }
    ink = visible_ink(image, box)
    return {
        **base,
        "status": "requires_calibration",
        "visible_ink": ink,
        "next_action": "Use schema_version=2 observations for multi-band/segmented calibrated samples; retain unknown identities.",
        "calibration_verified": False,
    }
