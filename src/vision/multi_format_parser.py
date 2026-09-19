"""Multimodal band-plot parsers for Phase 5 Plot-to-Physics.

The parser extracts the visual plot panel and curve skeleton from figures. It
intentionally does not infer Fermi level, VBM/CBM, or direct/indirect physics.
Those physical decisions are made by human calibration in the GUI and then by
the Physics Brain.

PDF/EPS support is optional. Install PyMuPDF manually if vector parsing is
needed:

    pip install PyMuPDF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".gif"}
VECTOR_SUFFIXES = {".pdf", ".eps"}


def cv_quality_level(score: float) -> str:
    """Map an aggregated 0-1 extraction-quality score to the traffic-light level."""
    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.6:
        return "green"
    if score >= 0.35:
        return "yellow"
    return "red"


_PANEL_SOURCE_SCORE = {
    "vision_detector_yolov8_pose": 0.9,
    "line_plot_frame": 0.8,
    "long_line_plot_frame": 0.8,
    "largest_gray_band_panel": 0.65,
    "largest_dark_plot_frame": 0.6,
    "fallback_full_image": 0.3,
}


_DOCUMENT_OCR_ENGINE = None


def _document_ocr_page(image: np.ndarray, number: int) -> dict:
    """OCR is a perception result, never an energy-axis or physics certificate."""
    global _DOCUMENT_OCR_ENGINE
    if _DOCUMENT_OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _DOCUMENT_OCR_ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    rows, timing = _DOCUMENT_OCR_ENGINE(image)
    regions = [
        {
            "box": np.asarray(row[0], dtype=float).tolist(),
            "text": str(row[1]),
            "ocr_score": float(row[2]),
        }
        for row in (rows or [])
    ]
    from src.vision.axis_ocr_review import review_axis_tokens

    return {
        "page": number,
        "engine": "rapidocr_onnxruntime",
        "regions": regions,
        "text": "\n".join(row["text"] for row in regions),
        "coordinate_unit": "pixel",
        "image_size": [image.shape[1], image.shape[0]],
        "ocr_seconds": [float(t) for t in (timing or [])],
        "axis_token_review": review_axis_tokens(regions),
    }


def _bounded_document_raster(payload: bytes):
    import io
    from PIL import Image
    from src.vision.calibration_contract import decode_raster_bytes

    if not isinstance(payload, bytes) or not 0 < len(payload) <= 10 * 1024 * 1024:
        raise ValueError("document payload must be nonempty bytes, at most 10 MiB")
    with Image.open(io.BytesIO(payload)) as header:
        if header.width * header.height > 4096 * 4096:
            raise ValueError("document raster exceeds 16777216 pixels")
    return decode_raster_bytes(payload)


def extract_band_image(
    payload: bytes,
    *,
    annotations: dict | None = None,
    calibration: dict | None = None,
    panel_selection: dict | None = None,
) -> dict:
    """Use the existing CV frontend; never infer eV or Fermi from pixel height."""
    import hashlib
    import tempfile

    with _bounded_document_raster(payload) as image:
        image_size = image.size
        from src.vision.panel_routing import selected_raster, frame_candidates

        if panel_selection is not None:
            if annotations is not None or calibration is not None:
                raise ValueError("Do not mix legacy calibration and source-bound ROI")
            return selected_raster(payload, image, panel_selection)
        candidates = frame_candidates(image) if annotations is None and calibration is None else []
        if len(candidates) > 1:
            return {
                "status": "requires_panel_selection",
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "image_size": list(image_size),
                "panel_bbox": None,
                "panel_candidates": candidates,
                "panel_candidates_total": len(candidates),
                "panel_candidates_truncated": False,
                "band_data": None,
                "confidence": None,
                "ood_flag": None,
                "human_audited": False,
                "skeleton_point_count": 0,
                "cv_quality": None,
                "next_action": "Classify each candidate from the source image, then submit one source-bound electronic_band ROI.",
            }
    with tempfile.TemporaryDirectory(prefix="band-upload-") as directory:
        path = Path(directory) / "figure.png"
        path.write_bytes(payload)
        parsed = MultiFormatParser(detector_path=None).parse_image(str(path))
    result = {
        "status": "requires_calibration",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "image_size": list(image_size),
        "band_data": None,
        "skeleton_point_count": int(len(parsed.skeleton_pixels)),
        "cv_quality": parsed.metadata.get("cv_quality"),
        "panel_bbox": parsed.metadata.get("panel_bbox"),
        "confidence": None,
        "ood_flag": None,
        "human_audited": False,
        "limitations": [
            "CV scores are heuristics, not calibrated probabilities.",
            "Provide explicit single-segment pixel axes, Fermi and valence/conduction annotations.",
        ],
    }
    if annotations is not None and calibration is not None:
        import json
        from src.vision.calibration_contract import (
            validate_manual_calibration,
            validate_manual_extrema,
        )
        from src.vision.physics_reconstructor import PhysicsReconstructor

        # This upload boundary must inspect original JSON types before the legacy
        # manual pipeline's float/NumPy conversions can erase bool or string types.
        import math

        def require_real_numbers(value, depth=0):
            if isinstance(value, list) and depth < 3:
                for item in value:
                    require_real_numbers(item, depth + 1)
                return
            if type(value) not in (int, float):
                raise ValueError(
                    "calibration fields require finite real numbers, not bool or strings"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("calibration fields require finite real numbers")

        if not isinstance(annotations, dict) or not isinstance(calibration, dict):
            raise ValueError("manual calibration must be mappings")
        for field in (
            "panel",
            "fermi_y",
            "xaxis_pts",
            "yaxis_pts",
            "vbm",
            "cbm",
            "vb_strokes",
            "cb_strokes",
        ):
            require_real_numbers(annotations.get(field))
        for field in ("x_values", "y_values"):
            require_real_numbers(calibration.get(field))
        validate_manual_calibration(annotations, calibration, image_size=image_size)
        ann, cal = annotations, calibration
        x, y, w, h = ann["panel"]
        reconstructed = PhysicsReconstructor().reconstruct_from_manual(
            source_path="upload-sha256:" + result["source_sha256"],
            panel_bbox=[x, y, x + w, y + h],
            y_calibration=[
                {"y": p[1], "value": v} for p, v in zip(ann["yaxis_pts"], cal["y_values"])
            ],
            x_calibration=[
                {"x": p[0], "value": v} for p, v in zip(ann["xaxis_pts"], cal["x_values"])
            ],
            vbm_pixel=dict(zip(("x", "y"), ann["vbm"])),
            cbm_pixel=dict(zip(("x", "y"), ann["cbm"])),
            valence_points=[dict(zip(("x", "y"), p)) for s in ann["vb_strokes"] for p in s],
            conduction_points=[dict(zip(("x", "y"), p)) for s in ann["cb_strokes"] for p in s],
            fermi_y_pixel=ann["fermi_y"],
        )
        validate_manual_extrema(ann, cal, reconstructed=reconstructed.flat_tensor)
        result.update(
            status="caller_calibrated_unverified",
            tensor_shape=list(reconstructed.raw_tensor.shape),
            calibration_sha256=hashlib.sha256(
                json.dumps(
                    {"annotations": ann, "calibration": cal}, sort_keys=True, allow_nan=False
                ).encode()
            ).hexdigest(),
            band_data={
                "energies_eV": [
                    reconstructed.vbm_energy.astype(float).tolist(),
                    reconstructed.cbm_energy.astype(float).tolist(),
                ],
                "k_distance": reconstructed.k_axis.astype(float).tolist(),
                "segment_ids": [0] * len(reconstructed.k_axis),
                "fermi_eV": 0.0,
                "k_unit": "relative",
                "band_roles": ["valence", "conduction"],
            },
        )
        result["limitations"].append(
            "Caller-supplied calibration is not an independent human audit; physical k units and mass remain unknown."
        )
    return result


def extract_document_text(
    payload: bytes, *, kind: str, max_pages: int = 3, page_start: int = 1
) -> dict:
    """Read supplied document bytes, without treating text as calibration evidence."""
    import hashlib
    import pymupdf

    if not isinstance(payload, bytes) or not 0 < len(payload) <= 10 * 1024 * 1024:
        raise ValueError("document payload must be nonempty bytes, at most 10 MiB")
    if kind not in {"pdf", "image"}:
        raise ValueError("document kind must be pdf or image")
    if type(max_pages) is not int or not 1 <= max_pages <= 5:
        raise ValueError("max_pages must be an integer from 1 to 5")
    if type(page_start) is not int or page_start < 1:
        raise ValueError("page_start must be a positive nonboolean integer")
    pages = []
    if kind == "image":
        if page_start != 1:
            raise ValueError("raster input only has page 1")
        with _bounded_document_raster(payload) as image:
            pages.append(_document_ocr_page(np.asarray(image)[:, :, ::-1].copy(), 1))
        total_pages = 1
    else:
        if not payload.lstrip().startswith(b"%PDF-"):
            raise ValueError("declared pdf requires a PDF header")
        with pymupdf.open(stream=payload, filetype="pdf") as document:
            if page_start > len(document):
                raise ValueError("page_start exceeds document pages")
            for index in range(page_start - 1, min(page_start - 1 + max_pages, len(document))):
                page = document[index]
                text = page.get_text()
                if text.strip():
                    pages.append(
                        {
                            "page": index + 1,
                            "engine": "pymupdf_text",
                            "text": text,
                            "coordinate_unit": "pdf_point",
                        }
                    )
                else:
                    if (
                        np.ceil(page.rect.width * 150 / 72) * np.ceil(page.rect.height * 150 / 72)
                        > 4096 * 4096
                    ):
                        raise ValueError("PDF page render exceeds 16777216 pixels")
                    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False)
                    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3
                    )
                    item = _document_ocr_page(image[:, :, ::-1].copy(), index + 1)
                    item["pdf_render_dpi"] = 150
                    pages.append(item)
            total_pages = len(document)
    from src.vision.axis_ocr_review import review_document_text

    for extracted_page in pages:
        extracted_page["text_quality"] = review_document_text(extracted_page["text"])
    return {
        "status": "ok",
        "text": "\n".join(p["text"] for p in pages),
        "pages": pages,
        "requires_native_visual_review": any(
            p["text_quality"]["status"] == "requires_visual_confirmation" for p in pages
        ),
        "content_trust": "untrusted_document_data",
        "document_instructions_are_authority": False,
        "total_pages": total_pages,
        "truncated": total_pages > len(pages),
        "page_start": page_start,
        "next_page": pages[-1]["page"] + 1 if pages and pages[-1]["page"] < total_pages else None,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "confidence": None,
        "ood_flag": None,
        "calibration_verified": False,
    }


def inspect_pdf_band_page(payload: bytes, *, page_number: int = 1) -> dict:
    """Expose supplied PDF page geometry without inventing axes or band energies."""
    import base64
    import pymupdf
    from collections import Counter

    checked = extract_document_text(payload, kind="pdf", max_pages=1, page_start=page_number)
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        page = document[page_number - 1]
        if np.ceil(page.rect.width * 2.5) * np.ceil(page.rect.height * 2.5) > 4096 * 4096:
            raise ValueError("PDF preview exceeds 16777216 pixels")
        pix = page.get_pixmap(dpi=180, colorspace=pymupdf.csRGB, alpha=False)
        drawings = page.get_drawings()
        words = page.get_text("words")
        if len(drawings) > 50000 or len(words) > 50000:
            raise ValueError("PDF page geometry exceeds bounded inspection budget")
        images = page.get_image_info()
        with pymupdf.open() as single_page:
            single_page.insert_pdf(
                document,
                from_page=page_number - 1,
                to_page=page_number - 1,
                links=False,
                annots=False,
                widgets=False,
            )
            source_pdf = single_page.tobytes(garbage=4, deflate=True)
        if len(source_pdf) > 10 * 1024 * 1024:
            raise ValueError("selected source PDF exceeds output byte budget")
        return {
            "status": "geometry_only",
            "source_sha256": checked["source_sha256"],
            "source_page_pdf_base64": base64.b64encode(source_pdf).decode(),
            "source_page_pdf_role": "original_graphics_copy_not_reconstruction",
            "page_number": page_number,
            "total_pages": len(document),
            "page_size_points": [float(page.rect.width), float(page.rect.height)],
            "preview_size_pixels": [pix.width, pix.height],
            "preview_dpi": 180,
            "preview_png_base64": base64.b64encode(pix.tobytes("png")).decode(),
            "text_words": [{"bbox": [float(v) for v in w[:4]], "text": str(w[4])} for w in words],
            "drawing_count": len(drawings),
            "drawing_types": dict(Counter(d.get("type", "unknown") for d in drawings)),
            "drawing_item_count": sum(len(d.get("items", [])) for d in drawings),
            "embedded_images": [
                {"bbox": list(i["bbox"]), "width": i["width"], "height": i["height"]}
                for i in images
            ],
            "band_data": None,
            "confidence": None,
            "ood_flag": None,
            "human_audited": False,
            "calibration_verified": False,
            "limitations": [
                "Page geometry and preview only; no inferred physical values.",
                "Electronic bands, phonons and superconducting gaps must remain distinct.",
            ],
        }


_PDF_DIGITIZATION_LIMITS = {
    "words": 50000,
    "frame_comparisons": 2000000,
    "panels": 16,
    "text_visibility_ocr_calls": 32,
    "flattened_points": 500000,
    "section_edge_tests": 25000000,
    "cross_section_samples": 60000,
    "visibility_checks": 2000000,
    "svg_bytes": 4 * 1024 * 1024,
    "csv_bytes": 8 * 1024 * 1024,
    "json_bytes": 16 * 1024 * 1024,
}


def _pdf_spend(budget, resource, count):
    used = budget["used"].get(resource, 0) + int(count)
    if used > budget["limits"][resource]:
        raise ValueError("PDF digitization budget exceeded: " + resource)
    budget["used"][resource] = used


def _pdf_neutral(color, upper=0.65):
    """Select neutral drawing ink, not red/blue coupling display layers."""
    return color is not None and max(color) <= upper and max(color) - min(color) < 0.04


def _pdf_axis_lines(drawings):
    """Axis-aligned centerlines from simple strokes or filled thin rectangles."""
    lines = []
    for path_id, drawing in enumerate(drawings):
        items = drawing.get("items", [])
        filled = drawing.get("fill") is not None
        color = drawing.get("fill") if filled else drawing.get("color")
        if not _pdf_neutral(color) or not 1 <= len(items) <= 4:
            continue
        for item in items:
            coords = []
            if item[0] == "re":
                r = item[1]
                if filled:
                    if r.width <= r.height * 0.02:
                        coords = [((r.x0 + r.x1) / 2, r.y0, (r.x0 + r.x1) / 2, r.y1)]
                    elif r.height <= r.width * 0.02:
                        coords = [(r.x0, (r.y0 + r.y1) / 2, r.x1, (r.y0 + r.y1) / 2)]
                else:
                    coords = [
                        (r.x0, r.y0, r.x0, r.y1),
                        (r.x1, r.y0, r.x1, r.y1),
                        (r.x0, r.y0, r.x1, r.y0),
                        (r.x0, r.y1, r.x1, r.y1),
                    ]
            elif item[0] == "l" and not filled:
                coords = [(*item[1], *item[2])]
            for x0, y0, x1, y1 in coords:
                if abs(x1 - x0) < 1e-4:
                    lines.append(
                        {
                            "axis": "v",
                            "fixed": (x0 + x1) / 2,
                            "lo": min(y0, y1),
                            "hi": max(y0, y1),
                            "path_id": path_id,
                        }
                    )
                elif abs(y1 - y0) < 1e-4:
                    lines.append(
                        {
                            "axis": "h",
                            "fixed": (y0 + y1) / 2,
                            "lo": min(x0, x1),
                            "hi": max(x0, x1),
                            "path_id": path_id,
                        }
                    )
    return lines


def _pdf_visible_text_words(page, words, budget):
    """Bounded native-text contribution, not dark-background occupancy.

    Compare the final render against a text-only-redacted in-memory page copy
    that retains drawings and images. Each native glyph must darken at least
    1% of its bbox pixels by 32/255 or more; unrelated background variation is
    identical in the counterfactual and cannot supply that evidence. These are
    conservative contrast bounds, not calibrated readability/confidence.
    Full text paint bounds must also fit the actual rectangular clip stack;
    unknown clip geometry refuses native text rather than trusting residual ink.
    Later painted bbox overlap is rejected; image-only overlap may pass only
    when local OCR of the final rendered word agrees exactly. Raster samples
    are not a font-semantic certificate.
    """
    import hashlib
    import pymupdf
    import re

    # Only these lexical classes can supply units, tick values, EF or panel labels.
    # Unrelated prose must not exhaust the calibration-visibility work budget.
    evidence_pattern = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\([A-Za-z0-9]{1,3}\))"
    words = [
        word
        for word in words
        if word["text"].strip("()[] ") in ("eV", "meV")
        or word["text"].replace("_", "").replace(" ", "") in ("EF", "E\u1da0")
        or re.fullmatch(evidence_pattern, word["text"].strip().replace("\u2212", "-"))
    ]
    spans, display = page.get_texttrace(), page.get_bboxlog()
    if len(spans) > 50000 or len(display) > 100000:
        raise ValueError("PDF text visibility budget exceeded")
    # Native texttrace boxes survive PDF clipping. Certify the active clip at
    # the actual paint operation, not a clip's outer bbox (which may hide holes).
    # Only a single rectangular path is understood; compound/curved/text/mask
    # clips are unknown and cannot authorize native semantics. q/Q and Form
    # XObject clips are observed by MuPDF itself, not parsed from PDF strings.
    if page.rotation:
        return []  # texttrace is unrotated; do not mix coordinate spaces.
    mupdf = pymupdf.mupdf
    clip_budget_error = None

    def spend_clip():
        # SWIG translates callback exceptions. Preserve the original admission
        # error, not its C++ wrapper, so the public refusal contract is stable.
        nonlocal clip_budget_error
        try:
            _pdf_spend(budget, "visibility_checks", 1)
        except ValueError as exc:
            clip_budget_error = exc
            raise

    class ClipPath(mupdf.FzPathWalker2):
        def __init__(self):
            super().__init__()
            self.ops = []
            self.invalid = False
            for name in ("moveto", "lineto", "curveto", "closepath"):
                getattr(self, "use_virtual_" + name)()

        def add(self, kind, *xy):
            spend_clip()
            if len(self.ops) < 6:
                self.ops.append((kind, xy))
            else:
                self.invalid = True

        def moveto(self, ctx, x, y):
            self.add("m", x, y)

        def lineto(self, ctx, x, y):
            self.add("l", x, y)

        def curveto(self, ctx, *xy):
            self.add("c")

        def closepath(self, ctx):
            self.add("h")

        def rectangle(self, ctm):
            if self.invalid or "".join(k for k, xy in self.ops) not in ("mlllh", "mllllh"):
                return None
            points = [
                pymupdf.Point(*xy) * pymupdf.Matrix(ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f)
                for k, xy in self.ops[:-1]
            ]
            if len(points) == 5:
                if points[-1] != points[0]:
                    return None
                points.pop()
            xs, ys = {p.x for p in points}, {p.y for p in points}
            if (
                len(xs) != 2
                or len(ys) != 2
                or len({tuple(p) for p in points}) != 4
                or not all(np.isfinite(tuple(p)).all() for p in points)
                or not all(
                    a.x == b.x or a.y == b.y for a, b in zip(points, points[1:] + points[:1])
                )
            ):
                return None
            return pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))

    class TextClipDevice(mupdf.FzDevice2):
        def __init__(self):
            super().__init__()
            self.stack = [pymupdf.Rect(page.rect)]
            self.rows = []
            self.unsupported = 0
            self.invalid = False
            for name in (
                "fill_path",
                "stroke_path",
                "fill_text",
                "stroke_text",
                "ignore_text",
                "fill_shade",
                "fill_image",
                "fill_image_mask",
                "clip_path",
                "clip_stroke_path",
                "clip_text",
                "clip_stroke_text",
                "clip_image_mask",
                "pop_clip",
                "begin_mask",
                "end_mask",
                "begin_tile",
                "end_tile",
            ):
                getattr(self, "use_virtual_" + name)()

        def record(self, kind):
            spend_clip()
            self.rows.append((kind, None if self.unsupported else self.stack[-1]))

        def fill_path(self, *args):
            self.record("fill-path")

        def stroke_path(self, *args):
            self.record("stroke-path")

        def fill_text(self, *args):
            self.record("fill-text")

        def stroke_text(self, *args):
            self.record("stroke-text")

        def ignore_text(self, *args):
            self.record("ignore-text")

        def fill_shade(self, *args):
            self.record("fill-shade")

        def fill_image(self, *args):
            self.record("fill-image")

        def fill_image_mask(self, *args):
            self.record("fill-image")

        def clip_path(self, ctx, path, even_odd, ctm, scissor):
            walker = ClipPath()
            mupdf.fz_walk_path(mupdf.FzPath(mupdf.ll_fz_keep_path(path)), walker, walker.m_internal)
            rect = walker.rectangle(ctm)
            self.push(rect)

        def push(self, rect):
            spend_clip()
            parent = self.stack[-1]
            self.stack.append(rect & parent if rect is not None and parent is not None else None)

        def unknown_clip(self, *args):
            self.push(None)

        clip_stroke_path = clip_text = clip_stroke_text = clip_image_mask = unknown_clip

        def pop_clip(self, *args):
            spend_clip()
            if len(self.stack) > 1:
                self.stack.pop()
            else:
                self.invalid = True

        def begin_mask(self, *args):
            self.unsupported += 1

        def end_mask(self, *args):
            self.unsupported -= 1
            self.push(None)

        def begin_tile(self, *args):
            self.unsupported += 1
            return 0

        def end_tile(self, *args):
            self.unsupported -= 1

    device = TextClipDevice()
    try:
        mupdf.fz_run_page(page.this, device, mupdf.FzMatrix(), mupdf.FzCookie())
    except Exception as exc:
        if clip_budget_error is not None:
            raise clip_budget_error from exc
        raise
    finally:
        mupdf.fz_close_device(device)
    # Bind device paint order exactly to the bboxlog/texttrace sequence numbers.
    # An unsupported traversal cannot silently attach a different text's clip.
    if (
        device.invalid
        or device.unsupported
        or len(device.stack) != 1
        or [kind for kind, clip in device.rows] != [kind for kind, box in display]
    ):
        return []
    _pdf_spend(budget, "visibility_checks", len(words) * max(1, len(spans)))
    scale = min(4.0, 2048 / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
    )
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    ink = (rgb.max(axis=2) <= 165) & (rgb.max(axis=2).astype(int) - rgb.min(axis=2) <= 20)
    digest = hashlib.sha256(pix.samples).hexdigest()
    # Remove text only, never the original page or any uploaded bytes. Keeping
    # graphics/images is essential: a black/noisy background must cancel out.
    with pymupdf.open() as without_text:
        without_text.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
        reference = without_text[0]
        reference.add_redact_annot(reference.rect, fill=False, cross_out=False)
        reference.apply_redactions(images=0, graphics=0, text=0)
        reference_pix = reference.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
        )
        if (reference_pix.x, reference_pix.y, reference_pix.width, reference_pix.height) != (
            pix.x,
            pix.y,
            pix.width,
            pix.height,
        ):
            return []  # Coordinate mismatch cannot certify native visibility.
        reference_rgb = np.frombuffer(reference_pix.samples, dtype=np.uint8).reshape(rgb.shape)
        contribution = ink & (
            (reference_rgb.astype(np.int16) - rgb.astype(np.int16)).min(axis=2) >= 32
        )
        reference_digest = hashlib.sha256(reference_pix.samples).hexdigest()
    kept = []
    for word in words:
        rect = pymupdf.Rect(word["bbox"])
        glyphs = []
        for span in spans:
            if not rect.intersects(pymupdf.Rect(span["bbox"])):
                continue
            _pdf_spend(budget, "visibility_checks", len(span["chars"]))
            for char in span["chars"]:
                box = pymupdf.Rect(char[3])
                if (
                    rect.contains(pymupdf.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
                    and not chr(char[0]).isspace()
                ):
                    glyphs.append((char, span))
        if "".join(chr(c[0]) for c, s in glyphs) != word["text"]:
            continue
        visible, image_overlap = bool(glyphs), False
        for char, span in glyphs:
            seqno = span["seqno"]
            active_clip = device.rows[seqno][1] if 0 <= seqno < len(device.rows) else None
            if (
                active_clip is None
                or not active_clip.contains(pymupdf.Rect(display[seqno][1]))
                or span["type"] not in (0, 1)
                or span["opacity"] < 0.99
                or not _pdf_neutral(span["color"])
            ):
                visible = False
                break
            box = pymupdf.Rect(char[3])
            later = display[span["seqno"] + 1 :]
            _pdf_spend(budget, "visibility_checks", len(later))
            overlaps = [
                kind
                for kind, bounds in later
                if kind != "ignore-text" and box.intersects(pymupdf.Rect(bounds))
            ]
            if any(kind != "fill-image" for kind in overlaps):
                visible = False
                break
            image_overlap = image_overlap or bool(overlaps)
            x0, y0 = int(np.floor(box.x0 * scale - pix.x)), int(np.floor(box.y0 * scale - pix.y))
            x1, y1 = int(np.ceil(box.x1 * scale - pix.x)), int(np.ceil(box.y1 * scale - pix.y))
            if not (
                0 <= x0 < x1 <= pix.width
                and 0 <= y0 < y1 <= pix.height
                and (
                    image_overlap
                    or np.count_nonzero(contribution[y0:y1, x0:x1])
                    >= max(2, int(np.ceil(0.01 * (x1 - x0) * (y1 - y0))))
                )
            ):
                visible = False
                break
        if visible and image_overlap:
            # A raster may faithfully reproduce covered native glyphs. In that
            # case require independent recognition of the final rendered word,
            # not merely ink at its stale bbox (e.g. a black overlay).
            _pdf_spend(budget, "text_visibility_ocr_calls", 1)
            clip = (rect + (-2, -2, 2, 2)) & page.rect
            zoom = min(8.0, 512 / max(clip.width, clip.height))
            sample = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), clip=clip, colorspace=pymupdf.csRGB, alpha=False
            )
            image = np.frombuffer(sample.samples, dtype=np.uint8).reshape(
                sample.height, sample.width, 3
            )
            try:
                ocr = _document_ocr_page(image[:, :, ::-1].copy(), page.number + 1)
                visible = (
                    "".join(r["text"] for r in ocr["regions"]) == word["text"]
                    and bool(ocr["regions"])
                    and all(r["ocr_score"] >= 0.9 for r in ocr["regions"])
                )
            except (ImportError, RuntimeError, ValueError):
                visible = False
        if visible:
            kept.append(
                {
                    **word,
                    "visibility_evidence": {
                        "method": (
                            "rendered_word_ocr_matches_native_text_after_image_overlap"
                            if image_overlap
                            else "opaque_native_glyphs_no_later_bbox_overlap_and_text_only_render_difference"
                        ),
                        "source_seqnos": sorted(set(s["seqno"] for c, s in glyphs)),
                        "clip_evidence": {
                            "method": "mupdf_paint_order_full_text_bounds_in_rectangular_clip",
                            "bounds_pdf": [
                                list(device.rows[i][1])
                                for i in sorted(set(s["seqno"] for c, s in glyphs))
                            ],
                            "unknown_clip_policy": "reject_native_text",
                        },
                        "render_scale": scale,
                        "render_rgb_sha256": digest,
                        "text_removed_render_rgb_sha256": reference_digest,
                        "native_contribution_min_channel_delta": 32,
                        "native_contribution_min_bbox_fraction": 0.01,
                        "independently_verified": False,
                        "rendered_word_ocr": (
                            {
                                **ocr,
                                "clip_bbox_pdf": list(clip),
                                "render_scale": zoom,
                                "render_rgb_sha256": hashlib.sha256(sample.samples).hexdigest(),
                            }
                            if image_overlap
                            else None
                        ),
                    },
                }
            )
    return kept


def _pdf_band_frames(drawings, words, page_rect, page_number, budget):
    import re

    lines = _pdf_axis_lines(drawings)
    minimum = min(page_rect.width, page_rect.height) * 0.03
    vertical = [
        line for line in lines if line["axis"] == "v" and line["hi"] - line["lo"] >= minimum
    ]
    horizontal = [
        line for line in lines if line["axis"] == "h" and line["hi"] - line["lo"] >= minimum
    ]
    _pdf_spend(budget, "frame_comparisons", len(vertical) ** 2 * max(1, len(horizontal)))
    frames = []
    for left in vertical:
        for right in vertical:
            w, h = right["fixed"] - left["fixed"], left["hi"] - left["lo"]
            tol = min(w, h) * 0.005
            if w < minimum or max(abs(right[k] - left[k]) for k in ("lo", "hi")) > tol:
                continue
            edges = [
                line
                for line in horizontal
                if abs(line["lo"] - left["fixed"]) <= tol
                and abs(line["hi"] - right["fixed"]) <= tol
            ]
            top = next((line for line in edges if abs(line["fixed"] - left["lo"]) <= tol), None)
            bottom = next((line for line in edges if abs(line["fixed"] - left["hi"]) <= tol), None)
            if top is None or bottom is None:
                continue
            bbox = [left["fixed"], top["fixed"], right["fixed"], bottom["fixed"]]
            if any(max(abs(a - b) for a, b in zip(bbox, f["bbox"])) < tol for f in frames):
                continue
            frames.append(
                {
                    "bbox": bbox,
                    "frame_source_path_ids": sorted(
                        set(line["path_id"] for line in (left, right, top, bottom))
                    ),
                }
            )
    _pdf_spend(budget, "panels", len(frames))
    panels, candidates = [], []
    for index, frame in enumerate(sorted(frames, key=lambda f: (f["bbox"][1], f["bbox"][0]))):
        x0, y0, x1, y1 = frame["bbox"]
        w, h = x1 - x0, y1 - y0
        units = [
            word
            for word in words
            if x0 - 0.3 * w <= word["bbox"][0] < x0
            and word["bbox"][2] <= x0
            and y0 <= sum(word["bbox"][1::2]) / 2 <= y1
            and word["text"].strip("()[] ") in ("eV", "meV")
        ]
        labels = [
            word["text"]
            for word in words
            if re.fullmatch(r"\([A-Za-z0-9]{1,3}\)", word["text"])
            and x0 - 0.1 * w <= word["bbox"][0] <= x0 + 0.3 * w
            and y0 - 0.18 * h <= word["bbox"][1] <= y0 + 0.12 * h
        ]
        distinct_units = {word["text"].strip("()[] ") for word in units}
        unit = next(iter(distinct_units)) if len(distinct_units) == 1 else None
        frame.update(
            panel_id=f"p{page_number}_panel_{index + 1}",
            label=labels[0] if labels else None,
            unit=unit,
            unit_evidence=units,
            calibration_verified=False,
            traces=[],
            status="needs_ocr_review",
            k_unit="relative",
            energy_mapping=None,
            tick_evidence=[],
            fermi_evidence=None,
        )
        if unit == "eV":
            panels.append(frame)
        else:
            frame["reason"] = (
                "conflicting_visible_energy_units"
                if len(distinct_units) > 1
                else "non_electronic_unit_meV"
                if unit == "meV"
                else "missing_ev_unit"
            )
            candidates.append(frame)
    return panels, candidates


def _pdf_flatten_cubic(points, tolerance, depth=0):
    """Bounded adaptive de Casteljau flattening in PDF points."""
    p = np.asarray(points, dtype=float)
    delta = p[3] - p[0]
    length = float(np.linalg.norm(delta))
    if length > 0:
        distance = (
            np.abs(delta[0] * (p[1:3, 1] - p[0, 1]) - delta[1] * (p[1:3, 0] - p[0, 0])) / length
        )
    else:
        distance = np.linalg.norm(p[1:3] - p[0], axis=1)
    if float(distance.max()) <= tolerance:
        return [p[0].tolist(), p[3].tolist()]
    if depth >= 12:
        raise ValueError("PDF cubic flattening exceeds subdivision budget")
    q = (p[:-1] + p[1:]) / 2
    r = (q[:-1] + q[1:]) / 2
    center = (r[0] + r[1]) / 2
    left = _pdf_flatten_cubic([p[0], q[0], r[0], center], tolerance, depth + 1)
    right = _pdf_flatten_cubic([center, r[1], q[2], p[3]], tolerance, depth + 1)
    return left[:-1] + right


def _pdf_linear_subpaths(drawing, tolerance, budget):
    """Retain PDF move-to/closure boundaries instead of stitching paths."""
    subpaths, current = [], []
    for item in drawing.get("items", []):
        if item[0] not in ("l", "c"):
            if current:
                subpaths.append(current)
                current = []
            continue
        piece = (
            [list(item[1]), list(item[2])]
            if item[0] == "l"
            else _pdf_flatten_cubic(item[1:5], tolerance)
        )
        _pdf_spend(budget, "flattened_points", len(piece))
        if current and max(abs(a - b) for a, b in zip(piece[0], current[-1])) > 1e-5:
            subpaths.append(current)
            current = []
        if not current:
            current.append(piece[0])
        current.extend(piece[1:])
    if current:
        subpaths.append(current)
    return subpaths


def _pdf_panel_traces(panel, drawings, budget):
    """Thin filled contours: unique vertical-section midlines, not physical errors."""
    x0, y0, x1, y1 = panel["bbox"]
    w, h = x1 - x0, y1 - y0
    breaks = sorted(
        set(
            (d["rect"].x0 + d["rect"].x1) / 2
            for d in drawings
            if _pdf_neutral(d.get("fill") or d.get("color"))
            and d["rect"].width < 0.01 * w
            and d["rect"].height > 0.9 * h
            and x0 + 0.01 * w < d["rect"].x0 < x1 - 0.01 * w
            and abs(d["rect"].y0 - y0) < 0.02 * h
            and abs(d["rect"].y1 - y1) < 0.02 * h
        )
    )
    panel["k_breaks_relative"] = [(x - x0) / w for x in breaks]
    traces = []
    for path_id, drawing in enumerate(drawings):
        r = drawing["rect"]
        filled = drawing.get("fill") is not None
        color = drawing.get("fill") if filled else drawing.get("color")
        if (
            path_id in panel["frame_source_path_ids"]
            or not _pdf_neutral(color, 0.25)
            or r.x1 <= x0
            or r.x0 >= x1
            or r.y1 <= y0
            or r.y0 >= y1
        ):
            continue
        if (r.width < 0.01 * w and r.height > 0.9 * h) or (
            r.height < 0.01 * h and r.width > 0.9 * w
        ):
            continue  # long straight/dashed guides; flat full-width bands remain ambiguous
        for sub_id, points in enumerate(_pdf_linear_subpaths(drawing, min(w, h) / 16384, budget)):
            polygon = np.asarray(points, dtype=float)
            # Decide eligibility from the source subpath before visibility splits
            # it into tiny dashes; a calibrated guide is still not a band.
            source_dashes = str(drawing.get("dashes") or "")
            dashed = bool(source_dashes.partition("[")[2].partition("]")[0].strip())
            ambiguous_guide = np.ptp(polygon[:, 1]) < 0.01 * h or dashed
            lo, hi = max(x0, float(polygon[:, 0].min())), min(x1, float(polygon[:, 0].max()))
            if hi - lo < 0.03 * w or len(points) < (4 if filled else 2):
                continue  # short glyph-like contours are not promoted to bands
            if filled and not np.array_equal(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])
            a, b = polygon[:-1], polygon[1:]
            runs, current, thicknesses = [], [], []
            # Strictly interior x samples avoid double-counting closed polygon vertices.
            count = max(3, int(np.ceil((hi - lo) / w * 512)))
            _pdf_spend(budget, "cross_section_samples", count)
            _pdf_spend(budget, "section_edge_tests", count * len(a))
            for x in np.linspace(lo, hi, count + 2)[1:-1]:
                if current and any(current[-1][0] <= boundary <= x for boundary in breaks):
                    runs.append((current, thicknesses))
                    current, thicknesses = [], []
                if any(abs(x - boundary) <= w / 1024 for boundary in breaks):
                    continue
                active = ((a[:, 0] <= x) & (x < b[:, 0])) | ((b[:, 0] <= x) & (x < a[:, 0]))
                aa, bb = a[active], b[active]
                yy = np.sort(
                    aa[:, 1] + (x - aa[:, 0]) * (bb[:, 1] - aa[:, 1]) / (bb[:, 0] - aa[:, 0])
                )
                width = (
                    float(yy[1] - yy[0])
                    if filled and len(yy) == 2
                    else float(drawing.get("width") or 0)
                )
                unique = len(yy) == (2 if filled else 1)
                if unique and 0 < width <= 0.02 * h and y0 < yy.mean() < y1:
                    current.append([float(x), float(yy.mean())])
                    thicknesses.append(width)
                elif current:
                    runs.append((current, thicknesses))
                    current, thicknesses = [], []
            if current:
                runs.append((current, thicknesses))
            for run_id, (run, widths) in enumerate(runs):
                if len(run) < 3:
                    continue
                traces.append(
                    {
                        "trace_id": f"{panel['panel_id']}_path{path_id}_sub{sub_id}_fragment{run_id}",
                        "source_path_id": path_id,
                        "source_subpath_id": sub_id,
                        "geometry_kind": "filled_contour_midline"
                        if filled
                        else "stroke_centerline",
                        "source_seqno": drawing.get("seqno"),
                        "points_pdf": run,
                        "k_relative": [(p[0] - x0) / w for p in run],
                        "k_unit": "relative",
                        "energy_eV": None,
                        "band_data": None,
                        "source_dashes": source_dashes,
                        "quantitative_exclusion_reason": (
                            "flat_or_dashed_source_ambiguous_with_guide"
                            if ambiguous_guide
                            else None
                        ),
                        "segments": [{"start": 0, "stop": len(run), "segment_id": 0}],
                        "drawing_thickness_pdf_points": [min(widths), max(widths)],
                        "physical_energy_uncertainty_eV": None,
                        "uncertainty_reasons": [
                            "visible_fragment_not_complete_band",
                            "high_symmetry_boundaries_not_interpolated",
                            "unique_thin_cross_section_only_no_branch_stitching",
                            "drawing_thickness_is_not_physical_uncertainty",
                        ],
                    }
                )
    return traces


def _pdf_visible_fragments(page, panel, words, budget):
    """Reject hidden vector points against a bounded rendering; do not fill holes."""
    import pymupdf

    bbox = pymupdf.Rect(panel["bbox"])
    scale = min(6.0, 2048 / max(bbox.width, bbox.height))
    if (np.ceil(bbox.width * scale) + 1) * (np.ceil(bbox.height * scale) + 1) > 4_000_000:
        raise ValueError("PDF panel visibility render exceeds pixel budget")
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale), clip=bbox, colorspace=pymupdf.csRGB, alpha=False
    )
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    mask = (image.max(axis=2) <= 165) & (image.max(axis=2).astype(int) - image.min(axis=2) <= 20)
    before = sum(len(t["points_pdf"]) for t in panel["traces"])
    words = [word for word in words if pymupdf.Rect(word["bbox"]).intersects(bbox)]
    _pdf_spend(budget, "visibility_checks", before * max(1, len(words)))
    result = []
    for trace in panel["traces"]:
        runs, current = [], []
        for index, (x, y) in enumerate(trace["points_pdf"]):
            ix, iy = int(x * scale - pix.x), int(y * scale - pix.y)
            text_overlap = any(
                b["bbox"][0] <= x <= b["bbox"][2] and b["bbox"][1] <= y <= b["bbox"][3]
                for b in words
            )
            # A half-pixel horizontal guard rejects ambiguous occlusion edges.
            left = int(x * scale - pix.x - 0.5)
            right = int(x * scale - pix.x + 0.5)
            visible = (
                0 <= left <= ix <= right < pix.width
                and 0 <= iy < pix.height
                and bool(mask[iy, left : right + 1].all())
                and not text_overlap
            )
            if visible:
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        for run_id, indices in enumerate(runs):
            if len(indices) < 3:
                continue
            item = {
                **trace,
                "trace_id": trace["trace_id"] + f"_visible{run_id}",
                "points_pdf": [trace["points_pdf"][i] for i in indices],
                "k_relative": [trace["k_relative"][i] for i in indices],
                "segments": [{"start": 0, "stop": len(indices), "segment_id": 0}],
                "uncertainty_reasons": trace["uncertainty_reasons"]
                + ["occlusion_gaps_not_interpolated"],
            }
            result.append(item)
    panel["visibility_evidence"] = {
        "method": "rendered_neutral_ink_at_vector_midline",
        "render_scale": scale,
        "image_size": [pix.width, pix.height],
        "clip_bbox_pdf": list(bbox),
        "candidate_point_count": before,
        "retained_point_count": sum(len(t["points_pdf"]) for t in result),
        "limitation": "Finite raster sampling is not an exact PDF clipping or layer-identity proof.",
    }
    panel["traces"] = result


def _pdf_tick_strip_ocr(page, panel):
    """OCR only bounded axis text locally, retaining the raster and model provenance."""
    import base64
    import hashlib
    from importlib.metadata import version
    import pymupdf
    import rapidocr_onnxruntime

    x0, y0, x1, y1 = panel["bbox"]
    w, h = x1 - x0, y1 - y0
    clip = pymupdf.Rect(x0 - 0.2 * w, y0 - 0.08 * h, x0 - 0.005 * w, y1 + 0.08 * h) & page.rect
    if clip.is_empty:
        raise ValueError("empty PDF tick-strip")
    scale = min(8.0, 2048 / max(clip.width, clip.height))
    if (np.ceil(clip.width * scale) + 1) * (np.ceil(clip.height * scale) + 1) > 4_000_000:
        raise ValueError("PDF tick-strip exceeds pixel budget")
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale), clip=clip, colorspace=pymupdf.csRGB, alpha=False
    )
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    evidence = _document_ocr_page(rgb[:, :, ::-1].copy(), page.number + 1)
    model_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"
    evidence.update(
        clip_bbox_pdf=list(clip),
        render_scale=scale,
        raster_origin_pixels=[pix.x, pix.y],
        calibration_verified=False,
        engine_version=version("rapidocr-onnxruntime"),
        model_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_dir.glob("*.onnx"))
        },
        png_base64=base64.b64encode(pix.tobytes("png")).decode(),
    )
    words = []
    for region in evidence["regions"]:
        box = np.asarray(region["box"], dtype=float)
        low = (box.min(axis=0) + [pix.x, pix.y]) / scale
        high = (box.max(axis=0) + [pix.x, pix.y]) / scale
        words.append(
            {
                "bbox": [*low.tolist(), *high.tolist()],
                "text": region["text"],
                "ocr_score": region["ocr_score"],
                "engine": evidence["engine"],
            }
        )
    return evidence, words


def _pdf_calibrate_panel(panel, drawings, words):
    """Fit numeric tick/geometry correspondences; require an explicit EF label."""
    import re

    x0, y0, x1, y1 = panel["bbox"]
    w, h = x1 - x0, y1 - y0
    lines = _pdf_axis_lines(drawings)
    tick_lines = [
        line
        for line in lines
        if line["axis"] == "h"
        and 0.002 * w <= line["hi"] - line["lo"] <= 0.06 * w
        and line["lo"] - 0.01 * w <= x0 <= line["hi"] + 0.01 * w
        and y0 - 0.01 * h <= line["fixed"] <= y1 + 0.01 * h
    ]
    panel["tick_geometry_candidates"] = tick_lines
    reference_lines = []
    for path_id, drawing in enumerate(drawings):
        r = drawing["rect"]
        if (
            _pdf_neutral(drawing.get("fill") or drawing.get("color"))
            and r.width >= 0.85 * w
            and r.height <= 0.012 * h
            and abs(r.x0 - x0) < 0.03 * w
            and abs(r.x1 - x1) < 0.03 * w
            and y0 + 0.03 * h < (r.y0 + r.y1) / 2 < y1 - 0.03 * h
        ):
            reference_lines.append((path_id, (r.y0 + r.y1) / 2))
    panel["reference_line_candidates"] = [
        {"source_path_id": pid, "y_pdf": y} for pid, y in reference_lines
    ]
    ticks = []
    for word in words:
        bx0, by0, bx1, by1 = word["bbox"]
        text = word["text"].strip().replace("\u2212", "-")
        if not (
            x0 - 0.2 * w <= bx0 < x0
            and bx1 <= x0
            and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
        ):
            continue
        matches = [
            line
            for line in tick_lines
            if abs(line["fixed"] - (by0 + by1) / 2) <= min(0.03 * h, 0.6 * (by1 - by0))
        ]
        if len(matches) != 1 or not np.isfinite(float(text)):
            continue
        line = matches[0]
        ticks.append(
            {
                "text": word["text"],
                "value_eV": float(text),
                "y_pdf": line["fixed"],
                "text_bbox": word["bbox"],
                "engine": word["engine"],
                "ocr_score": word.get("ocr_score"),
                "source_path_id": line["path_id"],
                "visibility_evidence": word.get(
                    "visibility_evidence", {"method": "rendered_tick_strip_ocr"}
                ),
            }
        )
    panel["tick_evidence"] = sorted(ticks, key=lambda tick: tick["y_pdf"])
    panel["calibration_warnings"] = []
    if len(ticks) < 3:
        panel["calibration_warnings"].append("at_least_three_numeric_tick_geometry_pairs_required")
        return
    yy = np.asarray([t["y_pdf"] for t in panel["tick_evidence"]])
    values = np.asarray([t["value_eV"] for t in panel["tick_evidence"]])
    differences = np.diff(values)
    if (
        np.any(np.diff(yy) <= 0.005 * h)
        or np.ptp(yy) < 0.5 * h
        or not (np.all(differences > 0) or np.all(differences < 0))
    ):
        panel["calibration_warnings"].append("tick_correspondence_is_ambiguous_or_not_monotone")
        return
    u = (yy - y0) / h
    slope, intercept = np.linalg.lstsq(np.column_stack([u, np.ones_like(u)]), values, rcond=None)[0]
    residual = float(np.max(np.abs(values - (slope * u + intercept))) / np.ptp(values))
    panel["calibration_candidate"] = {
        "slope_eV_per_pdf_point": float(slope / h),
        "axis_value_at_top_eV": float(intercept),
        "max_residual_fraction_of_tick_span": residual,
        "tick_y_domain_pdf": [float(yy.min()), float(yy.max())],
    }
    if residual > 0.01 or not np.isfinite(slope) or slope == 0:
        panel["calibration_warnings"].append("nonlinear_or_inconsistent_tick_ruler")
        return
    matches = []
    for word in words:
        if word["text"].replace("_", "").replace(" ", "") not in ("EF", "E\u1da0"):
            continue
        bx0, by0, bx1, by1 = word["bbox"]
        if not (x0 - 0.2 * w <= bx0 and bx1 <= x1 + 0.2 * w):
            continue
        for path_id, y in reference_lines:
            if abs(y - (by0 + by1) / 2) < min(0.04 * h, by1 - by0):
                matches.append(
                    {
                        "label_text": word["text"],
                        "label_bbox": word["bbox"],
                        "engine": word["engine"],
                        "ocr_score": word.get("ocr_score"),
                        "y_pdf": y,
                        "source_path_id": path_id,
                        "visibility_evidence": word.get(
                            "visibility_evidence", {"method": "rendered_tick_strip_ocr"}
                        ),
                    }
                )
    if len(matches) != 1:
        panel["calibration_warnings"].append("one_explicit_fermi_label_and_reference_line_required")
        return
    fermi = matches[0]
    if not yy.min() <= fermi["y_pdf"] <= yy.max():
        panel["calibration_warnings"].append("fermi_reference_outside_tick_domain")
        return
    panel["fermi_evidence"] = fermi
    panel["energy_mapping"] = {
        **panel["calibration_candidate"],
        "axis_value_at_fermi_eV": float(slope * (fermi["y_pdf"] - y0) / h + intercept),
        "energy_reference": "E_minus_EF",
        "fermi_y_pdf": fermi["y_pdf"],
    }
    panel["status"] = "digitized_unverified"
    for trace in panel["traces"]:
        if trace.get("quantitative_exclusion_reason") is not None:
            trace["uncertainty_reasons"].append(trace["quantitative_exclusion_reason"])
            continue
        y = np.asarray(trace["points_pdf"])[:, 1]
        if np.any(y < yy.min()) or np.any(y > yy.max()):
            trace["uncertainty_reasons"].append(
                "fragment_outside_numeric_tick_domain_no_extrapolation"
            )
            continue
        trace["energy_eV"] = ((y - fermi["y_pdf"]) * slope / h).tolist()
        trace["band_data"] = {
            "energies_eV": [trace["energy_eV"]],
            "k_distance": trace["k_relative"],
            "segment_ids": [0] * len(y),
            "fermi_eV": 0.0,
            "k_unit": "relative",
        }
        trace["uncertainty_reasons"].append(
            "automatic_image_calibration_not_independently_verified"
        )
    if not any(trace["band_data"] is not None for trace in panel["traces"]):
        panel["status"] = "geometry_only"


def _pdf_export_fragments(panels, page_rect, budget):
    """Export measured visible samples only; SVG contains no raw PDF markup."""
    import csv
    import io
    from html import escape

    if panels:
        left = min(p["bbox"][0] for p in panels)
        top = min(p["bbox"][1] for p in panels)
        right = max(p["bbox"][2] for p in panels)
        bottom = max(p["bbox"][3] for p in panels)
        padding = max(right - left, bottom - top) * 0.04
        view = (
            left - padding,
            top - padding - 6,
            right - left + 2 * padding,
            bottom - top + 2 * padding + 6,
        )
    else:
        view = (0, 0, page_rect.width, page_rect.height)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view[0]:g} {view[1]:g} {view[2]:g} {view[3]:g}">',
        "<title>Unverified visible electronic band fragments; k is relative</title>",
    ]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "panel_id",
            "trace_id",
            "source_path_id",
            "source_subpath_id",
            "point_index",
            "x_pdf",
            "y_pdf",
            "k_relative",
            "energy_eV",
        ]
    )
    for panel in panels:
        x0, y0, x1, y1 = panel["bbox"]
        pid = escape(panel["panel_id"], quote=True)
        svg.extend(
            [
                f'<defs><clipPath id="clip_{pid}"><rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}"/></clipPath></defs>',
                f'<text x="{x0:g}" y="{y0 - 3:g}" font-size="6">{escape(panel["label"] or panel["panel_id"])}: {escape(panel["status"])}</text>',
                f'<rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}" fill="none" stroke="black" stroke-width="0.3"/>',
                f'<g clip-path="url(#clip_{pid})">',
            ]
        )
        for trace in panel["traces"]:
            points = " ".join(f"{x:.6f},{y:.6f}" for x, y in trace["points_pdf"])
            svg.append(
                f'<polyline id="{escape(trace["trace_id"], quote=True)}" points="{points}" fill="none" stroke="black" stroke-width="0.25"/>'
            )
            energies = trace["energy_eV"] or [None] * len(trace["points_pdf"])
            for i, ((x, y), k, energy) in enumerate(
                zip(trace["points_pdf"], trace["k_relative"], energies)
            ):
                writer.writerow(
                    [
                        panel["panel_id"],
                        trace["trace_id"],
                        trace["source_path_id"],
                        trace["source_subpath_id"],
                        i,
                        x,
                        y,
                        k,
                        energy,
                    ]
                )
        svg.append("</g>")
    svg.append("</svg>")
    svg_text, csv_text = "\n".join(svg), stream.getvalue()
    _pdf_spend(budget, "svg_bytes", len(svg_text.encode()))
    _pdf_spend(budget, "csv_bytes", len(csv_text.encode()))
    return svg_text, csv_text


def digitize_pdf_band_panels(payload: bytes, *, page_number: int = 1) -> dict:
    """Bounded visible PDF geometry; absent calibration never implies eV."""
    import json

    budget = {"limits": dict(_PDF_DIGITIZATION_LIMITS), "used": {}}
    import hashlib
    import pymupdf

    if not isinstance(payload, bytes) or not 0 < len(payload) <= 10 * 1024 * 1024:
        raise ValueError("PDF payload must be nonempty bytes, at most 10 MiB")
    if type(page_number) is not int or page_number < 1:
        raise ValueError("page_number must be a positive nonboolean integer")
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        if page_number > len(document):
            raise ValueError("page_number exceeds document pages")
        page = document[page_number - 1]
        if page.rotation != 0:
            raise ValueError(
                "rotated PDF pages require an explicit coordinate transform; not supported"
            )
        if (
            not np.isfinite(list(page.rect)).all()
            or page.rect.is_empty
            or (np.ceil(page.rect.width * 2.5) + 1) * (np.ceil(page.rect.height * 2.5) + 1)
            > 4096 * 4096
        ):
            raise ValueError("PDF page exceeds geometry budget")
        drawings = page.get_drawings()
        if len(drawings) > 50000 or sum(len(d.get("items", [])) for d in drawings) > 500000:
            raise ValueError("PDF decoded geometry exceeds budget")
        words = [
            {"bbox": [float(v) for v in w[:4]], "text": str(w[4]), "engine": "pymupdf_text"}
            for w in page.get_text("words")
        ]
        _pdf_spend(budget, "words", len(words))
        native_words = words  # Raw bboxes may conservatively exclude geometry, never calibrate it.
        words = _pdf_visible_text_words(page, words, budget)
        panels, candidates = _pdf_band_frames(drawings, words, page.rect, page_number, budget)
        for panel in panels:
            panel["traces"] = _pdf_panel_traces(panel, drawings, budget)
            _pdf_visible_fragments(page, panel, native_words, budget)
            _pdf_calibrate_panel(panel, drawings, words)
            if panel["energy_mapping"] is None:
                try:
                    evidence, ocr_words = _pdf_tick_strip_ocr(page, panel)
                    panel["tick_strip_ocr"] = evidence
                    additions = [
                        o
                        for o in ocr_words
                        if o["ocr_score"] >= 0.9
                        and not any(
                            o["text"] == t["text"]
                            and abs(sum(o["bbox"][1::2]) - sum(t["bbox"][1::2]))
                            < t["bbox"][3] - t["bbox"][1]
                            for t in words
                        )
                    ]
                    _pdf_calibrate_panel(panel, drawings, words + additions)
                except (ImportError, RuntimeError, ValueError) as exc:
                    panel["tick_strip_ocr"] = {"status": "ocr_unavailable", "error": str(exc)[:400]}
        clean_svg, csv_text = _pdf_export_fragments(panels, page.rect, budget)
        numeric = bool(panels) and all(
            any(t["band_data"] is not None for t in p["traces"]) for p in panels
        )
        result = {
            "status": "digitized_unverified" if numeric else "geometry_only",
            "page_number": page_number,
            "resource_budget": budget,
            "total_pages": len(document),
            "panels": panels,
            "band_data": None,
            "clean_svg": clean_svg,
            "csv_text": csv_text,
            "needs_ocr_review": any(p["energy_mapping"] is None for p in panels)
            or any(p["reason"] == "missing_ev_unit" for p in candidates),
            "limitations": [
                "Only visible neutral-ink fragments, not complete physical band identities.",
                "Filled contours shorter than 3% panel width and full-width flat lines are ambiguous and omitted.",
                "No stitching across subpaths, high-symmetry guides, ambiguous sections or occlusion gaps.",
                "Dashed source paths and source subpaths with less than 1% panel-height variation remain geometry-only: flat bands and guides are not distinguished.",
                "OCR scores and drawing thickness are not physical uncertainty or calibrated confidence.",
                "Red/non-neutral coupling display layers are excluded; no coupling or lambda values inferred.",
                "Automatic tick/EF calibration is unverified, not an independent human audit.",
                "Native text requires full paint bounds contained in the actual rectangular clip stack, opaque neutral glyphs and rendered contribution. Unknown/compound/curved clips refuse native text. Later painted bbox overlap is rejected, except image-only overlap with exact local rendered-word OCR agreement; benign clips/overlaps may be rejected. This is not font-semantic certification.",
                "Relative k only: no effective mass, full-BZ gap, material insulation or Tc inference.",
            ],
            "calibration_candidates": candidates,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "confidence": None,
            "ood_flag": None,
            "human_audited": False,
            "calibration_verified": False,
        }
        raster_regions = [
            {
                "bbox_pdf_points": list(i["bbox"]),
                "width": i["width"],
                "height": i["height"],
                "page_number": page_number,
                "source_sha256": result["source_sha256"],
            }
            for i in page.get_image_info()
            if i["width"] >= 40 and i["height"] >= 40
        ]
        result["raster_regions"] = raster_regions
        result["carrier_kind"] = (
            "hybrid" if raster_regions and panels else "raster" if raster_regions else "vector"
        )
        if raster_regions and not panels:
            result.update(
                status="requires_raster_review",
                needs_ocr_review=True,
                next_action="Inspect the embedded raster(s); submit a source-bound pdf_point ROI and panel_kind. Electronic bands, phonons and transport must be selected separately.",
                empty_result_reason="no_vector_band_panels_with_embedded_raster",
            )
        elif not panels:
            result["empty_result_reason"] = "no_vector_band_panels"
        # Reserve space for the counter itself; recorded usage is a conservative byte bound.
        _pdf_spend(budget, "json_bytes", len(json.dumps(result, allow_nan=False).encode()) + 64)
        return result


@dataclass
class ParsedBandData:
    """Geometry extracted from a band-structure carrier."""

    source_path: str
    source_type: str
    k: np.ndarray
    energy: np.ndarray
    band_hint: np.ndarray
    fermi_energy: float = 0.0
    kpath_labels: List[Dict[str, Any]] = field(default_factory=list)
    skeleton_pixels: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None
    frame_gaps: Optional[np.ndarray] = None
    frame_numbers: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_points(self) -> np.ndarray:
        return np.stack([self.k, self.energy], axis=1).astype(np.float32)


class MultiFormatParser:
    """Extract `(k, E)` band traces from PDF/EPS, raster images, and videos."""

    def __init__(
        self,
        image_height_ev: float = 16.0,
        target_k_points: int = 128,
        max_video_frames: int = 48,
        threshold: int = 190,
        detector_path: Optional[
            str
        ] = "artifacts/models/vision_detector/band_plot_yolov8_pose_best.pt",
        detector_conf: float = 0.05,
    ) -> None:
        self.image_height_ev = float(image_height_ev)
        self.target_k_points = int(target_k_points)
        self.max_video_frames = int(max_video_frames)
        self.threshold = int(threshold)
        self.detector_path = str(detector_path) if detector_path else ""
        self.detector_conf = float(detector_conf)
        self._detector = None

    def parse(self, path: str) -> ParsedBandData:
        suffix = Path(path).suffix.lower()
        if suffix in VECTOR_SUFFIXES:
            return self.parse_vector(path)
        if suffix in IMAGE_SUFFIXES:
            return self.parse_image(path)
        if suffix in VIDEO_SUFFIXES:
            return self.parse_video(path)
        raise ValueError(f"Unsupported input format: {suffix}")

    def parse_vector(self, path: str) -> ParsedBandData:
        """Parse PDF path/line drawings when PyMuPDF is available."""

        suffix = Path(path).suffix.lower()
        if suffix == ".eps":
            raise RuntimeError(
                "EPS vector extraction requires converting EPS to PDF before parsing."
            )

        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF vector parsing requires PyMuPDF. Run: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(path)
        if doc.page_count == 0:
            raise ValueError(f"No pages found in {path}")

        page = doc[0]
        drawings = page.get_drawings()
        points: List[List[float]] = []
        for drawing in drawings:
            for item in drawing.get("items", []):
                op = item[0]
                if op == "l":
                    p0, p1 = item[1], item[2]
                    points.extend([[float(p0.x), float(p0.y)], [float(p1.x), float(p1.y)]])
                elif op == "re":
                    rect = item[1]
                    points.extend(
                        [
                            [float(rect.x0), float(rect.y0)],
                            [float(rect.x1), float(rect.y0)],
                            [float(rect.x1), float(rect.y1)],
                            [float(rect.x0), float(rect.y1)],
                        ]
                    )
                elif op == "c":
                    for p in item[1:]:
                        points.append([float(p.x), float(p.y)])

        if len(points) < 4:
            raise ValueError("No usable vector path points were detected in the PDF")

        xy = np.asarray(points, dtype=np.float32)
        vector_score = float(np.clip(len(xy) / 2000.0, 0.0, 0.9))
        vector_quality = {
            "score": vector_score,
            "level": cv_quality_level(vector_score),
            "detector_status": "not_applicable_vector",
            "components": {
                "panel_detection": "pdf_vector_drawings",
                "vector_points": int(len(xy)),
            },
        }
        return self._coords_to_band_data(
            path,
            "vector",
            xy,
            image_shape=(float(page.rect.height), float(page.rect.width)),
            metadata={
                "vector_points": int(len(xy)),
                "parser": "PyMuPDF",
                "cv_quality": vector_quality,
            },
        )

    def parse_image(self, path: str) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for image Plot-to-Physics parsing"
            ) from exc

        image = self._imread_unicode(path)
        if image is None:
            raise FileNotFoundError(path)

        panel, offset, panel_meta = self._select_physical_panel(image)
        binary, mask_mode = self._auto_band_mask(panel)
        skeleton = self._skeletonize(binary)
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 8:
            raise ValueError("No band-like skeleton was detected")

        overlay = image.copy()
        x0, y0 = offset
        self._draw_detection_overlay(overlay, panel_meta)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        cv_quality = self._compute_cv_quality(panel_meta, coords, panel.shape)
        parsed = self._coords_to_band_data(
            path,
            "image",
            coords,
            image_shape=panel.shape[:2],
            skeleton_pixels=coords,
            overlay_image=overlay,
            panel_image=panel,
            metadata={
                "skeleton_pixels": int(len(coords)),
                "mask_mode": mask_mode,
                "cv_quality": cv_quality,
                **panel_meta,
            },
        )
        self._draw_skeleton_overlay(overlay, panel.shape[:2], offset, skeleton)
        parsed.overlay_image = overlay
        return parsed

    def _select_physical_panel(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        """Prefer the complete coordinate-frame panel over a local detector crop."""

        geometry_panel = self._extract_band_panel(image)
        detected = self._detect_with_vision_model(image)
        if detected is None:
            return geometry_panel

        g_panel, g_offset, g_meta = geometry_panel
        d_panel, d_offset, d_meta = detected
        g_area = int(g_panel.shape[0] * g_panel.shape[1])
        d_area = int(d_panel.shape[0] * d_panel.shape[1])
        g_frame = g_meta.get("frame_crop") not in (None, "none") or "frame" in str(
            g_meta.get("panel_detection", "")
        )

        # Detector boxes can latch onto inset labels or small subplots. If the
        # geometry detector found a much larger axis-bounded frame, that is the
        # physically valid coordinate system for Plot-to-Physics.
        if g_frame and g_area >= max(int(d_area * 1.35), d_area + 5000):
            merged = {
                **g_meta,
                "panel_detection": f"{g_meta.get('panel_detection', 'geometry_frame')}_preferred_over_yolo_crop",
                "vision_detector_all_panels": d_meta.get("vision_detector_all_panels", []),
                "vision_detector_panel_count": d_meta.get("vision_detector_panel_count", 0),
                "rejected_yolo_panel_bbox": d_meta.get("panel_bbox"),
                "rejected_yolo_reason": "geometry frame is larger and axis-bounded",
            }
            return g_panel, g_offset, merged
        return detected

    def _compute_cv_quality(
        self,
        panel_meta: Dict[str, Any],
        skeleton_pixels: np.ndarray,
        panel_shape: tuple[int, ...],
    ) -> Dict[str, Any]:
        """Aggregate extraction quality into a 0-1 score with a traffic-light level.

        Components (all measured, no fabricated detector claims):
          - panel source reliability (geometry frame vs fallback) + detector conf
            only when vision weights exist (constitution §9: optional/unverified);
          - frame crop quality;
          - skeleton density and column (k-direction) occupancy;
          - panel resolution and skeleton point count.
        """
        height = float(panel_shape[0])
        width = float(panel_shape[1])
        area = max(height * width, 1.0)
        n_pixels = int(len(skeleton_pixels))

        source = str(panel_meta.get("panel_detection", "fallback_full_image"))
        base = 0.3
        for key, val in _PANEL_SOURCE_SCORE.items():
            if source.startswith(key):
                base = val
                break
        detector_conf = panel_meta.get("vision_detector_confidence")
        if detector_conf is not None:
            detector_status = "detector_confidence_used"
            detector_score = float(np.clip(float(detector_conf), 0.0, 1.0))
        else:
            detector_status = "optional_missing_score_excluded"
            detector_score = base
        panel_score = 0.6 * base + 0.4 * detector_score

        frame_crop = panel_meta.get("frame_crop")
        frame_score = 0.8 if frame_crop not in (None, "none") else 0.4

        if n_pixels > 0:
            density = float(n_pixels / area)
            xs = skeleton_pixels[:, 0]
            col_occ = float(len(np.unique(np.clip(xs, 0, width - 1).astype(int)))) / max(width, 1.0)
        else:
            density, col_occ = 0.0, 0.0
        density_score = float(np.clip(density / 0.02, 0.0, 1.0))
        res_score = float(np.clip(height / 400.0, 0.0, 1.0))
        count_score = float(np.clip(n_pixels / 2000.0, 0.0, 1.0))

        score = (
            0.25 * panel_score
            + 0.15 * frame_score
            + 0.20 * density_score
            + 0.20 * col_occ
            + 0.12 * res_score
            + 0.08 * count_score
        )
        score = float(np.clip(score, 0.0, 1.0))
        return {
            "score": score,
            "level": cv_quality_level(score),
            "detector_status": detector_status,
            "components": {
                "panel_detection": source,
                "panel_score": panel_score,
                "frame_crop": frame_crop if frame_crop not in (None, "none") else "none",
                "frame_score": frame_score,
                "skeleton_density": density,
                "column_occupancy": col_occ,
                "panel_resolution_score": res_score,
                "skeleton_pixels": n_pixels,
            },
        }

    def _detect_with_vision_model(
        self, image: np.ndarray
    ) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        """Use the trained pose model as a panel locator, not as the physics brain."""

        detector_path = Path(self.detector_path) if self.detector_path else None
        if detector_path is None or not detector_path.exists():
            return None

        try:
            detector = self._load_detector(detector_path)
            if detector is None:
                return None
            results = detector.predict(image, conf=self.detector_conf, verbose=False)
        except Exception:
            return None

        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        conf = (
            boxes.conf.cpu().numpy().astype(float)
            if getattr(boxes, "conf", None) is not None
            else np.ones(len(xyxy))
        )
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        h, w = image.shape[:2]
        order = np.argsort(-(conf * np.sqrt(np.maximum(areas, 1.0))))
        candidates: List[Dict[str, Any]] = []
        best_payload = None
        pad = max(4, int(0.01 * max(w, h)))

        for idx in order:
            raw_x0, raw_y0, raw_x1, raw_y1 = xyxy[int(idx)]
            x0 = max(int(round(raw_x0)) - pad, 0)
            y0 = max(int(round(raw_y0)) - pad, 0)
            x1 = min(int(round(raw_x1)) + pad, w)
            y1 = min(int(round(raw_y1)) + pad, h)

            box_w = x1 - x0
            box_h = y1 - y0
            if box_w < max(70, int(0.08 * w)) or box_h < max(60, int(0.08 * h)):
                continue
            if box_w * box_h > 0.96 * w * h:
                continue

            panel = image[y0:y1, x0:x1].copy()
            panel, inner_offset, frame_meta = self._crop_to_plot_frame(panel)
            px0 = x0 + inner_offset[0]
            py0 = y0 + inner_offset[1]
            px1 = px0 + panel.shape[1]
            py1 = py0 + panel.shape[0]
            keypoints = self._extract_detection_keypoints(result, int(idx), px0, py0, panel.shape)
            if keypoints and not all(
                bool(item.get("inside_panel", False)) for item in keypoints.values()
            ):
                keypoints = {}
            candidate_score = self._score_detected_panel(panel, float(conf[int(idx)]), frame_meta)

            candidate = {
                "panel_bbox": [int(px0), int(py0), int(px1), int(py1)],
                "confidence": float(conf[int(idx)]),
                "candidate_score": float(candidate_score),
                "raw_bbox": [float(raw_x0), float(raw_y0), float(raw_x1), float(raw_y1)],
                "keypoints": keypoints,
                **frame_meta,
            }
            candidates.append(candidate)
            if best_payload is None or candidate_score > best_payload[2].get(
                "candidate_score", -1e9
            ):
                best_payload = (panel, (int(px0), int(py0)), candidate)

        if best_payload is None:
            return None

        panel, offset, selected = best_payload
        meta = {
            "panel_bbox": selected["panel_bbox"],
            "panel_detection": "vision_detector_yolov8_pose",
            "vision_detector_path": str(detector_path),
            "vision_detector_confidence": float(selected["confidence"]),
            "vision_detector_candidate_score": float(
                selected.get("candidate_score", selected["confidence"])
            ),
            "vision_detector_keypoints": selected.get("keypoints", {}),
            "vision_detector_all_panels": candidates,
            "vision_detector_panel_count": len(candidates),
            **{
                k: v
                for k, v in selected.items()
                if k not in {"panel_bbox", "confidence", "raw_bbox", "keypoints"}
            },
        }
        return panel, offset, meta

    def _score_detected_panel(
        self, panel: np.ndarray, confidence: float, frame_meta: Dict[str, Any]
    ) -> float:
        """Rank detector boxes by detector confidence and band-plot geometry.

        DOS panels and text boxes can receive high detector confidence on real
        paper composites. A true band plot usually has a recoverable rectangular
        k-path frame and curve pixels spread across many x positions.
        """

        score = float(confidence)
        if frame_meta.get("frame_crop") not in (None, "none"):
            score += 0.18
        try:
            mask, _ = self._auto_band_mask(panel)
            height, width = mask.shape[:2]
            if height > 0 and width > 0:
                col_occ = float(((mask > 0).sum(axis=0) > 0).mean())
                row_occ = float(((mask > 0).sum(axis=1) > 0).mean())
                density = float((mask > 0).sum() / max(height * width, 1))
                score += 0.22 * col_occ + 0.08 * row_occ
                if density < 0.002:
                    score -= 0.15
                if col_occ < 0.18:
                    score -= 0.18
        except Exception:
            pass
        return score

    def _load_detector(self, detector_path: Path) -> Any:
        if self._detector is not None:
            return self._detector
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            return None
        self._detector = YOLO(str(detector_path))
        return self._detector

    def _extract_detection_keypoints(
        self,
        result: Any,
        index: int,
        x0: int,
        y0: int,
        panel_shape: Optional[tuple[int, int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
            return {}
        xy = keypoints_obj.xy.cpu().numpy()
        conf_arr = None
        if getattr(keypoints_obj, "conf", None) is not None:
            conf_arr = keypoints_obj.conf.cpu().numpy()
        if index >= len(xy) or xy[index].shape[0] < 2:
            return {}

        names = ("vbm", "cbm")
        out: Dict[str, Dict[str, float]] = {}
        for kp_idx, name in enumerate(names):
            px, py = xy[index][kp_idx]
            item = {"x": float(px - x0), "y": float(py - y0)}
            if conf_arr is not None and index < len(conf_arr) and kp_idx < conf_arr[index].shape[0]:
                item["confidence"] = float(conf_arr[index][kp_idx])
            if panel_shape is not None:
                height, width = panel_shape[:2]
                item["inside_panel"] = bool(0.0 <= item["x"] < width and 0.0 <= item["y"] < height)
            out[name] = item
        return out

    def _draw_detection_overlay(self, overlay: np.ndarray, panel_meta: Dict[str, Any]) -> None:
        try:
            import cv2
        except ImportError:
            return

        all_panels = panel_meta.get("vision_detector_all_panels") or []
        for candidate in all_panels:
            bx0, by0, bx1, by1 = [int(v) for v in candidate.get("panel_bbox", [0, 0, 0, 0])]
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 180, 255), 1)
            label = f"{float(candidate.get('confidence', 0.0)):.2f}"
            cv2.putText(
                overlay,
                label,
                (bx0, max(12, by0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 120, 255),
                1,
            )

        x0, y0, x1, y1 = [int(v) for v in panel_meta.get("panel_bbox", [0, 0, 0, 0])]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 255), 3)
        keypoints = panel_meta.get("vision_detector_keypoints", {})
        colors = {"vbm": (255, 0, 255), "cbm": (0, 255, 255)}
        for name, point in keypoints.items():
            px = int(round(x0 + float(point.get("x", 0.0))))
            py = int(round(y0 + float(point.get("y", 0.0))))
            cv2.circle(overlay, (px, py), 5, colors.get(name, (0, 255, 255)), -1)

    def _draw_skeleton_overlay(
        self,
        overlay: np.ndarray,
        panel_shape: tuple[int, int],
        offset: tuple[int, int],
        skeleton: np.ndarray,
    ) -> None:
        """Draw only CV-owned outputs: plot boundary and raw skeleton pixels.

        This intentionally does not draw E_F, VBM/CBM, band gaps, or
        valence/conduction labels. Those require human calibration in the GUI.
        """

        try:
            import cv2
        except ImportError:
            return

        x0, y0 = offset
        height, width = panel_shape[:2]
        x1, y1 = x0 + width, y0 + height
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 140, 0), 3)

        ys, xs = np.where(skeleton > 0)
        for px, py in zip(xs, ys):
            cv2.circle(overlay, (int(x0 + px), int(y0 + py)), 1, (0, 220, 255), -1)

    def parse_video(self, path: str) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for video Plot-to-Physics parsing"
            ) from exc

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.max_video_frames
        stride = max(frame_count // self.max_video_frames, 1)
        parsed_frames: List[ParsedBandData] = []
        frame_numbers: List[int] = []
        prev_gray = None
        flow_magnitudes: List[float] = []

        idx = 0
        while len(parsed_frames) < self.max_video_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            tmp = self._parse_frame_array(path, frame)
            parsed_frames.append(tmp)
            frame_numbers.append(idx)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitudes.append(float(np.linalg.norm(flow, axis=2).mean()))
            prev_gray = gray
            idx += 1

        cap.release()
        if not parsed_frames:
            raise ValueError("No parseable frames were found in the video")

        frame_gaps = np.asarray(
            [self._rough_gap_from_points(p) for p in parsed_frames], dtype=np.float32
        )
        base = parsed_frames[len(parsed_frames) // 2]
        base.frame_gaps = frame_gaps - frame_gaps[0]
        base.frame_numbers = np.asarray(frame_numbers, dtype=np.int32)
        base.source_type = "video"
        base.metadata.update(
            {
                "frames_used": len(parsed_frames),
                "delta_gap_ev": base.frame_gaps.tolist(),
                "mean_optical_flow": float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0,
            }
        )
        return base

    def _parse_frame_array(self, source_path: str, frame: np.ndarray) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for video Plot-to-Physics parsing"
            ) from exc

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = self._band_binary_mask(gray)
        skeleton = self._skeletonize(binary)
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 8:
            raise ValueError("Frame did not contain a usable band skeleton")
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        overlay = frame.copy()
        overlay[skeleton > 0] = (0, 0, 255)
        return self._coords_to_band_data(
            source_path,
            "video_frame",
            coords,
            image_shape=frame.shape[:2],
            skeleton_pixels=coords,
            overlay_image=overlay,
        )

    def _band_binary_mask(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, dark = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(blurred, 60, 160)
        mask = cv2.bitwise_or(dark, edges)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _extract_band_panel(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        """Find the main band-structure panel in screenshots or scanned plots."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        line_panel = self._extract_line_plot_panel(image)
        if line_panel is not None:
            return line_panel

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_bg = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 170) & (hsv[:, :, 2] < 245)).astype(
            np.uint8
        ) * 255
        upper = np.zeros_like(gray_bg)
        upper[: int(image.shape[0] * 0.62), :] = gray_bg[: int(image.shape[0] * 0.62), :]
        kernel = np.ones((13, 13), np.uint8)
        mask = cv2.morphologyEx(upper, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        candidates = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if area < 0.02 * image.shape[0] * image.shape[1]:
                continue
            if w < 150 or h < 120:
                continue
            aspect = w / max(h, 1)
            if aspect < 1.1:
                continue
            candidates.append((area, x, y, w, h))

        if not candidates:
            dark_crop = self._extract_dark_plot_panel(image)
            if dark_crop is not None:
                return dark_crop
            return (
                image,
                (0, 0),
                {
                    "panel_bbox": [0, 0, int(image.shape[1]), int(image.shape[0])],
                    "panel_detection": "fallback_full_image",
                },
            )

        # The band plot is normally the largest wide gray panel. DOS panels are
        # narrower and text blocks are outside the gray plotting background.
        _area, x, y, w, h = max(candidates, key=lambda item: (item[0], item[3]))
        pad = 3
        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w) + pad, image.shape[1])
        y1 = min(int(y + h) + pad, image.shape[0])
        panel, inner_offset, frame_meta = self._crop_to_plot_frame(image[y0:y1, x0:x1].copy())
        x0 += inner_offset[0]
        y0 += inner_offset[1]
        x1 = x0 + panel.shape[1]
        y1 = y0 + panel.shape[0]
        return (
            panel,
            (x0, y0),
            {
                "panel_bbox": [x0, y0, x1, y1],
                "panel_detection": "largest_gray_band_panel",
                **frame_meta,
            },
        )

    def _extract_dark_plot_panel(
        self, image: np.ndarray
    ) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < 95).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < image.shape[1] * 0.45 or h < image.shape[0] * 0.45:
                continue
            if w * h > image.shape[0] * image.shape[1] * 0.95:
                continue
            candidates.append((w * h, x, y, w, h))
        if not candidates:
            return None

        _area, x, y, w, h = max(candidates, key=lambda item: item[0])
        pad = 4
        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w) + pad, image.shape[1])
        y1 = min(int(y + h) + pad, image.shape[0])
        panel, inner_offset, frame_meta = self._crop_to_plot_frame(image[y0:y1, x0:x1].copy())
        x0 += inner_offset[0]
        y0 += inner_offset[1]
        x1 = x0 + panel.shape[1]
        y1 = y0 + panel.shape[0]
        return (
            panel,
            (x0, y0),
            {
                "panel_bbox": [x0, y0, x1, y1],
                "panel_detection": "largest_dark_plot_frame",
                **frame_meta,
            },
        )

    def _extract_line_plot_panel(
        self, image: np.ndarray
    ) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        height, width = gray.shape[:2]

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 4)))
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        col_counts = (v_lines > 0).sum(axis=0)
        cols = np.where(col_counts >= max(int(height * 0.45), 80))[0]
        groups = self._contiguous_groups(cols)
        centers = [int(round((g[0] + g[-1]) / 2.0)) for g in groups]

        usable = []
        for c in centers:
            ys = np.where(v_lines[:, c] > 0)[0]
            if len(ys) < max(int(height * 0.45), 80):
                continue
            usable.append((c, int(ys[0]), int(ys[-1]), int(len(ys))))
        if len(usable) < 2:
            return None
        min_top = min(item[1] for item in usable)
        max_bottom = max(item[2] for item in usable)
        top_tol = max(6, int(height * 0.03))
        bottom_tol = max(8, int(height * 0.06))
        frame_like = [
            item
            for item in usable
            if item[1] <= min_top + top_tol and item[2] >= max_bottom - bottom_tol
        ]
        if len(frame_like) >= 2:
            usable = frame_like

        x_left = min(item[0] for item in usable)
        x_right = max(item[0] for item in usable)
        y_top = int(np.median([item[1] for item in usable]))
        y_bottom = int(np.median([item[2] for item in usable]))
        if x_right - x_left < width * 0.35 or y_bottom - y_top < height * 0.40:
            return None

        x0 = max(x_left, 0)
        y0 = max(y_top, 0)
        x1 = min(x_right + 1, width)
        y1 = min(y_bottom + 1, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            image[y0:y1, x0:x1].copy(),
            (x0, y0),
            {
                "panel_bbox": [x0, y0, x1, y1],
                "panel_detection": "line_plot_frame",
                "frame_crop": "global_long_line_plot_frame",
                "frame_line_columns": [int(item[0]) for item in usable],
            },
        )

    def _crop_to_plot_frame(
        self, panel: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        robust = self._crop_to_long_line_frame(panel, gray)
        if robust is not None:
            return robust

        dark = (gray < 120).astype(np.uint8)
        row_counts = dark.sum(axis=1)
        min_run = max(int(panel.shape[1] * 0.45), 80)
        rows = np.where(row_counts >= min_run)[0]
        if len(rows) < 2:
            return panel, (0, 0), {"frame_crop": "none"}

        y_top = int(rows[0])
        y_bottom = int(rows[-1])
        if y_bottom - y_top < panel.shape[0] * 0.35:
            return panel, (0, 0), {"frame_crop": "none"}

        row_band = dark[max(0, y_top - 2) : min(panel.shape[0], y_top + 3), :].max(axis=0)
        if row_band.sum() < min_run:
            row_band = dark[max(0, y_bottom - 2) : min(panel.shape[0], y_bottom + 3), :].max(axis=0)
        xs = np.where(row_band > 0)[0]
        if len(xs) < 2:
            return panel, (0, 0), {"frame_crop": "none"}
        x_left = int(xs[0])
        x_right = int(xs[-1])
        if x_right - x_left < panel.shape[1] * 0.35:
            return panel, (0, 0), {"frame_crop": "none"}

        pad = 2
        x0 = max(x_left + pad, 0)
        y0 = max(y_top + pad, 0)
        x1 = min(x_right - pad, panel.shape[1])
        y1 = min(y_bottom - pad, panel.shape[0])
        if x1 <= x0 or y1 <= y0:
            return panel, (0, 0), {"frame_crop": "none"}
        return panel[y0:y1, x0:x1].copy(), (x0, y0), {"frame_crop": "plot_frame"}

    def _crop_to_long_line_frame(
        self, panel: np.ndarray, gray: np.ndarray
    ) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        height, width = gray.shape[:2]
        _, dark = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 4)))
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        col_counts = (v_lines > 0).sum(axis=0)
        cols = np.where(col_counts >= max(int(height * 0.45), 80))[0]
        groups = self._contiguous_groups(cols)
        if len(groups) < 2:
            return None

        centers = [int(round((g[0] + g[-1]) / 2.0)) for g in groups]
        line_heights = [int(col_counts[c]) for c in centers]
        usable = [
            (c, h) for c, h in zip(centers, line_heights) if h >= max(int(height * 0.55), 120)
        ]
        if len(usable) < 2:
            return None

        x_left = min(c for c, _h in usable)
        x_right = max(c for c, _h in usable)
        if x_right - x_left < width * 0.35:
            return None

        y_samples = []
        for c, _h in usable:
            ys = np.where(v_lines[:, c] > 0)[0]
            if len(ys):
                y_samples.append((int(ys[0]), int(ys[-1])))
        if len(y_samples) < 2:
            return None
        y_top = int(np.median([y0 for y0, _y1 in y_samples]))
        y_bottom = int(np.median([y1 for _y0, y1 in y_samples]))
        if y_bottom - y_top < height * 0.35:
            return None

        pad = 3
        x0 = max(x_left + pad, 0)
        y0 = max(y_top + pad, 0)
        x1 = min(x_right - pad, width)
        y1 = min(y_bottom - pad, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            panel[y0:y1, x0:x1].copy(),
            (x0, y0),
            {
                "frame_crop": "long_line_plot_frame",
                "frame_line_columns": [int(c) for c, _h in usable],
            },
        )

    def _contiguous_groups(self, values: np.ndarray) -> List[np.ndarray]:
        if len(values) == 0:
            return []
        breaks = np.where(np.diff(values) > 1)[0] + 1
        return [group for group in np.split(values, breaks) if len(group) > 0]

    def _auto_band_mask(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        # Color-agnostic skeleton extraction. CV must not decide whether a
        # curve is valence/conduction from color; human GUI calibration assigns
        # physical meaning later.
        gray = self._remove_ruling_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        mask = self._dark_curve_mask(gray)
        mask = self._remove_reference_line_rows(mask)
        mask = self._remove_reference_line_columns(mask)
        # Filter out small text-like blobs so letters aren't misidentified as
        # band-structure curves (connected components that are too compact or
        # too small relative to the image).
        mask = self._filter_text_blobs(mask)
        return mask, "color_agnostic_skeleton"

    def _filter_text_blobs(self, binary: np.ndarray) -> np.ndarray:
        """Remove connected components that look like text/letters/axis-labels.

        Text characters appear as compact dark blobs (e.g. axis labels,
        inset text, tick numbers). Band curves are long, thin curvilinear
        structures with very low aspect ratio (width >> height or vice versa).
        We filter aggressively:
          - area < 80 px: noise/punctuation
          - aspect > 0.20 and area < 800: too square, likely a letter/number
        """
        try:
            import cv2

            n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
        except ImportError:
            return binary

        cleaned = binary.copy()
        keep = np.ones(n_labels, dtype=bool)
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            # Skip the background label (i=0)
            if area < 80:
                # Tiny speck -- noise, punctuation, thin tick marks
                keep[i] = False
                continue
            aspect = min(w, h) / max(w, max(h, 1))
            if aspect > 0.20 and area < 800:
                # Nearly square or moderately elongated blob, with bounded area.
                # Band curves are very thin: aspect << 0.1 for long lines.
                # Letters/numbers/axis labels have aspect > 0.20.
                keep[i] = False
                continue
        # One label-indexed image pass, not one full-image comparison per speck.
        cleaned[~keep[labels]] = 0
        return cleaned

    def _dark_curve_mask(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _suppress_annotation_colors(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        red = ((hue <= 12) | (hue >= 168)) & (sat > 35) & (val > 45)
        cleaned = image.copy()
        cleaned[red] = (255, 255, 255)
        return cleaned

    def _remove_reference_line_rows(self, binary: np.ndarray) -> np.ndarray:
        cleaned = binary.copy()
        height, width = cleaned.shape[:2]
        row_counts = (cleaned > 0).sum(axis=1)
        rows_idx = np.arange(height)
        interior = (rows_idx > int(height * 0.08)) & (rows_idx < int(height * 0.92))
        threshold = max(int(width * 0.20), 80)
        candidates = np.where((row_counts >= threshold) & interior)[0]
        for group in self._contiguous_groups(candidates):
            if len(group) > max(3, int(height * 0.02)):
                continue
            y0 = max(int(group[0]) - 2, 0)
            y1 = min(int(group[-1]) + 3, height)
            cleaned[y0:y1, :] = 0
        return cleaned

    def _remove_reference_line_columns(self, binary: np.ndarray) -> np.ndarray:
        cleaned = binary.copy()
        height, width = cleaned.shape[:2]
        col_counts = (cleaned > 0).sum(axis=0)
        cols_idx = np.arange(width)
        interior = (cols_idx > int(width * 0.02)) & (cols_idx < int(width * 0.98))
        threshold = max(int(height * 0.35), 140)
        candidates = np.where((col_counts >= threshold) & interior)[0]
        for group in self._contiguous_groups(candidates):
            if len(group) > max(8, int(width * 0.03)):
                continue
            x0 = max(int(group[0]) - 2, 0)
            x1 = min(int(group[-1]) + 3, width)
            cleaned[:, x0:x1] = 0
        return cleaned

    def _remove_ruling_lines(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        _, dark = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, gray.shape[0] // 9)))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, gray.shape[1] // 8), 1))
        vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        ruling = cv2.bitwise_or(vertical, horizontal)
        cleaned_dark = cv2.bitwise_and(dark, cv2.bitwise_not(ruling))
        return cv2.bitwise_not(cleaned_dark)

    def _blue_band_mask(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        b, g, r = cv2.split(image)
        blue_hsv = (hue >= 85) & (hue <= 115) & (sat > 45) & (val > 45)
        blue_bgr = (b.astype(np.int16) > r.astype(np.int16) + 20) & (
            b.astype(np.int16) > g.astype(np.int16) + 5
        )
        mask = (blue_hsv | blue_bgr).astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _imread_unicode(self, path: str) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for image Plot-to-Physics parsing"
            ) from exc

        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(path)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for skeletonization") from exc

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary)

        img = (binary > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _coords_to_band_data(
        self,
        path: str,
        source_type: str,
        coords_xy: np.ndarray,
        image_shape: Any,
        skeleton_pixels: Optional[np.ndarray] = None,
        overlay_image: Optional[np.ndarray] = None,
        panel_image: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedBandData:
        height = float(image_shape[0])
        width = float(image_shape[1])
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_span = max(x_max - x_min, 1.0)

        k = np.clip((x - x_min) / x_span, 0.0, 1.0)
        # CV-owned output only: normalized image coordinates. These are not
        # physical energies and must not be used for final physics inference.
        energy = (1.0 - np.clip(y / max(height, 1.0), 0.0, 1.0)).astype(np.float32)
        band_hint = np.zeros_like(energy, dtype=np.int8)

        kpath_labels = self._estimate_kpath_labels(coords_xy, width)
        meta = metadata or {}
        meta.update(
            {
                "image_shape": [height, width],
                "requires_human_calibration": True,
                "cv_physics_inference": "disabled",
            }
        )
        return ParsedBandData(
            source_path=str(path),
            source_type=source_type,
            k=k.astype(np.float32),
            energy=energy.astype(np.float32),
            band_hint=band_hint,
            fermi_energy=0.0,
            kpath_labels=kpath_labels,
            skeleton_pixels=skeleton_pixels,
            overlay_image=overlay_image,
            metadata=meta,
        )

    # Fermi-level and band-type inference deliberately removed from CV parser.
    # Human calibration in scripts/gui_workbench.py owns physical axes, VBM/CBM,
    # and curve assignment before PhysicsBrainInvoker performs topology fusion.

    def _estimate_kpath_labels(self, coords_xy: np.ndarray, width: float) -> List[Dict[str, Any]]:
        x = coords_xy[:, 0]
        hist, bins = np.histogram(x, bins=32)
        threshold = max(float(hist.max()) * 0.65, 8.0)
        peaks = np.where(hist >= threshold)[0]
        ticks: List[Dict[str, Any]] = []
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]
        for i, peak in enumerate(peaks[:4]):
            x_center = 0.5 * (bins[peak] + bins[peak + 1])
            ticks.append(
                {
                    "index": int(round((x_center / max(width, 1.0)) * (self.target_k_points - 1))),
                    "label": labels[i],
                }
            )
        if not ticks:
            ticks = [
                {"index": 0, "label": "$\\Gamma$"},
                {"index": self.target_k_points - 1, "label": "X"},
            ]
        return ticks

    def _rough_gap_from_points(self, parsed: ParsedBandData) -> float:
        vb = parsed.energy[parsed.energy <= 0.0]
        cb = parsed.energy[parsed.energy >= 0.0]
        if len(vb) == 0 or len(cb) == 0:
            return 0.0
        return float(np.min(cb) - np.max(vb))
