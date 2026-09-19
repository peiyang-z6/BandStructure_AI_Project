"""Document/plot ingestion contracts; all generated fixtures are synthetic."""

import io
import json
import hashlib
import numpy as np
import pytest


def test_native_pdf_text_glyph_loss_requires_visual_confirmation():
    from src.vision.axis_ocr_review import review_document_text

    original = "Gamma replaced by \x05; lattice 4.496 \x01A; exponent \ufffd and private \ue000"
    result = review_document_text(original)
    assert result["status"] == "requires_visual_confirmation" and not result["auto_corrected"]
    assert result["suspicious_codepoints"]["U+0005"] == 1
    assert (
        review_document_text("Energy (eV)\nΓ–X\t−2")["status"] == "no_lexical_flags_not_certified"
    )


import pymupdf
from PIL import Image, ImageDraw, ImageFont
from src.vision import multi_format_parser as parser


def test_pdf_native_text_keeps_provenance_and_no_calibrated_confidence():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 40), "Synthetic band reference: Energy (eV), not calibrated.")
        payload = doc.tobytes()
    func = getattr(parser, "extract_document_text", None)
    assert callable(func), "bounded document extraction is not implemented"
    result = func(payload, kind="pdf")
    assert "Energy (eV)" in result["text"]
    assert result["pages"][0]["engine"] == "pymupdf_text"
    assert result["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["confidence"] is None and result["ood_flag"] is None
    assert result["calibration_verified"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("limit", [0, -1, True, 6])
def test_document_rejects_invalid_page_budget(limit):
    with pymupdf.open() as doc:
        doc.new_page()
        payload = doc.tobytes()
    with pytest.raises(ValueError, match="max_pages"):
        parser.extract_document_text(payload, kind="pdf", max_pages=limit)


@pytest.mark.parametrize("case", ["path", "empty", "oversize", "unsupported"])
def test_document_boundary_rejects_bad_upload(case):
    with pymupdf.open() as doc:
        doc.new_page()
        good = doc.tobytes()
    payload = {
        "path": "C:/Windows/win.ini",
        "empty": b"",
        "oversize": good + b" " * (10 * 1024 * 1024),
        "unsupported": good,
    }[case]
    with pytest.raises(ValueError, match="document payload|document kind"):
        parser.extract_document_text(payload, kind="html" if case == "unsupported" else "pdf")


def synthetic_text_png():
    image = Image.new("RGB", (800, 240), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text((40, 65), "Energy eV 12345", font=font, fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_raster_document_runs_real_ocr_not_inferred_calibration():
    result = parser.extract_document_text(synthetic_text_png(), kind="image")
    assert "Energy" in result["text"] and "12345" in result["text"]
    page = result["pages"][0]
    assert page["engine"] == "rapidocr_onnxruntime"
    assert page["regions"] and page["coordinate_unit"] == "pixel"
    assert all(0 <= region["ocr_score"] <= 1 for region in page["regions"])
    assert result["calibration_verified"] is False
    assert result["confidence"] is None


def test_scanned_pdf_falls_back_to_real_ocr():
    with pymupdf.open() as doc:
        page = doc.new_page(width=800, height=240)
        page.insert_image(page.rect, stream=synthetic_text_png())
        payload = doc.tobytes()
    result = parser.extract_document_text(payload, kind="pdf")
    assert "12345" in result["text"]
    assert result["pages"][0]["engine"] == "rapidocr_onnxruntime"
    assert result["pages"][0]["pdf_render_dpi"] == 150


def test_document_refuses_excessive_decoded_raster_pixels():
    image = Image.new("L", (4097, 4096), 255)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    with pytest.raises(ValueError, match="pixels"):
        parser.extract_document_text(buf.getvalue(), kind="image")


def test_pdf_format_is_not_silently_inferred_from_wrong_declared_kind():
    with pytest.raises(ValueError, match="PDF header"):
        parser.extract_document_text(synthetic_text_png(), kind="pdf")


def synthetic_band_png():
    image = Image.new("RGB", (641, 481), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 580, 440), outline="black", width=2)
    k = np.linspace(0, 1, 201)
    for energy in [-0.3 - (k - 0.5) ** 2, 1.2 + (k - 0.5) ** 2]:
        points = list(zip((60 + 520 * k).astype(int), (240 - 100 * energy).astype(int)))
        draw.line(points, fill="black", width=3)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_uncalibrated_band_image_returns_only_pixel_evidence():
    func = getattr(parser, "extract_band_image", None)
    assert callable(func), "pixel-only band extraction not implemented"
    result = func(synthetic_band_png())
    assert result["status"] == "requires_calibration"
    assert result["band_data"] is None
    assert result["skeleton_point_count"] > 10
    assert result["confidence"] is None
    assert result["human_audited"] is False


def synthetic_annotations():
    return {
        "panel": [60, 40, 520, 400],
        "fermi_y": 240,
        "xaxis_pts": [[60, 440], [580, 440]],
        "yaxis_pts": [[60, 40], [60, 440]],
        "vbm": [320, 270],
        "cbm": [320, 120],
        "vb_strokes": [[[60, 295], [190, 276.25], [320, 270], [450, 276.25], [580, 295]]],
        "cb_strokes": [[[60, 95], [190, 113.75], [320, 120], [450, 113.75], [580, 95]]],
    }, {"x_values": [0, 1], "y_values": [2, -2]}


@pytest.mark.parametrize(
    "field,values",
    [
        ("x_values", [False, True]),
        ("y_values", [True, False]),
        ("x_values", ["0", "1"]),
        ("y_values", ["2", "-2"]),
    ],
)
def test_upload_calibration_requires_real_numbers_without_coercion(field, values):
    ann, cal = synthetic_annotations()
    cal[field] = values
    with pytest.raises(ValueError, match="finite real numbers"):
        parser.extract_band_image(synthetic_band_png(), annotations=ann, calibration=cal)


def test_upload_annotation_numbers_cannot_hide_bool_or_numeric_strings():
    from copy import deepcopy

    ann, cal = synthetic_annotations()
    for field, indices in [
        ("panel", (0,)),
        ("fermi_y", ()),
        ("xaxis_pts", (0, 0)),
        ("yaxis_pts", (0, 1)),
        ("vbm", (0,)),
        ("cbm", (1,)),
        ("vb_strokes", (0, 0, 0)),
        ("cb_strokes", (0, 0, 1)),
    ]:
        for replacement in (True, "240"):
            altered = deepcopy(ann)
            target, key = altered, field
            for index in indices:
                target, key = target[key], index
            target[key] = replacement
            with pytest.raises(ValueError, match="finite real numbers"):
                parser.extract_band_image(
                    synthetic_band_png(), annotations=altered, calibration=cal
                )


def test_caller_calibration_uses_existing_6d_reconstructor_without_human_forgery():
    ann, cal = synthetic_annotations()
    result = parser.extract_band_image(synthetic_band_png(), annotations=ann, calibration=cal)
    assert result["status"] == "caller_calibrated_unverified"
    bands = np.asarray(result["band_data"]["energies_eV"])
    np.testing.assert_allclose(bands[1].min() - bands[0].max(), 1.5, rtol=1e-5, atol=1e-5)
    assert result["tensor_shape"] == [1, 2, 128, 3]
    assert result["band_data"]["k_unit"] == "relative"
    assert result["human_audited"] is False and result["confidence"] is None
    assert len(result["calibration_sha256"]) == 64


def test_pdf_rejects_huge_page_before_rendering(monkeypatch):
    with pymupdf.open() as doc:
        doc.new_page(width=10000, height=10000)
        payload = doc.tobytes()

    def prohibit_render(*args, **kwargs):
        raise AssertionError("oversize raster rendering reached")

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", prohibit_render)
    with pytest.raises(ValueError, match="pixels"):
        parser.extract_document_text(payload, kind="pdf")
