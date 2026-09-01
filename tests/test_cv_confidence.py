"""P2 CV confidence + brain uncertainty — RED-first tests."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.vision.multi_format_parser import MultiFormatParser, cv_quality_level
from src.vision.brain_invoker import compute_brain_uncertainty


def _synthetic_band_image(w=600, h=400, degrade=0.0, salt=0.0, seed=0):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 40, 30, 560, 370
    d.rectangle([x0, y0, x1, y1], outline="black", width=2)
    for i in range(9):
        tx = x0 + (x1 - x0) * i / 8
        d.line([tx, y1, tx, y1 - 6], fill="black", width=1)
    for i in range(6):
        ty = y0 + (y1 - y0) * i / 5
        d.line([x0, ty, x0 + 6, ty], fill="black", width=1)
    rng = np.random.default_rng(seed)
    for band in range(3):
        prev = None
        for x in range(x0 + 2, x1 - 2, 3):
            y = y0 + (y1 - y0) / 2 + (band - 1) * 70 + 50 * math.sin((x - x0) / 55.0 + band)
            if degrade:
                y += rng.normal(0, degrade)
            px, py = int(x), int(y)
            if prev is not None:
                d.line([prev, (px, py)], fill="black", width=2)
            prev = (px, py)
    if salt:
        arr = np.array(img.convert("RGB"))
        noise = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        mask = rng.random((h, w)) < salt
        arr[mask] = noise[mask]
        img = Image.fromarray(arr)
    return img


def _parse_score(tmp_path, name, img):
    path = tmp_path / name
    img.save(path)
    parser = MultiFormatParser(detector_path=None)
    try:
        parsed = parser.parse(str(path))
    except ValueError:
        return 0.0, None
    q = parsed.metadata["cv_quality"]
    return q["score"], q


def test_parser_attaches_cv_quality_metadata(tmp_path):
    _, q = _parse_score(tmp_path, "clean.png", _synthetic_band_image())
    assert q is not None
    assert 0.0 <= q["score"] <= 1.0
    assert q["level"] in ("green", "yellow", "red")
    assert q["detector_status"] == "optional_missing_score_excluded"
    assert "panel_detection" in q["components"]


def test_degraded_image_scores_lower_than_clean(tmp_path):
    clean_score, _ = _parse_score(tmp_path, "clean.png", _synthetic_band_image())
    degraded_score, _ = _parse_score(
        tmp_path, "degraded.png", _synthetic_band_image(degrade=25.0, salt=0.35)
    )
    assert clean_score > 0
    assert clean_score > degraded_score


def test_failed_extraction_counts_as_zero_confidence(tmp_path):
    blank = Image.new("RGB", (400, 300), "white")
    score, q = _parse_score(tmp_path, "blank.png", blank)
    assert score == 0.0 and q is None


def test_cv_quality_level_boundaries():
    assert cv_quality_level(1.0) == "green"
    assert cv_quality_level(0.6) == "green"
    assert cv_quality_level(0.599) == "yellow"
    assert cv_quality_level(0.35) == "yellow"
    assert cv_quality_level(0.349) == "red"
    assert cv_quality_level(0.0) == "red"


def _peaked_array(n=128, peak_idx=40, peak_val=0.9):
    arr = np.full(n, 0.02, dtype=np.float32)
    arr[peak_idx] = peak_val
    return arr


def test_brain_uncertainty_concentrated_scores_higher_than_uniform():
    concentrated = compute_brain_uncertainty(
        {"metal": 0.02, "direct": 0.95, "indirect": 0.03},
        vbm_probability=_peaked_array(),
        cbm_probability=_peaked_array(peak_idx=90),
        gap_ev=1.2,
    )
    uniform = compute_brain_uncertainty(
        {"metal": 1 / 3, "direct": 1 / 3, "indirect": 1 / 3},
        vbm_probability=np.full(128, 1 / 128, dtype=np.float32),
        cbm_probability=np.full(128, 1 / 128, dtype=np.float32),
        gap_ev=1.2,
    )
    assert concentrated["score"] > uniform["score"]
    assert 0.0 <= concentrated["entropy"] <= 1.0
    assert 0.0 <= uniform["entropy"] <= 1.0
    assert concentrated["level"] in ("green", "yellow", "red")


def test_brain_uncertainty_flags_implausible_gap():
    plausible = compute_brain_uncertainty(
        {"metal": 0.1, "direct": 0.8, "indirect": 0.1},
        _peaked_array(), _peaked_array(), gap_ev=1.5,
    )
    implausible = compute_brain_uncertainty(
        {"metal": 0.1, "direct": 0.8, "indirect": 0.1},
        _peaked_array(), _peaked_array(), gap_ev=-3.0,
    )
    assert plausible["gap_plausible"] is True
    assert implausible["gap_plausible"] is False
    assert implausible["score"] < plausible["score"]
