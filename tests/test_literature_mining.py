"""P3 literature-mining statistics and traversal — RED-first tests."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.literature_mining_pipeline import (
    iter_raster_images,
    summarize_records,
    store_experimental_band,
    write_report,
)


def test_iter_raster_images_lists_only_supported_formats(tmp_path):
    for name in ["a.png", "b.jpg", "c.JPEG", "d.bmp", "e.tiff", "f.txt", "g.pdf", "h.csv"]:
        (tmp_path / name).write_bytes(b"x")
    found = sorted(p.name for p in iter_raster_images(tmp_path))
    assert found == ["a.png", "b.jpg", "c.JPEG", "d.bmp", "e.tiff"]


def test_summarize_records_computes_success_rate_mean_confidence_and_direct_count():
    records = [
        {"status": "stored", "cv_score": 0.8, "prediction": {"predicted_type": "direct"}},
        {"status": "low_confidence", "cv_score": 0.3, "prediction": {"predicted_type": "indirect"}},
        {"status": "failed", "error": "boom"},
        {"status": "stored", "cv_score": 0.7, "prediction": {"predicted_type": "direct"}},
    ]
    stats = summarize_records(records)
    assert stats["success_rate"] == pytest.approx(0.5)
    assert stats["average_confidence"] == pytest.approx((0.8 + 0.3 + 0.7) / 3.0)
    assert stats["direct_gap_candidates"] == 2
    assert stats["total"] == 4


def test_summarize_records_empty_input():
    stats = summarize_records([])
    assert stats["total"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["average_confidence"] is None
    assert stats["direct_gap_candidates"] == 0


class _FakeReconstructed:
    raw_tensor = np.zeros((128, 6), dtype=np.float32)
    flat_tensor = np.zeros((128, 6), dtype=np.float32)
    metadata = {}


class _FakePrediction:
    vbm_probability = np.zeros(128, dtype=np.float32)
    cbm_probability = np.zeros(128, dtype=np.float32)
    line_mode_gap_ev = 1.1
    predicted_type = "direct"


def test_store_experimental_band_writes_cv_confidence_attr(tmp_path):
    h5_path = tmp_path / "experimental_bands.h5"
    confidence = {"cv_score": 0.77, "high_confidence": True, "reason": "physics_self_consistent"}
    store_experimental_band(
        h5_path,
        Path("fig1.png"),
        _FakeReconstructed(),
        _FakePrediction(),
        confidence,
    )
    with h5py.File(h5_path, "r") as h5:
        assert "fig1" in h5
        grp = h5["fig1"]
        assert abs(float(grp.attrs["cv_confidence"]) - 0.77) < 1e-6
        assert float(grp.attrs["line_mode_gap_ev"]) == 1.1
        assert grp.attrs["predicted_type"] == "direct"


def test_write_report_contains_required_statistics(tmp_path):
    summary = {
        "pdf_dir": str(tmp_path),
        "images_extracted": 10,
        "high_confidence": 6,
        "low_confidence": 2,
        "failed": 2,
        "success_rate": 0.6,
        "average_confidence": 0.62,
        "direct_gap_candidates": 3,
        "experimental_h5": str(tmp_path / "experimental_bands.h5"),
        "records": [
            {"image": "a.png", "status": "stored",
             "confidence": {"reason": "physics_self_consistent"},
             "prediction": {"line_mode_gap_ev": 1.2, "predicted_type": "direct"}},
            {"image": "b.png", "status": "failed", "error": "x"},
        ],
    }
    out = tmp_path / "report.md"
    write_report(summary, out)
    text = out.read_text(encoding="utf-8")
    assert "成功提取率" in text
    assert "60.0%" in text
    assert "平均置信度" in text
    assert "0.62" in text
    assert "潜在 Direct Gap 材料" in text
    assert "3" in text
