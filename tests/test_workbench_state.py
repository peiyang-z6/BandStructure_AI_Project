"""P1 workbench state persistence — RED-first tests for WorkbenchStateStore."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.gui_workbench import WorkbenchStateStore, annotations_from_json


def _rgb_image(w=320, h=200, color=(255, 255, 255)):
    img = Image.new("RGB", (w, h), color)
    # deterministic single-pixel detail to give unique content
    img.putpixel((10, 10), (255, 0, 0))
    return img


def _record(key, image_path="/tmp/a.png"):
    return {
        "schema_version": 1,
        "input_path": image_path,
        "image_key": key,
        "canvas_width": 320,
        "canvas_height": 200,
        "annotations": {
            "panel": [5.0, 5.0, 300.0, 180.0],
            "fermi_y": 90.0,
            "vb_strokes": [[[10.0, 100.0], [20.0, 110.0]]],
            "cb_strokes": [],
            "xaxis_pts": [[5.0, 185.0], [315.0, 185.0]],
            "yaxis_pts": [[5.0, 180.0], [5.0, 10.0]],
            "vbm": [120.0, 110.0],
            "cbm": [120.0, 70.0],
            "gap_type": "direct",
        },
        "calibration": {"y_values": [0.0, 5.0], "x_values": [0.0, 1.0]},
        "material_id": "GaAs",
        "gap_label": "direct",
    }


def test_roundtrip_preserves_annotations_and_calibration(tmp_path):
    store = WorkbenchStateStore(state_dir=tmp_path)
    img = _rgb_image()
    key = store.image_key(img)
    rec = _record(key)
    store.save(key, rec)
    loaded = store.load(key)
    assert loaded["annotations"]["vbm"] == [120.0, 110.0]
    assert loaded["annotations"]["vb_strokes"] == [[[10.0, 100.0], [20.0, 110.0]]]
    assert loaded["calibration"] == {"y_values": [0.0, 5.0], "x_values": [0.0, 1.0]}
    assert loaded["material_id"] == "GaAs"


def test_same_image_content_under_different_paths_shares_key(tmp_path):
    store = WorkbenchStateStore(state_dir=tmp_path)
    key_a = store.image_key(_rgb_image())
    key_b = store.image_key(_rgb_image())
    assert key_a == key_b
    # same state file is hit regardless of where the image was loaded from
    store.save(key_a, _record(key_a, image_path="/one/path/fig.png"))
    assert store.load(key_b)["input_path"] == "/one/path/fig.png"


def test_corrupt_state_file_returns_none(tmp_path):
    store = WorkbenchStateStore(state_dir=tmp_path)
    key = store.image_key(_rgb_image())
    (store.state_path(key)).write_text("{not valid json", encoding="utf-8")
    assert store.load(key) is None


def test_missing_state_returns_none(tmp_path):
    store = WorkbenchStateStore(state_dir=tmp_path)
    assert store.load("no-such-key") is None


def test_atomic_save_leaves_no_temp_files(tmp_path):
    store = WorkbenchStateStore(state_dir=tmp_path)
    key = store.image_key(_rgb_image())
    store.save(key, _record(key))
    leftovers = [p.name for p in Path(tmp_path).iterdir() if p.name != f"{key}.json"]
    assert leftovers == []


def test_annotations_from_json_rebuilds_canvas_shapes():
    data = _record("k")["annotations"]
    ann = annotations_from_json(data)
    assert ann["panel"] == (5.0, 5.0, 300.0, 180.0)
    assert ann["fermi_y"] == 90.0
    assert ann["vbm"] == (120.0, 110.0)
    assert ann["vb_strokes"] == [[(10.0, 100.0), (20.0, 110.0)]]
    assert ann["xaxis_pts"] == [(5.0, 185.0), (315.0, 185.0)]
    assert ann["gap_type"] == "direct"


def test_annotations_from_json_tolerates_nulls():
    ann = annotations_from_json(
        {"panel": None, "fermi_y": None, "vb_strokes": [], "cb_strokes": [],
         "xaxis_pts": [], "yaxis_pts": [], "vbm": None, "cbm": None, "gap_type": None}
    )
    assert ann["panel"] is None and ann["vbm"] is None and ann["fermi_y"] is None
