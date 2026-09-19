"""Public-release regressions; all fixtures here are synthetic, not human truth."""

import copy
import io
import json

import pytest
from PIL import Image, ImageDraw


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BAND_MCP_STORE_DIR", str(tmp_path / "store"))


def observation():
    return {
        "schema_version": 2,
        "source": {"document_id": "synthetic:release", "page_number": 1, "panel_label": "a"},
        "energy_unit": "eV",
        "energy_reference": {
            "kind": "fermi",
            "value_eV": 0.0,
            "observed": True,
            "evidence": "Synthetic declared Fermi reference",
        },
        "fermi_eV": 0.0,
        "k_unit": "relative",
        "k_distance": [0.0, 0.5, 1.0],
        "segment_ids": [0, 0, 0],
        "bands": [
            {"band_id": "v", "role": "valence", "energies_eV": [-2.0, -1.0, -2.0]},
            {"band_id": "c", "role": "conduction", "energies_eV": [2.0, 1.0, 2.0]},
        ],
        "calibration_evidence": {
            "energy_tick_count": 2,
            "k_anchor_count": 2,
            "fermi_reference_observed": True,
        },
        "ambiguities": [],
    }


def test_crossing_has_one_meaning_across_interfaces():
    from mcp_server.server import analyze_band_structure, analyze_visual_observations

    o = observation()
    o["bands"][0]["energies_eV"] = [-1.0, 0.0, 1.0]
    o["bands"][1]["energies_eV"] = [2.0, 2.5, 3.0]
    v2 = analyze_visual_observations(o)
    raw = analyze_band_structure(
        "numerical_data",
        {
            "energies_eV": [b["energies_eV"] for b in o["bands"]],
            "fermi_eV": 0.0,
            "k_unit": "relative",
            "k_distance": o["k_distance"],
            "segment_ids": o["segment_ids"],
        },
    )
    assert v2["line_mode_gap_eV"] == raw["line_mode_gap_eV"] == 0
    assert v2["line_mode_topology"] == raw["line_mode_topology"] == "metal"
    assert v2["eligible_for_scientific_acceptance"] is False


def test_missing_side_is_actionable_and_exportable():
    from mcp_server.server import request_missing_evidence, analyze_visual_observations

    o = observation()
    o["bands"] = o["bands"][:1]
    asked = request_missing_evidence(o)
    measured = analyze_visual_observations(o)
    assert asked["status"] == "needs_more_evidence"
    assert "conduction_band_evidence" in asked["missing_evidence"]
    assert measured["line_mode_gap_eV"] is None and measured["exports"]["csv_text"]


def test_physical_k_and_vbm_reference_do_not_invent_fermi():
    from mcp_server.server import analyze_visual_observations

    o = observation()
    o.pop("fermi_eV")
    o["energy_reference"]["kind"] = "vbm"
    o["bands"][0]["energies_eV"] = [-1.0, 0.0, -1.0]
    o["k_cartesian_invA"] = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    r = analyze_visual_observations(o)
    assert r["status"] == "ai_observation_unverified" and r["fermi_eV"] is None
    assert r["line_mode_gap_eV"] == 1 and r["line_mode_topology"] == "direct"


def test_realistic_448_band_input_keeps_all_bands():
    from mcp_server.server import analyze_visual_observations

    o = observation()
    o["bands"] = [
        {
            "band_id": str(i),
            "role": "valence" if i < 288 else "conduction",
            "energies_eV": [-2.0, -1.0, -2.0] if i < 288 else [2.0, 1.0, 2.0],
        }
        for i in range(448)
    ]
    r = analyze_visual_observations(o)
    assert r["observation_summary"]["band_count"] == 448
    assert r["coverage"]["observed_samples"] == 1344


def test_colour_annotation_does_not_remove_black_trace():
    from src.vision.panel_routing import visible_ink

    image = Image.new("RGB", (160, 160), "white")
    d = ImageDraw.Draw(image)
    d.line([(10, 120), (75, 50), (145, 120)], fill="black", width=3)
    before = visible_ink(image, [0, 0, 160, 160])
    d.ellipse((70, 25, 76, 31), fill="red")
    after = visible_ink(image, [0, 0, 160, 160])
    assert after["point_count"] >= before["point_count"]


def test_candidate_inventory_is_not_silently_truncated():
    from src.vision.panel_routing import frame_candidates

    image = Image.new("RGB", (1000, 800), "white")
    d = ImageDraw.Draw(image)
    for row in range(5):
        for column in range(8):
            x = 15 + column * 120
            y = 15 + row * 150
            d.rectangle((x, y, x + 100, y + 120), outline="black", width=2)
    assert len(frame_candidates(image)) == 40


def test_version_manifest_is_consistent():
    from pathlib import Path
    from mcp_server.server import DELIVERY_VERSION

    config = json.loads((Path(__file__).parents[1] / "mcp_server/config.json").read_text())
    assert config["delivery_version"] == DELIVERY_VERSION


def test_large_observation_export_has_a_paged_result_reference():
    from mcp_server.server import analyze_visual_observations

    o = observation()
    n = 2050
    o["k_distance"] = list(range(n))
    o["segment_ids"] = [0] * n
    for b in o["bands"]:
        b["energies_eV"] = [-1 if b["role"] == "valence" else 1] * n
    r = analyze_visual_observations(o)
    assert r["exports"]["result_id"] and r["exports"]["total_rows"] == 4100
