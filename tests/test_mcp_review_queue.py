"""Review workbench contracts: AI pre-review can never mint human approval."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def pending_queue():
    from mcp_server.server import prepare_human_audit_candidate

    observations = {
        "source": {
            "document_id": "sha256:" + "1" * 64,
            "page_number": 7,
            "panel_label": "(a)",
            "figure_label": "Figure 3(a)",
            "material_label": "BaB2",
        },
        "energy_unit": "eV",
        "fermi_eV": 0.0,
        "k_unit": "relative",
        "calibration_evidence": {
            "energy_tick_count": 2,
            "k_anchor_count": 8,
            "fermi_reference_observed": True,
        },
        "ambiguities": ["continuous band identity unresolved"],
        "human_confirmed": False,
    }
    candidate = prepare_human_audit_candidate(observations)["candidate_record"]
    return {
        "schema_version": 1,
        "status": "pending_human",
        "eligible_for_scientific_acceptance": False,
        "records": [candidate],
    }


def write_queue(path: Path) -> None:
    path.write_text(json.dumps(pending_queue()), encoding="utf-8")


def test_queue_load_verifies_candidate_hash_and_keeps_approval_blocked(tmp_path):
    from mcp_server.review_queue import load_pending_queue

    path = tmp_path / "queue.json"
    write_queue(path)
    loaded = load_pending_queue(path)
    assert loaded["status"] == "pending_human"
    assert loaded["record_count"] == 1
    review = loaded["records"][0]["review_state"]
    assert review["approval_ready"] is False
    assert "bands" in review["missing_evidence"]
    assert review["formal_approval_supported"] is False


def test_queue_load_rejects_tamper_and_self_approval(tmp_path):
    from mcp_server.review_queue import load_pending_queue

    data = pending_queue()
    path = tmp_path / "queue.json"
    for mutation in ("observation", "review", "eligible"):
        changed = json.loads(json.dumps(data))
        if mutation == "observation":
            changed["records"][0]["observations"]["fermi_eV"] = 1.0
        elif mutation == "review":
            changed["records"][0]["independent_human_review"] = {"decision": "approved"}
        else:
            changed["eligible_for_scientific_acceptance"] = True
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_pending_queue(path)


@pytest.mark.parametrize("reviewer_kind", ["ai_assisted", "human"])
def test_review_packet_never_accepts_approved_decision(reviewer_kind):
    from mcp_server.review_queue import make_review_packet

    record = pending_queue()["records"][0]
    with pytest.raises(ValueError, match="formal approval"):
        make_review_packet(record, "approved", reviewer_kind, "must not approve")


def test_ai_assisted_packet_is_explicitly_nonhuman_and_noneligible():
    from mcp_server.review_queue import make_review_packet

    record = pending_queue()["records"][0]
    packet = make_review_packet(
        record, "needs_more_evidence", "ai_assisted",
        "Axes and EF are visible; continuous branches remain unresolved.")
    assert packet["status"] == "ai_assisted_pre_review"
    assert packet["decision"] == "needs_more_evidence"
    assert packet["human_audited"] is False
    assert packet["eligible_for_scientific_acceptance"] is False
    assert packet["operator_authentication"] is None
    assert packet["record_id"] == record["record_id"]


def test_review_packet_save_is_hash_bound_and_refuses_overwrite(tmp_path):
    from mcp_server.review_queue import make_review_packet, save_review_packet

    packet = make_review_packet(
        pending_queue()["records"][0], "needs_more_evidence", "ai_assisted", "reviewed")
    result = save_review_packet(packet, tmp_path / "reviews")
    assert Path(result["path"]).is_file()
    assert len(result["sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_review_packet(packet, tmp_path / "reviews")


def test_review_packet_save_rejects_path_traversal_and_forged_approval(tmp_path):
    from mcp_server.review_queue import make_review_packet, save_review_packet

    packet = make_review_packet(
        pending_queue()["records"][0], "needs_more_evidence", "ai_assisted", "reviewed")
    traversal = {**packet, "record_id": "../../forged"}
    approved = {
        **packet,
        "status": "approved",
        "decision": "approved",
        "human_audited": True,
        "eligible_for_scientific_acceptance": True,
    }
    with pytest.raises(ValueError):
        save_review_packet(traversal, tmp_path / "reviews")
    with pytest.raises(ValueError):
        save_review_packet(approved, tmp_path / "reviews")
    assert not (tmp_path / "forged__ai_assisted__needs_more_evidence.json").exists()


def test_workbench_headless_check_validates_queue_and_image(tmp_path):
    from PIL import Image

    queue = tmp_path / "queue.json"
    image = tmp_path / "page.png"
    write_queue(queue)
    Image.new("RGB", (320, 240), "white").save(image)
    result = subprocess.run([
        sys.executable, "scripts/mcp_human_review_workbench.py",
        "--queue", str(queue), "--image", str(image),
        "--output-dir", str(tmp_path / "reviews"), "--headless-check",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ready_for_manual_review"
    assert payload["record_count"] == 1
    assert payload["approval_ready_count"] == 0
    assert payload["formal_approval_supported"] is False
