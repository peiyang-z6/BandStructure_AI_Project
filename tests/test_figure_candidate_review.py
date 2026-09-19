"""Rapid OA figure screening cannot mint independent human approval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest


def fixtures(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    pre = tmp_path / "pre"
    (dataset / "images").mkdir(parents=True)
    (dataset / "records").mkdir()
    (pre / "records").mkdir(parents=True)
    candidate_id = "epmc-band-" + "1" * 24
    image_path = dataset / "images" / f"{candidate_id}.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    record = {
        "schema_version": 1, "candidate_id": candidate_id,
        "group_id": "PMC1234567",
        "source": {"pmcid": "PMC1234567", "license": "CC-BY",
                   "figure_label": "Figure 1", "caption": "Electronic band structure."},
        "image": {"path": f"images/{candidate_id}.png", "sha256": image_sha,
                  "width": 320, "height": 240},
        "human_audited": False, "formal_approval": None,
        "operator_authentication": None, "eligible_for_scientific_acceptance": False,
    }
    record_rel = f"records/{candidate_id}.json"
    (dataset / record_rel).write_text(json.dumps(record), encoding="utf-8")
    dataset_manifest = {
        "schema_version": 1, "status": "complete_ai_assisted_figure_candidate_snapshot",
        "record_count": 1, "human_audited_count": 0, "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "records": [{"candidate_id": candidate_id, "group_id": "PMC1234567",
                     "record_path": record_rel, "image_path": record["image"]["path"],
                     "image_sha256": image_sha}],
    }
    dataset_manifest_path = dataset / "manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    dataset_sha = hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    pre_record = {
        "schema_version": 1, "candidate_id": candidate_id, "group_id": "PMC1234567",
        "image_sha256": image_sha, "status": "ai_assisted_panel_candidate",
        "panel_bbox_xyxy": [10, 20, 250, 200], "label_origin": "automated_cv_ai_assisted",
        "human_audited": False, "formal_approval": None,
        "operator_authentication": None, "eligible_for_scientific_acceptance": False,
    }
    pre_rel = f"records/{candidate_id}.json"
    (pre / pre_rel).write_text(json.dumps(pre_record), encoding="utf-8")
    pre_manifest = {
        "schema_version": 1, "status": "complete_ai_assisted_panel_preannotations",
        "source_manifest_sha256": dataset_sha, "record_count": 1,
        "human_audited_count": 0, "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "records": [{"candidate_id": candidate_id, "group_id": "PMC1234567",
                     "image_sha256": image_sha, "status": "ai_assisted_panel_candidate",
                     "record_path": pre_rel}],
    }
    pre_manifest_path = pre / "manifest.json"
    pre_manifest_path.write_text(json.dumps(pre_manifest), encoding="utf-8")
    return dataset_manifest_path, pre_manifest_path


def test_load_figure_review_queue_binds_manifests_and_image(tmp_path):
    from mcp_server.figure_review_queue import load_figure_review_queue

    dataset, pre = fixtures(tmp_path)
    queue = load_figure_review_queue(dataset, pre)
    assert queue["record_count"] == 1
    assert queue["formal_approval_supported"] is False
    assert queue["records"][0]["preannotation"]["panel_bbox_xyxy"] == [10, 20, 250, 200]


@pytest.mark.parametrize("kind", ["ai_assisted", "human"])
def test_screening_packet_rejects_approved_for_every_operator_kind(tmp_path, kind):
    from mcp_server.figure_review_queue import load_figure_review_queue, make_screening_packet

    record = load_figure_review_queue(*fixtures(tmp_path))["records"][0]
    with pytest.raises(ValueError, match="external authenticated registry"):
        make_screening_packet(record, "approved", kind, "no")


def test_ai_screening_packet_is_immutable_nonapproving(tmp_path):
    from mcp_server.figure_review_queue import (
        load_figure_review_queue, make_screening_packet, save_screening_packet)

    record = load_figure_review_queue(*fixtures(tmp_path))["records"][0]
    packet = make_screening_packet(record, "screen_positive", "ai_assisted", "visible plot")
    assert packet["status"] == "ai_assisted_figure_screening"
    assert packet["human_audited"] is False
    assert packet["eligible_for_scientific_acceptance"] is False
    saved = save_screening_packet(packet, tmp_path / "screening")
    assert Path(saved["path"]).is_file()
    with pytest.raises(FileExistsError):
        save_screening_packet(packet, tmp_path / "screening")


def test_freeze_evaluation_allocation_keeps_labels_and_approval_null(tmp_path):
    from mcp_server.figure_review_queue import (
        freeze_evaluation_allocation, load_figure_review_queue)

    queue = load_figure_review_queue(*fixtures(tmp_path))
    result = freeze_evaluation_allocation(queue, tmp_path / "allocation", "oa-band-eval-v1")
    allocation = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert allocation["record_count"] == 1
    assert allocation["document_group_count"] == 1
    assert allocation["frozen_before_human_review"] is True
    assert allocation["formal_approved_count"] == 0
    assert allocation["records"][0]["intended_split"] == "evaluation"
    assert allocation["records"][0]["reference_label"] is None
