"""Figure candidate preannotation remains hash-bound and non-human."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "records").mkdir()
    image_path = root / "images" / f"epmc-band-{'0' * 24}.png"
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 560, 410), outline="black", width=3)
    for offset in range(8):
        points = [(80 + x, 235 + int(80 * __import__("math").sin((x + offset * 12) / 60)))
                  for x in range(0, 481, 6)]
        draw.line(points, fill="black", width=2)
    image.save(image_path)
    payload = image_path.read_bytes()
    image_sha = hashlib.sha256(payload).hexdigest()
    candidate_id = "epmc-band-" + "0" * 24
    record = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source": {"pmcid": "PMC1234567", "license": "CC-BY"},
        "group_id": "PMC1234567",
        "image": {"path": f"images/{image_path.name}", "sha256": image_sha},
        "annotation": {"class": "electronic_band_structure_figure",
                       "level": "caption_matched_ai_candidate"},
        "human_audited": False,
        "formal_approval": None,
        "operator_authentication": None,
        "eligible_for_scientific_acceptance": False,
    }
    (root / "records" / f"{candidate_id}.json").write_text(
        json.dumps(record), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete_ai_assisted_figure_candidate_snapshot",
        "record_count": 1,
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "records": [{"candidate_id": candidate_id,
                     "record_path": f"records/{candidate_id}.json",
                     "image_path": f"images/{image_path.name}",
                     "image_sha256": image_sha}],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_preannotate_snapshot_is_nonapproving_and_hash_bound(tmp_path):
    from src.vision.figure_candidate_preannotator import preannotate_snapshot

    manifest_path = make_dataset(tmp_path)
    result = preannotate_snapshot(manifest_path, tmp_path / "preannotations", workers=1)
    output = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert output["record_count"] == 1
    assert output["formal_approved_count"] == 0
    assert output["human_audited_count"] == 0
    assert output["eligible_for_scientific_acceptance"] is False
    annotation = json.loads(next((tmp_path / "preannotations/records").glob("*.json")).read_text("utf-8"))
    assert annotation["label_origin"] == "automated_cv_ai_assisted"
    assert annotation["human_audited"] is False
    assert annotation["formal_approval"] is None
    assert annotation["image_sha256"] == output["records"][0]["image_sha256"]


def test_preannotate_snapshot_rejects_tampered_image(tmp_path):
    from src.vision.figure_candidate_preannotator import preannotate_snapshot

    manifest_path = make_dataset(tmp_path)
    image_path = next((manifest_path.parent / "images").glob("*.png"))
    image_path.write_bytes(image_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA"):
        preannotate_snapshot(manifest_path, tmp_path / "preannotations", workers=1)


def test_preannotate_snapshot_refuses_existing_output(tmp_path):
    from src.vision.figure_candidate_preannotator import preannotate_snapshot

    manifest_path = make_dataset(tmp_path)
    output = tmp_path / "preannotations"
    output.mkdir()
    with pytest.raises(FileExistsError):
        preannotate_snapshot(manifest_path, output, workers=1)
