"""Contracts for harvesting reusable electronic-band figures from Europe PMC."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest


ARTICLE_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <body>
    <fig id="F1">
      <label>Figure 1</label>
      <caption><p>Calculated electronic band structure and density of states of ZnO.</p></caption>
      <graphic xlink:href="paper-f1.jpg"/>
    </fig>
    <fig id="F2">
      <label>Figure 2</label>
      <caption><p>Photonic band structure of the resonator.</p></caption>
      <graphic xlink:href="paper-f2.jpg"/>
    </fig>
    <fig id="F3">
      <label>Figure 3</label>
      <caption><p>Electronic band structure reproduced with permission from Ref. 3.</p></caption>
      <graphic xlink:href="paper-f3.jpg"/>
    </fig>
  </body>
</article>
"""


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def zip_bytes(name: str = "paper-f1.jpg", payload: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload or image_bytes())
    return buffer.getvalue()


def test_parse_figures_keeps_electronic_and_rejects_non_electronic_or_third_party():
    from src.data.europe_pmc_figure_harvester import parse_electronic_band_figures

    figures = parse_electronic_band_figures(ARTICLE_XML)
    assert figures == [{
        "figure_id": "F1",
        "figure_label": "Figure 1",
        "caption": "Calculated electronic band structure and density of states of ZnO.",
        "asset_href": "paper-f1.jpg",
    }]


@pytest.mark.parametrize("value", ["cc by", "CC-BY", "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"])
def test_normalise_license_accepts_only_reusable_cc_by(value):
    from src.data.europe_pmc_figure_harvester import normalise_license

    assert normalise_license(value) == "CC-BY"


@pytest.mark.parametrize("value", ["cc by-nc", "cc by-nd", "copyright", "", None])
def test_normalise_license_rejects_restricted_or_unknown(value):
    from src.data.europe_pmc_figure_harvester import normalise_license

    with pytest.raises(ValueError):
        normalise_license(value)


def test_extract_zip_asset_is_exact_bounded_and_rejects_path_alias():
    from src.data.europe_pmc_figure_harvester import extract_zip_asset

    payload = image_bytes()
    assert extract_zip_asset(zip_bytes(payload=payload), "paper-f1.jpg") == payload
    with pytest.raises(ValueError):
        extract_zip_asset(zip_bytes("nested/paper-f1.jpg"), "../paper-f1.jpg")
    with pytest.raises(ValueError):
        extract_zip_asset(zip_bytes("other.jpg"), "paper-f1.jpg")


def test_candidate_record_is_source_bound_and_never_human_approved():
    from src.data.europe_pmc_figure_harvester import build_candidate_record

    article = {
        "pmcid": "PMC1234567",
        "doi": "10.1/example",
        "title": "A paper",
        "authorString": "A. Author",
        "journalTitle": "Journal",
        "pubYear": "2024",
        "license": "cc by",
        "isOpenAccess": "Y",
        "isRetracted": "N",
    }
    figure = {
        "figure_id": "F1",
        "figure_label": "Figure 1",
        "caption": "Electronic band structure of ZnO.",
        "asset_href": "paper-f1.jpg",
    }
    record = build_candidate_record(article, figure, image_bytes(), "a" * 64, "b" * 64)
    assert record["candidate_id"].startswith("epmc-band-")
    assert record["source"]["pmcid"] == "PMC1234567"
    assert record["source"]["license"] == "CC-BY"
    assert record["annotation"]["class"] == "electronic_band_structure_figure"
    assert record["annotation"]["level"] == "caption_matched_ai_candidate"
    assert record["human_audited"] is False
    assert record["eligible_for_scientific_acceptance"] is False
    assert record["formal_approval"] is None


def test_write_snapshot_refuses_symlink_and_freezes_nonapproving_manifest(tmp_path):
    from src.data.europe_pmc_figure_harvester import build_candidate_record, write_snapshot

    article = {
        "pmcid": "PMC1234567", "doi": "10.1/example", "title": "A paper",
        "authorString": "A. Author", "journalTitle": "Journal", "pubYear": "2024",
        "license": "cc by", "isOpenAccess": "Y", "isRetracted": "N",
    }
    figure = {"figure_id": "F1", "figure_label": "Figure 1",
              "caption": "Electronic band structure of ZnO.", "asset_href": "paper-f1.jpg"}
    record = build_candidate_record(article, figure, image_bytes(), "a" * 64, "b" * 64)
    result = write_snapshot(tmp_path / "snapshot", [(record, image_bytes())], target_count=1)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert manifest["formal_approved_count"] == 0
    assert manifest["human_audited_count"] == 0
    assert manifest["eligible_for_scientific_acceptance"] is False
    with pytest.raises(FileExistsError):
        write_snapshot(tmp_path / "snapshot", [(record, image_bytes())], target_count=1)
