"""Bounded security regressions: no network, shell payloads, or host OOM tests."""

import csv
import io
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "identifier", ["=1+1", "+1+1", "-1+1", "@SUM(1,1)", "\t=1+1", "\r=1+1", " \ufeff=1+1"]
)
@pytest.mark.parametrize("points", [3, 2050])
def test_band_csv_treats_untrusted_identifiers_as_literal_text(
    tmp_path, monkeypatch, identifier, points
):
    from mcp_server.observations_v2 import analyze
    from mcp_server.artifacts import get
    from tests.test_mcp_release_contracts import observation

    monkeypatch.setenv("BAND_MCP_STORE_DIR", str(tmp_path / "store"))
    data = observation()
    data["bands"][0]["band_id"] = identifier
    data["k_distance"] = list(range(points))
    data["segment_ids"] = [0] * points
    data["bands"][0]["energies_eV"] = [-1.0] * points
    data["bands"][1]["energies_eV"] = [1.0] * points
    result = analyze(data)
    exported = result["exports"]
    text = exported.get("csv_text") or get(exported["result_id"])[0].decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["band_id"] == "'" + identifier
    assert rows[0]["energy_eV"] == "-1.0"


def test_snapshot_loader_does_not_retain_all_image_payloads(tmp_path):
    from src.vision.figure_candidate_preannotator import _load_snapshot
    from tests.test_figure_candidate_preannotator import make_dataset

    manifest, loaded = _load_snapshot(make_dataset(tmp_path))
    assert manifest["record_count"] == 1
    assert not isinstance(loaded[0][1], bytes)


def test_snapshot_size_budget_precedes_read(tmp_path, monkeypatch):
    from src.vision import figure_candidate_preannotator as module
    from tests.test_figure_candidate_preannotator import make_dataset

    manifest = make_dataset(tmp_path)
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 16, raising=False)
    with pytest.raises(ValueError, match="image.*budget|image.*large"):
        module._load_snapshot(manifest)


def test_parser_subprocess_reports_enforced_memory_limit():
    import base64
    import pymupdf
    from mcp_server.server import _run_upload_worker

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((30, 40), "bounded-memory legitimate PDF")
        payload = base64.b64encode(document.tobytes()).decode()
    result = _run_upload_worker(
        "extract_document",
        {"payload_base64": payload, "kind": "pdf", "max_pages": 1, "page_start": 1},
    )
    assert result["status"] == "ok"
    assert result["resource_limits"]["memory_limit_enforced"] is True
    assert 128 * 1024 * 1024 <= result["resource_limits"]["memory_limit_bytes"] <= 2 * 1024**3


def test_real_child_allocation_cannot_exceed_small_limit():
    import subprocess, sys, os

    code = "from mcp_server.resource_limits import enforce_memory_limit; enforce_memory_limit(128*1024*1024)\ntry:\n x=bytearray(256*1024*1024)\nexcept MemoryError:\n print('allocation_blocked')\nelse:\n raise SystemExit('limit_not_enforced')"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0 and result.stdout.strip() == "allocation_blocked", result.stderr


def test_bounded_worker_output_and_timeout():
    import subprocess, sys, os
    from mcp_server.resource_limits import run_bounded, WorkerOutputLimit

    with pytest.raises(WorkerOutputLimit):
        run_bounded(
            [sys.executable, "-c", "import sys;sys.stdout.write('x'*10000)"],
            input=b"",
            env=os.environ.copy(),
            timeout=10,
            max_stdout=100,
        )
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded(
            [sys.executable, "-c", "import time;time.sleep(2)"],
            input=b"",
            env=os.environ.copy(),
            timeout=0.1,
        )


def test_limit_setup_failure_never_continues_to_document_parsing(monkeypatch):
    from mcp_server import upload_worker, resource_limits

    def denied():
        raise OSError("synthetic limit installation failure")

    monkeypatch.setattr(resource_limits, "enforce_memory_limit", denied)
    with pytest.raises(SystemExit) as exc:
        upload_worker.main()
    assert exc.value.code == 5


def test_small_component_filter_does_not_rescan_all_pixels_per_label(monkeypatch):
    import numpy as np, cv2
    from src.vision.multi_format_parser import MultiFormatParser

    class Labels(np.ndarray):
        calls = 0

        def __eq__(self, other):
            type(self).calls += 1
            return super().__eq__(other)

    labels = np.arange(256, dtype=np.int32).reshape(16, 16).view(Labels)
    stats = np.ones((256, 5), dtype=np.int32)
    monkeypatch.setattr(
        cv2, "connectedComponentsWithStats", lambda *a, **k: (256, labels, stats, None)
    )
    binary = np.ones((16, 16), dtype=np.uint8) * 255
    out = MultiFormatParser(detector_path=None)._filter_text_blobs(binary)
    assert Labels.calls <= 1
    assert np.count_nonzero(out) == 1


def test_snapshot_rechecks_data_between_admission_and_consumption(tmp_path):
    from src.vision.figure_candidate_preannotator import _load_snapshot, _preannotate
    from tests.test_figure_candidate_preannotator import make_dataset

    _, loaded = _load_snapshot(make_dataset(tmp_path))
    loaded[0][1].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        _preannotate(loaded[0])


def test_csv_safe_text_keeps_numeric_values_and_unicode_labels():
    from src.utils.csv_safety import spreadsheet_text

    assert spreadsheet_text(-2.5) == -2.5
    assert spreadsheet_text("Γ band") == "Γ band"
    assert spreadsheet_text(None) is None
    assert spreadsheet_text("\u200b=1+1").startswith("'")


def test_cache_writes_are_not_advertised_as_read_only():
    import asyncio
    from mcp_server.server import mcp

    tools = asyncio.run(mcp.list_tools())
    writable = {
        "analyze_visual_observations",
        "extract_band_from_image",
        "analyze_attachment",
        "export_attachment",
        "validate_axis_calibration",
    }
    assert {t.name for t in tools if t.annotations.readOnlyHint is False} == writable


def test_many_segment_csv_fallback_neutralizes_formula(tmp_path, monkeypatch):
    from mcp_server.observations_v2 import export_samples
    from tests.test_mcp_release_contracts import observation

    data = observation()
    n = 120
    data["k_distance"] = list(range(n))
    data["segment_ids"] = list(range(n))
    for band in data["bands"]:
        band["energies_eV"] = [-1.0] * n
    data["bands"][0]["band_id"] = "=1+1"
    result = export_samples(data)
    assert next(csv.DictReader(io.StringIO(result["csv_text"])))["band_id"] == "'=1+1"
