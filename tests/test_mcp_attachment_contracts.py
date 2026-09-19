"""Synthetic attachment, scientific-data, lifecycle and protocol regressions."""

import asyncio
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BAND_MCP_STORE_DIR", str(tmp_path / "store"))


def mesh():
    return {
        "data_kind": "uniform_mesh",
        "energies_eV": [[-2, -1], [1, 2]],
        "occupancies": [[2, 2], [0, 0]],
        "occupation_full": 2,
        "k_fractional": [[0, 0, 0], [0.5, 0, 0]],
        "k_weights": [0.5, 0.5],
        "lattice_A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "band_ids": ["v", "c"],
    }


def test_default_store_survives_minimal_client_environment(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from mcp_server.artifacts import store_directory

    monkeypatch.delenv("BAND_MCP_STORE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    platform_data = tmp_path / ("AppData/Local" if os.name == "nt" else ".local/share")
    monkeypatch.setenv("LOCALAPPDATA", str(platform_data))
    configured = store_directory()
    monkeypatch.delenv("LOCALAPPDATA")
    assert store_directory() == configured


def test_persisted_pdf_reference_is_idempotent():
    from mcp_server.artifacts import put

    first = put(b"{}", kind="pdf_geometry_json", deduplicate=True)
    second = put(b"{}", kind="pdf_geometry_json", deduplicate=True)
    assert first == second


def test_mesh_gap_is_not_a_path_or_global_gap():
    from mcp_server.scientific_data import analyze

    result = analyze(mesh())
    assert result["sampled_mesh_gap_eV"] == 2 and result["sampled_direct_gap_eV"] == 3
    assert result["line_mode_gap_eV"] is None and result["global_gap_eV"] is None
    assert list(result).index("sampled_mesh_gap_eV") < list(result).index("band_ids")
    assert (
        result["weighted_electron_count"] == 2 and not result["eligible_for_scientific_acceptance"]
    )


@pytest.mark.parametrize("energies", [[[-1, 1], [2, 3]], [[-2, -1], [1, 2]]])
def test_line_mode_reports_shape_for_generic_attachment_export(energies):
    from mcp_server.scientific_data import analyze

    result = analyze({"data_kind": "line_mode", "energies_eV": energies,
                      "fermi_eV": 0, "k_unit": "relative", "k_distance": [0, 1],
                      "segment_ids": [0, 0]})
    assert result["status"] != "refused"
    assert result["band_count"] == 2 and result["k_point_count"] == 2
    assert result["global_gap_eV"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("k_weights", [0, 1]),
        ("k_weights", [1, 1]),
        ("band_ids", ["x", "x"]),
        ("occupation_full", True),
        ("occupancies", [[3, 2], [0, 0]]),
        ("lattice_A", [[0, 0, 0]] * 3),
        ("data_kind", "unknown"),
        ("energies_eV", [[float("nan"), 0]]),
        ("physical_metadata", {"soc": "yes"}),
        ("physical_metadata", {"electron_count": 10**500}),
    ],
)
def test_invalid_mesh_is_refused(field, value):
    from mcp_server.scientific_data import analyze

    data = mesh()
    data[field] = value
    assert analyze(data)["status"] == "refused"


def test_partial_occupations_do_not_certify_metallicity():
    from mcp_server.scientific_data import analyze

    data = mesh()
    data["occupancies"][0][0] = 1.5
    result = analyze(data)
    assert result["status"] == "partial_mesh" and result["sampled_mesh_gap_eV"] is None


@pytest.mark.parametrize("mode,coords", [("Direct", "0.5 0 0"), ("Cartesian", "1 0 0")])
def test_poscar_coordinate_conventions(mode, coords):
    from mcp_server.scientific_data import parse_poscar

    result = parse_poscar(f"fixture\n1\n2 0 0\n0 2 0\n0 0 2\nSi\n1\n{mode}\n{coords}\n".encode())
    assert result["fractional_coordinates"] == [[0.5, 0, 0]]
    assert result["space_group"] is None and len(result["source_sha256"]) == 64


def test_xml_rejects_late_doctype(tmp_path):
    from mcp_server.scientific_data import parse_vasprun

    path = tmp_path / "vasprun.xml"
    path.write_bytes(b" " * (1024 * 1024) + b"<!DOCTYPE x><x/>")
    with pytest.raises(ValueError, match="XML_ENTITIES"):
        parse_vasprun(path)


def test_attachment_image_registered_bytes_and_no_arbitrary_path(tmp_path):
    from mcp_server.attachment_adapter import prepare
    from mcp_server.server import inspect_attachment, analyze_attachment

    path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(path)
    registered = prepare(path)
    result = inspect_attachment(registered["attachment_id"])
    assert result["source_sha256"] == registered["source_sha256"] and result["status"] == "ok"
    assert inspect_attachment(str(path))["status"] == "refused"
    assert analyze_attachment(registered["attachment_id"], "scientific")["status"] == "refused"


def test_result_survives_module_restart_and_sha_bound_pagination(tmp_path):
    import importlib
    from mcp_server import artifacts

    payload = ("样本,α\n" * 100).encode()
    saved = artifacts.put(payload, kind="test")
    artifacts = importlib.reload(artifacts)
    offset = 0
    parts = []
    while True:
        chunk = artifacts.read_chunk(saved["artifact_id"], offset, 17, saved["sha256"])
        parts.append(chunk["payload_chunk"])
        offset = chunk["next_offset"]
        if offset is None:
            break
    assert "".join(parts).encode() == payload
    with pytest.raises(ValueError, match="VERSION_CHANGED"):
        artifacts.read_chunk(saved["artifact_id"], 0, 10, "0" * 64)
    with pytest.raises(ValueError, match="Continuation"):
        artifacts.read_chunk(saved["artifact_id"], 1, 10)


def test_tamper_expiry_and_operator_prune():
    from mcp_server import artifacts as a

    saved = a.put(b"original", kind="test")
    data, _ = a._paths(saved["artifact_id"])
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="INTEGRITY"):
        a.get(saved["artifact_id"])
    expired = a.put(b"past", kind="test", ttl_seconds=-1)
    attachment = a.put(b"keep", kind="test", ttl_seconds=None, attachment=True)
    with pytest.raises(ValueError, match="EXPIRED"):
        a.get(expired["artifact_id"])
    assert expired["artifact_id"] in a.prune_expired()["expired_result_ids"]
    a.prune_expired(apply=True)
    assert a.get(attachment["artifact_id"])[0] == b"keep"


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_pdf_adapter_keeps_invertible_mapping(tmp_path, rotation):
    import pymupdf
    from mcp_server.attachment_adapter import prepare

    path = tmp_path / "rotated.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_rotation(rotation)
        doc.save(path)
    result = prepare(path, page_number=1, max_dimension=400)
    meta = result["metadata"]
    a, b, c, d, e, f = meta["pixel_to_unrotated_pdf_point"]
    assert meta["source_rotation"] == rotation and abs(a * d - b * c) > 0
    assert max(meta["image_size"]) <= 401
    assert meta["original_source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_anchor_sign_and_sample_binding(tmp_path):
    from mcp_server.attachment_adapter import prepare
    from mcp_server.evidence import validate_axes, review_calibrated_samples

    path = tmp_path / "axis.png"
    Image.new("RGB", (100, 100), "white").save(path)
    saved = prepare(path)
    e = [{"pixel": 10, "value": 2, "raw_text": "2"}, {"pixel": 90, "value": -2, "raw_text": "−2"}]
    k = [{"pixel": 10, "value": 0, "raw_text": "0"}, {"pixel": 90, "value": 1, "raw_text": "1"}]
    result = validate_axes(saved["attachment_id"], e, k)
    observation = {
        "calibration_id": result["calibration_id"],
        "source": {
            "attachment_id": saved["attachment_id"],
            "source_sha256": saved["source_sha256"],
        },
        "energy_unit": "eV",
        "k_unit": "relative",
        "k_distance": [0, 1],
        "bands": [{"energies_eV": [2, -2], "pixel_points": [[10, 10], [90, 90]]}],
    }
    assert review_calibrated_samples(observation) == []
    observation["bands"][0]["energies_eV"][0] = 1
    assert review_calibrated_samples(observation) == ["calibration.energy_mismatch"]
    bad = copy.deepcopy(e)
    bad[1]["raw_text"] = "2"
    with pytest.raises(ValueError, match="LABEL_VALUE"):
        validate_axes(saved["attachment_id"], bad, k)


@pytest.mark.parametrize(
    "report",
    [
        {"document_kind": [], "claims": []},
        {"document_kind": "computational", "claims": [{"kind": [], "basis": "reported"}]},
        {
            "document_kind": "computational",
            "claims": [{"kind": "property", "basis": "inferred", "value": float("inf")}],
        },
    ],
)
def test_malformed_material_evidence_is_not_an_exception(report):
    from mcp_server.evidence import review_material_report

    assert review_material_report(report)["status"] in {"refused", "needs_more_evidence"}


def test_computational_paper_cannot_be_experimental_sop():
    from mcp_server.evidence import review_material_report

    r = review_material_report(
        {
            "document_kind": "computational",
            "claims": [
                {
                    "kind": "experimental_synthesis",
                    "basis": "reported",
                    "value": "Synthetic example only",
                    "evidence": [
                        {
                            "document_id": "synthetic",
                            "page_number": 1,
                            "excerpt": "calculation",
                            "origin": "current_paper",
                        }
                    ],
                }
            ],
        }
    )
    assert r["status"] == "needs_more_evidence" and not r["human_audited"]


def test_unrelated_pdf_misses_run_concurrently():
    from mcp_server.pdf_result_cache import PDFResultCache

    cache = PDFResultCache()
    barrier = Barrier(2)

    def factory():
        barrier.wait(timeout=3)
        return "{}"

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(cache.get, key, factory) for key in ["first", "second"]]
        assert [f.result(timeout=5) for f in futures] == ["{}", "{}"]


def test_all_published_schemas_are_valid_and_match_inventory():
    from jsonschema import Draft202012Validator
    from mcp_server.server import mcp, get_service_status

    async def verify():
        tools = await mcp.list_tools()
        resources = await mcp.list_resources()
        assert len(tools) == get_service_status()["tool_count"] == 17
        assert len(resources) == get_service_status()["resource_count"] == 5
        for tool in tools:
            Draft202012Validator.check_schema(tool.inputSchema)
            if tool.outputSchema:
                Draft202012Validator.check_schema(tool.outputSchema)
        schema = next(t.inputSchema for t in tools if t.name == "analyze_electronic_data")
        Draft202012Validator(schema).validate({"data": mesh()})

    asyncio.run(verify())


def test_symbolic_k_labels_require_explicit_relative_plot_basis(tmp_path):
    from mcp_server.attachment_adapter import prepare
    from mcp_server.evidence import validate_axes

    path = tmp_path / "symbolic.png"
    Image.new("RGB", (200, 150), "white").save(path)
    identifier = prepare(path)["attachment_id"]
    energy = [
        {"pixel": 10, "value": 2, "raw_text": "2"},
        {"pixel": 110, "value": -2, "raw_text": "-2"},
    ]
    ticks = [
        {"pixel": 10, "value": 0, "raw_text": "Γ", "value_basis": "relative_plot_position"},
        {"pixel": 190, "value": 1, "raw_text": "X", "value_basis": "relative_plot_position"},
    ]
    result = validate_axes(identifier, energy, ticks)
    assert result["k_unit"] == "relative" and result["human_audited"] is False
    with pytest.raises(ValueError, match="SYMBOLIC"):
        validate_axes(identifier, energy, ticks, "angstrom^-1")
    ticks[1]["value"] = 0.5
    with pytest.raises(ValueError, match="RELATIVE_K"):
        validate_axes(identifier, energy, ticks)
