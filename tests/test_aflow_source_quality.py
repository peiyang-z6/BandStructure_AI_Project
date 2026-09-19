"""Offline source QA, not a generic detector or scientific validation.

Keep the original two-response regressions and the independently reviewed six
full responses/caches. Only scratch HDF5 copies are used; never the real HDF5,
network, outer arrays, or models. Energy mapping is not full-contract validity.
"""
from pathlib import Path
import hashlib
import json
import lzma
import os

import numpy as np
import pytest

from src.data.aflow_adapter import AFLOWAdapter

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/reports/p3_diagnostic_20260909/source_response_evidence"
BAD_ID = "aflow-46374655e4730ce2"
GOOD_ID = "aflow-0015a3db328a3b49"
SHA = {
    BAD_ID: "fdd89582c412d6e5a4b9502781e5400e97715075ae8591842a9b5c3d1c67b4aa",
    GOOD_ID: "8eb02ee4238273fdff849e6ae3d31d5349f17dece172957894089229440974c2",
}


def bound_payload(mid):
    content = (EVIDENCE / (mid + ".json.xz")).read_bytes()
    assert hashlib.sha256(content).hexdigest() == SHA[mid]
    return AFLOWAdapter._lenient_json(lzma.decompress(content).decode("utf-8-sig"))


def tiny_payload(**updates):
    return {"Efermi": 5.5, "n_bands": 2, "n_kpoints": 3,
            "bands_data": [[0, -50, 0], [0.5, -49, 0], [1, -48, 0]], **updates}


def test_bound_published_anomaly_is_quarantined_by_content_not_id():
    raw = bound_payload(BAD_ID)
    for mid in (BAD_ID, "different-identity-same-source-content"):
        with pytest.raises(ValueError, match="source_quarantine"):
            AFLOWAdapter._parse_band_payload(mid, raw)
    # The same ID with different, normal content must not be blacklisted.
    corrected = AFLOWAdapter._parse_band_payload(BAD_ID, tiny_payload())
    np.testing.assert_array_equal(corrected["energies"][1], 0)


def test_bound_normal_deep_valence_response_stays_supported():
    raw = bound_payload(GOOD_ID)
    parsed = AFLOWAdapter._parse_band_payload(GOOD_ID, raw)
    assert parsed["energies"].shape == (raw["n_bands"], raw["n_kpoints"])
    np.testing.assert_array_equal(parsed["energies"], np.asarray(raw["bands_data"], np.float32)[:, 1:].T)
    assert parsed["efermi"] == 0.0

@pytest.mark.parametrize("declared", [1, 3, 2.5, "2", True, None, -2])
def test_parser_rejects_inconsistent_or_illegal_declared_n_bands(declared):
    with pytest.raises(ValueError, match="declared_n_bands"):
        AFLOWAdapter._parse_band_payload("synthetic", tiny_payload(n_bands=declared))

@pytest.mark.parametrize("declared", [1, 4, 3.5, "3", True, None, -3])
def test_parser_rejects_inconsistent_or_illegal_declared_n_kpoints(declared):
    with pytest.raises(ValueError, match="declared_n_kpoints"):
        AFLOWAdapter._parse_band_payload("synthetic", tiny_payload(n_kpoints=declared))

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_parser_rejects_nonfinite_energy(value):
    raw = tiny_payload()
    raw["bands_data"][1][1] = value
    with pytest.raises(ValueError, match="nonfinite_energy"):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "1.2", [1.2]])
def test_parser_rejects_illegal_source_ef(value):
    with pytest.raises(ValueError, match="invalid_source_efermi"):
        AFLOWAdapter._parse_band_payload("synthetic", tiny_payload(Efermi=value))

@pytest.mark.parametrize("k", [[0, np.nan, 1], [0, 1, np.inf], [-np.inf, 0.5, 1],
                              [-1, 0, 1], [0, 0, 0]])
def test_parser_rejects_illegal_k_coordinate(k):
    raw = tiny_payload()
    for row, coordinate in zip(raw["bands_data"], k):
        row[0] = coordinate
    with pytest.raises(ValueError, match="invalid_k_distances"):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

@pytest.mark.parametrize("positions", [[0, np.nan], [0, np.inf], [0, 2], [0]])
def test_parser_rejects_invalid_k_label_positions(positions):
    raw = tiny_payload(kpoint_labels=["G", "X"], kpoint_positions=positions)
    with pytest.raises(ValueError, match="invalid_kpoint_positions"):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

def test_missing_spin_is_unknown_not_false_and_survives_storage(tmp_path):
    import h5py
    from src.data.band_store import to_jsonable
    result = AFLOWAdapter._parse_band_payload("unknown-spin", tiny_payload())
    assert result["is_spin_polarized"] is None
    assert result["spin_status"] == "unknown"
    assert result["source_quality"]["status"] == "ambiguous"
    assert "unknown_spin" in result["source_quality"]["reasons"]
    assert json.loads(json.dumps(to_jsonable(result)))["is_spin_polarized"] is None
    adapter = AFLOWAdapter(str(tmp_path))
    adapter.save_to_hdf5(result)
    with h5py.File(adapter.h5_path, "r") as handle:
        group = handle["unknown-spin"]
        assert "is_spin_polarized" not in group.attrs  # band_store omits None
        assert group.attrs["spin_status"] == "unknown"
        np.testing.assert_array_equal(group["energies"], result["energies"])

def test_declared_normalized_single_channel_contract_is_verified_valid():
    # These are explicit normalized importer declarations, NOT claims that the
    # historical AFLOW responses actually supplied these fields.
    raw = tiny_payload(band_schema="aflow_bandsdata_v1", band_layout="bands_k",
                       spin_channels=1, source_semantics_evidence="synthetic-contract-v1")
    result = AFLOWAdapter._parse_band_payload("synthetic", raw)
    assert result["source_quality"]["status"] == "verified_valid"
    assert result["source_quality"]["reasons"] == []
    assert result["is_spin_polarized"] is False
    assert result["spin_status"] == "declared"
    np.testing.assert_array_equal(result["energies"], [[-50, -49, -48], [0, 0, 0]])

def test_declared_spin_major_flat_retains_both_unsorted_channels():
    raw = tiny_payload(band_schema="aflow_bandsdata_v1", band_layout="spin_major_flat",
                       spin_channels=2, source_semantics_evidence="synthetic-two-spin-v1",
                       n_bands=4, bands_data=[[0, -50, 1, -49, 2],
                                             [0.5, -49, 2, -48, 3], [1, -48, 3, -47, 4]])
    result = AFLOWAdapter._parse_band_payload("synthetic-spin", raw)
    assert result["source_quality"]["status"] == "verified_valid"
    assert result["is_spin_polarized"] is True
    assert result["energies"].shape == (4, 3)
    np.testing.assert_array_equal(result["energies"], np.asarray(raw["bands_data"], np.float32)[:, 1:].T)

@pytest.mark.parametrize("updates, reason", [
    ({"spin_channels": 0}, "invalid_spin_channels"),
    ({"spin_channels": True}, "invalid_spin_channels"),
    ({"spin_channels": 1.5}, "invalid_spin_channels"),
    ({"spin_channels": 2, "band_layout": "bands_k"}, "inconsistent_spin_layout"),
    ({"spin_channels": 1, "band_layout": "spin_major_flat"}, "inconsistent_spin_layout"),
])
def test_contradictory_spin_declarations_are_invalid(updates, reason):
    raw = tiny_payload(band_schema="aflow_bandsdata_v1", band_layout="bands_k",
                       spin_channels=1, source_semantics_evidence="synthetic-contract-v1")
    raw.update(updates)
    with pytest.raises(ValueError, match=reason):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

def test_network_entry_defaults_formal_fail_closed_for_unknown_semantics(tmp_path, monkeypatch):
    compressed = lzma.compress(json.dumps(tiny_payload()).encode())
    adapter = AFLOWAdapter(str(tmp_path), rate_limit_delay=0)
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: compressed)
    result = adapter.fetch_band_structure("unknown", {"download_url": "https://offline.invalid/source"})
    assert "error" in result
    assert result["source_quality"]["status"] == "ambiguous"
    assert not (Path(adapter.json_cache_dir) / "unknown.json").exists()

def test_diagnostic_fetch_reports_unknown_without_promoting_valid(tmp_path, monkeypatch):
    compressed = lzma.compress(json.dumps(tiny_payload()).encode())
    adapter = AFLOWAdapter(str(tmp_path), rate_limit_delay=0)
    # API opt-in is explicit on the instance, never inferred from metadata.
    adapter.source_qa_mode = "diagnostic"
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: compressed)
    result = adapter.fetch_band_structure("unknown", {"download_url": "https://offline.invalid/source"})
    assert "error" not in result
    assert result["source_quality"]["status"] == "ambiguous"
    assert result["is_spin_polarized"] is None

def test_parsed_cache_quarantine_cannot_bypass_qa_or_redownload(tmp_path, monkeypatch):
    from src.data.band_store import to_jsonable
    raw = bound_payload(BAD_ID)
    table = np.asarray(raw["bands_data"], np.float32)
    cached = {"material_id": BAD_ID, "source": "aflow", "energies": table[:, 1:].T,
              "k_distances": table[:, 0], "efermi": 0.0,
              "source_efermi_absolute": raw["Efermi"], "num_bands": raw["n_bands"],
              "num_kpoints": raw["n_kpoints"], "source_sha256": SHA[BAD_ID],
              "source_quality": {"status": "verified_valid"}, "is_spin_polarized": False}
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    path = Path(adapter.json_cache_dir) / (BAD_ID + ".json")
    path.write_text(json.dumps(to_jsonable(cached)))
    before = path.read_bytes()
    def forbidden(url):
        pytest.fail("rejected cache attempted a network download")
    monkeypatch.setattr(adapter, "_request_bytes", forbidden)
    result = adapter.fetch_band_structure(BAD_ID, {"download_url": "https://offline.invalid/no"})
    assert "error" in result
    assert result["source_quality"]["status"] == "invalid"
    assert result["source_quality"]["reason"] == "source_quarantine_published_response_v1"
    assert path.read_bytes() == before

def test_unreadable_cache_is_read_error_without_redownload_or_rewrite(tmp_path, monkeypatch):
    adapter = AFLOWAdapter(str(tmp_path))
    path = Path(adapter.json_cache_dir) / "broken.json"
    path.write_text("{broken json")
    before = path.read_bytes()
    requests = []
    def forbidden(url):
        requests.append(url)
        raise OSError("no network in source QA")
    monkeypatch.setattr(adapter, "_request_bytes", forbidden)
    result = adapter.fetch_band_structure("broken", {"download_url": "https://offline.invalid/no"})
    assert requests == []
    assert result["source_quality"]["status"] == "read_error"
    assert path.read_bytes() == before

@pytest.mark.parametrize("column,value", [(0, True), (0, "0.5"), (1, True), (1, "-1"), (1, 1e300)])
def test_parser_rejects_coercible_non_numeric_or_unstorable_values(column, value):
    raw = tiny_payload()
    raw["bands_data"][1][column] = value
    with pytest.raises(ValueError, match="invalid_numeric_type|nonfinite_energy"):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

def test_declared_three_dimensional_spin_contract_validates_without_sorting():
    from src.data.aflow_adapter import assess_source_quality
    record = {"source": "mp", "source_schema": "mp_pymatgen_v1",
              "band_layout": "spin_band_k", "spin_channels": 2,
              "source_semantics_evidence": "synthetic-pymatgen-contract",
              "energies": np.array([[[-2, -2, -2], [1, 1, 1]],
                                     [[-3, -3, -3], [2, 2, 2]]]),
              "num_bands": 2, "num_kpoints": 3, "efermi": 0.3,
              "k_distances": np.array([0, 0.5, 1])}
    qa = assess_source_quality(record)
    assert qa["status"] == "verified_valid"
    assert qa["spin_channels"] == 2
    assert qa["layout"] == "spin_band_k"

@pytest.mark.parametrize("layout,channels,energies", [
    ("bands_k", 1, np.zeros((2, 2, 3))),
    ("spin_band_k", 2, np.zeros((3, 2, 3))),
    ("spin_band_k", 2, np.zeros((2, 3))),
    ("spin_major_flat", 2, np.zeros((3, 3))),
])
def test_declared_layout_must_match_actual_spin_axes(layout, channels, energies):
    from src.data.aflow_adapter import assess_source_quality
    qa = assess_source_quality({"source": "aflow", "source_schema": "aflow_bandsdata_v1",
        "band_layout": layout, "spin_channels": channels,
        "source_semantics_evidence": "synthetic-declaration", "energies": energies,
        "k_distances": [0, 0.5, 1], "efermi": 0.0})
    assert qa["status"] == "invalid"
    assert qa["reason"] == "inconsistent_spin_layout"

@pytest.mark.parametrize("key,value", [("material_id", "other-id"), ("source", "mp")])
def test_cache_identity_or_provider_mismatch_is_not_accepted(tmp_path, monkeypatch, key, value):
    from src.data.band_store import to_jsonable
    record = AFLOWAdapter._parse_band_payload("expected", tiny_payload())
    record[key] = value
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    path = Path(adapter.json_cache_dir) / "expected.json"
    path.write_text(json.dumps(to_jsonable(record)))
    before = path.read_bytes()
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: pytest.fail("network on cache rejection"))
    result = adapter.fetch_band_structure("expected")
    assert "error" in result
    assert result["source_quality"]["reason"] == "cache_identity_or_source_mismatch"
    assert path.read_bytes() == before

@pytest.mark.parametrize("kpoints", [[[0], [np.nan], [1]], [[0], [1]], [[0, 1], [0.5, 1], [1, 1]]])
def test_cached_kpoints_are_qa_checked_not_just_k_distances(tmp_path, monkeypatch, kpoints):
    from src.data.band_store import to_jsonable
    record = AFLOWAdapter._parse_band_payload("bad-kpoints", tiny_payload())
    record["kpoints"] = kpoints
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    path = Path(adapter.json_cache_dir) / "bad-kpoints.json"
    path.write_text(json.dumps(to_jsonable(record)))
    before = path.read_bytes()
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: pytest.fail("cache must not redownload"))
    result = adapter.fetch_band_structure("bad-kpoints")
    assert "error" in result
    assert result["source_quality"]["reason"] == "invalid_kpoints"
    assert before == path.read_bytes()

def test_failed_network_parse_retains_actual_response_sha_and_url(tmp_path, monkeypatch):
    raw = tiny_payload(n_bands=99)
    compressed = lzma.compress(json.dumps(raw).encode())
    adapter = AFLOWAdapter(str(tmp_path))
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: compressed)
    url = "https://offline.invalid/bad-response.json.xz"
    result = adapter.fetch_band_structure("invalid-source", {"download_url": url})
    assert result["source_quality"]["status"] == "invalid"
    assert result.get("source_sha256") == hashlib.sha256(compressed).hexdigest()
    assert result.get("source_url") == url
    assert not (Path(adapter.json_cache_dir) / "invalid-source.json").exists()

@pytest.mark.parametrize("payload", [None, [], 42])
def test_non_mapping_cached_record_is_explicit_invalid_not_a_crash(tmp_path, payload):
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    path = Path(adapter.json_cache_dir) / "bad-record.json"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    result = adapter.fetch_band_structure("bad-record")
    assert "error" in result
    assert result["source_quality"]["status"] == "invalid"
    assert result["source_quality"]["reason"] == "invalid_record_mapping"
    assert path.read_bytes() == before

@pytest.mark.parametrize("key,value", [("band_schema", np.nan), ("band_layout", ["bands_k"]),
                                       ("source_semantics_evidence", True)])
def test_semantics_metadata_cannot_forge_verified_valid(key, value):
    raw = tiny_payload(band_schema="aflow_bandsdata_v1", band_layout="bands_k",
                       spin_channels=1, source_semantics_evidence="synthetic-contract")
    raw[key] = value
    parsed = AFLOWAdapter._parse_band_payload("unknown-declaration", raw)
    qa = parsed["source_quality"]
    assert qa["status"] == "ambiguous"
    json.dumps(qa, allow_nan=False)


@pytest.mark.parametrize("k", [[0, 1e300, 2e300], [1e20, 1e20 + 1e5, 1e20 + 2e5]])
def test_parser_rejects_k_that_cannot_keep_finite_span_in_storage(k):
    raw = tiny_payload()
    for row, coordinate in zip(raw["bands_data"], k):
        row[0] = coordinate
    with pytest.raises(ValueError, match="invalid_k_distances"):
        AFLOWAdapter._parse_band_payload("synthetic", raw)

@pytest.mark.parametrize("key,value", [("num_bands", 9), ("num_kpoints", 9), ("efermi", np.nan),
    ("source_efermi_absolute", np.inf), ("energies", [[0, np.nan, 0], [1, 1, 1]]),
    ("k_distances", [0, np.nan, 1])])
def test_cache_recomputes_all_numeric_rules_without_writes(tmp_path, key, value):
    from src.data.aflow_adapter import assess_source_quality
    from src.data.band_store import to_jsonable
    record = AFLOWAdapter._parse_band_payload("cached", tiny_payload())
    record[key] = value
    record["source_quality"] = {"status": "verified_valid"}
    expected = assess_source_quality(record)
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    path = Path(adapter.json_cache_dir) / "cached.json"
    path.write_text(json.dumps(to_jsonable(record)))
    before = path.read_bytes()
    result = adapter.fetch_band_structure("cached")
    assert result["source_quality"] == expected
    assert result["source_quality"]["status"] == "invalid"
    assert "error" in result
    assert before == path.read_bytes()


@pytest.mark.parametrize("updates,reason", [({"band_schema": "future"}, "unknown_schema"),
    ({"band_layout": "interleaved"}, "unknown_layout"),
    ({"spin_channels": 4, "band_layout": "unknown"}, "unknown_spin"),
    ({"spin_channels": None}, "unknown_spin")])
def test_unrecognized_semantics_remain_ambiguous_not_invalid(updates, reason):
    raw = tiny_payload(band_schema="aflow_bandsdata_v1", band_layout="bands_k", spin_channels=1,
                       source_semantics_evidence="synthetic-contract")
    raw.update(updates)
    record = AFLOWAdapter._parse_band_payload("ambiguous", raw)
    assert record["source_quality"]["status"] == "ambiguous"
    assert reason in record["source_quality"]["reasons"]


@pytest.mark.parametrize("bands_data", [
    [[[0, -1, 1], [0.5, -1, 1], [1, -1, 1]]] * 2,
    {"up": [[0, -1, 1], [1, -1, 1]], "down": [[0, -2, 2], [1, -2, 2]]},
])
def test_unmapped_raw_layout_is_reported_ambiguous_not_material_invalid(tmp_path, monkeypatch, bands_data):
    raw = tiny_payload(bands_data=bands_data)
    compressed = lzma.compress(json.dumps(raw).encode())
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: compressed)
    result = adapter.fetch_band_structure("unsupported", {"download_url": "https://offline.invalid/layout"})
    assert "error" in result  # no eigenvalue axes were guessed
    assert result.get("source_quality", {}).get("status") == "ambiguous"
    assert result["source_quality"]["reason"] == "unsupported_raw_layout"
    assert "energies" not in result
    assert not (Path(adapter.json_cache_dir) / "unsupported.json").exists()


def test_declared_shifted_zero_cannot_hide_a_nonzero_canonical_ef():
    from src.data.aflow_adapter import assess_source_quality
    record = {"source": "mp", "source_schema": "mp_pymatgen_v1", "band_layout": "bands_k",
              "spin_channels": 1, "source_semantics_evidence": "synthetic-contract",
              "energies": [[-1, -1, -1], [1, 1, 1]], "k_distances": [0, 0.5, 1],
              "efermi": 2.0, "energy_reference": "fermi_shifted_zero"}
    qa = assess_source_quality(record)
    assert qa["status"] == "invalid"
    assert qa["reason"] == "inconsistent_energy_reference"


@pytest.mark.parametrize("raw", [None, {}, {"bands_data": [[0, True, 1], [1, -1, 1]], "Efermi": 0},
                                  {"bands_data": [[0, -1], [1, 1]], "Efermi": 0}])
def test_raw_mapping_failure_is_explicit_invalid_with_bound_bytes(tmp_path, monkeypatch, raw):
    compressed = lzma.compress(json.dumps(raw).encode())
    adapter = AFLOWAdapter(str(tmp_path), source_qa_mode="diagnostic")
    monkeypatch.setattr(adapter, "_request_bytes", lambda url: compressed)
    result = adapter.fetch_band_structure("malformed", {"download_url": "https://offline.invalid/malformed"})
    assert "error" in result
    assert result.get("source_quality", {}).get("status") == "invalid"
    assert result["source_quality"]["reason"] == "invalid_raw_payload"
    assert result["source_quality"]["rule_version"] == "aflow-source-qa-v1"
    assert result["source_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert not (Path(adapter.json_cache_dir) / "malformed.json").exists()


@pytest.mark.parametrize("response", [b"not an xz stream", lzma.compress(b"{broken JSON"), None])
def test_unreadable_response_is_read_error_not_material_invalid(tmp_path, monkeypatch, response):
    adapter = AFLOWAdapter(str(tmp_path))
    def read(url):
        if response is None:
            raise OSError("offline read failure")
        return response
    monkeypatch.setattr(adapter, "_request_bytes", read)
    result = adapter.fetch_band_structure("unreadable", {"download_url": "https://offline.invalid/read"})
    assert "error" in result
    assert result.get("source_quality", {}).get("status") == "read_error"
    assert result["source_quality"]["reason"] == "source_response_read_error"
    expected = None if response is None else hashlib.sha256(response).hexdigest()
    assert result["source_sha256"] == expected
    assert not (Path(adapter.json_cache_dir) / "unreadable.json").exists()


REVIEW_SHA256 = "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750"
REVIEW_ARCHIVE = (ROOT / "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence"
                  / "independent_source/review_report.json")


def _local_evidence_path(path):
    # The untouched review preserves its original Windows source paths.
    path = str(path).replace("\\", "/")
    if os.name == "posix" and path.startswith("C:/"):
        return Path("/mnt/c") / path[3:]
    return Path(path) if Path(path).is_absolute() else ROOT / path


def _review():
    path = _local_evidence_path(os.environ.get("AFLOW_SOURCE_REVIEW", REVIEW_ARCHIVE))
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == REVIEW_SHA256
    report = json.loads(content)
    bindings = report["qa_fingerprint_binding_table"]
    assert len(bindings) == len({r["material_id"] for r in bindings}) == 6
    assert sum(r["is_confirmed_anomaly"] for r in bindings) == 5
    return report


def _reviewed_source(mid):
    """Bind all k/energy cells to the exact table, not the review summary."""
    from src.data.aflow_adapter import source_content_fingerprint
    report = _review()
    index, binding = next((i, b) for i, b in enumerate(report["qa_fingerprint_binding_table"])
                          if b["material_id"] == mid)
    material = next(m for m in report["materials"] if m["material_id"] == mid)
    published = next(s for s in material["sources"] if s["url"] == binding["source_url"])
    wire = _local_evidence_path(published["wire_path"]).read_bytes()
    assert hashlib.sha256(wire).hexdigest() == binding["compressed_sha256"] == published["wire_sha256"]
    decoded = lzma.decompress(wire)
    assert hashlib.sha256(decoded).hexdigest() == published["decoded_sha256"]
    assert decoded == _local_evidence_path(published["decoded_path"]).read_bytes()
    payload = AFLOWAdapter._lenient_json(decoded.decode("utf-8-sig"))
    assert len(material["parsed_cache_checks"]) == 1
    cache = material["parsed_cache_checks"][0]
    cache_bytes = _local_evidence_path(cache["path"]).read_bytes()
    assert hashlib.sha256(cache_bytes).hexdigest() == cache["sha256"]
    record = json.loads(cache_bytes)
    table = np.asarray(payload["bands_data"], dtype="<f4")
    np.testing.assert_array_equal(table[:, 1:].T, record["energies"])
    np.testing.assert_array_equal(table[:, 0], record["k_distances"])
    assert record["material_id"] == mid
    assert record["source_sha256"] == binding["compressed_sha256"]
    assert binding["fingerprint_version"] == "aflow-k-energy-f32-v1"
    assert hashlib.sha256(str(table.shape).encode("ascii") + table.tobytes()).hexdigest() == binding["content_fingerprint"]
    assert source_content_fingerprint(record["energies"], record["k_distances"]) == binding["content_fingerprint"]
    assert material["full_float32_k_energy_fingerprint"] == binding["content_fingerprint"]
    return {"binding": binding, "index": index, "payload": payload, "record": record,
            "cache_bytes": cache_bytes, "wire": wire}


def _assert_review_citation(quality, source):
    binding = source["binding"]
    evidence = quality["quarantine_evidence"]
    assert quality["status"] == "invalid"
    assert quality["reason"] == "source_quarantine_published_response_v1"
    assert quality["content_sha256"] == binding["content_fingerprint"]
    assert evidence["evidence_version"] == "published-response-quarantine-20260909-v2"
    assert evidence["review_status"] == binding["review_verdict"] == "verified_invalid_published_energy"
    assert evidence["material_id_observed"] == binding["material_id"]
    assert evidence["compressed_sha256"] == binding["compressed_sha256"]
    assert evidence["source_url"] == binding["source_url"]
    assert evidence["evidence_json_pointer"] == f"/qa_fingerprint_binding_table/{source['index']}"
    assert evidence["evidence_report_sha256"] == REVIEW_SHA256
    assert evidence["evidence_path"] == REVIEW_ARCHIVE.relative_to(ROOT).as_posix()
    report_bytes = _local_evidence_path(evidence["evidence_path"]).read_bytes()
    assert hashlib.sha256(report_bytes).hexdigest() == REVIEW_SHA256
    assert json.loads(report_bytes)["qa_fingerprint_binding_table"][source["index"]] == binding


def test_existing_quarantine_cites_independent_primary_review():
    from src.data.aflow_adapter import assess_source_quality
    source = _reviewed_source(BAD_ID)
    _assert_review_citation(assess_source_quality(source["record"]), source)


def _copy_reviewed_hdf5(path, sources):
    """Small rejection fixture copied from caches; not a canonical HDF5 read."""
    from src.data.band_store import save_band_record
    assert not path.exists()
    for source in sources:
        record = dict(source["record"])
        for key in ("energies", "k_distances", "kpoints"):
            if key in record:
                record[key] = np.asarray(record[key], dtype=np.float32)
        save_band_record(str(path), record)


def _assert_reviewed_rejection(mid, tmp_path, monkeypatch):
    import h5py
    from scripts import prepare_p3_multiband as prepare
    from src.data.aflow_adapter import SourceQualityError, assess_source_quality
    source = _reviewed_source(mid)
    # Real parser, complete decompressed response, no columns dropped/changed.
    for identity in (mid, "different-identity-same-source-content"):
        with pytest.raises(SourceQualityError, match="source_quarantine") as caught:
            AFLOWAdapter._parse_band_payload(identity, source["payload"])
        _assert_review_citation(caught.value.quality, source)

    def no_download(url):
        pytest.fail("rejected cache must not download or overwrite evidence")

    # Both modes read byte-exact copies of the original cached JSON. This is a
    # cache HIT, not a reconstructed record asserting its own quality.
    for mode in ("formal", "diagnostic"):
        adapter = AFLOWAdapter(str(tmp_path / mode), source_qa_mode=mode)
        cache_path = Path(adapter.json_cache_dir) / (mid + ".json")
        cache_path.write_bytes(source["cache_bytes"])
        monkeypatch.setattr(adapter, "_request_bytes", no_download)
        result = adapter.fetch_band_structure(mid, {"download_url": source["binding"]["source_url"]})
        assert "error" in result
        _assert_review_citation(result["source_quality"], source)
        assert cache_path.read_bytes() == source["cache_bytes"]
        assert sorted(p.name for p in cache_path.parent.iterdir()) == [cache_path.name]

    path = tmp_path / "copied-source.h5"
    _copy_reviewed_hdf5(path, [source])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, "r") as handle:
        stored = prepare.read_hdf5_source_record(handle[mid])
        np.testing.assert_array_equal(stored["energies"], source["record"]["energies"])
        np.testing.assert_array_equal(stored["k_distances"], source["record"]["k_distances"])
        with pytest.raises(prepare.SampleExclusion, match="source_quarantine") as caught:
            prepare.build_one(handle[mid], {}, max_bands=16, n_k=256, delta_e=5.0, max_atoms=50)
        _assert_review_citation(json.loads(caught.value.detail), source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    # Provenance is NOT the execution key. Full-content quarantine still wins
    # against altered identity/SHA/URL and forged valid-declaration metadata.
    other = dict(source["record"], material_id="different-id", source_sha256="0" * 64,
                 source_url="https://offline.invalid/different", source_schema="aflow_bandsdata_v1",
                 band_layout="bands_k", spin_channels=1, source_semantics_evidence="forged-declaration")
    _assert_review_citation(assess_source_quality(other), source)
    normal = AFLOWAdapter._parse_band_payload(mid, tiny_payload())
    normal.update(source_sha256=source["binding"]["compressed_sha256"],
                  source_url=source["binding"]["source_url"])
    assert assess_source_quality(normal)["status"] == "ambiguous"
    assert "quarantine_evidence" not in assess_source_quality(normal)

    # In particular, the last k point (not merely a prefix or energies alone)
    # participates. Changing either last stored cell breaks exact binding, not
    # evidence of complete validity. Only temporary in-memory variants change.
    for key, cell in (("k_distances", -1), ("energies", (-1, -1))):
        variant = dict(source["record"])
        changed = np.array(variant[key], dtype=np.float32)
        changed[cell] = np.nextafter(changed[cell], np.float32(np.inf))
        variant[key] = changed
        quality = assess_source_quality(variant)
        assert quality["content_sha256"] != source["binding"]["content_fingerprint"]
        assert quality["status"] == "ambiguous"
        assert "quarantine_evidence" not in quality


def test_reviewed_phosphorus_response_is_invalid_at_all_three_entries(tmp_path, monkeypatch):
    _assert_reviewed_rejection("aflow-540aee2190a21eec", tmp_path, monkeypatch)


def test_reviewed_bromolybdenum_response_is_invalid_at_all_three_entries(tmp_path, monkeypatch):
    _assert_reviewed_rejection("aflow-7c7ee61f60207c27", tmp_path, monkeypatch)


def test_reviewed_thallium_response_is_invalid_at_all_three_entries(tmp_path, monkeypatch):
    _assert_reviewed_rejection("aflow-8bee7b2755ee75ce", tmp_path, monkeypatch)


def test_reviewed_cadmium_response_is_invalid_at_all_three_entries(tmp_path, monkeypatch):
    _assert_reviewed_rejection("aflow-c3dae23bece6500c", tmp_path, monkeypatch)


# Existing-behavior regressions below; not claimed as additional production TDD.
def test_reviewed_original_anomaly_still_fails_all_three_entries(tmp_path, monkeypatch):
    _assert_reviewed_rejection(BAD_ID, tmp_path, monkeypatch)


def test_reviewed_control_is_not_promoted_to_full_validity(tmp_path, monkeypatch):
    import h5py
    from scripts import prepare_p3_multiband as prepare
    from src.data.aflow_adapter import assess_source_quality
    source = _reviewed_source(GOOD_ID)
    assert not source["binding"]["is_confirmed_anomaly"]
    parsed = AFLOWAdapter._parse_band_payload(GOOD_ID, source["payload"])
    np.testing.assert_array_equal(parsed["energies"], source["record"]["energies"])
    np.testing.assert_array_equal(parsed["k_distances"], source["record"]["k_distances"])
    assert parsed["efermi"] == source["record"]["efermi"] == 0.0
    assert parsed["source_efermi_absolute"] == source["payload"]["Efermi"]
    assert parsed["is_spin_polarized"] is None
    for record in (parsed, source["record"]):
        qa = assess_source_quality(record)
        assert qa["status"] == "ambiguous"
        assert set(qa["reasons"]) == {"unknown_schema", "unknown_layout", "unknown_spin", "missing_semantics_evidence"}
        assert "quarantine_evidence" not in qa
    for mode in ("formal", "diagnostic"):
        adapter = AFLOWAdapter(str(tmp_path / mode), source_qa_mode=mode)
        path = Path(adapter.json_cache_dir) / (GOOD_ID + ".json")
        path.write_bytes(source["cache_bytes"])
        monkeypatch.setattr(adapter, "_request_bytes", lambda url: pytest.fail("no cache re-download"))
        result = adapter.fetch_band_structure(GOOD_ID, {"download_url": source["binding"]["source_url"]})
        assert ("error" in result) is (mode == "formal")
        assert result["source_quality"]["status"] == "ambiguous"
        assert path.read_bytes() == source["cache_bytes"]
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    path = tmp_path / "normal-control.h5"
    _copy_reviewed_hdf5(path, [source])
    before = path.read_bytes()
    with h5py.File(path, "r") as handle:
        with pytest.raises(prepare.SampleExclusion, match="source_ambiguous") as caught:
            prepare.build_one(handle[GOOD_ID], {}, max_bands=16, n_k=256, delta_e=5.0, max_atoms=50)
        assert json.loads(caught.value.detail)["status"] == "ambiguous"
    assert path.read_bytes() == before


def test_six_reviewed_sources_have_closed_prepare_qa_counts(tmp_path, monkeypatch):
    import h5py
    from scripts import prepare_p3_multiband as prepare
    from src.data.aflow_adapter import SOURCE_QUARANTINE
    bindings = _review()["qa_fingerprint_binding_table"]
    assert set(SOURCE_QUARANTINE) == {b["content_fingerprint"] for b in bindings if b["is_confirmed_anomaly"]}
    sources = [_reviewed_source(b["material_id"]) for b in bindings]
    h5, split, out = tmp_path / "six-source-copies.h5", tmp_path / "fixture-split.npz", tmp_path / "qa"
    _copy_reviewed_hdf5(h5, sources)
    # Only an explicit six-record fixture partition, not a canonical split or
    # scientific group membership claim. There are no outer keys or arrays.
    ids = [b["material_id"] for b in bindings]
    np.savez(split, material_ids_train=np.array(ids), groups_train=np.arange(6, dtype=np.int32))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (h5, split)}

    def forbidden(*args, **kwargs):
        pytest.fail("rejected source must stop before selection, interpolation, or graph construction")

    for name in ("select_fermi_bands", "extract_fermi_bands", "path_segments", "build_crystal_graph"):
        monkeypatch.setattr(prepare, name, forbidden)
    with h5py.File(h5, "r") as handle:
        assert set(handle) == set(ids)
        for source in sources:
            mid = source["binding"]["material_id"]
            record = prepare.read_hdf5_source_record(handle[mid])
            np.testing.assert_array_equal(record["energies"], source["record"]["energies"])
            np.testing.assert_array_equal(record["k_distances"], source["record"]["k_distances"])
            with pytest.raises(prepare.SampleExclusion):
                prepare.build_one(handle[mid], {}, max_bands=16, n_k=256, delta_e=5.0, max_atoms=50)
    report = prepare.source_qa_ledger(h5, split, out, split="train")
    assert report["counts"] == {"requested": 6, "valid": 0, "ambiguous": 1, "invalid": 5, "read_error": 0}
    assert [row["material_id"] for row in report["records"]] == ids
    for row, source in zip(report["records"], sources):
        assert row["content_sha256"] == source["binding"]["content_fingerprint"]
        if source["binding"]["is_confirmed_anomaly"]:
            _assert_review_citation(row, source)
        else:
            assert row["status"] == "ambiguous"
    assert report == json.loads((out / "source_qa_train.json").read_text())
    assert sorted(p.name for p in out.iterdir()) == ["source_qa_train.json"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())
