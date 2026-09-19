"""P1 v2 contracts; synthetic cases never certify external structures."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.data import structure_sidecar as sc


def structure():
    return {
        "material_id": "aflow-0000000000000001",
        "source": "aflow", "aurl": "https://example.invalid/material/1",
        "lattice": [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
        "species": ["Na", "Cl"], "species_per_atom": ["Na", "Cl"],
        "fractional_coordinates": [[0., 0., 0.], [.5, .5, .5]],
        "structure_conventions": {
            "lattice_units": "angstrom", "lattice_vectors": "rows",
            "coordinates": "fractional", "coordinate_basis": "direct_lattice",
            "cell_setting": "source_cell",
        },
    }


@pytest.mark.parametrize("field,value", [
    ("species_per_atom", ["Cl", "Na"]),
    ("lattice", [[4.000000001, 0, 0], [0, 4, 0], [0, 0, 4]]),
    ("fractional_coordinates", [[0, 0, 0], [.500000001, .5, .5]]),
    ("structure_conventions", {"lattice_units": "bohr", "lattice_vectors": "rows",
      "coordinates": "fractional", "coordinate_basis": "direct_lattice", "cell_setting": "source_cell"}),
])
def test_v2_hash_binds_complete_atom_and_basis_semantics(field, value):
    original = structure()
    changed = {**original, field: value}
    assert sc.compute_structure_sha256(original) != sc.compute_structure_sha256(changed)

@pytest.mark.parametrize("field,value", [
    ("lattice", [[0, 0, 0]] * 3),
    ("lattice", [[float("inf"), 0, 0], [0, 4, 0], [0, 0, 4]]),
    ("lattice", [[-4, 0, 0], [0, 4, 0], [0, 0, 4]]),
    ("lattice", [[1, 0], [0, 1]]),
    ("fractional_coordinates", []),
    ("fractional_coordinates", [[0, 0, 0]]),
    ("species_per_atom", ["Na"]),
    ("species_per_atom", ["Na", "NotAnElement"]),
    ("species", ["Na", "O"]),
    ("structure_conventions", {}),
])
def test_v2_hash_rejects_invalid_structure(field, value):
    with pytest.raises(ValueError):
        sc.compute_structure_sha256({**structure(), field: value})

@pytest.mark.parametrize("geometry", [
    [4, 4, 4, 90, 90, 0], [4, 4, 4, 90, 90, 180],
    [4, 4, 4, 10, 10, 170], [-4, 4, 4, 90, 90, 90],
    [float("nan"), 4, 4, 90, 90, 90], [4, 4, 4, 90, 90, float("inf")],
])
def test_geometry_refuses_impossible_cell_without_clamping(geometry):
    with pytest.raises(ValueError):
        sc.geometry_to_lattice_vectors(*geometry)
    assert sc.build_lattice_from_geometry(geometry) is None

def test_legacy_validation_is_explicit_and_never_certifies_atom_assignment():
    assert hasattr(sc, "verify_structure_sha256"), "explicit versioned hash verification API missing"
    rec = structure()
    legacy = sc.compute_structure_sha256(rec, version="legacy-v1")
    rec["structure_sha256"] = legacy
    assert sc.verify_structure_sha256(rec, version="legacy-v1") is True
    assert sc.verify_structure_sha256(rec, version="structure-v2") is False
    rec["species_per_atom"] = ["Cl", "Na"]
    assert sc.verify_structure_sha256(rec, version="legacy-v1") is True
    rec["structure_sha256"] = "not-a-sha"
    assert sc.verify_structure_sha256(rec, version="legacy-v1") is False

def test_mapping_preserves_unknown_protocol_and_source_without_zero_imputation():
    unknown = sc.sidecar_record_from_aflow_fields({
        "auid": "aflow:0000000000000001", "source": "aflow_mirror",
        "spin_cell": "0", "spin_atom": 0,
        "dft_type": ["PAW_PBE", "additional_protocol"],
    })
    assert unknown["source"] == "aflow_mirror"
    assert unknown["spin_cell"] == 0
    assert unknown["dft_type"] == ["PAW_PBE", "additional_protocol"]
    for field in ("magnetic_moments", "spin_polarized", "soc", "hubbard_u",
                  "pseudopotential_version", "kpath_convention", "kpoints_3d"):
        assert unknown[field] is None
        assert field in unknown["missing_fields"]
    known = sc.sidecar_record_from_aflow_fields({
        "auid": "aflow:0000000000000001", "soc": False,
        "spin_polarized": True, "hubbard_u": {"enabled": False},
        "species_pp_version": ["PAW_PBE Si 05Jan2001"],
        "protocol_evidence": {"source": "local-INCAR", "sha256": "a" * 64},
    })
    assert known["soc"] is False
    assert known["spin_polarized"] is True
    assert known["hubbard_u"] == {"enabled": False}
    assert known["pseudopotential_version"] == ["PAW_PBE Si 05Jan2001"]
    assert known["protocol_evidence"]["source"] == "local-INCAR"

def test_field_quality_recomputes_missing_and_keeps_basic_structure_distinct():
    assert hasattr(sc, "audit_structure_record"), "field quality audit API missing"
    rec = structure()
    rec.update({"spin_cell": 0, "spin_atom": 0, "missing_fields": [],
                "dft_functional": "", "pseudopotential": [],
                "reciprocal_lattice": sc.reciprocal_lattice_from_vectors(np.asarray(rec["lattice"])),
                "structure_sha256": sc.compute_structure_sha256(rec),
                "structure_hash_version": "structure-v2"})
    out = sc.audit_structure_record(rec)
    assert out["status"] == "ambiguous"
    assert out["basic_structure_status"] == "valid"
    assert out["field_quality"]["spin_cell"]["status"] == "valid"
    assert out["field_quality"]["spin_atom"]["status"] == "ambiguous"
    assert out["field_quality"]["structure_sha256"]["status"] == "valid"
    for field in ("soc", "hubbard_u", "spin_polarized", "kpoints_3d", "dft_functional", "pseudopotential"):
        assert field in out["missing_fields"]
        assert out["field_quality"][field]["status"] == "ambiguous"
    assert rec["missing_fields"] == []  # audit does not rewrite input
    broken = sc.audit_structure_record({**rec, "lattice": [[0, 0, 0]] * 3})
    assert broken["status"] == "invalid"
    assert broken["basic_structure_status"] == "invalid"
    assert broken["field_quality"]["lattice"]["status"] == "invalid"

@pytest.mark.parametrize("field,value,status", [
    ("reciprocal_lattice", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "invalid"),
    ("soc", "false", "invalid"), ("spin_polarized", 0, "invalid"),
    ("hubbard_u", 0, "ambiguous"),
    ("hubbard_u", {"enabled": True, "values_eV": [float("nan")]}, "invalid"),
    ("pseudopotential", [None], "invalid"),
    ("pseudopotential_version", [""], "invalid"),
    ("dft_functional", ["PBE"], "invalid"),
    ("structure_basis_evidence", {"verified": True}, "ambiguous"),
    ("protocol_evidence", {"verified": True}, "ambiguous"),
    ("kpoints_3d", {"segments": []}, "invalid"),
    ("kpoints_dense", {"verification_status": "verified"}, "ambiguous"),
])
def test_field_quality_checks_values_and_does_not_trust_present_or_verified_flag(field, value, status):
    out = sc.audit_structure_record({**structure(), field: value})
    assert out["field_quality"][field]["status"] == status

def test_quality_ledger_conserves_requested_ids_including_missing_duplicates_extras_and_read_errors():
    from scripts.build_structure_sidecar import coverage_report
    one = structure()
    duplicate = {**structure(), "material_id": "aflow-0000000000000002"}
    errored = {"material_id": "aflow-0000000000000003", "read_error": "unreadable JSON row"}
    extra = {**structure(), "material_id": "aflow-0000000000000005"}
    ids = {f"aflow-{i:016x}" for i in range(1, 5)}
    out = coverage_report([one, duplicate, duplicate, errored, extra], ids)
    assert out["requested"] == 4
    assert sum(out[s] for s in sc.QUALITY_STATES) == out["requested"]
    assert out["invalid"] == 1
    assert out["read_error"] == 2
    assert out["ambiguous"] == 1
    assert out["id_integrity"]["duplicate_ids"] == [duplicate["material_id"]]
    assert out["id_integrity"]["extra_ids"] == [extra["material_id"]]
    assert out["id_integrity"]["missing_ids"] == ["aflow-0000000000000004"]
    assert out["id_integrity"]["valid"] is False
    assert {r["material_id"] for r in out["records"]} == ids
    assert out["field_present"]["soc"] == 0
    assert out["missing_field_breakdown"]["soc"] == 4
    for counts in out["field_quality"].values():
        assert sum(counts[s] for s in sc.QUALITY_STATES) == 4

@pytest.mark.parametrize("text", [
    "G X\n20\nAutomatic\nreciprocal\n0 0 0\n.5 0 0\n",
    "G X\n0\nLine-mode\nreciprocal\n0 0 0\n.5 0 0\n",
    "G X\n20\nLine-mode\nunknown\n0 0 0\n.5 0 0\n",
    "G X\n20\nLine-mode\nreciprocal\n0 0 0\nnan 0 0\n",
    "G X\n20\nLine-mode\nreciprocal\n0 0 0\n.5 0 0\n0 0 0\n",
    "G X\n20\nLine-mode\nreciprocal\n0 0 0\n0 0 0\n",
])
def test_kpoints_parser_rejects_malformed_line_mode_instead_of_partial_success(text):
    from scripts.build_structure_sidecar import parse_kpoints_bands
    assert "error" in parse_kpoints_bands(text)

def test_dense_path_constructs_segment_safe_grid_but_requires_basis_proof(tmp_path):
    assert hasattr(sc, "build_dense_kpath"), "dense-k validation API missing"
    from scripts.build_structure_sidecar import parse_kpoints_bands
    text = "G-X X-M\n3\nLine-mode\nreciprocal\n0 0 0\n.5 0 0\n.5 0 0\n.5 .5 0\n"
    bits = parse_kpoints_bands(text)
    out = sc.build_dense_kpath(bits, structure())
    assert out["construction_status"] == "valid"
    assert out["segment_count"] == 2 and out["point_count"] == 6
    assert out["segment_ids"] == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(out["fractional_points"], [[0,0,0], [.25,0,0], [.5,0,0], [.5,0,0], [.5,.25,0], [.5,.5,0]])
    assert out["verification_status"] == "unverified"
    assert out["basis_status"] == "unverified"
    assert out["alignment_status"] == "unverified"
    assert out["segment_lengths"] is None and out["cartesian_points"] is None
    # A present but nonexistent evidence artifact is not a proof.
    fake = {"poscar_path": str(tmp_path / "missing"), "poscar_sha256": "a"*64,
            "kpoints_path": str(tmp_path / "missing2"), "kpoints_sha256": "b"*64}
    assert sc.build_dense_kpath(bits, structure(), basis_evidence=fake)["basis_status"] == "unverified"

def path_evidence(tmp_path):
    import hashlib
    poscar = "NaCl synthetic\n1.0\n4 0 0\n0 4 0\n0 0 4\nNa Cl\n1 1\nDirect\n0 0 0\n.5 .5 .5\n"
    text = "G-X X-M\n3\nLine-mode\nreciprocal\n0 0 0\n.5 0 0\n.5 0 0\n.5 .5 0\n"
    pp, kp = tmp_path / "POSCAR.bands", tmp_path / "KPOINTS.bands"
    pp.write_text(poscar)
    kp.write_text(text)
    evidence = {"poscar_path": str(pp), "poscar_sha256": hashlib.sha256(pp.read_bytes()).hexdigest(),
                "kpoints_path": str(kp), "kpoints_sha256": hashlib.sha256(kp.read_bytes()).hexdigest(),
                "relationship": "same_calculation_cell", "source": "synthetic-test-calculation"}
    return sc.parse_kpoints_bands(text), evidence


def test_dense_path_proves_basis_from_existing_digest_bound_poscar_and_kpoints(tmp_path):
    bits, evidence = path_evidence(tmp_path)
    out = sc.build_dense_kpath(bits, structure(), basis_evidence=evidence)
    assert out["basis_status"] == "verified", out["reasons"]
    assert out["alignment_status"] == "unverified"
    assert out["verification_status"] == "unverified"
    np.testing.assert_allclose(out["segment_lengths"], [np.pi/4, np.pi/4])
    np.testing.assert_allclose(out["distance"], [0, np.pi/8, np.pi/4, np.pi/4, 3*np.pi/8, np.pi/2])
    assert out["length_units"] == "angstrom^-1 (2pi)"
    tampered = {**evidence, "poscar_sha256": "0"*64}
    assert sc.build_dense_kpath(bits, structure(), basis_evidence=tampered)["basis_status"] == "unverified"
    mismatch = {**structure(), "species_per_atom": ["Cl", "Na"]}
    assert sc.build_dense_kpath(bits, mismatch, basis_evidence=evidence)["basis_status"] == "unverified"
    changed_bits = deepcopy(bits)
    changed_bits["segments"][0]["k_to"][0] = .25
    assert sc.build_dense_kpath(changed_bits, structure(), basis_evidence=evidence)["basis_status"] == "unverified"

def test_dense_alignment_checks_every_point_segment_and_source_artifact(tmp_path):
    import hashlib
    bits, evidence = path_evidence(tmp_path)
    band = {"material_id": structure()["material_id"], "coordinate_mode": "reciprocal",
            "poscar_sha256": evidence["poscar_sha256"], "source": "synthetic independent dense dump",
            "fractional_points": [[0,0,0],[.25,0,0],[.5,0,0],[.5,0,0],[.5,.25,0],[.5,.5,0]],
            "segment_ids": [0,0,0,1,1,1]}
    path = tmp_path / "band_kpoints.json"
    def audit(payload):
        path.write_text(json.dumps(payload))
        band_proof = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return sc.build_dense_kpath(bits, structure(), basis_evidence=evidence, band_evidence=band_proof)
    out = audit(band)
    assert out["verification_status"] == "verified"
    assert out["alignment_status"] == "verified"
    for changed in (
        {**band, "fractional_points": band["fractional_points"][:-1]},
        {**band, "segment_ids": [0,0,1,1,1,1]},
        {**band, "fractional_points": [[.1,0,0]] + band["fractional_points"][1:]},
        {**band, "poscar_sha256": "f"*64},
    ):
        assert audit(changed)["verification_status"] != "verified"

def test_enrichment_versions_hash_without_mutating_legacy_or_guessing_basis():
    assert hasattr(sc, "enrich_structure_record"), "versioned enrichment API missing"
    rec = structure()
    conventions = rec.pop("structure_conventions")
    rec["structure_sha256"] = sc.compute_structure_sha256(rec, version="legacy-v1")
    rec["missing_fields"] = []
    before = deepcopy(rec)
    out = sc.enrich_structure_record(rec)
    assert rec == before
    assert out["schema_version"] == "p1-sidecar-v2"
    assert out["legacy_structure_sha256"] == rec["structure_sha256"]
    assert out["legacy_hash_verified"] is True
    assert out["structure_sha256"] is None
    assert "structure_conventions" in out["missing_fields"]
    assert "soc" in out["missing_fields"]
    explicit = sc.enrich_structure_record(rec, conventions=conventions)
    assert explicit["structure_hash_version"] == "structure-v2"
    assert sc.verify_structure_sha256(explicit, version="structure-v2")
    assert explicit["source"] == before["source"]
    assert explicit["quality"]["status"] == "ambiguous"  # semantics hash is not external proof

@pytest.mark.parametrize("species,n,expected,origin", [
    (["Si"], 2, ["Si", "Si"], "single_species"),
    (["Na", "Cl"], 2, ["Na", "Cl"], "legacy_one_of_each_order_unverified"),
    (["Na", "Cl"], 3, None, "ambiguous"),
])
def test_species_resolution_only_uses_legal_cardinality_fallback(species, n, expected, origin):
    rec = structure()
    rec.pop("species_per_atom")
    rec["species"] = species
    rec["fractional_coordinates"] = [[0, 0, 0]] * n
    out = sc.enrich_structure_record(rec)
    assert out["species_per_atom"] == expected
    assert out["species_assignment_source"] == origin
    if expected is None:
        assert out["structure_sha256"] is None
    else:
        assert sc.verify_structure_sha256(out, version="structure-v2")
    assert "species_per_atom" not in rec

def test_enrichment_cannot_launder_corrupt_input_hash_or_endpoint_read_error():
    rec = structure()
    rec["structure_sha256"] = "0"*64
    rec["structure_hash_version"] = "structure-v2"
    out = sc.enrich_structure_record(rec)
    assert out["quality"]["status"] == "invalid"
    assert out["input_hash_verification"]["valid"] is False
    failed = sc.enrich_structure_record({**structure(), "field_read_errors": {"soc": "HTTP 500"}})
    assert failed["quality"]["field_quality"]["soc"]["status"] == "read_error"
    assert failed["quality"]["status"] == "read_error"

def test_new_mapping_emits_v2_semantics_and_reports_invalid_geometry_without_crashing():
    fields = {"auid": "aflow:0000000000000001", "geometry": [4,4,4,90,90,90],
              "species": ["Si"], "positions_fractional": [[0,0,0],[.5,.5,.5]]}
    out = sc.sidecar_record_from_aflow_fields(fields)
    assert out["schema_version"] == "p1-sidecar-v2"
    assert out["species_per_atom"] == ["Si", "Si"]
    assert sc.verify_structure_sha256(out, version="structure-v2")
    assert out["structure_conventions"]["lattice_units"] == "angstrom"
    bad = sc.sidecar_record_from_aflow_fields({**fields, "geometry": [4,4,4,10,10,170],
                                              "positions_fractional": [["bad",0,0]]})
    assert bad["quality"]["status"] == "invalid"
    assert bad["structure_sha256"] is None
    assert bad["raw_fields"]["geometry"] == [4,4,4,10,10,170]

@pytest.mark.parametrize("auid", ["aflow:abc", "aflow:0000000000000001 ", " aflow:0000000000000001", "AFLOW:0000000000000001", "0000000000000001"])
def test_auid_format_validated_before_mapping_without_normalization(auid):
    with pytest.raises(ValueError, match="AUID"):
        sc.sidecar_record_from_aflow_fields({"auid": auid})

def test_bulk_build_clips_to_fixed_ids_and_rejects_duplicate_rows(tmp_path):
    import h5py
    from scripts.build_structure_sidecar import build_records
    path = tmp_path / "ids.h5"
    with h5py.File(path, "w") as f:
        f.create_group("aflow-0000000000000001")
    row = {"auid": "aflow:0000000000000001"}
    records, report = build_records(str(path), [row, {"auid": "aflow:0000000000000002"}], False, 1)
    assert [r["material_id"] for r in records] == ["aflow-0000000000000001"]
    assert report["clipped_incoming_ids"] == ["aflow-0000000000000002"]
    with pytest.raises(ValueError, match="duplicate"):
        build_records(str(path), [row, deepcopy(row)], False, 1)

def test_offline_audit_and_enrich_clis_write_new_version_and_protect_existing_outputs(tmp_path):
    import hashlib
    import subprocess
    import sys
    legacy = structure()
    conventions = legacy.pop("structure_conventions")
    legacy["structure_sha256"] = sc.compute_structure_sha256(legacy, version="legacy-v1")
    old = tmp_path / "legacy.json"
    old.write_text(json.dumps([legacy]))
    ids, conv = tmp_path / "ids.json", tmp_path / "conventions.json"
    ids.write_text(json.dumps([legacy["material_id"]]))
    conv.write_text(json.dumps(conventions))
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    base = [sys.executable, "scripts/build_structure_sidecar.py", "--sidecar", str(old), "--ids-json", str(ids)]
    ledger = tmp_path / "audit.v2.json"
    result = subprocess.run(base + ["--mode", "audit", "--out", str(ledger)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(ledger.read_text())
    assert report["schema_version"] == "p1-quality-ledger-v2" and report["requested"] == 1
    assert report["field_present"]["soc"] == 0
    new = tmp_path / "sidecar.v2.json"
    cmd = base + ["--mode", "enrich", "--conventions-json", str(conv), "--out", str(new)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    enriched = json.loads(new.read_text())
    assert sc.verify_structure_sha256(enriched[0], version="structure-v2")
    report = json.loads(new.with_name("sidecar.v2_report.json").read_text())
    assert "field_quality" in report and report["id_integrity"]["valid"]
    new_hash = hashlib.sha256(new.read_bytes()).hexdigest()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert hashlib.sha256(new.read_bytes()).hexdigest() == new_hash
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before

def test_fill_merge_is_fixed_id_nonempty_preserving_and_conflicts_are_visible():
    assert hasattr(sc, "merge_sidecar_records"), "fixed-ID merge API missing"
    original = {**structure(), "dft_functional": "PBE", "soc": False}
    before = deepcopy(original)
    ids = [original["material_id"], "aflow-0000000000000002"]
    incoming = [
        {"material_id": ids[0], "dft_functional": "LDA", "soc": None, "spin_polarized": True},
        {"auid": "aflow:0000000000000002", "species": ["Si"], "source": "aflow_actual_mirror"},
        {"auid": "aflow:0000000000000003"},
    ]
    merged, report = sc.merge_sidecar_records([original], incoming, ids)
    assert original == before
    assert [r["material_id"] for r in merged] == ids
    assert merged[0]["dft_functional"] == "PBE" and merged[0]["soc"] is False
    assert merged[0]["spin_polarized"] is True
    assert merged[0]["merge_conflicts"][0]["field"] == "dft_functional"
    assert report["invalid"] == 1
    assert report["clipped_incoming_ids"] == ["aflow-0000000000000003"]
    assert merged[1]["source"] == "aflow_actual_mirror"
    assert report["gap_fill_added"] == 1 and "field_quality" in report
    with pytest.raises(ValueError, match="duplicate"):
        sc.merge_sidecar_records([original, deepcopy(original)], [], ids)
    with pytest.raises(ValueError, match="duplicate"):
        sc.merge_sidecar_records([original], [incoming[1], deepcopy(incoming[1])], ids)

def test_fill_cli_requires_new_output_and_keeps_full_coverage_ledger(tmp_path):
    import subprocess
    import sys
    original, incoming, ids = tmp_path / "old.json", tmp_path / "incoming.json", tmp_path / "ids.json"
    original.write_text(json.dumps([structure()]))
    incoming.write_text(json.dumps([{"auid": "aflow:0000000000000002", "species": ["Si"]},
                                    {"auid": "aflow:0000000000000003"}]))
    ids.write_text(json.dumps([structure()["material_id"], "aflow-0000000000000002"]))
    before = original.read_bytes()
    out = tmp_path / "filled.v2.json"
    base = [sys.executable, "scripts/fill_sidecar_gaps.py", "--sidecar", str(original),
            "--incoming-json", str(incoming), "--ids-json", str(ids)]
    result = subprocess.run(base + ["--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert original.read_bytes() == before
    report = json.loads(out.with_name("filled.v2_report.json").read_text())
    assert report["requested"] == 2 and "field_quality" in report
    assert report["gap_fill_added"] == 1 and report["clipped_incoming_ids"] == ["aflow-0000000000000003"]
    assert {r["material_id"] for r in json.loads(out.read_text())} == set(json.loads(ids.read_text()))
    assert subprocess.run(base, capture_output=True).returncode != 0
    assert subprocess.run(base + ["--out", str(original)], capture_output=True).returncode != 0
    assert original.read_bytes() == before

def test_fetch_failures_are_field_read_errors_with_actual_source_url(monkeypatch):
    from scripts import fill_sidecar_gaps as fill
    def fail(*args, **kwargs):
        raise OSError("controlled unavailable source")
    monkeypatch.setattr(fill.urllib.request, "urlopen", fail)
    rec = fill.fetch_one("aflow-0000000000000001", "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/test", False, retries=1)
    assert rec["quality"]["status"] == "read_error"
    assert rec["quality"]["field_quality"]["lattice"]["status"] == "read_error"
    assert "controlled unavailable source" in rec["field_read_errors"]["lattice"]
    assert rec["field_provenance"]["lattice"]["url"].endswith("/?geometry")
    assert rec["aurl"] == "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/test"

def complete_evidenced_record(tmp_path):
    import hashlib
    rec = structure()
    bits, basis = path_evidence(tmp_path)
    rec.update({"spin_cell": 0., "spin_atom": [0., 0.], "magnetic_moments": [0., 0.],
                "spin_polarized": False, "soc": False, "hubbard_u": {"enabled": False},
                "dft_functional": "PBE", "dft_type": ["PBE"], "pseudopotential": ["Na", "Cl"],
                "pseudopotential_version": ["test-v1", "test-v1"], "code": "vasp.test",
                "kpath_segments": ["G-X", "X-M"], "kpath_convention": "VASP_line_mode",
                "reciprocal_lattice": sc.reciprocal_lattice_from_vectors(np.asarray(rec["lattice"])).tolist(),
                "reciprocal_convention": "rows_2pi_inverse_transpose",
                "structure_basis_evidence": basis, "kpoint_basis_evidence": basis, "kpoints_3d": bits})
    protocol_fields = ("spin_cell", "spin_atom", "magnetic_moments", "spin_polarized", "soc", "hubbard_u",
                       "dft_functional", "dft_type", "pseudopotential", "pseudopotential_version", "code")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"material_id": rec["material_id"], "source": rec["source"],
                                         "fields": {f: rec[f] for f in protocol_fields}}))
    rec["protocol_evidence"] = {"path": str(protocol_path), "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    band_path = tmp_path / "band.json"
    band_path.write_text(json.dumps({"material_id": rec["material_id"], "coordinate_mode": "reciprocal",
                                     "poscar_sha256": basis["poscar_sha256"], "source": "synthetic-test",
                                     "fractional_points": [[0,0,0],[.25,0,0],[.5,0,0],[.5,0,0],[.5,.25,0],[.5,.5,0]],
                                     "segment_ids": [0,0,0,1,1,1]}))
    rec["band_kpoint_evidence"] = {"path": str(band_path), "sha256": hashlib.sha256(band_path.read_bytes()).hexdigest()}
    rec["kpoints_dense"] = sc.build_dense_kpath(bits, rec, basis_evidence=basis, band_evidence=rec["band_kpoint_evidence"])
    rec["structure_sha256"] = sc.compute_structure_sha256(rec)
    rec["structure_hash_version"] = "structure-v2"
    return rec


def test_full_quality_gate_reverifies_sources_and_dense_payload_not_claim_flags(tmp_path):
    rec = complete_evidenced_record(tmp_path)
    out = sc.audit_structure_record(rec)
    assert out["status"] == "valid", out["field_quality"]
    report = sc.build_quality_ledger([rec], [rec["material_id"]])
    assert report["valid"] == 1 and report["full_contract_passed"]
    tampered = deepcopy(rec)
    tampered["kpoints_dense"]["fractional_points"][1][0] = .3
    assert sc.audit_structure_record(tampered)["field_quality"]["kpoints_dense"]["status"] == "invalid"
    mismatched = deepcopy(rec)
    mismatched["kpath_segments"] = ["G-X"]
    assert sc.audit_structure_record(mismatched)["field_quality"]["kpath_segments"]["status"] == "invalid"
    from pathlib import Path
    Path(rec["protocol_evidence"]["path"]).unlink()
    assert sc.audit_structure_record(rec)["field_quality"]["protocol_evidence"]["status"] == "ambiguous"

def test_enrich_builds_dense_only_as_reverified_derivative(tmp_path):
    rec = complete_evidenced_record(tmp_path)
    rec.pop("kpoints_dense")
    out = sc.enrich_structure_record(rec)
    assert out["kpoints_dense"]["verification_status"] == "verified"
    assert out["quality"]["status"] == "valid"
    absent = deepcopy(rec)
    absent.pop("kpoint_basis_evidence")
    unknown = sc.enrich_structure_record(absent)
    assert unknown["kpoints_dense"]["verification_status"] == "unverified"
    assert unknown["kpoints_dense"]["segment_lengths"] is None

def test_enrich_cli_rejects_duplicate_input_instead_of_publishing_it(tmp_path):
    import subprocess
    import sys
    old, ids, out = tmp_path / "old.json", tmp_path / "ids.json", tmp_path / "new.json"
    old.write_text(json.dumps([structure(), structure()]))
    ids.write_text(json.dumps([structure()["material_id"]]))
    result = subprocess.run([sys.executable, "scripts/build_structure_sidecar.py", "--mode", "enrich",
                             "--sidecar", str(old), "--ids-json", str(ids), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert not out.exists()

def test_json_duplicate_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('[{"material_id":"aflow-0000000000000001","material_id":"aflow-0000000000000002"}]')
    with pytest.raises(ValueError, match="duplicate JSON"):
        sc.load_sidecar_records(path)

def test_audit_cli_materializes_read_error_ledger_for_unreadable_sidecar(tmp_path):
    import subprocess
    import sys
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([structure()["material_id"], "aflow-0000000000000002"]))
    out = tmp_path / "read_error.v2.json"
    result = subprocess.run([sys.executable, "scripts/build_structure_sidecar.py", "--mode", "audit",
                             "--sidecar", str(tmp_path / "absent.json"), "--ids-json", str(ids), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    ledger = json.loads(out.read_text())
    assert ledger["requested"] == ledger["read_error"] == 2
    assert ledger["input_read_error"]
    assert ledger["full_contract_passed"] is False

def test_audit_derives_no_physical_reciprocal_from_nonfinite_or_left_handed_lattice():
    for lattice in ([[float("nan"),0,0],[0,1,0],[0,0,1]], [[-1,0,0],[0,1,0],[0,0,1]]):
        with pytest.raises(ValueError):
            sc.reciprocal_lattice_from_vectors(np.asarray(lattice))

def test_dense_path_refuses_allocation_above_explicit_bounded_limit():
    bits = sc.parse_kpoints_bands("G-X\n100\nLine-mode\nreciprocal\n0 0 0\n.5 0 0\n")
    out = sc.build_dense_kpath(bits, structure(), max_points=10)
    assert out["construction_status"] == "invalid"
    assert out["fractional_points"] is None
    assert "limit" in " ".join(out["reasons"])

def test_record_and_ledger_read_error_outcomes_are_identical():
    rec = {"material_id": structure()["material_id"], "read_error": "corrupt source record"}
    direct = sc.audit_structure_record(rec)
    ledger = sc.build_quality_ledger([rec], [rec["material_id"]])
    assert direct["status"] == "read_error"
    assert direct["field_quality"] == ledger["records"][0]["field_quality"]

def test_atomic_writer_failure_and_competing_processes_never_replace_output(tmp_path):
    import subprocess
    import sys
    target = tmp_path / "atomic.json"
    with pytest.raises(ValueError):
        sc.atomic_write_json(target, {"value": float("nan")})
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    code = """import sys
from src.data.structure_sidecar import atomic_write_json
try:
    atomic_write_json(sys.argv[1], {'writer':sys.argv[2]})
except FileExistsError:
    sys.exit(17)
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(target), str(i)]) for i in range(2)]
    assert sorted(p.wait(timeout=20) for p in processes) == [0, 17]
    assert json.loads(target.read_text())["writer"] in ("0", "1")
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        sc.atomic_write_json(target, {"replacement": True})
    assert target.read_bytes() == before


def test_hdf5_id_loading_reads_only_root_keys(monkeypatch, tmp_path):
    import h5py
    path = tmp_path / "keys_only.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group(structure()["material_id"])
        group.create_dataset("energies", data=[[1., 2.]])
    def forbid(*args, **kwargs):
        raise AssertionError("HDF5 dataset read forbidden in P1 ledger")
    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbid)
    assert sc.load_fixed_ids(h5_path=path) == [structure()["material_id"]]

def test_bulk_kpoints_failure_is_not_silently_lost_from_field_ledger(monkeypatch, tmp_path):
    import h5py
    from scripts import build_structure_sidecar as build
    path = tmp_path / "ids.h5"
    with h5py.File(path, "w") as f:
        f.create_group(structure()["material_id"])
    monkeypatch.setattr(build, "fetch_kpoints_bits", lambda *a: {"error": "controlled KPOINTS read failure"})
    row = {"auid": "aflow:0000000000000001", "aurl": "aflowlib.duke.edu:AFLOWDATA/test"}
    records, report = build.build_records(str(path), [row], True, 1)
    assert report["field_quality"]["kpoints_3d"]["read_error"] == 1
    assert records[0]["quality"]["field_quality"]["kpoints_3d"]["status"] == "read_error"
    assert "kpoints_3d" in records[0]["missing_fields"]

def test_malformed_local_protocol_evidence_stays_unverified_without_aborting_ledger(tmp_path):
    import hashlib
    rec = complete_evidenced_record(tmp_path)
    from pathlib import Path
    path = Path(rec["protocol_evidence"]["path"])
    path.write_text('["not a protocol object"]')
    rec["protocol_evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = sc.build_quality_ledger([rec], [rec["material_id"]])
    assert ledger["requested"] == ledger["ambiguous"] == 1
    assert ledger["records"][0]["field_quality"]["protocol_evidence"]["status"] == "ambiguous"

def test_malformed_band_evidence_cannot_crash_or_pass_dense_gate(tmp_path):
    import hashlib
    rec = complete_evidenced_record(tmp_path)
    from pathlib import Path
    path = Path(rec["band_kpoint_evidence"]["path"])
    path.write_text('[]')
    rec["band_kpoint_evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = sc.build_quality_ledger([rec], [rec["material_id"]])
    assert ledger["requested"] == ledger["invalid"] == 1
    assert ledger["records"][0]["field_quality"]["kpoints_dense"]["status"] == "invalid"

def test_invalid_nonfinite_protocol_still_has_strict_json_audit_ledger(tmp_path):
    import subprocess
    import sys
    bad = {**structure(), "hubbard_u": {"enabled": True, "values_eV": [float("nan")]}}
    old, ids, out = tmp_path / "nonfinite.json", tmp_path / "ids.json", tmp_path / "ledger.json"
    old.write_text(json.dumps([bad]))
    ids.write_text(json.dumps([bad["material_id"]]))
    result = subprocess.run([sys.executable, "scripts/build_structure_sidecar.py", "--mode", "audit",
                             "--sidecar", str(old), "--ids-json", str(ids), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    def reject_constant(value):
        raise ValueError("nonstandard JSON: " + value)
    ledger = json.loads(out.read_text(), parse_constant=reject_constant)
    assert ledger["invalid"] == ledger["requested"] == 1
    assert ledger["records"][0]["protocol"]["hubbard_u"]["values_eV"] == [{"__p1_nonfinite_float__": "NaN"}]

def test_v2_serializer_preserves_invalid_nonfinite_as_tagged_source_data(tmp_path):
    rec = sc.enrich_structure_record({**structure(), "fractional_coordinates": [[float("nan"), 0, 0], [.5,.5,.5]]})
    encoded = sc.StructureSidecarSchema().serialize(rec)
    text = json.dumps(encoded, allow_nan=False)
    assert json.loads(text)["fractional_coordinates"][0][0] == {"__p1_nonfinite_float__": "NaN"}
    assert encoded["quality"]["status"] == "invalid"

@pytest.mark.parametrize("script,mode", [("build_structure_sidecar.py", ["--mode", "enrich"]), ("fill_sidecar_gaps.py", [])])
def test_new_version_clis_keep_invalid_nonfinite_records_without_dropping_ids(tmp_path, script, mode):
    import subprocess
    import sys
    bad = {**structure(), "hubbard_u": {"enabled": True, "values_eV": [float("nan")]}}
    old, ids, out = tmp_path / "bad.json", tmp_path / "ids.json", tmp_path / "new.json"
    old.write_text(json.dumps([bad]))
    ids.write_text(json.dumps([bad["material_id"]]))
    result = subprocess.run([sys.executable, "scripts/" + script, *mode, "--sidecar", str(old),
                             "--ids-json", str(ids), "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = json.loads(out.read_text())
    assert rows[0]["material_id"] == bad["material_id"]
    assert rows[0]["hubbard_u"]["values_eV"] == [{"__p1_nonfinite_float__": "NaN"}]
    assert json.loads(out.with_name("new_report.json").read_text())["invalid"] == 1

@pytest.mark.parametrize("field,value", [
    ("auid", "aflow:0000000000000002"),
    ("reciprocal_convention", "no_2pi"),
    ("pseudopotential", ["Na"]), ("pseudopotential_version", ["test-v1"]),
    ("spin_cell", False), ("spin_atom", [False, False]),
])
def test_field_gate_rejects_identity_convention_and_protocol_cardinality_conflicts(field, value):
    record = {**structure(), field: value}
    key = "material_id" if field == "auid" else field
    assert sc.audit_structure_record(record)["field_quality"][key]["status"] == "invalid"

def test_fill_can_resolve_read_error_placeholder_without_erasing_prior_error():
    original = {"material_id": structure()["material_id"], "read_error": "previous source unavailable"}
    merged, ledger = sc.merge_sidecar_records([original], [structure()], [original["material_id"]])
    assert ledger["read_error"] == 0
    assert merged[0]["read_error_history"] == ["previous source unavailable"]
    assert original["read_error"] == "previous source unavailable"

def test_ledger_exposes_nonempty_merge_conflict_reason_not_just_invalid_count():
    record = {**structure(), "merge_conflicts": [{"field": "dft_functional", "existing": "PBE", "incoming": "LDA"}]}
    ledger = sc.build_quality_ledger([record], [record["material_id"]])
    assert ledger["records"][0]["merge_conflicts"] == record["merge_conflicts"]
    assert ledger["invalid"] == 1

def test_basis_gate_never_accepts_dummy_species_inferred_from_vasp4_counts(tmp_path):
    import hashlib
    from pathlib import Path
    bits, evidence = path_evidence(tmp_path)
    poscar = Path(evidence["poscar_path"])
    # Counts-only VASP4 must not be silently assigned pymatgen's H/He defaults.
    poscar.write_text("counts-only synthetic\n1\n4 0 0\n0 4 0\n0 0 4\n1 1\nDirect\n0 0 0\n.5 .5 .5\n")
    evidence["poscar_sha256"] = hashlib.sha256(poscar.read_bytes()).hexdigest()
    rec = {**structure(), "species": ["H", "He"], "species_per_atom": ["H", "He"]}
    import warnings
    with warnings.catch_warnings(record=True):
        result = sc.build_dense_kpath(bits, rec, basis_evidence=evidence)
    assert result["basis_status"] == "unverified"
    assert result["segment_lengths"] is None
