"""Regression tests for the P1 structure sidecar builder.

Contract (constitution 5.0 §8 P1):
- Sidecar is a READ-ONLY pairing keyed by material_id; the immutable HDF5
  (aflow_bands.h5) is never modified.
- Per material, the sidecar stores at least: lattice 3x3, species,
  fractional_coordinates, magnetic/spin, DFT functional/U/pseudopotential,
  reciprocal lattice, 3D fractional k-points, k-path convention, source and
  structure SHA-256.
- Lattice 3x3 is constructed from the AFLOW `geometry` six-vector
  [a, b, c, alpha, beta, gamma] (degrees) using the standard crystallographic
  convention.
- Sidecar material_id set must equal the HDF5 group set (coverage audit).
- Non-empty precedence / fallback marking mirrors constitution §4.
"""
import hashlib
import json

import numpy as np
import pytest

from src.data.structure_sidecar import (
    StructureSidecarSchema,
    build_lattice_from_geometry,
    compute_structure_sha256,
    geometry_to_lattice_vectors,
    sidecar_record_from_aflow_fields,
)


def test_geometry_to_lattice_vectors_cubic():
    a, b, c = 4.0, 4.0, 4.0
    alpha = beta = gamma = 90.0
    lat = geometry_to_lattice_vectors(a, b, c, alpha, beta, gamma)
    assert lat.shape == (3, 3)
    assert np.allclose(lat[0], [4.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(lat[1], [0.0, 4.0, 0.0], atol=1e-6)
    assert np.allclose(lat[2], [0.0, 0.0, 4.0], atol=1e-6)


def test_geometry_to_lattice_vectors_monoclinic():
    # monoclinic: alpha=80.32 (angle b-c), beta=gamma=90 (matches mP12 example)
    lat = geometry_to_lattice_vectors(5.281256, 5.220942, 5.400221, 80.31969, 90.0, 90.0)
    assert lat.shape == (3, 3)
    assert np.isfinite(lat).all()
    # |a| = a, |b| = b, |c| = c
    assert np.isclose(np.linalg.norm(lat[0]), 5.281256, atol=1e-5)
    assert np.isclose(np.linalg.norm(lat[1]), 5.220942, atol=1e-5)
    assert np.isclose(np.linalg.norm(lat[2]), 5.400221, atol=1e-5)
    # alpha = angle(b,c) = 80.32, gamma = angle(a,b) = 90
    assert np.isclose(
        np.degrees(np.arccos(np.dot(lat[1], lat[2]) / (5.220942 * 5.400221))),
        80.31969, atol=1e-3,
    )
    assert np.isclose(
        np.degrees(np.arccos(np.dot(lat[0], lat[1]) / (5.281256 * 5.220942))),
        90.0, atol=1e-3,
    )


def test_build_lattice_from_geometry_accepts_aflow_list():
    # AFLOW `?geometry` returns a bare list [a,b,c,alpha,beta,gamma]
    lat = build_lattice_from_geometry([4.0, 4.0, 4.0, 90.0, 90.0, 90.0])
    assert lat.shape == (3, 3)


def test_compute_structure_sha256_deterministic():
    spec = {"lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]], "species": ["O", "Zr"],
            "fractional_coordinates": [[0, 0, 0], [0.5, 0.5, 0.5]]}
    h1 = compute_structure_sha256(spec)
    h2 = compute_structure_sha256(spec)
    assert h1 == h2
    assert len(h1) == 64
    assert h1 != compute_structure_sha256({**spec, "fractional_coordinates": [[0, 0, 0], [0.5, 0.5, 0.4]]})


def test_sidecar_record_from_aflow_fields_full():
    fields = {
        "auid": "aflow:000d6c8ff8e9b2b5",
        "geometry": "[4.0,4.0,4.0,90,90,90]",
        "species": ["Al", "Hf", "Zr"],
        "positions_fractional": [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        "dft_type": ["PAW_PBE"],
        "spin_cell": 0,
        "spin_atom": [0, 0, 0],
        "species_pp": ["Al", "Hf_pv", "Zr_sv"],
        "kpoints_bands_path": ["\\Gamma-X", "X-M"],
        "kpoints": [[9, 9, 9], [11, 11, 9], ["\\Gamma-X", "X-M"], 20],
        "code": "vasp.4.6.35",
    }
    rec = sidecar_record_from_aflow_fields(fields)
    assert rec["material_id"] == "aflow-000d6c8ff8e9b2b5"
    assert rec["lattice"].shape == (3, 3)
    assert rec["species"] == ["Al", "Hf", "Zr"]
    assert len(rec["fractional_coordinates"]) == 3
    assert rec["dft_functional"] == "PAW_PBE"
    assert rec["spin_cell"] == 0
    assert rec["kpath_segments"] == ["\\Gamma-X", "X-M"]
    assert rec["structure_sha256"] and len(rec["structure_sha256"]) == 64


def test_sidecar_record_missing_optional_fields_marked():
    rec = sidecar_record_from_aflow_fields({"auid": "aflow:x", "geometry": "[4,4,4,90,90,90]"})
    assert rec["dft_functional"] is None
    assert rec["spin_cell"] is None
    assert rec["missing_fields"]  # fallback/missing must be marked
    assert "dft_functional" in rec["missing_fields"]


def test_schema_roundtrip_and_coverage():
    rec = sidecar_record_from_aflow_fields({
        "auid": "aflow:abc", "geometry": "[4,4,4,90,90,90]",
        "species": ["Si"], "positions_fractional": [[0, 0, 0]],
    })
    schema = StructureSidecarSchema()
    payload = schema.serialize(rec)
    parsed = schema.deserialize(payload)
    assert parsed["material_id"] == "aflow-abc"
    assert parsed["lattice"].shape == (3, 3)
    assert parsed["species"] == ["Si"]
