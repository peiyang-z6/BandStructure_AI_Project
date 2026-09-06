"""Regression tests for the P1 sidecar coverage audit contract.

Constitution 5.0 §8 P1: the sidecar material_id set must equal the HDF5
group set (coverage audit), and per-field missing counts must be reported.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.structure_sidecar import sidecar_record_from_aflow_fields


def _records():
    return [
        sidecar_record_from_aflow_fields({
            "auid": "aflow:aaa", "geometry": "[4,4,4,90,90,90]",
            "species": ["Si"], "positions_fractional": [[0, 0, 0]],
        }),
        sidecar_record_from_aflow_fields({
            "auid": "aflow:bbb", "geometry": "[5,5,5,90,90,90]",
            "species": ["Ge"], "positions_fractional": [[0.25, 0.25, 0.25]],
        }),
    ]


def test_sidecar_material_id_maps_to_aflow_prefix():
    rec = _records()[0]
    assert rec["material_id"] == "aflow-aaa"
    # AFLOW auid "aflow:aaa" must become "aflow-aaa" (hyphen), matching HDF5.
    assert rec["material_id"].startswith("aflow-")


def test_every_record_has_material_id_and_structure_hash():
    for rec in _records():
        assert rec["material_id"]
        assert rec["structure_sha256"] and len(rec["structure_sha256"]) == 64


def test_missing_fields_breakdown_counts_per_field():
    # A record with only geometry must report every absent field.
    rec = sidecar_record_from_aflow_fields({"auid": "aflow:zzz", "geometry": "[4,4,4,90,90,90]"})
    assert "species" in rec["missing_fields"]
    assert "fractional_coordinates" in rec["missing_fields"]
    assert "dft_functional" in rec["missing_fields"]
    assert "spin_cell" in rec["missing_fields"]
    assert "pseudopotential" in rec["missing_fields"]
    assert "kpath_segments" in rec["missing_fields"]
