"""Regression tests for AFLOW metadata field backfill (P0B).

Contract:
- material_id mapping: 'aflow-' + auid (lowercase hex as served).
- Non-empty precedence: an incoming field must NOT overwrite an existing
  non-empty value (constitution §4 — empty/missing must not downgrade).
- Only whitelisted fields are merged; everything else in the record is
  untouched.
- Coverage accounting: per-field counts before/after, missing IDs reported.
"""
import copy

import pytest

from scripts.backfill_aflow_metadata_fields import (
    BACKFILL_FIELDS,
    merge_backfill_rows,
    row_to_backfill,
)


def test_row_to_backfill_maps_auid_and_keeps_only_whitelisted_fields():
    raw = {
        "auid": "9c5b1a44b0f8a485",
        "prototype": "ABC2",
        "species": "Nb,Sc,Tc",
        "species_pp": "Nb_pv,Sc_sv,Tc_pv",
        "year": 1994,
        "experiment": "LDAU2",
        "Egap": 1.23,
        "compound": "NbScTc2",
        "aurl": "aflowlib.duke.edu:AFLOWDATA/LIB3_WEB/...",
    }
    out = row_to_backfill(raw)
    assert out["material_id"] == "aflow-9c5b1a44b0f8a485"
    assert set(out.keys()) <= {"material_id"} | set(BACKFILL_FIELDS)
    assert out["prototype"] == "ABC2"
    assert "Egap" not in out and "compound" not in out


def test_row_to_backfill_accepts_prefixed_auid_as_served_by_aflux():
    raw = {"auid": "aflow:000d6c8ff8e9b2b5", "prototype": "ABC2", "year": 1994}
    out = row_to_backfill(raw)
    assert out["material_id"] == "aflow-000d6c8ff8e9b2b5"


def test_row_to_backfill_keeps_first_entry_of_aflowlib_date_list():
    raw = {
        "auid": "aflow:000d6c8ff8e9b2b5",
        "aflowlib_date": ["20150623_00:05:22_GMT-4", "20150624_17:56:11_GMT-4"],
    }
    out = row_to_backfill(raw)
    assert out["aflowlib_date"] == "20150623_00:05:22_GMT-4"


def test_row_to_backfill_skips_rows_without_auid():
    assert row_to_backfill({"prototype": "A2B"}) is None
    assert row_to_backfill({}) is None


def test_api_root_includes_query_separator():
    # Regression: a bare '/aflux/' URL (no '?') makes AFLOW return 404 on
    # every request — cost three failed background runs.
    import urllib.parse

    from scripts.backfill_aflow_metadata_fields import API_ROOT

    assert API_ROOT.endswith("aflux/?")
    aflux = "Egap(0*,*5),natoms(1*,*50),prototype(),paging(1,10)"
    url = API_ROOT + urllib.parse.quote(aflux, safe="(),*")
    assert "aflux/?" in url
    assert "?" in url and url.count("?") == 1


def test_merge_does_not_downgrade_non_empty_values():
    existing = [
        {"material_id": "aflow-0001", "prototype": "ABC", "year": 2001},
        {"material_id": "aflow-0002", "prototype": None, "species": "Fe,O"},
    ]
    incoming = [
        {"material_id": "aflow-0001", "prototype": "XYZ", "year": 1999},
        {"material_id": "aflow-0002", "prototype": "ABC", "species": ""},
        {"material_id": "aflow-0003", "prototype": "A2BC", "species": "Na,Cl"},
    ]
    updated, stats = merge_backfill_rows(existing, incoming)
    by_id = {m["material_id"]: m for m in updated}
    # non-empty existing value wins
    assert by_id["aflow-0001"]["prototype"] == "ABC"
    assert by_id["aflow-0001"]["year"] == 2001
    # empty existing value is filled
    assert by_id["aflow-0002"]["prototype"] == "ABC"
    assert by_id["aflow-0002"]["species"] == "Fe,O"  # incoming empty does not downgrade
    # new id appended
    assert by_id["aflow-0003"]["prototype"] == "A2BC"
    assert stats["rows_incoming"] == 3
    assert stats["new_ids"] == 1
    assert stats["fields_filled"] >= 2


def test_merge_is_idempotent():
    existing = [{"material_id": "aflow-0001", "prototype": "ABC"}]
    incoming = [{"material_id": "aflow-0001", "prototype": "ABC"}]
    updated, stats = merge_backfill_rows(existing, incoming)
    assert len(updated) == 1
    assert stats["fields_filled"] == 0
