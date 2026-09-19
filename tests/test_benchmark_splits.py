"""Regression tests for the seven-split benchmark suite (P0C)."""
import numpy as np
import pytest

from src.data.benchmark_splits import (
    build_all_splits,
    build_leave_element_split,
    build_temporal_split,
    _composition_key,
    _parse_aurl,
    _prototype_key,
    _source_protocol_key,
    _temporal_year,
)


def _samples(n=120):
    import random

    random.seed(7)
    sgs = list(range(10, 34))
    el_sets = [["Na", "Cl"], ["Si"], ["Ga", "As"], ["Fe"], ["Ti", "O"], ["C"]]
    out = []
    for i in range(n):
        el = el_sets[i % len(el_sets)]
        out.append(
            {
                "material_id": f"aflow-test-{i}",
                "spacegroup_number": sgs[i % len(sgs)],
                "formula_pretty": "".join(f"{e}{random.randint(1,3)}" for e in el),
                "prototype": f"P{i % 7}",
                "source": "AFLOW",
                "dft_functional": "PBE",
                "species": el,
                "aurl": (
                    f"aflowlib.duke.edu:AFLOWDATA/LIB3_WEB/X/entry{i}"
                    if i % 3
                    else f"aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/FCC/entry{i}"
                ),
                "aflowlib_date": f"{2010 + i % 12}0623_00:05:22_GMT-4",
            }
        )
    return out


def _canonical(samples):
    return {f"{part}_material_ids": [s["material_id"] for s in samples
                                    if (s["spacegroup_number"] < 28) == (part == "train")]
            for part in ("train", "test")}


def test_parse_aurl_catalog_and_functional():
    assert _parse_aurl("aflowlib.duke.edu:AFLOWDATA/LIB3_WEB/X/e") == ("LIB3_WEB", "unknown")
    assert _parse_aurl("aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/FCC/e") == ("ICSD_WEB", "unknown")
    assert _parse_aurl("aflowlib.duke.edu:AFLOWDATA/LIB1_WEB/X/e") == ("LIB1_WEB", "unknown")
    assert _parse_aurl("") == ("unknown", "unknown")


def test_composition_key_reduced_formula():
    assert _composition_key("Cl1Na1") == _composition_key("Na1Cl1")
    assert _composition_key("Na2Cl2") == _composition_key("Na1Cl1")
    assert _composition_key("") == "unknown"


def test_prototype_key_prefers_backfilled_prototype():
    s = {"prototype": "ABC2", "pearson_symbol": "cF8", "spacegroup_number": 225, "num_sites": 4}
    assert _prototype_key(s) == "ABC2"
    assert _prototype_key({}) == "|".join(["unknown", "sgunknown", "nunknown"])


def test_source_protocol_key():
    s = {"aurl": "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/FCC/O2_ICSD_173933"}
    assert _source_protocol_key(s) == "unknown|ICSD_WEB|unknown"


def test_temporal_year_parse():
    assert _temporal_year("20150623_00:05:22_GMT-4") == 2015
    assert _temporal_year(None) is None
    assert _temporal_year("") is None


def test_all_seven_splits_exist_and_are_disjoint():
    samples = _samples(120)
    sgs = np.asarray([s["spacegroup_number"] for s in samples], dtype=np.int32)
    splits = build_all_splits(samples, sgs, train_size=0.8, random_state=42, canonical_split=_canonical(samples))
    assert set(splits.keys()) == {
        "random",
        "space_group",
        "composition",
        "prototype",
        "leave_element",
        "source_protocol",
        "temporal",
    }
    for name, split in splits.items():
        overlap = np.intersect1d(split["train_idx"], split["test_idx"])
        assert overlap.size == 0, f"{name} has overlap"
        assert split["num_train"] > 0 and split["num_test"] > 0
        assert split["num_train"] + split["num_test"] == 120


def test_spacegroup_split_has_no_group_overlap():
    samples = _samples(200)
    sgs = np.asarray([s["spacegroup_number"] for s in samples], dtype=np.int32)
    splits = build_all_splits(samples, sgs, train_size=0.8, random_state=42, canonical_split=_canonical(samples))
    split = splits["space_group"]
    train_sgs = set(sgs[split["train_idx"]].tolist())
    test_sgs = set(sgs[split["test_idx"]].tolist())
    assert not train_sgs.intersection(test_sgs)


def test_composition_split_has_no_composition_overlap():
    samples = _samples(200)
    sgs = np.asarray([s["spacegroup_number"] for s in samples], dtype=np.int32)
    splits = build_all_splits(samples, sgs, train_size=0.8, random_state=42, canonical_split=_canonical(samples))
    split = splits["composition"]
    def keys(idx):
        return {_composition_key(samples[i]["formula_pretty"]) for i in idx}
    assert not keys(split["train_idx"]).intersection(keys(split["test_idx"]))


def test_temporal_split_is_chronological():
    samples = _samples(200)
    sgs = np.asarray([s["spacegroup_number"] for s in samples], dtype=np.int32)
    split = build_temporal_split(samples, 0.8, np.random.RandomState(42))
    train_dates = [str(samples[i]["aflowlib_date"]) for i in split[0]]
    test_dates = [str(samples[i]["aflowlib_date"]) for i in split[1]]
    # every test date is strictly newer than every dated train date
    train_dated = [d for d in train_dates if d]
    assert max(train_dated) <= min(test_dates)


def test_temporal_split_excludes_undated_samples():
    samples = _samples(60)
    samples[5]["aflowlib_date"] = ""
    samples[9]["aflowlib_date"] = None
    split = build_temporal_split(samples, 0.8, np.random.RandomState(42))
    for i in (5, 9):
        assert i not in set(split[0]) | set(split[1]), f"undated sample {i} must be excluded"
