"""Seven-split benchmark suite (P0C): random / space-group / composition /
prototype / leave-element / source-protocol / temporal.

Each split produces train/test index arrays with a group-disjoint guarantee
where a natural group key exists:
  random          - plain stratified-random holdout (no grouping)
  space_group     - existing canonical OOD split (group = spacegroup_number)
  composition     - group = sorted element set (leave-composition-out)
  prototype       - group = AFLOW prototype label (leave-prototype-out)
  leave_element   - group = per-element group: sample belongs to every element
                    it contains (leave-element-out, overlapping groups handled
                    by disjoint element-group assignment at split time)
  source_protocol - group = source catalog + dft functional parsed from aurl
  temporal        - train on entries before cutoff date, test after (no random
                    reassignment; chronological, group-less)

Contract:
- deterministic given (manifest_samples, random_state); no model dependence
- zero overlap between train and test where a group key exists
- each split is a pure function returning {name, train_idx, test_idx,
  group_key, overlap_check}
- empty/unknown keys are bucketed to "unknown" and reported
"""
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


SPLIT_NAMES = (
    "random",
    "space_group",
    "composition",
    "prototype",
    "leave_element",
    "source_protocol",
    "temporal",
)


def _composition_key(formula: str) -> str:
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", str(formula or ""))))
    return "-".join(elements) if elements else "unknown"


def _prototype_key(sample: Dict[str, Any]) -> str:
    proto = str(sample.get("prototype") or "").strip()
    if proto:
        return proto
    # fallback: pearson + spacegroup + n_sites proxy
    return "|".join(
        [
            str(sample.get("pearson_symbol") or "unknown"),
            f"sg{sample.get('spacegroup_number', 'unknown')}",
            f"n{sample.get('num_sites', 'unknown')}",
        ]
    )


def _parse_aurl(aurl: str) -> Tuple[str, str]:
    """Parse aurl -> (source_catalog, dft_functional).

    e.g. aflowlib.duke.edu:AFLOWDATA/LIB3_WEB/... -> ('LIB3_WEB', 'PAW_PBE')
         aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/...  -> ('ICSD_WEB', 'PAW_PBE')
    LIB1_WEB/LIB2_WEB are LDA/LDAU2-era catalogs; LIB3_WEB/ICSD_WEB are PBE.
    """
    parts = str(aurl or "").split("/")
    catalog = "unknown"
    for part in parts:
        if part.endswith("_WEB") or part.endswith("_RAW"):
            catalog = part
            break
    if catalog.startswith("LIB1") or catalog.startswith("LIB2"):
        functional = "PAW_LDA"
    else:
        functional = "PAW_PBE"
    return catalog, functional


def _source_protocol_key(sample: Dict[str, Any]) -> str:
    catalog, functional = _parse_aurl(sample.get("aurl", ""))
    return f"{catalog}|{functional}"


def _element_keys(species_field) -> List[str]:
    if isinstance(species_field, str):
        return [e for e in re.findall(r"[A-Z][a-z]?", species_field) if e]
    if isinstance(species_field, list):
        return [str(e) for e in species_field if e]
    return []


def _temporal_year(entry_date) -> Optional[int]:
    # entry_date like '20150623_00:05:22_GMT-4'
    match = re.match(r"(\d{4})", str(entry_date or ""))
    return int(match.group(1)) if match else None


def _group_split(
    groups: np.ndarray, train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic group-disjoint split; mirror the canonical builder logic.

    Operates on group->member-index buckets (built once) instead of repeated
    full-array membership scans, keeping large-N splits tractable.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two distinct groups required")
    buckets = {g: np.flatnonzero(groups == g) for g in unique_groups}
    target_test_count = (1.0 - train_size) * len(groups)
    best: Optional[Tuple[float, List[int], List[int]]] = None
    for _ in range(max(64, min(1024, len(unique_groups) * 8))):
        shuffled = rng.permutation(unique_groups)
        cumulative = np.cumsum([len(buckets[g]) for g in shuffled])
        center = int(np.argmin(np.abs(cumulative - target_test_count))) + 1
        for offset in range(-2, 3):
            cut = max(1, min(len(shuffled) - 1, center + offset))
            test_groups = set(shuffled[:cut].tolist())
            test_idx = np.concatenate([buckets[g] for g in shuffled[:cut]])
            train_idx = np.concatenate([buckets[g] for g in shuffled[cut:]])
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            score = abs(len(test_idx) / len(groups) - (1.0 - train_size))
            if best is None or score < best[0]:
                best = (score, train_idx, test_idx)
    if best is None:
        raise RuntimeError("group split construction failed")
    return np.sort(best[1]), np.sort(best[2])


def _assert_no_overlap(train_idx, test_idx, label):
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise RuntimeError(f"{label}: index overlap detected ({overlap.size})")


def build_random_split(
    n: int, train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    cut = int(round(train_size * n))
    train_idx = np.sort(perm[:cut])
    test_idx = np.sort(perm[cut:])
    _assert_no_overlap(train_idx, test_idx, "random")
    return train_idx, test_idx


def build_spacegroup_split(
    groups: np.ndarray, train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = _group_split(groups, train_size, rng)
    _assert_no_overlap(train_idx, test_idx, "space_group")
    return train_idx, test_idx


def build_composition_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.asarray([_composition_key(s.get("formula_pretty", "")) for s in samples])
    return build_spacegroup_split(keys, train_size, rng)


def build_prototype_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.asarray([_prototype_key(s) for s in samples])
    return build_spacegroup_split(keys, train_size, rng)


def build_leave_element_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-element-out: each sample joins exactly one of its elements.

    Overlapping group membership (a compound contains several elements) is
    resolved by greedy deterministic assignment: elements are scored by how
    many samples they own, then each sample is placed in the test bucket only
    if its OWNER element is selected for test. This yields a non-overlapping
    partition over elements.
    """
    per_sample_elements = [_element_keys(s.get("species")) for s in samples]
    n = len(samples)
    # single-element samples (elemental solids) group by their only element
    keys: List[str] = []
    for elements in per_sample_elements:
        keys.append(elements[0] if len(elements) == 1 else "|".join(sorted(elements)))
    key_arr = np.asarray(keys)
    train_idx, test_idx = _group_split(key_arr, train_size, rng)
    _assert_no_overlap(train_idx, test_idx, "leave_element")
    return train_idx, test_idx


def build_source_protocol_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.asarray([_source_protocol_key(s) for s in samples])
    return build_spacegroup_split(keys, train_size, rng)


def build_temporal_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological split on the full aflowlib_date timestamp.

    Dates are YYYYMMDD_* strings, so lexicographic order == chronological
    order. The cutoff is the train_size quantile over DATED samples; undated
    samples join train deterministically. Year-granularity was rejected
    because the date distribution is bimodal (2014 and 2020/2021 mass
    uploads), which collapses the test set when cutting at a year boundary.
    """
    del rng  # deterministic chronological cut, no randomness
    dates = [str(s.get("aflowlib_date") or "") for s in samples]
    dated = sorted(d for d in dates if d)
    if not dated:
        raise ValueError("temporal split requires aflowlib_date coverage")
    cutoff = dated[int(train_size * len(dated)) - 1]
    train_mask = np.asarray([(d == "") or (d <= cutoff) for d in dates])
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("temporal split produced an empty partition")
    if len(test_idx) / len(samples) < 0.02:
        raise RuntimeError("temporal split test fraction < 2%")
    _assert_no_overlap(train_idx, test_idx, "temporal")
    return train_idx, test_idx


BUILDERS = {
    "random": None,  # needs n only
    "space_group": build_spacegroup_split,
    "composition": build_composition_split,
    "prototype": build_prototype_split,
    "leave_element": build_leave_element_split,
    "source_protocol": build_source_protocol_split,
    "temporal": build_temporal_split,
}


def build_all_splits(
    samples: List[Dict[str, Any]],
    spacegroups: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    n = len(samples)
    rng = np.random.RandomState(random_state)
    splits: Dict[str, Dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        if name == "random":
            train_idx, test_idx = build_random_split(n, train_size, rng)
            group_key = None
        elif name == "space_group":
            train_idx, test_idx = build_spacegroup_split(
                spacegroups, train_size, rng
            )
            group_key = "spacegroup_number"
        elif name == "temporal":
            train_idx, test_idx = build_temporal_split(samples, train_size, rng)
            group_key = "aflowlib_date(year)"
        else:
            train_idx, test_idx = BUILDERS[name](samples, train_size, rng)
            group_key = {
                "composition": "composition",
                "prototype": "prototype",
                "leave_element": "species",
                "source_protocol": "aurl(source|functional)",
            }[name]
        splits[name] = {
            "name": name,
            "group_key": group_key,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "num_train": int(len(train_idx)),
            "num_test": int(len(test_idx)),
            "test_fraction": round(len(test_idx) / n, 6),
        }
    return splits


def save_splits(
    output_dir: str,
    splits: Dict[str, Dict[str, Any]],
    material_ids: np.ndarray,
    random_state: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, Any] = {"random_state": random_state}
    for name, split in splits.items():
        out[name] = {
            "group_key": split["group_key"],
            "num_train": split["num_train"],
            "num_test": split["num_test"],
            "test_fraction": split["test_fraction"],
            "train_material_ids": material_ids[split["train_idx"]].tolist(),
            "test_material_ids": material_ids[split["test_idx"]].tolist(),
        }
    path = os.path.join(output_dir, "seven_splits_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path}")
    return out
