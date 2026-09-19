"""Versioned P0 split construction, never a frozen-model eligibility claim.

Canonical space-group IDs are reused exactly. Composition includes reduced
stoichiometry; leave-element holds out every occurrence of a selected element.
Reported source/catalog/functional keys are not full DFT-protocol provenance.
Unknown species/protocols/dates are explicitly excluded; prototype proxies are
identified in scope. Temporal means UTC catalog-entry chronology, not blind DFT.
The separate evaluation gate must recheck every test against actual fit AND
selection IDs/groups; a new split's nominal train pool cannot certify that.
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
    from pymatgen.core import Composition

    if not formula:
        return "unknown"
    try:
        composition = Composition(str(formula), strict=True).reduced_composition
        return json.dumps(composition.get_el_amt_dict(), sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        return "unknown"


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
    """Extract catalog only; a directory cannot certify a DFT functional."""
    parts = str(aurl or "").split("/")
    catalog = "unknown"
    for part in parts:
        if part.endswith("_WEB") or part.endswith("_RAW"):
            catalog = part
            break
    return catalog, "unknown"


def _source_protocol_key(sample: Dict[str, Any]) -> str:
    catalog = sample.get("source_catalog") or _parse_aurl(sample.get("aurl", ""))[0]
    functional = sample.get("dft_functional") or sample.get("dft_type") or "unknown"
    if isinstance(functional, list):
        functional = ",".join(sorted(str(x) for x in functional)) or "unknown"
    source = sample.get("source") or "unknown"
    return f"{source}|{catalog}|{functional}"


def _element_keys(species_field) -> List[str]:
    from pymatgen.core import Element

    if isinstance(species_field, str):
        if re.sub(r"[A-Z][a-z]?|[,;\s]+", "", species_field):
            return []
        species_field = re.findall(r"[A-Z][a-z]?", species_field)
    if not isinstance(species_field, (list, tuple)) or not species_field:
        return []
    if any(not isinstance(e, str) or not Element.is_valid_symbol(e) for e in species_field):
        return []
    return sorted(set(species_field))


def _temporal_year(entry_date) -> Optional[int]:
    # entry_date like '20150623_00:05:22_GMT-4'
    match = re.match(r"(\d{4})", str(entry_date or ""))
    return int(match.group(1)) if match else None


def _temporal_timestamp(entry_date):
    """AFLOW catalog timestamp in UTC; malformed/unknown offsets stay unknown."""
    from datetime import datetime, timedelta, timezone

    match = re.fullmatch(r"(\d{8}_\d{2}:\d{2}:\d{2})_GMT([+-]\d{1,2})(?::(\d{2}))?", str(entry_date or ""))
    if not match:
        return None
    try:
        hours = int(match[2])
        minutes = int(match[3] or 0) * (-1 if match[2].startswith("-") else 1)
        offset = timezone(timedelta(hours=hours, minutes=minutes))
        return datetime.strptime(match[1], "%Y%m%d_%H:%M:%S").replace(tzinfo=offset).astimezone(timezone.utc)
    except ValueError:
        return None


def _group_split(
    groups: np.ndarray, train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic group balancing for NON-canonical benchmark axes.

    Operates on group->member-index buckets (built once) instead of repeated
    full-array membership scans, keeping large-N splits tractable.
    """
    unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique_groups) < 2:
        raise ValueError("at least two distinct groups required")
    target_test_count = (1.0 - train_size) * len(groups)
    best_score = float("inf")
    best_test_groups = None
    for _ in range(max(64, min(1024, len(unique_groups) * 8))):
        shuffled = rng.permutation(len(unique_groups))
        cumulative = np.cumsum(counts[shuffled])
        center = int(np.argmin(np.abs(cumulative - target_test_count))) + 1
        for offset in range(-2, 3):
            cut = max(1, min(len(shuffled) - 1, center + offset))
            score = abs(int(cumulative[cut - 1]) / len(groups) - (1.0 - train_size))
            if score < best_score:
                best_score = score
                best_test_groups = shuffled[:cut].copy()
    if best_test_groups is None:
        raise RuntimeError("group split construction failed")
    is_test = np.isin(inverse, best_test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


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
    """Hold out one element and ALL compounds containing it.

    Choose the element closest to the requested test fraction; seeded ordering
    breaks ties. Unknown species are excluded, never silently assigned to fit.
    Every test row therefore contains an element absent from every train row.
    """
    per_sample = [set(_element_keys(s.get("species"))) for s in samples]
    known = np.asarray([i for i, elements in enumerate(per_sample) if elements], dtype=int)
    elements = sorted(set().union(*per_sample))
    candidates = []
    for element in rng.permutation(elements):
        test = np.asarray([i for i in known if element in per_sample[i]], dtype=int)
        train = np.asarray([i for i in known if element not in per_sample[i]], dtype=int)
        if len(train) and len(test):
            candidates.append((abs(len(test) / len(known) - (1 - train_size)), train, test))
    if not candidates:
        raise ValueError("leave_element requires a nonempty genuinely unseen-element partition")
    _, train_idx, test_idx = min(candidates, key=lambda item: item[0])
    return train_idx, test_idx


def build_source_protocol_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.asarray([_source_protocol_key(s) for s in samples])
    known = np.asarray([i for i, key in enumerate(keys)
                        if key.split("|")[0] != "unknown" and key.split("|")[-1] != "unknown"], dtype=int)
    train, test = build_spacegroup_split(keys[known], train_size, rng)
    return known[train], known[test]


def build_temporal_split(
    samples: List[Dict[str, Any]], train_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological split on the full aflowlib_date timestamp.

    The cutoff uses only dated samples. Undated samples are excluded from
    both partitions and must be reported separately. This is catalog-entry
    chronology, NOT a new-calculation blind test.
    """
    del rng  # deterministic chronological cut, no randomness
    dates = [_temporal_timestamp(s.get("aflowlib_date")) for s in samples]
    dated = sorted(d for d in dates if d)
    if not dated:
        raise ValueError("temporal split requires aflowlib_date coverage")
    cutoff = dated[int(train_size * len(dated)) - 1]
    train_idx = np.asarray([i for i, d in enumerate(dates) if d and d <= cutoff], dtype=int)
    test_idx = np.asarray([i for i, d in enumerate(dates) if d and d > cutoff], dtype=int)
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
    canonical_split: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    n = len(samples)
    rng = np.random.RandomState(random_state)
    splits: Dict[str, Dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        if name == "random":
            train_idx, test_idx = build_random_split(n, train_size, rng)
            group_key = None
        elif name == "space_group":
            if canonical_split is None:
                raise ValueError("canonical_split with exact train/test IDs is required")
            positions = {str(s["material_id"]): i for i, s in enumerate(samples)}
            train_ids = canonical_split.get("train_material_ids", [])
            test_ids = canonical_split.get("test_material_ids", [])
            all_ids = train_ids + test_ids
            if (not train_ids or not test_ids or len(set(all_ids)) != len(all_ids)
                    or len(positions) != n or set(all_ids) != set(positions)):
                raise ValueError("canonical IDs must uniquely partition all samples")
            train_idx, test_idx = (
                np.asarray([positions[mid] for mid in canonical_split[f"{part}_material_ids"]], dtype=int)
                for part in ("train", "test")
            )
            if len(spacegroups) != n or set(spacegroups[train_idx]) & set(spacegroups[test_idx]):
                raise ValueError("canonical spacegroup evidence has overlap or wrong length")
            group_key = "spacegroup_number"
        else:
            group_key = {
                "composition": "reduced_stoichiometric_composition",
                "prototype": "prototype_or_explicit_proxy",
                "leave_element": "held_out_element",
                "source_protocol": "source|catalog|reported_functional",
                "temporal": "aflowlib_date(timestamp)",
            }[name]
            try:
                train_idx, test_idx = BUILDERS[name](samples, train_size, rng)
            except (ValueError, RuntimeError) as exc:
                splits[name] = {"reason": str(exc)}
                train_idx = test_idx = np.asarray([], dtype=int)
        excluded_idx = np.setdiff1d(np.arange(n), np.concatenate([train_idx, test_idx]))
        scope = {"claim": group_key or "sample_random_holdout_not_group_ood"}
        if name == "prototype":
            scope["proxy_count"] = sum(not s.get("prototype") for s in samples)
        elif name == "temporal":
            scope["claim"] = "catalog_entry_time_not_blind_dft"
            scope["undated_policy"] = "exclude"
        elif name == "source_protocol":
            used = np.concatenate([train_idx, test_idx])
            scope["source_count"] = len({samples[i].get("source") for i in used})
            scope["claim"] = "reported_source_catalog_functional_key_disjoint_not_full_protocol"
            scope["unknown_policy"] = "exclude_unknown_source_or_functional"
        elif name == "leave_element":
            train_elements = set().union(*[set(_element_keys(samples[i].get("species"))) for i in train_idx])
            test_elements = set().union(*[set(_element_keys(samples[i].get("species"))) for i in test_idx])
            scope["held_out_elements"] = sorted(test_elements - train_elements)
            scope["unknown_policy"] = "exclude"
        splits[name] = {
            **splits.get(name, {}),
            "status": "available" if len(train_idx) and len(test_idx) else "unavailable",
            "scope": scope,
            "excluded_idx": excluded_idx,
            "num_excluded": int(len(excluded_idx)),
            "name": name,
            "group_key": group_key,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "num_train": int(len(train_idx)),
            "num_test": int(len(test_idx)),
            "test_fraction": round(len(test_idx) / n, 6),
        }
    splits["space_group"]["canonical_reused"] = True
    fields = ("material_id", "formula_pretty", "prototype", "pearson_symbol", "num_sites", "species",
              "source", "source_catalog", "dft_functional", "dft_type", "aurl", "aflowlib_date")
    splits["space_group"]["identity_metadata"] = {
        s["material_id"]: {**{key: s.get(key) for key in fields}, "spacegroup_number": int(spacegroups[i])}
        for i, s in enumerate(samples)
    }
    return splits


def save_splits(
    output_dir: str,
    splits: Dict[str, Dict[str, Any]],
    material_ids: np.ndarray,
    random_state: int,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, Any] = {"schema_version": 2, "random_state": random_state,
                           "provenance": provenance or {}}
    for name, split in splits.items():
        out[name] = {
            **{key: value for key, value in split.items() if not key.endswith("_idx")},
            "excluded_material_ids": material_ids[split["excluded_idx"]].tolist(),
            "train_material_ids": material_ids[split["train_idx"]].tolist(),
            "test_material_ids": material_ids[split["test_idx"]].tolist(),
        }
    path = os.path.join(output_dir, "seven_splits_manifest.v2.json")
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path}")
    return out
