"""Group bootstrap and error stratification (P0D).

Group bootstrap: resample at the GROUP level (spacegroup) so that bootstrap
confidence intervals respect the OOD unit of independence. For each of the
three P0 tasks we produce:
  - point estimate (macro-averaged accuracy / F1 over classes)
  - 95% percentile CI from B group-bootstrap resamples
Error stratification: per-stratum metric tables with keyed strata:
  - provider type (metal / direct / indirect)
  - disagreement label (agree / disagree)
  - spacegroup-number bands (1-74, 75-167, 168-230)
  - num_sites bands (1, 2, 3, 4+)
  - source catalog (from aurl)
"""
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Accuracy CI via group-level bootstrap.

    y_true/y_pred are integer class labels. groups give the unit of
    independence (spacegroup id). Resampling draws groups with replacement
    and recomputes accuracy on the pooled members.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred, groups must share length")
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    point = float(np.mean(y_true == y_pred))
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[g] for g in chosen])
        stats.append(float(np.mean(y_true[idx] == y_pred[idx])))
    stats = np.asarray(stats)
    return {
        "point_estimate": point,
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "bootstrap_mean": float(stats.mean()),
        "bootstrap_std": float(stats.std()),
        "n_boot": n_boot,
        "n_groups": int(len(unique_groups)),
        "n_samples": int(len(y_true)),
    }


def macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Class-macro-averaged accuracy (robust to class imbalance)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    accs = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        accs.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(accs)) if accs else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat.tolist()


def stratify_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stratum_fn: Sequence[Any],
    stratum_name: str = "stratum",
) -> Dict[str, Any]:
    """Per-stratum accuracy table for error stratification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    strata = np.asarray(stratum_fn)
    out = {"stratum": stratum_name, "rows": []}
    for key in sorted(set(strata.tolist()), key=str):
        mask = strata == key
        n = int(mask.sum())
        correct = int(np.sum(y_true[mask] == y_pred[mask]))
        row = {
            "key": str(key),
            "n": n,
            "accuracy": round(correct / n, 6) if n else None,
        }
        if n:
            classes = np.unique(y_true[mask])
            row["class_distribution"] = {
                str(c): int(np.sum(y_true[mask] == c)) for c in classes
            }
        out["rows"].append(row)
    return out


def spacegroup_band(sg: int) -> str:
    sg = int(sg)
    if sg <= 74:
        return "1-74"
    if sg <= 167:
        return "75-167"
    return "168-230"


def num_sites_band(n_sites: Optional[int]) -> str:
    if n_sites is None:
        return "unknown"
    n = int(n_sites)
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def source_catalog(sample: Dict[str, Any]) -> str:
    from src.data.benchmark_splits import _parse_aurl

    catalog, _ = _parse_aurl(sample.get("aurl", ""))
    return catalog


def build_error_stratification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    strata_specs = [
        ("provider_type", lambda s: s.get("gap_type", "unknown")),
        ("spacegroup_band", lambda s: spacegroup_band(s.get("spacegroup_number"))),
        ("num_sites_band", lambda s: num_sites_band(s.get("num_sites"))),
        ("source_catalog", lambda s: source_catalog(s)),
    ]
    out = {"task": task_name}
    for name, fn in strata_specs:
        out[name] = stratify_errors(y_true, y_pred, [fn(s) for s in samples], stratum_name=name)
    return out


def summarize_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray],
    task_name: str,
    n_classes: int,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    summary = {
        "task": task_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes),
    }
    if groups is not None:
        summary["group_bootstrap"] = group_bootstrap_ci(
            y_true, y_pred, groups, n_boot=n_boot, random_state=random_state
        )
    return summary
