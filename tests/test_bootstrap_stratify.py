"""Regression tests for group bootstrap + error stratification (P0D)."""
import numpy as np

from src.evaluation.bootstrap_stratify import (
    build_error_stratification,
    group_bootstrap_ci,
    macro_accuracy,
    spacegroup_band,
    stratify_errors,
    summarize_task,
)


def test_group_bootstrap_ci_basic():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(10), 10)
    y_true = (rng.uniform(size=len(groups)) > 0.5).astype(int)
    y_pred = y_true.copy()
    y_pred[3] = 1 - y_pred[3]  # one error
    ci = group_bootstrap_ci(y_true, y_pred, groups, n_boot=200, random_state=1)
    assert 0.0 <= ci["ci_low"] <= ci["point_estimate"] <= ci["ci_high"] <= 1.0
    assert ci["n_groups"] == 10 and ci["n_samples"] == 100


def test_group_bootstrap_ci_requires_matching_lengths():
    with __import__("pytest").raises(ValueError):
        group_bootstrap_ci(np.zeros(3), np.zeros(4), np.zeros(3))


def test_macro_accuracy_is_class_balanced():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    assert macro_accuracy(y_true, y_pred) == 1.0
    y_pred = np.array([0, 0, 0, 0])
    assert macro_accuracy(y_true, y_pred) == 0.5  # class 0 perfect, class 1 zero


def test_stratify_errors_table():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    strata = ["a", "a", "b", "b", "b"]
    out = stratify_errors(y_true, y_pred, strata, stratum_name="test")
    by_key = {row["key"]: row for row in out["rows"]}
    assert by_key["a"]["accuracy"] == 0.5
    assert by_key["b"]["accuracy"] == 1.0
    assert out["stratum"] == "test"


def test_spacegroup_bands():
    assert spacegroup_band(1) == "1-74"
    assert spacegroup_band(74) == "1-74"
    assert spacegroup_band(75) == "75-167"
    assert spacegroup_band(167) == "75-167"
    assert spacegroup_band(168) == "168-230"
    assert spacegroup_band(230) == "168-230"


def test_build_error_stratification_runs_on_toy_samples():
    samples = [
        {"gap_type": 0, "spacegroup_number": 12, "num_sites": 1, "aurl": "x:AFLOWDATA/LIB3_WEB/a"},
        {"gap_type": 1, "spacegroup_number": 100, "num_sites": 2, "aurl": "x:AFLOWDATA/ICSD_WEB/a"},
        {"gap_type": 2, "spacegroup_number": 200, "num_sites": 4, "aurl": "x:AFLOWDATA/LIB3_WEB/a"},
    ]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    out = build_error_stratification(y_true, y_pred, "line_mode_topology", samples)
    for key in ("provider_type", "spacegroup_band", "num_sites_band", "source_catalog"):
        assert key in out and out[key]["rows"]


def test_summarize_task_smoke():
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    groups = np.array([1, 1, 2, 2, 3])
    s = summarize_task(y_true, y_pred, groups, "t1", 3, n_boot=50)
    assert s["accuracy"] == 1.0
    assert s["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert "group_bootstrap" in s
