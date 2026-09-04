"""Regression tests for 3-seed aggregation (P0D)."""
import json
import os

import numpy as np

from scripts.aggregate_three_seed_results import _agg


def test_agg_mean_std():
    out = _agg([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0
    assert abs(out["std"] - 1.0) < 1e-9
    assert out["min"] == 1.0 and out["max"] == 3.0 and out["n"] == 3


def test_agg_single_value_std_zero():
    out = _agg([0.5])
    assert out["std"] == 0.0


def test_aggregate_end_to_end(tmp_path):
    from scripts.aggregate_three_seed_results import aggregate

    for i, seed in enumerate(("a", "b")):
        d = tmp_path / seed
        d.mkdir()
        splits = {
            "space_group": {
                "tasks": {
                    "line_mode_topology": {
                        "accuracy": 0.9 + i * 0.02,
                        "macro_accuracy": 0.88 + i * 0.02,
                        "n_samples": 1000,
                        "group_bootstrap": {"ci_low": 0.89 - i * 0.01, "ci_high": 0.91 + i * 0.01},
                    },
                    "provider_global_electronic_type": {
                        "accuracy": 0.85 + i * 0.02,
                        "macro_accuracy": 0.83 + i * 0.02,
                        "n_samples": 1000,
                        "group_bootstrap": {},
                    },
                    "line_global_disagreement": {
                        "accuracy": 0.80 + i * 0.02,
                        "macro_accuracy": 0.78 + i * 0.02,
                        "n_samples": 1000,
                        "group_bootstrap": {"ci_low": 0.79, "ci_high": 0.81},
                    },
                }
            }
        }
        (d / "seven_split_evaluation.json").write_text(
            json.dumps({"seeds": 1, "splits": splits}), encoding="utf-8"
        )
    agg = aggregate([str(tmp_path / "a"), str(tmp_path / "b")])
    task = agg["splits"]["space_group"]["tasks"]["line_mode_topology"]
    assert task["accuracy"]["mean"] == 0.91
    assert task["accuracy"]["n"] == 2
    assert task["ci95_low"]["mean"] == 0.885