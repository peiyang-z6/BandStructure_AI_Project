"""Aggregate 3-seed x seven-split three-task benchmark results (P0D).

Input: three experiment report dirs (one per seed), each containing
seven_split_evaluation.json produced by evaluate_seven_splits.py.
Output: three_seed_aggregate.json with per-task / per-split mean±std and
95% bootstrap CIs averaged across seeds (per-split CI reported individually).
"""
import argparse
import json
import os
import sys

import numpy as np

TASKS = (
    "line_mode_topology",
    "provider_global_electronic_type",
    "line_global_disagreement",
)


def load_eval(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


def aggregate(report_dirs: list[str]) -> dict:
    seed_results = []
    for report_dir in report_dirs:
        path = os.path.join(report_dir, "seven_split_evaluation.json")
        if not os.path.isfile(path):
            print(f"SKIP missing eval: {path}")
            continue
        seed_results.append(load_eval(path))
    if not seed_results:
        raise RuntimeError("no seven_split_evaluation.json found in any report dir")

    split_names = [name for name in seed_results[0]["splits"].keys()]
    out = {"seeds": len(seed_results), "splits": {}}
    for split_name in split_names:
        split_out = {"tasks": {}}
        for task in TASKS:
            accs, macros, ci_lows, ci_highs = [], [], [], []
            n_samples = None
            for result in seed_results:
                task_summary = (
                    result.get("splits", {}).get(split_name, {}).get("tasks", {}).get(task)
                )
                if not task_summary:
                    continue
                accs.append(task_summary["accuracy"])
                macros.append(task_summary["macro_accuracy"])
                boot = task_summary.get("group_bootstrap") or {}
                if boot:
                    ci_lows.append(boot["ci_low"])
                    ci_highs.append(boot["ci_high"])
                n_samples = task_summary["n_samples"]
            if not accs:
                split_out["tasks"][task] = {"error": "metric missing in some seeds"}
                continue
            split_out["tasks"][task] = {
                "accuracy": _agg(accs),
                "macro_accuracy": _agg(macros),
                "ci95_low": _agg(ci_lows) if ci_lows else None,
                "ci95_high": _agg(ci_highs) if ci_highs else None,
                "n_samples_per_split": n_samples,
            }
        out["splits"][split_name] = split_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 3-seed seven-split results")
    parser.add_argument("--report-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = aggregate(args.report_dirs)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.output} ({out['seeds']} seeds, {len(out['splits'])} splits)")


if __name__ == "__main__":
    main()