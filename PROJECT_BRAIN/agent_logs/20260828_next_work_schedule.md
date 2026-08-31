# 2026-08-28 — Next Work Schedule (v6 metric-fix, post-formal-training)

Status: **schedule updated from real disk evidence; formal training completed on V100, final gate fix syncing pending server connectivity**

## What happened (verified)

- v6 code audit found invalid last-batch epoch metrics in v5 and other scientific/reporting defects; all fixed in-place with TDD (branch `v6-metricfix-20260827`, tags `v6-metricfix-code-20260827`, `v6-metricfix-sync-20260828`, `v6-metricfix-sync-r2-20260828`, `v6-metricfix-sync-r3-20260828`).
- Diagrams updated to 2026-08-27 pair (structure + flow); v4 dated pair retained as historical.
- Server sync verified twice (r2: 130 files mismatch=0, remote pytest 134/134); retained v4 baseline restored on server (39 files mismatch=0); `pymatgen`/`mp-api` installed with `numpy<2`, `pip check` clean, Conv1D GPU probe finite.
- V100 formal run: SSL 60 epochs and supervised 60 epochs completed GPU-only; supervised canonical best epoch=47 on aggregate `val_loss`; best/last/accepted frozen before outer access; evaluation-only produced metrics/predictions/MC/plots.
- Formal run ended at the final artifact content gate with an `ImportError` caused by a remote `scripts` package name collision. Fixed in r3 with lightweight `src/utils/selection_manifest.py` (no TensorFlow import at gate time). Local tests: Stage-0 `117 passed in 26.06s`, full WSL `135 passed in 211.56s`.

## Immediate tasks (auto-retrying in background)

1. Apply r3 core archive on server and rerun remote compile/full pytest.
2. Rerun `run_full_pipeline.py` WITHOUT `--fresh/--force-*` so training stages are reused and only the final gate + model-brain manifest complete.
3. Package v6 models/checkpoints/reports/logs and return them to local; verify archive/member SHA-256; recompute metrics from `ood_test_predictions.json`; load models and run finite forward.
4. Write the final dated v6 training report with real metrics.

## Then

5. Promote v6 to latest accepted only after step 4 passes; keep v5 as immutable historical run (checkpoint-selection caveat) and v4 as baseline.
6. Update README/dev_context/CONSTITUTION with final v6 numbers and the r3 gate fix; commit and tag final docs.
7. Remove local one-shot scripts from Temp (security hygiene).

## Deferred (unchanged priority, no new direction)

- 3-seed mean/variance and group-bootstrap uncertainty on the fixed v5 outer split.
- Uncertainty calibration for DFT acquisition (raw vs tolerance coverage already separated).
- Materials Project dual-source formal data (network/ASN block remains).
- Phase C (crystal structure → multi-band) remains **pending**; schema/k-path/multi-band/OOD contracts still unfrozen.

## Not changing

- Main chain `E(k) → (N,2,128,3) → MBM → gap/type → GUI` untouched.
- v4/v5 raw/tensor/model artifacts immutable.
- Provider labels remain target/audit only; outer test remains final-evaluation-only.
