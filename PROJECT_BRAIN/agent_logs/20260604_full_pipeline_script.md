# 2026-06-04 Full Pipeline Script

## Decision

The training flow is complete: downloaded data, OOD tensors, SSL MBM pretraining, supervised fine-tuning, and validation reports all exist.

The current model brain is represented by:

- `models/ssl_mbm_pretrained.keras`
- `models/ssl_mbm_norm_stats.json`
- `models/finetuned_gap_predictor.weights.h5`
- `models/finetuned_gap_predictor_config.json`
- `data_cache/ood_tensors/band_tensors_ood_split.npz`
- `reports/finetune_supervised/metrics_summary.json`

## Added

- `scripts/run_full_pipeline.py`

## Behavior

- Runs download -> OOD tensors -> SSL -> supervised fine-tuning -> manifest.
- Defaults to reusing complete existing stages.
- `--fresh` rebuilds processed/training/report artifacts while keeping raw data.
- `--clean-raw` removes raw data only when explicitly combined with `--fresh`.
- `--status-only` prints artifact status and writes the model brain manifest.
- The manifest is saved to `models/physics_model_brain_manifest.json`.

## Verification Plan

- Run `python scripts/run_full_pipeline.py --help`.
- Run `python scripts/run_full_pipeline.py --status-only`.
