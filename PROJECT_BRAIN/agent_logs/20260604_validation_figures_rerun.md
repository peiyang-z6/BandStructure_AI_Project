# 2026-06-04 Validation Figures Rerun

## Request

Add the validation figures described in `C:\Users\PeiYang\Desktop\效果检验.txt`, then regenerate the Step 3 training results under `reports` while avoiding ambiguity with previous outputs.

## Changes

- Extended `scripts/finetune_supervised.py` to generate:
  - training curves
  - learning-rate curve
  - parity plots with MAE/R2
  - OOD error histogram
  - confusion matrix
  - one-vs-rest ROC curves
  - OOD band overlay examples
  - VBM/CBM curvature zoom plots
  - latent t-SNE by spacegroup
  - physics violation rate bar chart
- Added `ReduceLROnPlateau` and learning-rate logging.
- Added deterministic `--random-state`.
- Corrected physics curvature validation to compute d2E/dk2 from reconstructed energy curves instead of trusting the reconstructed curvature channel.

## Regenerated Outputs

- Removed previous Step 3 outputs before rerun:
  - `reports/finetune_supervised`
  - `reports/finetune_supervised_report_20260604.md`
  - `checkpoints/finetune_supervised`
  - `models/finetuned_gap_predictor.weights.h5`
  - `models/finetuned_gap_predictor_config.json`
- Re-ran supervised fine-tuning.
- Final run early-stopped after 47 epochs.

## Final OOD Metrics

- MAE: 0.560982 eV
- RMSE: 1.027373 eV
- R2: 0.559989
- type accuracy: 0.830508
- direct AUC: 0.783688
- indirect AUC: 0.780142
- physics violation rates:
  - negative predicted gap: 0.0
  - VBM positive curvature: 0.0
  - CBM negative curvature: 0.0

## Main Artifacts

- `reports/finetune_supervised/metrics_summary.json`
- `reports/finetune_supervised/training_curves.png`
- `reports/finetune_supervised/learning_rate_curve.png`
- `reports/finetune_supervised/parity_plot_ood_test.png`
- `reports/finetune_supervised/error_distribution_ood_test.png`
- `reports/finetune_supervised/confusion_matrix_ood_test.png`
- `reports/finetune_supervised/roc_curves_ood_test.png`
- `reports/finetune_supervised/band_overlay_examples_ood_test.png`
- `reports/finetune_supervised/curvature_zoom_examples_ood_test.png`
- `reports/finetune_supervised/latent_tsne_spacegroups.png`
- `reports/finetune_supervised/physics_violation_rates_ood_test.png`
