# 2026-06-05 Effective-Mass Visualization

## Trigger

The final defense figures needed to show that the model captures band curvature, not only energy alignment.

## Changes

- Updated `src/utils/visualizer.py`.
- `plot_band_overlay` now computes second-derivative curvature at true and predicted VBM/CBM extrema.
- The figure annotates effective-mass proxies using `m* ~= 1 / curvature`.
- Annotation color is green when predicted and true curvature signs agree.
- Annotation color is red when the predicted curvature sign flips relative to truth.
- True extrema are drawn as filled markers; predicted extrema are drawn as open markers.

## Fresh-Run Artifacts

- `reports/finetune_supervised/band_overlay_examples_ood_test.png`
- `reports/finetune_supervised/curvature_zoom_examples_ood_test.png`
- `reports/finetune_supervised/metrics_summary.json`

## Latest Fresh-Run Metrics

- OOD Line-mode Gap MAE: `0.493444 eV`
- OOD Type Accuracy: `0.847458`
- OOD Macro F1: `0.522589`
- OOD Direct Gap Recall: `0.750000`
- OOD R2: `0.772382`

## Note

The visualization upgrade succeeded. Because the full pipeline retrains SSL and fine-tuning from scratch, classification metrics can fluctuate across fresh runs on the small OOD split.
