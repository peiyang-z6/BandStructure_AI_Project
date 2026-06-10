# 2026-06-04 Supervised Fine-tuning and Physics Validation

## Completed

- Added `scripts/finetune_supervised.py` for Step 3.
- Preserved the OOD tensor split from `data_cache/ood_tensors/band_tensors_ood_split.npz`.
- Loaded `models/ssl_mbm_pretrained.keras` and froze the first 2/4 transformer blocks.
- Added two supervised heads:
  - band gap regression
  - metal/direct/indirect gap type classification
- Added physics validation using `E_CBM - E_VBM = E_gap` from processed tensor extrema.
- Generated parity plots for train and OOD test.
- Saved model as weights plus config for robust Keras subclass restoration.

## Main Results

- Training early-stopped after 45 epochs.
- Best validation gap MAE: 0.563009 eV.
- OOD test:
  - MAE: 0.563009 eV
  - RMSE: 0.994923 eV
  - type accuracy: 0.796610
  - parity correlation: 0.799903
  - tensor-gap MAE: 0.370484 eV

## Artifacts

- `models/finetuned_gap_predictor.weights.h5`
- `models/finetuned_gap_predictor_config.json`
- `checkpoints/finetune_supervised/best.weights.h5`
- `reports/finetune_supervised/metrics_summary.json`
- `reports/finetune_supervised/ood_test_predictions.json`
- `reports/finetune_supervised/parity_plot_ood_test.png`
- `reports/finetune_supervised/parity_plot_train.png`
- `reports/finetune_supervised_report_20260604.md`

## Notes

- `scripts/main_pipeline.py` was not used because it is an older broad pipeline and does not encode this Step 3 flow over OOD tensors plus the MBM SSL encoder.
- The 0-byte `models/finetuned_gap_predictor.keras` artifact from the initial subclass save attempt was removed.
- A fresh model reconstruction and `load_weights` test passed.
