# 2026-06-04 Focal Loss Single-Weight Calibration

## Scope

- Removed manual Dataset sample weights from `scripts/finetune_supervised.py`.
- Kept class imbalance correction inside `FocalLoss(class_weights=...)` as the only type-head weighting mechanism.
- Renamed model input parameter naming from `num_bands` to `num_features` in `src/models/band_structure_encoder.py`.
- Regenerated all non-download artifacts from the existing raw Materials Project cache.

## Fresh Run Results

- OOD tensor shape: `(200, 2, 128, 3)`
- OOD split: `141 / 59`, grouped by `spacegroup_number`, group overlap `[]`
- SSL MBM best validation loss: `0.27048`
- Supervised OOD MAE: `0.597 eV`
- Supervised OOD RMSE: `0.968 eV`
- OOD Accuracy: `0.763`
- OOD Macro F1: `0.467`
- OOD Direct Gap Recall: `0.750`
- OOD confusion counts `[metal, direct, indirect]`: `[[0, 0, 0], [0, 9, 3], [0, 11, 36]]`

## Notes

The calibration removed the previous Focal Loss double-weighting path that over-pushed indirect samples into the direct class. Direct recall remains meaningful without collapsing indirect classification.
