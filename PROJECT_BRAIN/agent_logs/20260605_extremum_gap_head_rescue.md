# 2026-06-05 Extremum Gap Head Rescue

## Trigger

The supervised model used a pooled encoder vector for gap regression. This erased local VBM/CBM extrema and caused poor line-mode gap prediction.

## Changes

- Replaced pooled Dense gap regression with a sequence-local `ExtremumExpectedGapHead`.
- The new head predicts VBM/CBM k-point probability distributions with Conv1D logits.
- It denormalizes VBM_E and CBM_E channels back to eV inside the model.
- The supervised gap loss and MAE now operate on physical eV values.
- Raised `type_weight` from `0.5` to `1.0`.
- Changed `freeze_layers` from `3` to `2`, unfreezing the final two transformer blocks.
- Restored `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)`.
- Kept EarlyStopping patience at `20`.

## Fresh-Run Outcome

- OOD Line-mode Gap MAE: `0.470519 eV`
- OOD Line-mode Gap RMSE: `0.692908 eV`
- OOD Model vs Global DFT MAE: `0.681300 eV`
- OOD DFT Gap Residual MAE: `0.250708 eV`
- OOD Type Accuracy: `0.694915`
- OOD Macro F1: `0.418768`
- OOD Direct Gap Recall: `0.666667`
- OOD R2: `0.801997`

## Interpretation

The main failure mode was local-extremum information loss, not only label mismatch. Preserving sequence-local extrema improved the line-mode gap MAE from about `1.21 eV` to `0.47 eV`.
