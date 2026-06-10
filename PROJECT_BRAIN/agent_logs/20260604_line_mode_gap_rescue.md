# 2026-06-04 Line-mode Gap Rescue

## Physics Correction

- Regression target changed from global MP DFT gap to line-mode tensor gap:
  `E_CBM - E_VBM`.
- Global DFT gap is retained as an information-bottleneck residual:
  `tensor_gap - global_dft_gap`.
- Reports now distinguish:
  - Line-mode Gap MAE: model prediction vs tensor extrema.
  - DFT Gap Residual: tensor extrema vs global MP DFT gap.

## Fine-tuning Rescue

- Disabled Focal Loss for the 141-sample training regime.
- Classification now uses Categorical Crossentropy with class sample weights.
- Set conservative defaults:
  - learning rate `2e-5`
  - type loss weight `0.5`
  - freeze first 3 of 4 transformer blocks
  - no ReduceLROnPlateau

## Rationale

The model cannot infer global band extrema that are absent from the line-mode
path. Aligning the regression target to the observed tensor preserves data
fidelity and makes the remaining global DFT residual an honest measurement of
the line-mode information bottleneck.
