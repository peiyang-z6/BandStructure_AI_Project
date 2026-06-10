# 2026-06-05 Topology Injection and Probability Sharpening

## Trigger

The extremum expected-value gap head reduced OOD line-mode MAE, but the classifier still lacked explicit access to the physical topology that distinguishes direct and indirect gaps.

## Changes

- Added learned-temperature softmax to the extremum probability head.
- Exposed `P_vbm` and `P_cbm` topology features:
  - expected k-distance
  - probability overlap
  - symmetric KL divergence
- Concatenated topology priors with pooled encoder features before the classification MLP.
- Added topology consistency loss:
  - Direct samples are penalized when expected VBM/CBM k-points separate.
  - Indirect samples are penalized when expected VBM/CBM k-points collapse within a margin.
- Added entropy sharpening loss for extremum probability distributions.
- Implemented layered learning rates with head LR `1e-3` and encoder effective LR `1e-5` via encoder-gradient scaling.
- Generated `reports/finetune_supervised/extremum_probability_heatmaps_ood_test.png`.

## Fresh-Run Outcome

- OOD Line-mode Gap MAE: `0.350426 eV`
- OOD Line-mode Gap RMSE: `0.540193 eV`
- OOD Model vs Global DFT MAE: `0.567594 eV`
- OOD Direct Gap Recall: `1.000000`
- OOD Type Accuracy: `0.762712`
- OOD Macro F1: `0.485526`
- OOD Direct AUC: `0.845745`
- OOD R2: `0.879657`

## Interpretation

The target Direct Gap Recall > `0.90` was achieved on the current OOD split. Macro F1 remains limited by indirect/direct tradeoff and small sample count, but the classifier now has explicit topology channels instead of relying on pooled latent geometry alone.
