# 2026-06-04 Constitutional Pipeline, SSL, Fine-tune, and Visualization Audit

## Data Pipeline

- Confirmed `GroupShuffleSplit` is used with `spacegroup_number` groups.
- Added `tests/test_data_pipeline.py`.
- Test prints train/test spacegroup intersection and asserts it is empty.
- Test validates tensor contract `(N, 2, 128, 3)` and flattened 6D features:
  `[VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]`.

## SSL Loss

- Added `src/models/losses.py` with formula comments.
- Added differentiable soft-window `BandCurvatureLoss`.
- Added `src/engine/ssl_trainer.py` with TensorBoard logs for:
  `mse_loss`, `curvature_loss`, `symmetry_loss`, `curvature_weight`, and `symmetry_weight`.
- `scripts/train_ssl.py` now uses the shared SSL trainer.

## Fine-tuning

- Added `src/engine/finetune_trainer.py` for freeze strategy reporting.
- Fine-tuning reports now include `finetune_strategy_summary.json`.
- Classification uses multiclass softmax Focal Loss from `src/models/losses.py`.

## Visualization

- Parity plots include a `y=x` line with MAE and R2 in the legend.
- Band overlays keep kpath labels, high-symmetry vertical guide lines, and red Fermi-level reference.
