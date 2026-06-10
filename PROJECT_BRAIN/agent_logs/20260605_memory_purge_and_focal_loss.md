# 2026-06-05 Memory Purge and Focal Loss Repair

## Purpose

Fix the direct/indirect classification accuracy illusion and purge stale project memory that referenced deleted legacy modules.

## Constitution Read

Read `PROJECT_BRAIN/CONSTITUTION.md` and preserved the active rules:

- Keep the 4-step pipeline.
- Preserve protected data/model/report assets.
- Maintain OOD split by `spacegroup_number`.
- Do not use sample-wise random splitting.

## Tensor Feature Repair

Updated `src/data/ood_tensor_builder.py`:

- Old tensor: `(N, 2, 128, 2)`
- New tensor: `(N, 2, 128, 3)`
- Flattened model input:
  - `VBM_E`
  - `VBM_curv`
  - `VBM_k_dist`
  - `CBM_E`
  - `CBM_curv`
  - `CBM_k_dist`

The k-distance channels expose whether VBM and CBM extrema coincide, which directly supports direct/indirect classification.

## SSL Repair

Updated `scripts/train_ssl.py`:

- Dynamic feature indexing for 2 bands x C channels.
- Curvature sign/consistency losses now locate CBM energy and curvature from `features_per_band`.
- Saved `ssl_mbm_final_epoch100.keras`.
- Restored `ckpt-best` before saving `ssl_mbm_pretrained.keras`.

New SSL MBM best validation loss: `0.29776`.

Baseline SSL MBM best validation loss before this repair: `0.38232`.

## Focal Loss and Imbalance Repair

Updated `scripts/finetune_supervised.py`:

- Added class-weight computation from `type_train`.
- Added class-weighted Focal Loss with `gamma=2.0`, `alpha=0.25`.
- Added sample weights for the classification output to handle multi-output Keras training robustly.
- Added Macro F1 and Direct Gap Recall to metrics.
- Added Markdown report generation.

Class weights:

- metal: `1.0`
- direct: `1.905405`
- indirect: `0.677885`

## Results

Baseline supervised OOD MAE before repair: `0.561 eV`.

Current supervised OOD results after repair:

- OOD MAE: `0.737 eV`
- OOD RMSE: `1.036 eV`
- OOD Accuracy: `0.678`
- OOD Macro F1: `0.435`
- Direct Gap Recall: `1.000`
- Direct AUC: `0.8067`
- Indirect AUC: `0.8050`

Confusion matrix effect:

- Direct gaps: `12/12` recovered.
- Indirect gaps: `28/47` retained as indirect, `19/47` shifted to direct.

Interpretation: the repair successfully breaks the "all indirect" accuracy illusion, but it over-corrects toward direct predictions. Future work should tune focal alpha/gamma, type loss weight, or decision thresholds.

## Memory Purge

Rewrote `PROJECT_BRAIN/dev_context.md` to remove old Phase 1-10 and deleted module descriptions. The file now only describes the current simplified 4-step pipeline.
