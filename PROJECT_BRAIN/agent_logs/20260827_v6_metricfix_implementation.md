# 2026-08-27 — v6 Metric-Fix Implementation and GPU Retrain Execution

Status: **local implementation and smoke complete; final regression/sync/server formal training pending**

## Authorization

User approved the revised complete plan with these decisions:

- canonical checkpoint monitor: complete inner-validation aggregate `val_loss`;
- secret-free canonical core commit/tag allowed;
- v6 disables dimensionally invalid curvature-magnitude consistency;
- v6 disables whole-path virtual-strain heuristic;
- update diagrams, synchronize local/server, run GPU-only SSL+supervised training, return results and update schedule.

## Frozen experiment identity

- candidate: `aflow_noleak_v6_30k_seed42_metricfix`
- immutable input: v5 raw 30k + `ood_tensors_v5_30000_seed42`
- historical v5: retained with invalid checkpoint-selection caveat
- retained v4: immutable baseline; model-brain manifest timestamp incident recorded separately

## Git baseline

- branch: `v6-metricfix-20260827`
- pre-v6 audited snapshot: `025d5ce`
- verified metric-fix code: `7126056230a204cdd71d915b3a1db7be5ec4e2a7`
- code tag: `v6-metricfix-code-20260827`
- pre-snapshot regression: `92 passed in 414.21s`
- secret gate: staged host/password/private-key matches = 0
- actual `configs/api_keys.env`, raw/processed data, and trained artifacts remained ignored/uncommitted

## Confirmed root causes

1. Supervised custom train/test steps returned current-batch scalars; old epoch metrics and checkpoint selection represented the final partial validation batch (20 samples).
2. Line-mode target is the exact input function `min(CBM_E)-max(VBM_E)`; analytic baseline error is zero.
3. SSL span overlap/segment truncation produced actual mask mean about 0.14387; validation corruption changed each epoch; batch means were averaged without denominators.
4. Curvature-magnitude consistency compared separately standardized curvature to index-space second differences of separately standardized energy.
5. Post-hoc curvature and whole-path augmentation were not segment-safe.
6. Report template hard-coded 2026-06-04 and Materials Project paths; MC tolerance coverage was mislabeled as raw 95% coverage.
7. Pipeline hard-coded v4 paths and status-only wrote the model-brain manifest.
8. v5 lacked supervised-last and one consolidated final artifact inventory.

## RED→GREEN changes

### Supervised

- Stateful full-epoch trackers for gap loss/MAE, weighted type loss, accuracy, Macro F1, and auxiliaries.
- Canonical `val_loss` checkpoint/early-stop monitor.
- `best.weights.h5`, `last.weights.h5`, accepted-restored-best separated.
- Atomic `inner_selection_manifest.json` with paths/bytes/SHA, monitor and best epoch.
- Frozen-manifest hash gate before outer evaluation.
- `--train-only` does not load outer arrays/sample manifest; `--evaluation-only` loads outer only after gate.
- Accepted weights are not rewritten during report/config generation.
- SSL-only mask token and projection/reconstruction heads frozen for supervised fine-tuning.
- Keras Dataset fit explicitly uses `shuffle=False`; dataset itself owns shuffling.
- Whole-path augmentation default OFF.
- Segment IDs retained and used for post-hoc curvature/validator.
- Report provenance/runtime paths fixed; analytic baseline/global DFT/model approximation separated.
- Sample/group-macro/mismatch metrics and raw/tolerance MC coverage added.
- MC dropout now honors bounded `batch_size`.
- Parity axes correctly name analytic line-mode and learned soft-extremum quantities.

### SSL

- Stateless-seed-capable segment-safe span union; each sample actual mask in 15–30%.
- Validation mask fixed by random state + batch index.
- MSE/MAE weighted by masked elements, mask fraction by positions, auxiliaries by samples.
- `ssl_history.json` atomically updated each epoch.
- Curvature-magnitude consistency default 0 and hard-disabled against adaptive revival.
- Whole-path strain default OFF; explicit enable retained only for experimental compatibility.
- GPU placement gate on encoder and reconstruction tensors.

### Pipeline

- Explicit experiment ID/raw HDF5/OOD tensor dir/report date.
- Explicit immutable inputs cannot be downloaded into, cleaned, or rebuilt.
- Supervised orchestration fixed to train-only → evaluation-only.
- v6 required-artifact gate includes best/last/inner-selection manifest.
- `--status-only` is now no-write.
- GPU/provenance flags propagate to child commands.

## Local verification completed

- Final Stage-0: `116 passed in 27.64s`.
- Final local compileall: exit 0.
- Final local full pytest: `134 passed in 227.52s`.
- Immutable v4/v5 raw/tensor/model/prediction SHA-256 values all matched frozen baselines.
- Real `run_full_pipeline.py --source aflow --status-only` no-write probe preserved v4 manifest SHA `91a7b2...a32c`.
- Local supervised 1-epoch GPU smoke on RTX 4060:
  - full-epoch metrics logged;
  - best/last/accepted and inner manifest generated;
  - train-only confirmed outer arrays not loaded;
  - evaluation-only passed hash gate, generated metrics/predictions/MC/plots/reports;
  - accepted hash remained unchanged;
  - analytic baseline recorded as 0;
  - dated report and latest alias byte-identical.
- Local SSL 1-epoch GPU smoke:
  - validation actual mask ≈ 0.25015;
  - symmetry weight = 0;
  - best/last, final/pretrained model and `ssl_history.json` generated.
- Diagrams created and visually checked:
  - `PROJECT_BRAIN/diagrams/project_structure_diagram_20260827.html/.png`
  - `PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260827.html/.png`
  - PNG 1600×1500; main diagram, cards and footer complete, no obvious clipping/overlap.
- Retrospective v5 manifest: 61 files, 290,169,865 bytes.

## Remote read-only preflight

- 2× Tesla V100-SXM2-16GB idle at inventory time;
- about 3.23 TB free;
- TensorFlow 2.21 / CUDA build 12.5.1 / sm_70;
- Conv1D forward/backward on `/GPU:0`, finite gradients;
- remote v5 data/models/predictions match local hashes;
- remote code predates current metric-fix branch.

## Independent pre-commit review remediation

The first independent logic review returned `passed=false`. Every reported/reproducible issue was handled by additional RED→GREEN tests:

1. Rejected `.` / `..` / hidden/traversal experiment IDs and unsupported sources.
2. Restricted explicit raw/OOD inputs to `data/raw/<source>/` and `data/processed/<source>/`; bound sibling snapshot metadata/raw manifest and tensor audit.
3. Prevented fresh/clean/download/build from touching explicit immutable inputs or shared vision directories.
4. Rejected best/last/accepted path collisions in both freeze and validation paths.
5. Bound evaluation checkpoint path to the manifest-declared best state.
6. Recorded SSL pre-adaptation selection weights separately from post-adaptation weights.
7. Saved `epoch_var` before best/last checkpoints; checkpoint epochs now match history.
8. Required nonempty regular artifact files and validated selection hashes, SSL history schema/mask/weights, and SSL checkpoint epochs before manifest publication.
9. Moved final required/content gates before model-brain manifest write.
10. Corrected remaining report labels and made curvature zoom segment-aware.

Delegation-provider retries failed to return a parseable second verdict due provider/network failures. A separate local Codex CLI read-only review of the **current working tree** independently marked these nine core gates verified: explicit input containment, immutable cleanup, distinct supervised states, manifest hash/size validation, manifest-bound evaluation, one-based supervised best epoch, SSL best epoch before checkpoint save, nonempty regular files, and content gate before manifest publication. Its only warning was missing programmatic source allowlisting; that warning was then fixed through RED→GREEN. Codex could not run tests in its pyenv, so WSL project tests remain the execution authority.

## Remote synchronization and V100 smoke corrective action

- Applied core archive tag `v6-metricfix-sync-20260828`, commit `7cbd3541ee6fc797097660aea029d2ad45221dd3`.
- Archive SHA-256 `62c8422fd4ba633a3c59bd60956abc19ac6cb74f6056fca1998dcd15b4b7e684`; 127 file hashes verified before and after apply; secrets excluded; v5 artifacts not overwritten.
- Historical server cleanup had removed retained v4 assets and `pymatgen/mp-api`; restored 39 v4 files (mismatch=0), installed project-declared `pymatgen`/`mp-api` with `numpy<2`, `pip check` clean, Conv1D `/GPU:0` finite, MP tests 4/4, full remote pytest 134/134.
- Remote v4/v5 hashes matched local frozen values after restoration.
- First V100 smoke: SSL 1 epoch and supervised train-only 1 epoch passed, but evaluation-only failed at Conv1D inference because the newly uncompiled evaluation model allowed Keras/XLA auto-JIT; V100 cuDNN autotuner could not select a supported `convBiasActivationForward` config.
- Root-cause TDD fix: evaluation now restores frozen best **before** `model.compile(jit_compile=False)`. This avoids both optimizer-state restore warnings and XLA inference autotuning. Local Stage-0/full tests passed after the change; corrected source must be resynchronized before rerunning remote smoke.

## Pending gates

1. Manifest-driven core sync, remote readback and remote tests.
2. Remote 1-epoch SSL/supervised smoke.
3. Full V100 SSL + supervised train-only + evaluation-only.
4. Result archive return, local metrics recomputation/model load/hash acceptance.
5. Final README/dev_context/constitution/training report/schedule closure.

No v6 formal metric is claimed before these gates complete.
