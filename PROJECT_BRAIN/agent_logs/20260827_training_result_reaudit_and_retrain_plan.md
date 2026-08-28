# 2026-08-27 — Training Result Reaudit and Retrain Plan Gate

Status: **local audit completed; remote read-only inventory blocked on SSH re-authorization; implementation not yet approved**

Audit timestamp: `2026-08-27T23:17:06+08:00`

## User request

1. 检查当前训练结果是否有问题；
2. 对确认的问题进行原地修复；
3. 更新项目结构图和端到端流程图；
4. 安全同步本机与 GPU 服务器；
5. 修改完成后重新训练，将真实结果回传本机；
6. 更新项目日程和治理文档。

## Frozen contracts

- 主链路保持 `E(k) → (N,2,128,3) → Transformer/MBM → gap+type heads → GUI`；不得建立第二套训练框架。
- `aflow_noleak_v4_seed42` 与 `aflow_noleak_v5_30k_seed42` 的 raw/tensor/model/checkpoint/report 继续保留为 immutable historical artifacts。
- outer OOD test 不得参与 normalization、checkpoint selection、early stopping、阈值选择或超参数调优。
- 新训练必须创建新的 versioned experiment；GPU 不可见或关键训练算子落到 CPU 时 fail-fast。
- provider labels 只作 target/audit；AFLOW feature reference 保持 canonical `E_F=0`。
- Phase C 保持 pending。

## Local artifact integrity

本轮重新计算下列 SHA-256，全部与冻结基线一致：

- v4 raw HDF5: `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`
- v5 raw HDF5: `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`
- v5 full tensor: `9c5edea5e97f1c9b2561bc1c2324bcdd22741b5aea69e7d7c4c25116375a1ff7`
- v5 OOD split: `c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`
- v5 SSL model: `8a57975a5461f8ebc5a18d424ce91ed752a4f6bfa36e7cc8cdb30edf7b9cc6c1`
- v5 supervised portable weights: `91f42fc3bb67ab676ad369c01822b8f6b58bc2676dd968406e38fa3359b1a5be`
- v5 supervised best checkpoint: `3f55d2756a320c5a172da61155981747d18195dfa45e98423e5589964ef0e7f6`
- v5 outer predictions: `4d1aa9cc7e0fa5ef97ea7a08e8b7fe1f5c19529c69871737c07047e4bcce54e7`

Portable load/forward smoke also passed. Conclusion: current problem is not corrupted data/model bytes.

## Verified current outer metrics

Independent recomputation from all 6,019 rows in `ood_test_predictions.json` gives:

- line-mode tensor-gap MAE: `0.000352150382 eV`
- line-mode tensor-gap RMSE: `0.001863863436 eV`
- type accuracy: `0.947333444094`
- type Macro F1: `0.932648282183`

Therefore documents claiming RMSE `0.000595` or accuracy `0.941186` are incorrect. The gap result is an E(k)-derived reconstruction metric, not structure-to-gap prediction accuracy. Global DFT-gap residual MAE remains `0.640224159 eV` and must be reported separately.

## Confirmed blocker: epoch metrics are last-batch metrics

Root cause is in `scripts/finetune_supervised.py`:

- custom `train_step`/`test_step` return raw current-batch tensors from `_metrics_dict`;
- no Keras metric trackers aggregate across batches;
- formal supervised batch size is 32;
- inner train/validation sizes are 20,329 / 3,604, with final partial batches of 9 / 20;
- every logged train `type_acc` is an exact multiple of `1/9` and every logged `val_type_acc` is an exact multiple of `1/20`.

Thus `training_log.csv` records the last partial batch, not a full epoch. `ModelCheckpoint(monitor="val_gap_mae")` selected epoch 52 using only the final 20 validation samples. Current outer metrics are reproducible for that checkpoint, but checkpoint selection and all quoted inner-validation metrics are scientifically invalid. This requires a RED→GREEN fix and a new full training run.

## Additional confirmed defects / reporting drift

1. `save_markdown_report()` hard-codes report date `2026-06-04`, Materials Project tensor paths, and MP output directories; AFLOW retraining reproduces a wrong report.
2. `README.md`, `dev_context.md`, execution log and latest report disagree on SSL masked MAE, supervised inner MAE, outer RMSE, accuracy and transfer hashes/count scope.
3. Current MC report labels tolerance-augmented coverage as 95% CI coverage. ID-joined recomputation without tolerance gives raw 95% coverage `0.896992856`; error-vs-std Pearson correlation is `0.752116759`. New reports must separate raw calibration from tolerance diagnostics.
4. `metrics_summary.json` stores absolute server output paths, reducing portability; new manifests should store project-relative paths plus explicit generation root provenance.
5. `run_full_pipeline.py` still hard-codes AFLOW v4 paths and experiment ID. A new 30k run is therefore unsafe without explicit snapshot/experiment parameterization.
6. `run_full_pipeline.py --status-only` writes `physics_model_brain_manifest.json`, despite the read-only-looking command name. This requires an explicit write flag or true no-write status behavior.
7. The v5 split manifest records 5,384 provider-metal vs line-mode crossing mismatches. This is a declared domain mismatch (global provider label versus sampled high-symmetry path), not permission to alter features from labels; it must remain visible in the report.
8. SSL metrics aggregate batch means rather than weighting by batch/masked-element count. This is smaller than the supervised blocker but should receive a focused weighted-aggregation regression before a full SSL retrain.

## Audit-side-effect incident and restoration obligation

During this audit, `run_full_pipeline.py --status-only` rewrote:

`artifacts/models/aflow_noleak_v4_seed42/physics_model_brain_manifest.json`

The older `20260604_full_pipeline_script.md` log did document that `--status-only` writes this file. The CLI name is not read-only-safe, but the immediate incident also reflects an audit-operator error: that historical behavior was not checked before invocation.

Frozen pre-audit evidence from `artifact_provenance.json`:

- bytes: `11220`
- SHA-256: `00cd2f1029f73c4a7ae28abda3424b875ed318b851dca6cd458f7c5f9c3eb724`

Current regenerated file:

- bytes: `11220`
- SHA-256: `91a7b2979b02cb06b8eed252c0b512dcd53d5742d4638c20155d278e06dda32c`
- regenerated `created_at`: `2026-08-27T11:03:28.143524`

The field layout, artifact statuses and v4 metrics remain present; the hash changed at least because `created_at` was regenerated. No raw data, tensor, model, checkpoint or prediction file changed. A read-only search of the retained server project, local project parent, Temp, Downloads and Recycle Bin found no exact original copy. The old SHA-256 preimage will not be guessed. Final acceptance must therefore preserve the regenerated file, append explicit provenance for this incident, and stop treating the old manifest hash as current; if the user later supplies a backup with SHA-256 `00cd2f...b724`, it can be restored byte-for-byte.

## Remote audit state

Read-only inventory completed at `2026-08-27T16:23:39Z` using the password previously supplied by the user; no credential was printed or persisted in the project.

- Host: Linux, 24 logical CPUs, about 66.1 GB RAM, about 3.23 TB free under the user home filesystem.
- GPUs: `2 × Tesla V100-SXM2-16GB`, both idle; no active SSL/supervised/full-pipeline process.
- Conda environment `bandstructure_gpu30k` still exists.
- TensorFlow 2.21.0 is CUDA-enabled; build CUDA is 12.5.1 and `sm_70` is included.
- A real Conv1D forward/backward probe ran on `/GPU:0` and returned finite gradients. GPU-only retraining is currently feasible.
- Server v5 data: 30,000 raw groups, zero temporary groups, train/test shapes `23,933/6,019`, group overlap zero.
- Server v5 raw manifest, tensor audit, SSL model, supervised weights/checkpoint, metrics and predictions match local hashes.
- `train_ssl.py`, `finetune_supervised.py`, `ssl_trainer.py` and `ood_tensor_builder.py` currently match local bytes; therefore the supervised last-batch metric defect exists on both sides.
- Server is behind the local late-hardening state for `run_full_pipeline.py`, `test_stage0_robustness.py`, README, constitution and dev context.
- Safe sync direction after local fixes is local core source/tests/governance → server. Existing v5 data/model artifacts are identical and must not be overwritten or re-transferred.
- The retained server tree contains no v4 `physics_model_brain_manifest.json`; exact recovery from the server is unavailable.
- Host, password and connection details are intentionally omitted from this log.

## Complete execution plan (pending explicit approval)

### Phase A — RED→GREEN repair in existing files

1. **Full-epoch supervised metrics** — `tests/test_stage0_robustness.py` → `scripts/finetune_supervised.py`.
   - RED fixture uses unequal validation batches and proves current history/checkpoint metric equals the last batch instead of the weighted full set.
   - GREEN adds Keras trackers with `metrics` reset semantics; loss components are sample-weighted, gap MAE uses every sample, type accuracy and confusion-derived Macro F1 use every sample.
   - Canonical checkpoint/early-stop signal is recommended to become aggregate inner `val_loss`, the already-defined joint gap+type+physics objective. Aggregate gap MAE/accuracy/Macro F1 remain separate diagnostics.
   - Outer OOD data is not loaded into any callback or selection path.

2. **Report correctness and portability** — `tests/test_stage0_robustness.py` → `scripts/finetune_supervised.py`.
   - RED asserts runtime experiment/date/source/tensor paths and project-relative output paths; current hard-coded MP/2026-06-04 report fails.
   - GREEN derives report metadata from CLI/run manifest and stores relative paths plus explicit remote generation-root provenance.
   - Uncertainty output reports raw 95% coverage separately from tolerance-based diagnostic coverage; neither is silently renamed.
   - Line-mode E(k)-derived reconstruction, global DFT-gap residual and type classification remain distinct metric scopes.

3. **Version-safe pipeline entry** — `tests/test_stage0_robustness.py` → `scripts/run_full_pipeline.py`.
   - RED requires explicit `--experiment-id` and immutable `--tensor-npz`/snapshot input for non-baseline training.
   - RED hashes the model-brain manifest before/after status inspection and requires no change.
   - GREEN parameterizes the existing pipeline in place; `--status-only` becomes no-write, while a separate explicit manifest-write option carries the side effect.
   - Required-artifact fail-closed and GPU propagation from the 2026-08-26 hardening remain unchanged.

4. **SSL metric denominator** — `tests/test_stage0_robustness.py` → `src/engine/ssl_trainer.py`.
   - RED uses unequal batches/mask counts.
   - GREEN weights reconstruction MAE/MSE by actual masked elements and batch-level objective terms by sample count, eliminating last-partial-batch bias without changing MBM architecture or loss definitions.

5. **Incident provenance**.
   - Do not rewrite historical `artifact_provenance.json`.
   - Add a supplemental transfer/provenance record containing the v4 manifest old hash, regenerated hash, timestamp and this audit log reference.
   - Preserve the regenerated file; restore only if an independently supplied copy matches the exact old SHA-256.

### Phase B — local verification gate

Run in WSL conda environment `bandstructure-ai`:

1. each new test RED, then GREEN, one vertical slice at a time;
2. all affected targeted tests;
3. `python -m compileall -q src scripts tests`;
4. complete `python -m pytest -q`;
5. `git diff --check` and small-diff/line-ending review;
6. read-only hashes for v4/v5 raw, v5 full/split, SSL, supervised checkpoint and predictions;
7. status-only no-write hash regression on the regenerated v4 manifest.

No server upload begins unless every gate passes.

### Phase C — project structure and flow diagrams

Keep the dated 2026-08-24 diagrams as historical records and add the current dated pair under `PROJECT_BRAIN/diagrams/`, then update `PROJECT_BRAIN/diagrams/README.md`:

1. project structure: source/data/snapshots/experiments/reports/GUI/governance and local↔server boundary;
2. end-to-end flow: AFLOW persistence → immutable snapshot → `(N,2,128,3)` → group-disjoint outer split → inner-only MBM/normalization → inner-only supervised selection → one-time outer evaluation → artifact acceptance → GUI;
3. explicitly show GPU-only server gate, no CPU fallback, hash manifests, return verification, and Phase C as pending.

Deliver self-contained HTML plus rendered PNG, with programmatic checks for title/SVG/content and visual inspection for clipping/overlap.

### Phase D — manifest-driven local→server synchronization

1. Freeze local pre-sync manifest and remote pre-sync inventory.
2. Build a small core package containing only approved source/tests/safe config templates/diagrams/governance files.
3. Exclude `configs/api_keys.env`, credentials, reference documents, caches, raw-response cache, v4/v5 models/checkpoints/reports, and unchanged 30k raw/tensors.
4. Upload to a temporary staging directory; verify archive and every member hash.
5. Back up only the remote core files being replaced, then apply them to the existing server project. Do not duplicate or overwrite identical data/artifacts.
6. Re-read remote files and run compile/full pytest there before any training.

### Phase E — new GPU-only formal experiment

Proposed ID: `aflow_noleak_v6_30k_seed42_metricfix`.

1. Reuse the immutable v5 30k raw/split as read-only input; do not redownload or rebuild it.
2. Confirm no trainer is running and rerun the Conv1D forward/backward GPU gate.
3. Run one-epoch SSL and supervised smoke into disposable v6 smoke directories.
4. Run full MBM SSL from scratch, then full supervised fine-tuning; preserve the v5 hyperparameters except for the corrected aggregation/selection semantics.
5. Use one V100 explicitly because distributed multi-GPU training is not implemented/validated; the second GPU remains unused rather than introducing an architectural change.
6. Fail immediately if TensorFlow sees no GPU, a critical tensor is placed on CPU, gradients become non-finite, required checkpoint/log is absent, or duplicate trainer processes appear.
7. Record GPU utilization/memory at fixed intervals, best/last checkpoints, complete stdout/stderr and environment lock.
8. Evaluate the outer 6,019-sample OOD split exactly once after the canonical inner-selected checkpoint is frozen.

### Phase F — return and acceptance

1. Generate per-file and archive SHA-256 manifests on the server.
2. Return only the new v6 models, best/last checkpoints, logs, GPU trace, predictions, metrics, figures, reports and acceptance manifests.
3. Verify local archive size/hash/member count; verify every file hash after extraction.
4. Recompute all headline metrics from returned predictions rather than trusting markdown.
5. Load SSL and supervised models locally, run finite forward inference, and verify class probabilities sum to one.
6. Rehash v4/v5 immutable assets; any unexpected change rejects v6 acceptance.
7. Promote v6 to latest accepted only when every required artifact and gate passes. Retain v5 as an immutable historical run with an explicit checkpoint-selection caveat.

### Phase G — schedule and governance closure

Update together:

- `README.md`;
- `PROJECT_BRAIN/dev_context.md`;
- `PROJECT_BRAIN/CONSTITUTION.md` (metric aggregation, explicit experiment identity, status no-write and acceptance rules);
- new dated training execution/report log;
- new `PROJECT_BRAIN/agent_logs/20260827_next_work_schedule.md` rather than rewriting the historical 2026-08-24 schedule;
- diagram index and sync/acceptance manifests.

The schedule will separate completed v6 acceptance, immediate follow-up, deferred items and Phase C pending status. No metric is copied from a log when it can be recomputed from a machine-readable artifact.

## Non-goals

- no change to the `E(k) → 6D tensor → Transformer/MBM → supervised heads → GUI` direction;
- no second downloader/store/training framework;
- no provider-label-conditioned feature construction;
- no outer-test-driven model selection;
- no multi-GPU refactor;
- no overwrite/deletion of v4/v5 raw/tensor/model assets;
- no credential persistence in packages, logs or reports.

## Late independent-review addendum

Two read-only independent audits completed after the initial plan was presented. Their high-impact claims were rechecked against the current disk before inclusion. The second audit's statement that server access was unavailable is superseded by the later successful password-authorized SSH inventory and GPU probe already recorded above.

### Additional verified scientific blockers

1. **The line-mode gap target has an exact zero-error algebraic baseline.** Loading `X_test` and computing `min(CBM_E)-max(VBM_E)` in float32 is element-for-element identical to all 6,019 saved targets. Baseline MAE/RMSE are exactly zero, while the learned soft-extremum head has MAE `0.000352150382 eV` and RMSE `0.001863863436 eV`. The head may remain useful for differentiable extremum localization, but its gap number is approximation error, not predictive skill. New reports must show the exact baseline and must not label this as independent DFT prediction.

2. **Outer-test procedural blindness is incomplete.** No outer sample enters normalization, gradients or callbacks, but the current script loads and prints outer-test shape/group/class statistics before fit and suggests rebuilding the split when classes are absent. There is no evidence that this changed the v5 run, but the training and final outer evaluation must be separated so training cannot inspect outer metadata before checkpoint freeze.

3. **SSL masking violates the declared actual-ratio gate and validation is stochastic.** Across 54 formal epochs, logged actual mask fraction averages `0.14387037`, is below `0.15` in 50 epochs, and ranges `0.133–0.156`; best epoch 39 reports `0.144`. Overlapping starts and segment truncation reduce coverage, and validation generates a fresh random mask each epoch, so early-stopping comparisons do not use fixed corruption.

4. **The SSL curvature-magnitude consistency term is dimensionally invalid.** It subtracts separately standardized explicit curvature channels from index-space second differences of separately standardized energy channels. With no physical k-coordinate supplied to the loss, these quantities do not share units or scale. The v6 run must not keep this magnitude term enabled as if it were physical; the scale-invariant curvature-sign loss can remain.

5. **Several post-hoc/augmentation paths are not segment-safe.** `_local_curvature` in the report and `PhysicsValidator` can cross branch boundaries. `VirtualStrainAugmentation` interpolates the whole concatenated index axis without segment IDs and is not a true lattice-strain transform. Until a structure/k-vector contract exists, v6 should disable this heuristic augmentation by default; post-hoc curvature must be made segment-aware.

6. **The OOD claim is only space-group OOD.** Group and material-ID overlap are zero and no exact tensor signature crosses train/test, but composition/prototype/source OOD is not established. Exact formula overlap covers 1,027/6,019 test samples; a no-spacegroup Pearson-symbol+site-count proxy covers 5,442/6,019. Reports and diagrams must use the precise term `space-group-disjoint outer OOD`.

7. **Provider-label/domain mismatch is material.** Outer provider-metal versus line-mode crossing mismatch is 1,257/6,019 (`20.8839%`). Type accuracy is about `0.98887` on feature/label-match samples and `0.78998` on mismatch samples. This is a global-provider-label versus sampled-line-path limitation, not permission to condition features on labels.

8. **Sample weighting hides group variability.** Sample-weighted outer accuracy is `0.947333`, while spacegroup-macro accuracy is about `0.939857`; the largest test group is about 20.42% of test samples. v6 reports must include group-macro metrics and clearly defer multi-seed/group-bootstrap uncertainty to the next schedule item.

### Additional reproducibility/artifact blockers

- Current Git state is `master` at `1ce9aa66847ca6ee63f004666a24125fc82bd6f9`, zero tags, and 76 status entries (`33 modified`, `2 deleted`, `41 untracked`, including this new audit log). It cannot be used as a reproducible sync source without classification and a secret-free canonical commit/tag.
- The historical 2026-08-25 transfer manifest no longer represents current core files; a new manifest/package is mandatory.
- v5 supervised artifacts contain only `best.weights.h5`; an independent epoch-60/last state is not recoverable locally or in the retained server project. The missing file must be recorded as a historical gap, not synthesized.
- The current 61 files across v5 model/checkpoint/report/log directories lack one consolidated final manifest. A retrospective read-only v5 inventory should be generated without rewriting historical artifacts.

## Plan amendments required before execution

1. Add a **Git reconciliation gate** before synchronization: classify all 76 status entries, verify exclusions/secrets, preserve user changes, and form a canonical core commit/tag bound to the new transfer manifest. Data/artifacts remain outside Git.
2. Split supervised execution into **train-only** and **evaluation-only** invocations within the existing script/pipeline. Train-only does not load outer arrays or metadata; it freezes and hashes best/last/accepted checkpoints. Evaluation-only requires that frozen manifest before loading outer test.
3. Save three distinct supervised states in v6: canonical best, final/last epoch, and portable accepted-restored-best. Verify each hash and loadability.
4. Extend SSL TDD scope to enforce actual mask fraction in the 15–30% contract, fixed deterministic validation corruption, weighted metric denominators, and `consistency_weight=0` for the dimensionally invalid magnitude term until a real k-coordinate contract exists.
5. Disable the current whole-path `VirtualStrainAugmentation` for v6; retain it only as a documented experimental heuristic until it becomes segment-aware and physically named.
6. Make report/validator curvature windows segment-aware and correct all parity/axis labels to distinguish exact line-mode tensor gap from provider global DFT gap.
7. Add exact algebraic baseline, group-macro metrics, mismatch-stratified metrics and raw uncertainty coverage to v6 reports.
8. Generate a consolidated retrospective v5 manifest and a companion record for the missing supervised-last artifact and historical download-report schema, without changing old result files.

No production source code has been modified during this audit.
