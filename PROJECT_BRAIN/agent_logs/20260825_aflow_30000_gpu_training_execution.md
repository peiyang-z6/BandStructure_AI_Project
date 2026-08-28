# 2026-08-25 — AFLOW 30,000 + GPU-only Formal Training Execution

Status: completed

## User-authorized scope

1. 将 AFLOW canonical cache 从 6,443 扩展到总计 30,000 个可用 line-mode E(k) HDF5 groups；
2. 使用有界多线程下载；
3. 依照既有 no-leak 合同重建 6D tensor 和 group-disjoint OOD split；
4. 选择性打包核心代码与全部下载数据，传到既有 NVIDIA V100 服务器；
5. SSL 与 supervised 均必须使用 NVIDIA GPU，禁止 CPU fallback；
6. 回传真实模型、checkpoint、日志、预测、指标和 provenance；
7. 生成任务完成当日训练报告。

## Frozen experiment identity

- New experiment: `aflow_noleak_v5_30k_seed42`
- Previous accepted baseline: `aflow_noleak_v4_seed42`
- v4 在 v5 完成全链验收前不得覆盖或删除。

## Preflight facts

- Current canonical HDF5 groups: 6,443; temporary groups: 0.
- Current raw JSON responses: 6,443.
- Current metadata candidate records: 10,149.
- Current exclusion registry: 1,066.
- Local free disk: 515.73 GiB.
- Full regression gate after GPU fail-fast changes: 42 passed.
- AFLOW live probe: metadata reachable; usable payload parsed as `(50,180)`, canonical `E_F=0`, source SHA-256 present.
- Concurrency harness: 4 workers reached concurrency 4; all HDF5 writes occurred only on the caller/main thread; exact target count preserved.

## Download contract

- Staged total targets: 8,000 → 12,000 → 16,000 → 20,000 → 24,000 → 27,000 → 30,000.
- Bounded workers: begin with 12; do not use concurrent HDF5 writers.
- Thread-local adapters perform network fetches; the main thread serializes HDF5 commits.
- Persist every usable raw response and exclusion reason; target means final cached groups, not additions.
- AFLOW supplemental and HDF5 variants remain isolated and are not silently mixed into canonical training input.

## Tensor/science contract (unchanged)

- AFLOW feature reference `E_F=0`; absolute Efermi provenance only.
- Provider labels are target/audit only.
- Label-free occupied/empty envelopes, PCHIP, segment-aware curvature/crossing.
- Tensor `(N,2,128,3)` plus segment IDs.
- Outer split spacegroup 80/20, seed 42, group overlap zero.
- Outer test is final-evaluation-only.

## GPU-only gate

- `train_ssl.py --require-gpu` already existed.
- `finetune_supervised.py --require-gpu` was added through RED→GREEN tests.
- `run_full_pipeline.py --require-gpu` now propagates to supervised fine-tuning as well as SSL.
- Historical environment is not accepted merely because GPUs are visible: its TensorFlow 2.21 build targets CUDA 12.5.1 while Python CUDA packages are 12.9 and cuDNN is 9.24; V100 Conv1D hangs at first forward pass.
- A separate environment will be created and accepted only if an explicit `/GPU:0` Conv1D forward+backward probe returns finite gradients, followed by 1-epoch real SSL and supervised smokes.
- Any GPU probe failure aborts formal training; CPU fallback is forbidden for this run.

## Package / transfer exclusions

Exclude reference material, secrets, `.git`, caches, legacy models/checkpoints/reports/logs, optional vision datasets, transfer archives and temporary scripts. Include current source, scripts, tests, safe configs/templates, canonical raw HDF5+metadata+reports/exclusions, and the newly built formal processed tensor+manifest.

## Corrective action recorded during execution

- The first staged downloader invocation appended to `data/raw/aflow/aflow_bands.h5`, which temporarily changed the byte hash referenced by the v4 relocation audit. This was detected by the full regression test once the active file had passed 16,000 groups.
- The running download was not interrupted because moving an HDF5 file during serialized commits would risk corruption.
- The exact immutable 6,443-group v4 file was recovered from the previously verified server mirror into `data/raw/aflow/provenance/baselines/aflow_bands_6443_bb261f1e.h5`.
- Recovered baseline evidence: 587,757,153 bytes; SHA-256 `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`; 6,443 groups; zero temporary groups.
- After the downloader exits, the expanded HDF5/metadata/JSON/exclusion/report set will be moved into a dedicated immutable 30k snapshot. The exact recovered v4 HDF5 will then be restored to the historical canonical path before the full test suite is rerun.
- No downloaded records were discarded and no v4 model/checkpoint/report was overwritten.
- At the 20,000-group stage, the downloader recorded 4,000 successes, 2,035 line-mode-unavailable records and 10 transient failures from 6,045 attempts; cumulative permanent exclusions reached 7,400. This measured yield proved that the old 40,000-candidate cap could not support 30,000 usable records.
- A RED→GREEN regression changed the bounded metadata policy from a 40k hard cap to `min(2 × target, 80k)`, so target 30k queries 60k candidates. New Python stage processes use the corrected policy without altering HDF5 serialization.
- A second live constraint appeared: equal 5-bin quotas stop early when high-gap bins are exhausted and do not reallocate unused quota. The final resume therefore uses one `[0,5] eV` pool (`gap_bins=1`) only after the first >21k records have already preserved stratified high-gap coverage; final train/test class distributions remain acceptance gates.
- The isolated server environment `bandstructure_gpu30k` passed an explicit V100 Conv1D forward/backward probe with CUDA 12.5.82 and cuDNN 9.3.0.75. Keras 3 `jit_compile='auto'` was separately proven to invoke an incompatible V100 XLA cuDNN autotuner; supervised compile now sets `jit_compile=False` through RED→GREEN tests and completed a real 1-epoch GPU smoke.
- Supervised reports no longer use the hard-coded `20260604` filename or escape the experiment directory; `--report-date` now freezes the task date, and `--fresh` no longer deletes sibling experiment reports.

## Completed data snapshot evidence

- Final download completed `2026-08-25T21:41:25+08:00`: 30,000 HDF5 groups, 30,000 raw JSON, zero temporary groups.
- Final resume stage: 7,307 success, 5,110 line-mode unavailable, 64 transient failures; 22,693 were already cached; cumulative permanent exclusions 14,141.
- Immutable raw HDF5 SHA-256: `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`; 30k JSON hash-manifest SHA-256: `d90e6f9fd1289207ad00195f105bb34895186cbd522e1596669634084dc48625`.
- v4 baseline restored byte-identically: 6,443 groups, 6,443 JSON, HDF5 SHA-256 `bb261f1e…b62d`, metadata SHA-256 `daf66250…eff`.
- Tensor builder produced 29,952 valid samples and skipped 48 with audited reasons; full/split SHA-256 are `9c5edea5…1ff7` / `c99d2164…b7a5`.
- Outer split train/test 23,933/6,019; groups 161/45; overlap 0; all three target classes exist in both splits.
- The first tensor wrapper used obsolete flag spellings `--output-dir/--target-k-points`; raw snapshot had already completed safely. The builder was resumed with the actual CLI `--output/--target-k`; tensor/hash validator passed. A final wrapper test name was stale, then the two actual v4/v5 immutability tests passed (`2 passed`).

## Completion evidence

- Local transfer archive：30,069 members；4,526,311,026 bytes；SHA-256 `5f9ea0d4683786db93c2422115687794ab1057f463478c2598669984d977a007`。服务器逐 member hash verification mismatch=0 后原子发布为 `BandStructure_AI_30k_20260825`。
- GPU environment：TensorFlow 2.21、CUDA runtime/NVCC 12.5.82、cuDNN 9.3.0.75；V100 Conv1D forward/backward 与 finite gradients 均在 `/GPU:0`；CPU fallback 禁止且未发生。
- SSL：best epoch 39，stopped epoch 54；best val masked MSE/MAE 0.384270/0.395004；best/last/restored model 已保存。
- SSL reload smoke 首次因未导入 Keras registered `SSLEncoder` 而 rc=1；同一磁盘模型补入注册 import 后在 GPU finite forward 通过，未重训 SSL。
- Supervised fit：inner train/val 20,329/3,604，完整 60 epochs；best inner gap epoch 52，MAE 0.000341 eV。训练后 t-SNE 一次性编码 29,952 样本导致 OOM；RED→GREEN 增加 batch_size=128 latent extraction 与显式 `--evaluation-only`。随后从 best checkpoint 恢复，history rows=60、fit_rerun=False，未覆盖训练 CSV 或 checkpoint。
- Final outer OOD (6,019)：line-mode tensor-gap MAE/RMSE 0.000352/0.000595 eV；model-vs-global-DFT gap MAE 0.640281 eV；type accuracy 0.941186；Macro F1 0.932648；direct-gap recall 0.943662。line-mode 与 global DFT 指标在报告中分开命名。
- GPU monitor：145 samples；GPU0 max utilization 89%，max memory 15,114/16,384 MiB。accepted supervised model 从磁盘重载后 gap/type outputs 均在 `/GPU:0` 且 finite。
- Server artifact manifest：50 files、75,143,804 bytes，逐文件 mismatch=0。Results archive：63 members、214,984,160 bytes、SHA-256 `704468f9ff22bc8d43c92bd6d0b5511f1a6cf827e746cc9220a4da77df83030f`。
- Local import：55 new + 8 byte-identical reused；62 content entries 再次逐 SHA 验证通过。结果归档保存在 `artifacts/reports/aflow_noleak_v5_30k_seed42/archives/`。
- Final WSL compileall/pytest：`55 passed in 213.38s`；此前随机 test flake 经 10-run RED 复现并改为 deterministic logits，随后 10/10 通过。
- `tests/smoke_latest_model.py` 已切到 v5 portable paths；2 条 outer OOD 推理 gap/type finite、probability sums=1。`PhysicsBrainInvoker` defaults 与 GUI t-SNE 同步切到 v5。
- 正式报告：`artifacts/reports/aflow_noleak_v5_30k_seed42/latest_training_report_20260825.md`。任务于 2026-08-25 启动，训练/报告跨午夜完成，报告保留任务日期并记录实际生成时间。
- Phase C 仍为 pending；本任务未实现 crystal graph→multi-band sequence。
- Remote cleanup：确认无训练进程后终止唯一 completed/sleep screen；删除旧 `~/BandStructure AI` mirror 和 26 个 `.cache` 传输/烟测临时对象；保留 `~/BandStructure_AI_30k_20260825` 与 `bandstructure_gpu30k` 环境。
- Security cleanup：远端 ephemeral SSH public key 删除 1 条、同 tag remaining=0，其他 8 条 authorized keys 保留；新连接确认该 key 无法认证。Windows Temp 删除 41 个任务临时文件（含私钥、公钥、4.5 GB transfer archive），释放 4,529,323,910 bytes；项目内 results archive/manifests 保留。

## 2026-08-27 correction note

本节保留上方 2026-08-25/26 原始执行记录，但以 2026-08-27 对磁盘 JSON/CSV/predictions/log 的只读重算纠正其叙述性漂移：

- SSL epoch 39 正确 val masked MSE/MAE 是 `0.384270 / 0.156710`（标准化六特征），不是 0.395004；
- reported supervised best gap MAE 是 `0.000348125 eV`，但旧 epoch metric 实际只来自最后 20 条 validation 样本，因此 epoch 52 selection 被判定无效；
- outer learned line-mode MAE/RMSE 是 `0.000352150 / 0.001863863 eV`；
- type accuracy/Macro F1 是 `0.947333444 / 0.932648282`；
- GPU monitor 是 225 samples，不是 145；
- line-mode target 的解析 input baseline MAE/RMSE 为 0；model-vs-global-DFT MAE 为 0.640281 eV；
- v5 supervised 独立 final/last checkpoint 不可恢复；现存 best 与 accepted-restored-best 保留；
- 现存 v5 四类 artifact 为 61 files，统一 retrospective manifest 见 `PROJECT_BRAIN/transfer_manifests/v5_consolidated_artifact_inventory_20260827.json`；
- v5 状态改为 immutable historical run with invalid checkpoint selection；不得继续把旧表格视为当前无保留验收结论。

完整审计与 v6 修复方案见 `PROJECT_BRAIN/agent_logs/20260827_training_result_reaudit_and_retrain_plan.md`。
