# BandStructure AI Project — Dev Context

Last updated: 2026-08-28
Current root: `C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`
Branch: `v6-metricfix-20260827`
Pre-v6 audited snapshot commit: `025d5ce`
Verified metric-fix code commit: `3c00aef4468a617a5bceec5b7262037d025baa63`
Code tags: `v6-metricfix-code-20260827`, `v6-metricfix-sync-20260828`, `v6-metricfix-sync-r2-20260828`, `v6-metricfix-sync-r3-20260828`, `v6-metricfix-sync-r4-20260828`

## Current State

项目主链保持不变：

```text
完整 line-mode E(k)
  → (N,2,128,3) 6D tensor + segment IDs
  → Transformer / Masked Band Modeling
  → learned soft-extremum gap + metal/direct/indirect head
  → tkinter Plot-to-Physics
```

`aflow_noleak_v6_30k_seed42_metricfix` 已正式验收为 **latest accepted**：服务器 Tesla V100 GPU-only 完成 SSL 60 epochs + 监督 60 epochs（canonical aggregate `val_loss` best epoch 47），最终 artifact gate 与 model-brain manifest 已通过；产物回传（52 files / 74,071,297 bytes / SHA-256 `6735d815951b8081c114c11055ab8c5e122f1cc3ca466d0fc25c3145ef0a8184`）逐成员哈希匹配，本地模型加载/前向有限、概率和为 1，selection manifest 校验通过。

v6 正式指标（6,019 outer，独立重算）：learned line-mode MAE/RMSE `0.000407/0.007109 eV`（解析 baseline 0/0）、model-vs-global-DFT MAE `0.640244 eV`、type accuracy `0.958797`、Macro F1 `0.949251`、spacegroup-macro `0.965795`、mismatch accuracy `0.830549`、MC raw 95% coverage `0.803954`。

## Why v6 Is Required

v5 `aflow_noleak_v5_30k_seed42` 的字节资产仍完整，但监督模型选择被审计判定无效：

- custom `train_step/test_step` 返回当前 batch 裸标量，没有 epoch trackers；
- batch size=32，inner train/val=20,329/3,604，末批=9/20；
- 60 个 train accuracy 全是 `1/9` 倍数，60 个 val accuracy 全是 `1/20` 倍数；
- reported best epoch 52 / inner gap MAE 只代表最后 20 条 validation 样本；
- 已保存 artifacts 无法恢复每 epoch 的正确全 validation 指标。

此外：

- line-mode target 是输入函数 `min(CBM_E)-max(VBM_E)`；解析 baseline MAE/RMSE=0；
- v5 模型 learned soft-extremum MAE/RMSE=0.000352150/0.001863863 eV，是近似误差而非独立 DFT 预测精度；
- model-vs-global-DFT MAE=0.640281 eV；
- v5 type accuracy/Macro F1=0.947333/0.932648，spacegroup-macro accuracy=0.939857；
- outer provider-metal/line-mode mismatch=1,257/6,019，mismatch subset accuracy 约 0.78998；
- v5 MC raw 95% interval coverage 约 0.898，旧 1.0 是加入 0.5 eV tolerance 后的 diagnostic。

v5 保留为 `immutable historical run with invalid checkpoint selection`。61 个现存 v5 artifact 的统一哈希清单：

`PROJECT_BRAIN/transfer_manifests/v5_consolidated_artifact_inventory_20260827.json`

## Immutable Data Inputs

### v5 30k raw

- path: `data/raw/aflow/snapshots/aflow_30000_20260825/`
- HDF5 groups: 30,000；temporary groups: 0
- HDF5 SHA-256: `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`
- raw JSON: 30,000

### v5 processed tensor

- path: `data/processed/aflow/ood_tensors_v5_30000_seed42/`
- full shape: `(29952,2,128,3)`
- outer train/test: 23,933 / 6,019
- train/test spacegroups: 161 / 45；overlap=0
- full NPZ SHA-256: `9c5edea5e97f1c9b2561bc1c2324bcdd22741b5aea69e7d7c4c25116375a1ff7`
- split NPZ SHA-256: `c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`

v6 只读复用上述输入；`--fresh` 和 downloader/tensor builder 均不得删除、扩展或重建显式 raw/ood 输入。

### retained v4

- raw groups: 6,443
- raw SHA-256: `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`
- v4 model-brain manifest 的 `created_at` 在 2026-08-27 audit 中被历史 `--status-only` 副作用重生成；raw/tensor/model/checkpoint/predictions 未变化。
- incident: `PROJECT_BRAIN/transfer_manifests/v4_manifest_regeneration_incident_20260827.json`
- `run_full_pipeline --status-only` 现已 TDD 保护为 no-write。

## v6 Code Contracts Implemented Locally

### Supervised

- Keras stateful full-epoch trackers：gap loss/MAE、weighted type loss、accuracy、Macro F1、auxiliary terms；
- aggregate `val_loss` 是唯一 canonical checkpoint/early-stop monitor；
- `best.weights.h5`、`last.weights.h5`、accepted-restored-best 三态分离；
- `inner_selection_manifest.json` 在 outer access 前冻结三态 bytes/SHA；
- `--train-only` 不读取 outer arrays 或 outer sample manifest；
- `--evaluation-only` 在 outer load 前回验 selection manifest；
- SSL-only mask token、projection/reconstruction heads 在 supervised 阶段冻结；
- whole-path k-warp augmentation 默认 OFF；
- report/runtime config 使用 experiment/source/date/project-relative paths；
- analytic baseline、global DFT residual、sample/group/mismatch metrics 分 scope；
- parity 图不再把 line-mode target 标成 DFT gap；
- segment-aware post-hoc curvature；
- MC raw/tolerance coverage 分开，MC batch_size 真正有界。

### SSL

- actual span mask 每样本强制落在 15–30%，目标 25%；
- segment-safe span union；
- validation mask 由 `random_state + batch_index` stateless 固定；
- MSE/MAE 按 masked elements、mask fraction 按 positions、辅助项按 samples 加权；
- `ssl_history.json` 每 epoch 原子写入；
- dimensionally invalid curvature-magnitude consistency 默认 `0.0` 且不能被 adaptive scheduler 复活；
- curvature-sign loss保留；
- whole-path strain heuristic 默认 OFF；
- SSL encoder/reconstruction tensor 必须位于 GPU when `--require-gpu`。

### Pipeline

- explicit `--experiment-id` / `--raw-h5` / `--ood-dir` / `--report-date`；
- v4 默认 layout 保留用于历史复现；
- explicit v6 artifacts 与 v4/v5 隔离；
- immutable raw/OOD inputs 不下载、不清理、不重建；
- supervised 固定两阶段：train-only 后 evaluation-only；
- v6 required artifacts 新增 supervised best/last/selection manifest；
- `--status-only` no-write；
- required-artifact gate 继续 fail-closed。

## Local Verification So Far

- pre-v6 baseline：`92 passed in 414.21s`；
- 修复后 Stage-0：`118 passed in 26.13s`；
- 最新完整 WSL compileall + pytest：compileall exit 0；`136 passed in 197.67s`；
- local supervised smoke：1 epoch on RTX 4060，aggregate metrics、best/last/accepted、train-only outer isolation 全通过；
- local evaluation-only smoke：selection hash gate、64 outer samples、MC/report/plots、accepted hash unchanged 全通过；
- local SSL smoke：1 epoch，val actual mask≈0.25015、symmetry weight=0、best/last/history/model 全通过；
- 最终 compile/full pytest、immutable rehash 与 diff check 仍需在同步前重跑。

## Diagrams

当前：

- `PROJECT_BRAIN/diagrams/project_structure_diagram_20260827.html/.png`
- `PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260827.html/.png`

PNG 以 1600×1500 重渲染并完成视觉检查；主图、四卡片、footer 无裁切/明显重叠。20260824 v4 图保留为 historical snapshot。

## Remote Server Preflight

只读 preflight 已完成：

- Linux；24 logical CPUs；约 66 GB RAM；约 3.23 TB free；
- 2× Tesla V100-SXM2-16GB，盘点时均空闲；
- Conda env `bandstructure_gpu30k` 存在；
- TensorFlow 2.21，CUDA build 12.5.1，包含 sm_70；
- Conv1D forward/backward 位于 `/GPU:0`，gradients finite；
- remote v5 data/model/predictions 与 local hashes 一致；
- remote code 落后于本机当前 metric-fix branch；同步方向应为本地 canonical core → 服务器，不重传相同 v5 data/artifacts。

连接信息和密码不得写入项目或报告。

## Next Execution Steps

1. 完成 final full pytest、diff/line endings、immutable hash gate；
2. 形成 v6 metric-fix canonical commit/tag；
3. 生成 secret-excluded core sync manifest/archive；
4. staging 上传、逐文件回验、远端 compile/full pytest；
5. V100 一轮 SSL/supervised smoke；
6. 新 ID 下完整 SSL → supervised train-only → evaluation-only；
7. 回传 v6 models/checkpoints/logs/predictions/reports/manifests；
8. 本地重算 metrics、模型加载/forward、v4/v5 rehash；
9. 全部通过后才 promote v6，并更新 dated schedule/final report。

## Current Blockers / Deferred Scope

- v6 正式训练尚未完成；
- 只有 seed=42；3-seed 与 group-bootstrap 留到新日程；
- Phase C crystal structure schema、统一 k-path、multi-band target 与新 OOD 合同尚未冻结；
- Materials Project 正式双源仍受出口网络封禁；
- 当前模型仍是 E(k) analyzer，不是 structure→bands predictor。
