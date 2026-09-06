# BandStructure AI Project — Dev Context

Last updated: 2026-09-04
Current root: `C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`
Branch: `v7-60k-20260903`（P0 起由 `v6-metricfix-20260827` 改名，与 v7 状态对齐）
Pre-v6 audited snapshot commit: `025d5ce`
Verified metric-fix code commit: `3c00aef4468a617a5bceec5b7262037d025baa63`
v7 acceptance code commit: `ed783a8`
P0 reframing code commit: `a2b281c`
Code tags: `v6-metricfix-code-20260827`, `v6-metricfix-sync-20260828`, `v6-metricfix-sync-r2-20260828`, `v6-metricfix-sync-r3-20260828`, `v6-metricfix-sync-r4-20260828`, `v7-60k-final-acceptance-20260903`

## Current State

项目主链保持不变：

```text
完整 line-mode E(k)
  → (N,2,128,3) 6D tensor + segment IDs
  → Transformer / Masked Band Modeling
  → learned soft-extremum gap + 三任务分类 head（topology / provider / disagreement）
  → tkinter Plot-to-Physics
```

**`aflow_noleak_v7_60k_seed42` 为 latest accepted**（2026-09-03）：AFLOW 60,000 条（主通道 55,476 + 代理通道唯一 4,524 seed 42 抽样合并 = 60,000，metadata 60,000 ID 一致；张量 59,899 样本、101 条无边缘包络跳过已记录）。V100 GPU-only：SSL 50 epochs（early stopping）、监督 51 epochs（best epoch 31 by aggregate `val_loss`；best/last/accepted 冻结后 evaluation-only）。两次 GPU OOM（11,987 outer-test 全批量 reconstruct / extremum 热力图前向）已 TDD 修复（`4367f87`、`ed783a8`），全量回归 164 passed。产物回传 SHA-256 `9c46d1122be333ad903cff3eaab6fb802bc7f4c47ad3761051993b701d6cd45c` 校验一致，本地加载/前向通过，v4/v5/v6 immutable rehash 一致。

### P0 科研基准重构（2026-09-04 启动，进行中）

用户 P0 指令：line-mode gap MAE 退出主结果位；标签拆为三任务；3 seeds + group bootstrap + 错误分层；固定七类拆分；README 清理；分支改名。完整方案：`PROJECT_BRAIN/agent_logs/20260903_P0_scientific_reframing_plan.md`；宪法 4.13（§5 P0 科研基准条款）。

已完成（本地，commit `a2b281c` + `78174c2`，196 passed）：

- **P0A 三任务标签**：`derive_line_mode_topology` / `derive_provider_global_type` / `derive_line_global_disagreement` 纯函数（ood_tensor_builder）+ `scripts/derive_three_task_labels.py`（冻结张量后处理派生，不改 v4–v7 NPZ）。实测 59,899 全样本覆盖：line_mode_topology {metal 47,333 / direct 3,739 / indirect 8,827}；provider type {40,234 / 5,769 / 13,896}；disagreement {agree 52,481 / conflict 7,418}。产物 `three_task_labels.npz` + `three_task_labels_report.json` 已落盘 tensor 目录。
- **P0B 元数据回填**：`scripts/backfill_aflow_metadata_fields.py`（AFLUX `prototype()/species()/species_pp()/aflowlib_date()` 请求，207 页全量；非空不回退；合并后裁剪回 60,000 HDF5 ID）。实测 coverage 58,250/60,000（四个字段一致）。修复了一个 URL 缺 `?` 的 bug（三次后台失败根因，回归测试锁定）。
- **P0C 七拆分**：`src/data/benchmark_splits.py` + `scripts/build_seven_splits.py`；实测 random/space_group/composition/prototype/leave_element 各 47,919/11,980；source_protocol 59,359/540（aurl 目录分布极端偏斜：ICSD_WEB 34,798 / LIB3_WEB 24,630 / LIB1_WEB 567 / LIB2_WEB 5）；temporal 48,269/11,630（按 aflowlib_date 时间戳分位数，双峰年份分布使按年切不可行）。manifest `seven_splits_manifest.json` + 紧凑版 `seven_splits_test_ids.json`。
- **P0D 评估**：`src/evaluation/bootstrap_stratify.py`（spacegroup-level group bootstrap 95% CI、macro accuracy、错误分层四轴）；`scripts/evaluate_seven_splits.py`（冻结模型跨拆分三任务评估，分块前向）；`finetune_supervised.py` 内 `evaluate_three_tasks` 接入 metrics_summary 的 `primary_results` 段。
- **模型三 head**：`SupervisedBandGapModel` 增加 `topology_head`（3 类）+ `disagreement_head`（2 类）；type head 作为 provider head 保持纯净（`topology_rule_weight` 默认 0，规则先验只喂 topology head）；`make_tf_dataset` 支持三任务标签；`load_dataset` 从 sidecar `three_task_labels.npz` 按 material_id 对齐加载。smoke 测试用 `skip_mismatch` 兼容 v7 冻结权重。
- **P0E 文档**：README 折叠为单一「当前状态」+「历史里程碑」表；科学边界 7 条更新；宪法 4.13；dev_context 本段。
- 待办：服务器 3 seeds {42, 2024, 7} 全链训练（~12h V100）→ 冻结模型七拆分评估 + 3-seed 汇总报告 → 分支 `v6-metricfix-20260827` → `v7-60k-20260903`。

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

1. ✅ 服务器传输 P0 更新脚本与数据产物（litterbox 中转，MD5 双向校验一致）；
2. ✅ 3 seeds {42, 2024, 7} 全链训练完成（SSL + supervised + evaluation-only + 七拆分评估）；
3. ✅ 每 seed 冻结模型 `evaluate_seven_splits.py`（三任务 × 七拆分 + group bootstrap + 错误分层）；
4. ✅ 3-seed 汇总报告 `three_seed_aggregate.json` + `P0_scientific_reframing_report.md`；
5. ⏳ 分支已本地改名 `v7-60k-20260903`（远程推送与旧分支删除待 GitHub 凭据解除后执行）。

## Current Blockers / Deferred Scope

- GitHub 远程分支推送（本地改名已完成；远程待 GitHub 凭据解除）；
- Phase C crystal structure schema、统一 k-path、multi-band target 与新 OOD 合同尚未冻结；
- Materials Project 正式双源仍受出口网络封禁；
- 当前模型仍是 E(k) analyzer，不是 structure→bands predictor。
