# BandStructure AI Project — Dev Context

Last updated: 2026-09-08
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

**差异化方向（2026-09-06）**：跨模态"晶体结构 — 数值能带 — 论文/实验能带图像"模型，用于检索/匹配/可信拒识/主动 DFT 闭环。P1→P5 顺序见宪法 5.0 §8。P0（科研基准重构）与 P1 结构 sidecar 已完成；P2 未验收，P3 正按用户限域授权进行探索性纠错和对照训练，详见宪法 5.1 §8。

**`aflow_noleak_v7_60k_seed42` 为 latest accepted**（2026-09-03）：AFLOW 60,000 条（主通道 55,476 + 代理通道唯一 4,524 seed 42 抽样合并 = 60,000，metadata 60,000 ID 一致；张量 59,899 样本、101 条无边缘包络跳过已记录）。V100 GPU-only：SSL 50 epochs（early stopping）、监督 51 epochs（best epoch 31 by aggregate `val_loss`；best/last/accepted 冻结后 evaluation-only）。两次 GPU OOM 已 TDD 修复（`4367f87`、`ed783a8`），全量回归 199 passed。产物回传 SHA-256 校验一致，本地加载/前向通过，v4/v5/v6 immutable rehash 一致。

### P0 科研基准重构（已完成，2026-09-06）

用户 P0 指令：line-mode gap MAE 退出主结果位；标签拆为三任务；3 seeds + group bootstrap + 错误分层；固定七类拆分；README 清理；分支改名。宪法 4.13 §5（已并入 5.0）。

- **三任务标签**：`derive_line_mode_topology` / `derive_provider_global_type` / `derive_line_global_disagreement`（ood_tensor_builder）+ `scripts/derive_three_task_labels.py`（冻结张量后处理，不改 v4–v7 NPZ）。59,899 全样本覆盖；disagreement conflict 7,418（12.4%）。
- **元数据回填**：`scripts/backfill_aflow_metadata_fields.py`，prototype/species/species_pp/aflowlib_date 覆盖 58,250/60,000。
- **七拆分**：`src/data/benchmark_splits.py` + `scripts/build_seven_splits.py`，全部 seed 42、group-disjoint、manifest 落盘。
- **评估**：`src/evaluation/bootstrap_stratify.py`（spacegroup 级 group bootstrap 95% CI + 错误分层四轴）；`scripts/evaluate_seven_splits.py`（冻结模型七拆分评估）；`scripts/aggregate_three_seed_results.py`（3-seed 汇总）。
- **模型三 head**：`SupervisedBandGapModel` 增 topology_head（3 类）+ disagreement_head（2 类）；provider head 去规则先验。
- **3 seeds {42, 2024, 7} 全链**（服务器 V100）完成。主结果（outer OOD 11,987，3-seed 均值）：line_mode_topology `0.9932±0.0016`、provider_global_electronic_type `0.9371±0.0062`、line_global_disagreement `0.9666±0.0016`。
- **分支改名**：`v6-metricfix-20260827` → `v7-60k-20260903`（本地；远程推送待 GitHub 凭据解除）。
- 全量回归 **199 passed**。产物见 `artifacts/reports/aflow_noleak_v7_60k_seed{42,2024,7}/`。

### 方向调整（2026-09-06，用户批准改宪法）

按 `Desktop/questions and directions.md` 调整开发方向。审计结论：文档四处"164 passed / 分支 v6"已过时（P0 后为 199 passed / 分支 v7-60k-20260903）；短板 4（HDF5 缺结构字段）属实；P1 结构 sidecar 的 AFLOW REST 端点实测可达（`?geometry`/`?positions_fractional`/`?species`/`?dft_type`/`?spin_cell`/`?files`，`?lattice` 404）。宪法升 5.0。完整审计：`PROJECT_BRAIN/agent_logs/20260906_direction_reframing_audit.md`。

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

## P3 本轮实证状态（2026-09-08）

- 排序 OT 旧 run 的 60 epochs 已完成；旧报告 train/test sorted_band_mae 为 2.7861583315 / 3.7084333312 eV。旧 gap 实现和训练目标有缺陷，不作科学验收结论。
- 已回传旧模型、报告、对应代码和训练日志；归档 SHA 双端一致，见 `artifacts/reports/p3_audit_20260908/legacy_inventory.json`。
- 本轮复现 padding 改写 VBM、deep-valence/high-conduction 掩盖 crossing、丢失跨越 band、重复 k 距离跨分支插值、缺边缘误标金属、NaN loss 被置零、Keras 重载预测变化等问题，并逐项回归修正。
- outer-train 固定抽样 1,000 个材料中，旧选择器对 329 个漏掉全部 EF-straddling band；这是抽样数据诊断，不能外推为全量金属比例。
- 原地新增 `attention_layers=0/2` 的 MLP/k-attention 对照，支持 segment attention mask；两版 Keras 重载预测一致性测试通过。
- 修正版数据独立保存到 `p3_multiband_v2_20260908/`，旧 P3 及 v4–v7 资产不覆盖。核心六文件与两个入口均完成独立复审，当前源文件 SHA 与审查快照一致；本地及服务器完整回归均为 **456 passed**。
- 本地 GPU 与服务器两张 V100 的真实晶体端到端 smoke 均通过：重载差均为 0，smoke holdout ID/group overlap 均为 0，权重/预测/凭据 SHA 已读回。服务器 headless、OpenCV/历史资产缺项、conda 激活问题已分别实证修复；未跳过测试，未启用 CPU fallback。该结果不代表正式 outer 精度。
- checkpoint 保存模型/优化器/epoch 并实测恢复，但无 CLI resume，非空 run 目录不可覆盖。训练侧原子数最大为 50，容量审计不代替新版全量质量与覆盖统计。
- 受控流程已真实启动：独立全量 prepare → 两张 GPU 上同数据、seed 42、batch 32、最多 180 epochs 的 MLP/两层 attention 对照 → 双方冻结后 outer 评估。完成凭据/中性 final commit 未发布前不放行训练；本轮最终 valid/exclusion 计数与精度尚待结果。
- 纠错依据见 `agent_logs/20260908_P3_reaudit_attention_execution.md`；最新服务器验证、失败保留与运行合同见 `agent_logs/20260908_P3_remote_controlled_execution.md`。远端源码快照在流程运行中保持冻结，本地文档更新不直接覆盖该快照。

## Next Execution Steps

P0、P1 已完成。P2 检索未验收。P3 按宪法 5.1 的限域授权进行探索性实现与对照训练；不能把该授权写成 P2/P3 科学验收通过。

1. ✅ **P1 结构 sidecar 补全**：`aflow_structure_sidecar.json`（60,000 记录，ID 与 HDF5 完全对齐）。详见 `agent_logs/20260906_P1_structure_sidecar.md`；
2. ⚠️ **P2 跨模态检索（未验收）**：脚手架完成（CGCNN + 等变编码器 + InfoNCE + 检索指标）。两版正式训练均未达标：v1 纯 TF CGCNN（train recall@10=0.39 / test recall@1=0.0035 过拟合）、v2 e3nn 等变 + 增强描述子（train recall@10=0.0526 / test recall@10=0.0054，无 OOD 泛化）。原因尚待系统诊断：单一 cosine mean 不能证明或排除 collapse，也不能独自定位 InfoNCE 退化根因。P2 检索留作后续优化；
3. 🔄 **P3 variable multi-band decoder（进行中、未验收）**：最新状态见 `agent_logs/20260908_P3_remote_controlled_execution.md`。独立复审、本地/服务器 456 项完整回归及双 V100 端到端验证已通过；新版全量数据流程已进入 MLP/自注意力受控训练，双方冻结后才做 outer 评估。尚无本轮最终精度，正式 Bandformer 对照与完整物理指标仍未完成；
4. P4 校准不确定性 + 主动获取；
5. P5 外部验证集（MP/JARVIS source-OOD + 论文/ARPES 图像 + 新 DFT blind test）；
6. ⏳ 分支远程推送（本地已改名 `v7-60k-20260903`；待 GitHub 凭据解除）。

## Current Blockers / Deferred Scope

- GitHub 远程分支推送（本地改名已完成；远程待 GitHub 凭据解除）；
- P2 检索未达标，现有结果未证明 OOD 泛化；根因仍待系统复核，不能仅凭 InfoNCE loss 或单一 cosine 统计认定退化解。P3 按宪法 5.1 限域授权探索，不代表 P2 验收通过；
- P1 sidecar 39 个 `ICSD_WEB/HEX` 材料结构字段 AFLOW 端点 HTTP 500（源端缺陷，已按宪法 §4 标记 missing_fields，不伪造）；
- Materials Project 正式双源仍受出口网络封禁；
- P3 旧选带比例与金属比例受已证实的 crossing 丢失缺陷影响，不能用于修正版结论；以新 schema=2 prepare_report 的实测覆盖、容量截断与 exclusion 数为准。
