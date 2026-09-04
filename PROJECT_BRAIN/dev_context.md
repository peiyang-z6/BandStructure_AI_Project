# BandStructure AI Project — Dev Context

Last updated: 2026-09-03
Current root: `C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`
Branch: `v6-metricfix-20260827`
Pre-v6 audited snapshot commit: `025d5ce`
Verified metric-fix code commit: `3c00aef4468a617a5bceec5b7262037d025baa63`
v7 acceptance code commit: `ed783a8`
Code tags: `v6-metricfix-code-20260827`, `v6-metricfix-sync-20260828`, `v6-metricfix-sync-r2-20260828`, `v6-metricfix-sync-r3-20260828`, `v6-metricfix-sync-r4-20260828`, `v7-60k-final-acceptance-20260903`

## Current State

项目主链保持不变：

```text
完整 line-mode E(k)
  → (N,2,128,3) 6D tensor + segment IDs
  → Transformer / Masked Band Modeling
  → learned soft-extremum gap + metal/direct/indirect head
  → tkinter Plot-to-Physics
```

**`aflow_noleak_v7_60k_seed42` 为 latest accepted**（2026-09-03）：AFLOW 60,000 条（主通道 55,476 + 代理通道唯一 4,524 seed 42 抽样合并 = 60,000，metadata 60,000 ID 一致；张量 59,899 样本、101 条无边缘包络跳过已记录）。V100 GPU-only：SSL 50 epochs（early stopping）、监督 51 epochs（best epoch 31 by aggregate `val_loss`；best/last/accepted 冻结后 evaluation-only）。两次 GPU OOM（11,987 outer-test 全批量 reconstruct / extremum 热力图前向）已 TDD 修复（`4367f87`、`ed783a8`），全量回归 164 passed。产物回传 SHA-256 `9c46d1122be333ad903cff3eaab6fb802bc7f4c47ad3761051993b701d6cd45c` 校验一致，本地加载/前向通过，v4/v5/v6 immutable rehash 一致。

v7 正式指标（11,987 outer，55 spacegroups，overlap=0）：line-mode MAE/RMSE `1.54e-05/5.54e-04 eV`（解析 baseline 0/0）、model-vs-global-DFT MAE `0.3767 eV`、type accuracy `0.9251`、Macro F1 `0.8788`、spacegroup-macro `0.9296`、mismatch accuracy `0.7441`（1,723 条）、MC raw 95% coverage `0.7974`。回归优于 v6、分类略降（60k 尾部样本更难），如实报告。

v6 正式指标（历史 accepted，6,019 outer）：learned line-mode MAE/RMSE `0.000407/0.007109 eV`（解析 baseline 0/0）、model-vs-global-DFT MAE `0.640244 eV`、type accuracy `0.958797`、Macro F1 `0.949251`、spacegroup-macro `0.965795`、mismatch accuracy `0.830549`、MC raw 95% coverage `0.803954`。

### Phase 6 启动

Phase 6 启动：完成环境固化、GUI 状态持久化、CV 不确定性估计，并建立文献挖掘 Pipeline 原型。

- **P0 环境固化**：`requirements.txt` 全部核心依赖 `==` 实测锁定（tensorflow==2.21.0, keras==3.15.1, mp-api==0.46.4, emmet-core==0.87.1 等 27 项，与 v6 验收运行时一致）；`environment.yml` = `conda env export --no-builds` 全量快照（27 conda + 120 pip）。验证：pin 与 pip freeze 逐项一致、`pip check` 干净。
- **P1 GUI 状态持久化**：`gui_workbench.py` 内 `WorkbenchStateStore`（`data/annotations/workbench_state/`，图像内容 SHA-256 寻址，原子写）；标注/定标/材料信息 800ms 防抖自动保存 + 换图/关闭保存 + 同图重开自动恢复。宪法 §9（tkinter only）下选择后端 JSON 缓存分支。
- **P2 CV 置信度**：`multi_format_parser.py` 聚合 `cv_quality`（0–1 分 + green/yellow/red；detector 权重缺失时 `optional_missing_score_excluded`）；`brain_invoker.py` `compute_brain_uncertainty`（熵 + 极值峰锐度 + gap 合理性；明确不用 MC-Dropout）；GUI 右侧质量指示灯 + 黄/红警告文案。
- **P3 文献挖掘**：`literature_mining_pipeline.py` 原地升级——PDF 页内图 + PNG/JPG/JPEG/BMP 直接遍历；记录含 CV/脑置信度；h5 attrs `cv_confidence`；《文献挖掘摘要报告》统计成功提取率、平均置信度、潜在 Direct Gap 材料数量。输出默认 `data/raw/experimental/experimental_bands.h5`（宪法 §3 兼容，不建根级 `data_cache/`——用户已批准）。
- **附带**：latest-model 指针（brain_invoker 默认路径 / GUI t-SNE / smoke_latest_model）v5→v6 并更新测试期望。
- TDD：新增 `tests/test_workbench_state.py`（7）、`tests/test_cv_confidence.py`（6）、`tests/test_literature_mining.py`（5），全部 RED→GREEN。

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
