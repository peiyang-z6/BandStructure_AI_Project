# 2026-08-22 Stage 1: 30000-Record Download + SSL Retrain on Metal-Fixed Tensors

## Scope

承接 20260813 阶段（2400 条 + metal 召回修复），本阶段：
1. 修复下载器元数据瓶颈，将 AFLOW 下载扩展到 30000 条；
2. 在 metal 修复后的张量分布上重训 SSL + 微调，验证 metal + direct 召回双双改善；
3. 打包传输到 V100 服务器，跑通完整训练流程。

## Download Bottleneck Fix (root cause)

- **现象**：`--target 30000` 卡死在候选查询阶段数小时，日志 0 字节。
- **根因**：`batch_download.py` 设 `limit = target*3 = 90000`，`fetch_metadata` 需分 bin 逐页
  拉取 90000 条 AFLUX 候选元数据（每页 500，数百页慢查询）；且 `response.read()` 对慢速
  AFLUX 响应无硬超时保护，雪上加霜。
- **修复**（`batch_download.py`）：`limit = max(100, min(target*2, 40000))`（候选池与下载
  缺口匹配，不再过度放大），`page_size` 固定 500（AFLUX 上限）。
- **修复**（`aflow_adapter.py`）：`fetch_metadata` 加逐页进度打印（`[AFLUX] bin X/5 page Y
  (bucket Z/quota)`），分阶段下载可监控。
- **提速**：`--workers 8`（worker 适配器 rate_limit_delay=0），下载速度 ~5 → ~15 条/分钟。

## Download Execution

- staged 循环：4000 → 8000 → 12000 → … → 30000（每阶段断点续传，候选池按当前缓存重建）。
- 进行中：当前缓存 ~2900 条，8 workers 稳定推进。

## SSL Retrain on Metal-Fixed Tensors (direct 修复验证) — RESULTS

- 动机：metal 费米锚定改变了张量分布（metal 带隙→0），旧 SSL encoder（在旧分布训练）
  与新分布失配，导致带隙回归精度下降（0.17→0.38 eV）。
- 操作：在 `aflow_ood_tensors_metalfix`（2400 条）上重训 SSL（GPU，60 epoch），
  val loss 收敛至 ~0.003（优于旧分布的 ~0.005），再用重训 encoder 微调（CPU）。

### 三次训练 OOD 测试对比（关键结果）

| 指标 | 基线（修复前） | v1（旧 SSL） | v2（重训 SSL） |
|---|---|---|---|
| metal 召回 | 0.000 | 0.711 | **0.974** |
| metal AUC | 0.836 | 0.928 | **0.933** |
| OOD 带隙 MAE | 0.173 | 0.381 | **0.248** eV |
| OOD R² | 0.990 | 0.893 | **0.968** |
| Macro F1 | 0.383 | 0.564 | 0.469 |
| direct 召回 | 0.253 | 0.253 | 0.253 |
| indirect 召回 | 0.888 | 0.643 | 0.643 |

### 结论
- **metal 修复 + SSL 重训彻底成功**：metal 召回 0%→97.4%，带隙回归回升（MAE 0.25, R² 0.97）。
- **新权衡**：metal 阈值 0.15 eV 偏宽松，indirect 召回降至 64%（部分 indirect 误判为 metal）；
  Macro F1 0.469。需校准 metal 阈值平衡三类。
- **direct 召回仍卡 25%**：VBM/CBM k 位置重合特征学习不足，是更深层问题（Stage 3 重点）。

## Direct Recall Root Cause (诊断，待 SSL 重训验证)

- direct 误判去向：108→indirect，41→direct，13→metal（n=162）。
- 关键：误判样本带隙分布（med 3.26 eV）与 direct 整体（med 3.07 eV）几乎重合——
  **带隙大小不是区分 direct/indirect 的有效特征**，区分关键是 VBM/CBM 是否同 k 点（拓扑特征）。
- 当前 topology 规则已含此特征但权重仅 0.3，learned head 对拓扑特征学习不足。
- 这是模型能力问题，需重训 SSL 让 encoder 学到拓扑区分度；属 Stage 3 算法调优范畴。

## Known Issues (本阶段遗留)

- 下载全程 ~30h（8 workers），需后台持续运行。
- SSL 重训后的微调结果已收集：metal 召回 97.4%、带隙 MAE 0.248 eV；direct 召回仍 25%，
  Macro F1 因 metal 阈值偏宽松降至 0.469。
- MP 数据源仍不可用（WSL IP 封 + 服务器 S3 被墙），单源 AFLOW 走完本阶段。
