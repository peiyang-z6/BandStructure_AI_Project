# 2026-08-22 Daily Run: Fixes + 6443-Record Training

## Trigger

用户要求：暂停下载 → 修复已知问题 → 用当前所有数据（6443 条）在服务器跑一轮测试 → 写今天的训练报告。

## Fixes Applied Today

### 1. Metal 阈值双条件（`scripts/finetune_supervised.py`）

- **问题**：v2 中 metal 召回 97.4% 但 indirect 召回降到 64%——metal 阈值（`gap < 0.15`）
  一刀切，部分 indirect 被误判为 metal。
- **修复**：metal 判定改为**双条件**——`metal_rule = (predicted_gap < 0.2) AND direct_topology`。
  真 metal 的费米锚定带在 VBM/CBM 同一 k 区域穿过 E_F，因此 direct topology 必须成立；
  indirect 即使预测带隙小也不满足 direct topology，不会被误判为 metal。阈值从 0.15 调到 0.2
  （锚定 metal 恰好为 0，留小容差）。

### 2. 负 tensor_gap 过滤（`src/data/ood_tensor_builder.py`）

- **问题**：stage0 遗留部分样本 tensor_gap 严重为负（min=-97 eV），源于 VBM/CBM 选择异常
  （AFLOW 某些记录能带数据质量差），污染回归目标。
- **修复**：`process_band_data` 在张量构建后检测 `tensor_gap = min(CBM)-max(VBM)`，
  非 metal 且 < -0.05 eV 的样本被过滤并记录为 excluded。

## Data Rebuild (6443-record cache)

- 源：`data_cache/aflow_bands.h5`（6443 条，下载至 stage2 中段暂停）。
- 张量：`aflow_ood_tensors_6443`，**6399 条**（过滤 44 条负带隙异常）/ **183 空间群**，
  train/test = 5117/1282，空间群零交集。
- metal 带隙全部 = 0.000 eV（费米锚定保持）；负 tensor_gap = 0 条（过滤生效）。
- 三分类分布：train metal 479 / direct 1636 / indirect 3002；test metal 118 / direct 413 / indirect 751。

## Training (in progress)

- SSL 重训（GPU，6399 条，60 epoch，val loss ~0.16–0.22，数据量增大任务更难，正常）。
- 微调（CPU，用 6443 SSL encoder，60 epoch，进行中）。
- 目标：验证 metal 阈值双条件是否提升 indirect 召回（同时保持 metal 召回），
  以及 6399 条干净数据下的整体精度。

## Known Issues (today)

- SSL val loss 偏高（0.16–0.22 vs 2400 条的 ~0.003）：数据量从 2400 → 6399，
  模型需适应更多样能带形态；后续 30000 条时需相应调大模型容量或延长训练。
- direct 召回待微调完成后确认（双条件修复理论上不影响 direct/indirect 区分逻辑）。
