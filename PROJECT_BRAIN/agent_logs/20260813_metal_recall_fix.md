# 2026-08-13 Metal Recall Root-Cause Fix (Tensor + Classifier)

## Trigger

服务器基线训练（`20260813_server_migration_training.md`）得到带隙 OOD MAE=0.173 eV、R²=0.990，
但三分类 **metal 召回 = 0%**（38 条 metal 全部误判，36→indirect，2→direct）。
用户批准 A+B+C 全方案彻底修复。

## Root Cause (three layers, all confirmed with data)

1. **张量层（主因）**：`_select_vbm_cbm_indices` 对 metal 选费米面两侧的占据/空带边
   （通常是**不同带**）→ metal 的张量带隙（E_CBM−E_VBM）med=0.62 eV（假带隙），
   与半导体无法区分。
2. **分类器 topology 规则**：`topology_type = [0, 0.02+0.96·d, 0.98−0.96·d]`，
   **metal 通道概率恒为 0**，且 `topology_rule_weight=0.7` 压过 learned head（0.3）。
3. **类别不均衡**：metal 仅 153/1920 = 8%（class_weight=4.18 不足以对抗前两层）。

## Fixes Applied

### A. Tensor layer — `src/data/ood_tensor_builder.py`
- `_select_vbm_cbm_indices(..., is_metal=False)`：新增 is_metal 参数。
  metal 时返回**费米面穿过带**作为 VBM=CBM 同一索引。
- `_build_sample_tensor(..., is_metal=False)`：metal 时把两个 slot **锚定到费米能级**
  （vbm 平移使 max=E_F，cbm 平移使 min=E_F），曲率/k-distance（平移不变）保留 metal 色散物理。
- `process_band_data`：把 `is_metal` 传入 `_build_sample_tensor`。
- **验证**：metalfix 张量 metal 样本 tensor_gap **全部 = 0.000 eV**（train 153 + test 38），
  与 direct/indirect（med ~2.5 eV）清晰区分。6D 张量契约（形状）未变。

### B. Classifier — `scripts/finetune_supervised.py`
- topology 规则增加 **metal 分支**：`metal_rule = (predicted_gap < 0.15 eV)`，
  `topology_type[0] = 0.90·metal_rule`，半导体分支乘以 `(1−metal_rule)`；
  不再硬性把 metal 概率置 0。
- `topology_rule_weight` 0.7 → **0.3**（learned head 主导）。
- `metal_rule` 显式 `tf.reshape(..., tf.shape(direct_rule))` 修复广播维度
  （否则 type_pred 变 `(B,B,3)` 触发 rank mismatch）。

### 已知遗留（非本次范围，报告中标注）
- direct/indirect 仍有少数负 tensor_gap（min≈−56/−98 eV），为 stage0 遗留的
  VBM/CBM 选择异常样本；med 正常（+2.5 eV），不影响主体，留待后续数据质量清洗。

## Verification
- shape_check：topology_type 输出恒为 (B,3)（gap 为 (B,) 或 (B,1) 均通过）。
- metalfix 微调在服务器 CPU 上正常进入训练循环（无 cuDNN 崩溃，无 shape 崩溃）。

## Notes
- 张量契约改变（metal 带隙→0）后，本次微调复用了旧 SSL encoder 以快速验证分类修复；
  严格做法应在新张量分布上重训 SSL（Stage 3 一并处理）。
- 服务器端 `finetune_supervised.py` 的旧 cuDNN 临时 hack 已被最终代码覆盖
  （CPU 全量训练规避 cuDNN ABI 问题，代码本身不含 device hack）。
