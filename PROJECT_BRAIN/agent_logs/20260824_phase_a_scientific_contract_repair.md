# 2026-08-24 — Phase A Scientific Contract Repair

## Trigger

模型交接审计确认：旧 tensor builder 使用 provider `is_metal` 改变输入（费米锚定），同时以该标签作为分类 target，构成 target-conditioned feature 风险；AFLOW `bands_data` 能量已相对 E_F=0，但旧 HDF5 保存的是绝对 `Efermi`；曲率和 crossing 判断未隔离 k-path segments。

## Root Causes

1. AFLOW `bands_data` 是 `E-E_F` 坐标，原始 `Efermi` 仅为绝对源值；混用参考系使大量非金属被误判穿越。
2. 全路径 band min/max 会跨不连续 high-symmetry segments，制造假 crossing。
3. provider metal label 被用于锚定输入，导致分类 target 改变 features。
4. 两条固定 band 在 crossing/重排序情况下可产生极端负 tensor gap。
5. legacy 部署 gate 默认开启，仅适用于旧标签锚定张量，不适用于无泄漏表示。

## Implemented Fixes

### A1 — Label-independent features

- `AFLOWAdapter._parse_band_payload`：canonical `efermi=0.0`；保留 `source_efermi_absolute` 和 `energy_reference=fermi_shifted_zero`。
- builder 对旧 AFLOW HDF5 同样规范化特征参考为 0 eV。
- `is_metal` 仅做 target/audit，不再进入 `_build_sample_tensor`。
- line-mode crossing 由 E(k)+E_F、并在单个 segment 内推断；只做 audit，不改变 features。
- VBM/CBM 特征改为逐 k 点 occupied/empty edge envelopes；PCHIP 重采样并限制在 E_F 两侧，天然非负 tensor gap。

### A2 — Segment-aware physics

- 为 raw/resampled k-path 构建 `segment_ids`。
- `_curvature` 局部拟合禁止跨 segment；边界不足 3 点时不跨段回退。
- NPZ 新增 `segment_ids_train/test`（shape N×128），样本 manifest 保存 symmetry labels + segment IDs。

### A3 — Manifest and audit

- `metal_feature_audit`：crossing count、provider label mismatch count/IDs。
- `distribution_audit`：composition/prototype overlap、train/test source distribution。
- `provenance`：source H5、metadata、builder、full/split NPZ SHA-256 与生成时间。
- formula/pearson/num_sites 写入样本 metadata。

### Legacy gate scope

- `ExtremumExpectedGapHead` 的 Fermi-anchor gate 改为默认关闭。
- 只有 legacy 锚定快照显式 `--enable-metal-anchor-gate` 才启用。
- 新无泄漏快照不使用该 gate。

## Final Phase A Snapshot

`data_cache/aflow_ood_tensors_6443_noleak_v4/`

- 6441/6443 valid samples；2 条因无法在所有 k 点构造 occupied/empty envelope 被隔离。
- 183 groups；train/test=5153/1288；spacegroup overlap=0。
- classes train metal/direct/indirect=476/1655/3022；test=121/410/757。
- tensor gap < -0.05 eV：0。
- exact-zero gap：train 37、test 8；不再等同 provider metal，不用于默认 gate。
- segment_ids shapes=(5153,128)/(1288,128)。
- crossing inferred=2374；provider mismatch=1791（27.8%），作为 AFLOW protocol/line-mode discrepancy audit，不进入 features。
- composition overlap=199；prototype overlap=0（prototype key 含 spacegroup，因此仅作初步 audit，尚非独立 prototype holdout）。

## Verification

- Local tests: 25 passed。
- Server tests: 25 passed。
- SSL smoke: 1 epoch GPU, inner train/val groups disjoint, outer OOD untouched。
- Supervised smoke: 1 epoch CPU；legacy gate=False；report generated；outer OOD final-only。
- download processes=0（按用户要求保持暂停）。

## Scientific Consequence

2026-08-24 之前基于 provider-metal Fermi anchoring 的分类/部署门控结果降级为 **legacy diagnostic**，不可再称为无泄漏独立 metal 分类基线。正式基线需在 noleak_v4 + Phase B MBM 升级后重训。

## Next

进入 Phase B：learnable mask token、span/segment-aware masking、warmup+cosine、early stopping、gradient clipping、masked-position-aware validator；完成后在 noleak_v4 做多 seed baseline，再进入 crystal graph→multi-band E(k)。
