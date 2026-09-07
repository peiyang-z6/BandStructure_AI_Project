# P3 Variable Multi-Band Decoder — 开发方案（2026-09-07）

## 治理状态（如实记录）

- **P2 跨模态检索未验收**：v1 纯 TF CGCNN 与 v2 e3nn 等变编码器 + 增强描述子均未达标。
  - v1：train s2b recall@10=0.39，test recall@1=0.0035（≈随机）→ 严重过拟合。
  - v2（1+2 结合）：train s2b recall@10=0.0526，test recall@10=0.0054（random 0.00084）；诊断：结构嵌入无 collapse（pairwise cos mean 0.034）但 InfoNCE batch-64 学到退化解，全局检索失效。
- 宪法 5.0 §8 规定"P2 通过后再进入 P3""后续阶段不得在前置阶段验收前启动正式实现"。
- **用户指令（2026-09-07）"进行第三阶段，也就是下一个阶段"**：用户作为 §1 授权者，在已知 P2 负面结果后仍指示推进 P3。本方案按用户授权推进，P3 启动为**跳过 P2 验收的授权决策**，记录于此；P2 检索留作后续优化（不阻塞 P3 生成任务，因生成与检索是不同训练目标）。

## 数据合同审计结论（已实测，非猜测）

- **完整多条 band 数据已齐备，无需重新下载**。实测 `data/raw/aflow/snapshots/aflow_60000_20260831/aflow_bands.h5`（60,000 groups）：
  - 每 group 存 `energies` shape `(num_bands, num_kpoints)`，**num_bands 可变**（实测 82 / 84 / 96 条），**num_kpoints 可变**（实测 200 / 280 点）。
  - `k_distances`/`kpoints`（累计 k 距离）、`kpath_labels`（高对称点）、`num_sites`、`spacegroup_number`、`is_metal`、`band_gap`、`efermi`（E_F 已移至 0）齐全。
- **结构数据齐备**：P1 sidecar 60,000 记录（lattice/species_per_atom/fractional_coordinates/functional/spin/倒格子），59,961 条 pymatgen 可构造。
- **拆分齐备**：七类拆分（含 canonical space-group OOD，train 47,912 / test 11,948，空间群零交集）。
- 结论：P3 是**纯模型+训练阶段**，无数据下载、无新 sidecar。

## Bandformer 调研结论（arXiv 2411.16483 + 开源 github.com/qmatyanlab/Bandformer）

- **架构**：晶体图（节点=原子 one-hot，边=距离 Gaussian 展开）→ Graph Transformer encoder（biased multi-head attention，边特征作 attention bias，ResNet+LayerNorm）→ 全局池化晶体特征 → graph2seq decoder（k-point 坐标 positional encoding → self-attention → graph2seq attention → MLP）。
- **关键设计**：① band center / band dispersion 分解（LayerNorm 会归一化每条 band，不同 band 中心不同，故分离预测均值与偏离）；② v2 用 rFFT 处理连续 band 序列。
- **数据与指标（v2）**：MP 27,772 band structures，费米附近多条 band，band MAE **0.304 eV**，gap MAE **0.251 eV**（非金属）。公开拆分普通 90/10，固定 band 数。
- **我们的差异化**：AFLOW 60k（>2× MP 数据量）+ 空间群隔离 OOD 拆分（非普通 90/10）+ variable band 数（mask）+ band-set matching（Hungarian/OT）+ 物理约束。

## P3 分步方案（每步有退出标准）

### 3a. 数据准备 — `scripts/prepare_p3_multiband.py`
- 从 HDF5 提取**费米附近 band**：对每条 band 求全局 min/max，取落在 [E_F−ΔE, E_F+ΔE] 窗口内的 band（ΔE 可调，默认 5 eV）；金属无 gap，按窗口选费米跨越 band。band 数上限 `max_bands`（默认 16），不足则 padding + `band_mask`。
- 统一 k 网格插值到固定 `n_k`（默认 256，shape-preserving，复用现有插值，禁 cubic-spline 过冲）。
- 关联 P1 sidecar 结构 + 复用七拆分（不重建拆分，manifest 落盘）。
- 输出：`data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband/`（train/test NPZ：(N, max_bands, n_k) + band_mask + k 坐标 + 结构图）。
- **退出标准**：band 数分布直方图 + mask 覆盖率报告；金属/绝缘体 band 选取 sanity check；张量 SHA 落盘。

### 3b. Bandformer baseline 复现/适配
- 适配 Bandformer 开源代码到 AFLOW 数据（写 AFLOW→Bandformer train.pt 适配器），或按论文架构实现等价版本（Graph Transformer + graph2seq + center/dispersion 分解）。
- 在**相同数据与拆分**（3a 产出）训练，得到 baseline 指标。
- **退出标准**：train/test band MAE + gap MAE 复现到论文量级（0.3 eV band MAE 附近）或记录无法复现的原因。

### 3c. 本项目 variable multi-band decoder — `src/models/multiband_decoder.py`
- **结构编码器**：复用 P2 等变编码器（e3nn）或 CGCNN，输出结构嵌入（P2 产物可复用，但 P3 生成任务下重新训练而非冻结）。
- **decoder**：结构嵌入 + k 坐标 positional encoding → 条件解码 E_n(k)（Transformer graph2seq 或更简单 MLP 先跑通）。
- **band-set matching**：Hungarian/最优传输处理 band 交换与 crossing（可变 band 数用 mask 排除 padding）。
- **物理约束 loss**（分优先级，先 core 后进阶）：
  - core：segment continuity（相邻 k 点能量差正则）、VBM/CBM 位置、metal/direct/indirect topology；
  - 进阶：高对称点简并、时间反演对称、曲率/有效质量。
- **退出标准**：train/test band MAE、gap MAE、extremum k-error、effective-mass error；对比 3b Bandformer baseline。

### 3d. 正式对比 + 报告
- Bandformer vs 本项目，相同数据/拆分，主指标 full-band MAE、extremum k-error、effective-mass error、Recall@K/mAP（P5 预留）。
- 报告：`artifacts/reports/aflow_noleak_v7_60k_seed42/p3_multiband_report.md`。

## 关键设计决策（待用户确认的默认值）

1. **max_bands=16、n_k=256、ΔE=5 eV**（可调，先跑通再调优）。
2. **decoder 起步用简单 MLP/Transformer**，先拿到可训练 baseline，再逐步加 graph2seq + FFT（避免一步到位 Bandformer 全架构的调试成本）。
3. **结构编码器重新训练**（P3 生成任务，不冻结 P2 检索权重）。

## Non-goals（明确不做）

- 不重新下载数据、不新建 sidecar、不重建拆分、不改 immutable HDF5/NPZ。
- 不从头实现完整 DeepH 类哈密顿量网络（宪法 §1 禁止）。
- 不在本阶段实现 P4 校准不确定性 / P5 外部验证集。
- 不修 P2 检索（留作后续，除非用户要求）。

## 验证义务（宪法 §10）

Python compile + 全量回归测试 + 张量 shape/group count/SHA + 模型 load/forward smoke + 服务器状态回验 + git diff 规模。
