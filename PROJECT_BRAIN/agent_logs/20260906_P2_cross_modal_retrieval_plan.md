# 2026-09-06 P2 跨模态检索 — 审计与方案

## 目标（宪法 5.0 §8 P2）

保留 MBM 作 band encoder，结构编码器复用成熟 GNN（不从零发明），
structure–band 对比学习对齐 60k 配对嵌入，建立 ANN 向量索引实现
"能带图 → 最相似材料/结构/置信度" 检索。P2 优先于 multi-band 生成。

## 审计结论（实测 2026-09-06）

### 现有 band encoder

- `src/models/band_structure_encoder.py`：纯 TensorFlow/Keras Transformer。
  - `BandStructureEncoder`：band_projection → positional encoding → 6 层
    MultiHeadAttention + FFN → (N, seq_len, d_model)。
  - `SSLEncoder`：encoder + MeanPooling + projection_head(128) +
    reconstruction_head。`call(..., return_features=True)` 返回 pooled
    (N, d_model=256) 特征。
- 实际训练配置（v7）：输入 (N, seq_len=128, num_features=6)（由
  (N,2,128,3) 展平）；d_model=128、num_heads=4、num_layers=4、dff=512、
  projection_dim=64。
- 张量：`band_tensors_ood_split.npz` X_train (47912,2,128,3)、
  X_test (11987,2,128,3)，material_ids 完整（U64）。

### 环境（关键框架约束）

- conda env `bandstructure-ai`：Python 3.11、TF 2.21、numpy 1.26、
  pymatgen 2026.7.31。
- **无** torch / alignn / matgl / e3nn / faiss / hnswlib。
- 本地 WSL 可见 GPU（RTX 4060）；服务器 2×V100。

→ 宪法要求"复用 ALIGNN/MatGL/e3nn 类成熟 GNN"，但环境是纯 TF 生态。
结构编码器框架选型是 P2 的首个决策点（已 clarify 待用户回复）。

### P1 sidecar 数据合同缺陷（P2 前置，已发现并修复中）

- `species` 字段是 AFLOW 去重元素列表（如 ['Co','Sc','Zn']），
  `fractional_coordinates` 按原子排列（4 行）→ **56,099/59,961 条
  len(species) != len(coords)**，pymatgen `Structure(lattice, species,
  coords)` 直接失败。
- 根因：`?species` 端点返回去重列表；权威 per-atom 顺序在 CONTCAR.relax
  （VASP POSCAR，含符号行 + 计数行）。
- 修复：`scripts/fix_sidecar_species.py` 拉 CONTCAR.relax 解析 per-atom
  species，写 sidecar 新增 `species_per_atom` 字段（不动原 `species`，
  向后兼容）。VASP4（仅计数行无符号行）用 sidecar `species` 列表 + POSCAR
  计数组合还原（已验证 POSCAR 符号顺序与 sidecar species 8/8 一致）。
- 测试：`tests/test_poscar_species_parse.py`（5 passed，含 VASP4 两分支）。

## 方案草案（待结构编码器决策后定稿）

### P2a 结构编码器
- 输入：sidecar 的 lattice/species_per_atom/fractional_coordinates。
- 选项 A（e3nn 等变 GNN）：装 torch+e3nn，物理最强，双框架代价。
- 选项 B（纯 TF CGCNN 式消息传递）：复用 Keras，单框架，零新依赖。
- 选项 C（pymatgen 描述子 + MLP）：最快可运行基线，P3 再升级。

### P2b 对比学习
- structure encoder 输出 → 共享投影（同 dim=128/64），InfoNCE 对齐
  60k (structure, band) 配对。
- 冻结 band encoder（v7 权重）或联合微调，待定。

### P2c ANN 检索
- 无 faiss/hnswlib；60k × 64 维 brute-force 余弦检索可行（~15MB 矩阵），
  或 pip 装 faiss-cpu。先 brute-force 出基线指标（Recall@K/mAP）。

### 评估
- Recall@1/5/10、mAP、结构↔能带互检索；跨七拆分（尤其 space-group OOD）
  的检索泛化。

## 未决

- 结构编码器框架选型（clarify 超时，待用户回复）。
- 对比学习是否联合微调 band encoder。
