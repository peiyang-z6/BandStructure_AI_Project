# 2026-08-24 — Model Transition Audit and Updated Development Plan

## Purpose

模型切换后重新审计当前线程、宪法、dev_context、20260812–20260824 日志、核心源码、真实张量/模型/报告，并与近期电子结构机器学习路线对照。本日志更新计划；候选拓展方向在用户选择前保持 `PENDING`，本次不修改核心模型方向。

## Historical Context Reconstruction

Hermes 历史数据库中未找到独立的 BandStructure/Kimi K3 已结束会话；唯一 Kimi K3 历史会话是量子力学网页，与本项目无关。BandStructure 的模型工作保存在当前线程和项目 `PROJECT_BRAIN/agent_logs/`，因此本次以项目磁盘产物为事实来源。

已重建的关键进度：

1. Stage 0 修复 MP/AFLOW 下载、真实 k 坐标、全自旋选择、spacegroup OOD 泄漏、GUI Fermi/反向标定。
2. AFLOW 缓存扩到 6443 条；过滤后形成 6399 条、183 空间群的正式快照。
3. metal 费米锚定和分类先验修复后，metal recall 从 0 提升到 97.5%。
4. 2026-08-24 训练：raw OOD gap MAE=0.1868 eV，classification accuracy=0.679、Macro F1=0.658，metal/direct/indirect recall=0.975/0.651/0.648。
5. 部署费米锚定门控后 gap MAE=0.0645 eV、RMSE=0.2358 eV、R²=0.9882；raw 与 gated 指标分开保存。
6. 下载已暂停于 6443 条；本地/服务器 tests=19 passed，核心文件 SHA-256 一致。

## Central Scientific Finding: What the Current Model Actually Does

当前训练入口读取 `band_tensors_ood_split.npz`，输入是由完整 E(k) 提取的 VBM/CBM 能量、曲率和极值距离序列；代码中没有晶体原子种类、坐标、晶格或 crystal graph 输入。因此：

- 当前核心是 **E(k) → 表征/重建 → gap/type 分析器**；
- 它适合能带数字化、去噪、自动分析、质检、相似检索和图像回流；
- 它不是 **crystal structure → E(k)** 预测器，不能按当前形式替代未知晶体的 DFT；
- gap/type 已可从输入 E(k) 直接计算，监督任务具有“从答案提取摘要”的循环性，不能把高指标表述为从晶体结构预测能带。

直接从晶体图预测完整带结构是可行的研究方向：Bandformer 将晶体图编码和连续 k-path 序列解码结合，在 52,861 条 Materials Project 能带上报告了 0.14 eV band-energy MAE。[3] 另一条路线不是直接回归 E(k)，而是学习满足 E(3) 等变性的 DFT/紧束缚 Hamiltonian，再对角化得到能带；DeepH-E3 与 HamGNN 证明该路线可扩展到大超胞、缺陷和 Moiré 系统，并严格保留旋转/反演等物理对称性。[1][2]

## Is the Current Deep-Learning Scheme Reasonable?

### Reasonable parts

1. **MBM sequence representation is reasonable for E(k) analysis.** Band structure is a piecewise-continuous k-path sequence；Transformer/graph-to-sequence 处理连续色散与极值是合理归纳偏置。[3]
2. **Physics-guided SSL is directionally correct.** 物理相关预任务、masking 与结构扰动在材料 GNN 中能够改善下游性质预测，但效果依赖预训练/下游分布匹配。[4]
3. **Group-disjoint OOD is better than sample random split.** 当前 spacegroup OOD 合同优于随机拆分；材料发现基准也强调回归平均误差不足以代表真实筛选价值，必须用接近部署任务的 OOD 与分类指标。[5]
4. **Uncertainty + DFT prefilter is practical.** ML 更适合作为 DFT 前置筛选，而不是未经校准直接替代高保真计算。[5][6]

### Major shortcomings

#### P0 — Scientific validity

1. **Task circularity:** E(k) 已是输入，gap/type 是其确定性派生量；当前指标主要证明自动分析能力，不证明结构→能带预测能力。
2. **Target leakage risk:** `is_metal` provider 标签被用于 metal tensor Fermi anchoring，同时又是分类目标。必须改为仅由 energies+E_F 检测穿越，标签只做监督，不得改变输入。
3. **Lossy 6D representation:** 仅保留两条边缘带，丢失多带简并、轨道成分、spin identity、SOC、波函数、DOS/Fermi surface；不能可靠推导光学跃迁、拓扑不变量和完整输运。
4. **k-path semantic mismatch:** 不同空间群的 Γ–X–… 路径、段数和物理长度不同，当前 128 点位置编码未显式提供 segment boundaries/symmetry labels；跨段曲率可能无物理意义。

#### P1 — Model/training

5. SSL 仍用零填充 mask（标准化均值附近），无 learnable mask token；仍为随机单点 mask，无 span masking。
6. SSL 无 warmup+cosine、early stopping、gradient clipping；6399 条 best val loss=0.1206，表明现容量/训练策略难以随数据扩展。
7. `finetune_supervised.py` 已达 1500+ 行；模型、训练、评估、报告仍集中在单脚本，已有 `finetune_trainer.py` 未承担应有职责。
8. 只有 seed=42 单次结果，无三 seed 方差、linear probe 和系统消融。

#### P2 — Data/evaluation

9. OOD 只有 spacegroup；还缺 element/composition holdout、prototype holdout、source holdout 和 gap-range extrapolation。
10. AFLOW/MP/JARVIS 的泛函、赝势、SOC 和磁性协议不同；source-aware 校准未落地。
11. PBE 类标签与实验带隙存在系统偏差；项目不能把 DFT-label accuracy 等同实验 accuracy。
12. 尚无完整 run manifest（数据 SHA-256、代码 commit/config hash、环境锁、split IDs、seed）。

#### P3 — Application boundary

13. 当前 `material_classifier.py` 依赖过期路径/样本数字的 docstring；识别元素/结构主要是 metadata lookup 或 formula fallback，不是从能带图唯一反演晶体结构。
14. `inverse_generator.py` 只输出元素概率与空间群的轻量原型，不生成满足化学计量、晶格稳定性和可合成性的真实结构；不能称为材料逆设计器。

## Updated Core Development Plan

### Phase A — Scientific-contract repair (must precede 30000 training)

1. **Remove label-conditioned input construction** in `src/data/ood_tensor_builder.py`：metal crossing 仅由 E(k)+E_F 推断；provider `is_metal` 只作为 target。重建 6443 快照并检查 metal 指标是否仍成立。
2. **Segment-aware tensor**：把 k-path segment boundary/symmetry-point id 纳入 attention mask 或附加 metadata；曲率损失禁止跨段差分。6D 主张量不变，辅助 mask/metadata 旁路保存。
3. **Declare two product modes**：
   - Analyzer mode：E(k)/图像 → 物理参数、质检、检索；
   - Predictor mode：crystal structure → E(k)/Hamiltonian；
   两者共享物理 heads/reporting，但不得混淆指标。
4. **Evaluation matrix**：spacegroup OOD + composition/prototype OOD + source OOD；所有阈值只在 inner validation 校准。
5. **Rebuild manifest**：数据/代码/配置/环境/split/seed hash 全记录；无 manifest 的结果不得升级为正式科研结论。

### Phase B — MBM upgrade in existing files

1. `band_structure_encoder.py`：learnable mask token；
2. `ssl_trainer.py`：span masking（5–15 k points）、segment-aware masking、gradient clipping；
3. `train_ssl.py`：single warmup+cosine、early stopping、resume、mixed precision；
4. 3-seed ablation：zero vs token、point vs span、physics loss on/off；
5. linear probe + full fine-tune；评价 masked-position reconstruction，不再用全序列 decoder 物理分数。

### Phase C — True crystal-to-band predictive capability (pending route selection)

**Route C1 — Direct graph-to-sequence (recommended near-term):** 在现有 Transformer 中系统性替换/扩展输入投影为 crystal graph encoder，decoder 以标准高对称 k-path 为条件输出多条 E_n(k)。这一方向与 Bandformer 类似，数据要求与现有 30000–50000 line-mode 数据最匹配。[3]

**Route C2 — Hamiltonian surrogate (high-value long-term):** 预测 E(3)-equivariant Hamiltonian，随后对角化得到 band/DOS/wavefunction/Berry-related quantities。物理完备性和缺陷/异质结构迁移更强，但需要原子轨道基 Hamiltonian 标签，当前 AFLOW bandsdata 不足以训练。[1][2]

### Phase D — Productization and applications (pending user selection)

项目不应只做“看图识别”和“生成一张曲线”，还可形成以下四类实用产品路线：

1. **Electronic-structure QA + searchable fingerprint platform**：自动发现 DFT 错误（Fermi、spin、k-path、负 gap、断带），按能带形状检索相似材料；最贴合当前 E(k) encoder，开发风险最低。
2. **Uncertainty-guided high-throughput screening / active learning**：按目标 gap、directness、平带、曲率、成本/毒性/稳定性进行多目标排序；不确定样本自动提交 DFT，形成 propose→calculate→learn 闭环。ML 作为 DFT 预筛是成熟的应用定位。[5][6]
3. **Device-oriented electronic design workbench**：面向光伏/LED/功率半导体/透明导体/热电材料，输出候选排名、解释、置信度和 DFT 验证清单。真正的光学强度、迁移率和热电系数需要增加 DOS、物理 k 单位、速度/散射或光学矩阵元，不能仅由当前 6D 张量宣称。
4. **Advanced quantum-material simulation**：应变/缺陷/掺杂/异质结的能带变化、拓扑材料筛选、Moiré 大超胞；建议走 Hamiltonian route，因等变 Hamiltonian 可导出比 E(k) 曲线更多的电子结构量。[1][2]
5. **Experimental-computational discrepancy analysis**：把 ARPES/STS/论文图像与 DFT/ML 对齐，检测带重整化、缺陷态、实验偏移；现有 tkinter 标定与图像回流可直接演化为该方向。

材料 AI 的实际价值通常来自“ML 预筛 → 少量高保真 DFT/实验验证”，而不是单一指标模型本身；高通量库、主动学习与实验/计算闭环是更可信的产品形态。[5][6]

## Recommended Portfolio

**Primary recommendation:** C1 direct crystal graph→multi-band sequence + D1 QA/fingerprint + D2 active-learning screening。

理由：

- 保留当前 E(k) MBM/GUI 成果，不推翻主线；
- 解决项目最大的概念缺口（没有结构输入）；
- 30000 AFLOW line-mode 数据可直接作为 C1 初始监督集；
- QA/fingerprint 可在 C1 完成前先交付实用价值；
- Hamiltonian route 保留为后续研究分支，不在当前数据条件下强行实现。

## Execution Gate

在用户选择产品路线前，只执行审计、计划与文档更新。选择后：

1. 将所选路线写入 `PROJECT_BRAIN` 日程；
2. 先执行 Phase A 科学合同修复并重跑 6443 基线；
3. 再执行 Phase B MBM 升级；
4. 根据所选 C/D 路线进入结构预测或应用开发。

## Sources

[1] https://www.nature.com/articles/s41467-023-38468-8 — DeepH-E3
[2] https://www.nature.com/articles/s41524-023-01130-4 — HamGNN
[3] https://arxiv.org/html/2411.16483v1 — Bandformer
[4] https://arxiv.org/html/2401.05223 — Physics-guided dual SSL
[5] https://www.nature.com/articles/s42256-025-01055-1 — Matbench Discovery
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC7067066 — Materials Science in the AI age
