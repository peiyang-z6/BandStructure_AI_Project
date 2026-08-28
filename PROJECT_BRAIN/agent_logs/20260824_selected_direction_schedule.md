# 2026-08-24 — Selected Direction Schedule

## User Selection

选定主路线：

> **真实晶体结构 → 完整能带预测 + DFT 质检/相似检索 + 不确定性主动学习**

该路线保持项目核心大方向，不建立冗余并列程序；沿现有数据、Transformer、物理 heads、GUI 和报告链路做系统性升级。

## Scope Boundary

- Analyzer mode：E(k)/图像 → 参数、质检、相似检索（保留并强化现有能力）。
- Predictor mode：crystal structure → multi-band E_n(k)（新增真实预测能力）。
- 两种 mode 共享同一物理标签、OOD/uncertainty/reporting 合同；指标分别报告，禁止将 Analyzer 指标包装为结构预测指标。
- Hamiltonian/SOC/拓扑路线暂不纳入当前里程碑，待 Predictor mode 稳定后再评估。

## Schedule

### Milestone A — Scientific-contract repair（立即执行）

1. 移除 `is_metal` 标签条件化输入：metal crossing 只能由 energies + E_F 推断；provider 标签只作为 target。
2. 加入 k-path segment boundary / symmetry-point metadata；曲率和一致性损失不得跨段。
3. 扩展 manifest：金属标签/特征一致性、formula/prototype/source overlap audit、数据/代码 hash。
4. 重建 6443 快照，比较修复前后 metal/direct/indirect 与 raw/gated gap 指标。

**Exit criteria**：无 target-conditioned feature；tensor contract 保持 `(N,2,128,3)`；segment_ids `(N,128)`；spacegroup train/test 零交集；全套测试通过；完成 1 epoch SSL + 监督冒烟。

### Milestone B — MBM representation upgrade

1. learnable mask token 替换零填充；
2. 5–15 点 span masking + segment-aware masking；
3. warmup+cosine、early stopping、gradient clipping、resume；
4. 3-seed linear probe/full fine-tune；mask/physics-loss 消融；
5. masked-position-aware reconstruction validator。

**Exit criteria**：同一 6443 manifest 上，inner-val/linear-probe 不劣于当前基线；外层 OOD 只最终评估；三 seed 均值±标准差完整。

### Milestone C — Crystal graph → multi-band E(k)

1. 在现有模型中引入晶体图输入：元素、分数坐标、晶格、邻接、空间群/对称信息；
2. 结构 encoder 与 k-path-conditioned sequence decoder 共享现有 Transformer/heads；
3. 从两条边缘带起步，逐步扩展到 Fermi 附近多条带；
4. Analyzer encoder 与 Predictor decoder 做表征蒸馏/一致性训练；
5. 评价 band-energy MAE、gap/type、极值位置、dispersion 及多类 OOD。

**Exit criteria**：输入不含 E(k) 时能输出多带序列；structure/prototype OOD 有独立报告；不得依赖测试集校准。

### Milestone D — Product applications

1. DFT QA：Fermi/spin/k-path/negative-gap/discontinuity/provider-label consistency；
2. Band fingerprint：SSL embedding 相似材料检索、聚类、异常检测；
3. Uncertainty active learning：高不确定/高价值候选自动形成 DFT 任务队列；
4. Multi-objective screening：gap/directness/dispersion/uncertainty + 稳定性/毒性/成本 metadata 排序；
5. GUI：批量输入、候选表、相似材料、异常原因、导出 CSV/DFT 清单。

**Exit criteria**：端到端演示“结构候选 → 预测/质检 → 不确定性排序 → DFT 验证队列”；报告明确区分预测值、门控值和 provider/DFT 标签。

## Data Schedule

- 当前正式快照：AFLOW 6443 缓存 / 6399 clean tensors。
- 30000 条下载保持暂停，待 Milestone A 完成后从 6443 条断点续传。
- 下载恢复前固定 staged targets、8 workers、候选池上限、exclusion registry 和 snapshot hash。

## Immediate Next Action

执行 Milestone A1–A3；完成后重建 6443 张量并运行测试与冒烟，不等待 30000 条下载。
