# 2026-08-24 — Next Work Schedule after Formal noleak_v4 Baseline

## Starting point

- Phase B：完成。
- Canonical experiment：`aflow_noleak_v4_seed42`。
- Outer gap MAE：0.049747 eV；Macro F1：0.858339。
- 首要分类瓶颈：direct recall 0.765854；direct→indirect 92 条。
- 当前模型仍是 E(k) 分析器，不是 structure→multi-band predictor。
- AFLOW 下载继续暂停在正式 6443 条，未经新指令不扩大。

## Global rules

1. Outer space-group split 固定 seed=42 且不参与任何选择。
2. 多 seed 只改变初始化/训练随机性；若改变 inner split，必须单独声明。
3. 不恢复 label-conditioned feature、legacy metal gate 或 zero-gap hard gate。
4. 每阶段退出前必须有 manifest、测试、真实 smoke、报告和日期日志。
5. Phase C 不直接写新并列框架；升级现有 adapter/store/builder/encoder/trainer/pipeline/GUI 链路。

---

## Stage 1 — Baseline freeze and engineering closure

**计划工期：1–2 个工作日**

### Tasks

- 将本次迁移后的目录、路径合同、测试和报告作为 canonical baseline 提交/打 tag。
- 对 `run_full_pipeline.py --source aflow --status-only` 做最终 artifact 状态清单。
- 显式冻结或从 supervised optimizer 排除 `mask_token`，清除预期但干扰审计的 no-gradient warning。
- 移除 `fit(..., shuffle=True)` 的冗余参数；保持 `tf.data` 真实 shuffle 不变。
- 对 supervised GPU 失败建立最小 Conv1D/CUDA repro；只在明确 root cause 后决定修复或继续 CPU。
- 固化环境、代码、数据、模型与报告 SHA-256。

### Exit criteria

- Full pytest 全绿；模型 load+forward smoke 全绿。
- Pipeline status 对所有非 vision 核心 artifact 为 `exists=true`。
- CPU/GPU 策略有可复现说明，不再依赖临时 shell 习惯。
- Git diff 无行尾伪修改、无 secret、无临时脚本。

---

## Stage 2 — Three-seed formal baseline and direct/indirect error audit

**计划工期：3–5 个工作日（取决于服务器 CPU/GPU 修复）**

### Tasks

- 在同一 outer split 与同一数据 manifest 上补 2 个预注册训练 seeds，与 seed42 组成 3-seed。
- 分别报告 mean±std：gap MAE/RMSE/R²、accuracy、Macro F1、每类 precision/recall/F1、physics score。
- 对 137 条 direct↔indirect 错误做材料级审计：
  - VBM/CBM extremum segment；
  - extremum k-distance；
  - degeneracy/crossing；
  - provider directness protocol；
  - PCHIP 与原始 k 点一致性。
- 对 30 条 indirect→metal 误差检查近零 tensor gap 与 provider label mismatch。
- 只在 inner validation 上评估：class weights、type-head calibration、topology auxiliary loss；outer test 最后一次性评估。

### Exit criteria

- 3-seed 报告完整，outer test 使用次数与配置可审计。
- Direct recall 的变化有 seed variance 与置信区间，不以单次提升宣称改进。
- 任何 feature 修改都通过 provider-label-invariance 和 segment-aware regression tests。

---

## Stage 3 — Uncertainty calibration, DFT QA and similarity retrieval

**计划工期：3–5 个工作日**

### Tasks

- 在 inner validation 上建立 uncertainty calibration：coverage-width、reliability、risk-coverage、error-vs-uncertainty。
- 比较 MC dropout、deep ensemble（利用 3 seeds）和轻量 conformal residual calibration。
- 将当前审计指标系统化为 DFT QA：
  - Fermi reference inconsistency；
  - segment jump；
  - negative/inverted tensor gap；
  - provider global gap vs line-mode gap；
  - curvature/sign anomaly；
  - missing spin/channel；
  - reconstruction error outlier。
- 暴露现有 SSL encoder 的 pooled embedding，建立可审计 nearest-neighbor 检索；结果显示 source、spacegroup、composition 与距离。
- GUI 只增加 QA/检索结果页，不改 tkinter 主框架。

### Exit criteria

- Uncertainty 在独立 outer OOD 上有未调参的 calibration 报告。
- 高不确定样本的真实误差显著高于低不确定样本，才允许进入主动学习。
- 检索结果可回链 material ID、raw record、tensor manifest 和模型 hash。

---

## Stage 4 — Phase C structure data contract freeze

**计划工期：3–5 个工作日；此阶段先做数据合同，不写正式模型**

### Required structure schema

- lattice matrix `(3,3)`；
- species / atomic numbers；
- fractional coordinates；
- occupancy 与 disorder policy；
- periodic neighbor graph、cutoff 与 edge vectors；
- space group / symmetry；
- spin polarization、magnetism、SOC flags；
- functional、pseudopotential、U、k-mesh 等计算协议；
- structure hash / prototype ID。

### Required multi-band target schema

- 统一 reciprocal k-path 与 segment IDs；
- k-point fractional coordinates 与 cumulative distance；
- Fermi 附近固定窗口的多条 valence/conduction bands；
- padding/band mask；
- degeneracy 与 band-order ambiguity policy；
- spin/SOC channel policy；
- canonical energy reference；
- 缺失/低质量/协议冲突隔离原因。

### Tasks

- 审计 AFLOW/MP 当前 raw records 是否真正含上述字段，禁止从文档推断。
- 先构建 100–500 条 structure+band pilot snapshot。
- 扩展现有 `aflow_adapter.py`、`band_store.py` 与 `ood_tensor_builder.py` 的 schema/provenance；不另建第二套 downloader/store。
- 建立 structure/composition/prototype/source OOD manifest。

### Exit criteria

- 100% pilot records 通过 schema validation 或进入有原因的 quarantine。
- 同一 structure hash 不跨 split。
- multi-band target 可从 HDF5 读回并与原始 E(k) 数值比对。
- 用户审阅并批准数据合同后才进入模型实现。

---

## Stage 5 — Minimal crystal graph → multi-band prototype

**计划工期：1–2 周；需 Stage 4 明确批准**

### In-place upgrade targets

- `src/models/band_structure_encoder.py`：在现有序列 encoder 中加入结构条件上下文与 k-conditioned decoding；保留当前 6D encoder 接口。
- `src/engine/ssl_trainer.py`：支持 structure-conditioned band masking，不复制训练器。
- `src/engine/finetune_trainer.py`：支持 multi-band losses 和 uncertainty heads。
- `scripts/run_full_pipeline.py`：统一数据/训练/评估入口。
- 当前 6D 模型继续作为 E(k) QA、检索和下游 gap/type baseline。

### Prototype order

1. 16 样本 overfit，证明 graph→sequence 数据流正确；
2. 100–500 pilot group-disjoint 训练；
3. 报告 band-energy MAE、gap/type、extrema、crossing、segment continuity；
4. 与无结构 baseline、仅 composition baseline 比较；
5. 通过后才扩大数据和模型。

### Exit criteria

- 结构输入置换、周期平移和等价胞选择测试通过。
- 不依赖 target E(k) 作为输入；否则不算 structure→band prediction。
- 模型输出多带 Eₙ(k)，不只输出 gap 标量。
- 外层 OOD 从未用于模型选择。

---

## Stage 6 — Active-learning loop and GUI integration

**计划工期：1 周以上；需 uncertainty calibration 与 Phase C 原型均通过**

- acquisition score = uncertainty + scientific value + diversity + compute cost；
- 只将高价值/高不确定结构加入 DFT queue；
- 保存提交结构、计算协议、DFT 输出、失败原因和回流模型版本；
- GUI 展示候选、相似材料、QA flags、uncertainty 与是否建议 DFT；
- 不把模型输出直接称为实验结论。

## Immediate recommended next action

先执行 **Stage 1**，随后补 3-seed。Phase C 的第一步是 Stage 4 数据合同审计，不应立即开始写 graph model。
