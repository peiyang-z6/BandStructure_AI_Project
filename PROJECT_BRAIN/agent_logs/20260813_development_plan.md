# 2026-08-13 Development Plan: Stage 1-4 Systematic Upgrade

## Context

基于 CONSTITUTION v4.1、dev_context（2026-08-12 Stage 0 基线）、README 与全部
agent_logs 的完整审计，以及源码复核（ssl_trainer / losses / band_structure_encoder /
finetune_supervised 等），在与用户（Cherry Studio 科研助手对话背景）讨论后形成本方案。

**本日志只记录方案，不修改任何代码与既有文档。** 方案遵循两条红线：

1. 不改变项目大方向（MP/AFLOW E(k) → 6D 张量 → MBM SSL → 监督微调 → tkinter Phase 5）；
2. 不做并列式"新模块"拓展，所有工作为对现有程序的功能性升级与系统性 bug 修复。

## Current State Assessment

- 数据层：MP/AFLOW 双适配器、断点续传、原子写入已修好；但仅有 12 条 AFLOW 冒烟数据，
  无 metal 类，MP 源从未实际运行。
- 张量层：6D 契约、真实 k 坐标、全自旋边缘态、group 分层划分符合宪法。
- SSL / 监督微调：仅 1 epoch 冒烟；历史 100 epoch 结果（0.29776 / MAE 0.47 eV 等）
  对应的模型与报告不在仓库中，按 dev_context 只能视为历史陈述。
- GUI：tkinter 重写完成，Fermi 零点 / 反向轴 / 缩放不变标注已修。
- 环境：WSL2 + RTX 4060 + TF 2.21 GPU 链路打通，conda 环境 `bandstructure-ai`。

结论：管线科学性已修复，但**没有任何可宣称的精度基线**。

## Confirmed Problems (by severity)

### P0 — 阻塞性

1. 数据规模不足：12 样本 / 11 空间群 / 无 metal 类，三分类目标无从谈起。
2. 无可复现精度基线：后续任何算法改动没有对照系。

### P1 — 设计缺陷（源码确认）

3. SSL 掩码用 0 填充（ssl_trainer.py `masked_batch = batch * (1.0 - mask)`），
   归一化后 0≈均值，模型走捷径，应改可学习 mask token。
4. 逐点随机掩码：能带曲线平滑，邻居插值即可重建，学不到全局物理；
   应改连续段（span）掩码，覆盖率仍守宪法 15–30%。
5. SSL 训练器无学习率调度、无早停（历史 epoch 92 最优、100 已过拟合）；
   允许单一 warmup+cosine，禁止叠加互相覆盖的控制器。
   （2026-08-13 源码复核更正：虚拟应变增强已在 `src/data/band_structure_dataset.py`
   的 `VirtualStrainAugmentation` 实现并默认启用（k 空间应变 + 重插值），
   原方案"未实现"系误判；但增强幅度 strain_scale=0.01 是否充分、以及能量缩放
   与极值扰动两类物理增强仍缺失，列入 Stage 3 消融。）
7. Focal loss 过度矫正未闭环（20260605 自查：19/47 indirect 被误判为 direct）；
   需在 inner-val 上以 Macro F1 为准则调 gamma/alpha 或阈值。
8. `finetune_supervised.py` 1485 行上帝脚本；逻辑应归位到已有的
   `src/engine/finetune_trainer.py` 与 `src/models/`（代码归位，不是新模块）。

### P2 — 科学与工程欠债

9. 物理损失自适应权重为开环启发式，两权重耦合，无消融验证。
10. 曲率一致性损失假设等距网格，跨高对称段边界的差分在物理上错误，应按段索引置零。
11. 有效质量停留在 relative proxy；MP/AFLOW 原始记录含笛卡尔 k 坐标（有物理单位），
    核心流水线可输出真 m_e 单位有效质量（6D 契约不变，追加 NPZ 字段）。
12. MP/AFLOW domain shift 只有原则没有可执行判据。
13. 无实验 manifest 制度（数据快照哈希 + 环境锁 + 配置哈希）——历史数字失效的根源。
14. DFT(PBE) 带隙系统性低估 30–50% 无任何声明与处理，报告与答辩存在风险。

## Development Plan (six directions)

### 1. 材料数据下载（12 条 → 可训练规模）

- 规模目标反推自 OOD 统计需求：空间群 ≥ 60（测试侧 ~12 群 × 每群 ≥ 3 条）；
  总样本 800–1500；metal 类 ≥ 150；natoms ≤ 50 保持。
- AFLOW 七层带隙分层采样：[0,0.1] / [0.1,0.5] / [0.5,1.0] / [1.0,2.0] /
  [2.0,3.5] / [3.5,5.0] / [5.0,8.0] eV，复用现有 gap-strata 交错逻辑，属调用策略升级。
- 配置 MP key 后跑 MP 源冒烟（50–100 条），与 AFLOW 同空间群样本做字段级对照，
  同时作为 domain shift 首次定量测量。
- 定义合并判据并写入宪法 §3：共有化学式/空间群样本带隙差 |μ|<0.1 eV 且 σ<0.3 eV
  方可合并训练；否则只允许"AFLOW 预训练→MP 微调"或分列报告。
- 下载侧：AFLUX 请求加固定限速应对 429；续传 manifest 增加分层独立计数。

### 2. 深度学习训练框架（归位与复现性）

- 微调逻辑归位：ExtremumExpectedGapHead / type head / 物理辅助损失 / 指标迁入
  `src/engine/finetune_trainer.py` 与 `src/models/`；脚本收敛为 argparse + 调用（<200 行）。
- 统一实验配置：单一 YAML（configs/train_ssl.yaml、configs/finetune.yaml）+
  argparse 覆盖；配置哈希写入 run manifest。
- SSL 训练器：单一 WarmupCosine 调度 + inner-val 早停（patience 15）+
  梯度裁剪（global norm 1.0）+ mixed_float16（冒烟验证数值稳定性）。
- 多 seed 评估：正式结果一律 3 seed 均值±标准差；run_full_pipeline.py 加 --seeds。
- 评估报告自动生成为微调最后一步：MAE/RMSE/R²、混淆矩阵、Macro F1、per-class AUC、
  按带隙分层误差表、校准曲线（对接学术论文报告模板需求）。

### 3. 自监督学习算法

- 可学习 mask token 替换零填充（SSLEncoder 内联修改，~20 行）。
- Span masking：连续段长 5–15，总覆盖率 15–30%（宪法区间）。
- 虚拟应变增强（宪法合规补实现，仅作用于训练集，参数入 manifest）：
  能量缩放 ±10%；k 路径重采样扰动；VBM/CBM 间距扰动。
- 物理损失审计：跨高对称段边界差分权重置零；2×2 消融
  {曲率符号损失}×{自适应权重}，3 seed，用 inner-val 数据决定取舍。
- SSL 质量代理评估：linear probe（冻结 encoder 训线性头）对比 full fine-tune，
  作为表征质量度量写入报告。

### 4. 后续功能开发（契约内延伸）

- 真有效质量：张量构建时对 VBM/CBM 邻域二次拟合 + ℏ²/d²E/dk² 换算，
  输出 m*_vbm/m*_cbm 至 NPZ 附加字段，微调作为小权重辅助回归目标。6D 契约不变。
- MC Dropout 不确定性接线（src/utils/mc_dropout.py 已存在未用）：T=30 前向，
  GUI 与报告显示 "Eg = 1.35 ± 0.12 eV"；与 PBE 系统性偏差分开陈述。
- Focal loss 闭环：inner-val 上网格搜索 / 阈值平移 / 温度缩放三选一，
  以 Macro F1 最大化为准则。
- GUI 批量筛选页签：批量输入 → 排序表（带隙、类型、置信度、不确定性）→ 导出 CSV。
- inverse_generator.py 明确冻结，待主线精度达标后重启。

### 5. 工程鲁棒性

- 实验 manifest 制度：reports/<run_id>/manifest.json 含数据快照 SHA-256、git commit、
  配置哈希、pip freeze、GPU/TF 版本、seed、空间群划分清单；
  "没有 manifest 的数字不许进报告"写入宪法 §4。
- 契约测试三件套：张量不变量（形状、VBM≤CBM 能量序、k_dist 范围）；
  损失单元测试（对称输入一致性损失=0；伪造正曲率 VBM 符号损失>0）；
  任意 seed 下 split 泄漏测试。
- 检查点卫生：keep-last-3 滚动（best 永久保留）。
- 训练 --resume 从断点 epoch 恢复（checkpoint 已含 optimizer 状态，仅差接线）。
- requirements*.txt 四文件职责注释（core/gpu/vision/dev），manifest 记录所用集合。

### 6. 项目实用性

- 自动报告章节对接附件1学术论文模板：方法、实验（分层数据表、多 seed 指标）、
  结果（混淆矩阵、校准曲线）、局限（DFT 低估声明、OOD 定义）。
- GUI 结果页加物理直译（如"预测带隙 1.4 eV 直接 → 类似 GaAs，适合发光器件"）。
- README 给出"从零到报告"5 条命令；run_full_pipeline.py 补齐为完整链路。
- 三条已知局限写入 README 与报告：PBE 低估、AFLOW/MP 协议差异、有效质量单位前提。

## Staging

| 阶段 | 内容 | 出口判据 |
|---|---|---|
| Stage 1 数据与基线 | 方向一全部 + manifest 制度 + 契约测试 | ≥800 AFLOW + ≥100 MP、≥60 空间群、含 metal；可复现快照哈希 |
| Stage 2 训练框架修复 | 微调归位、YAML 配置、SSL 调度/早停/混合精度、多 seed | 同一快照完整跑通，产出第一份带 manifest 的正式基线报告 |
| Stage 3 SSL 算法升级 | mask token、span、虚拟应变、损失消融、linear probe | 3-seed 指标优于 Stage 2 基线且报告可查 |
| Stage 4 功能与打磨 | 有效质量、MC Dropout、focal 闭环、批量筛选、报告对接 | 端到端 demo：GUI 批量筛选 → CSV + 报告初稿 |

## Explicit Non-Goals

- 不改 6D 张量契约与空间群 OOD 外层契约（80/20、seed=42、零交集）；
- 不重启 Gradio / 不新增并列训练框架 / 不启动逆设计；
- 不删除 PROJECT_BRAIN/、data_cache/、models/、checkpoints/、reports/ 任何内容。

## Next Action

进入 Stage 1：环境核查 → AFLOW 七层分层下载（后台长任务）→ MP key 配置与冒烟 →
manifest 与契约测试落地。执行前需用户确认下载规模与 MP key 可用性。
