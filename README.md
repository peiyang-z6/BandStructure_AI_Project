# BandStructure AI Project

面向晶体能带结构的物理约束深度学习项目。项目大方向保持不变：

```text
完整 line-mode E(k)
  → (N, 2, 128, 3) 6D 物理张量
  → Masked Band Modeling 自监督预训练
  → 带隙回归 + 三任务分类（line_mode_topology / provider_global_electronic_type / line_global_disagreement）
  → tkinter Plot-to-Physics 工作台
```

**差异化核心方向（2026-09-06）**：建立"晶体结构 — 数值能带 — 论文/实验能带图像"的物理约束跨模态模型，用于检索、匹配、可信拒识与主动 DFT 闭环；structure→multi-band Eₙ(k) 预测是其中一个任务，而非全部卖点。开发顺序 P1→P5：结构 sidecar 补全（P1）、跨模态检索（P2）、variable multi-band decoder（P3）、校准不确定性 + 主动获取（P4）、外部验证集（P5）。详见宪法 §8 与 `PROJECT_BRAIN/agent_logs/20260906_direction_reframing_audit.md`。

## 当前状态

**latest accepted：`aflow_noleak_v7_60k_seed42`**（2026-09-03 v7 60k 最终验收）。AFLOW 数据 **60,000 条**（55,476 主通道 + 4,524 代理通道唯一，seed 42 抽样合并；张量 59,899 样本，101 条无边缘包络跳过）。服务器 Tesla V100 GPU-only：SSL 50 epochs（early stopping）+ 监督 51 epochs（best epoch 31 by aggregate `val_loss`）+ 冻结后 evaluation-only。两次 GPU OOM 已 TDD 修复（commit `4367f87`、`ed783a8`，分块前向），全量回归 164 passed；产物回传 SHA-256 校验通过。

| 项目 | v7 正式值（11,987 outer） |
|---|---:|
| learned line-mode MAE / RMSE | **1.54e-05 / 5.54e-04 eV** |
| model vs global DFT MAE | **0.3767 eV** |
| type accuracy / Macro F1 | **0.9251 / 0.8788** |
| spacegroup-macro accuracy | **0.9296** |
| mismatch accuracy（1,723） | **0.7441** |
| MC raw 95% coverage | **0.7974** |

**P0 科研基准重构（已完成，2026-09-06）**：line-mode gap MAE 退出主结果位；标签拆为三任务 —— `line_mode_topology`（线模式路径可观察）、`provider_global_electronic_type`（uniform/DOS/数据库来源）、`line_global_disagreement`（预测二者冲突的二分类）。3 seeds {42, 2024, 7} 全链训练 + 冻结模型七类拆分评估（random / space-group / composition / prototype / leave-element / source-protocol / temporal）+ spacegroup group-bootstrap 95% CI + 错误分层四轴。完整报告见 `artifacts/reports/aflow_noleak_v7_60k_seed42/P0_scientific_reframing_report.md`。

**三任务主结果（outer OOD 11,987，3 seeds 均值±std）**：line_mode_topology `0.9932±0.0016`、provider_global_electronic_type `0.9371±0.0062`、line_global_disagreement `0.9666±0.0016`。

### P3 探索性进展与纠错（2026-09-08，未验收）

P1 已完成，P2 检索仍未验收。用户授权先行推进 P3；该限域例外已记录到宪法 5.1，不代表跨过科学验收。

排序 OT 的旧 60-epoch run 已完成，但本轮复核发现选带会漏掉费米跨越 band、padding 会改写 VBM、插值可能跨不连续分支、Keras 重载预测不一致等问题。旧模型/报告/对应代码已归档并验证双端 SHA；旧 gap 数值不作为有效精度结论。

已原地增加可选 k 点自注意力，保持 MLP 对照；实现 inner-only 选模、best/last/accepted 冻结和 evaluation-only。EF 接触、loss 溢出、prepare 完成／冒烟凭据、outer ID/group 互斥和样本轴门禁均已通过独立复审。**本地与服务器完整回归均为 456 passed**；本地 GPU 及服务器双 V100 的真实晶体端到端冒烟均通过，重载预测差为 0。

受控流程已完成并于 **2026-09-09（UTC）完成结果复核**：新版train 47,879条，outer有效11,936条；MLP stop72/best52、两层自注意力stop65/best45。相同数据/inner split、seed42、batch32及训练规则，双方冻结后才做outer评估；不代表等参数量或等实际算力。

| P3 outer指标 | MLP | 两层自注意力 |
|---|---:|---:|
| 逐k谱OT MAE（eV） | 4.083630 | 3.951537 |
| 共同可解析11,893条的gap MAE（eV） | 0.794580 | 0.899375 |
| 可解析目标零隙上的假gap率 | 31.0217% | 41.4431% |
| gap可解析覆盖率 | 99.9078% | 99.7319% |

谱误差点估计下降3.23%，但55空间群/2,000次配对bootstrap的差值95%区间跨零；gap与金属诊断退化，**不能称自注意力全面更好或P3验收通过**。45文件（3,319,777,019 bytes）已回传逐文件SHA核验；全量预测重算一致；原V100重放与本机独立重载差均0。本机GPU关键算子、last/optimizer恢复通过，但跨V100/RTX严格逐点等价（rtol=atol=1e−5）未通过，失败保留、未放宽门限。

保留模型/优化器/epoch恢复状态，但尚无CLI resume。逐k谱OT不等于轨迹匹配，scalar k不支持物理有效质量结论；Bandformer同数据对照、P3七拆分/多seed及完整物理指标仍缺。最新记录：[最终结果与限制](PROJECT_BRAIN/agent_logs/20260909_P3_controlled_final_results.md)。

记录：[纠错与审查](PROJECT_BRAIN/agent_logs/20260908_P3_reaudit_attention_execution.md) · [服务器验证与受控流程](PROJECT_BRAIN/agent_logs/20260908_P3_remote_controlled_execution.md)。旧无完成凭据的 P3 NPZ 必须重新 prepare 到新目录，冒烟输入不得作为正式数据。初次服务器环境失败已保留，最终通过没有跳过测试或伪造历史资产。

## 历史里程碑（摘要）

| 日期 | 里程碑 | 要点 |
|---|---|---|
| 2026-09-06 | **P0 科研基准重构** | 三任务标签/三 head、3 seeds、七拆分、group bootstrap、错误分层 |
| 2026-09-03 | **v7 60k 验收** | 60k 数据、v7 训练验收为 latest accepted（详见上表） |
| 2026-08-31 | Phase 6 加固 | 环境固化（requirements `==` 锁定）；GUI 状态持久化（JSON 缓存，宪法 §9 无 Gradio）；CV 置信度+质量灯；文献挖掘 pipeline（P3 → `data/raw/experimental/experimental_bands.h5`） |
| 2026-08-28 | v6 metric-fix 验收 | 修复 checkpoint-selection（aggregate inner `val_loss`、best/last/accepted 冻结）；完整回归 92→164 passed |
| 2026-08-27 | v5 审计判定 | v5 字节完整但 checkpoint-selection 无效，降为 immutable historical run |
| 2026-08-26 | 下载可靠性治理 | canonical HDF5 hash+lock、cursor v2、write-ahead 对账；只治理未来任务，不改 v4/v5 |
| 2026-08-24 | v4 首个 no-leak 基线 | 6,443 条，outer 1,288，type acc 0.8657 |

各里程碑完整报告：v7 见 `artifacts/reports/aflow_noleak_v7_60k_seed42/v7_final_training_report.md`；v6 见 `artifacts/reports/aflow_noleak_v6_30k_seed42_metricfix/v6_final_training_report.md`；v5 审计见 `PROJECT_BRAIN/agent_logs/20260827_training_result_reaudit_and_retrain_plan.md`；v4 见 `artifacts/reports/aflow_noleak_v4_seed42/latest_training_report_20260824.md`。

## 新项目根目录

运行项目已迁移至：

`C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`

论文、Word、图片、表格等资料保留在外层兄弟目录：

`C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\资料`

`资料/` 不得放进运行根目录。

## 目录结构

```text
BandStructure_AI_Project/
├── data/
│   ├── raw/
│   │   ├── aflow/
│   │   │   ├── aflow_bands.h5              # byte-identical v4 6,443 baseline
│   │   │   ├── json_cache/aflow/            # v4 6,443 原始响应
│   │   │   ├── snapshots/
│   │   │   │   └── aflow_30000_20260825/   # immutable v5 30k raw+JSON+hash manifest
│   │   │   ├── supplemental/                # 20 条早期唯一记录，隔离于正式快照
│   │   │   └── provenance/                  # baseline 恢复、变体与下载审计
│   │   └── materials_project/
│   │       ├── mp_bands.h5                  # 12 条已下载 smoke 记录合并集
│   │       └── provenance/
│   └── processed/
│       └── aflow/
│           ├── ood_tensors/                 # 历史正式 noleak_v4 张量
│           └── ood_tensors_v5_30000_seed42/ # v5 29,952 no-leak 张量
├── artifacts/
│   ├── {models,checkpoints,reports,logs}/aflow_noleak_v6_30k_seed42_metricfix/ # 新版本正式输出（训练前为空）
│   ├── {models,checkpoints,reports,logs}/aflow_noleak_v5_30k_seed42/           # immutable historical v5
│   └── {models,checkpoints,reports,logs}/aflow_noleak_v4_seed42/               # retained baseline
├── configs/
├── scripts/
├── src/
│   ├── data/
│   ├── engine/
│   ├── models/
│   ├── utils/
│   └── vision/
├── tests/
├── PROJECT_BRAIN/
├── README.md
└── requirements*.txt
```

`src` 现有模块职责不变；没有增加平行数据层、平行训练器或第二套模型框架。`scripts/` 保持稳定的扁平入口，职责分类见 `scripts/README.md`。

## 项目结构图与运行流程图

- 当前项目结构图：`PROJECT_BRAIN/diagrams/project_structure_diagram_20260827.html`
- 当前端到端流程图：`PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260827.html`
- 2026-08-24 v4 图保留为 historical snapshot；图示说明：`PROJECT_BRAIN/diagrams/README.md`

两张图均为静态、可缩放、离线可打开的 HTML/SVG。

## 数据与张量合同

原始记录至少包含：

- `energies`: `(bands,k)` 或 `(spin,bands,k)`，单位 eV；
- 真实累计 `k_distances` 与 segment/symmetry 信息；
- `spacegroup_number`、source、material ID、URL 与校验信息；
- provider gap/type 标签仅用于 target/audit。

固定张量：

```text
X: (N, 2, 128, 3)
band:    [occupied edge envelope, empty edge envelope]
channel: [energy, curvature, normalized distance to extremum]

flatten → (N, 128, 6)
[VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]
```

强制合同：

- AFLOW `bands_data` 使用 canonical `E_F=0`；原始绝对 `Efermi` 仅作 provenance；
- edge envelopes 只由 E(k)+E_F 构造；provider 标签不得改变输入、锚定、mask 或插值；
- 使用 PCHIP shape-preserving interpolation；
- curvature、crossing、span masking 和物理损失均不得跨 k-path segment；
- outer split 为 spacegroup 80/20、seed=42、零交集；outer test 不参与模型选择；
- composition overlap、prototype overlap、label/feature mismatch 与 SHA-256 必须进入 manifest。

最新 v5 raw HDF5 SHA-256：

`d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`

最新 v5 split NPZ SHA-256：

`c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`

保留的 v4 raw/split SHA-256：`bb261f1e…b62d` / `c6652b85…ae59`。

## WSL2 + Conda 快速开始

```bash
conda activate bandstructure-ai
cd /mnt/c/Users/PeiYang/Documents/'AI Project'/'BandStructure AI Project'/BandStructure_AI_Project

# 完整回归测试
python -m pytest -q

# 最新模型真实加载与推理 smoke
CUDA_VISIBLE_DEVICES='' python tests/smoke_latest_model.py
```

`scripts/run_full_pipeline.py --source aflow` 保留为 v4 复现实验入口，不是 v6 正式命令。v4/v5 都是 immutable snapshot；继续训练必须显式给出新的 experiment ID 与只读输入。

### v6 GPU-only 正式命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_full_pipeline.py \
  --source aflow \
  --experiment-id aflow_noleak_v6_30k_seed42_metricfix \
  --raw-h5 data/raw/aflow/snapshots/aflow_30000_20260825/aflow_bands.h5 \
  --ood-dir data/processed/aflow/ood_tensors_v5_30000_seed42 \
  --target 30000 \
  --report-date 20260827 \
  --ssl-epochs 60 --ssl-batch-size 32 \
  --mask-ratio 0.25 --sign-weight 2.0 --consistency-weight 0.0 \
  --finetune-epochs 60 --finetune-batch-size 32 \
  --learning-rate 0.001 --encoder-learning-rate 0.00001 \
  --type-weight 2.0 --freeze-layers 2 \
  --topology-weight 0.3 --entropy-weight 0.02 --extremum-weight 1.0 \
  --fresh --force-ssl --force-finetune --skip-vision --require-gpu
```

该命令不得在 CPU 或未通过 Conv1D GPU forward/backward preflight 的环境运行。

### 重现 v4 基线张量（仅复现，不改 v5）

```bash
python scripts/build_ood_tensors.py \
  --h5 data/raw/aflow/aflow_bands.h5 \
  --metadata data/raw/aflow/aflow_metadata.json \
  --output data/processed/aflow/ood_tensors
```

### 启动 tkinter 工作台

```bash
python scripts/gui_workbench.py
```

当前物理模型链路已验收；最新 vision detector 权重在清理前目录中不存在，因此自动图像检测属于可选未验收能力，人工标定与物理模型调用仍保留。

## 环境

- WSL2 / Ubuntu 24.04
- Conda env：`bandstructure-ai`
- Python 3.11
- TensorFlow 2.21
- 本地 RTX 4060；服务器 2×V100 16GB
- `requirements-gpu.txt`：核心 GPU 链路
- `requirements-vision.txt`：可选 YOLO/Ultralytics

Materials Project 密钥只允许放在未跟踪文件 `configs/api_keys.env`。模板为 `configs/api_keys.env.example`。不得输出、写入报告或提交真实密钥。

## 当前科学边界

1. 当前正式方向仍以已计算 E(k) 为输入，是能带分析/表征模型；结构→多能带预测是 P3 阶段目标，尚未实现。
2. line-mode gap target 是输入函数；解析 baseline MAE/RMSE 为 0，learned gap head 的误差只衡量 soft-extremum approximation。**P0 起 line-mode gap MAE 不再是主结果**；主结果是三任务（line_mode_topology / provider_global_electronic_type / line_global_disagreement）。
3. v5 checkpoint selection 因 last-batch metric 无效（immutable historical）；v6/v7 均以 aggregate inner `val_loss` 为 canonical selection。
4. P0 已落地 3 seeds {42, 2024, 7} 与 spacegroup group-bootstrap 95% CI（`src/evaluation/bootstrap_stratify.py`）。
5. P0 已固定七类拆分（random / space-group / composition / prototype / leave-element / source-protocol / temporal，见 `src/data/benchmark_splits.py`），冻结模型跨拆分评估。
6. provider/line-mode 冲突以三任务中的 `line_global_disagreement` 显式建模（冲突率 12.4%）。
7. MC-dropout raw 95% coverage 0.7974 尚不足以驱动主动学习（P4 才升级为校准不确定性）；当前不得把 MC 方差当 DFT 选择依据。
8. HDF5 尚无 lattice/species/fractional_coordinates——结构字段缺失是 P1 的数据合同问题（审计见 `20260906_direction_reframing_audit.md`）。

## 不可破坏约束

- 不改变 E(k)→6D→MBM→监督 heads→tkinter 主链；
- 不删除 `PROJECT_BRAIN/`、`configs/api_keys.env`、`data/raw/`、`data/processed/` 或当前正式 `artifacts/`；
- 不把 `资料/` 移入运行根目录；
- 不提交/输出真实 API key；
- outer test 不参与 early stopping、checkpoint、阈值或超参数选择；
- 结构性修改必须同步 README、dev_context、CONSTITUTION 和日期日志；
- 不修改既有 immutable HDF5（结构字段走只读 sidecar）；
- 不单纯扩数据到 100k，不从零实现 DeepH 类哈密顿量网络。

## 下一阶段

按宪法 §8 的 P1→P5 顺序推进。当前 P0 已完成（三任务基准 + 3 seeds + 七拆分 + group bootstrap + 错误分层）。下一步是 **P1：为 60k 数据补全结构 sidecar**（lattice/species/fractional_coordinates 等，AFLOW REST 端点实测可达）。完整方向评估见：

`PROJECT_BRAIN/agent_logs/20260906_direction_reframing_audit.md`
