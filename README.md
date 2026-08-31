# BandStructure AI Project

面向晶体能带结构的物理约束深度学习项目。项目大方向保持不变：

```text
完整 line-mode E(k)
  → (N, 2, 128, 3) 6D 物理张量
  → Masked Band Modeling 自监督预训练
  → 带隙回归 + metal/direct/indirect 分类
  → tkinter Plot-to-Physics 工作台
```

后续经批准的扩展方向是**在现有链路上**增加 crystal graph → multi-band Eₙ(k)、DFT 质检/相似检索和不确定性主动学习，不建立并列冗余框架。

## 当前状态（2026-08-28 最终验收）

### v6 metric-fix：已正式验收为 latest accepted

新实验：`aflow_noleak_v6_30k_seed42_metricfix`。GPU-only 在服务器 Tesla V100 完成 60 epochs SSL + 60 epochs 监督训练；修复后的最终 gate 通过，model-brain manifest 已发布，产物已回传并逐文件哈希校验、本地模型加载/前向通过。

| 项目 | v6 正式值（6,019 outer，独立重算） |
|---|---:|
| learned line-mode MAE / RMSE | **0.000407 / 0.007109 eV** |
| analytic identity baseline MAE / RMSE | **0 / 0 eV** |
| tensor vs global DFT MAE | 0.640224 eV |
| model vs global DFT MAE | 0.640244 eV |
| line-mode approximation R² | 0.999984 |
| type accuracy | **0.958797** |
| Macro F1 | **0.949251** |
| spacegroup-macro accuracy | **0.965795** |
| feature/label mismatch accuracy (1,257) | 0.830549 |
| MC raw 95% coverage | 0.803954 |
| selection | aggregate inner `val_loss`；best epoch 47 |
| SSL | 60 epochs，best epoch 49，val mask fraction 0.248 |

关键修复已落地并被 TDD 锁定：

- 监督 epoch 指标为完整 inner validation 聚合；`val_loss` canonical checkpoint 选择；`best/last/accepted` 三态 SHA 冻结后才允许 outer 评估；
- SSL actual mask 强制 15–30%，validation corruption 固定；magnitude-consistency hard-disabled；
- 报告日期/来源/路径全部运行时生成；MC raw/tolerance coverage 分开；
- pipeline 显式 `--experiment-id/--raw-h5/--ood-dir/--report-date`，v6 不覆盖 v4/v5；
- r3：`src/utils/selection_manifest.py` 轻量校验器（无 TensorFlow 依赖）；r4：入口脚本将项目根加入 `sys.path`（`python scripts/run_full_pipeline.py` 直接启动不再被 `scripts` 命名遮蔽）。

完整报告：`artifacts/reports/aflow_noleak_v6_30k_seed42_metricfix/v6_final_training_report.md`；回传归档 52 files / 74,071,297 bytes / SHA-256 `6735d815951b8081c114c11055ab8c5e122f1cc3ca466d0fc25c3145ef0a8184`。验收清单：`PROJECT_BRAIN/transfer_manifests/local_final_acceptance_v6_20260828.json`。

### v5 30k：字节完整，但 checkpoint-selection 已被审计判定无效

`aflow_noleak_v5_30k_seed42` 的 raw/tensor/model/predictions SHA-256 和模型加载仍全部通过；但是旧自定义 `train_step/test_step` 只把最后一个 validation batch 写入 epoch 日志，epoch 52 实际只代表最后 20 条 validation 样本。因此 v5 保留为 **immutable historical run with invalid checkpoint selection**，不能再作为无保留的科学 latest accepted。

磁盘 predictions 的真实重算结果：

| 项目 | v5 审计真值 |
|---|---:|
| AFLOW raw HDF5 / raw JSON | 30,000 / 30,000 |
| 有效 no-leak 张量 | 29,952（跳过 48） |
| Outer train/test | 23,933 / 6,019 |
| Train/test space groups / overlap | 161 / 45 / **0** |
| SSL best / stopped（历史日志事实） | epoch 39 / 54 |
| SSL val masked MSE / MAE（标准化 6D） | 0.384270 / 0.156710 |
| Supervised reported best / completed | epoch 52 / 60（**selection invalid**） |
| Outer learned line-mode MAE / RMSE | **0.000352 / 0.001864 eV** |
| Analytic line-mode identity baseline MAE / RMSE | **0 / 0 eV** |
| Outer model-vs-global-DFT gap MAE | **0.640281 eV** |
| Type accuracy / Macro F1 | **0.947333 / 0.932648** |
| Spacegroup-macro accuracy | **0.939857** |

这一 gap target 是输入 E(k) 的解析函数 `min(CBM_E)-max(VBM_E)`；模型数值只能称为 learned soft-extremum approximation，不能称为未知结构或 DFT replacement 的预测精度。v5 完整审计见 `PROJECT_BRAIN/agent_logs/20260827_training_result_reaudit_and_retrain_plan.md`，61 文件 retrospective manifest 见 `PROJECT_BRAIN/transfer_manifests/v5_consolidated_artifact_inventory_20260827.json`。

v5 raw 位于 `data/raw/aflow/snapshots/aflow_30000_20260825/`。历史 v4 6,443 raw、张量和训练 artifacts 保留；v4 model-brain manifest 时间戳重生成 incident 已单独记录，未影响 v4 raw/tensor/model/checkpoint/predictions。

## 未来下载任务可靠性治理（2026-08-26）

本轮只原地加固现有 `BandStore` / `RobustBandDownloader` / full pipeline，不重新下载数据、不重建张量、不重训模型：

- canonical HDF5 使用稳定物理语义 hash 与跨进程 single-writer lock；真实 v4/v5 legacy schema 可幂等读取，identity/provenance、`source_efermi_absolute`、缺失字段/`None` 不制造伪 variant，stored hash 不能掩盖 canonical 实际内容变化；
- 同 ID 物理冲突保留原 canonical 并隔离到 `provenance/h5_variants/`；POSIX 与原生 Windows child-process lock contention 均已验证；
- candidate catalog/cursor 与 canonical metadata sidecar 分离；rich merge、历史 duplicate conflict audit、HDF5-absent stale-ID 清理以及 metadata+provenance write-ahead transaction 保证对账无损且可恢复；
- AFLUX cursor v2 固定 `page_size`、保存每个 gap-bin 的未消费页尾，并将 page size 纳入 query fingerprint；candidate limit 不再因 gap-bin 数量越界，空 bin quota 仍可重分配；
- concurrent futures 按完成顺序立即在协调线程处理，peer `KeyboardInterrupt` 不再丢弃先完成结果；MP batch 返回数与请求数不一致时 fail-fast；HDF5 save 仍只在协调线程执行；
- report 使用真实 `persisted/target` 并输出 `target_reached`/终止原因；未达到 target 时 downloader 非零退出，`run_full_pipeline.py` 同时执行下载 count/report 前置门禁与未显式跳过 required artifacts 的最终 fail-closed 门禁。

这些修改只治理未来下载任务；latest accepted 仍为 `aflow_noleak_v5_30k_seed42`，v4/v5 immutable 数据、模型和指标没有被改写。最终验证：Stage-0 全文件 `74 passed in 21.40s`，完整 WSL 回归 `92 passed in 217.63s`，latest v5 model smoke 退出 0，原生 Windows child-process lock probe 退出 0；四个 immutable 核心文件 SHA-256/shape/group count 全部匹配，`mismatches=[]`。

## 上一正式基线（2026-08-24）

### Phase B 已完成

首个无 provider-label-conditioned feature 的正式实验：

`aflow_noleak_v4_seed42`

| 项目 | 正式结果 |
|---|---:|
| AFLOW raw cache | 6,443 |
| 有效 noleak 张量 | 6,441 |
| 空间群 | 183 |
| Outer train/test | 5,153 / 1,288 |
| Train/test group overlap | 0 |
| SSL best / early-stop | epoch 37 / 52 |
| Supervised best / early-stop | epoch 11 / 31 |
| Outer line-mode gap MAE | **0.049747 eV** |
| Outer line-mode gap RMSE | **0.159700 eV** |
| Outer R² | **0.995591** |
| Type accuracy | **0.865683** |
| Macro F1 | **0.858339** |

完整报告：

`artifacts/reports/aflow_noleak_v4_seed42/latest_training_report_20260824.md`

旧 6,399/6,443 target-conditioned 结果全部降级为 **legacy diagnostic**，未迁入当前 artifacts，不能作为正式无泄漏结论。

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

1. 当前正式方向仍以已计算 E(k) 为输入，是能带分析/表征模型，不是未知晶体结构→完整能带或 DFT replacement。
2. line-mode gap target 是输入函数；解析 baseline MAE/RMSE 为 0，learned gap head 的误差只衡量 soft-extremum approximation。
3. v5 checkpoint selection 因 last-batch metric 无效；v6 正式 V100 结果尚未产生。
4. 当前只有 seed=42；3-seed 均值/方差与 group-bootstrap 仍待后续日程。
5. 当前 outer 是 space-group-disjoint OOD，不是 composition/prototype/source OOD。
6. provider-metal/line-mode feature mismatch 在 v5 outer 为 1,257/6,019；报告必须给出 mismatch 分层性能。
7. MC-dropout 必须同时报告 raw 95% interval coverage 与 tolerance diagnostic；后者不能称为校准置信区间。

## 不可破坏约束

- 不改变 E(k)→6D→MBM→监督 heads→tkinter 主链；
- 不删除 `PROJECT_BRAIN/`、`configs/api_keys.env`、`data/raw/`、`data/processed/` 或当前正式 `artifacts/`；
- 不把 `资料/` 移入运行根目录；
- 不提交/输出真实 API key；
- outer test 不参与 early stopping、checkpoint、阈值或超参数选择；
- 结构性修改必须同步 README、dev_context、CONSTITUTION 和日期日志。

## 下一阶段

Phase B 已完成。Phase C 在执行任何代码前必须先冻结结构数据合同，随后在现有 encoder/trainer/report 链路上增加 crystal graph→multi-band sequence；详细日程见：

`PROJECT_BRAIN/agent_logs/20260824_next_work_schedule.md`
