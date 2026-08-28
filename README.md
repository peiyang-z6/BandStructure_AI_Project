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

## 30k 正式状态（任务日期 2026-08-25，跨午夜完成）

最新已接受实验：`aflow_noleak_v5_30k_seed42`。30k immutable raw、no-leak tensor、GPU-only SSL/监督训练、outer OOD、回传哈希与本地模型 smoke 均已完成；完整报告：

`artifacts/reports/aflow_noleak_v5_30k_seed42/latest_training_report_20260825.md`

| 项目 | v5 正式验收 |
|---|---:|
| AFLOW raw HDF5 / raw JSON | 30,000 / 30,000 |
| Raw HDF5 SHA-256 | `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3` |
| 有效 no-leak 张量 | 29,952（跳过 48） |
| Outer train/test | 23,933 / 6,019 |
| Train/test space groups / overlap | 161 / 45 / **0** |
| SSL best / stopped | epoch 39 / 54 |
| Supervised inner best / completed | epoch 52 / 60 |
| Outer line-mode gap MAE / RMSE | **0.000352 / 0.000595 eV** |
| Outer model-vs-global-DFT gap MAE | **0.640281 eV** |
| Type accuracy / Macro F1 | **0.941186 / 0.932648** |
| GPU evidence | Tesla V100；max 89%；15,114 MiB；无 CPU fallback |

v5 raw 位于 `data/raw/aflow/snapshots/aflow_30000_20260825/`。历史 v4 6,443 byte-identical raw、张量和训练 artifacts 全部保留为可复现基线，未被覆盖。

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
│   ├── {models,checkpoints,reports,logs}/aflow_noleak_v5_30k_seed42/ # latest accepted GPU-only v5
│   └── {models,checkpoints,reports,logs}/aflow_noleak_v4_seed42/     # retained baseline
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

- 详细项目结构图：`PROJECT_BRAIN/diagrams/project_structure_diagram_20260824.html`
- 详细运行流程图：`PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260824.html`
- 图示说明：`PROJECT_BRAIN/diagrams/README.md`

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

`scripts/run_full_pipeline.py --source aflow` 保留为 v4 复现实验入口，不是 v5 latest 状态命令。v4/v5 都是 immutable snapshot；继续扩容必须先建立新的 snapshot/experiment ID，不得向二者原地追加。

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

1. 当前正式模型以已计算 E(k) 为输入，是能带分析/表征模型，不是未知晶体结构→完整能带预测器。
2. v5 的 0.000352 eV 是 line-mode tensor-gap MAE；对应 model-vs-global-DFT gap MAE 为 0.640281 eV，均不是实验带隙误差。
3. 当前是单 seed=42，尚缺 3-seed 均值/方差。
4. space-group OOD 不等于 composition/prototype/source OOD；v5 composition overlap=1,474。
5. v5 direct-gap recall=0.943662；仍需用多 seed 和更多 OOD 维度确认稳定性。
6. MC-dropout coverage 尚未做严格 calibration，暂不能直接驱动高成本 DFT 队列。

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
