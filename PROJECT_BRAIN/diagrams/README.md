# BandStructure AI Project Diagrams

本目录保存迁移后 canonical 项目的可缩放、离线 HTML/SVG 图。

## 1. 项目结构图

- 可缩放版本：`project_structure_diagram_20260824.html`
- 快速预览：`project_structure_diagram_20260824.png`

展示：

- 外层 `资料/` 与运行根目录的隔离；
- `PROJECT_BRAIN`、配置、测试和 Git 治理；
- `scripts/` 与 `src/{data,models,engine,utils,vision}` 的职责；
- AFLOW/MP raw 数据和 formal processed tensor；
- 统一实验 `aflow_noleak_v4_seed42` 的 models/checkpoints/reports/logs；
- 当前数据量、验证状态和 Phase C 边界。

## 2. 运行流程图

- 可缩放版本：`runtime_flow_diagram_20260824.html`
- 快速预览：`runtime_flow_diagram_20260824.png`

按四条泳道展示：

1. 数据获取：AFLOW/MP → downloader → adapters → band store → raw/provenance；
2. 张量合同：E_F=0、edge envelopes、PCHIP、segment-aware features、6D tensor、group split；
3. 训练评估：inner split → MBM → SSL encoder → supervised → artifacts → final outer OOD；
4. 在线推理：E(k)/论文图 → 校准与重建 → 6D → PhysicsBrainInvoker → gap/type/QA → tkinter。

## 科学闸门

- provider labels 只能作为 target/audit；
- outer OOD test 只在最终评估进入；
- supplemental/variant 数据不混入 formal snapshot；
- 当前模型是 E(k) 分析器，不是 structure→multi-band predictor；
- vision detector 权重当前不存在，图中明确标为 optional/missing。

两张图均为单文件 HTML，CSS 与 SVG 内联，无 JavaScript，可直接在现代浏览器离线打开并缩放。
