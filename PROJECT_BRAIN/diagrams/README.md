# BandStructure AI Project Diagrams

本目录保存 canonical 项目的可缩放、离线 HTML/SVG 图及 PNG 快速预览。

## 当前图（2026-08-27，v6 metric-fix）

### 1. 项目结构图

- 可缩放版本：`project_structure_diagram_20260827.html`
- 快速预览：`project_structure_diagram_20260827.png`

展示：

- AFLOW canonical store、immutable v4/v5 snapshots；
- `E(k) → (N,2,128,3) → Transformer/MBM → supervised heads → GUI` 单一主架构；
- v6 models/checkpoints/reports/logs 的版本化边界；
- best / last / accepted-restored-best 三态；
- Windows+WSL 本地验收与单张 V100 GPU-only 正式训练边界；
- Git tag、sync manifest、SHA-256、PROJECT_BRAIN 治理；
- Phase C crystal-structure predictor 明确标为 **pending**。

### 2. 端到端流程图

- 可缩放版本：`runtime_flow_diagram_20260827.html`
- 快速预览：`runtime_flow_diagram_20260827.png`

按四条泳道展示：

1. immutable v5 raw/tensor 输入与 space-group-disjoint outer split；
2. inner-only normalization、MBM、GPU gate、监督 train-only 与 aggregate `val_loss` 选模；
3. best/last/accepted hash 冻结后，evaluation-only 才加载 outer test；
4. server artifact manifest → 本地回传 → 重算指标/模型加载 → latest promotion/GUI。

图中明确区分：

- analytic line-mode tensor gap（输入函数，解析 baseline 零误差）；
- learned soft-extremum approximation；
- provider global DFT residual；
- sample accuracy、spacegroup-macro、feature/label mismatch strata；
- raw 95% interval coverage 与 tolerance diagnostic。

## 历史图（2026-08-24，v4 snapshot）

以下文件保留为不可改写的历史结构/流程快照，不再代表当前 latest：

- `project_structure_diagram_20260824.html`
- `project_structure_diagram_20260824.png`
- `runtime_flow_diagram_20260824.html`
- `runtime_flow_diagram_20260824.png`

它们记录 v4、6,443 raw、6,441 tensors 和当时的 artifact/test 状态。

## 当前科学闸门

- provider labels 只能作为 target/audit，不能改变特征；
- outer OOD test 不参与 normalization、checkpoint、early stopping 或调参；
- outer OOD 的准确名称是 **space-group-disjoint OOD**；
- SSL actual mask fraction 必须在 15–30%，validation corruption 固定；
- 无物理 k-coordinate 时 curvature-magnitude consistency 保持 hard-disabled；
- whole-path virtual-strain heuristic 默认关闭；
- 正式训练关键 tensor 必须位于 NVIDIA GPU，CPU fallback 立即失败；
- v4/v5 raw/tensor/model artifacts 不覆盖；v6 使用新 experiment ID。

所有 HTML 均为单文件，CSS 与 SVG 内联，无 JavaScript，可直接在现代浏览器离线打开并缩放。PNG 已以 1600×1500 重渲染并完成裁切/重叠视觉检查。
