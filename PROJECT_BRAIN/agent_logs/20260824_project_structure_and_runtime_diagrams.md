# 2026-08-24 — Project Structure and Runtime Flow Diagrams

## Scope

根据迁移后的真实磁盘目录、`README.md`、`scripts/README.md`、`data/README.md`、`artifacts/README.md` 与宪法生成两张详细技术图；未改变代码、数据、模型或训练结果。

## Artifacts

- `PROJECT_BRAIN/diagrams/project_structure_diagram_20260824.html`
- `PROJECT_BRAIN/diagrams/project_structure_diagram_20260824.png`
- `PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260824.html`
- `PROJECT_BRAIN/diagrams/runtime_flow_diagram_20260824.png`
- `PROJECT_BRAIN/diagrams/README.md`

## Diagram contracts

### Structure diagram

- 资料目录保持在 runtime root 外；
- 入口脚本与核心 `src` 实现分离；
- raw/processed 与 four-way artifacts 清晰分类；
- formal experiment ID 统一为 `aflow_noleak_v4_seed42`；
- 显示 AFLOW 6443 canonical、20 supplemental、MP 12、processed 6441；
- 显示 39 tests、model smoke 与 Phase C pending。

### Runtime diagram

- 明确数据获取、张量构建、训练评估和在线推理四条泳道；
- 显示 provider-label-invariant features、segment-aware processing 与 group-disjoint split；
- outer OOD test 使用独立红色路径绕过训练，仅进入 final evaluation；
- 显示 MBM best/stop=37/52、supervised best/stop=11/31；
- 显示 tkinter feedback loop 和 optional/missing vision detector；
- 明确当前输出是 line-mode gap/type/QA，不是 structure→full-band。

## Verification

- Edge headless 实际渲染两张 HTML；退出码 0；
- 结构图截图：1800×1700，250,564 bytes；
- 流程图截图：1800×2000，305,343 bytes；
- 视觉回验未发现文本裁切、框体重叠、箭头越界或图例溢出；
- HTML 文件为纯静态单文件，SVG 使用固定 viewBox，可浏览器缩放；
- 图中路径和数字均来自当前磁盘事实，未引用已删除 legacy artifacts。
