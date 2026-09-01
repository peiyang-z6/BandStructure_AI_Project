# Phase 6 执行日志：系统加固与数据飞轮（2026-08-31）

状态：完成。方案：`20260831_phase6_execution_plan.md`（用户已批准三项决策：锁实测版本 / 宪法兼容路径 / v6 指针随本轮修复）。

## 验证基线

- 全量回归：`compileall` exit 0；`154 passed in 28.20s`（Phase 6 前为 136；新增 18 项：P1×7、P2×6、P3×5）。
- `pip check` 无冲突；27 个 requirements pin 与 `pip freeze` 逐项一致。
- 端到端文献挖掘 smoke（真实 CV + v6 Brain + h5 + 报告）：2 张合成图 → stored 1 / low_confidence 1 / failed 0；成功提取率 50.0%、平均置信度 0.785、Direct 候选 0；h5 group `clean_band` attrs `cv_confidence=0.8343`、`line_mode_gap_ev=0.0475`、`predicted_type=indirect`。注：合成图无 Fermi 人工标定，brain gap≈0 是未标定输入的预期表现，不构成缺陷。

## P0 环境固化

- `requirements.txt` 重写为 27 项 `==` 精确锁（核心：tensorflow==2.21.0, keras==3.15.1, mp-api==0.46.4, emmet-core==0.87.1, numpy==1.26.4, h5py==3.14.0, scipy==1.17.1, scikit-learn==1.9.0, pandas==3.0.5, protobuf==7.35.1, pymatgen==2026.5.4, pillow==12.3.0, matplotlib==3.11.1, opencv-python-headless==4.11.0.86, pymupdf==1.28.2, optuna==4.9.0, plotly==6.9.0, requests==2.34.2, python-dotenv==1.2.2, pyyaml==6.0.3, pytest==8.4.2, pytest-cov==7.1.0, joblib==1.5.3, spglib==2.7.0, monty==2026.7.16, tensorboard==2.21.0, pymatgen-core==2026.7.31）。
- `environment.yml`：`conda env export --no-builds`（name bandstructure-ai，27 conda + 120 pip 依赖，Python 3.11.15）。
- 决策记录：用户指令示例 2.16.1/3.3.3 与本环境不符；经批准锁定实测 2.21.0/3.15.1（v6 验收运行时）。

## P1 GUI 状态持久化

- `scripts/gui_workbench.py`：
  - `annotations_from_json()`：JSON → 画布内存结构（panel/vbm/cbm 元组、strokes/axis 坐标元组列表）；
  - `WorkbenchStateStore`：目录 `data/annotations/workbench_state/`；`image_key` = 图像内容 SHA-256（归一化 PNG 字节）+ 尺寸；原子写（`.tmp` → `replace`）；损坏 JSON/缺失文件返回 None；
  - `DrawingCanvas(on_change=...)`：每次重绘后回调；`restore_annotations()`；
  - `BandStructureWorkbench`：`_schedule_state_save`（800ms 防抖）、`_autosave_state`（换图前与关闭时）、`_restore_state_if_present`（加载后恢复标注+定标+材料信息，状态栏提示）；
  - `_render_pdf_to_pil` 拆分使 PDF 与位图统一拿到 PIL 对象以计算 key。
- TDD：`tests/test_workbench_state.py` 7 项 RED→GREEN。

## P2 CV 置信度与不确定性

- `src/vision/multi_format_parser.py`：
  - `cv_quality_level()`：≥0.6 green / ≥0.35 yellow / 否则 red；
  - `_compute_cv_quality()`：panel 来源可靠性（分档表）+ detector 置信度（仅权重存在时，否则 `optional_missing_score_excluded`）+ frame 裁剪 + 骨架密度 + k 向列占位 + 面板分辨率 + 骨架点数，加权合成 0–1；
  - `parse_image` 与 `parse_vector` 都写入 `metadata["cv_quality"]`。
- `src/vision/brain_invoker.py`：
  - `compute_brain_uncertainty()`：type softmax 归一化熵 + VBM/CBM 峰锐度（1−mean/peak 取最差）+ gap 物理合理性；明确不用 MC-Dropout（eval 路径无 dropout 层，方差恒 0 会误导）；
  - `predict()` 将结果写入 `metadata["confidence"]`。
- `scripts/gui_workbench.py`：右侧面板新增"提取质量指示灯"（绿/黄/红 + 分数），黄/红显示指定警告文案；Recognition 结果追加 Confidence 行，gap 越界时追加物理范围警告。
- TDD：`tests/test_cv_confidence.py` 6 项 RED→GREEN（含清晰 vs 加噪图像分数排序、失败提取记 0 分、等级边界、熵/峰锐度排序、gap 合理性）。

## P3 文献挖掘 Pipeline

- `scripts/literature_mining_pipeline.py` 原地升级（用户批准不新建 `literature_mining.py`/不建根级 `data_cache/`）：
  - `iter_raster_images()`：目录内 png/jpg/jpeg/bmp/tiff 直接遍历（与 PDF 页内图合并）；
  - `summarize_records()`：成功率（stored/total）、平均 CV 置信度（有分数的记录）、潜在 Direct Gap 候选数（stored 且 predicted_type=direct）；
  - 每记录写入 `cv_score` + `confidence` 含 `cv_score/cv_level/brain_score`；
  - `store_experimental_band` h5 attrs 新增 `cv_confidence`；
  - `write_report` 改名为《文献挖掘摘要报告》，表格加 CV score 列，Summary 加三项统计。
- 默认输出：`data/raw/experimental/experimental_bands.h5`（宪法 §3 兼容）。
- TDD：`tests/test_literature_mining.py` 5 项 RED→GREEN（其中 1 项为测试自身 key 假设修正：h5 key 为文件 stem）。

## 附带修复

latest-model 指针 v5→v6（用户批准随本轮）：`src/vision/brain_invoker.py` 默认四路径、`scripts/gui_workbench.py` TSNE_IMAGE、`tests/smoke_latest_model.py` EXPERIMENT_ID → `aflow_noleak_v6_30k_seed42_metricfix`；`tests/test_project_layout.py` 三处期望同步更新（RED→GREEN）。

## 治理同步

- `README.md`：新增"环境固化（Phase 6 P0）"与"当前状态（2026-08-31 Phase 6）"两节；
- `PROJECT_BRAIN/dev_context.md`：新增 "### Phase 6 启动" 节，含用户指定原句；
- `PROJECT_BRAIN/CONSTITUTION.md`：Version 4.10→4.11；§9 新增工作台状态持久化、CV 质量分、脑不确定性纪律三条；
- 本日志 + `20260831_phase6_execution_plan.md`。

## 遗留 / 边界

- 无 vision detector 权重（`artifacts/models/vision_detector/` 不存在）：CV 质量分不含 detector 项、GUI 与报告均如实标注 unverified；detector 权重训练不在本阶段范围。
- GUI 无头环境无法自动化点击流；持久化与质量灯逻辑由纯函数/存储层测试覆盖，UI 手工验证待 WSL GUI 会话。
- 文献挖掘真实 PDF/PNG 语料运行（含 Fermi 标定图）留作下一轮数据飞轮任务。
- `data/raw/experimental/` 目录在首次真实运行 `store_experimental_band` 时创建。
