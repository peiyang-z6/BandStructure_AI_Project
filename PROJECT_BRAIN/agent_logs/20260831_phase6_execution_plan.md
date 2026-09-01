# Phase 6 执行方案：系统加固与数据飞轮（2026-08-31）

状态：方案待批准。本文件为新增的唯一项目文件，其余项目文件均未修改。

## 0. 磁盘审计事实（先于方案，docs vs disk 已核对）

### 环境（WSL2 conda `bandstructure-ai`, Python 3.11.15, 实测 pip list）
| 包 | 实测版本 | 包 | 实测版本 |
|---|---|---|---|
| tensorflow | 2.21.0 | mp-api | 0.46.4 |
| keras | 3.15.1 | emmet-core | 0.87.1 |
| numpy | 1.26.4 | pymatgen | 2026.5.4 |
| h5py | 3.14.0 | pandas | 3.0.5 |
| scipy | 1.17.1 | protobuf | 7.35.1 |
| scikit-learn | 1.9.0 | requests | 2.34.2 |
| matplotlib | 3.11.1 | joblib | 1.5.3 |
| pillow | 12.3.0 | optuna | 4.9.0 |
| pytest | 8.4.2 | tensorboard | 2.21.0 |
| opencv (cv2) | 4.11.0 | pymupdf | 1.28.2 |
| pip / setuptools / wheel | 26.2.1 / 84.0.0 / 0.47.0 | | |

- 用户指令中的示例版本（tensorflow==2.16.1, keras==3.3.3）**与本环境不符**。v6 正式验收训练运行于 tf 2.21.0 / keras 3.15.1，锁到示例版本会破坏已验证运行时的可复现性。故按"检查实际版本→锁定实际版本"执行，锁 2.21.0/3.15.1。
- `ultralytics`/`torch` 未安装——与宪法 §9（vision detector 可选/未验收）一致；`artifacts/models/vision_detector/` 无权重文件。P0 不引入它们。
- Windows 原生 Python 无 numpy/PIL/cv2；GUI/CV 全程在 WSL 环境运行（记录于 README/environment.yml 注释）。

### 代码事实
- GUI：`scripts/gui_workbench.py`（1,249 行，原生 tkinter，宪法 §9 禁止 Gradio/web）。标注全部存图像像素坐标；定标参数为 `_y1/_y2/_x1/_x2_var` + 材料 ID/标签；**无任何持久化**。
- CV：`src/vision/multi_format_parser.py`（1,010 行）。几何 panel 检测（line/gray/dark 多种）+ 可选 YOLO-pose 检测；骨架提取；metadata 含 detector 置信度（仅当权重存在），**无聚合质量分**。
- Brain：`src/vision/brain_invoker.py`。默认模型路径仍指 **v5**（v6 已是 latest accepted，指针过期；`tests/test_project_layout.py` 62-73 行断言 v5）。type 分类概率已有，可用熵+峰值锐度做不确定性（模型 eval 路径无 dropout，MC-Dropout 不适用——如实实现）。
- 文献挖掘：`scripts/literature_mining_pipeline.py` 已存在（PDF→图→parse→reconstruct→brain→h5→摘要），**缺**：直接遍历 PNG/JPG、CV 置信度、平均置信度/成功率/直接带隙候选统计。
- 宪法硬约束：§3 禁止根级 `data_cache/`；`tests/test_project_layout.py` 的 `test_runtime_code_contains_no_deprecated_root_paths` 会拒绝在 src/scripts 中出现 `data_cache` 字样。

## 1. [P0] 环境固化与依赖锁死

1. 重写 `requirements.txt`：全部核心依赖 `==` 精确锁定（上表实测值），按功能区注释（核心/材料 API/报告/GUI/Plot-to-Physics/开发），`--no-deps` 之外的安装即刷到可复现状态。
2. 生成 `environment.yml`：`conda env export` 全量精确快照（Python 3.11.15 + 全部 pip 依赖的 conda/pip 双段），目标：`conda env create -f environment.yml` 可重建。
3. 验证：`pip check`、`pip install --dry-run -r requirements.txt`（确认所有 pin 可解析且与现有环境一致）、`conda env export` 往返校验；README 增加复现说明段（WSL 运行 GUI/CV 的说明）。
4. 不改动 conda 环境本身（只固化描述文件）。

## 2. [P1] GUI 工作台状态持久化（tkinter + 后端 JSON 缓存）

按宪法 §9 用"后端 JSON 文件缓存"分支（无浏览器 localStorage / 无 Gradio）：

1. `gui_workbench.py` 内新增 `WorkbenchStateStore` 类：
   - 目录 `data/annotations/workbench_state/`（宪法兼容，训练标注同级新子目录）；
   - key = 图像内容 SHA-256（前 256KB）+ 尺寸；另存路径 registry（同一张图改名/复制后仍能命中）；
   - state = 全部 annotations（图像像素坐标，缩放不变）+ 定标值 y1/y2/x1/x2 + 材料 ID/gap 标签；
   - 原子写（临时文件 + 同目录 rename）。
2. 触发点：标注变化 → 800ms 防抖自动保存；换图前保存当前图；窗口关闭时保存；加载图片后按 key 恢复（无人为确认弹窗，状态栏提示"Restored previous session"）。
3. TDD：先 RED `tests/test_workbench_state.py`（save/restore 往返、content-key 命中、损坏 JSON 容错、未知图 no-op、定标浮点值往返），再 GREEN。

## 3. [P2] CV 提取置信度与不确定性反馈

1. `multi_format_parser.py`：
   - `parse_image` 后调用新增 `_compute_cv_quality(panel_meta, skeleton, panel)` 生成 `metadata["cv_quality"]`：
     分量 = detector 置信度（若有权重；无权重记 unverified）、panel 检测来源质量（轴框裁剪 vs fallback 全域）、骨架密度（曲线像素/panel 面积）、列占位（k 向覆盖）、面板分辨率（px 高）、骨架点数；
     加权合成 0–1 分 → 绿(≥0.6)/黄(≥0.35)/红(<0.35)。
   - 不伪造 YOLO：无权重时明确 `detector_status: optional_missing_score_excluded`。
2. `brain_invoker.py`：
   - 新增 `compute_brain_uncertainty(prediction)`：type-prob 归一化熵 + VBM/CBM 峰值锐度（peak vs mean）+ gap 物理合理性；
   - 输出 `BrainPrediction.metadata["confidence"]`。**不使用 MC-Dropout**（eval 路径无 dropout 层，方差恒 0 会误导）。
3. GUI 右侧面板：新增"提取质量指示灯"标签行（● 绿/黄/红 + 分数），CV 运行后与 Recognize 后分别更新；黄/红时显示用户指定警告文案："⚠️ 图像退化严重/特征模糊，提取结果可能存在误差，建议人工仔细复核定标点。"
4. TDD：合成 band 图（清晰 vs 加噪 vs 无曲线）→ 三级标签断言；脑置信度对高/低熵分布断言。

## 4. [P3] 文献挖掘 Pipeline

按"系统化原地升级"原则，**升级现有 `scripts/literature_mining_pipeline.py`**（即用户所说批处理脚本，避免平行模块；如需入口别名可另行说明）：

1. 输入遍历：目录下 `*.pdf`（页内图像抽取，已有）+ 直接 `*.png/*.jpg/*.jpeg/*.bmp`。
2. 每图记录 CV 置信度（P2 输出）+ 物理自洽置信度（已有 `physics_confidence`）合并为 record.confidence。
3. h5 默认路径：`data/raw/experimental/experimental_bands.h5`（宪法兼容，替代用户指令中的根级 `data_cache/`——若坚持 `data_cache/` 需同时在宪法 §3 与 deprecation 测试中豁免 `data_cache`，见下方待决点）。attrs 增加 `cv_confidence` 与 `mean_confidence`。
4. 《文献挖掘摘要报告》（markdown）新增：成功提取率、平均置信度、潜在 Direct Gap 材料数量；`literature_mining_summary.json` 同步字段。
5. TDD：小合成图集 → 统计字段正确、h5 attrs 正确、失败图计入分母（成功率口径）。

## 5. 附带修复（小改，不改大方向）：latest-model 指针 v5 → v6

`brain_invoker.py` 默认路径、`gui_workbench.py` t-SNE 图路径、`smoke_latest_model.py` 指向 v6（`aflow_noleak_v6_30k_seed42_metricfix`，文件已核实存在），并同步更新 `test_brain_invoker_defaults_resolve_to_latest_formal_artifacts`、`test_gui_tsne_path_targets_latest_formal_experiment`、`test_latest_model_smoke_targets_v5_portable_paths` 的期望值。

## 6. 治理闭环（宪法 §10）

结构性修改同步：README、dev_context（写入用户指定原句："Phase 6 启动：完成环境固化、GUI 状态持久化、CV 不确定性估计，并建立文献挖掘 Pipeline 原型。"）、宪法（§3 若按用户 data_cache 豁免则改；§7/§9 增补置信度纪律：无 detector 权重时不得声称已验收）、日期日志 `20260831_phase6_execution.md`。验证：compileall + 全量 pytest + GUI/CV WSL smoke + git diff 行尾检查。提交并打 tag `phase6-hardening-20260831`。

## 非目标（本阶段不做）

- 不重训、不动 v6 artifacts 与数据目录；
- 不引入 Gradio/web GUI、不安装 torch/ultralytics；
- 不用 MC-Dropout 假装不确定性；
- 不写任何凭据路径/密钥。

## 待批决策

D1. P0 版本基准：锁实测（tf==2.21.0 / keras==3.15.1，与 v6 验收一致）——推荐；还是按示例降级（2.16.1/3.3.3）。
D2. P3 输出路径与脚本载体：升级现有 `literature_mining_pipeline.py`、输出 `data/raw/experimental/experimental_bands.h5`（宪法兼容）——推荐；还是按字面新建 `scripts/literature_mining.py` + 根级 `data_cache/` 并豁免宪法/测试。
D3. latest-model 指针 v5→v6 修复是否随本轮一并执行——推荐一并修复。