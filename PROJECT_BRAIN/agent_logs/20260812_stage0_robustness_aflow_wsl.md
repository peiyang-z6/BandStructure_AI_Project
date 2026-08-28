# 2026-08-12 Stage 0: Robustness, AFLOW, WSL2

## Scope

在不改变项目核心 4 步与 Phase 5 方向的前提下，修复已知数据/评估/GUI 问题，接入
第二个完整能带源，并建立 WSL2 Conda 冒烟基线。

## Confirmed Problems

1. 仓库内容曾整体位于重复嵌套目录；经逐文件哈希比对后恢复到 Git 根目录。
2. SSL 与监督微调使用外层 OOD test 做 validation/checkpoint，构成评估泄漏。
3. 旧 split 声称 stratified，实际只随机 group 后截断，并以 round(gap) 近似类别。
4. 默认 `min_gap=0.1` 排除金属，与三分类目标矛盾。
5. 断点续传目标按“新增数”计算，已有缓存时会超额下载。
6. 曲率按等距索引计算，忽略真实 k-path distance；spin-polarized 数据只取首个自旋。
7. type label 从相同输入曲线推导却被当作独立监督标签。
8. GUI 反向标定分母错误、Fermi 线未参与零点、曲率倒数误标为 `m_e` 有效质量。
9. 过时 Gradio 启动脚本与宪法冲突，human vision split 存在同源泄漏风险。
10. 旧 requirements 的 `emmet-core<0.84` 与当前 mp-api 版本线不兼容。

## Changes

- 增加 AFLOW AFLUX E(k) 数据源和通用持久化层；MP/AFLOW 共用下载入口但保留独立缓存。
- 下载加入退避重试、SHA-256、JSON cache、HDF5 临时 group 写入与上游空文件分类。
- 修复总目标计数、真实 k 坐标、全自旋边缘态选择、provider type labels。
- 修复 group split，并建立外层 train 内的 inner validation；外层 test 只最终评估。
- 修复 GUI 校准、Fermi 零点、置信度和有效质量标签；禁用 Gradio。
- 更新 WSL/Python 3.11 requirements、README、constitution、dev context 与测试。

## Verification Evidence

- WSL regression tests: 12 passed.
- Python compilation: all `src/`, `scripts/`, `tests/` Python files passed.
- An AFLUX interval bug was found during evidence review: `min*,max*` clustered results
  at/above the upper threshold. It was corrected to the documented AFLUX interval form
  `min*,*max`; the `natoms` filter is now applied server-side as well.
- AFLOW candidate metadata is queried in gap strata and interleaved, preventing small
  targets from being composed only of rows nearest the lower interval boundary.
- A 5-record live stratified smoke covered 0.2001, 0.2603, 0.3205, 0.3805 and 0.4404 eV;
  all records also satisfied `natoms <= 50`.
- Corrected AFLOW resume smoke: 12 valid cached records in the 0.2--0.5 eV query;
  7 unusable upstream empty-band entries were isolated without terminating the run.
- Corrected tensor build: `(12, 2, 128, 3)`, 11 spacegroups, 10/2 split, zero overlap.
- First TensorFlow 2.21 smoke reached the training loop and exposed that TensorBoard is no longer
  installed transitively; `tensorboard` was made an explicit runtime dependency before rerun.
- The corrected smoke dataset is isolated under `data_cache/stage0_wsl_range/` and is not a performance claim.
- TensorFlow 2.21 enumerated `/physical_device:GPU:0`; corrected-data `--require-gpu`
  1-epoch MBM completed,
  restored the best checkpoint and exported both Keras models. A fresh-process reload
  then restored the SSL encoder and completed projection and reconstruction forwards.

## Scientific Caveats

- AFLOW and MP use non-identical calculation protocols. Do not concatenate without source-aware calibration.
- A 12-sample smoke set is too small for reliable supervised metrics and has no metal class.
- Effective mass in `m_e` requires physical reciprocal-length units and unit conversion; current GUI only exposes a proxy.
