# 20260903_P0_scientific_reframing_plan

## 背景

用户指令（P0，本阶段）：修正科研问题与基准——line-mode gap MAE 退出主结果位；标签拆成三个明确任务；3 seeds + group bootstrap + 错误分层；固定七类拆分；清理 README 历史叠加段；分支命名与 v7 对齐。

## 磁盘审计事实（2026-09-03 实测）

1. **现有标签**：`y_type`（3 类 metal/direct/indirect）纯来自 provider（`gap_type_source`/`is_metal`/`is_direct`），`is_direct` 仅 19,703/60,000 非空（金属为空）；tensor builder 内部已有 `vbm_idx/cbm_idx/metal_feature_inferred` 但**未存盘**（NPZ 无此字段）。
2. **七类拆分所需字段现状**：
   - space-group：`spacegroup_number` 60,000/60,000 ✅
   - composition：`formula_pretty` ✅ 可派生
   - source/protocol：`aurl` 60,000/60,000 ✅（ICSD_WEB/LIB1/LIB2/LIB3、LDA/LDAU2 可解析）
   - random：无需字段 ✅
   - prototype：metadata 0 覆盖 ❌ → 需 AFLUX 元数据回填（同查询重翻页，只取 metadata，约 207 页）
   - leave-element：`species` 0 覆盖 ❌ → 同上回填
   - temporal：`year` 0 覆盖 ❌ → 同上回填
   - 本地 json_cache 是**按材料 ID 的 band 数据**缓存，不含 catalog 行（已抽查 25,478 文件结构，非 catalog 响应）→ 不能免下载。
3. **line_mode_topology 可派生性**：X 的 channel 2 = extremum k-distance，金属/直接/间接可由线模式路径自身判定（crossing → metal；|argmin kdist_VBM − argmin kdist_CBM| ≤ 1 → direct；否则 indirect）。无需重下数据，仅需在张量构建/标签派生阶段落盘。
4. **现有 mismatch 层**：v7 已有 feature_label_mismatch strata（1,723/11,987），即 line_global_disagreement 的原型，但无独立 head、无 3 seeds、无 bootstrap。
5. **README**：5 个叠堆"当前状态"段（2026-09-03/08-31/08-28/08-26/08-24）。
6. **分支**：本地 `v6-metricfix-20260827`（HEAD=1d5c9fe）+ `master`；remote `origin/master` 为默认分支。改名涉及远程操作。
7. **训练预算**：v7 全链（SSL 50ep + sup 51ep）单 seed 约 4h V100。3 seeds 全链 ≈ 12h；若"每拆分重训"×7×3 不现实（≈84h），见决策 D3。

## P0 方案（6 个子任务）

### P0A 标签三任务重构（核心科研修正）

- **T1 line_mode_topology**（3 类：metal/direct/indirect）：由线模式路径自身派生（crossing/k-距离重合），与 provider 标签无关。落盘到 NPZ/manifest。
- **T2 provider_global_electronic_type**（3 类）：即现有 provider 标签（Egap_type），保持 target/audit 角色（宪法 §5 已限定）。
- **T3 line_global_disagreement**（2 类）：T1 ≠ T2 时为 1。派生标签为 ground truth；模型加独立二分类 head 专门预测冲突（同时报告"派生式判定" argmax 不一致率作为解析对照）。
- 主结果排序：T3（冲突检测）与 T1/T2 分层精度为主表；line-mode gap MAE 降为次级附注（保留解析基线 0/0 的既有说明）。
- 模型结构：现有 gap head + type head 不动；type head 训练目标改为 T2，新增 topology head（T1）与 disagreement head（T3）。三个分类 head 均为独立输出。改动文件：`src/models/`（模型类）、`scripts/finetune_supervised.py`。

### P0B 元数据回填（prototype/species/year）

- 新脚本 `scripts/backfill_aflow_metadata_fields.py`：与下载器同 query fingerprint（Egap(0,5)+natoms(1,50)+page_size 500）重翻 catalog，只取 `prototype, species, species_pp, year, experiment` 字段，按 material_id 合并进 `aflow_metadata.json`（非空字段不回退已有记录，宪法 §4）。~207 页，单出口 ~35–60 分钟，0 band 数据下载。
- 覆盖审计：回填后逐字段输出覆盖率，不达标的字段在 manifest 标注。

### P0C 七类拆分基准套件

- 新模块 `src/data/benchmark_splits.py`（或在 tensor builder 内扩展）：固定七类拆分定义，全部 seed 42、0 group overlap、manifest 落盘（group key、train/test 规模、overlap、分层统计）：
  1. random：sample-level stratified shuffle；
  2. space-group：现有定义（保持不变，canonical 训练拆分）；
  3. composition：按公式组成集合分组；
  4. prototype：AFLOW prototype 标签分组（P0B 回填后）；
  5. leave-element：leave-one-element-out（对 species 集合按元素分组；至少覆盖出现频次 top-k 元素）；
  6. source/protocol：aurl 解析（ICSD_WEB / LIB1 / LIB2 / LIB3 × 泛函）分组；
  7. temporal：year 分桶（老文献 vs 新文献，P0B 回填后）。
- 每拆分的定义与统计写入 `data/processed/aflow/benchmark_splits/*.json` + 拆分审计报告。

### P0D 3 seeds + group bootstrap + 错误分层

- 3 seeds：{42, 2024, 7}（或按用户偏好）——canonical space-group 拆分下 SSL+sup 全链 ×3，单卡 V100 顺跑。每个 seed 产出独立 experiment dir（`aflow_noleak_v7_60k_seed{42,2024,7}` 或 seed 后缀变体命名，待定）。
- group bootstrap：对七类拆分的 test 集合做 spacegroup-level bootstrap（B=1000），给出 95% CI（percentile），只作用于评估，不改训练。
- 错误分层：每拆分 × 每任务输出分层表——T2 层（metal/direct/indirect）、T1 层、T3 冲突/一致层、provider-feature mismatch 层、year/prototype/source 层。
- 汇总报告 `artifacts/reports/benchmark_suite/*.md`：三任务 × 七拆分 × 3 seeds 均值±std 主表，line-mode gap MAE 作附注。

### P0E README 清理 + 分支命名对齐

- README：删除/折叠 5 个历史"当前状态"段为单一"当前状态"+ 一个"历史里程碑"折叠小节；同步 P0 三任务框架。
- 分支：`v6-metricfix-20260827` → 新名（决策 D4）；本地 rename + push + remote 默认分支处理 + 服务器侧说明。

### 治理

- 每子任务 TDD（新代码先 RED 测试）+ 全量回归；结构性修改同步 README/dev_context/宪法/dated log；宪法升 4.13（三任务基准合同、七拆分定义、3-seed 要求）。

## 明确不做（Non-goals）

- 不改 6D 张量合同与费米锚定；不引入新数据源；不删 v4/v5/v6/v7 既有 artifact；不动 outer OOD test 不得用于选模的宪法条款；不做每拆分 × 每 seed 全量重训矩阵（P0 只训 canonical 拆分 3 seeds，其余拆分做冻结模型评估；按拆分重训留 P1 待批）。

## 待拍板决策（4 项）

- D1 三任务 head 结构：3 个独立 head（推荐）vs 复用现有 type head 输出做 T1/T3 派生（零结构改动、更快）。
- D2 3 seeds 命名与数值：seed ∈ {42, 2024, 7} 及 experiment 命名（推荐 `aflow_noleak_v7_60k_seedXX` 三目录，canonical 拆分一致）。
- D3 七拆分评估方式：P0 冻结模型跨拆分评估（推荐，~1h GPU）vs 每拆分重训（×7×3，~84h，不推荐）。
- D4 分支新名：`v7-60k-20260903`（推荐）vs `main`（需同时迁移 origin 默认分支与服务器）。
