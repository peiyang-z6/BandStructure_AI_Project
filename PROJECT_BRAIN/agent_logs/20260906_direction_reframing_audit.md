# 2026-09-06 方向调整审计（questions and directions.md 落地前核查）

## 用户指令

按 `Desktop/questions and directions.md` 调整项目开发方向，已批准修改项目宪法。

## 磁盘审计结果

### 文档声明 vs 磁盘现状（两处文档已过时）

1. 文档称"完整回归 164 passed"——**已过时**：P0 阶段后为 **199 passed**（新增 29 个 P0 测试）。
2. 文档称"分支仍叫 `v6-metricfix-20260827`"——**已过时**：P0E 已改名 `v7-60k-20260903`。

其余事实性声明核实如下。

### 短板 4（HDF5 缺结构字段）——属实

实测 `aflow_60000_20260831/aflow_bands.h5` 首个 group：
- keys: `energies`, `k_distances`, `kpoints`, `metadata`
- **无 lattice、species、fractional_coordinates、reciprocal lattice、3D fractional k-points**

`aflow_metadata.json`（60,000 条）每记录 keys:
`aflowlib_date, aurl, band_file, band_gap, download_url, formula_pretty, gap_type_source, is_direct, is_metal, material_id, num_sites, pearson_symbol, prototype, source, source_id, spacegroup_number, species, species_pp`

- **有** species（元素符号列表）、species_pp（势函数后缀）、prototype、aurl、aflowlib_date；
- **无** lattice 3×3、fractional_coordinates、magnetic/SOC、functional/U、reciprocal lattice、3D k-points、k-path convention、结构 SHA。

结论：文档"Phase C 首先是数据合同问题，不是模型问题"成立。宪法 §8 已把 lattice/species/fractional coordinates 列为前置条件，正确。

### P1 结构 sidecar 补全路径——实测可行

对 LIB3_WEB 与 ICSD_WEB 各一例实测 AFLOW REST 端点（`aflowlib.duke.edu/AFLOWDATA/<aurl>/`）：

| 端点 | 状态 | 返回 |
|---|---|---|
| `?geometry` | 200 | `[a,b,c,α,β,γ]` 晶格参数 |
| `?positions_fractional` | 200 | 分数坐标列表 |
| `?positions_cartesian` | 200 | 笛卡尔坐标列表 |
| `?species` | 200 | 元素列表 |
| `?dft_type` | 200 | `["PAW_PBE"]` |
| `?spin_cell` | 200 | 自旋标量 |
| `?enthalpy_formation_atom` | 200 | 形成焓 |
| `?files` | 200 | 文件清单（含 `CONTCAR.relax`、`.cif`） |
| `?lattice` | 404 | 不存在（用 `?geometry` 替代） |

结论：P1 的结构 sidecar 全部字段均可通过只读 HTTP 端点按 AUID 逐个获取，无需下载大体积文件（CONTCAR/CIF 为小文本），路径现实可行。

## 方向评估结论

文档的三项差异化定位判断成立，与磁盘现状一致：

1. line-mode gap 是 identity baseline=0 的解析函数（宪法 §6 已固定），不能作为主贡献——P0 已把主结果改为三任务。
2. provider global label 与 line-mode 可见特征的域错配是真实科学问题（P0 的 `line_global_disagreement` 任务已显式建模，冲突率 12.4%）。
3. MC coverage 0.7974 不足以驱动主动学习（P0 未解决，留 P4）。
4. 结构字段缺失是 P1 的数据合同问题（本文实测确认）。

推荐执行顺序（文档 P0 已完成；P1–P5 见新宪法条款）：先补结构 sidecar（P1），再做跨模态检索（P2），然后 multi-band decoder（P3）、校准不确定性（P4）、外部验证集（P5）。
