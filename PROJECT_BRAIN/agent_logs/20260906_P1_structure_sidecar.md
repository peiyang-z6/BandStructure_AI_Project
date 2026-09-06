# 2026-09-06 P1 结构 sidecar 补全 — 实施日志

## 目标（宪法 5.0 §8 P1）

为 60k AFLOW 能带补全结构，不改 immutable HDF5，新增只读配对 sidecar：

- lattice 3×3、species、fractional_coordinates、magnetic/spin、DFT
  functional/U/pseudopotential、reciprocal lattice、3D fractional k-points、
  k-path convention、source、structure SHA-256。

## 数据源探测（实测 2026-09-06）

AFLOW REST 端点逐字段验证：

| 字段 | 端点 | 结果 |
|---|---|---|
| 晶格参数 | `?geometry` | 200，`[a,b,c,α,β,γ]` |
| 分数坐标 | `?positions_fractional` | 200 |
| 笛卡尔坐标 | `?positions_cartesian` | 200 |
| 元素 | `?species` | 200 |
| 功能 | `?dft_type` | 200，`["PAW_PBE"]` |
| 自旋 | `?spin_cell`/`?spin_atom`/`?spinD` | 200 |
| 势 | `?species_pp`/`?species_pp_version` | 200 |
| k 路径 | `?kpoints_bands_path` | 200 |
| k 网格 | `?kpoints` | 200，`[relax_mesh,static_mesh,segments,nkpts]` |
| 倒格子参数 | `?reciprocal_geometry` | 200 |
| VASP 版本 | `?code` | 200 |
| 3D 逐点 k | `KPOINTS.bands` | 200（line-mode 段端点） |
| `?lattice` | — | **404**（用 `?geometry` 构造） |
| `?kpoints_bands` | — | **404** |
| `?LDAU`/`?LDAU2`（U 值） | — | **404** |

AFLUX 批量分页返回全部结构字段（原生类型），`kpoints()` 只给 mesh+段标签+nkpts，
**不含逐点 3D 坐标**——那在 KPOINTS.bands 文件（每 AURL ~1KB）。

## 实现（TDD）

- `src/data/structure_sidecar.py`：lattice 构造（`geometry_to_lattice_vectors`，
  标准晶体学约定）、倒格子、结构 SHA-256、sidecar 记录映射、schema 序列化。
- `scripts/build_structure_sidecar.py`：AFLUX 批量分页（复用 P0B 同款指纹）
  + KPOINTS.bands 3D 坐标补充 + 覆盖审计报告。
- 测试：`test_structure_sidecar.py`（7）、`test_kpoints_bands_parse.py`（3）、
  `test_structure_sidecar_coverage.py`（3）= 13 passed。

## 关键 bug 修复

1. `compute_structure_sha256` 对 numpy 数组 `or` 真值歧义 → `is not None` 判断。
2. monoclinic 测试断言 α 角写错（80.32 写成 90）→ 修正。
3. KPOINTS.bands nkpts 在第 2 行（"20 ! 20 grids"）非 header → 改行号。
4. 段端点步长：空白行过滤后端点成对连续 → `i += 2`（原 `+=3` 假设空白行保留）。

## 运行

- bulk：103,084 行 / 207 页 / 486.8s（workers 8）。
- HDF5 groups：60,000（sidecar 只保留其中 ID）。
- KPOINTS.bands 补充：58,250 请求（workers 12），完成。

## 覆盖缺口（重要发现）

首轮覆盖审计：sidecar 58,250 条 vs HDF5 60,000 groups → **1,750 个 ID 缺失**。

根因：这 1,750 个 ID 的 metadata 里 `Egap`/`natoms` 均为 `None`，即它们不在
AFLUX 查询指纹 `Egap(0*,*5),natoms(1*,*50)` 覆盖范围内——是 60k 下载时从
另一指纹合并进来的样本（见 memory：main 55,476 + merged 4,524）。它们有
有效 aurl，per-AURL REST 端点实测全部 200 可达。

处理：`scripts/fill_sidecar_gaps.py` 对这 1,750 个 ID 逐个拉 per-AURL 端点
（10 字段 + KPOINTS.bands），合并回 sidecar（非空优先，宪法 §4）。完成。

## 最终覆盖（100% ID 对齐，39 条源端缺陷）

- sidecar records = HDF5 groups = **60,000**，ID 集合完全相等（缺失 0）。
- 字段覆盖：lattice/species/fractional_coordinates/dft_functional/spin_cell/
  spin_atom/pseudopotential/reciprocal_lattice/kpath_segments/structure_sha256
  各 **59,961/60,000**；kpoints_3d **59,977/60,000**。
- 39 个记录缺全部结构字段：全部为 `ICSD_WEB/HEX` 目录，AFLOW REST 端点
  （?geometry/?positions_fractional/?species/?dft_type/?files）**全部 HTTP 500**。
  这是 AFLOW 服务器端数据缺陷（这些 HEX 条目损坏/不完整），非网络、非代码
  问题。按宪法 §4 如实标记 missing_fields，不伪造、不静默。
