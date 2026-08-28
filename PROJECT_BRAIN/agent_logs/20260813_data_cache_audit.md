# 2026-08-13 Data Cache Audit: Stage 1 Partially Pre-executed

## Trigger

模型服务商配置修复后恢复工作，在启动 Stage 1 下载前做环境核查，发现
data_cache 中已存在大量昨日（2026-08-13 15:41–18:12）的下载产物，
dev_context.md（2026-08-12）中的"12 条 AFLOW 冒烟、MP 未运行"声明已过期。

## Audit Evidence (read-only, no code changes)

### AFLOW cache（远早于方案预期）

- `aflow_bands.h5`：183 MB，**2000 个完整能带 group**（energies/k_distances/kpoints/metadata 齐全）。
- `aflow_metadata.json`：**5250 条元数据**（候选池），覆盖 **163 个空间群**。
- 元数据带隙分布（候选池 5250 条）：min=0.000，中位=3.072 eV，max=7.986 eV；
  **metal（is_metal=True）= 750 条**；`gap_type_source` 分布：
  metal 713、half-metal 37、insulator-indirect 2898、insulator-direct 1316，
  另有 spin-polarized 变体 indirect 215 / direct 71（AFLOW 提供方标签，合宪）。
- HDF5 实缓存 2000 条的带隙分布：min=0.000，中位=4.019 eV，max=7.900 eV；
  其中 metal（gap<0.05）**仅 146 条**（候选池 750 条 metal 中大部分尚未下载）。
- 数据质量字段完整：每条含 `source_sha256`、`download_url`、`kpath_labels`、
  `efermi`、`spacegroup_number`、`is_spin_polarized`，满足宪法 §3 可追溯要求。
- `aflow_excluded.json`：上游空 bands_data 记录已被隔离（`unexpected bands_data shape ()`），
  符合 stage0 修复的空文件分类逻辑。

### MP cache（key 已配置，但未成功下载）

- `configs/api_keys.env` 存在；`stage1_keycheck/`、`stage1_smoke_mp/`、
  `mp_batch_probe/`、`mp_batch_smoke/`、`mp_concurrency_smoke/`、`mp_reuse_smoke/`、
  `mp_single_client_smoke/`、`mp_throughput_smoke/` 多个冒烟目录已建立。
- `mp_throughput_smoke/mp_bands.h5` 仅 61 KB（接近空）；`stage1_smoke_mp/mp_download_report.json`
  显示 total_cached=0、success=0——**MP 源已完成 key 校验与客户端冒烟，但正式下载未跑通/未执行**。

## Consequences for the Plan (20260813_development_plan.md)

1. **P0-数据规模瓶颈对 AFLOW 已大幅缓解**：2000 条实缓存 / 163 空间群，
   已超过方案第一阶段目标（≥800 条 / ≥60 空间群）。
2. **metal 类缺口仍在**：实缓存 metal 仅 146 条，低于方案目标 ≥150；
   但候选池还有 ~600 条 metal 可续传补齐——`batch_download.py` 断点续传按总量计数，
   直接用 `--target` 补 metal 区间即可，无需改代码。
3. **带隙分层不均**：实缓存中位 4.019 eV，[0.2,0.5] eV 半导体应用区候选为 0。
   若应用报告侧重 1–2 eV 半导体，需要在续传时刻意加低带隙分层权重。
4. **MP 源是 Stage 1 唯一未闭环项**：key 已配置，需诊断为何 success=0
   （网络/字段映射/协议差异），这是 domain shift 对照实验的前提。
5. dev_context.md 的 Stage 0 声明需在下次宪法允许的结构更新中修订
   （本次仅追加日志，不改既有文档）。

## Tensor Build Verification (2026-08-13, on the 2000-record cache)

`build_ood_tensors.py` 在 2000 条实缓存上一次通过，宪法 §5 契约在大批量下成立：

- `X shape: (2000, 2, 128, 3)`，`y shape: (2000,)`；
- **156 个空间群**（train 123 / test 33），train/test = 1600/400，**空间群零交集**；
- 三分类标签（提供方标签，合宪 §4）：train {metal 117, indirect 516, direct 967}，
  test {metal 29, indirect 128, direct 243} —— 类别不均衡确认存在（focal loss 依据）；
- 带隙分布：train min=0.000 med=4.010 max=7.900 eV；test min=0.000 med=4.077 max=7.844；
- metal（gap<0.05）：train 117 / test 29，低于方案目标 150，需续传补齐；
- 张量通道范围（前 500 条抽样）：VBM_E/CBM_E 归一化前 eV 量级合理，
  曲率通道原始数值量级大（std ~2.8e4），验证了 `standardize_features` 的必要性。

## Next Action (unchanged direction)

- ✅ 张量构建与 OOD 划分已在 2000 条上验证（本日志）。
- 并行诊断 MP 下载链：live log 显示 MP 新 API 走 S3 `delta_log` 端点
  `Connection refused (os error 111)` —— 网络层拒连（代理/防火墙/S3 端点不可达），
  需确认 WSL 的 HTTP(S)_PROXY 是否覆盖 S3 域名；MP key 本身已通过 keycheck。
- 视三分类均衡决定是否续传 metal（候选池尚有 ~600 条 metal 未下载）。
