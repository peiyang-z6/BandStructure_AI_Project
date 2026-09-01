# 60,000 数据扩容 + V100 训练执行方案（2026-08-31）

状态：方案待批准。本文件为新增唯一项目文件，其余未修改。

## 0. 磁盘审计事实（docs vs disk 已核对）

- 现行 latest accepted：`aflow_noleak_v6_30k_seed42_metricfix`（30k，2026-08-28 验收，不可覆盖）。
- 30k immutable snapshot：`data/raw/aflow/snapshots/aflow_30000_20260825/`（h5 30,000 groups / 2.3 GB，json_cache 15 GB，SHA 冻结）。
- v4 baseline `data/raw/aflow/aflow_bands.h5`（6,443 groups）为历史 canonical，勿动。
- 本地磁盘 466 GB 可用；60k raw 预计 ≈ 4.6 GB h5 + ≈ 30 GB json_cache，无压力。
- 服务器 10.201.91.203（zhao，2×V100）：TCP 可达（SSH 显示 password auth 等待输入）；已有 `bandstructure_gpu30k` GPU 环境（TF 2.21 / CUDA 12.5.82 / cuDNN 9.3.0.75，v6 即此环境训练）。
- AFLOW keyless 探针返回 200。
- **发现阻塞**：候选上限 `src/data/batch_download.py:768` = `min(target*2, 80_000)`。60k target → 仅 80k 候选；现有 30k 运行的 metadata 已缓存 ~60k 候选，历史良率 ~50–59%（30k usable 用掉 ~60k 候选）。按此良率 +30k usable 需要 +51k~60k 新候选，80k 上限差 20k+ 缺口 → 必须先 TDD 提升上限（拟 `min(target*2, 160_000)`，保留良率回归测试）。

## 1. 数据扩容（30k → 60k）

1. 新目录 `data/raw/aflow/snapshots/aflow_60000_20260831/`：字节级复制 30k 快照全部成员（h5/json_cache/metadata/exclusion/报告/manifest），SHA 与 30k manifest 一致 → 保证 60k = 已验收 30k + 新 30k，不重下；
2. TDD：`metadata_candidate_limit` 上限 80k→160k（RED→GREEN + 现有下载器回归）；gap-bin 枯竭时沿用 30k 历史先例启用 `[0,5] eV` 单 pool 续跑；
3. 下载器对 60k 目录 `target=60000` 续跑（semantic dedupe、单写者锁、原子 report）；`target_reached=true` 与 group 计数 60,000 为硬门槛；
4. 冻结：`raw_snapshot_manifest.json`（immutable=true、SHA-256 全套）；30k 快照不动。

## 2. 张量与 split（新版本目录）

- 新 `data/processed/aflow/ood_tensors_v7_60000_seed42/`：
  - full + ood_split NPZ（(N,2,128,3)+segment IDs，PCHIP，E_F=0，segment-aware curvature）；
  - split spacegroup 80/20 seed 42 overlap=0；tensor/split SHA + audit manifest；预期 ≈59,900 有效（按 30k→29,952 跳过率外推）。

## 3. 同步到服务器（manifest 驱动）

- 上传：核心代码（当前 HEAD，含 Phase 6）+ 60k raw 增量（服务器已有 30k 镜像，按物哈希比对只补新 30k 记录）+ v7 张量/manifest；
- 双端 inventory + per-file SHA，mismatch=0 才训练；凭据仅交互 PTT，不落盘。

## 4. V100 GPU-only 训练（新实验）

- 实验 ID：`aflow_noleak_v7_60k_seed42`（v6 目录/产物绝不覆盖）；
- 配置镜像 v6 验收合同：SSL 60 epochs（actual mask 15–30%、固定 val corruption、curvature-sign only）+ supervised 60 epochs（aggregate `val_loss` canonical、best/last/accepted 冻结、`--train-only`→`--evaluation-only`、`jit_compile=False`、batch 32、`--require-gpu`）；
- 单 V100；启动前 V100 Conv1D 前后向 probe（finite gradients）+ 1-epoch 双 smoke；GPU monitor 15s 采样；60k 预计 SSL ~4–6h、监督 ~8–12h（全程后台监控 + 续跑脚本模式）。
- outer OOD 仅 evaluation-only 使用；不参与选模。

## 5. 回传与本地验收

- 结果归档（模型/checkpoint/日志/预测/指标/报告/manifest）回传，逐文件 SHA；
- 本地：指标从 `ood_test_predictions.json` 独立重算并与 `metrics_summary.json` 一致；模型 load + 真实 forward finite；selection manifest 校验；v4/v5/v6 哈希不变。
- 验收后提升 latest accepted；更新 `brain_invoker` 默认路径 / GUI t-SNE / `smoke_latest_model.py` → v7 + 测试期望（沿用 Phase 6 已批准的指针模式）。

## 6. 报告与治理

- 新报告 `artifacts/reports/aflow_noleak_v7_60k_seed42/latest_training_report_20260831.md`（含解析 baseline、mismatch strata、MC raw coverage 等 v6 口径全套）；
- README / dev_context（写入“数据规模扩展至 60,000 条并在 V100 GPU-only 完成 v7 训练与回传验收”）/ 宪法版本号与 §3 实验行 / 日期 agent log / 日程同步；
- 全量回归 + compileall + git diff 行尾检查；commit + tag。

## 非目标

- 不覆盖 v4/v5/v6；不动 `aflow_bands.h5`（v4 baseline）；不引入多 GPU 分布式；不换超参搜索；不做 Phase C。

## 风险与预算

- 下载：+30k usable 需 ~+55k 候选（AFLOW keyless，无配额风险），预计数小时；
- 训练：单 V100 < 24h；若 candidate 良率仍不足 → 停止并如实报告 yield，绝不伪造 `target_reached`（宪法 §4 fail-closed）。
- 60k 全链磁盘 ≈ +45 GB（本地 466 GB 可用；服务器侧训练前查 df）。

## 待批决策

D1 实验命名；D2 快照策略（复制 30k + 增量 30k，含候选上限 TDD 修正）；D3 训练预算。
