# 2026-08-24 — Project Cleanup and Root Migration Execution

## Scope

按用户明确授权，将运行项目迁移到：

`C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`

参考资料保留在外层 `资料/`，未进入运行根目录。主方向 E(k)→6D→MBM→gap/type→tkinter 未改变。

## Pre-migration audit

- 旧根目录：3078 files，约 2.08 GB。
- 旧 `data_cache`：2508 files，约 1.80 GB。
- 本地 AFLOW HDF5 实际只有 2400 groups，文档声称的 6443 全量位于服务器。
- 本地 JSON cache=2400；服务器增量 JSON=4043；文件名零重叠。
- 服务器正式 SSL 已完成；监督训练在 epoch 31 early-stop 后完成，训练进程回验为 0。

## Migration and data consolidation

1. `.git`、源码、测试、配置、依赖和 `PROJECT_BRAIN` 迁入新根。
2. 旧 artifacts 与临时文件先移入外层 quarantine，正式验收前未直接删除。
3. 服务器导出包使用 zstd，传输包 SHA-256：
   `ed7ca12d23777e206d1703a65ee264865d05df859c2350115eed1f6f4c5bf00f`。
4. 服务器 6443 HDF5、metadata、4043 JSON 与本地 2400 JSON 合并。
5. HDF5 material IDs 验证：本地 2400 全部属于正式 6443。
6. JSON 合并：2400+4043=6443，重叠 0。
7. 早期 AFLOW smoke 审计发现 20 个正式 6443 之外的唯一 ID，改存 `data/raw/aflow/supplemental/`，不混入 formal cache。
8. 发现 2 个 legacy HDF5 与 canonical 同 ID group 内容不同；完整变体存入 `data/raw/aflow/provenance/h5_variants/`。
9. MP 多批 smoke 合并为 12 unique HDF5 groups、12 JSON 和 3000 metadata；原报告进入 provenance。
10. 正式 processed snapshot 统一为 `data/processed/aflow/ood_tensors/`。

## Artifact consolidation

统一实验名：`aflow_noleak_v4_seed42`。

保留：

- restored-best SSL encoder；
- epoch-52 final SSL model；
- SSL best/last checkpoint；
- supervised best checkpoint；
- finetuned weights/config；
- metrics、predictions、figures、完整 SSL/finetune/TensorBoard logs；
- latest Chinese training report 与 artifact provenance。

未迁入：legacy target-conditioned models/reports、periodic SSL 10/20/30/40/50 checkpoints、smoke artifacts。

最终 artifact provenance 审计 33 个非自引用文件；除 `metrics_summary.json` 和 `finetuned_config.json` 因路径重定位有意改写外，所有有服务器原始哈希的非重写文件 SHA-256 mismatch=0。

## Runtime path update

- 旧 `data_cache/` → `data/raw/` + `data/processed/`。
- 旧根级 `models/checkpoints/reports/logs` → `artifacts/<type>/<experiment_id>/`。
- `src` 模块边界不变；没有新建平行 data/model/trainer 框架。
- `scripts` 保持稳定扁平入口，职责另写 `scripts/README.md`。
- `pipeline_layout('aflow')` 的 canonical run 指向 `aflow_noleak_v4_seed42`。
- `PhysicsBrainInvoker` 默认路径指向迁移后的真实模型文件。
- 新增路径回归测试，禁止 runtime Python 再引用旧根路径。

## Formal training result

- SSL best epoch=37；early-stop epoch=52；best masked MAE=0.13883。
- Supervised best inner-val epoch=11；early-stop epoch=31。
- Outer gap MAE=0.049747 eV；RMSE=0.159700 eV；R²=0.995591。
- Accuracy=0.865683；Macro F1=0.858339。
- Recall：metal 0.983471、direct 0.765854、indirect 0.900925。

完整报告：

`artifacts/reports/aflow_noleak_v4_seed42/latest_training_report_20260824.md`

## Verification evidence

- 路径 RED tests 均先失败并在更新后通过。
- Final full suite：`39 passed in 41.57s`。
- 最新模型真实 load-weights + outer OOD inference smoke：通过。
- 输入 `(2,128,6)`；gap/type 全有限；type probability sum=1。
- Formal HDF5 与 tensor manifest 原始 SHA-256 字节级一致。
- `资料/` 不在运行根目录。

## Corrections to the original plan

原计划假设所有早期 AFLOW smoke 都只是 6443 主缓存的重复子集；真实审计发现 20 个唯一 ID 和 2 个同 ID 内容变体。执行中没有删除它们，而是分别进入 supplemental 与 provenance。该修正以磁盘事实为准，并保持 formal 6443 snapshot 不变。

## Final cleanup result

- 删除 82 个隔离/临时条目，共 458 files、1,210,262,345 bytes（1.127 GiB）。
- 外层 quarantine：已删除并回读确认不存在。
- 931 MB zstd 传输归档：已删除。
- 本轮及前序 Phase A/B 系统 Temp 项目脚本：已删除；含 SSH/Paramiko 标记的剩余脚本为 0。
- `.pytest_cache`、`__pycache__` 与 `.pyc`：已删除；最终 `.pyc` 搜索结果为 0。
- 删除后回验：AFLOW canonical/supplemental/MP HDF5 groups 仍为 6443/20/12，正式 HDF5 SHA-256 未改变，监督模型仍存在。
- 外层目录最终只包含 `BandStructure_AI_Project/` 与 `资料/`。

最终 inventory：`PROJECT_BRAIN/post_migration_inventory_20260824.json`。
