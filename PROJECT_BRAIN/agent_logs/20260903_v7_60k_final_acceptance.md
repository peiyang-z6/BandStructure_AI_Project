# 20260903_v7_60k_final_acceptance

## 概要

`aflow_noleak_v7_60k_seed42`（60,000 条 AFLOW）完成下载、合并、张量构建、V100 训练与最终验收，提升为 **latest accepted**。宪法升 4.11→4.12。

## 数据链

- 下载：主通道 55,476 + 代理通道（10808 出口）40,259（其中唯一 5,032），seed 42 抽样合并 4,524 条 → **恰好 60,000**（用户决策"只补到 60k 为止"，其余 508 条代理唯一记录丢弃并记录）。
- 合并脚本 `scripts/merge_proxy_to_60000.py`（commit `76fdeed`）+ metadata 对齐脚本 `scripts/reconcile_metadata_60000.py`：metadata 60,000 ID 与 HDF5 完全一致。
- 张量：`data/processed/aflow/ood_tensors_v7_60000_seed42`，59,899 样本（101 条无 occupied/empty 边缘包络跳过，理由已记录）；split spacegroup 80/20 seed 42，train/test=47,912/11,987，155/55 组，**overlap=0**。
- 下载期修复：metadata 候选上限 80k→160k（`729b016`）；AFLOW bandsdata 尾随逗号容错（`6ee751c`）。

## 训练（V100 GPU-only，镜像 v6 合同）

- SSL：合同 60 epochs → 实际 50（early stopping），mask 0.25 / sign 2.0 / consistency 0.0。
- 监督 train-only：合同 60 → 实际 51，**best epoch 31**（aggregate inner `val_loss=0.744769`），best/last/accepted 冻结后 evaluation-only 才访问 outer。
- 服务器: `/home/zhao/BandStructure_AI_60k_20260903`（同步归档 3.18GB/161 文件，SHA 双向一致）。

## 训练中两次 OOM 与修复（TDD）

11,987 outer-test 全批量前向超出 16GB V100：

1. `reconstruct_raw_tensors` 全批量 `encoder.reconstruct` → `(11987,4,128,128)` 注意力 softmax ~3.1GB/层 OOM。修复 `reconstruct_encoder_chunks`（1024/块，commit `4367f87`）。
2. `save_extremum_probability_heatmaps` 全批量 `extremum_probabilities` 同样 OOM。修复 `extremum_probabilities_chunks`（commit `ed783a8`）。

回归测试 `tests/test_supervised_eval_chunked_reconstruct.py`（8 测试），全量 164 passed。修复后 evaluation-only 重跑成功。

## 最终指标（11,987 outer；v6 参考 6,019 outer）

| 指标 | v7 | v6 |
|---|---|---|
| line-mode MAE / RMSE (eV) | 1.54e-05 / 5.54e-04 | 4.07e-04 / 7.11e-03 |
| model vs global DFT MAE | 0.3767 | 0.6402 |
| type acc / Macro F1 | 0.9251 / 0.8788 | 0.9588 / 0.9493 |
| spacegroup-macro | 0.9296 | 0.9658 |
| mismatch accuracy (n) | 0.7441 (1,723) | 0.8305 (1,257) |
| MC raw 95% coverage | 0.7974 | 0.8040 |

回归显著提升；分类略降（60k 尾部样本更难，mismatch 层扩大），如实报告。

## 验收与治理

- 产物回传归档 SHA-256 `9c46d1122be333ad903cff3eaab6fb802bc7f4c47ad3761051993b701d6cd45c`（65.5MB），本地校验一致。
- 本地模型加载 + 真实 forward smoke 通过；selection manifest（best epoch 31、`outer_accessed=false`）校验通过；v4/v5/v6 immutable rehash 与冻结基线一致。
- latest-model 指针 v6→v7：`brain_invoker` 默认路径、GUI t-SNE、`smoke_latest_model.py`；`tests/test_project_layout.py` 期望同步（10 passed）。
- 文档同步：README、dev_context、CONSTITUTION 4.12、本日志；验收 manifest `PROJECT_BRAIN/transfer_manifests/local_final_acceptance_v7_20260903.json`；最终报告 `artifacts/reports/aflow_noleak_v7_60k_seed42/v7_final_training_report.md`。
