# 2026-08-24 — Phase B MBM Representation and Training Upgrade

## Scope

在 Phase A noleak_v4 张量合同上升级现有 `SSLEncoder` / `MBMTrainer` / `train_ssl.py`，不新增并列训练框架。

## Implemented

1. `SSLEncoder.mask_token`：trainable feature token；零填充 mask 已移除。
2. Span masking：每 batch 随机 5–15 k points；start 随机，掩码不跨 `segment_ids`。
3. `train_ssl.py` 读取 `segment_ids_train` 并与 group-disjoint inner split 对齐；旧 NPZ 自动全零 fallback。
4. Curvature/sign consistency loss 的二阶差分只在同 segment 三点窗口计算；跨段点不进入一致性均值。
5. 单一 warmup+cosine schedule，final step 精确到 min LR；无第二个 LR controller。
6. global gradient clipping（默认 1.0）。
7. early stopping（默认 patience 15 / min_delta 1e-5）。
8. `ckpt-last` + epoch/best/patience state；`--resume` 已通过 epoch 2→3 续训验证。
9. masked-position metrics：masked MSE、masked MAE、实际 mask fraction；不再以全序列 decoder 物理分数评价 MBM。
10. legacy metal-anchor gate 默认关闭；仅 `--enable-metal-anchor-gate` 显式兼容旧快照。

## Tests and Smoke

- Local full suite：31 passed。
- Server full suite：31 passed。
- RTX 4060 span/token/schedule/segment-loss smoke：passed。
- V100 span/token smoke：passed。
- Schedule boundary unit test：step0=0, warmup end=base, final=min。
- Segment boundary unit tests：mask 与 curvature/loss 均不跨段。
- Early stopping constant-val test：patience=2 在 epoch3 停止。
- Resume smoke：restored `ckpt-last (epoch=2)`，从 epoch3 继续并输出 `final_epoch3.keras`。

## Formal seed42 MBM

Dataset: `data_cache/aflow_ood_tensors_6443_noleak_v4/`

- d_model=128, heads=4, layers=4, dff=256；mask ratio target=0.25；actual span mask fraction≈0.17–0.18。
- warmup=5 epochs, cosine floor=1e-6, clip norm=1.0, patience=15。
- early stopped at epoch 52。
- final saved: `models/noleak_v4_b1_seed42/ssl_mbm_final_epoch52.keras`。
- val masked MAE in later epochs≈0.136–0.145；masked MSE≈0.17–0.23。
- supervised seed42 fine-tuning is running on CPU; final metrics pending.

## Notes

- Actual span mask fraction is below nominal 0.25 because spans are truncated at short segments and overlap; it remains within constitutional 15–30% range. Formal ablation should compare oversampling starts vs current behavior.
- Formal 3-seed result remains pending; seed42 is the first unbiased noleak baseline, not yet the final reported result.
