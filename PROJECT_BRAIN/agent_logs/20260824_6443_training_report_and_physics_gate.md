# 2026-08-24 — 6443-Record Training, Data-Quality Fixes, and Deployment Gap Gate

## User Request

暂停 30000 条下载，先修复已知问题；用当前全部数据在 V100 服务器完成一轮测试；生成今日训练报告。

## Download State

- staged AFLOW 下载已停止，服务器无 `batch_download` / `download_staged` 进程。
- 停止时 HDF5 缓存：**6443 条**。
- 后续恢复下载必须从 6443 条续传，不重新开始。

## Fixes

### 1. Metal topology prior

- 旧：`metal_rule = predicted_gap < 0.15 eV`，会把部分小带隙 indirect 误判为 metal。
- 新：`metal_rule = (predicted_gap < 0.2 eV) AND direct_topology`。
- 依据：费米锚定 metal 既有近零 tensor gap，又有 VBM/CBM k 拓扑一致性；双条件能抑制 indirect 假阳性。

### 2. Negative tensor-gap filter

- 非 metal 若 `min(CBM_E)-max(VBM_E)<-0.05 eV`，视为 VBM/CBM 选择异常并隔离。
- 6443 条缓存重建后保留 **6399 条**，过滤 44 条异常；有效集中负 tensor gap=0。

### 3. Fermi-anchor deployment gap gate

- 发现 OOD 原始回归最大误差 16.49 eV；前 10 个最大误差全部为 tensor gap=0 的费米锚定 metal。
- 新门控：若输入 tensor gap 绝对值≤1e-4 eV，则部署带隙确定性输出 0 eV；门控只使用输入特征，不读取测试标签。
- 新增 `test_extremum_gap_head_enforces_fermi_anchored_zero_gap`；定向测试通过，全套 **19 passed**。

### 4. Physics-score scope

- 原报告把 MBM 解码器全序列重建物理分数（0.0452）混入监督模型总分。
- MBM 只在掩码位置优化，不能用全序列重建分数评价监督 gap head。
- 修正为：监督物理分数（gap 一致性+非负性）与 SSL 重建物理分数单独报告。

## Tensor Snapshot

- 输出：`data_cache/aflow_ood_tensors_6443/`
- Shape：`(6399,2,128,3)`；183 空间群；train/test=5117/1282；空间群交集 0。
- 类别：train metal/direct/indirect=479/1636/3002；test=118/413/751。

## Training

### SSL

- `d_model=128`, 4 heads, 4 layers, 60 epochs, GPU。
- 最佳 inner-val loss：**0.12059（epoch 44）**。
- 输出：`models/server_6443/ssl_mbm_pretrained.keras`。

### Fine-tuning

- 60 epochs；best inner-val gap MAE=**0.03110 eV**；CPU（V100/CUDA13 cuDNN Conv1D ABI workaround）。
- 输出：`models/server_6443/finetuned_6443.weights.h5`。

## OOD Results

### Raw network

- gap MAE=0.1868 eV, RMSE=0.9965 eV, R²=0.7895, parity corr=0.9025。
- type accuracy=0.6794, Macro F1=0.6578。
- recall metal/direct/indirect=0.9746/0.6513/0.6485。
- AUC metal/direct/indirect=0.9723/0.7834/0.8371。
- 误差中位数=0.0029 eV；P95=0.645 eV；>5 eV 仅 14 条（1.09%），但主导 RMSE/R²。

### Physics-gated deployment

- gated samples=118（与 OOD metal 数相同）。
- gap MAE=**0.0645 eV**, RMSE=**0.2358 eV**, R²=**0.9882**。
- P95=0.2665 eV；最大误差=2.651 eV；负预测率=0。
- 分类指标不变。

### Physics and uncertainty

- corrected supervised physics score=0.9431。
- SSL full-sequence reconstruction physics score=0.0452（诊断项，不作为监督模型结论）。
- MC Dropout 95% CI coverage=97.97%，mean/median uncertainty=0.2306/0.0613 eV。

## Verification

- Python compile: passed。
- Targeted Fermi-anchor gate regression: passed。
- Full suite: **19 passed**。
- Server artifacts pulled to local; tarball size 90.0 MB, transfer size verified。
- Report metrics read back from local JSON and server JSON before report generation。

## Artifacts

- Daily report: `reports/每日训练报告_20260824_6443条修复验证.md`
- Raw metrics: `reports/server_ft_6443/metrics_summary.json`
- Deployment metrics: `reports/server_ft_6443/physics_constrained_deployment_metrics.json`
- Scope correction: `reports/server_ft_6443/physics_metric_scope_correction.json`
- Tensor manifest: `data_cache/aflow_ood_tensors_6443/ood_split_manifest.json`
- Models: `models/server_6443/`
- Checkpoints: `checkpoints/server_ssl_6443/`, `checkpoints/server_ft_6443/`

## Remaining Issues

1. 6399 条下 SSL best val loss 0.1206，说明 30000 条正式预训练需扩大模型容量或延长 epoch。
2. direct/indirect recall 均约 65%，仍需在 inner validation 上做阈值校准与 topology 特征增强。
3. SSL 重建评价需改成 masked-position-aware validator。
4. MP 双源仍受 WSL IP 封禁与服务器 S3 网络阻断限制。
