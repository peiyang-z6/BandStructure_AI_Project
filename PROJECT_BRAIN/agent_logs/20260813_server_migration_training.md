# 2026-08-13 Server Migration + Training Run ([REDACTED])

## Scope

按用户要求：检查 MP key 连通性 → 下载 2000 条 → 打包传输到服务器 → 配置兼容环境 →
完整训练 + 报告。全程不改项目大方向，仅做环境适配与小修复。

## 1. MP API 连通性结论（重要）

- 三个 key 字段：`MP_API_KEY_NEW`(32位，格式正确)、`MP_API_KEY`(16位，位数不足)、`MP_API_KEY_LEGACY`(空)。
- **决定性证据**：`api.materialsproject.org` 端点可达，但带 NEW key 与**不带 key** 均返回
  **同一 403**："Your IP address or ASN has been (temporarily) blocked ... due to inefficient
  or abusive traffic ... tips-for-large-downloads"。
- **根因**：WSL 代理 `127.0.0.1:10808` 出口 IP 因批量下载（mp_download_live.log 显示 /10000 级请求）
  被 MP 反滥用封禁。**与 key 无关，换 key 无效**。
- **决策（用户确认）**：MP 数据放到服务器下载（服务器为另一 IP，未被封）。

## 2. 数据下载（AFLOW，达标）

- 断点续传 `--target 2400`：**总缓存 2400 条**（本次新增 400，失败 0）。
- 候选池元数据扩到 9509 条 / 179 空间群。
- 上游空能带隔离累计 598 条（不污染数据）。
- 最终张量 `aflow_ood_tensors_final`：**X=(2400,2,128,3)，173 空间群，train/test=1920/480，零交集**。
- 三分类（提供方标签，合宪 §4）：train {metal 153, indirect 649, direct 1118}，
  test {metal 38, indirect 162, direct 280}；metal 类达标（≥150）。中位带隙 3.451 eV。

## 3. 服务器环境（已验证兼容）

- OS Ubuntu 26.04 LTS x86_64；CPU Xeon Gold 5118 24 核；内存 45GB；磁盘 3.0T 可用。
- GPU 2× Tesla V100-SXM2-16GB（驱动 580.173.02，CUDA 13.0 运行库）。
- conda 26.5.3 于 `~/miniconda3`，**已有环境 `bandstructure_ai`**：Python 3.11.15 +
  TF 2.21.0（识别双 GPU）+ h5py 3.14 + numpy 1.26.4 + sklearn 1.9.0 + mp_api。
- pip 走清华镜像，PyPI 可达。
- 服务器上 `py_compile` 全过、`pytest` **18 passed**。

## 4. 打包与传输

- 排除 PDF/图片/Word/Excel/`data_cache/json_cache`/`.git`/`models`/`checkpoints`/`logs` 及各 smoke 目录。
- 打包 188MB → SFTP 传 `/home/zhao/bandstructure_ai_pkg.tar.gz`（21s，内网）→
  解压到 `~/BandStructure AI/`，大小校验一致，临时包已清理。

## 5. 训练执行与 cuDNN 兼容性修复（关键工程发现）

### SSL（GPU，成功）

- 60 epoch，~2.2s/epoch，双 V100；val total loss 收敛至 ~0.005 量级。
- 产物：`models/server/ssl_mbm_pretrained.keras`（best 恢复）、`ssl_mbm_final_epoch60.keras`、
  `ssl_mbm_norm_stats.json`、`checkpoints/server_ssl/ckpt-*` 齐全。

### 微调的 cuDNN 失败与定位

- 现象：`ExtremumExpectedGapHead` 的 Conv1D 触发 `No algorithm worked! /
  <unknown cudnn status: 5003>`（前向）与 `Conv2DBackpropFilter` 失败（反向）。
- 根因（确诊）：TF 2.21 pip 自带 **CUDA 12.5.1 + cuDNN 9**，服务器驱动加载 **CUDA 13.0**
  运行库，二者 ABI 不兼容 → V100（sm_70）上 Conv backward op 无法实例化。
- 这是**环境二进制不兼容，非代码 bug**。SSL 的 Transformer（无卷积）不受影响，
  仅微调的 extremum Conv1D head 触发。
- 失败的尝试（均已回滚）：`TF_USE_CUDNN=0`（XLA 忽略）、XLA flag（flag 不存在/不生效）、
  `tf.device("/CPU:0")` 单独（自定义 train_step 的梯度仍 colocate 到 GPU）。

### 最终方案（务实）

- **微调改纯 CPU 训练**：`CUDA_VISIBLE_DEVICES=""` 隐藏 GPU，全部 op 走 CPU。
  服务器 24 核足够（~26s/epoch），彻底绕开 cuDNN/CUDA ABI 问题。
- 恢复 `finetune_supervised.py` 为原始版本（`.bak_convfix` 备份），不在代码里留 hack。
- **结果**：微调成功进入训练，Epoch 1-6 正常，val_gap_mae 起步 ~0.33-0.42 eV。

## 6. 观察到的训练稳定性问题（待 Stage 2/3 处理）

- warmup 期 train gap_loss 在 epoch 2 冲至 ~281 后回落：学习率爬升 + extremum head
  温度/权重初始化导致的瞬时尖峰，val 未发散。后续可用更平缓 warmup / 梯度裁剪 / 降低
  extremum_weight 初值缓解（属 Stage 3 算法调优，非本次范围）。

## Protected / Compliance

- 未删除 PROJECT_BRAIN/、data_cache/、models/、checkpoints/、reports/ 任何内容。
- `configs/api_keys.env` 传输系用户明确要求（服务器下载 MP 所需），不在 git 跟踪。
- 6D 张量契约、空间群 OOD 外层契约、单一 LR 控制器均未改动。

## Next

- 监控微调至 60 epoch 完成，收集 reports/server_ft 训练报告。
- 服务器侧另行下载 MP 数据（IP 未封），做 domain shift 对照。
- 将服务器训练报告与（可选）修复说明同步回本地 PROJECT_BRAIN。
