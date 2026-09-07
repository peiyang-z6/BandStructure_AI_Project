# 2026-09-07 P2 服务器部署 — 正式训练日志

## 目标

P2 跨模态检索脚手架已在本地 smoke 通过，本阶段将代码 + 数据部署到服务器
V100 跑正式对比训练（40 epochs + 全量检索评估）。

## 传输（litterbox 中转，MD5 双向校验）

- 打包 `p2_deploy.tar.gz`（20M，含 sidecar 130M→压缩后 20M + 5 个 P2 代码文件）。
- 本地 MD5 `c2c599dbc9e87caa0ce3ad1cacdfd0e9`；服务器下载后 MD5 一致。
- 传输踩坑：20M 单文件下载卡在 13-15M（跨境链路到 catbox CDN 不稳），
  `wget -c` 断点续传一次成功（实际大小 20,967,944 字节，非 20,000,000）；
  分片（4×5M）反而全 FAIL（当时链路恰好最差）。教训：大文件用 `wget -c`
  断点续传循环重试，不要用分片。

## 服务器部署

- 服务器项目目录 `/home/zhao/BandStructure_AI_60k_20260903`（P0 已建）。
- 已有：张量 NPZ `band_tensors_ood_split.npz`（142M）、v7 模型
  `ssl_mbm_pretrained.keras`（2.5M）、conda env `bandstructure_gpu30k`
  （pymatgen 2026.8.13、TF 2.21）、2×V100（16G，空闲）。
- 部署 5 个代码文件 + sidecar；`prepare_p2_pairs.py` 服务器端重跑生成配对
  数据（train 1.4G / test 345M，652.5s）。

## 正式训练

- 命令：`train_p2_contrastive.py --epochs 40 --batch-size 64 --temperature 0.07
  --require-gpu`。
- 验证：V100-0 可见、GPU-only gate 通过、epoch 1 loss 3.3952 → 2 loss 3.0776
  （下降正常）。
- 训练进程 pid 117946（V100-0 2497 MiB，51% GPU util）。
- 日志 `/home/zhao/p2_train.log`，产物目录 `artifacts/models/aflow_noleak_v7_60k_seed42/p2_contrastive/`。

## 待办

- ✅ 训练完成 + 全量检索评估（train 47,912 / test 11,948 双向 Recall@K/mAP）。
- ✅ 回传产物（p2_report.json + structure_encoder.keras），MD5 `4dfe9c580f...` 双向一致。
- 本地解包、验证、更新 dev_context + README + 提交。

## 正式训练结果（2026-09-07，诚实基线）

40 epochs（V100-0），loss 3.3952 → 2.0895（单调下降）。但检索泛化失败：

| 方向 | train recall@1 / @10 / mAP | test recall@1 / @10 / mAP |
|---|---|---|
| structure→band | 0.0745 / 0.3925 / 0.1726 | **0.0035 / 0.0180 / 0.0116** |
| band→structure | 0.0155 / 0.1145 / 0.0535 | **0.0005 / 0.0085 / 0.0063** |

- test recall@1 ≈ 1/286 = 0.0035，**等于随机水平**。
- train 集有记忆（median_rank 19），test 集无泛化（median_rank 635）。
- 结论：**40 epochs 的 InfoNCE 让 CGCNN 结构编码器记住了训练配对，但未学到可
  泛化的结构→能带语义**。这是严重过拟合，不是 bug。

### 根因假设（待诊断确认）

1. **只训练结构编码器 + 冻结 band encoder**：结构编码器容量足以记忆 47,912
   样本，40 epochs 无早停/正则，直接过拟合。
2. **空间群 OOD 过强**：test 是全新空间群，纯结构特征（one-hot 原子 + 距离）
   不足以跨空间群泛化到能带拓扑。
3. 对比学习温度 0.07 + batch 64 可能过小，负样本不足。
4. 结构→能带本身是高难度映射（需要电子结构计算级别的物理），单凭 CGCNN
   one-hot 特征 + 3 层消息传递在 40 epochs 内不够。

### 下一步（需用户决策）

- 诊断 train/test embedding 分布（是否 collapse）；
- 尝试：加早停/正则、增加 epochs、更大 batch、降低温度、联合微调 band
  encoder、或接受"P2 检索需要更强结构编码器（e3nn）"的判断。
- 宪法 §8 P2 要求"P2 通过后再进 P3"——当前 P2 未达标，不应进入 P3。
