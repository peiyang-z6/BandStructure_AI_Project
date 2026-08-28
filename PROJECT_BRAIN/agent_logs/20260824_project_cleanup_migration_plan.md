# 2026-08-24 — Project Cleanup and Root Migration Plan

## User-approved scope

将运行项目迁移到：

`C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`

参考论文、Word、图片、表格继续保留在外层兄弟目录 `资料/`，不得进入运行项目根目录。

## Protected scientific direction

不改变完整 line-mode E(k) → `(N,2,128,3)` 6D 张量 → MBM SSL → gap/type heads → tkinter GUI。此次只做目录、路径、产物与日志治理，不新增并列训练框架。

## Target layout

```text
BandStructure_AI_Project/
├── data/
│   ├── raw/
│   │   ├── aflow/
│   │   └── materials_project/
│   └── processed/
│       └── aflow/ood_tensors/
├── artifacts/
│   ├── models/aflow_noleak_v4_seed42/
│   ├── checkpoints/aflow_noleak_v4_seed42/
│   ├── reports/aflow_noleak_v4_seed42/
│   └── logs/aflow_noleak_v4_seed42/
├── configs/
├── scripts/
├── src/{data,engine,models,utils,vision}/
├── tests/
├── PROJECT_BRAIN/
├── README.md
└── requirements*.txt
```

## Retention rules

### Keep

1. 全部已下载原始数据，包括 AFLOW 主缓存、JSON cache、下载元数据/报告/排除记录，以及 MP/AFLOW 历史 smoke 中真实下载的 HDF5/JSON/metadata。
2. 当前唯一正式 processed snapshot：`aflow_ood_tensors_6443_noleak_v4`。
3. 最新正式实验：`aflow_noleak_v4_seed42` 的 SSL、监督权重、best/last checkpoint、训练日志、完整报告与图。
4. 全部源代码、测试、配置、依赖文件和 `PROJECT_BRAIN` 历史审计日志。
5. `configs/api_keys.env`，但不输出、不写入报告、不提交 Git。

### Remove after verification

1. `stage0_*`、`stage1_*`、`phaseB_*_smoke` 的模型/checkpoint/log/report。
2. legacy `server*`、`metalfix*`、旧 6443 target-conditioned 模型、checkpoint 与报告。
3. legacy processed tensors（旧 6443、metalfix、stage smoke OOD tensors、`.sync_backup_20260813`）。
4. `__pycache__`、`.pytest_cache`、`.sync_backup_20260813`、空 probe/keycheck 目录、0-byte `=3.5,`。
5. 临时远程脚本 `scripts/server_ssh.py`（含不应留在项目中的连接辅助信息）。
6. 根目录旧 `.log`；正式日志和数据下载日志迁入分类位置后删除旧副本。

## Migration safety

1. 先生成文件级 SHA-256 清单；
2. 同一磁盘使用 move，跨目录后核对文件数、总字节和关键哈希；
3. 服务器正式微调结束并下载完整产物后，才删除旧模型/报告；
4. 路径更新采用回归测试先失败、再修改；
5. 最后运行 Python compile、完整 pytest、manifest/spacegroup 审计和真实 SSL smoke；
6. 同步更新 README、dev_context、CONSTITUTION 与本日志。

## Explicit non-goals

- 不删除任何参考资料；
- 不删除任何尚未被确认保留的下载数据；
- 不重新实现 `src` 的平行模块；
- 不把 legacy 结果重新包装成 noleak 正式指标；
- 不在此次清理中启动 Phase C 模型开发。
