# P3 服务器验证与受控流程启动记录

日期：2026-09-08

## 1. 范围与结论

本记录承接 `20260908_P3_reaudit_attention_execution.md`，不改写前序失败记录。宪法 5.1 的限域探索授权不等于 P2/P3 科学验收；6D/MBM/三任务/tkinter 主链、已有 immutable 数据与已验收模型不改写。

工程门禁已通过，受控流程已真实启动。启动后的已读回状态为 `controlled / running / prepare_full`（控制进程 18978，初始准备子进程 18981）。这只说明新版全量数据正在生成，不能把它写成两模型已完成训练或精度达标。进度以持久化状态文件为准。

## 2. 已验证的证据

| 检查 | 实测结果 |
|---|---|
| 核心六文件独立复审 | 通过；当前 SHA 与审查快照一致 |
| prepare/train 及对应测试四文件复审 | 通过；151 项定向测试，零安全/逻辑阻塞 |
| 本地完整回归 | 456 passed |
| 服务器完整回归 | 456 tests，0 failures，0 errors，0 skipped |
| 本地真实晶体 GPU CLI smoke | MLP / 两层 attention 均通过，重载最大差 0 |
| 服务器真实晶体 GPU CLI smoke | 两张 Tesla V100-SXM2-16GB 上分别通过；TensorFlow 2.21.0 |
| 服务器产物读回 | best/last/accepted 权重与预测 NPZ 的 SHA 匹配；优化器恢复通过；重载最大差均为 0 |
| smoke 的隔离 | fit+validation 与 smoke holdout 的全部 ID/group 交集均为 0 |

服务器 smoke 的 64 个训练样本与 16 个 holdout 样本全部来自 canonical outer-train；没有读取 canonical outer-test arrays。该验证不提供正式 outer 精度。服务端报告的实际字段是 `smoke_report.json` / `runs`，状态为 `passed_gpu_cli_smoke_after_entry_gates`。

最终证据位于 `artifacts/reports/p3_audit_20260908/deployment_local/`：
- `remote_runtime_receipt.json`
- `remote_full_regression_complete_environment.xml`
- `remote_gpu_smoke_report_conda.json`
- `controlled_plan.json`、`controlled_started.json`、`controlled_initial_state.json`

## 3. 部署、失败及环境修复

- 代码/测试/治理文件共 106 个逐文件校验；43 个目标发生同步变化，78 个原文件先行备份并读回。三份既有 60k 原始输入的 bytes/SHA 双端一致，没有重传或改写。
- 部署前检查了非 P3 差异：既有 canonical P0 辅助函数、latest accepted 的 v6→v7 指针，以及换行差异；没有借同步新增非 P3 功能。
- 第一次服务器完整回归在收集阶段因 headless `TkAgg` 失败。使用用户缓存目录内的 Xvfb，并先实测 Tk root 初始化；没有改变生产 GUI 后端，也没有跳过工作台测试。
- 第二次完整回归为 **447 passed / 9 failed**。原因是缺少 OpenCV、缺失历史 immutable 布局资产，以及遗留临时 `e3nn_smoke.py`。安装与本地一致的 `opencv-python-headless==4.11.0.86`（`--no-deps`），未升级 TensorFlow/NumPy；遗留脚本按原 SHA 归档保留。
- 对缺失历史资产先做双端清单，再镜像 26 个本地真实原件；逐文件 bytes/SHA 相符，**零已有资产覆盖**，没有建立空文件或伪造测试数据。清单与验证见 `historical_manifest.json`、`historical_remote_before.json`、`historical_assets_verified.json`。
- 直接调用环境解释器的 GPU smoke 被 GPU-only 门禁拒绝：它继承 base 环境、缺少 NVIDIA 动态库搜索路径。对照实测 `conda run` 激活后可识别两张 V100，随后在独立 `_conda` 目录重跑 CLI smoke 并成功。没有启用 CPU fallback。
- 以上失败日志和失败 smoke 目录均保留；最终全量通过没有覆盖或抹去早期失败。

## 4. 冻结的受控实验合同

- 新数据目录：`data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband_v2_20260908/`，旧 P3 和 v4–v7 资产不覆盖。
- 数据参数：16 bands、256 k、选带窗口 5 eV、`max_atoms=50`；保留分段、spin/band 原始索引、显式排除记录和 split completion / final commit。窗口用于选带，不裁平宽色散能量。
- MLP：`attention_layers=0`，GPU 0；对照：`attention_layers=2`、4 heads，GPU 1。
- 两组共同参数：seed 42、d_model 128、batch 32、学习率 0.001、最多 180 epochs、warmup 5、patience 20、min_delta 0.0001；没有 CPU fallback，也没有 CLI resume。
- 两组消费同一份已提交数据；确定性、group-disjoint 的 85/15 inner split；完整 `val_loss` 独占选模。
- 两组都完成 best/last/accepted 冻结及代码/数据核验后，才进行各自 outer evaluation；不得根据任何一组 outer 结果调整另一组配置。
- 控制流程已通过完整激活的 conda 环境 detached 启动。后台只读监控已建立；远端任务不依赖交互式 SSH 保持运行。

数据准备的 valid/exclusion 总数尚待 final commit 后核算；不把旧 P3 的有效计数冒充新版结果。历史 outer 已经有过诊断，不能把本轮外层集合称为从未接触的盲集。

## 5. 边界与后续

- 尚未取得本轮受控训练的最终准确率，P2/P3 科学状态仍为未验收。
- per-k 谱 OT 不等于 trajectory Hungarian，不恢复简并处真实 band character；归一化 scalar k 不支持物理有效质量结论。
- 正式 Bandformer 同数据对照、完整物理指标、P3 七拆分/多 seed、P4/P5 均仍待完成。
- 当前服务器代码/治理文件作为冻结快照保持不变；本地进展文档可以前进，但不在运行中覆盖远端冻结文件。结束后再按双端清单同步结果和文档。
- 临时传输与进度接口仅为 SSH 通道中的 loopback 运维接口，不是新增 web GUI；只读监控与控制辅助文件在任务结束、结果回传核验后清理。

## 6. 2026-09-08 后续读回：已进入真实训练

以 `training_progress_verified.json` 的 `2026-09-08T15:35:41.201380+00:00` 快照为准：
- 新数据 final commit 已发布，train 凭据标记 `smoke_only=false`。训练侧保留 47,912 个唯一 ID/原始行，其中 47,879 有效、33 排除；排除记录逐 ID 持久化并读回核验。
- 33 条排除全部为 `invalid_band_or_k`，详情均为 `low-quality segment: fewer than two source k points`。没有伪造或跨不连续分支补出 k 点。
- MLP 已保存 12 个 epoch，最近 inner-val loss 为 1.4241049338204206；两层 attention 已保存 8 个 epoch，最近 inner-val loss 为 1.4104009085211315。两进程实际占用不同 V100，均在运行。
- `history.json` 的 `seconds` 是从训练循环开始计的累计时间，不能直接当单 epoch 耗时。最近连续差分得到约 100.23 / 136.36 秒每 epoch；若跑满 180 轮，快照时剩余约 4.68 / 6.51 小时，早停可能缩短。已有 18 小时只读监控足以覆盖该估计，没有更改训练参数。
- 此处没有读取真实 outer arrays 或全局 prepare 报告；上述数值只是 inner 训练进度，不是最终预测精度或模型优劣结论。
- 历史原件镜像核验后，中转大压缩包与 staging 已清理，原件副本和清单/备份均保留；进度服务已切换为 GET-only，并实测 POST 返回 501。
