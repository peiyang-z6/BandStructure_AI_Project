# BandStructure AI Project Constitution

Version: 4.9
Updated: 2026-08-27
Status: active project rules

## 1. Mission

本项目构建物理约束的晶体能带 AI 流水线，大方向保持为：

1. 从可追溯的数据源下载完整 line-mode E(k)；
2. 构建固定形状 6D 能带张量与 group-disjoint OOD 划分；
3. 使用 Masked Band Modeling 进行自监督预训练；
4. 微调用于带隙回归和 metal/direct/indirect 分类；
5. 通过原生 tkinter GUI 完成人工标定的 Plot-to-Physics 推理与训练数据回流；
6. 在上述现有链路上逐步扩展 crystal graph→multi-band Eₙ(k)、DFT QA/检索与不确定性主动学习。

允许系统性修复和兼容性升级，不得改写成与主链无关的独立模块集合。

## 2. Canonical Project Root and Protected Assets

运行根目录：

`C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`

参考资料目录是外层兄弟目录 `资料/`，不得放入运行根目录。

除非用户明确指定准确目标并完成迁移/哈希验证，否则禁止删除：

- `PROJECT_BRAIN/` 和本宪法；
- `configs/api_keys.env`；
- `data/raw/` 中已下载的 canonical、supplemental 与 provenance 数据；
- `data/processed/` 中当前正式张量与 manifest；
- `artifacts/` 中当前正式 models、best/last checkpoints、reports 和 logs；
- 外层 `资料/` 中的论文、文档、图像与表格。

旧 smoke/legacy 派生物只有在确认不含唯一下载记录、最新训练产物或唯一日志，并已形成迁移/删除清单后才可删除。

## 3. Directory Contract

```text
BandStructure_AI_Project/
├── data/{raw,processed}/
├── artifacts/{models,checkpoints,reports,logs}/<experiment_id>/
├── src/{data,engine,models,utils,vision}/
├── scripts/
├── tests/
├── configs/
└── PROJECT_BRAIN/
```

- `aflow_noleak_v5_30k_seed42` 保留为 immutable historical run；2026-08-27 审计确认其 supervised checkpoint selection 只代表最后 20 条 validation 样本，不能继续作为无保留的科学 latest accepted。`aflow_noleak_v6_30k_seed42_metricfix` 在完成服务器 GPU 训练、回传和本地验收前只能称 candidate；`aflow_noleak_v4_seed42` 继续作为 immutable baseline。
- 不重新创建根级 `data_cache/`、`models/`、`checkpoints/`、`reports/` 或 `logs/`。
- `src` 的现有责任边界优先于新建平行模块；入口脚本保持稳定，确需移动时必须先加路径回归测试。
- 临时文件、远程连接脚本、`__pycache__`、`.pytest_cache`、下载归档和迁移 quarantine 不得留在最终运行根目录。

## 4. Data Sources and Provenance

- 正式数据源为 Materials Project 与 AFLOW；source-aware 校准前分别缓存、分别审计。
- 只有包含完整 E(k)、费米参考、k-path 和空间群信息的数据才可进入核心 6D 流水线。
- 必须保留 source、原始 ID、下载 URL、标签来源、下载报告与校验信息。
- canonical training cache 与 supplemental/variant 数据必须隔离；不得为了“保留全部数据”而把 smoke 唯一记录静默混入正式训练快照。
- HDF5/JSON 去重必须用 material ID 与稳定物理语义哈希；identity/download provenance、`source_efermi_absolute`、字段缺失与显式 `None` 不得制造伪 variant，已有 `_semantic_sha256` attr 不得替代对 canonical datasets/attrs 实际内容的回验。同 ID 内容不一致时保留 variant，禁止覆盖。
- canonical HDF5 写入、ID readback 与 metadata reconciliation 必须持有同一 cache 的跨进程单 writer lock；同 ID、同语义内容幂等跳过，同 ID、不同语义内容只能写入隔离的 provenance variant。锁回归必须使用真实 child-process contention，并兼容 POSIX `fcntl` 与 Windows `msvcrt`。
- candidate catalog/cursor 与 canonical metadata sidecar 必须分离；canonical metadata 的 material ID 集合必须始终与 canonical HDF5 完全一致（包括 HDF5 不存在时清除 stale IDs），空值或缺失字段不得降级已有丰富记录，历史 duplicate 与 incoming 的非空冲突都必须保留幂等审计 variant。
- canonical metadata 与 conflict provenance 的跨文件提交必须使用可恢复 write-ahead checkpoint；exclusion、candidate cursor、download report 和其它 JSON rename 必须同目录原子替换，并在支持的平台同步父目录。异常中断后必须可从磁盘状态续传，不能依赖进程内计数。
- AFLUX 分页必须固定 `page_size`，将其纳入 query fingerprint，并在 per-gap-bin cursor 中保存未消费页尾；candidate `limit` 是硬上限，gap-bin 缺额可以重分配但不得跳过或丢弃候选。MP batch adapter 成功返回数必须与请求数完全一致。
- 并发 fetch 的每个 completed outcome 必须在协调线程即时处理；同批另一 future 抛 `KeyboardInterrupt`/`SystemExit` 时，已完成的 exclusion/save/stats 不得丢失。payload in-flight 必须有界，canonical HDF5 仍只能由协调线程串行提交。
- 下载完成主口径只能是 persisted canonical count。report 至少包含 `initial_cached`、`requested_target`、`attempted`、`submitted`、`completed`、`peak_in_flight`、`total_cached`、`target_reached` 与 `termination_reason`；`target_reached=false` 或 HDF5 count 小于 target 时 downloader 必须非零退出，full pipeline 必须在 tensor/training 前停止，并在结束前对未显式跳过阶段的 required artifacts 执行 fail-closed 门禁。
- 已验收 raw/processed snapshot 是 immutable：扩容必须创建新的版本目录、manifest 与 experiment ID，不得向历史 snapshot 原地追加。历史 v4 6,443 baseline 与 v5 30,000 snapshot 必须物理分离并分别保留哈希。
- 正式数据交付必须包含原始响应、失败/排除记录、下载日志、逐文件或聚合 SHA-256、tensor/split audit 和代码 builder hash。
- 用户指定 GPU-only 的正式训练必须在启动前验证目标 NVIDIA 设备及关键 forward/backward 算子；无 GPU、GPU op 失败或 CPU fallback 均应停止而非继续。
- API 密钥不得写入日志、报告、测试夹具、终端输出或 Git 跟踪文件。

## 5. Leakage and Evaluation Rules

- 外层 OOD 以 `spacegroup_number` 为 group key；固定 `train_size=0.8`、`random_state=42`、group overlap=0。
- 外层 OOD test 仅用于训练结束后的最终评估；不得用于 early stopping、checkpoint、学习率、阈值、calibration、超参数选择或归一化。
- canonical supervised 执行必须拆成 `train-only` 与 `evaluation-only`：train-only 不得读取 outer arrays、outer class/group statistics 或 outer sample manifest；best/last/accepted 的 bytes/SHA 和 aggregate inner monitor 冻结后，evaluation-only 才可加载 outer test。
- 模型选择只使用 outer-train 内部的 group-disjoint validation，并基于完整 validation 聚合；最后一个 batch 的裸标量不得作为 epoch metric 或 checkpoint monitor。
- 当前 OOD 只能准确命名为 **space-group-disjoint OOD**；没有独立证据时不得声称 composition/prototype/source OOD。
- provider 标签只能作为 target/audit，不得改变输入能带选择、费米锚定、mask、插值或 feature construction。
- fallback 标签必须在 manifest/report 标注，不能包装成独立标签验证。
- composition/prototype/source overlap 必须与 space-group overlap 分开报告。
- 旧 target-conditioned 结果永久标记为 legacy diagnostic，不得恢复为正式 baseline。

## 6. Tensor Contract

```text
X: (N, 2, seq_len, 3)
band axis:    0=occupied/VBM-like edge, 1=empty/CBM-like edge
channel axis: 0=energy, 1=curvature, 2=extremum k-distance

flattened input: (N, seq_len, 6)
[VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]
```

- AFLOW `bands_data` 使用 canonical `E_F=0`；绝对 `Efermi` 只作 provenance。
- 所有样本统一由 E(k)+E_F 构造 label-free occupied/empty edge envelopes。
- shape-preserving interpolation 优先；不得用会制造费米附近过冲的普通 cubic spline。
- k-path segment IDs 必须进入 NPZ/manifest；curvature、crossing、span mask 与局部 physics loss 不得跨 segment。
- 自旋极化数据不得只取第一个 spin channel。
- 无倒易长度单位时只能报告 relative curvature proxy，不能伪称以 `m_e` 为单位的有效质量。
- line-mode crossing、tensor gap 和 provider global gap 必须分开命名。
- 当 regression target 等于输入函数 `min(CBM_E)-max(VBM_E)` 时，报告必须同时给出解析 identity baseline（MAE/RMSE=0）并将 learned head 解释为 soft-extremum approximation；不得包装为独立 DFT 或未知结构预测精度。

## 7. Training Rules

### SSL

- learnable mask token；
- 5–15 point segment-aware span masking；
- **实际** per-sample mask fraction 必须在 15–30%，不能只检查请求参数；span overlap/segment truncation 后必须补足或重新选择；
- training corruption 可随机，inner validation corruption 必须由固定 stateless seed 复现；
- masked-position MSE/MAE 按真实 masked elements 加权，mask fraction 按 positions，加权结果进入 machine-readable history；
- curvature-sign loss 可在 segment 内使用；没有物理 k-coordinate/统一量纲时，curvature-magnitude consistency 必须 hard-disabled 且自适应调度不得复活；
- whole-path index warp 不得称为真实 lattice strain，且在未实现 segment-aware/structure-aware 合同前默认关闭；
- 单一 warmup+cosine schedule；
- gradient clipping；
- best/last checkpoint、resume、atomic history 与 early stopping。

### Supervised

- 冻结早期 encoder 层，并冻结 supervised forward 不使用的 SSL projection/reconstruction heads 与 mask token；
- 极值期望 gap head、type head 与审计明确的物理辅助损失；
- 归一化统计只来自 inner-fit；
- epoch loss/MAE/accuracy/Macro F1 必须用 stateful trackers 覆盖完整数据，canonical checkpoint/early stopping 监控 aggregate inner `val_loss`；
- 必须分别保存 canonical best、final/last 与 accepted-restored-best，并在 outer access 前写入 bytes/SHA selection manifest；
- legacy zero-gap/metal anchor gate 默认关闭；
- whole-path augmentation 默认关闭；
- 类别权重和阈值只能由 inner validation 确定。

冒烟只能验证可运行性，不得作为科研精度结论。监督物理分数与 SSL reconstruction 物理分数必须分开报告。正式分类报告至少包含 sample-weighted、group-macro 和 provider/feature mismatch strata；MC uncertainty 必须分开 raw interval coverage 与 tolerance diagnostic，后者不得命名为校准置信区间。

正式训练至少保留：数据/代码 manifest、配置、best/last/accepted 状态、完整日志、预测、最终 metrics 与 artifact SHA-256。

## 8. Phase C Extension Rules

Phase C 不得直接从“结构”跳到黑箱标量 gap。进入代码前必须先冻结：

- lattice、species、fractional coordinates 与 periodic neighbor graph；
- spin/SOC/magnetism 与计算协议字段；
- 统一 k-path 与 segment representation；
- multi-band target、band ordering/degeneracy 与能量 reference；
- structure/composition/prototype/source OOD；
- 缺失字段与低质量标签隔离规则。

Phase C 应复用现有 encoder/trainer/report/provenance 机制，现有 6D 模型继续作为 QA、检索和下游基线。

## 9. GUI Rules

- GUI 只使用 `scripts/gui_workbench.py` 的原生 tkinter；不重新引入 Gradio/web GUI。
- 真实图像的 6D 重建依赖人工标定坐标轴、Fermi、VBM/CBM 与曲线点。
- 标注使用原图像像素坐标；缩放和平移不得改变标签。
- Fermi 标注必须参与能量零点校准；反向轴保留正确符号。
- 人工标注 train/val 按来源图像或文档分组。
- 缺少正式 vision detector 权重时，自动检测必须标记为 optional/unverified，不能声称已验收。

## 10. Documentation and Verification

结构性修改必须同步：

- `README.md`；
- `PROJECT_BRAIN/dev_context.md`；
- 本宪法；
- `PROJECT_BRAIN/agent_logs/` 日期日志。

每次数据/训练/目录升级至少验证：

1. Python compile；
2. 完整单元/回归测试；
3. 真实数据 group count、shape、split overlap 与 SHA-256；
4. 关键模型 load-weights 与真实 forward smoke；
5. 外部/服务器状态变更的读取回验；
6. Git diff 规模与行尾污染检查。

“完成”必须有磁盘产物与真实命令输出，不得仅凭计划、日志片段或进程启动状态宣称。
