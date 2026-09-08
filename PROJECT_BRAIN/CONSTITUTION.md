# BandStructure AI Project Constitution

Version: 5.1
Updated: 2026-09-08
Status: active project rules

## 1. Mission

本项目构建物理约束的晶体能带 AI 流水线。**差异化核心方向**（2026-09-06 起）：

> 建立"晶体结构 — 数值能带 — 论文/实验能带图像"的物理约束跨模态模型，用于检索、匹配、可信拒识与主动 DFT 闭环；structure→multi-band Eₙ(k) 预测是其中一个任务，而非全部卖点。

具体大方向保持为：

1. 从可追溯的数据源下载完整 line-mode E(k)；
2. 构建固定形状 6D 能带张量与 group-disjoint OOD 划分；
3. 使用 Masked Band Modeling 进行自监督预训练；
4. 微调用于带隙回归、三任务分类（line_mode_topology / provider_global_electronic_type / line_global_disagreement）；
5. 通过原生 tkinter GUI 完成人工标定的 Plot-to-Physics 推理与训练数据回流；
6. 在现有链路上按 P1→P5 顺序扩展：结构 sidecar 补全（P1）、跨模态检索（P2）、variable multi-band decoder（P3）、校准不确定性 + 主动获取（P4）、外部验证集（P5）。

允许系统性修复和兼容性升级，不得改写成与主链无关的独立模块集合。禁止单纯为扩数据量而扩到 100k，禁止从头实现完整 DeepH 类哈密顿量网络——两者都偏离差异化定位。

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

- `aflow_noleak_v7_60k_seed42` 为 latest accepted（2026-09-03：60,000 条 AFLOW、V100 GPU-only、SSL 50 epochs + 监督 51 epochs（best epoch 31 by aggregate `val_loss`）、best/last/accepted 冻结后 evaluation-only、两次 GPU OOM 已 TDD 分块修复、产物回传 SHA 校验一致）。`aflow_noleak_v6_30k_seed42_metricfix` 转为历史 accepted 保留为 immutable（GPU-only、aggregate `val_loss` 选模、best/last/accepted 三态冻结、最终 content gate + model-brain manifest 通过）。`aflow_noleak_v5_30k_seed42` 保留为 immutable historical run；2026-08-27 审计确认其 supervised checkpoint selection 只代表最后 20 条 validation 样本，不能继续作为无保留的科学 latest accepted。`aflow_noleak_v4_seed42` 继续作为 immutable baseline。
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

### P0 科研基准（2026-09-04 起生效）

- **主结果**：三任务分类基准 —— `line_mode_topology`（线模式路径可观察）、`provider_global_electronic_type`（uniform/DOS/数据库来源）、`line_global_disagreement`（预测二者冲突的二分类）。**line-mode gap MAE 不再作为主结果**，只作为 secondary learned-identity diagnostic 报告。
- 三任务标签必须由冻结张量后处理派生（`scripts/derive_three_task_labels.py`），不重建历史张量、不改写 v4–v7 NPZ。
- 监督模型含三个独立分类 head（topology / provider / disagreement）；provider head 不得混入 line-mode 规则先验。
- **七类拆分固定**：random / space-group / composition / prototype / leave-element / source-protocol / temporal，全部 `random_state=42`、group-disjoint、manifest 落盘；P0 阶段采用 canonical space-group 拆分训练、冻结模型跨拆分评估；每类拆分的全量重训矩阵需单独批准。
- **3 seeds {42, 2024, 7}**：每 seed 独立 experiment ID；汇总报告三任务 × 七拆分 × 3 seeds 的均值/方差。
- **group bootstrap**：以 spacegroup 为 resampling 单元，报告 95% percentile CI；不得用 sample-level bootstrap 冒充 group-level 不确定性。
- **错误分层**：至少按 provider type、spacegroup band、num_sites band、source catalog 四轴分层报告；disagreement 层必须单独给出。
- 元数据回填（prototype/species/species_pp/aflowlib_date）遵循非空不回退；回填记录必须裁剪到 HDF5 现有 ID 集合。

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
- 极值期望 gap head、**三个分类 head（type/provider、topology、disagreement）** 与审计明确的物理辅助损失；
- 归一化统计只来自 inner-fit；
- epoch loss/MAE/accuracy/Macro F1 必须用 stateful trackers 覆盖完整数据，canonical checkpoint/early stopping 监控 aggregate inner `val_loss`；
- 必须分别保存 canonical best、final/last 与 accepted-restored-best，并在 outer access 前写入 bytes/SHA selection manifest；
- legacy zero-gap/metal anchor gate 默认关闭；
- whole-path augmentation 默认关闭；
- 类别权重和阈值只能由 inner validation 确定。

冒烟只能验证可运行性，不得作为科研精度结论。监督物理分数与 SSL reconstruction 物理分数必须分开报告。正式分类报告至少包含 sample-weighted、group-macro 和 provider/feature mismatch strata；MC uncertainty 必须分开 raw interval coverage 与 tolerance diagnostic，后者不得命名为校准置信区间。

正式训练至少保留：数据/代码 manifest、配置、best/last/accepted 状态、完整日志、预测、最终 metrics 与 artifact SHA-256。

pipeline 入口合同：`scripts/run_full_pipeline.py` 必须以脚本方式直接可启动（项目根在 `sys.path`）；最终 artifact 校验必须引用无 TensorFlow 依赖的轻量模块（`src/utils/selection_manifest.py`），不得在 gate 处动态导入训练脚本或依赖同名的第三方 `scripts` 包；evaluation-only 必须先恢复 frozen best 再 `compile(jit_compile=False)`。

## 8. 跨模态扩展路线（P1–P5）

差异化定位（§1）要求按以下顺序扩展，每阶段有明确的前置冻结条件与退出标准；后续阶段不得在前置阶段验收前启动正式实现。

**限域授权记录（2026-09-08）**：依据用户“进行第三阶段”“按 3 → 1 → 2 执行”及“请继续”的批准，允许在 P2 尚未验收时先行进行 P3 探索性实现、纠错和受控对照训练。此例外不表示 P2/P3 已通过，不改变主链，不解除 immutable 资产、GPU-only、inner-only 选模和最终验收规则，也不允许自动越级进入 P4/P5。审计与执行边界见 `agent_logs/20260908_P3_reaudit_attention_execution.md`。

### P1 结构 sidecar 补全

- **不得修改既有 immutable HDF5**；新增只读配对 sidecar，按 material_id 关联。
- sidecar 字段（至少）：lattice 3×3、species、fractional_coordinates、magnetic moments/spin/SOC、DFT functional/U/pseudopotential、reciprocal lattice、3D fractional k-points、k-path convention、source 与结构 SHA-256。
- 数据来源：AFLOW REST 端点（`?geometry`/`?positions_fractional`/`?species`/`?dft_type`/`?spin_cell`/`?files` 实测可达；`?lattice` 404，用 `?geometry` 替代），按 AUID 逐个获取，不下载大文件。
- 覆盖审计：sidecar 与 HDF5 的 material_id 集合必须一致；缺失/低质量字段隔离规则同宪法 §4 provenance。
- 完成后 `structure` 相关字段才可进入 §6 之后的结构编码器。

### P2 跨模态检索（优先于 multi-band 生成）

- 保留当前 MBM 作为 band encoder；结构编码器复用 ALIGNN/MatGL/e3nn 类成熟 GNN，不从零发明。
- 用 structure–band 对比学习在 60k 配对数据上对齐共享物理嵌入。
- GUI/论文图像经人工定标或 CV 重建后送入同一 band encoder。
- 建立 ANN 向量索引：上传一张能带图 → 返回最相似材料、结构与置信度。
- 这是最快形成独特、可演示、可发表结果的一步，P2 通过后再进入 P3。

### P3 variable multi-band decoder

- 不固定为六条 band；用 mask 支持可变 band 数。
- band 交换与 crossing 用 band-set matching（Hungarian/最优传输）处理。
- 同时约束：segment continuity、高对称点简并、时间反演对称、VBM/CBM 位置、曲率/有效质量、metal/direct/indirect topology。
- 在相同数据与拆分下正式对比 Bandformer，而非只与当前 v7 比。

### P4 校准不确定性与主动获取

- 用深度 ensemble 或异方差头估计不确定性；在独立 calibration split 上做 conformal calibration。
- 报告 coverage、interval width、ECE/Brier、risk–coverage、错误拒识率。
- 主动学习 acquisition 组合：校准不确定性、latent-space diversity、source/prototype novelty、预期物理信息增益。
- 不得继续把当前 MC-dropout 方差直接当 DFT 选择依据。

### P5 真实外部验证集

- 至少三类外部证据：Materials Project/JARVIS source-OOD 数值能带、人工校准论文图像或公开 ARPES、少量新 DFT 计算作为时间后移 blind test。
- 主指标包括 full-band MAE、extremum k-error、effective-mass error、Recall@K/mAP、open-set AUROC、校准覆盖率；不得只剩 gap 与三分类准确率。

现有 6D 模型继续作为 QA、检索和下游基线；现有 encoder/trainer/report/provenance 机制复用。

## 9. GUI Rules

- GUI 只使用 `scripts/gui_workbench.py` 的原生 tkinter；不重新引入 Gradio/web GUI。
- 真实图像的 6D 重建依赖人工标定坐标轴、Fermi、VBM/CBM 与曲线点。
- 标注使用原图像像素坐标；缩放和平移不得改变标签。
- Fermi 标注必须参与能量零点校准；反向轴保留正确符号。
- 人工标注 train/val 按来源图像或文档分组。
- 缺少正式 vision detector 权重时，自动检测必须标记为 optional/unverified，不能声称已验收。
- 工作台状态（标注、定标值、材料信息）必须按图像内容寻址持久化到后端 JSON 缓存，重开同一图像自动恢复；状态只存图像像素坐标，不依赖缩放/平移。
- CV 提取必须输出聚合质量分（detector 置信度仅在权重存在时计入；panel 来源、骨架密度、k 向覆盖、分辨率必须实测）。GUI 提取质量指示灯绿/黄/红，黄/红必须提示人工复核。
- 脑推理不确定性只能用真实输出信号（softmax 熵、极值峰锐度、gap 合理性）；没有 dropout 层的 eval 路径不得虚构 MC-Dropout 方差。

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
