# BandStructure AI Project Constitution

Version: 6.5
Updated: 2026-09-14（MCP参考感知与来源绑定修复）
Status: active project rules

**2026-09-13 开放论文图候选规则（当前）**：批量论文内容只能经明确允许机器获取的官方开放接口采集，并逐条保留许可、文档/图/资产SHA和来源标识；禁止绕过主站的系统下载限制。图注命中、CV框、OCR或Computer Use筛查都只是AI预标注。必须在真人看到标签前冻结完整evaluation抽样框，并按文档ID分组防泄漏。非目标图、缺框图和困难图不得事后静默剔除；应作为抽样框中的reference label结果记录。只有真实独立reviewer逐图给出目标/非目标、参考框及认证绑定后，才能称为金标准或正式评测集。当前300条CC-BY图来自239个PMCID group，正式真人approved仍为0。

**2026-09-13 MCP复核边界规则（当前）**：Computer Use、浏览器自动化、AI Agent操作GUI和AI生成的复核说明均属于AI辅助预审，不能计为独立真人reviewer，也不得产生`approved`、`human_audited=true`、operator authentication或科学资格。本地工作台只允许不可覆盖地记录`needs_more_evidence`/`rejected`；正式批准必须在submission目录外的认证operator registry中由真实独立人员完成，并绑定原始文档、页面/面板、观察与证据SHA。当前3条PRX候选均为AI辅助`needs_more_evidence`，approved=0。

**2026-09-13 MCP-first P2规则与工程验收（当前）**：本地stdio配置可面向Hermes、Claude、Cursor、VS Code、Codex生成，但模板生成、配置读回、健康连接和真实Agent调用必须分级报告。不得将本地stdio宣传为OpenAI Responses远程MCP。上传PDF/图像在配置启用时必须使用独立子进程、最小环境和硬超时；不得继承provider凭据。该边界不等于Windows OS sandbox，不能声称限制峰值内存、文件系统或网络。配置包不得包含key/token/password，必须逐文件SHA绑定。

**2026-09-13 MCP-first P1规则与工程验收（当前）**：AI可以预填真实文档观察和生成稳定候选SHA，但MCP是只读服务，永远不得签发`human_audited=true`、reviewer身份、operator authentication或科学资格。P1候选固定`pending_human`；正式真人集必须由外部认证reviewer对照原页，绑定文档/图像/观察/证据SHA，并由submission目录外的operator registry冻结group/split。正式evaluation门槛仍为200–500条。当前PRX三个面板候选全部`eligible=false`、approved=0。

**2026-09-13 MCP-first P0规则与工程验收（当前）**：MCP 是主要产品与交付面；AI 客户端自身文档视觉/OCR/文字理解是主感知路径。MCP只接收来源绑定的结构化观察，负责分析规划、缺证追问、合同校验、有限解析测量和拒识。本地RapidOCR/PDF几何提取仅作可选fallback。旧60k/MBM/GNN/P2/P3/tkinter作为受保护研究资产或可选后端保留，不得以MCP可调用性补签其科学验收。AI自报感知分、OCR分、调用者人工确认均不得成为物理confidence、OOD或独立真人审计。P0最终定向回归315/315；SDK1.30与Hermes SDK2协议演练通过；这只是P0工程验收，不是图像定量、模型或科学验收。

**2026-09-12 MCP/PDF修复最终状态（当前）**：工程`PASS`仅限本轮修复和已测边界；`PDF-VIS-001-R1`同色文字与`IR-CLIP-001`裁剪标签漏洞已闭合，可见正例保留。最终父级与独审各306项定向回归通过；独审两SDK各126调用/63语义例通过。父级新鲜真实Hermes SDK2为87调用/61检查，论文14/14页、3面板、168片段/7302点、0定量，科学`NOT_ACCEPTED/false`、完整数值重建仍未完成。原301过/5导入失败已归因为混合PYTHONPATH遮蔽，原失败保留；正确普通pytest入口无代码改动全过。125源复审时保持，复审后仅同步状态文档；配置未改。历史FAIL/UNKNOWN、真人0、训练/旧资产/P4门禁保持；本有限窗口结束，不再自动加修或复审。 [最终修复与重测报告](agent_logs/20260912_mcp_pdf_renewed_final.md)。

> 下列按日期保留的旧执行状态仅用于追溯；当前结论以本段为准，不回写历史证据。

**2026-09-11 论文MCP实战已封存：工程FAIL、科学不放行（覆盖下方历史状态）**：唯一限域修复及最终独审结束；剩余`PDF-VIS-001-R1`：同色字/背景不可见文字仍可参与刻度或EF定标，已在两SDK真实MCP复现。260项定向回归及父63调用/21检查通过不能抵消该反例。本文14/14页、3面板、168几何片段/7302点、0定量，完整数值重建未完成；仅交付原图与未定标片段。停止本自动修复窗口，不再沿用旧MCP代码PASS认证新增功能。既有注册配置未改，PDF自动定量禁止作为可靠物理结论；历史UNKNOWN、真人0、训练/旧资产/P4门禁保持。 [最终报告](agent_logs/20260911_prxen_mcp_practical_final.md)。

**2026-09-10 MCP交付最终状态（覆盖下方历史执行状态，不改写历史结果）**：测量/OCR首版已交付，唯一新审`code_passed=true`，原5代码阻断全部闭合、无新产品阻断；165项定向测试及81独立用例在两SDK各通过。历史helper绑定缺失与共享GREEN偏差保留，过程UNKNOWN、无保留总验收false。最终父核exit0，119冻结源保持，代码/示例包CRC与逐文件核验通过，独审证据已归档。**Hermes已注册（19:02续办核验）**：用户要求继续后已在default启用bandstructure；12注册项（8业务工具＋4资源/提示包装）和20次注册层调用通过，新建真实Hermes会话的2次MCP调用及对应返回已读回核验。其他配置与119源文件保持不变；此前取消记录仍保留。当前旧会话未热加载新增工具，新会话已验证可用。 部署细节见[启用记录](agent_logs/20260910_mcp_hermes_enabled.md)。不是完整60k跨模态模型；ANN/预测/校准/真人集/主动DFT仍blocked。本窗口结束，无待回传、不再自动迭代；详见[最终报告](agent_logs/20260910_mcp_delivery_final.md)。

**2026-09-10本窗口最终收尾（当前状态）**：deleg_d78d3b96两项已交接并父核，无本窗口待回传任务，不自动新增循环。联合353独立检查352过/1失败，JIF-L1空白语义证据被上游发布而对应test消费拒绝；正常train分区通过，非训练绕过。TYPE/NUMERIC等仅局部PASS，过程UNKNOWN。K01原记录/原字节回放/A参考仍78；持有O_PATH的B及父侧持有源FD的C参考为0，仅有限可行性，原隔离器与K02–K13未实现/批准。两审查加父回执1041文件已归档(不含归档清单，不复制raw)，R106限定源码不变。未合入候选、未全仓/训练/P4，科学与旧UNKNOWN保留。详见 `PROJECT_BRAIN/agent_logs/20260910_single_review_window_final.md`。

**2026-09-10新授权整改（覆盖旧批次停编状态，不改写历史结果）**：用户明确要求“请继续修复，修复结束后立即开始p4，如需用户选择的时候，就按照默认最优选项选择，不要让用户确认。”现恢复新的原地修复/复审/完整回归，默认最优选择直接执行；修复后直接衔接P4前置合同与实现验证。121基线/106Python已核SHA同上一冻结，六旧对象仅核stat。本轮不覆盖历史、不扩材料、不正式训练、不真实raw/outer/full prepare、不补签；旧UNKNOWN/TDD偏差仍保留。工程修复不冒充科学准入，P4正式校准/主动DFT仍受真实独立calibration及科学证据门禁。详见`PROJECT_BRAIN/agent_logs/20260910_renewed_repair_and_p4.md`。

**v2整仓及只读归因已交接（20260909批次，覆盖下文旧状态）**：完整1951项精确同节点/0排除，1950通过、1个注释layout失败、0错误、0跳过；280旧问题现通过、0新测试失败，106源码/6旧资产记录/23源证据及保护器均核稳。deleg_ae4bbfc2已正式交接，父核315证据项并完成专项归档；有界回放确认私有TemporaryDirectory清理中dir_fd相对data被G按项目CWD误解析，非真实项目目录。但原PID2293两行缺raw args/fd/节点/历史inode，不能用回放对象补签历史，完整历史绑定保持UNKNOWN、review_passed=false，不再等待该子任务。P2的JSON重复键与极短k轴反例、源QA覆盖、P3-L1-R1及TDD流程偏差仍阻断；不新增修复或全仓、不正式训练，P4 NO-GO。读取范围仍仅原v4/30k限定元数据/SHA和既有六源的精确路径/SHA，不开放60k目录或其他材料。详见`PROJECT_BRAIN/agent_logs/20260910_denial_attribution_final.md`。

**追加授权与第二轮父级预检（2026-09-09，覆盖下文旧执行状态）**：用户明确回复“同意两项授权”，允许指定v4/30k旧HDF5只读组名/数量与旧manifest/SHA校验，并单独追加P3-L1/L2两项prepare原地TDD修复及新独审；不扩展P2第三轮，不读取当前60k/outer或能量数组，不改写历史资产、不扩材料、不正式训练。第二轮六路已全部交接，父级预检核10份OWN源码、112份快照/106份Python编译及原JUnit；P3格式桥新独审PASS，prepare两项仍待修后独审。四路最终只读复审与P3限域修复已派`deleg_a21da317`；最终共同源码全仓回归须在P3停止编辑后运行。各子流计数不能相加或当作全仓通过。科学门槛与人工0/BLOCKED不变，P4 NO-GO。详见`PROJECT_BRAIN/agent_logs/20260909_additional_authorization.md`。

**循环1新独审与第二轮限定修复（2026-09-09，覆盖下文旧执行状态）**：六路交接已全部消费；上游/runtime/ANN新独审均FAIL，程序计数7个独立阻断问题。runtime L1–L7、ANN S1/L1/L3/L4仅该层工程局部通过，不能抵消上游raw/anchor缺口或真实认证缺失。P3两补丁已回传，父级核JUnit与11份冻结文件SHA，现做新独审；五异常＋一对照的源语义已独立复核，原包168文件逐SHA匹配，QA需补四项内容指纹。已冻111份代码/治理，第二轮四路小修＋两P3独审进行；最终共同源码回归和修后独审尚未通过。最多两轮后仍失败即报告剩余问题，不自动扩改。人工0/BLOCKED，P0/P1/物理/calibration/跨设备门槛不变，P4 NO-GO；不覆盖历史、不扩材料、不正式训练。详见`PROJECT_BRAIN/agent_logs/20260909_fix_cycle2.md`及`artifacts/reports/remediation_20260909/{p2_fix_cycle2_start,cycle1_reviewed_evidence}/`。

**P2中期回验补注（2026-09-09，不改变规则或5.1授权）**：运行合同与原生HNSW查询的合成CPU制品已父级读回，新图-runtime适配接口已联通；旧缓存不得借此升级来源或池化语义。修复循环1前1,099项CPU通过不替代真实数据、GPU、人工图像审计或独立复审；提交的人工清单合格记录为0，P2科学与P4仍不放行。证据见`artifacts/reports/remediation_20260909/parent_p2_handoff/summary.json`及整改执行台账。

**独立审查更正（2026-09-09，修复循环1）**：P2运行与检索两项独审均FAIL，共13项安全/逻辑缺陷；1,099项常规CPU通过与合成回放只证明先前快照的已测路径，不能抵消独立负例。“接口接通”不等于v3语义/来源/冻结/审核合同验收。当前按上游实际凭据产出、运行门禁、图像/审核三路进行修复；不正式训练、不全量准备、不补签历史来源，P4 NO-GO。修前快照、父级3项RED和责任边界见整改执行台账及`artifacts/reports/remediation_20260909/p2_review_fix_cycle1/`。

本段仅更正验收状态，不改变5.1方向、模型/损失/邻居参数与授权边界。

## 1. Mission

本项目构建面向 AI Agent 的物理约束能带分析 MCP。**主要方向**（2026-09-13 起）：

> AI 工具负责理解原始文档；MCP把能带分析方法、证据需求、可审计计算和诚实拒识固化为跨Agent工具协议，而不是继续以单独GUI应用为主要交付。

具体主链为：

1. AI 客户端在原始附件上执行自身文档视觉/OCR和文字理解；
2. AI 提交带文档ID、页码、面板、轴、刻度、Fermi、曲线和歧义的结构化观察；
3. MCP先规划和检查证据，缺失/矛盾时提出下一观察问题，不猜测；
4. 证据齐全后只执行声明范围内的sampled-path解析测量；
5. 本地OCR/PDF几何解析作为可选fallback，不替代AI主路径或人工审计；
6. 旧数据、模型、检索和GUI仅在完成独立来源/校准/OOD门禁后作为可选后端接入。

允许系统性修复和兼容性升级。禁止把未实现工具、未校准confidence/OOD、模型旧指标或合成图成功包装成MCP完整能力。

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
├── mcp_server/
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

**执行结果记录（2026-09-09 UTC复核）**：上述seed42受控对照已完成、双方权重冻结及最终预测已核验归档；谱误差点估计改善并不伴随gap/金属诊断改善，跨环境严格逐点等价亦未通过。该记录不变更5.1授权、latest-accepted指针或阶段退出标准，P2/P3仍未科学验收，不放行P4/P5。最终结果与保留失败见 `agent_logs/20260909_P3_controlled_final_results.md`。

**整改中期状态注记（2026-09-09）**：用户已批准修复前序数据、实现和科学验收缺口。当前代码的完整CPU回归1,099项通过，不代表P0–P3科学通过；旧七拆分的seen池问题、P1完整合同unknown/路径问题和真实模型/数据证据仍须分项验收。新三任务formal与P3旧格式兼容分别验证，禁止借legacy兼容补造历史来源或放宽formal门禁。该注记不改变大方向、latest-accepted工程指针、5.1探索性授权和P4/P5退出标准。证据见`agent_logs/20260909_remediation_execution.md`。

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

**2026-09-10限域衔接授权**：用户明确要求续修完成后立即开始P4，普通方案默认最优选项。本轮修复/新复审后，可开始合成CPU范围的P4合同和原地实现验证，不再以旧批次“不自动新增修复”阻止本轮。此为正式科学准入前的限域工程工作，不表示P0–P3已科学通过；不取消独立calibration、真实来源、冻结/无泄漏、immutable、不正式训练/全量prepare/真实outer读取和主动DFT执行边界。

- 用深度 ensemble 或异方差头估计不确定性；在独立 calibration split 上做 conformal calibration。
- 报告 coverage、interval width、ECE/Brier、risk–coverage、错误拒识率。
- 主动学习 acquisition 组合：校准不确定性、latent-space diversity、source/prototype novelty、预期物理信息增益。
- 不得继续把当前 MC-dropout 方差直接当 DFT 选择依据。

### P5 真实外部验证集

- 至少三类外部证据：Materials Project/JARVIS source-OOD 数值能带、人工校准论文图像或公开 ARPES、少量新 DFT 计算作为时间后移 blind test。
- 主指标包括 full-band MAE、extremum k-error、effective-mass error、Recall@K/mAP、open-set AUROC、校准覆盖率；不得只剩 gap 与三分类准确率。

现有 6D 模型继续作为 QA、检索和下游基线；现有 encoder/trainer/report/provenance 机制复用。

## 9. Legacy GUI Rules

- GUI 是保留的本地人工标注工具，不是MCP-first产品入口；如继续使用，只保留 `scripts/gui_workbench.py` 的原生 tkinter。
- 真实图像的 6D 重建依赖人工标定坐标轴、Fermi、VBM/CBM 与曲线点。
- 标注使用原图像像素坐标；缩放和平移不得改变标签。
- Fermi 标注必须参与能量零点校准；反向轴保留正确符号。
- 人工标注 train/val 按来源图像或文档分组。
- 缺少正式 vision detector 权重时，自动检测必须标记为 optional/unverified，不能声称已验收。
- 工作台状态（标注、定标值、材料信息）必须按图像内容寻址持久化到后端 JSON 缓存，重开同一图像自动恢复；状态只存图像像素坐标，不依赖缩放/平移。
- CV 提取必须输出聚合质量分（detector 置信度仅在权重存在时计入；panel 来源、骨架密度、k 向覆盖、分辨率必须实测）。GUI 提取质量指示灯绿/黄/红，黄/红必须提示人工复核。
- 脑推理不确定性只能用真实输出信号（softmax 熵、极值峰锐度、gap 合理性）；没有 dropout 层的 eval 路径不得虚构 MC-Dropout 方差。

## 9.1 MCP Human Audit Rules

- AI预填、AI感知分、`human_confirmed`自声明或MCP候选SHA都不是人审认证。
- `prepare_human_audit_candidate`只能生成`pending_human`记录，不得写审查registry或批准记录。
- 独立reviewer必须查看原始页面/面板，逐项确认轴、单位、Fermi、连续曲线、路径分段及歧义。
- operator registry必须位于submission目录外，并冻结review证据SHA及document/group/split分配。
- 200–500条evaluation记录门槛不得由合成图、AI自审、重复图或未绑定证据填充。

## 9.2 MCP Portability and Isolation Rules

- 本地客户端模板必须使用绝对Python/server路径、stdio、受控环境变量和前台进程；生成不等于安装。
- 原生客户端状态按`template/generated`、`configured/readback`、`connected`、`agent-called`分层，禁止混称。
- 上传worker只继承运行所需OS变量；AI provider凭据和其它非必要环境不得进入worker。
- worker超时、崩溃、非法/超大输出必须fail-closed，不能退回父进程内静默解析。
- Windows VS Code没有官方stdio sandbox；子进程超时隔离不得命名为OS sandbox。
- Streamable HTTP、认证、限流和远程OpenAI MCP在实现及独立验证前保持未交付。

## 10. Documentation and Verification

### 2026-09-14 MCP合同补充

- 隔离PDF续取必须绑定同一父进程不可变结果；过期/重启不允许重算后拼接。进程有界缓存不是OS sandbox。
- CV候选不具备电子/声子/输运语义。宿主ROI须绑定源SHA、页号、坐标单位和类别；空矢量结果须显式声明位图回退/缺证原因。
- 新v2允许fermi/vbm/arbitrary参考、稀疏多带和不连续路径；EF未知保持null，缺失不插值。旧GUI/v1要求EF的合同不强加给v2。
- OCR更正必须保留原文本及来源SHA/坐标单位/更正证据；AI来源的更正和图像导出都不形成独立人工认证。
- 版本、启动时间和构建SHA只能证明对应进程快照，不能把新进程通过冒充旧会话热加载。工程通过与完整DFT/科学准入分别记录。

### 原有同步要求

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
