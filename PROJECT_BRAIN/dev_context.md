# BandStructure AI Project — Dev Context

Last updated: 2026-08-26
Current root: `C:\Users\PeiYang\Documents\AI Project\BandStructure AI Project\BandStructure_AI_Project`

## Current State: v5 30k Accepted on NVIDIA GPU

项目主链仍为完整 E(k) → `(N,2,128,3)` 6D 张量 → MBM SSL → gap/type 微调 → tkinter Plot-to-Physics。最新已接受实验为 `aflow_noleak_v5_30k_seed42`；30k immutable raw、29,952 no-leak tensors、space-group-disjoint split、V100 GPU-only SSL/监督训练、回传哈希与本地 smoke 均已验收。`aflow_noleak_v4_seed42` 保留为 immutable 可复现基线。

### Latest formal result (v5)

- SSL best epoch 39；stopped epoch 54；best val masked MSE 0.384270，masked MAE 0.395004。
- Supervised inner best gap epoch 52；完整运行 60 epochs；inner best gap MAE 0.000341 eV。
- Outer line-mode gap MAE 0.000352 eV；RMSE 0.000595 eV；R² 1.000000。
- Outer model-vs-global-DFT gap MAE 0.640281 eV；不得与 line-mode 指标混称。
- Type accuracy 0.941186；Macro F1 0.932648；direct-gap recall 0.943662。
- 正式设备 Tesla V100-SXM2-16GB；GPU0 max utilization 89%，max memory 15,114 MiB；无 CPU fallback。
- Latest report：`artifacts/reports/aflow_noleak_v5_30k_seed42/latest_training_report_20260825.md`。

### Retained formal baseline (v4)

- Outer line-mode gap MAE 0.049747 eV；Type accuracy 0.865683；Macro F1 0.858339。
- 报告：`artifacts/reports/aflow_noleak_v4_seed42/latest_training_report_20260824.md`。
- 旧 6399/6443 target-conditioned 模型与报告仅为 legacy diagnostic，未进入当前 artifacts。

### v5 data acceptance

- Raw snapshot：30,000 HDF5 groups、30,000 JSON、0 temporary groups；HDF5 SHA-256 `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`。
- Raw JSON hash manifest：30,000 entries；SHA-256 `d90e6f9fd1289207ad00195f105bb34895186cbd522e1596669634084dc48625`。
- Metadata candidates：80,856；permanent exclusions：14,141；最终 resume success/no-data/failed：7,307/5,110/64。
- 有效 tensor：29,952；skipped 48；outer train/test：23,933/6,019；space groups：161/45；overlap=0。
- Full/split NPZ SHA-256：`9c5edea5e97f1c9b2561bc1c2324bcdd22741b5aea69e7d7c4c25116375a1ff7` / `c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`。
- Class counts train metal/direct/indirect：11,024/3,681/9,228；outer：2,768/923/2,328。
- v4 raw HDF5/metadata 已 byte-identical 恢复，v4 relocation tests 通过。

## Project Layout

- `data/raw/aflow/aflow_bands.h5`：byte-identical v4 6,443 baseline；`data/raw/aflow/snapshots/aflow_30000_20260825/`：immutable v5 raw snapshot。
- `data/processed/aflow/ood_tensors/`：v4 张量；`data/processed/aflow/ood_tensors_v5_30000_seed42/`：v5 已验收张量。
- `artifacts/{models,checkpoints,reports,logs}/aflow_noleak_v5_30k_seed42/`：latest accepted；v4 同名四类目录保留为 baseline。
- `src/{data,engine,models,utils,vision}/`：原有实现模块，不建立并列框架。
- `scripts/`：稳定入口；分类说明在 `scripts/README.md`。
- `PROJECT_BRAIN/`：宪法、上下文、迁移 manifest 与日期日志。
- 外层 `资料/`：参考论文/文档/图片，不属于运行根目录。

## Data Layer

### AFLOW

- v4 baseline HDF5：6,443 groups；SHA-256 `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`；metadata SHA-256 `daf662506f33fe990da97548afbbfb85c21add20566447651855c612608d2eff`。
- v5 snapshot：30,000 groups/JSON；HDF5 SHA-256 `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`；逐 JSON hash manifest 已保存。
- v4 baseline 与 v5 snapshot 物理分离；以后任何继续扩容都必须建立新 snapshot，不得原地追加 immutable 文件。
- 早期 smoke 中有 20 个不属于正式快照的唯一 ID；保存在 `data/raw/aflow/supplemental/`。
- 2 个 legacy HDF5 与 canonical 的同 ID group 内容不同，完整变体保存在 `data/raw/aflow/provenance/h5_variants/`。

### Future download/cache reliability contract (implemented 2026-08-26)

- `src/data/band_store.py` 现按稳定物理语义哈希处理同 ID：真实 v4/v5 legacy root/metadata 重叠 schema 可幂等读取；identity/provenance、`source_efermi_absolute`、缺失/`None` 被排除，group 实际 datasets/attrs 会重算而不盲信 stored hash。相同内容跳过，不同内容保留 canonical 并隔离为 variant。
- HDF5 commit/readback/metadata reconciliation 与完整 downloader run 均有跨进程 single-writer lock；WSL/POSIX 与原生 Windows `msvcrt` 均通过真实 child-process contention 探针。
- candidate catalog/cursor 与 canonical metadata sidecar 分离；metadata rich fold 不允许空值降级，历史 duplicate 与 incoming 非空冲突均写幂等 provenance audit；HDF5 不存在时 stale canonical IDs 也会清除。
- metadata canonical + conflict provenance 使用 write-ahead transaction；其它 JSON 使用 fsync temporary + same-directory replace，并在 POSIX 同步父目录。中断后 journal 可重放，不留下永久半提交状态。
- AFLUX cursor v2 固定 `page_size`、把 page size 纳入 query fingerprint，并逐 gap-bin 保存未消费页尾；candidate hard limit 不越界，duplicate page/真实耗尽分离，空 gap-bin quota 可转移。
- concurrent futures 每项完成后立即由协调线程处理；peer `KeyboardInterrupt/SystemExit` 不丢失同批已完成 exclusion/save/stats。MP batch 返回 cardinality 不一致即 fail-fast；`peak_in_flight <= workers` 且 HDF5 save 仍位于协调线程。
- report 主口径为 persisted canonical count，并包含 `target_reached`/`termination_reason`；target 未达抛非零异常。full pipeline 在 tensor/training 前复核 HDF5 count/report，并在结束前对未显式跳过阶段的 required artifacts 执行 fail-closed gate。
- 本轮只使用临时测试 cache，未对 v4/v5 immutable snapshots 执行 downloader、metadata 对账或任何写操作；Phase C 仍为 pending。

### Materials Project

- 已有 smoke 下载合并为 12 unique groups、12 JSON、3000 条 candidate metadata。
- 原下载报告保存在 `data/raw/materials_project/provenance/legacy_reports/`。
- API key 有效，但出口 IP/ASN 封禁问题尚未解除；不得通过换 key 规避。

## Tensor and Split Layer

- v4 tensor：`(6441,2,128,3)`；outer train/test 5,153/1,288；groups 146/37；overlap=0。
- v5 tensor：`(29952,2,128,3)`；outer train/test 23,933/6,019；groups 161/45；overlap=0；48 records skipped with reasons in manifest。
- AFLOW energy reference：canonical `E_F=0`；绝对 `Efermi` 仅作 provenance。
- Label-free occupied/empty edge envelopes + PCHIP。
- Segment-aware curvature、crossing、span masking 与 physics loss。
- NPZ 保存 `segment_ids_*` 与 symmetry labels。
- Manifest 保存 composition/prototype/source overlap、metal mismatch 与输入/输出 SHA-256。
- Full/split NPZ 与迁移前原始 hashes 字节级一致。

## Training Layer

### SSL / Phase B

已实现并正式训练：

- learnable mask token；
- 5–15 点 segment-aware span masking；
- masked-position MSE/MAE 与 actual mask fraction；
- segment-aware curvature/symmetry loss；
- warmup+cosine；
- global gradient clipping；
- best/last/resume checkpoint；
- early stopping。

### Supervised

- 冻结前 2/4 Transformer blocks；
- encoder effective LR 1e-5，heads LR 1e-3；
- class weights、极值期望 gap head、type head 和物理辅助损失；
- legacy metal anchor gate=False；
- outer test evaluation-only；
- Keras XLA JIT 显式关闭，避免 V100 cuDNN autotuner 不兼容；
- post-processing latent feature extraction 使用 batch_size=128，避免 29,952 样本一次性 attention OOM；
- `--evaluation-only` 可从 best checkpoint/history 恢复评估而不重跑 fit。

正式 v5 SSL 与 supervised 均在 Tesla V100 上执行。环境为 TF 2.21、CUDA runtime/NVCC 12.5.82、cuDNN 9.3.0.75；GPU forward/backward、训练 monitor 与 accepted model reload 均有磁盘证据，无 CPU fallback。

### Known metric boundaries

- `best_inner_val_gap_mae=0.000341 eV` 只用于 checkpoint 选择。
- outer line-mode tensor-gap MAE 为 0.000352 eV；model-vs-global-DFT gap MAE 为 0.640281 eV。
- direct-gap recall 0.943662；Macro F1 0.932648。
- composition overlap=1,474；当前只可称 space-group OOD。
- MC-dropout 尚未完成独立 calibration，不能直接作为 DFT acquisition 置信区间。

## GUI / Vision

- 原生 tkinter 是唯一 GUI。
- 人工坐标轴/Fermi/VBM/CBM 标定与物理模型调用路径保留。
- 最新 vision detector 权重在迁移前 artifacts 中不存在；自动检测不得声称已验收。
- `brain_invoker.py` 默认路径与 GUI t-SNE 已切换到 `aflow_noleak_v5_30k_seed42`，并有回归测试保证文件存在。

## Verification Evidence

- Final WSL compileall + pytest（2026-08-26 late-review hardening 后）：compileall exit 0；Stage-0 `74 passed in 21.40s`；完整回归 `92 passed in 217.63s`。
- `tests/smoke_latest_model.py`：实际加载 v5 split、norm、SSL `.keras` 和监督 weights；2 条 outer OOD 推理 gap/type 均有限、type probability sums=1，portable local paths 生效。
- Immutable readback：v4/v5 HDF5 groups=6,443/30,000、temporary groups=0；v5 full/split shapes 与 manifests 相等；四个 SHA-256 全匹配，`mismatches=[]`。
- Lock portability：4 个 WSL child-process contention 回归通过；原生 Windows msvcrt child-process probe 输出 `WINDOWS_CHILD_LOCK_REJECTED`、exit 0。
- 服务器 accepted-model reload：gap/type tensors 均在 `/GPU:0`，finite=True；formal status=`all/completed`。
- GPU monitor：145 samples；GPU0 max utilization 89%，max memory 15,114/16,384 MiB。
- Server transfer：30,069 archive members；server receive verification 逐文件 SHA mismatch=0。
- Returned results：63 archive members；62 content files 独立 hash validation 全通过；artifact manifest 50 files/75,143,804 bytes mismatch=0。
- Data：v4 6,443 baseline + immutable v5 30,000 snapshot；v5 tensor 29,952、outer 23,933/6,019、group overlap 0。
- `git diff --check` exit 0；仅有既有 Git LF→CRLF warning，无空白错误。

## Current Blockers

1. Phase C 尚未冻结 crystal structure schema、统一 k-path 和 multi-band target contract。
2. 当前结构数据不完整；AFLOW band cache 不能直接提供训练 crystal graph 所需的 lattice/species/fractional coordinates 全合同。
3. 只有 seed=42；3-seed 方差与 composition/prototype/source 多维 OOD 未完成。
4. uncertainty calibration 与 active-learning acquisition function 未验收。
5. MP 双源正式数据仍受网络封禁阻塞。

## Next Authorized Planning Target

下一阶段只做 Phase C 合同与最小数据闭环，不直接重写模型。执行顺序见：

`PROJECT_BRAIN/agent_logs/20260824_next_work_schedule.md`
