# 2026-08-26 — Downloader Durability and Provenance Hardening

Status: completed — late-review blockers fixed and reaccepted

## User approval

用户明确批准：“批准完整修复，并运行全量回归”。本轮只对现有下载/存储/pipeline 链路做原地治理修复，不建立并行框架。

## Trigger

2026-08-25 正式任务结束后，较早派出的只读审计延迟返回。其 GPU、候选上限、打包传输等结论已被后续真实执行取代，但当前源码仍存在以下未来运行风险：

1. 同 material ID 会删除并替换 canonical HDF5 group；
2. 没有同一 cache 的跨进程 writer lock；
3. exclusion、metadata 和 report 仅在正常结束后保存；
4. metadata 同 ID 整条 last-write-wins，候选目录与 canonical sidecar 混用；
5. report 缺少机器可读 target gate，target 未达仍可正常退出；
6. run stats 跨同一 downloader 实例调用累计；
7. 本地候选未优先复用，分页 cursor/补池状态未持久化；
8. pipeline 未在 downloader 之后独立读取 count/report 做防御性门禁。

## Frozen assets — pre-change readback

本轮禁止修改或重建下列 accepted assets：

- v4 raw HDF5：6,443 groups，0 temporary groups，SHA-256 `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`；
- v5 raw HDF5：30,000 groups，0 temporary groups，SHA-256 `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`；
- v5 full tensor：SHA-256 `9c5edea5e97f1c9b2561bc1c2324bcdd22741b5aea69e7d7c4c25116375a1ff7`；
- v5 split tensor：train `(23933,2,128,3)`、test `(6019,2,128,3)`、group overlap=0，SHA-256 `c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`；
- latest accepted experiment：`aflow_noleak_v5_30k_seed42`。

Git pre-state：branch `master`，HEAD `1ce9aa66847ca6ee63f004666a24125fc82bd6f9`，工作树已包含此前任务的大量未提交变更；本轮不得覆盖或回滚它们。

## Approved in-place implementation order

1. `src/data/band_store.py`：semantic/content hashing、idempotent skip、variant isolation、cross-process lock；
2. `src/data/batch_download.py`：run-local stats、atomic checkpoints、interrupt recovery、target gate、candidate catalog/cursor；
3. `src/data/aflow_adapter.py` / `src/data/mp_adapter.py`：仅调整现有 adapter 的返回/metadata 接口；
4. `scripts/run_full_pipeline.py`：下载后 count/report 双重门禁；
5. `tests/test_stage0_robustness.py`：每项行为严格 RED→GREEN；
6. `README.md`、`PROJECT_BRAIN/dev_context.md`、`PROJECT_BRAIN/CONSTITUTION.md` 与本日志同步；
7. WSL compileall、完整 pytest、latest model smoke、immutable hash readback、Git diff/line-ending audit。

## Non-goals

- 不访问 AFLOW/MP 网络，不下载新数据；
- 不写入 v4/v5 raw/processed snapshots；
- 不重新训练或覆盖任何模型、checkpoint、metrics、report；
- 不开始 Phase C；
- 不清理用户已有未提交工作树。

## Execution evidence

### 2026-08-26 correction after late review

在首次 `78 passed` 验收后，先前派出的两项只读独立审查延迟返回，并指出当前测试未覆盖的真实 schema/cursor/interrupt/final-artifact 边界。源码核对确认相关路径仍存在，因此此前“completed”状态作废，重新进入 RED→GREEN：

- 真实 legacy HDF5 的 metadata/root 重叠键可能导致 semantic hash 不兼容；
- provenance-only 与 `None`/缺失字段仍可能产生伪 variant；
- metadata audit/reconciliation 仍有幂等与跨文件 checkpoint 缺口；
- AFLOW 非整页 quota 恢复可能跳过页尾记录；
- workers>1 时单个 future 的 `BaseException` 会丢弃同批已完成结果；
- pipeline 最终 required artifacts 缺失仍只 WARN。

以下首次验收证据保留为历史中间结果，不再代表最终工作树。只有这些新增 RED 全部修复并重跑完整验收后，才能再次标记 completed。

### RED → GREEN implementation

1. `BandStore`
   - semantic SHA-256 排除易变下载 provenance/metadata 文本，保留 E(k)、k-path 与能量参考等物理内容；legacy HDF5 JSON attrs 先解析再 hash，空格/键序不产生伪冲突；
   - same ID + same physics 返回 `duplicate`，不重写 canonical；
   - same ID + different physics 返回 `variant`，冲突记录写入 `provenance/h5_variants/<cache>_variants.h5`；
   - HDF5 commit、ID readback 与 metadata reconciliation 使用同一非阻塞跨进程 advisory lock；完整 downloader run 另持有 source cache lock。
2. Metadata/provenance
   - rich-field merge：新空值不覆盖旧非空值，非空冲突保留 canonical 并写 metadata variant audit；
   - `*_candidate_catalog.json` / `*_candidate_cursor.json` 与 canonical `*_metadata.json` 分离；
   - 启动对账使 canonical metadata material IDs 与 HDF5 IDs 完全相等；adapter 透明返回真实 commit status。
3. Resume/progress
   - exclusions、cursor、canonical metadata 和 report 原子写入；每批 checkpoint，`KeyboardInterrupt/SystemExit` checkpoint 后重新抛出；
   - 每次调用重置 run-local stats；同一实例 target 3→5 只新增 2；
   - 进度主口径为 `persisted/target`；report 包含 attempted/submitted/completed/in-flight/peak/target/termination 字段；
   - target 未达时先保存 report，再抛 `DownloadTargetNotReached` 形成非零 CLI 退出。
4. Candidate paging
   - 本地 catalog 优先于 metadata API；generic page cursor 与 AFLOW exact per-gap-bin cursor 均按 query fingerprint 跨实例续传；
   - duplicate metadata page 与真正 API exhausted 分开处理；
   - AFLOW 固定 gap-bin quota 增加 deficit reallocation；
   - payload future 数不超过 `workers`，HDF5 save 仍只在协调线程。
5. Pipeline
   - downloader subprocess 原有 `check=True` 保留；
   - 增加 HDF5 persisted count + 本次 report 双门禁，`target_reached=false`、target/total 不一致或 report 缺失时在 tensor stage 前停止。

### Late-review RED → GREEN completion

延迟独立审查指出的路径均先构造最小失败测试，再做原地修复；Stage-0 test 数从 60 增至 74（14 条新增回归，另修正真实 legacy fixture）：

1. semantic hash：真实 v4/v5 root/metadata 重叠 schema、absolute source Efermi provenance、`None`/缺失等价、stale stored hash 回验；
2. metadata：冲突重放幂等、历史 duplicate rich fold/conflict audit、HDF5 absent stale-ID 清理、canonical+variant write-ahead transaction 与中断恢复；
3. AFLUX：cursor v2 固定 page size、逐 bin 保存未消费页尾、page size query identity、candidate hard limit；
4. downloader：`as_completed` outcome 立即由协调线程处理，peer interrupt 不丢先完成结果；MP batch cardinality mismatch fail-fast；
5. pipeline：stage 零退出但 required artifact 缺失时最终 fail-closed；显式 `--skip-vision` 只豁免 vision 两项；
6. lock tests：四处同进程/fcntl-only 夹具替换为真实 child-process contention；另用原生 Windows Python 验证 `msvcrt` 分支。

### Current verification

- 定向 `compileall`：exit 0；
- 完整 Stage-0 robustness：`74 passed in 21.40s`；
- cursor/paging 组合：`10 passed, 61 deselected`；真实 WSL child-process lock 回归：`4 passed`；
- 所有测试 cache 均位于临时目录；没有调用真实 AFLOW/MP 网络；
- v4/v5 immutable 数据与 accepted artifacts 未作为任何测试输出目标。

### Final acceptance

- WSL `python -m compileall -q src scripts tests`：exit 0；
- WSL 完整 `python -m pytest -q`：`92 passed in 217.63s (0:03:37)`；
- `CUDA_VISIBLE_DEVICES='' python tests/smoke_latest_model.py`：exit 0；实际加载 `aflow_noleak_v5_30k_seed42` split/norm/SSL/supervised artifacts，2 条推理 finite，type probability sums 均为 1，device=`CPU:0`；显式隐藏 GPU 后的 `CUDA_ERROR_NO_DEVICE` 初始化日志为预期，不是 fallback failure；
- 原生 Windows child-process lock probe：输出 `WINDOWS_CHILD_LOCK_REJECTED`，exit 0；未向错误的 Windows Python 安装 h5py，探针仅 stub 未调用的 h5py import，被测锁函数为项目真实实现；
- 修改后 immutable readback：
  - v4 HDF5：6,443 groups、0 tmp，SHA-256 `bb261f1e3b3f602e3463ca32831e15b67f53bd5f9f2d951614ec1441b8c0b62d`；
  - v5 HDF5：30,000 groups、0 tmp，SHA-256 `d9927f0425de6232a24b8cea2eb8d0c5820e0aa9e29222b7feb621ebc7be08f3`；
  - v5 full NPZ：`X=(29952,2,128,3)`，SHA-256 `9c5edea5e97f1c9b2561bc1c2324bcdd22741b5aea69e7d7c4c25116375a1ff7`；
  - v5 split NPZ：train/test `23933/6019`，SHA-256 `c99d21647489bec3c4a20cafd209ef136b83da966af67e0594f4dc5d0aa5b7a5`；
- verifier 结果 `mismatches=[]`，所有 group/tmp/shape checks=true；四个 SHA-256 与修改前冻结基线逐字一致；本轮未重建、下载或训练 v4/v5；
- `git diff --check`：exit 0；仅输出仓库既有 LF→CRLF working-copy 提示，无空白错误。

### Post-acceptance stale background notification reconciliation

- 最终答复后收到旧后台进程 `proc_ac1abf9c2b91` 的延迟失败通知；该进程输出为 `1 failed, 74 passed`，总计仅收集 75 项，明显早于最终工作树的 92 项套件，且失败行为正是 cursor v2 GREEN 前的中间态。
- 该进程通知到达时已不在 process registry，无法再读取启动时间；因此没有仅凭进程身份推断，而是在当前冻结工作树重新执行失败节点：`test_aflow_metadata_resumes_each_gap_bin_from_its_exact_cursor` 为 `1 passed in 0.76s`。
- 随后不修改任何 source/test 文件，串行重跑通知中的完全相同命令 `python -m compileall -q src scripts tests && python -m pytest -q`：exit 0，`92 passed in 212.68s (0:03:32)`。
- 因此该通知属于修复过程中启动、完成后才送达的中间态 run，不代表最终工作树；当前验收仍为通过。
