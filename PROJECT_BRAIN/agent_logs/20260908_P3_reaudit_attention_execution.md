# 2026-09-08 P3 继续推进：先纠正数据/评估，再比较 k 自注意力

## 授权与状态

用户已批准“按 3 → 1 → 2 执行”，本轮要求“请继续”。本轮保持结构—能带—图像跨模态主方向、既有 6D/MBM/三任务/tkinter 主链和 immutable v4–v7 资产。只原地升级 P3 提取、指标、decoder 和训练入口。

P2 仍未验收。用户先行推进 P3 的授权仅覆盖探索性实现与对照训练，不等于 P2 通过，也不等于 P3 正式验收；不自动解除 P4/P5 门禁。宪法本轮记录此限域例外，保留其余数据、GPU、内外层隔离和验收规则。

**本日志当前为进行中的工程/科学纠错记录，不是训练验收报告。** 后续结果必须追加实际产物、SHA 与命令输出。

## 已核实的旧训练

- 本地起点 commit：`5977c49`，分支 `v7-60k-20260903`，开始审计时工作区干净。
- 本地与服务器的 `multiband_decoder.py`、`multiband_metrics.py`、`multiband.py`、`train_p3_decoder.py` 在修改前 SHA-256 完全相同。
- 排序 OT 版训练确已完成 60 epochs：

| 旧报告字段 | outer-train | outer-test |
|---|---:|---:|
| band_mae（固定 slot） | 8.643047904861563 | 5.428568070080634 |
| sorted_band_mae | 2.7861583315234917 | 3.708433331214173 |
| gap_mae（旧实现，不作为有效科学结论） | 0.9345557689666748 | 1.342875599861145 |

首 epoch loss 5.004480257530875，最后 epoch 2.834154849695427。历史 loss 与模型架构/数据合同均未满足本轮新协议，不能从不同 loss 的数字直接推出改进，也不与异数据集 Bandformer 论文值作倍数比较。

服务器旧模型、报告、对应代码、预处理报告与训练日志只读打包回传。本轮未删除旧服务器产物。归档保存为 `artifacts/reports/p3_audit_20260908/legacy.tar.gz`，双端确认 SHA-256：

`0409813e03a3799ad7ec4059243748b71ec0eea9bb4bee2743ff48f090bd2b88`

逐成员哈希、旧指标与状态标注见同目录 `legacy_inventory.json`。前轮曾删除最早 masked-MAE run 的原输出目录；本轮不假定其权重仍可恢复，也不虚构重新评估。

## 复现并纠正的问题

1. **padding 改写 VBM**：旧代码将 valence sentinel 乘 false mask，得到 0；负 VBM 被错误抬到 0。回归反例真实 gap=3.0，旧函数返回 2.5。
2. **crossing 被误判为绝缘体**：deep occupied + high empty 不排除同时存在部分占据 band。`[-2,-2] / [-1,+1] / [3,3]` 被旧函数错误标为绝缘体。
3. **选带丢掉真正的费米跨越轨迹**：同一条件错误进入绝缘体双侧选择，从而跳过 crossing band。仅从 outer-train 用 seed 42 抽查 1,000 个原始材料：787 个含全路径 EF-straddling band，其中 329 个被旧选择器漏掉全部 straddling band。此统计是抽样、全路径能量区间诊断，不能称为全量金属比例或严格 segment 内 crossing 比例。
4. **EF 区间距离错误**：`min(abs(band_min-EF),abs(band_max-EF))` 不等于 EF 到能量区间的距离；宽跨越 band 会被近 EF 的非跨越 band 挤出。现在跨越优先，区间内距离为零，tie 按原始 index。
5. **错误拼接不同 k 分支**：旧 PCHIP helper 在重复累计距离处平均能量，再跨边界插值。反例 `[-2,-2,3,3]`、k=`[0,.5,.5,1]` 被插出不存在的中间能量。P3 改为按重复距离分段后独立插值；不改冻结 6D builder。
6. **缺边缘不等于金属**：只见到 occupied 或只见到 empty 时标 unknown，不能把缺失证据直接编码为 zero gap。EF 恰为 0 的 band 只属于 occupied，不能同时充当 CBM。
7. **gap 评估选择偏差**：真实金属预测出正 gap 是实际误差。剔除金属只改变评估群体，不是“修复误差膨胀”。新报告分 all-resolved / target-insulator / target-metal，并显式给出 unknown 和 coverage。预测 gap 必须从预测本身推导。
8. **排序 loss 静默吞掉 NaN/Inf**：有效预测 NaN 或误差溢出曾被置零。现在有效数据与误差非有限时失败；仅 padding 排除。
9. **Keras 多输入子类的重载不等价**：独立回归在 0 层和 2 层 decoder 均复现保存后重载预测变化；补全 `get_build_config/build_from_config` 后两者实际预测一致，不以 save() 成功代替验证。

以上修正本轮均用明确失败的回归反例后再修改对应生产路径。不能将前轮“先写实现后补测试”的历史追认为严格 TDD。

## 当前实施设计

### 修正版数据（独立版本）

- 新目录：`data/processed/aflow/ood_tensors_v7_60000_seed42/p3_multiband_v2_20260908/`。
- 保持 max_bands=16、n_k=256、delta_e=5 eV；保留完整被选 band，不裁剪真实色散。
- 复用 immutable 外层 split 的 ID 与 group；新增可追溯 selected_band_indices、segment_ids、明确 exclusion 原因与完整输入/代码/输出 SHA。
- 使用 Unicode IDs，不在新 NPZ 中创建 object/pickle ID 字段。spin 数据保留所有通道的 flattened index。
- 任何已有目标派生文件都禁止隐式覆盖。失败或零 valid 不得包装为完成。

### 同入口对照实验

- `MultiBandDecoder(attention_layers=0)`：MLP 对照；`attention_layers=2, attention_heads=4`：k 自注意力实验。
- 自注意力在 `(B,K,D)` 上做 pre-norm MHA + FFN，然后加 band slot embedding；避免 `(B*bands,K,D)` 的显存放大。
- 若有 segment IDs，attention mask 禁止跨 segment；实际梯度回归验证同段其他 k 对输出有影响、异段无影响。
- 两版使用相同修正版数据、seed 42、同一 group-disjoint inner fit/val、相同排序 OT loss 与训练预算。最多 180 epochs、patience 20、warmup 5 epochs + cosine、gradient clipnorm=1。预算/默认值不依据 outer test 调整。
- train-only 不读 outer NPZ；全部 inner-val 按有效 band×k 元素加权，独占 checkpoint/early stopping。保存 best/last/accepted、配置、数据/代码哈希、history、恢复信息与冻结 selection manifest。
- evaluation-only 在确认选择清单后才加载 outer 数据，保存真实预测与 ID；模型重载预测一致性和 GPU critical forward/backward 检查先于长训练。

## 明确保留的边界

- 逐 k 排序是等基数 1D **能量谱 OT**；并非整条轨迹 Hungarian、波函数 band 身份跟踪或 continuity loss。
- mask 表达至多 16 条的外给有效数量，不是推理时自动预测任意 band 数。
- scalar [0,1] k 是相对路径位置；不是三维倒空间 query。完整 reciprocal-space conditioning、物理简并/时间反演/曲率/有效质量、Bandformer 同数据正式对照仍未完成。
- 旧 outer test 已在先前迭代查看，不将其称为全新 blind external test。本轮选模仅用 inner-val；全新外部证据仍留给 P5。
- P2/P3 均不能因单次 smoke 或代码测试通过自动标为科学验收。

## 验证与产物（待本轮完成后追加）

已经有旧 run 双端哈希归档和逐条 RED→GREEN 输出。

本轮已完成的进一步实测：
- 父代理三组定向回归：`44 passed in 10.50s`（multiband、multiband_metrics、multiband_decoder）。这不是全量项目测试计数。
- 真实 outer-train 材料 `aflow-00fc6d45577f6562`，d_model=128、2 层/4 头 k attention，在本地 RTX 4060 运行 5 次编译 forward/backward；预测、loss、全部梯度及关键 MatMul/attention/sort 图算子均验证 GPU placement，finite 检查通过。
- 实际保存/重载后预测最大绝对差 `0.0`；模型 SHA `52a072a3502ff87c4abce3292a4bcd151fdeb7c4ebebbd7f535fe36c1c815249`。证据：`artifacts/reports/p3_audit_20260908/gpu_smoke/smoke_report.json`。
- 此 smoke 将同一个晶体重复成 batch=2，只验证可运行性/设备/重载，不作为收敛或泛化精度证据，也不能代替服务器 V100 端到端验证。
- 部署前重新计算本地/服务器三个 immutable 输入（60k 原始 HDF5、结构 sidecar、v7 outer split NPZ）的 SHA-256 与字节数，三项全部一致。证据：同审计目录下 `local_inputs_inventory.json`、`remote_predeploy_inventory.json` 与 `input_comparison.json`。服务器代码更新前清单共 12 项；这不等于新代码已部署，也不等于修正版数据已生成。

仍待完成：训练/数据入口并行结果集成、独立代码审查、完整 pytest、修正版全量计数、双端部署哈希和受控训练。

### 独立审查第一轮：未通过

独立 reviewer 未发现安全问题，但确认一个新增 EF 边界回归。父代理已在 WSL 复现：能带 `[-1,-1] / [0,1] / [2,2]` 的 `derive_gap` 错误返回 3 eV；容量 2 的选择器返回 `[0,2]`，把最低点接触 EF、随后向上色散的 `[0,1]` 漏掉。按当前窗口占据约定，该例应保留 `[0,1]` 两个 band 索引、返回 1 eV。完全平坦于 EF 的带与仅最低点接触 EF 的带必须区分，前文“strict >”规则不能直接推广到所有 EF 接触情况。

另一个非新增回归的数值风险：逐元素 OT 误差有限，最终 reduce_sum 仍可能溢出；最终 loss 也必须有限或明确失败。已交由第三方修复代理执行限定范围 RED→GREEN，随后重新独立审查。本轮复审通过前，不生成最终修正版数据、不部署或启动新长训。

### 第一轮修复后的集成验证

- 第三方完成 EF 接触边界及最终 loss 检查的限域修复。选择器和指标统一 `(min>=EF)&(max>EF)`；零隙双侧接触不标为绝缘体，孤立 EF 平带仍 unknown；eager/编译模式的最终归约溢出明确失败。
- 父代理在 WSL、CUDA 禁用条件下重新执行完整 `tests/`：**`387 passed in 70.34s`**。证据：`artifacts/reports/p3_audit_20260908/full_regression_after_core_fix.xml`。这不是 GPU 验证或科学精度验收。
- 两个入口分别由父代理复测：prepare **43 passed in 3.84s**；train **39 passed in 38.59s**，包括独立 CLI train/evaluate 与权重/优化器恢复。训练保存恢复状态，但没有 CLI resume；非空 run 目录拒绝覆盖。
- 训练侧 47,912 个 material ID 的原子数审计最大值为 50，超过默认容量 50 的数量为 0。此项只核对容量，不替代全量图构建质量检查；证据：`train_structure_capacity.json`。
- 核心六文件复审与两个入口的独立审查仍在进行。修复后真实 GPU 端到端 smoke 已启动，输入仅从 canonical outer-train 抽取，另划 group-disjoint smoke train/holdout；没有访问 canonical outer-test arrays，不能报告为 outer OOD 精度。新证据独立保存至 `gpu_entry_smoke_after_fix/`，不覆盖先前 smoke。

### 本地 GPU 端到端结果

该 smoke 已正常退出 0。真实结构/HDF5 准备有效样本为 smoke train 64、smoke holdout 16，全部来自 canonical outer-train；这不是正式 outer 评估。0 层 MLP 和 2 层 attention 均完成独立 CLI train→冻结→evaluate：预测、loss、全部梯度实测在 GPU；优化器/epoch 状态恢复通过；两版权重重载最大预测差均为 0.0；父代理读回 best/last/accepted 文件，SHA 和字节数全部匹配。验证摘要：`artifacts/reports/p3_audit_20260908/gpu_entry_smoke_verified.json`。本结果仍仅代表本地可运行性，不能替代 V100 端到端验证、独立审查或正式精度验收。

### 独立审查结果与入口门禁补齐

核心六文件复审已经通过；父代理重新计算六个文件 SHA，与冻结审查快照全部一致。见 `core_review_round2_verdict.json`。

prepare/train 入口首轮审查未通过，新增两个明确阻塞：
1. 训练/评估没有验证 prepare 的完成提交，也不识别输入的 smoke_only；缺报告或失败准备留下的 NPZ 仍可被加载，冒烟数据可能被命令参数标为 formal。
2. evaluation-only 未把 outer ID/groups 与冻结 fit+validation 池核对互斥；既有 synthetic 正常测试甚至使用了重叠空间群。父代理在仓库外复现：没有准备报告仍加载 10 个 synthetic 样本，训练与模拟 outer 的 group 交集为 `[1,2,3]`。未执行训练，也未据此认定 canonical 固定快照发生泄漏。

已委派第三方严格 TDD 限域修复：每 split 完成凭据及最终提交保护、正式模式拒绝冒烟输入、冻结后 outer ID/group 零交集门禁，并补原始样本轴长度检查。训练不得为校验而读取含 outer 统计的全局报告。修复范围仅两入口和两个测试；不重开已通过的核心物理逻辑，不新增 CLI resume。上述 387 项和 GPU 冒烟是补门禁前代码的证据，入口修复后必须重跑并再次独立审查；当前仍不部署、不生成正式全量派生物、不启动长训。

### 入口门禁修复后的集成验证

- 第三方已返回四文件限域修复；父代理重新读取两入口并在 WSL、CUDA 禁用条件下运行完整 `tests/`：**`456 passed in 112.03s`**。证据：`artifacts/reports/p3_audit_20260908/full_regression_after_entry_gates.xml`。这是本轮入口修改后的结果，不沿用旧 387 项。
- 固定目录新增 `p3_train_complete.json` / `p3_test_complete.json`（`p3_split_completion` v1）和最后发布的中性 `p3_prepare_commit.json`（`p3_prepare_commit` v1）。缺失/失败/篡改提交在 NPZ load 前拒绝；prepare `--limit` 输入要求下游显式 `--smoke-limit > 0`。CLI 无新增参数，旧无凭据 NPZ 只能重新 prepare 到新目录，不能手工补凭据冒充新数据。
- 训练只验证 train 凭据；冻结后 evaluate 在推理前核对原始 outer 全部行与冻结 fit+validation 的 ID/group 零交集，包含 invalid 行及 smoke 前缀外行。实际消费数组在 valid 索引前检查原始样本轴长度。
- 已冻结四文件完整 diff 和 SHA 静态记录，送第二轮独立复审：`entry_review_round2.diff`、`entry_review_round2_static.json`。静态扫描无 literal secret/shell execution/eval；`allow_pickle=True` 仅限已注明可信边界的历史本地冻结源 NPZ，新 P3 输入为 `allow_pickle=False`。四文件 patch 已用 `git apply --numstat` 完整解析，不使用截断工具输出作为审查依据。
- 新版真实晶体 GPU 端到端 smoke 已退出 0，独立目录 `gpu_entry_smoke_after_entry_gates/` 未覆盖旧产物。prepare 与 downstream 均显式 smoke：有效 train 64、holdout 16，均来自 canonical outer-train，未访问 canonical outer-test arrays。
- MLP / 两层 attention 均完成 GPU train→freeze→evaluate；prediction、loss、gradients 实测位于 GPU，优化器恢复通过，真实重载预测差均为 0.0。父代理读回两版 best/last/accepted 及 predictions 文件并重新计算 SHA/bytes，全部匹配；新完成凭据、入口四文件审查 SHA、历史核心六文件审查 SHA 也全部匹配。模拟 holdout 与冻结训练池的 ID/group overlap 均为 0。摘要：`gpu_entry_smoke_after_entry_gates_verified.json`；这是本地可运行性验证，不是精度或服务器 V100 验证。
- 入口第二轮独立复审已通过（151 项定向测试、零阻塞）；父代理再次计算当前四文件及核心六文件 SHA/bytes，均与冻结审查快照一致，见 `entry_review_round2_verdict.json`。本地工程门禁齐备，开始仅代码／测试／治理文档的备份式部署；immutable 原始输入不重传，旧数据和模型不覆盖。服务器 V100 端到端尚待执行，尚未启动全量新派生物及受控长训。P2/P3 科学验收状态不变。
