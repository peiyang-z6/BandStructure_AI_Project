# P3 受控对照最终结果与归档

日期：2026-09-09（UTC）。状态：**受控执行完成；P2/P3科学阶段未验收**。

## 授权与范围

延续用户已批准的P3限域方案和后续“请继续”。本轮仅恢复观测通道、核验已完成训练/评估、回传及记录结果，没有重启训练、修改超参数/生产源码、重建immutable张量或变更主方向。新资料只进入既有data/processed、artifacts和PROJECT_BRAIN边界。未提升GUI或latest-accepted指针，未启动P4/P5。

## 结果

- 新train原始47,912、有效47,879；33条分段不足两个源k点。
- 新outer原始11,987、有效11,936；排除51条：39缺结构、7分段不足、5 k距离质量不合格。旧11,948等旧pipeline计数保留为历史，不能混用于新版。
- 相同inner split、seed42、batch32、上限180轮、patience20；MLP stop72/best52，attention stop65/best45。双方冻结后才做本轮outer评价；不是等参数量/等实际算力试验，也不是未接触过的全新盲测。
- 逐k谱OT MAE：MLP4.083630、attention3.951537 eV；共同可解析11,893条gap MAE：0.794580/0.899375 eV；可解析目标零隙的假gap率31.0217%/41.4431%；各自unknown11/32。
- 配对空间群bootstrap（55组、2,000次、seed42）：谱差值attention−MLP的95%区间[-0.463505,0.476848]跨零；共同gap差值区间[0.036869,0.152813]。单训练seed、无多重比较调整，仅冻结后描述性证据。
- 独立只读复核 `deleg_cc435c96` 不支持“attention全面更好/P3验收”。其未读取NPZ/权重；父代理后续实际验证了共同集合、逐类计数、假gap分子分母、unknown原因及分层，不把审核者的聚合反推包装成逐样本验证。

## 接收与真实运行核验

- 45份新派生数据/权重三态/checkpoint/预测/日志/凭据，合计3,319,777,019 bytes；双端逐文件SHA一致，三immutable输入和运行代码一致，无覆盖历史文件。
- 全量11,936保存预测与目标/身份/mask/segment/k轴对齐，生产指标逐项重算一致。
- 最新本地全量回归456 passed、0 failures/errors/skipped；服务器冻结代码此前全量也为456 passed。未跳过测试或改为CPU正式训练。
- 每模型预先选定64真实outer行：原V100按原批次重放保存预测，差均0；各环境内独立构建/重载差均0。
- 本机RTX4060真实forward/loss及25/57个gradients均在GPU且finite；last/optimizer完整恢复，epoch72/65、iterations91,584/82,680，与last.weights一致；没有优化器更新，冻结文件未改。
- **保留两次失败**：RTX batch8对V100 batch32的严格逐点比较失败；对齐原batch32后仍不满足rtol=atol=1e−5（最大差MLP0.000900269、attention0.000114202 eV）。后续诊断将同机重载/恢复与跨环境等价分开记录，未放宽门限或把失败改成通过。具体GPU内核/确定性设置逐算子归因未完成。
- 原绝对路径凭据不改写，本地只作内存路径映射；不是CLI可迁移或CLI resume验收，后者仍未实现。

## 冻结结论与后续边界

per-k谱OT不是trajectory Hungarian或band身份/连续性验收；16槽及给定mask不是任意带数自动推断；5eV只是选带条件；scalar k不是真实3D倒空间。Bandformer同数据对照、P3七拆分/多seed、完整物理/结构规模/provider/source分层仍缺。继续保留两组结果，不按outer重新选权重或宣称已满足完整P3合同。

下一步仅在后续方案中研究金属假gap、16槽子集退化、trajectory/物理指标及跨环境数值可复现性；本轮不自动追加训练。P2检索也仍需科学验收。

## 证据与同步边界

最终报告与全部详细证据位于 `artifacts/reports/p3_audit_20260908/controlled_final_20260909/`，主报告 `P3_controlled_final_report.md`；机器清单为 `PROJECT_BRAIN/transfer_manifests/p3_controlled_final_20260909.json`。

服务器保留训练时源码/治理快照；最终报告、本机验收证据及更新治理文档以独立快照镜像，不覆盖旧部署凭据。传输回验/临时资源清理以该证据目录中的独立收据为准；未验证的状态不在此预先宣称完成。
