# 2026-09-06 P0 科研基准重构 — 完成日志

## 目标（用户 P0 指令）

- 不再把 line-mode gap MAE 作为主结果；
- 标签拆为三任务：line_mode_topology / provider_global_electronic_type / line_global_disagreement；
- 3 seeds + group bootstrap + 错误分层；
- 固定七类拆分（random / space-group / composition / prototype / leave-element / source-protocol / temporal）；
- README 清理叠加历史状态；分支改名对齐 v7。

## 执行摘要

- P0A 三任务标签派生（冻结张量后处理）+ 模型三独立分类 head；
- P0B AFLUX 元数据回填 prototype/species/species_pp/aflowlib_date（58,250/60,000；year/experiment AFLUX 不支持输出）；
- P0C 七拆分基准套件（seed 42、group-disjoint、manifest 落盘）；
- P0D group bootstrap（spacegroup 级重采样 95% CI）+ 错误分层四轴 + 七拆分冻结评估器 + 3-seed 汇总器；
- P0E README 折叠、宪法 4.13、dev_context 同步、分支改名 `v7-60k-20260903`；
- 全量回归 199 passed（新增 29 个 P0 测试）。

## 3-seed 训练（服务器 V100）

- 每 seed：SSL（seed 42 复用 v7 编码器；2024/7 全量 SSL 60ep）+ 监督 60ep train-only + evaluation-only + 七拆分冻结评估；
- 主结果（outer OOD 11,987，space-group canonical 拆分）：
  - line_mode_topology：0.9932±0.0016（3 seeds）
  - provider_global_electronic_type：0.9371±0.0062
  - line_global_disagreement：0.9666±0.0016
- 七拆分跨域评估全部 >0.94（source_protocol 仅 540 test，宽 CI，如实报告）。

## 关键坑（已修复并测试锁定）

1. AFLUX 附加字段请求语法 `prototype()/species()/species_pp()/aflowlib_date()`；`year()/experiment()` 不支持输出（空响应）。
2. 回填脚本 URL 缺 `?`（`aflux/` → `aflux/?`）导致三次后台 404 失败——回归测试 `test_api_root_includes_query_separator` 锁定。
3. `aflowlib_date` 是列表（取首元素为入库时间）；年份分布双峰，temporal 拆分按完整时间戳分位数而非按年。
4. aurl 目录分布极端偏斜（ICSD_WEB/LIB3_WEB 占绝大多数），source_protocol 拆分 test 仅 540。
5. 回填合并产生 44,683 个新 ID（catalog 全量），须裁剪回 60,000 HDF5 ID。
6. 服务器传输：scp 需交互密码、git push 卡 GitHub 凭据弹窗、pinggy 认证被拒、跨网段 80 端口不通——最终用 litterbox 中转（双向 MD5 校验一致）。
7. v7 冻结权重缺新 head 变量：smoke 测试 `skip_mismatch=True` 兼容加载。

## 产物

- `artifacts/reports/aflow_noleak_v7_60k_seed{42,2024,7}/`：metrics_summary.json（primary_results.three_task_benchmark）、seven_split_evaluation.json；
- `artifacts/reports/aflow_noleak_v7_60k_seed42/three_seed_aggregate.json`、`P0_scientific_reframing_report.md`；
- 回传归档 SHA-256 `1c01a7fb8925a6ff2596eea3d8ee4ee4`。

## 待办

- GitHub 远程推送分支改名（本地完成；远程待凭据解除）。
