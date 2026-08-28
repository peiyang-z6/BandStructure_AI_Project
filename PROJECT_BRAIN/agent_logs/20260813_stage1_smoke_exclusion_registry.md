# 2026-08-13 Stage 1: Data-layer fixes, WSL smoke, MP block discovery

## Scope

在不改变核心 4 步与 Phase 5 方向的前提下，修复数据层已知问题，在 WSL2 conda
(`bandstructure-ai`, Python 3.11.15, TF 2.21.0, RTX 4060) 完成全链路冒烟。

## Confirmed Problems (fixed)

1. **断点续传重复抓取已知无效记录**：`downloaded_ids` 只记录成功项，AFLOW 上游空
   `bands_data` 记录每次续传都会被重新下载（stage0 曾单次隔离 7 条）。
   修复：新增 `{source}_excluded.json` 持久化排除注册表（原子写入），候选过滤与
   摘要统计均接入；文件可人工编辑以便上游修复后重试。
2. **AFLOW `Egap_type` 缺失时被静默标为 indirect**：`"direct" in ""` 为 False 导致
   `is_direct=False`，相当于伪造 provider 标签，违反宪法 §4。
   修复：`gap_type` 为空时 `is_direct=None`，回退到极值推断并在 manifest 标注来源。
3. **CLI 未暴露分层数**：`--gap-bins` 直通 AFLOW `fetch_metadata` 的 gap_bins，
   支持 Stage 1 的宽带隙区间分层采样。

## Verification Evidence

- WSL `pytest -q`: 18 passed（含 2 条新增回归：排除注册表续传行为、空 Egap_type 不伪造标签）。
- `python -m compileall src scripts tests`：全部通过。
- AFLOW 真实下载（隔离目录 `data_cache/stage1_smoke/`，0–5 eV、5 层、natoms≤50）：
  15 条成功 + 4 条 no_data 隔离；二次运行 target=16 时 4 条排除记录零重试，新增 1 条。
- 张量构建：`(16, 2, 128, 3)`，15 个空间群，train/test=13/3，交集 0；
  类别分布 train {metal:1, direct:4, indirect:8}，metal 类首次进入冒烟集；
  所有分类标签均来自 provider 字段。
- GPU `--require-gpu` 1 epoch MBM：train=1.13768 val=0.80772，best checkpoint
  恢复与 `.keras` 导出正常；新进程重载后 projection/reconstruction 前向通过。

## External Blocker (not a code bug)

- Materials Project API 返回 403：当前出口 IP/ASN 被 MP 以"inefficient or abusive
  traffic"为由临时封禁。下载器按设计优雅降级（0 候选、明确 ERROR 日志、写报告）。
  需要用户按 MP 提示向 support@materialsproject.org 申诉（附公网 IP），或更换网络。
- figshare API 从本机同样返回 403，提示该出口 IP 可能被多个材料数据服务列入黑名单；
  AFLOW (aflow.org / aflowlib.duke.edu) 不受影响。

## Deferred

- 第三数据源（JARVIS-DFT）：bulk 数据托管于 figshare，当前网络不可达；且接入需
  宪法 §3 更新。决定：Stage 1 数据扩量走 AFLOW 单源，MP 解封后再做双源对照。
- 冒烟数据仅证明链路可运行，不代表模型精度（同 stage0 约定）。

## Files Changed

- `src/data/batch_download.py`：排除注册表 + `--gap-bins`。
- `src/data/aflow_adapter.py`：空 `Egap_type` 标签修复。
- `tests/test_stage0_robustness.py`：+2 回归测试。
