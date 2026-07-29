# CMJCC 最终论文准备清单

> **数字见 [`THESIS_OFFICIAL_RESULTS.md`](THESIS_OFFICIAL_RESULTS.md)；写作口径见
> [`CMJCC_Thesis_Writing_Guide.md`](CMJCC_Thesis_Writing_Guide.md)。** 本清单只管"做完了没有"。
>
> 旧版（写于 canonical oracle v3.0.0 之前，含已作废的 id 与数字）：
> `docs/legacy/CMJCC_Final_Thesis_Readiness_Checklist.SUPERSEDED.md`。

**当前状态：代码与自动实验已封存。** 下一阶段是论文写作与人工标注。

---

## 1. 已完成

### 代码与评测修正

- [x] **字段 arity 契约**（`llm/field_validation.py` + `orchestration/cmjcc.py`）。
      修掉了 hybrid 里 1/378 run 的 `AttributeError: 'list' object has no attribute 'strip'`；
      本轮 hybrid `crashed = 0`。
- [x] **全轮次归档完整性**。`model_calls.jsonl` 覆盖所有轮次并带 `turn_index` / `turn_run_id`；
      新增 `turn_records.jsonl`、`run_totals.json`；澄清改写调用与失败尝试都留有记录
      （只记异常类名，`call_id` 带 `#failed{n}` 后缀）。此前多轮运行**只导出最后一轮**，
      实测 3 个调用里有 2 个不可见。
- [x] **声明式 canonical oracle v3.0.0**。42/42 declared，0 system-derived；
      参考答案（字段值、hard/soft、unknown 处理、澄清答案）声明在场景文件里，
      纳入 `scenarios_fingerprint` 与 experiment id。
- [x] **指标与报告修正**：`precision_at_5` / `mean_graded_relevance` 在标签空间为空时返回 N/A；
      `unknown_hard_rate` 空分母返回 N/A；每行只报一个估计量；p 值科学记数法；
      §5.6 渲染检索 / top-k / 抽取来源三张此前从未出现的表；case study 带 repeat 标注；
      §1/§12/§13 措辞按后端派生。
- [x] **execution / analysis 指纹分离**。`experiment_id` 只由 execution fingerprint 派生，
      分析层改动不再作废实验；白名单（含 `jobrec_eval/simulated_user.py`）由静态 AST 扫描测试守护。
- [x] **长跑保护**。畸形 HTTP 200（非 JSON 体、缺 `choices`）转为 `LLMError` 走降级路径；
      单个 run 崩溃不再拖垮整批，缺口计入 manifest 并在报告中声明。
- [x] **澄清答案 fail-fast**。`clarification_expected` 场景缺声明答案时，
      `ExperimentRunner.run()` 第一行即抛错，一个 run 都不花。

### 实验与归档

- [x] **deterministic `exp-e748800507ef`**：210/210，crashed 0，verify OK，replay 210/210 0 diff。
- [x] **hybrid `exp-6db1e87daed5`**：378/378，crashed 0，verify OK，replay 378/378 0 diff，
      622 次调用 0 失败。
- [x] **两者身份一致**：`commit_hash f7970b81f653`、`execution_fingerprint f3ef9775f6b6d08a`。
- [x] **精简 release 已入库**：`final_release/`（100 文件、4.11 MB）+ 自校验 `checksums.json`。
- [x] **完整 bundle 归档**：`dist/` 下两个 ZIP，经 secrets 扫描、ZIP 完整性、
      解压后重验 checksums（5305 / 9505 个文件）后取 SHA-256，记录在 `provenance.json`。
- [x] **长期记忆验证集 `exp-1fe49fedbd22`**：10 runs 已归档，两棵树 verify OK。
      **仅作补充工程验证**，不可引用排序质量数字。

### 质量门禁

- [x] `ruff check src tests scripts` 通过
- [x] `mypy src` = 14（既有基线，未增加）
- [x] `pytest -m "not postgres and not perf"` = **840 passed**
- [x] `pytest tests/perf` = 22 passed
- [x] `pytest -m postgres` = 5 passed（真实 PostgreSQL）
- [x] coverage（`src/jobrec/*`）= 92%

---

## 2. 未完成

- [ ] **人工标注**。工具链与人工标签指标通路已完成并端到端验证
      （`jobrec_eval/annotation_ui/`、`--relevance-source human`、
      加权 kappa 与 oracle-vs-human 对比），但**尚无真人标注数据**。
      这是剩余的最大构念效度缺口，必须作为局限写入 §12。
- [ ] **长期记忆验证集的声明式参考答案**。目前 oracle 为 `system_derived_pass`（0/5 declared）。
      若论文要引用该集的排序数字，需先声明其参考答案并重算分析（分析层，无需重跑实验）。
- [ ] **Chapter 3 方法描述改写**，以匹配声明式 oracle 与已修正的指标定义。
- [ ] **Chapter 5–7 定稿**。

---

## 3. 封存后的操作纪律

- **不再修改运行代码，不再重跑 deterministic 或 hybrid。**
- 分析层（metrics / report / statistics）若需修正，可在已保存 bundle 上重算：
  `python -m jobrec_eval.cli pipeline --experiment-dir <bundle 目录> --out-root <新目录>`，
  这**不会**作废实验身份（execution fingerprint 不变）。
- 重算后须重建 release：`python scripts/build_final_release.py --write`，
  必要时再 `python scripts/build_bundle_archives.py --write`。
- 引用数字前先核对 `THESIS_OFFICIAL_RESULTS.md`。
