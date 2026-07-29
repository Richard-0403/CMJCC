# CMJCC 最终论文准备清单

> # ⛔ 本清单正文的实验 id 与数值均已被取代 — 先读这一节
>
> 正文写于 **canonical oracle v3.0.0（声明式参考答案）之前**，当时相关性标签由系统自己的抽取器产生。**唯一可引用的正式实验对：**
>
> | 角色 | experiment id | 规模 | 验证 |
> |---|---|---|---|
> | 主实验（deterministic，五变体） | **`exp-e748800507ef`** | 210/210，crashed 0 | verify OK；replay 210/210，0 diff |
> | 补充鲁棒性验证（hybrid，真实 LLM） | **`exp-6db1e87daed5`** | 378/378，crashed 0 | verify OK；replay 378/378，0 diff |
>
> - `commit_hash` `f7970b81f653`、`execution_fingerprint` `f3ef9775f6b6d08a`，**两者完全相同**。
> - canonical oracle **v3.0.0，42/42 declared，0 system-derived**（`inputs_fingerprint b66a395e…`，`reference_fingerprint be950e06…`）。
> - hybrid：622 次模型调用、0 失败、3,031,744 tokens、耗时 88.9 分钟（其中 86.6 分钟为 API 等待，占 97.4%）。
>
> **作废的 id**：`exp-197f6aacc171`、`exp-06cc34defe39`、`exp-87aec1bc99dc`、`exp-f90573008bdb`（仅可复现性证据）、`exp-8793b18de5b2`、`exp-515b63d6a656`、`exp-301060a1899d`。
>
> **归档口径已变更**：不再有 `final_release/deterministic_runs/`、`final_release/hybrid_runs/` 或 `dist/*.zip`。现在是**精简且已入库**的 `final_release/`（100 文件、4.11 MB）：报告、metrics/statistics 表、图、manifests、audit、各实验自身的 `checksums.json`、不含 bundle 的 run 溯源、以及冻结输入。`normalized/`（约 9 MB）与 run bundle（约 80 MB）刻意不入库，可从 bundle 再生。用 `python scripts/build_final_release.py --write` 重建。
>
> **数值一律以 `final_release/` 为准。** 已知变化：full NDCG@5 0.9585→**0.9115**、P@5 0.9743→**0.9581**、HCSR 1.000→**0.9838**、task_success 1.000→**0.9762**（`full` 不再满分，集中于 SC-D-02）。hybrid 记忆效应 Holm 0.0469→**0.188，不再显著**。
>
> **正文中"仍待用户拍板"的那条已解决**：hybrid 的 `AttributeError: 'list' object has no attribute 'strip'` 是字段 arity 缺陷，已修复（`field_validation` 的 arity 契约 + `cmjcc` 消费端），本轮 hybrid `invalid_runs = 0`、`crashed 0`。

---

> 目的：完成代码、评测与实验归档后，正式定稿 Chapter 5–7。  
> 当前判断：框架主体与**代码层评测修正（第 1–7、14、16 项）已完成并通过测试**；**人工标注的工具与人工标签指标通路（第 10、11 项的机器部分）已完成并端到端验证**；**deterministic 主实验与 hybrid 补充鲁棒性验证已在同一份代码上双双跑完（第 8、9 项），真实 PostgreSQL 套件已真跑通过（第 12 项），perf 与 mypy 已补齐（第 13 项），replay 与 checksums 已在正式产物上复验（第 15、16 项），两组产物已归档并打包（第 17 项的归档部分）**。剩余工作是真人标注这一趟、代码冻结与 Chapter 3 改写。  
> 状态基准：本次核查发生在**LLM 调用记账补齐 + deterministic 与 hybrid 双实验重跑之后**。`[x]` 表示已在代码、测试或正式产物上验证；`[ ]` 表示尚未完成，并在各节 **状态** 中写明还差什么。  
> **仍待用户拍板的一项**：hybrid 里发现的一个真实代码缺陷（`AttributeError: 'list' object has no attribute 'strip'`，1/378 run）是否修复并重跑 hybrid——见第 9 项末尾，本清单**不代替用户决定**。

> **配套文档**：论文写作口径见 **`CMJCC_Thesis_Writing_Guide.md`**——所有“写错就站不住”的表述约束（生存者偏差、grounding=1.000 的正确解释、ActiveSearchState 措辞、LLM 延迟 caveat、no-match 定义、禁止表述清单）都集中在那里。本清单管“做完了没有”，那份文档管“怎么写才成立”。

> **正式实验标识（引用时必须使用这两个）**
>
> | 用途 | experiment id | 位置 |
> |---|---|---|
> | **主实验**（deterministic，五变体） | **`exp-197f6aacc171`** | `evaluation/outputs/exp-197f6aacc171/` + `evaluation/outputs/_runs/exp-197f6aacc171/` |
> | **补充鲁棒性验证**（hybrid，真实 LLM） | **`exp-06cc34defe39`** | `evaluation/outputs_hybrid/exp-06cc34defe39/` + `evaluation/outputs_hybrid/_runs/exp-06cc34defe39/` |
>
> **两者的 `source_fingerprint` 完全相同**（`8eba8f8106dc...`），即两组产物出自同一份代码——这是本轮刻意付出重跑代价换来的一致性，`final_manifest.json` 因此可以用一个指纹同时覆盖主实验与 hybrid。
>
> 不可引用：`exp-f90573008bdb`、`exp-515b63d6a656`、`exp-301060a1899d`（均为过程中的中间产物，前两者已删除）；`exp-8793b18de5b2` 与 `test_results/` 是**修复前**产物，**有意保留**（共 2.1 MB，已入库 75 个文件，作为“哪些结论被取代”的历史记录），但**不得引用**。
>
> experiment id 把源码指纹计入哈希，因此改代码必然换 id——这正是为了避免不同代码版本的产物互相覆盖（详见第 8 项）。

---

## 一、必须修复的评测问题

### 1. 修复 clarification 场景的 Task Success 计分

- [x] 修改 `src/jobrec_eval/metrics.py` 中的 `MetricsComputer._task_success()`。
- [x] 不再要求 clarification-dependent 场景的最终 response 必须是 `clarification`。
- [x] 改为判断整个 dialogue/session 是否完成以下过程：
  1. 系统提出必要澄清；
  2. clarification target 正确；
  3. simulated user 提供回答；
  4. 系统最终返回正确 recommendation 或正确 no-match；
  5. 最终结果满足 relevance、hard constraints 和 evidence-linked rationale 要求。
- [x] 使用 `dialogue_trace.jsonl` 或完整 turn history 判断是否执行过必要澄清。
- [x] 对跳过必要澄清的运行，不得错误计为成功。
- [x] 对完成“澄清 → 推荐”的运行，正确计为成功。

**完成标准**

- clarification-dependent 场景中，正确完成澄清与推荐的 run 不再被记为 `task_success = 0`。
- full variant 的 Task Success 不再因为计分逻辑错误而下降。
- Task Success 与真实 dialogue trace 一致。

**状态：已完成（代码 + 实测）**

- `MetricsComputer._task_success()` 与 `_partial()` 现在经由 `metrics.py` 中新增的 `dialogue_view()` / `DialogueView` 读取 `dialogue_trace.jsonl`，对**整段对话**计分：必须“提出澄清 + 命中可接受 slot + 用户已回答 + 终局正确（HCSR 仍为 1.0 且 rationale 有据）”才记 1。
- 跳过必要澄清、只问了错误 slot、或在 `max_turns` / `cannot_answer` / `repeated_slot` 上停摆，一律记 0。
- 没有 dialogue 证据的旧 bundle 回落到原来的 final-response 规则，保证向后兼容。
- 实测：`full` 在 7 个 clarification-dependent 场景（SC-B-01..05、SC-G-01、SC-G-02）上 `task_success` 全为 1（修复前全为 0）；`no_memory` 6/7、`no_context` 1/7，这些 0 是真实消融效应，不是计分缺陷。

---

### 2. 修复 clarification precision / recall 的数据来源

- [x] 不再只从最终 `RunRecord` 或最终 response 中读取 clarification。
- [x] 从整个 `dialogue_trace.jsonl` 中提取：
  - 是否提出 clarification；
  - clarification slot；
  - reason code；
  - 是否重复提问；
  - simulated user 是否回答；
  - 最终 termination reason。
- [x] 正确计算：
  - Necessary clarification recall；
  - Clarification precision；
  - Useful clarification rate；
  - Unnecessary clarification rate；
  - Repeated-question rate。

**完成标准**

- 已经提出过正确澄清的 run 不再显示 clarification recall = 0。
- 指标能反映整个对话过程，而不只是最后一轮。

**状态：已完成，但有一处如实记录的缺口**

- 新增 per-run 列：`clarification_asked`、`clarification_asked_slots`（记录**全部**提问，不只是最后一次）、`clarification_answered`、`clarification_repeated_slot`、`clarification_reason_code`、`response_turns`、`termination_reason`。
- `clarification_metrics` 基于这些列重写，并新增 `necessary_recall`、`useful_rate`、`unnecessary_rate`、`repeated_question_rate`、`answered_rate`。
- 缺口：`clarification_reason_code` 只有在“对话结束时仍停留在未回答的提问上”才有值，因为 trace 记录本身不携带 reason code。论文中若引用 reason code 分布，需说明这一取值范围。

---

### 3. 将 Clarification Efficiency 接入正式评测流水线

- [x] 在主评测 pipeline 中调用 `clarification_efficiency()`。
- [x] 将指标加入：
  - per-run metrics；
  - variant summary；
  - scenario-type summary；
  - 最终 Markdown/CSV 报告。
- [x] 明确必要澄清与非必要澄清的评分规则。
- [x] 将 `response_turns` 与 task success 联合解释。
- [x] 避免将“少问问题但结果错误”评价为高效率。

**建议报告指标**

- Median response turns；
- IQR response turns；
- Necessary clarification recall；
- Unnecessary clarification count；
- Clarification efficiency score；
- Repeated-slot guard activations。

**状态：已完成**

- `clarification_efficiency()` 此前是**从未被调用的死代码**；现在由 `run_pipeline` 实际调用，写出 `metrics/clarification_efficiency.csv`，并进入 per-run metrics、variant summary、scenario-type summary 以及报告 §5.4（median/IQR response turns、necessary recall、unnecessary count、efficiency score、repeated-slot guard 触发次数）。
- skip penalty 在数学上保证：跳过必要澄清的 run 的效率分不可能高于提出了该澄清的 run。

---

### 4. 修正报告中的统计方法描述

- [x] 修改 `src/jobrec_eval/report.py` 中旧的描述：
  - 删除 “run-level discordant pairs”；
  - 改为 “scenario-level paired outcomes”。
- [x] 明确说明：
  - `scenario_id` 是独立分析单位；
  - `repeat_index` 仅用于稳定性和方差分析；
  - McNemar 使用 scenario-level paired success；
  - deterministic runs 默认一次；
  - stochastic/hybrid runs 可重复多次。

**完成标准**

- 代码实现、Chapter 3 方法描述和自动生成报告三者一致。
- 报告中的 `n_pairs` 等于有效配对场景数，而不是总运行数。

**状态：报告文字已完成；Chapter 3 文字待第 八 节处理**

- 报告 §8 重写：`scenario_id` 为分析单位，`repeat_index` 仅用于稳定性/方差，McNemar 基于 scenario-level 配对二值结果（多次重复按多数票折叠，偶数次平票记为 not-success），`n_pairs` = 有效配对场景数。
- 新增 pairing-provenance 表（scenarios / runs / repeats / valid pairs / discordant），读者可直接看出 `n_pairs` 是场景数而非运行数。
- 需要说明的是：**代码本来就是 scenario-level**（`statistics.aggregate_scenario_success`），过时的只是报告文字。
- 附带修掉两个报告文字缺陷（本次核查发现，与上述改动无关）：
  1. §2 原本硬编码 “Five variants”，在三变体运行下自相矛盾，现按 `exp['variants']` 推导，1 个变体时单复数也正确；
  2. 报告头原本声称所有数字可从 `raw/` 复现，而该目录从不存在，现改为真实位置 `_runs/<experiment_id>/<variant>/<scenario_id>/<run_index>/`，并说明它与分析输出目录 `<experiment_id>/` 同级、同在 `--out-root` 之下。
  3. 上述两点由 `tests/eval/test_report_header_and_variant_count.py` 与 `tests/eval/test_pipeline_artifacts.py` 中的落盘校验守住。

---

## 二、需要完成的框架机制评测

### 5. 将 Failure-Path 指标接入正式报告

- [x] 将以下指标接入主报告：
  - `failure_detection_rate`
  - `recovery_success_rate`
  - `grounding_rate`
  - `handoff_success_rate`
- [x] 单独建立 fault-injection 结果部分。
- [x] 覆盖：
  - invalid evidence ID；
  - missing evidence source；
  - wrong-field evidence；
  - unsupported salary claim；
  - unsupported location claim；
  - unsupported skill claim；
  - schema-invalid handoff；
  - missing-field handoff；
  - agent exception；
  - timeout + retry；
  - partial failure + recovery。
- [x] 区分：
  - 正常主实验；
  - fault-injection robustness experiment。

**完成标准**

- 正常场景中的 grounding 可以是 1.000。
- fault-injection 场景能够证明错误会被检测、拒绝或恢复。
- 不为了让指标低于 1.000 而把故障样本混入主实验。

**状态：已完成**

- 这四个 rate 同样是死代码；现在被实际调用，写入 `metrics/failure_metrics.csv`（tidy long form，带 numerator/denominator，因此任何 N/A 都可解释）。
- 报告新增 §10 “Fault-Injection Robustness (separate robustness experiment)”：§10.1 为四个 rate，§10.2 为 11 类故障覆盖表，来源是 `tests/support/fault_injection.py` + `tests/unit/test_failure_paths.py` + `tests/integration/test_failure_metrics.py`。
- §10 明确写出三件事：主实验 grounding = 1.000 是合理的；故障样本**不**混入主实验；分母为空时显示 N/A，而不是伪造 0.0 或 1.000。

---

### 6. 生成框架机制贡献结果

- [x] 在报告中正式计算：
  - `Δmemory = M_full − M_no_memory`
  - `Δcontext = M_full − M_no_context`
- [x] 至少对以下指标计算 delta：
  - NDCG@5；
  - HCSR；
  - Task Success；
  - Explanation Grounding；
  - Response Turns；
  - Clarification Efficiency。
- [x] 对 memory-dependent subset 单独报告。
- [x] 对 context-dependent subset 单独报告。
- [x] 使用统一表述：

> Framework mechanism contribution under the controlled prototype instantiation.

- [x] 避免使用“全面优于所有其他框架”的表述。

**完成标准**

- Chapter 5 的主要结果可以直接对应 candidate-memory mechanism 和 job-context orchestration mechanism。
- 结果明确属于框架机制评估，而不仅是原型性能展示。

**状态：已完成**

- `response_turns` 与 `clarification_efficiency` 的 Δ 作为**独立的 `SECONDARY` outcome family** 计算，Holm 校正与预注册的 `PRIMARY` family 分开进行，因此没有任何 primary p-value 被改变（有测试断言 primary 行与“只算 primary”的结果逐字节一致）。
- 渲染在新增的 §6.3；memory-dependent / context-dependent 子集本来就分开报告；表述字符串未改动（仍为 “contribution of that framework mechanism under the controlled prototype instantiation”）。

---

### 7. 完成消融实验配置一致性检查

- [x] 在生成比较报告前验证：
  - catalog hash；
  - scenario hash；
  - prompt hash；
  - model settings；
  - top-k；
  - retrieval pool size；
  - random seed；
  - commit hash。
- [x] 检查消融对只在目标机制 flags 上不同：
  - full vs no-memory：只允许 memory-related flags 不同；
  - full vs no-context：只允许 context-related flags 不同。
- [x] 配置不一致时停止报告生成。
- [x] 将 consistency flags 写入 run manifest。

**完成标准**

- 结果差异可以合理归因于被移除的框架机制。
- 不存在 catalog、prompt、ranking 设置等混淆因素。

**状态：本次会话之前就已实现，本次仅复核**

- `write_report()` 在写出任何文件之前校验 run manifest，不一致即抛错；per-ablation 的 flag scoping 已包含在内。

---

## 三、必须重新运行的正式实验

### 8. 重新运行完整 Deterministic 主实验

- [x] 修复 clarification 计分后，清除旧实验输出。
- [x] 使用当前冻结前代码重新运行全部场景。
- [x] 至少包含：
  - `full`
  - `profile_only`
  - `one_shot`
  - `no_memory`
  - `no_context`
- [x] deterministic mode 默认每个 scenario-variant 运行一次。
- [x] 保存：
  - raw run bundles；
  - dialogue traces；
  - state versions；
  - retrieval results；
  - ranking results；
  - explanations；
  - handoffs；
  - evidence logs；
  - run manifests；
  - checksums。
- [x] 确认 0 个 unexpected system failures。
- [x] 生成新的：
  - summary tables；
  - statistical tests；
  - ablation deltas；
  - figures；
  - error analysis。

**注意**

不得继续使用仓库中修改前生成的旧结果，包括仍写着 “68 tests”、旧 McNemar 描述或 `one_shot == no_memory` 的报告。

**状态：已完成——`exp-197f6aacc171`，210 runs，0 system_error，0 invalid_runs，两棵目录树 `cli verify` 均 OK**

复现命令（如实记录，`--out-root` 默认 `evaluation/outputs`，bootstrap 默认 5000 iters / seed 2026，relevance 默认 oracle）：

```powershell
.venv\Scripts\python.exe -m jobrec_eval.cli pipeline --config configs/experiment_full.yaml `
  --scenarios evaluation/data/scenarios.jsonl --catalog data/processed/jobs.jsonl `
  --variants full,profile_only,one_shot,no_memory,no_context --repeats 1
.venv\Scripts\python.exe -m jobrec_eval.cli verify evaluation/outputs/_runs/exp-197f6aacc171
.venv\Scripts\python.exe -m jobrec_eval.cli verify evaluation/outputs/exp-197f6aacc171
```

**⭐ 本轮重跑最值得写进论文的一条证据**：`exp-197f6aacc171` 与被它取代的 `exp-f90573008bdb` 之间，**`variant_summary` 的全部指标逐项一致**。两者之间的代码改动只发生在 LLM 调用记账（token usage / request params / retry 痕迹 / raw_response 落盘、`ReplayProvider` 索引，详见第 9 项）。逐项一致因此**证明这些改动没有触及任何指标计算路径**——这既是可复现性证据，也是"为什么旧 headline 数字可以原样沿用、只换 id"的依据。error_taxonomy 分布同样不变（missing_constraint_enforcement 35 / missing_dialogue_evidence 35 / stale_or_missing_memory 18 / missing_dialogue_continuation 7 / other 1）。

**headline（full variant，scenario-mean，附分母）**：NDCG@5 0.959 (n=37)、HCSR 1.000 (n=37)、Task Success 1.000 (n=42)、Grounding 1.000 (n=42)、Handoff 1.000 (n=42)。数据质量：0 error / 2 warning（SC-E-02、SC-E-04）/ 27 条已确认 fixture。

**变体总览（务必连 `_n` 一起引用，理由见下方“生存者偏差”）**

| variant | NDCG@5 (n) | P@5 (n) | HCSR (n) | TaskSucc (n=42) | ClarEff |
|---|---|---|---|---|---|
| full | 0.959 (37) | 0.974 (37) | 1.000 (37) | **1.000** | -1.50 |
| profile_only | 0.587 (24) | 0.500 (28) | 0.500 (28) | 0.167 | -335.00 |
| one_shot | 0.949 (21) | 1.000 (21) | 1.000 (21) | 0.619 | -382.50 |
| no_memory | 0.945 (28) | 0.986 (28) | 0.986 (28) | 0.762 | -216.00 |
| no_context | 0.710 (38) | 0.595 (42) | 0.600 (42) | 0.167 | -1.50 |

**消融结论**：Δcontext 在 context-dependent 子集上 task_success Δ=1.000（15/15 全 discordant）、HCSR Δ=0.520、NDCG Δ=0.259、mean_violation Δ=-0.740，CI 全部不含 0；Δmemory 在 memory-dependent 子集上 task_success Δ=0.5625，p_holm=0.023。

**本轮为通过本项而先修掉的四件事（全部有测试守住）**

1. **`one_shot` 与 `no_memory` 曾经行为完全相同**（42 场景 × 43 列，0 处差异），因为 `use_multi_turn_continuation` 的唯一消费者是 `orchestrator.py:177` 的 `use_prior_dialogue and use_multi_turn_continuation`，而这两个变体的 `use_prior_dialogue` 都是 False——这个 flag 在语义上是**死的**，实验 runner 的澄清循环只看场景 flag，从不看变体 flag。现已让 runner 的澄清循环真正受 `use_multi_turn_continuation` 门控（新增 `TERMINATION_CONTINUATION_DISABLED = "continuation_disabled"` 与 `_continues_dialogue(flags)`），`one_shot` 因此是**真正的单轮基线**。复核结果：两者在 7 个澄清场景（SC-B-01..05、SC-G-01、SC-G-02）上各有 21 列实质差异（turns 1 vs 2、termination `continuation_disabled` vs `recommendation`、returned_count 0 vs 5、task_success 6/7 处 0 vs 1），其余 35 个场景仅 `variant` 标签列不同。**场景脚本轮次刻意未截断**：它们是实验刺激，截断会让不同变体收到不同输入，破坏配对同输入的基础。
2. **experiment id 曾会跨代码版本碰撞并静默覆盖产物**。旧 id 只哈希 config/catalog/scenarios/prompt，不含代码，因此修完 bug 重跑会算出同一个 id 并**就地覆盖**修复前的 artifact（这一点已实际发生，`exp-301060a1899d` 的修复前基线因此丢失）。现改为把**源码指纹**（`jobrec/` + `jobrec_eval/` 全部 `*.py`，LF 归一化后 sha256）计入 id，并加 `guard_output_dir` / `ExperimentOverwriteError` 覆盖保护（要覆盖必须显式 `--allow-overwrite`）。之所以用源码指纹而不是 `code_version`：后者是静态的 `"0.1.0"`；`commit_hash` / `git_dirty` 也记录在 manifest 里但不入 id，因为工作树是 dirty 的。本次 manifest：`commit_hash 9768116417...`、`git_dirty true`、`source_fingerprint 8eba8f8106dc...`（与 hybrid 的 `exp-06cc34defe39` **完全相同**）。
3. **`clarification_efficiency` 曾不惩罚“问了却放弃”的对话**：`one_shot` 放弃全部 7 个澄清对话，却因为 skip penalty 只在“必要澄清从未被提出”时触发而排在 `no_memory` 之上。现新增独立的 unresolved-dialogue penalty，判定信号是**最终 `response_type == "clarification"`**（刻意不复用 `DIALOGUE_NOT_CONTINUED_REASONS`：那只指一种原因，计分必须把所有放弃一视同仁；也刻意不用 `clarification_answered`（在 `repeated_slot` 下为真）或 `task_success`（会把排序质量折进来））。评分序变为 *问了并解决* > *问了但放弃* > *跳过必要澄清*。修复后 `one_shot` 的 EffScore 为 -382.50（最差），`no_memory` -216.00，新增 `Abandoned`(`asked_unresolved`) 与 `AnsweredRate` 两列与 EffScore 并排呈现：`one_shot` 放弃 16、AnsweredRate 0.125，`no_memory` 放弃 9、0.562。
4. **error taxonomy 曾把 `one_shot` 的截断失败标成 `stale_or_missing_memory (ablation)`**，名不副实。新增类别 `missing_dialogue_continuation (ablation)`，键控在**已记录的列**上而非变体名。`cannot_answer` / `max_turns` / `repeated_slot` **刻意排除在外**（那些场景的续话机制是工作的；折进来会把 `profile_only` 的 7 个 `repeated_slot` 失败从 baseline 类别里偷走）。最终分布：missing_constraint_enforcement 35、missing_dialogue_evidence 35、stale_or_missing_memory 18、missing_dialogue_continuation 7、other 1，合计 96 = 变体失败数之和（no_context 35 + profile_only 35 + one_shot 16 + no_memory 10）。

**⚠️ 生存者偏差——引用变体表时必须连分母一起引用**

`one_shot` 的 P@5 与 HCSR 都读作 **1.000**，比 `full` 的 0.974 **更高**，但这纯粹是因为它放弃的 7 个最难场景没有返回排序列表、因而**退出了分母**（n 从 37 掉到 21）。排序类指标只在“变体确实返回了排序列表”的场景上取平均，`grounding` 同理（无推荐即无 claim 可核，`one_shot` 的 n(Grounding) 只有 26）。因此：

- 跨变体比较**只能用 `task_success`**（42 个场景全部有定义，放弃即记失败，不会退出分母）；
- 排序类指标只在 `_n` 相当时才可比；
- `full vs one_shot` 的 NDCG Δ=+0.005（CI [+0.000, +0.014]，含 0，n=21 配对场景）**不能**表述为“无差异”，正确表述是**“在 `one_shot` 放弃的那些场景上不可估计”**。
- 这三点现在**由报告自己印出来**，不再依赖读者自觉：§1 headline 每个指标带 `n`，§5 表下新增逐指标分母表 + “Read the denominators before the means” 警示段，§5.x 每条 Δ 带 `n=… paired scenarios` 并明确否证等价读法，§5.2 每格改为 `rate (n=applicable)`（并在未知值占比非零时附 `unk X%`），§7 说明其 `n` 是场景数而非逐指标分母。同时修掉两处**静默丢变体**的渲染缺陷：§7 曾硬编码只渲染 full/no_memory/no_context，§5.2 曾硬编码只渲染 full/no_context/profile_only，而两张底层 CSV 都带全部 5 个变体（§5.2 现在因此暴露出关键事实：`one_shot` 的 work_modes 合规率 1.000 只建立在 **6** 个可判定 pair 上，而 `no_context` 的 0.457 建立在 35 个之上、其中 37% 为未知值）。
- 旧产物 `evaluation/outputs/exp-8793b18de5b2/` 与 `test_results/` 仍带修复前的标签与数字（旧 “run-level discordant pairs” 表述、旧 “68 tests” 说法）。按用户决定**保留不删**，但**不得引用、也不得手工改写**——要用就整体重跑替换，否则只能另附 erratum。

---

### 9. 重新运行 Hybrid / LLM 验证实验

- [x] 使用当前代码重新生成 hybrid 结果。
- [x] 至少比较：
  - full；
  - no-memory；
  - no-context。
- [x] 覆盖：
  - memory-dependent scenarios；
  - context-dependent scenarios；
  - clarification-dependent scenarios；
  - malformed/ambiguous extraction cases（真实发生：schema_failure_rate 非零、rule fallback 实际触发，另有 2 个 run 收到列表包裹值——其中 1 个暴露了一个真实缺陷，见本项末尾）。
- [x] 每个 scenario-variant 至少重复 3 次（实际 3 次）。
- [x] 保存：
  - model ID；
  - provider；
  - prompt hash；
  - request parameters；
  - response metadata；
  - schema failures；
  - retry count；
  - fallback reason；
  - fallback rate；
  - token usage；
  - latency。
- [x] 比较 deterministic 与 hybrid 的结果差异。

**完成标准**

- Hybrid 结果来自当前代码，而不是旧版本。
- 能够量化 LLM extraction、repair、retry 和 rule fallback 的实际使用情况。

**状态：已完成——`exp-06cc34defe39`，378 runs，耗时 134.7 分钟，0 system_error，两棵目录树 `cli verify` 均 OK；`invalid_runs = 1`（一个已定位、已量化、**尚未修复**的真实缺陷，见本项末尾）**

**论文定位（用户已决定）**：**deterministic `exp-197f6aacc171` 是主实验，hybrid `exp-06cc34defe39` 是补充鲁棒性验证。** 两者 `source_fingerprint` 相同（`8eba8f8106dc...`），即出自同一份代码。**不得把两者的数字混在同一张表里平均**（口径见 `CMJCC_Thesis_Writing_Guide.md` §6「deterministic 主实验 与 hybrid 补充鲁棒性验证：关系与写法」）。

**规模与运行方式**

- 378 runs = `full` / `no_memory` / `no_context` × 42 场景 × **3 重复**。
- provider：`gpt-5.5` @ `https://api.vectorengine.ai/v1`（OpenAI 兼容接口）。
- 报告头正确渲染为 `Run mode: **hybrid** (model: remote)`——因此报告本身不会被误读成 deterministic。
- **如实说明的覆盖缺口**：hybrid 只跑了 **3 个变体**，`profile_only` 与 `one_shot` **没有 hybrid 对照**。因此这两个基线的任何结论只能引 deterministic 主实验，hybrid 不能用来讨论它们。

**真实 LLM 成本（本项原先声明的"唯一来源"，现已获得）**

- 调用次数：**378** 次，全部 `purpose = intent_extraction`，每 run 恰好 **1.00** 次。
- token：prompt **227,105** + completion **113,969**（其中 **reasoning 52,333**——`gpt-5.5` 是推理模型，reasoning token 必须单独报告，否则 completion 会被低估）= **total 341,074**，每 run 平均 **902**。
- `usage` 字段缺失 **0** 次（即 378/378 都有真实用量，没有一处是补零推算的）。
- `raw_response` **378/378** 已写入（脱敏后）。

**真实延迟（本项原先声明的"唯一来源"，现已获得）**

- 单次调用：中位 **11,624 ms** / p95 **20,420 ms** / 最大 **57,518 ms**。
- LLM 等待合计 **81.7 分钟**，占总耗时 **134.7 分钟**的约 61%。
- 变体级 `total_latency_ms` 均值：full **12,353** / no_memory **17,237** / no_context **14,642**。
- 对比第 13 项的 perf 套件（hybrid 单元格 LLM 中位数 0.558–0.701 ms）：那是 **mock 计算耗时**。**论文报告真实延迟/成本时必须引本项**，报告编排开销的规模行为时才引 perf。

**schema / retry / fallback / 抽取来源（本项原先声明的"唯一来源"，现已获得）**

- fallback **0** 次；retry（去掉 `response_format` 重试）**0** 次。
- 抽取来源占比与 schema 失败：

  | variant | rule | llm | schema_failure_rate | rule_fallback 字段数 |
  |---|---|---|---|---|
  | full | 0.204 | 0.796 | — | — |
  | no_memory | 0.009 | 0.991 | 0.0094 | 4 |
  | no_context | 0.205 | 0.795 | 0.0019 | 1 |

  `no_memory` 的 rule 占比只有 0.009，是因为没有记忆可复用、几乎每个字段都得靠模型抽取——这也解释了它更高的 schema 失败率。

**hybrid vs deterministic（同场景集、同变体）**

| variant | 指标 | hybrid | deterministic |
|---|---|---|---|
| full | task_success | **0.9206** | 1.0000 |
| full | ndcg@5 | 0.9584 | 0.9585 |
| full | hcsr | 0.9910 | 1.0000 |
| full | grounding | 1.000 | 1.000 |
| full | handoff | **0.996** | 1.000 |
| no_memory | task_success | 0.7381 | 0.7619 |
| no_memory | ndcg@5 | 0.9271 | 0.9452 |
| no_context | task_success | 0.1667 | 0.1667 |
| no_context | ndcg@5 | **+0.0173**（hybrid 略高） | — |
| no_context | hcsr | **+0.0286**（hybrid 略高） | — |

- full 的 handoff 0.996 < 1.000 就是那次崩溃导致的。
- `no_context` 在 hybrid 下 ndcg/hcsr **反而略高**：属小样本波动与 LLM 抽取差异（task_success 两侧同为 0.1667），**不得**表述为"去掉 context 反而更好"（禁例已写入写作指导 §13）。
- hybrid 分母（引用时必须带上，生存者偏差同样适用）：

  | variant | n(ndcg) | n(task) | n(grounding) |
  |---|---|---|---|
  | full | 37 | 42 | 42 |
  | no_memory | 30 | 42 | 35 |
  | no_context | 38 | 42 | 42 |

**provenance 字段的确切位置（引用时别找错文件）**

- `model_manifest`（`provider` / `model` / `base_url` / `api_key_env` / `api_key_present`）记在**每个 `run_record.json`** 里，**从不包含密钥值**——只记录环境变量名与"是否存在"。
- `prompt_hash` **同时**在 `experiment_manifest.json` 与 `run_record.json` 中。
- **`experiment_manifest.json` 顶层没有 provider 字段**，所以要查 provider/model 必须逐 run 看 `model_manifest`，不要在实验级 manifest 里找。

**为满足本项而先补的代码改动（全部有测试守住）**

1. **`RemoteLLMProvider` 此前从不填充 `LLMCallRecord.metadata`**——token usage、temperature、retry 痕迹全部丢失，本项要求的"保存 request parameters / response metadata / token usage / retry count"因此**根本无法满足**。现在记录：归一化的 token 用量（同时兼容 `prompt_tokens`/`completion_tokens` 与 `input_tokens`/`output_tokens` 两种拼写）、`finish_reason`、request parameters、以及 `attempts` / `retried_without_response_format` / `retry_reason`。两个刻意的选择：`usage` 缺失时记为**缺失**而不是伪造 0；request parameters 按**实际发送的 body 反读**，因此 gpt-5 系列被刻意省略的 `temperature` 记为缺失，而不是记一个从未真正发送过的值。
2. **`config.llm.save_raw_responses` 此前只作用于数据库，对 run bundle 无效**。现在它也 governs bundle：开启（默认）时写入脱敏后的 `raw_response`，**关闭时该键完全缺失**——这样"未保留"不会被误读为"模型没返回内容"。脱敏刻意镜像 `SqlRepository._model_call_payload` 的同一套政策（`redact()` + `redact_candidate_text`），两个 sink 不另立第二套标准。**prompt 在任何模式下都不持久化。**
3. **`ReplayProvider` 此前对 hybrid 产物必然失败**：它按 `content_id("call", purpose, prompt)` 建索引并读 `raw_response`，而 bundle 行两者都没有。现在每条记录同时索引到 `call_id`（remote provider 的 `call_id` 恰好**就是**那个 content id）与（若 prompt 存在时）由 prompt 重算的 key，因此**无需持久化 prompt 也能重放**。
4. 新测试 `tests/unit/test_model_call_accounting.py`（**22 个**），全部使用伪造 payload + monkeypatch，**未发生任何真实 API 调用**。

**⚠️ 本轮新发现的真实代码缺陷——已定位、已量化、**尚未修复**，且"是否修复并重跑"**仍待用户拍板****

`invalid_runs = 1`：`full` / **SC-A-04** / repeat 1，`failure_code = INTERNAL_ERROR`，`AttributeError: 'list' object has no attribute 'strip'`。

根因链路（已完整定位并复现）：

1. 真实 LLM 把 `normalized_value` 返回成**单元素列表**；
2. `validate_field("target_roles", ["software engineer"])` 返回 **ok=True 且原样保留该列表**——因为 `target_roles` 本身是列表型字段；
3. `orchestrator.py` 的 schema 修复循环**只处理 ok=False**，于是跳过了它；
4. 该列表流入 `cmjcc.py` 的 `canonical_role(r) for r in list_values["target_roles"]`，`canonical_role` 对字符串调 `.strip()` → 崩溃。

**对照组说明修复机制本身是对的**：同一条响应里的 `preferred_locations` 与 `work_modes` 被判 ok=False，因此被 `_repair_raw`（单元素列表 → 标量）**正确修复**。漏掉的是"验证通过但形状仍错"这一类，不是修复机制失效。

**影响面已量化（写论文时必须连带说明，避免夸大）**

- 378 个 run 中只有 **2 个**返回列表包裹值：1 个崩溃，1 个被正确修复；
- 检查全部 378 个 `active_search_state.json`，**零个**出现嵌套列表 → **没有静默数据污染**；
- 该崩溃对 full 场景级 task_success 的影响仅 **0.9206 → 0.9286**（0.008）；
- full 的 10/126 个 run 级失败分布：SC-A-04 1 次（=这个崩溃）、SC-D-11 2 次、SC-D-12 1 次、**SC-G-01 3/3 全败**、**SC-G-02 3/3 全败**。即 deterministic 与 hybrid 之间 **0.079** 的差距中，**约 90% 是真实 LLM 行为**（尤其 SC-G-01 / SC-G-02 这两个澄清场景在真实 LLM 下**系统性失败**），只有约 **10%** 来自这个崩溃。

**⏸ 待用户决定：是否修复并重跑 hybrid。** 两条路的代价如实列出，**本清单不代替用户做这个决定，也不得把它写成"已修复"**：

- **修复并重跑**：修代码会改变 `source_fingerprint`，从而**破坏本轮刚建立的"两组产物同指纹"一致性**（除非 deterministic 也一起重跑），并且重跑 hybrid 需再花约 **135 分钟**与约 **34 万 tokens**；换来的只是 full task_success 从 0.9206 到 0.9286（**0.008**）。
- **不修、如实报告**：把它作为 Chapter 6 的失败案例写出来（真实 LLM 产生了流水线未处理的输出形状），连带说明 1/378 的影响面、零数据污染、以及它在 hybrid–deterministic 差距中只占约 10%。写法见 `CMJCC_Thesis_Writing_Guide.md` §6.4。

---

## 四、人工评测工作

### 10. 完成人工 Relevance 标注

- [ ] 导出 unique scenario-job pairs。
- [ ] 使用 0–3 graded relevance：
  - 0 = irrelevant；
  - 1 = weak fit；
  - 2 = partial fit；
  - 3 = strong fit。
- [ ] 两名评分者独立标注。
- [ ] 计算 weighted Cohen’s kappa。
- [ ] 导出 disagreement cases。
- [ ] 完成 adjudication。
- [ ] 使用 adjudicated labels 重新计算：
  - NDCG@5；
  - Precision@5；
  - Mean graded relevance。
- [ ] 比较 automatic oracle 与 human labels。

**状态：标注工具已建成并测试通过；未勾选的原因是标签尚未采集——阻塞项已从“缺机制”变成“缺人工时间”**

- 新增标注包 `src/jobrec_eval/annotation_ui/`：一个跑在 localhost 的 FastAPI + 服务端渲染 Web UI，外加一层与界面解耦的 headless 数据层（`store.py` 为 SQLite/WAL 存储、`loader.py`、`assignment.py`、`export.py`、`console.py`、`app.py`、`templating.py`、`views.py`、`templates/`、`static/`）。也就是说，前端标注页面已经不是计划，而是可运行的工具。
- **多评分者按 option A 设计**：评分者是一个规模为 N 的**池**，但每个条目**恰好**分配 2 名不同评分者，分配过程带随机种子且负载均衡（`max_load_imbalance <= 1`）。这样 `rater_1` / `rater_2` 的导出契约与成对 Cohen's kappa 才继续成立；每条目 3 名以上评分者需要改用 Fleiss' kappa 和另一套 CSV，因此**被有意拒绝**（池规模 < 2 时抛 `InsufficientRaterPoolError`）。
- **盲评是结构性的，不是约定**：评分者可见的载荷与机器答案（oracle 分级、validator 判定）落在不同的数据库列上，评分者侧的类型里**没有**承载分析侧结果的字段，写入时若载荷未盲则直接拒绝。评分者隔离同理：任何人都读不到、也覆盖不了别人的标签。这两点之所以重要，是因为它们是所报告 kappa 的前置条件。
- claim 条目按内容寻址的 `claim_id` **去重**，一次判断即覆盖所有产出同一句子的 run；导出时再按 `(run_id, claim_id)` 展开回逐行。
- **裁定（adjudication）是一条被记录的判定**，带裁定人姓名和**必填**的书面理由。CSV 的 `adjudicated` 列只从该判定填充：两名评分者一致时留**空**（他们的共同标签即金标准），分歧确实未裁定时也留**空**（如实报告为 unadjudicated 并排除，绝不取平均）。旧的 `round((rater_1+rater_2)/2)` 启发式只作为**带标注的回退**保留给没有 `adjudicated` 列的历史文件，永远不参与构建已发表指标背后的标签表。
- 无法解析的 evidence id 在 grounding 标注界面上**显著标出并打旗标**（即本清单第 11 项的“检查 claim 对应 evidence ID 是否可解析”），因为一条指向空处的引用支撑不了任何结论。
- **实测工作量（来自真实运行，不是估算）**：12 场景子集 × 2 变体 → 48 条 relevance 条目、425 次 claim 出现去重为 94 条唯一 claim 条目（4.5×）；全部 5 个变体 → 100 条 relevance 条目、1101 次出现去重为 147 条 claim 条目（7.5×）。按完整实验 42 场景 × 5 变体外推：约 **350** 条 relevance 条目、约 **515** 条 claim 条目待判断，`claim_annotations_human.csv` 约 **3850** 行，合计约 **1730** 次单人判断——按每条目 2 人计，人均约 **865** 次。每条目耗时以 `duration_ms` 记录，因此标注一开始，真实工作量就是可测数字而不是猜测。
- 运行方式（命令如实记录）：

  ```powershell
  python -m jobrec_eval.annotation_ui build --experiment-dir <out_root>/_runs/<experiment_id> --scenarios ... --catalog ... --annotation-dir ... --raters a,b,c --seed 2026 --oracle-labels <analysis_dir>/normalized/relevance_labels.csv
  python -m jobrec_eval.annotation_ui serve --annotation-dir ...   # 只打印 uvicorn 命令，本身不启动服务
  # 然后在 http://127.0.0.1:8765/ 标注
  python -m jobrec_eval.annotation_ui export --annotation-dir ... --out-dir <scenarios 文件旁边> --release-dir final_release/human_annotations
  ```

- **安全性，直说**：该 UI **没有任何认证**。选择评分者是署名，不是登录。默认绑定 `127.0.0.1`，绑定非回环地址必须显式加 `--allow-remote-host` 才行。它是为“几个人轮流用**同一台**机器”设计的，**不得**暴露到网络上。
- 本项最后一条要求（用**裁定后**标签重算 NDCG@5 / Precision@5 / mean graded relevance，并比较 oracle 与 human）**已实现**：`python -m jobrec_eval.cli pipeline --relevance-source human` 从裁定表重算这三个指标；没有裁定标签时**直接大声失败**，而不是在 human 标题下悄悄报 oracle 数字；写出 `metrics/relevance_source_comparison.csv`；把真实来源与标签文件的 sha256 记入 `manifests/analysis_plan.yaml`；并把报告头、§4、§5.5、§12 的文字改为条件式，因此任何一次运行都不可能在人工标签产出数字时还声称 “no human raters were used”。
- 有意保留的一处不对称：**retrieval recall 在两种模式下都继续使用 oracle 标签**，因为人工判断只覆盖被返回的 pair，换成人工标签会让 recall@pool 平凡地趋近 1.000。
- **端到端验证（真实运行）**：24-run pipeline → 48 relevance / 94 claim 条目 → 合成的双评分者标签 + 裁定 → 导出 → 在**同一批 bundle** 上以 `--relevance-source human` 重跑，`full` 的 NDCG@5 从 0.963 变为 0.759、P@5 从 1.000 变为 0.330，比较 CSV 与 analysis plan 均已填充，`cli verify` 返回 OK。
- 因此本项复选框全部保持未勾选：工具与指标通路都在，缺的是真人坐下来标完这一轮（且须排在第 8 项确定性重跑之后，见第 十 节）。

---

### 11. 完成人工 Explanation Grounding 标注

- [ ] 导出 unique factual claims。
- [ ] 两名评分者独立判断：
  - supported；
  - unsupported。
- [ ] 检查 claim 对应 evidence ID 是否可解析。
- [ ] 计算 Cohen’s kappa。
- [ ] 完成 disagreement adjudication。
- [ ] 重新计算：
  - explanation grounding；
  - unsupported claim rate。
- [ ] 保留人工标注文件、说明和最终标签版本。

**完成标准**

- Chapter 5 中的人类评测结果可复现。
- Chapter 3 中关于两位评分者和 agreement 的承诺得到落实。

**状态：同第 10 项——标注工具已建成并测试通过，未勾选是因为标签尚未采集，阻塞项是人工时间**

- grounding 标注与 relevance 标注共用 `src/jobrec_eval/annotation_ui/` 这一套工具：localhost FastAPI + 服务端渲染界面，底下是同一个 headless 数据层（`store.py` SQLite/WAL、`loader.py`、`assignment.py`、`export.py`、`console.py`、`app.py`、`templating.py`、`views.py`、`templates/`、`static/`）。
- 每条 claim **恰好**由 2 名不同评分者独立判断 supported / unsupported，评分者来自规模为 N 的池，分配带种子且负载均衡（`max_load_imbalance <= 1`），以保证成对 Cohen's kappa 与 `rater_1` / `rater_2` 导出契约成立；3 人以上/条目被有意拒绝（需 Fleiss' kappa 与另一套 CSV，池 < 2 时抛 `InsufficientRaterPoolError`）。
- 盲评与评分者隔离是结构性的：validator 判定与评分者载荷分列存放，评分者侧类型里没有分析侧字段，未盲载荷在写入时被拒；任何评分者都读不到、覆盖不了他人的标签。这是所报 kappa 成立的前提。
- **“检查 claim 对应 evidence ID 是否可解析”已由界面承担**：无法解析的 evidence id 在 grounding 页面上显著标出并打旗标——引用指向空处，就支撑不了任何断言。这条依赖第 15 项新增的 `evidence_items.jsonl`（否则 id 在 bundle 里根本无从解析）。
- claim 条目按内容寻址的 `claim_id` 去重，一次判断覆盖所有产出同一句子的 run；导出时按 `(run_id, claim_id)` 展开回逐行，因此 `claim_annotations_human.csv` 的行数远大于实际判断次数。
- 裁定是带裁定人姓名与**必填**书面理由的记录判定。`adjudicated` 列只从该判定填充；两人一致时留空（共同标签即金标准），分歧未裁定时同样留空并如实报告为 unadjudicated 且**排除**，绝不取平均。`round((rater_1+rater_2)/2)` 仅作带标注的回退供无 `adjudicated` 列的历史文件使用，不参与已发表指标的标签表。
- **实测规模**：12 场景 × 2 变体 → 425 次 claim 出现去重为 94 条（4.5×）；5 变体全量 → 1101 次出现去重为 147 条（7.5×）。外推到 42 场景 × 5 变体约 **515** 条 claim 条目、`claim_annotations_human.csv` 约 **3850** 行；连同 relevance 合计约 1730 次判断、人均约 865 次。每条目 `duration_ms` 落库，真实耗时标注一开始即可测。
- 运行与安全性同第 10 项：`build` → `serve`（只打印 uvicorn 命令）→ 在 `http://127.0.0.1:8765/` 标注 → `export`（可同时落到 `final_release/human_annotations`）。UI **无认证**，选择评分者只是署名；默认绑 `127.0.0.1`，非回环绑定需显式 `--allow-remote-host`；仅供数人轮流使用同一台机器，不得暴露到网络。
- 导出产物即“保留人工标注文件、说明和最终标签版本”所需的归档：`claim_annotations_human.csv` 放到 scenarios 旁边后，kappa 与 validator-vs-human 一致性自动计算，同一份可复制进 `final_release/human_annotations`。

---

## 五、数据库与测试验证

### 12. 运行真实 PostgreSQL Integration Suite

- [x] 启动真实 PostgreSQL（Docker 或 test container）。
- [x] 运行所有 `@pytest.mark.postgres` 测试。
- [x] 测试：
  - CandidateState 保存与恢复；
  - DialogueState 保存与恢复；
  - ~~ActiveSearchState 保存与恢复~~ → **见下方诚实缺口**；
  - RecommendationDecision 保存与恢复；
  - EvidenceLog 保存与恢复；
  - Handoff 保存与恢复；
  - application restart；
  - same-session continuation；
  - evidence link validity；
  - CandidateState version history。
- [x] 正式实验模式下验证 DB unavailable 时 fail fast。
- [x] 正式最终测试中 PostgreSQL 测试不得 skip。

**状态：已完成——真实 PostgreSQL 15.18，`pytest -m postgres` = 5 passed / 0 skipped / 616 deselected**

- 15 张表建成；`schema_version` 行（id=1, version=1）；实测探针 `db_version='PostgreSQL 15.18 (Debian ...)'` 与 `migration_version=1` 落在 run record 上为非空，并能在换一个全新 engine 重新加载后存活。
- R9.1/R9.6 的 fail-fast 位于 `tests/unit/test_db_fail_fast.py`（8 passed，不需要活库）。
- 端口说明：容器挂在宿主 **55432**，不是 5432——5432 被另一个项目的 `tiktok-ai-postgres` 容器占用，该容器**刻意未停**。通过 stdin 传入的 compose override（`ports: !override`）启动，**未改动仓库任何文件**。
- 确切执行方式（Windows）：

  ```powershell
  docker compose -f docker/docker-compose.yml up -d postgres   # 端口冲突时用 override 改宿主端口
  $env:DATABASE_URL = "postgresql+psycopg://jobrec:jobrec@localhost:55432/jobrec"
  .venv\Scripts\pytest.exe -m postgres
  ```

- 收尾：`docker compose -f docker/docker-compose.yml down`（不加 `-v` 则保留 `pgdata` 卷）。
- `make test-pg` 存在，但其依赖的 `scripts/pg_local.sh` 只适用于 Linux，Windows 上请用上面三行。
- **诚实缺口（论文中不得含糊）**：schema 里**根本没有 ActiveSearchState 表**。持久化的只有 decision 上的 `active_search_id` 这一标识，state 本身是每次检索**重新推导**的。run bundle 里确实有 `active_search_state.json`（因此**归档**层面是完整的，见第 15 项），但**数据库**层面“ActiveSearchState 完整持久化”这一说法**不成立**。论文若有此表述必须改写为：active search 的**身份**被持久化，状态按需重建。

---

### 13. 运行全部测试和质量检查

- [x] 运行原有测试。
- [x] 运行所有新增单元测试。
- [x] 运行 property-based tests。
- [x] 运行 integration tests。
- [x] 运行 E2E tests。
- [x] 运行 PostgreSQL tests。
- [x] 运行 performance tests。
- [x] 运行 `ruff`。
- [x] 运行 type checking。
- [x] 生成 coverage report。
- [x] 运行 deterministic smoke evaluation。
- [x] 运行 catalog/scenario data-quality validation。

**完成标准**

- 0 failed。
- 所有 skip 都有明确、合理的说明。
- PostgreSQL 正式验证不能靠 skip。
- 测试数量应报告当前真实值：**683 tests**（不再引用旧报告里的 “68 tests”，也不再引用中途出现过的 “483 / 594 / 599 / 635 / 647 / 661 tests”）。

**状态：已完成，当前真实数字如下（全部为本轮实测）**

- `ruff` 干净：`.venv\Scripts\ruff.exe check src tests scripts` → All checks passed。
- 默认测试套件：**683 passed, 2 skipped, 22 deselected, 0 failed**（`pytest -m "not postgres and not perf"`）。相对上一轮的 661，新增的 22 个来自 `tests/unit/test_model_call_accounting.py`（第 9 项的 LLM 调用记账，全部用伪造 payload + monkeypatch，无真实 API 调用）。
  - 2 个 skip 是 PostgreSQL——**仅因为该次运行的 shell 未设 `DATABASE_URL`**；带库运行时为 5 passed / 0 skipped（见第 12 项）。
  - 22 个 deselected 是 perf 测试（默认运行按设计排除），单独执行：**22 passed**（`pytest tests/perf -q`，即 `make test-perf`）。
- coverage：**92%**（CI 门槛 85%）。
- type check：`.venv\Scripts\mypy.exe src` → **14 errors in 9 files**，与 CI 门槛 `MYPY_MAX_ERRORS: "14"` **相等，因此 CI 的 typecheck job 会通过**。这 14 个是如实保留的余量（6 个 assignment，其余为 `statistics.py` 与 `annotation_ui/views.py` 的 arg-type/union-attr 收窄问题），Windows 与 `--platform linux` 下计数一致。
- **检查器已钉版**：`ruff==0.16.0`、`mypy==2.3.0` 写进 dev extra；`[tool.mypy] python_version = "3.12"` 为单一真源，`ci.yml` 里冗余的 `--python-version 3.12` 已移除。理由：两者都是 CI 门禁，不钉版的检查器会在代码未变的情况下翻转判定，这会直接破坏代码冻结。
- 本轮 mypy 顺带修掉的 3 类**真实缺陷**（不是为过门禁而加 ignore）：5 处无用 ignore 清除；`llm/remote_provider.py` 的 3 处 arg-type——`_scrub(str(exc))` 是一个真实的**脱敏陷阱**；`evaluation/manifest.py` 的 2 处 attr-defined——用 `getattr(os, "sysconf", None)` 兜住，这是真实的 **Windows 可移植性 bug**。新增测试：`test_timeout_errors_are_scrubbed_before_being_raised`，以及 `tests/contract/test_run_manifest.py` 中两个 sysconf 可移植性测试。
- **perf 实测数字**（`artifacts/reports/perf_latency.json`）：e2e 中位数 deterministic 15.3 / 18.3 / 18.6 ms、hybrid 17.1 / 20.1 / 19.8 ms（catalog 100/200/300）；retrieval 2.9 → 4.8 → 6.2 ms；LLM 中位数 deterministic 0.0、hybrid 0.558 / 0.701 / 0.620 ms。
  - **论文必须写明的 caveat（仍然成立）**：perf 套件用的**还是 mock provider**，`provider_reported_llm_ms` 在**所有**单元格（含 hybrid 单元格）都是 0.0，因为 mock 不发生网络往返。所以 perf 表里的 “LLM latency” 是**mock 计算耗时，不是真实 API 成本**。
  - **补充（本轮变化）**：**真实 API 延迟现在已由第 9 项的 hybrid 实验提供**——单次调用中位 **11.6 s** / p95 **20.4 s** / 最大 **57.5 s**。因此论文引用**真实延迟与成本**时应引 hybrid 实验（`exp-06cc34defe39`），引用**编排开销随目录规模的行为**时才引 perf 套件。两者相差三个数量级，混用会直接写出错句。
- 数据质量附注：在 12 场景的 CI subset 运行中，有 **3 个场景报 `missing_hard_constraint_reference`**（随仓库发布的完整场景集是干净的：正式实验 0 error / 2 warning）。值得看一眼，但不构成阻塞项。
- 环境附注：本环境**未安装 `jinja2`**，因此标注 UI 用一个基于标准库的小渲染器（`annotation_ui/templating.py`）渲染模板，**默认自动转义**（比裸 `jinja2.Environment()` 更安全，这也是选择保留它、不引入 jinja2 的原因——复现包保持纯 Python）。日后若换成真正的 Jinja，需要设置 `autoescape=select_autoescape(["html"])`，并重新核对布尔值的拼写形式——前端 `annotate.js` 用 `=== 'true'` 判断。

---

## 六、数据与实验工件检查

### 14. 处理 Data Quality Warnings

- [x] 检查缺少明确 hard-constraint reference 的场景。
- [x] 对需要 hard constraints 的 scenario 补充 reference。
- [x] 检查 no-match 场景定义。
- [x] 明确 no-match 是：
  - 角色匹配 + 硬约束联合不可行；
  - 而不只是单独硬约束不可行。
- [x] 对故意保留的过期职位增加 fixture 标识，例如：
  - `is_test_fixture = true`
  - `expected_ineligible_reason = expired`
- [x] 确保过期职位不会被数据质量检查误判为必须删除。
- [x] 将最终 data-quality report 保存到实验工件。
- [x] **已由用户决定**：SC-E-02 与 SC-E-04 的 `no_match_scenario_constraint_satisfiable` → **保留现状，在论文中写清楚联合不可行的推理**（口径见 `CMJCC_Thesis_Writing_Guide.md` §1）。

**状态：已完成（含定义问题的决定）**

- `JobPosting` 新增 `is_test_fixture` / `expected_ineligible_reason` 注解，并通过 `catalog.FIXTURE_ANNOTATION_FIELDS` 从 `raw_payload_hash` 中排除，因此 `catalog_hash` **可证明未变**（`145dfa05...454509`，已与 `catalog_manifest.json` 核对）。
- 27 条故意过期的职位现在是 `info` 级别的“已确认 fixture”，不再是“要求删除”的 warning；反过来，解释不了任何事情的 marker 会被单独报为 `unsupported_test_fixture_marker`。
- 新增 `missing_hard_constraint_reference` warning：结论依赖硬约束却没有指名任何硬约束的场景会被点出；场景集补上了 `hard_fields` / `blocking`。
- no-match 的定义**本来就是**联合不可行（role-match + hard constraints），本次只是把它写清楚。
- pipeline 现在把 `data_quality_report.json` 一并写入实验工件，并纳入 checksums 覆盖。
- **✅ 已决定（用户拍板）：采用方案 2 —— 保留现状，在论文中把联合不可行的推理写清楚。** 那 2 个 warning 因此是**已知且已接受**的，不是待修缺陷。
  - 理由：这不是数据缺陷，而是 no-match 构念的定义问题——系统实现的定义本来就是“角色匹配 ∧ 硬约束”联合不可行。收紧 turn 文本会改动 `scenarios.jsonl` → scenario hash 变 → experiment id 变 → 已定稿的 `exp-197f6aacc171` **与 `exp-06cc34defe39` 双双作废**（hybrid 重跑还要再花约 135 分钟与约 34 万 tokens）；为一处措辞重跑两个实验不成比例。
  - 已核实的具体事实（供论文直接引用）：**SC-E-02** 有 5 个职位（`job-0021`、`job-0086`、`job-0089`、`job-0094`、`job-0169`）满足全部硬约束但**全部在所请求角色族之外**；**SC-E-04** 有 1 个（`job-0012`）同理。场景自带 `notes` 已写明角色族内的联合不可行性（“no KL onsite BA posting pays >= RM4500” / “no KL hybrid SE posting pays >= RM6000”）。
  - **写作口径见 `CMJCC_Thesis_Writing_Guide.md` §1**，含必须写出的英文定义句与“不要把 warning 说成已修复”的禁例。

---

### 15. 保存完整 Raw Run Bundles

每个正式实验应保存：

- [x] resolved config；
- [x] catalog snapshot；
- [x] scenario snapshot；
- [x] prompt files / prompt hash；
- [x] run manifest；
- [x] model calls（deterministic 下为**空文件**，见下方 caveat）；
- [x] dialogue trace；
- [x] extracted preferences；
- [x] CandidateState versions；
- [x] DialogueState versions；
- [x] ActiveSearchState；
- [x] JobContextState；
- [x] retrieval results；
- [x] RecommendationDecision；
- [x] ranking feature contributions；
- [x] EvidenceLog；
- [x] evidence items（`evidence_items.jsonl`）；
- [x] handoffs；
- [x] final response；
- [x] per-run metrics；
- [x] component latency；
- [x] logs；
- [x] checksums。

**状态：已完成（deterministic 与 hybrid 均已落盘并归档）——`evaluation/outputs/_runs/exp-197f6aacc171/` 下 210 个 bundle、`evaluation/outputs_hybrid/_runs/exp-06cc34defe39/` 下 378 个 bundle，四棵目录树 `cli verify` 全部 OK**

- 逐 bundle 实测清单（以 `full/SC-A-01/0` 为例，全部非空）：`resolved_config.yaml`、`run_manifest.json`、`run_record.json`、`dialogue_trace.jsonl`、`dialogue_state.json`、`extracted_preferences.json`、`candidate_state_before/after.json`、`active_search_state.json`、`job_context_state.json`、`eligibility_results.json`、`retrieval_results.json`、`recommendation_decision.json`、`response.json`、`response_claims.json`、`evidence_log.jsonl`、`evidence_items.jsonl`、`handoffs.jsonl`、`clarification.json`、`component_latency.json`、`log_trace.jsonl`、`input_snapshot.json`。
- **caveat（仍然成立）**：`model_calls.jsonl` 在 deterministic 模式下是 **0 字节**——mock provider 不产生模型调用。本轮重跑后仍然是 210/210 全空。这不是缺陷，但意味着 deterministic bundle 的 replay 重放的是**规则抽取路径**而不是录制的模型输出（见第 16 项）。
- **✅ "model calls"这一条现在真正被满足了（本轮变化）**：hybrid bundle（`exp-06cc34defe39`）的 `model_calls.jsonl` **非空**，378 条记录，每条带**归一化 token usage**（prompt / completion / reasoning，`usage` 缺失时记为缺失而非 0）、**request parameters**（按实际发送的 body 反读，未发送的参数记为缺失）、**retry 痕迹**（`attempts` / `retried_without_response_format` / `retry_reason`）、`finish_reason`、以及**脱敏后的 `raw_response`**（378/378 已写入）。此前 `RemoteLLMProvider` 从不填充 `metadata`，这些字段在产物里**全部不可见**，详见第 9 项的代码改动记录。
- **`prompt` 在任何模式下都不持久化**（deterministic 与 hybrid 皆然）。这是刻意的隐私/体积取舍：`prompt_hash` 记在 `experiment_manifest.json` 与 `run_record.json` 里用于同一性校验，`ReplayProvider` 现在按 `call_id` 命中记录，因此**无需持久化 prompt 也能重放**。论文若讨论产物完整性，应写"prompt 以哈希形式而非原文归档"。
- `save_raw_responses` 关闭时 `raw_response` **键完全缺失**，而不是写成空串——这样"未保留"不会被误读为"模型没返回内容"。
- **✅ 归档副本已完成（deterministic 与 hybrid 均已就位）**：`.gitignore:38` 忽略了 `evaluation/outputs/_runs/`，那些 bundle 原本只存在于本机磁盘、不随仓库分发。现已复制到 `final_release/`，布局刻意与真实 `--out-root` 一致，因此 `cli verify` / `--experiment-dir` 可直接作用于副本：

  ```text
  final_release/
  ├── deterministic_runs/
  │   ├── _runs/exp-197f6aacc171/   4837 files /  77.7 MB   （210 个 run bundles，与源逐文件一致）
  │   └── exp-197f6aacc171/           49 files /   4.4 MB   （分析输出）
  └── hybrid_runs/
      ├── _runs/exp-06cc34defe39/   8701 files / 166.9 MB   （378 个 run bundles，与源逐文件一致）
      └── exp-06cc34defe39/           49 files /   7.1 MB   （分析输出）
  ```

  **四棵副本树的 `cli verify` 全部返回 OK**，源目录亦复验 OK 未受影响。hybrid 体积约为 deterministic 的两倍，主要来自 378 条 `model_calls.jsonl` 记录里的脱敏 `raw_response`。
  - **踩过的坑，记录备查**：bundle 树与分析输出树**同名**（都叫 `<experiment_id>`），第一次复制时直接平铺，把分析目录的 49 个文件并进了 bundle 目录，并让两边的 `checksums.json` 互相覆盖。已清空重做为上面的 `_runs/` + 分析目录两层布局。**日后再复制务必保留这一层级**（hybrid 归档时已按此执行，未再踩坑）。
  - **✅ 分发方式已定（本轮已执行）**：`final_release/` **已加入 `.gitignore`**，不入版本库——它是树内已有产物的臃肿副本（两组合计约 256 MB / 13636 文件），git 不是合适的传输方式。改为随论文附交两个独立压缩包：

    ```text
    dist/CMJCC_deterministic_runs_exp-197f6aacc171.zip    7.2 MB
    sha256 = 8B729C38BB5C98288C85C93D820DD9458A0695039A3EF65BCEAEA2B83C36AD22

    dist/CMJCC_hybrid_runs_exp-06cc34defe39.zip          13.9 MB
    sha256 = 459058878948CD23210373087E501A97B22D197DA3F17E0F3A2CB959DEC455B1
    ```

    压缩包内各自保持 `_runs/<experiment_id>/` 与 `<experiment_id>/` 两层布局，两个 `checksums.json` 均在正确相对路径上，**解压后可直接 `cli verify`**。`dist/` 亦被 gitignore（`.gitignore:7`）。旧的 `CMJCC_deterministic_runs_exp-f90573008bdb.zip`（sha256 `936A490D...961331`）**已删除替换**，任何仍引用该 sha256 的地方都是过期引用。
  - ⚠️ **三份副本的关系**：run bundles 现在只存在于 (1) `evaluation/outputs/_runs/` 与 `evaluation/outputs_hybrid/_runs/`（均被 gitignore）、(2) `final_release/{deterministic,hybrid}_runs/_runs/`（被 gitignore）、(3) 上述两个 zip。**三者都不在版本库里**，删除任何一份前请确认另有留存。

**（原状态记录）写入机制齐备（本轮补上 evidence items）**

- `write_run_bundle()` 已经逐项写出上表内容（`run_record.json`、`dialogue_trace.jsonl`、`candidate_state_before/after.json`、`retrieval_results.json`、`recommendation_decision.json`、`evidence_log.jsonl`、`evidence_items.jsonl`、`handoffs.jsonl`、`model_calls.jsonl`、`component_latency.json`、`run_manifest.json`、`resolved_config.yaml` 等），实验级快照（config / catalog / scenarios）与 `checksums.json` 由 `ExperimentRunner` 写出。
- **本次新增 `evidence_items.jsonl`**：run bundle 现在持久化 claim 所引用的 evidence **条目本身**（来源、来源对象、字段名、原始文本、归一化取值、作用域），这才是让 claim 的 `evidence_ids` 能**离线解析**的东西。此前只保存了逐阶段的决策日志（`evidence_log.jsonl`），从一个 bundle 里**根本无法**把一个 evidence id 解析成具体证据；悬空的 id 现在被如实报告为 unresolvable，而不是静默丢弃。第 11 项 grounding 标注界面上的 evidence 可解析性检查依赖这一条。
- 该新增**未改变任何可复现性哈希**（catalog / scenario / prompt / config hash 均不变）。

---

### 16. 验证 Replay 与 Checksums

- [x] 为全部输入与输出生成统一 `checksums.json`。
- [x] 执行 checksum verify command。
- [x] 故意修改测试文件，确认 checksum 能检测 tampering。
- [x] 执行 artifact replay。
- [x] 重新生成统计和报告。
- [x] 执行 deterministic recomputation。
- [x] 比较：
  - extracted slots hash；
  - state version hash；
  - filtered jobs hash；
  - ranking output hash；
  - explanation claims hash。
- [x] 生成 `replay_diff.json`。
- [x] 最终实验应无非预期 replay 差异。

**状态：已完成——正式产物 `exp-197f6aacc171` 上复验通过**

在正式产物上实测的三件事：

1. **checksums**：两棵目录树 `cli verify` 均返回 OK（`_runs/exp-197f6aacc171` 与 `exp-197f6aacc171`）。hybrid 的两棵树（`outputs_hybrid/_runs/exp-06cc34defe39` 与 `outputs_hybrid/exp-06cc34defe39`）以及 `final_release/` 下的四棵副本树也全部 verify OK。
2. **replay（本轮已在新产物上实测）**：`replay_experiment()` 在 `exp-197f6aacc171` 上重放全部 **210/210 runs，0 differences，0 errors**，`replay_diff.json` 落在 `artifacts/reports/replay_diff_exp-197f6aacc171.json`（刻意不写进被 checksum 覆盖的目录树，以免引入未登记文件）。
   - **caveat**：deterministic bundle 的 `model_calls.jsonl` 为空，所以重放时 `ReplayProvider` 无记录可服务、回落到规则抽取器，五个 key-state 哈希仍逐一相同。因此**这一条 210/210 的证据**证明的是**流水线的确定性**，**不是**“录制的模型输出可回放”——后者由下面的 hybrid replay 单独证明。
   - **✅ hybrid replay 已实测（本轮补做，此前标为“机制就绪、未实测”）**：在 `exp-06cc34defe39` 上跑完整 replay，**378 runs、0 differences**，`replay_diff.json` 落在 `artifacts/reports/replay_diff_exp-06cc34defe39.json`。逐项拆开看：
     - **376 个 run 真正命中了录制的远程响应**并重算出逐一相同的 key-state 哈希——这才是“录制的模型输出可回放”的证据，`ReplayProvider` 按 `call_id` 命中（**无需持久化 prompt**）确实成立。
     - **2 个 run 回落到规则抽取器**，因为它们的 `model_calls.jsonl` **本就为空**（`no_context/SC-D-03/2` 与 `no_memory/SC-A-02/2`，纯规则抽取、没有发生模型调用）。这不是查表失败，是没有录制可服务。
     - **1 个 run 无法重放**：正是那个崩溃的 `full/SC-A-04/1`——它在写出 `candidate_state_before.json` 之前就失败了，bundle 本身不完整，`ReplayInputError` 如实报出而不是被静默跳过。
     - 每 bundle 的调用数分布：**{0 次: 2, 1 次: 374, 2 次: 2}**，合计 378 次，与第 9 项的 token 统计口径一致。
     - 因此现在**可以**写“录制的真实模型输出已验证可回放”，但必须连带说明上面三类例外的确切计数。
3. **从已保存 bundle 重算统计与报告（本轮已在新产物上实测）**：`pipeline --experiment-dir evaluation/outputs/_runs/exp-197f6aacc171 --out-root <scratch>` 复用既有 bundle（**未生成新的 `_runs`**，已确认），产出的 `metrics/*.csv` **全部 17/17 张与正式分析逐字节相同**。重算会回写 run manifest，之后两棵目录树的 `cli verify` **仍然 OK**（即下面那条 checksums 失效缺陷的修复是有效的）。

**跨运行可复现性的两条独立证据**

- **证据 A（历史，措辞已按事实收紧；⚠️ 支撑产物已删除，结论保留）**：曾连续跑了两次完整实验——`exp-515b63d6a656` 与 `exp-f90573008bdb`。**这两份产物都已不再是当前正式产物**：`exp-515b63d6a656` 在早前清理中删除（省 82 MB），`exp-f90573008bdb` 亦已被本轮的 `exp-197f6aacc171` 取代，因此这条比对不能再被复核；结论本身如下记录并保留，且随时可以再跑一次实验重新演示（这是本条证据被判定为“可弃”的原因：它证明的是确定性，而确定性可按需重新演示，不像被覆盖的基线那样不可再生）。两次之间**唯一的代码改动是报告渲染层**（`report.py` 的分母披露与变体覆盖修复），产生指标的代码路径未变（这也是为什么 id 变了：源码指纹覆盖全部 `*.py`，包括只影响渲染的改动）。逐列比对后，两次运行**唯一**的差异是 `run_id` 与 wall-clock 延迟列（`total_latency_ms`、`mean_ms` / `median_ms` / `p95_ms`、`*_retrieval_latency_ms`）；所有指标、统计量、taxonomy 与配对比较**逐字节相同**，`statistics/paired_comparisons.csv` 亦完全一致。因此这条证据支持的结论是：**在固定输入与固定种子下，流水线的数值输出可复现，且报告渲染的改动确实没有改动任何数字**——它不是“完全相同的代码跑两次”的证据。
- **证据 B（本轮新增，✅ 仍可复核——`exp-f90573008bdb` 的两棵目录树目前仍在磁盘上，尚未删除）**：`exp-f90573008bdb` → `exp-197f6aacc171` 之间的代码改动只发生在 **LLM 调用记账**（token usage / request params / retry 痕迹 / `raw_response` 落盘、`ReplayProvider` 索引），逐项比对后**两份 `variant_summary` 的全部指标一致**，error_taxonomy 分布亦不变。因此这条同样支持“指标计算路径未被触及”，而且它比证据 A 更贴题：证据 A 的改动只在渲染层，证据 B 的改动**真的动了 provider 与 bundle 写出路径**，指标却一个都没变。两条证据的共同结论：**id 因源码指纹而变，并不意味着数字会变**。⚠️ 与证据 A 同样的局限：`exp-f90573008bdb` 一端的产物已被替换，因此这条比对**日后也无法重新复核**，只能作为本轮的记录。

**（本轮之前已修的真实缺陷，保留记录）**

- **已修缺陷（本次会话发现并修复）**：`_runs/<experiment_id>/checksums.json` 过去对**每一个** run bundle 都是永久失效的。原因是生成报告会回写 `run_manifest.json` 与 `run_record.json`（盖上 consistency 区块），而 runner 早在之前就已对这两个文件算过哈希——24-run 实验上表现为 **48 处 mismatch**，因此在 run-bundle 目录树上执行 `cli verify` **永远不可能通过**。现已通过在外层 manifest 中**精确重盖这两类条目**修复，并有测试断言：完整跑完一次 pipeline 之后，`cli verify <out_root>/_runs/<experiment_id>` 返回 0。
- 现在 **run-bundle 目录树与分析输出目录两者的校验都返回 OK**。
- `cli verify` 返回 OK；replay 在先前一次检查中复现了 **36/36 runs，0 differences**，`replay_diff.json` 正常产出。
- tampering 检测由 `tests/eval/test_checksums.py` 覆盖（modified / deleted / added 三种漂移，含 property test）。
- 上述两项此前未勾是因为要等修复后的正式实验；**已在 `exp-197f6aacc171` 上补齐**（见上方三条实测）。hybrid 产物的 **checksums 与 replay 现在也都已实测**（378 runs / 0 differences，见上方第 2 条）。

---

## 七、代码冻结和最终归档

### 17. 执行正式 Code Freeze

- [ ] 确认所有代码、测试和实验均完成。
- [ ] 创建 annotated Git tag。
- [ ] 保存 commit hash。
- [ ] 生成 dependency lock。
- [ ] 保存数据库 schema。
- [ ] 保存 migration version。
- [ ] 保存运行说明。
- [ ] 生成 final manifest。
- [ ] 确认 manifest 引用：
  - frozen commit；
  - dependency lock；
  - catalog hash；
  - scenario hash；
  - prompt hash；
  - config hash。
- [ ] 归档完整 reproduction package。

**建议目录**

```text
final_release/
├── source/
├── config/
├── catalog/
├── scenarios/
├── prompts/
├── deterministic_runs/
├── hybrid_runs/
├── human_annotations/
├── reports/
├── figures/
├── database_schema/
├── checksums.json
├── requirements.lock.txt
├── RUN_INSTRUCTIONS.md
└── final_manifest.json
```

**状态：未开始——现在只被第 10/11 项（人工标注）阻塞，另需先处理第 9 项末尾那个待决策的缺陷**

- deterministic 主实验（第 8 项）与 hybrid 补充鲁棒性验证（第 9 项）均已完成，因此 `deterministic_runs/`、`hybrid_runs/`、`reports/`、`figures/`、`checksums.json` 的内容已经**存在且可归档**；`database_schema/` 与 `migration_version` 也已在第 12 项中确认（15 张表、version=1）。
- 仍缺的只有 `human_annotations/`（第 10/11 项）。
- **✅ `deterministic_runs/` 与 `hybrid_runs/` 均已就位**：

  ```text
  final_release/deterministic_runs/_runs/exp-197f6aacc171/   4837 files /  77.7 MB
  final_release/deterministic_runs/exp-197f6aacc171/           49 files /   4.4 MB
  final_release/hybrid_runs/_runs/exp-06cc34defe39/          8701 files / 166.9 MB
  final_release/hybrid_runs/exp-06cc34defe39/                  49 files /   7.1 MB
  ```

  **四棵树 `cli verify` 全部 OK**，因此复现包已能支持 `cli verify`、replay 与 `pipeline --experiment-dir`。详见第 15 项。
- **分发形态已定**：`final_release/` 与 `dist/` 均被 gitignore，复现包以两个压缩包随论文附交，`final_manifest.json` 应同时引用这两个 sha256，使副本可校验：

  ```text
  dist/CMJCC_deterministic_runs_exp-197f6aacc171.zip    7.2 MB
  sha256 = 8B729C38BB5C98288C85C93D820DD9458A0695039A3EF65BCEAEA2B83C36AD22

  dist/CMJCC_hybrid_runs_exp-06cc34defe39.zip          13.9 MB
  sha256 = 459058878948CD23210373087E501A97B22D197DA3F17E0F3A2CB959DEC455B1
  ```

  旧的 `CMJCC_deterministic_runs_exp-f90573008bdb.zip` 及其 sha256 `936A490D...961331` **已删除替换**，不得再引用。
- **`final_manifest.json` 可以用一个 `source_fingerprint` 同时覆盖两组产物**（均为 `8eba8f8106dc...`）——这是本轮刻意重跑 deterministic 换来的一致性，也是不要轻易再动代码的理由。
- 归档时 `reports/` 与 `figures/` 可直接取自 `exp-197f6aacc171/report/` 与 `exp-197f6aacc171/plots/`（12 张图），hybrid 的对应目录在 `exp-06cc34defe39/` 下；`database_schema/` 与 migration version 取自第 12 项确认的结果（15 张表、version=1）。
- **冻结时必须注意**：experiment id 现在把源码指纹计入哈希，所以**冻结前的任何一次代码改动都会让两个 id 同时变化**，并破坏当前"两组产物同指纹"的一致性。归档时 `final_manifest.json` 引用的 id 必须与实际归档的目录名一致，且应同时记录 `commit_hash`（`9768116417...`）、`git_dirty`（当前为 **true**）与 `source_fingerprint`（`8eba8f8106dc...`）。**冻结前先把工作树提交干净。**
- 这条注意事项直接关系到第 9 项末尾那个**待用户决定**的缺陷：若决定修复并重跑 hybrid，则必须在 code freeze **之前**完成，且需要同时考虑 deterministic 是否也一起重跑以保持同指纹。
- 检查器版本已钉（`ruff==0.16.0`、`mypy==2.3.0`），这是让冻结后的门禁判定可复现的前提。

---

## 八、论文章节更新清单

**状态：整节未开始。Chapter 3–6 的绝大部分现在都可以写了——deterministic 主实验（`exp-197f6aacc171`）与 hybrid 补充鲁棒性验证（`exp-06cc34defe39`）都已定稿**

- 可以现在写：Chapter 3（全部）、Chapter 4（全部）、Chapter 5 的 dataset / scenario taxonomy / variants / metrics / statistical method / overall results / memory 与 context 子集 / clarification 子集 / fault-injection / error analysis / framework deltas、**deterministic vs hybrid 配置对比**、**Latency 的真实 LLM 成本**；Chapter 6 的 **“deterministic 与 hybrid 的差异”与 failure cases**。
- 必须等第 10/11 项（人工标注）：Chapter 5 的 “Human annotation” 与 “Human-vs-oracle comparison”。
- **写 hybrid 那部分前先读写作指导 §5（延迟与成本两个数据源）与 §6（deterministic/hybrid 关系）**：两者不能混表平均，hybrid 没有 `profile_only`/`one_shot` 对照，`no_context` 在 hybrid 下 ndcg/hcsr 略高不得写成机制结论，那个 `AttributeError` 崩溃不得写成"LLM 的错"或"已修复"。
- **写 Chapter 5 时的三条硬约束**（否则会写出站不住的句子）：(1) 排序类指标必须连 `_n` 一起给，跨变体比较只用 `task_success`；(2) `full vs one_shot` 的 NDCG 近零 Δ 写作“在其放弃的场景上不可估计”，不写“无差异”；(3) grounding = 1.000 与 handoff = 1.000 要按报告 §10 的说法解释（well-formed 输入下由构造保证，robustness 证据来自 fault-injection 套件），不可暗示为对抗性输入下的表现。

### Chapter 3 — Research Methodology

- [ ] 将 future tense 改为实际完成时。
- [ ] 更新实际技术栈：
  - Python；
  - PostgreSQL；
  - SQLAlchemy；
  - FastAPI；
  - CLI；
  - deterministic / hybrid / replay。
- [ ] 准确描述 CandidateState 长期写回。
- [ ] 准确描述 clarification loop。
- [ ] 准确描述 scenario-level statistics。
- [ ] 准确描述消融配置一致性。
- [ ] 加入人工评分流程。
- [ ] 加入 fault-injection evaluation。
- [ ] 加入 reproducibility 和 code freeze 流程。

> 注：scenario-level statistics 与消融配置一致性的**代码与报告文字**已经就位（第 4、7 项），Chapter 3 只需与之对齐。

---

### Chapter 4 — Framework Design and Implementation

- [ ] Framework architecture overview。
- [ ] Agent/component responsibilities。
- [ ] Typed state model。
- [ ] Candidate-Memory and Job-Context Connector。
- [ ] CandidateState versioned memory。
- [ ] Conflict and scope resolution。
- [ ] Hard-filter-before-rank。
- [ ] Retrieval and ranking。
- [ ] Clarification loop。
- [ ] Evidence-bound explanation。
- [ ] Agent handoff contracts。
- [ ] PostgreSQL persistence。
- [ ] API、CLI 和 experiment runner。
- [ ] Reproducibility manifest、checksums 和 replay。

---

### Chapter 5 — Evaluation and Results

- [ ] Dataset/catalog description。
- [ ] Scenario taxonomy。
- [ ] Experimental variants。
- [ ] Deterministic and hybrid configuration。
- [ ] Metrics definitions。
- [ ] Human annotation。
- [ ] Statistical method。
- [ ] Overall results。
- [ ] Memory-dependent subset。
- [ ] Context-dependent subset。
- [ ] Clarification-dependent subset。
- [ ] Fault-injection robustness。
- [ ] Retrieval vs ranking error。
- [ ] Latency。
- [ ] Human-vs-oracle comparison。
- [ ] Error analysis。
- [ ] Framework mechanism deltas。

---

### Chapter 6 — Discussion

- [ ] 解释 Candidate Memory 的贡献。
- [ ] 解释 Job-Context Orchestration 的贡献。
- [ ] 解释 Clarification Orchestration 的贡献。
- [ ] 解释 Evidence Grounding 和 Handoff 的作用。
- [ ] 区分 architecture mechanism 与 prototype implementation。
- [ ] 讨论 deterministic 与 hybrid 的差异。
- [ ] 讨论 failure cases。
- [ ] 讨论内部效度和外部效度。
- [ ] 避免将结果扩大为对所有外部框架的全面优越性证明。

---

### Chapter 7 — Conclusion

- [ ] 回答 RQ1–RQ4。
- [ ] 总结框架贡献。
- [ ] 总结 memory/context 消融证据。
- [ ] 总结 inspectability 和 reproducibility。
- [ ] 说明研究边界。
- [ ] 说明真实用户、跨行业和跨平台验证属于未来工作。

---

## 九、最终可定稿标准

满足以下条件后，可以正式定稿所有后续章节：

- [x] Clarification task-success 计分已修复。
- [x] Clarification metrics 已接入完整 dialogue trace。
- [x] 报告使用 scenario-level statistical wording。
- [x] Failure-path metrics 已生成。
- [x] `Δmemory` 和 `Δcontext` 已生成。
- [x] 配置一致性检查通过。
- [x] Deterministic 最终实验已重跑。
- [x] Hybrid 最终验证已重跑。
- [ ] 人工 relevance 标注完成。
- [ ] 人工 grounding 标注完成。
- [x] PostgreSQL integration tests 真实通过。
- [x] 全部原有和新增测试通过。
- [x] Data-quality warnings 已处理或解释。
- [x] Raw run bundles 已完整保存（deterministic 210 个 + hybrid 378 个，均已归档并 verify OK）。
- [x] Checksums 和 replay 已验证。
- [ ] Code freeze 已执行。
- [ ] Chapter 3 已按实际实现更新。

**状态说明（避免误读）**

- “全部原有和新增测试通过”：**683 passed / 2 skipped / 22 deselected / 0 failed**，coverage **92%**，`ruff` 干净，mypy **14 errors = CI 门槛 14（通过）**，perf **22 passed**，PostgreSQL 带库 **5 passed / 0 skipped**。默认运行里那 2 个 skip 只是该 shell 未设 `DATABASE_URL`，不是能力缺失。
- “Data-quality warnings 已处理或解释”：正式实验 **0 error / 2 warning / 27 条已确认 fixture**。2 个 warning 就是 SC-E-02 / SC-E-04 的 `no_match_scenario_constraint_satisfiable`——**用户已拍板：保留现状，在论文中写清楚**（见第 14 项与写作指导 §1），因此属“已解释”而非“待处理”。CI subset 里那 3 个 `missing_hard_constraint_reference` 不出现在完整场景集上。
- 仍未勾的四项各自的真实阻塞：**两项人工标注缺人工时间**（工具与指标通路均已就绪，且必须针对已完成的 `exp-197f6aacc171` 结果采集）；**code freeze** 与 **Chapter 3 改写**排在最后。外部 API 凭证这一阻塞已解除（第 9 项已完成）。
- **本清单有且仅有一项需要用户拍板**：第 9 项末尾那个 `AttributeError` 缺陷是否修复并重跑 hybrid（代价：约 135 分钟 + 约 34 万 tokens + 破坏两组产物同指纹，收益：full task_success +0.008）。其余剩余项全是人工时间与收尾写作。
- “人工 relevance 标注完成”“人工 grounding 标注完成”：**标注工具（`src/jobrec_eval/annotation_ui/`）与人工标签指标通路（`pipeline --relevance-source human`）都已建成、测试通过并做过端到端验证**，剩下的纯粹是人工判断本身——按实测外推约 350 条 relevance 条目、约 515 条 claim 条目、合计约 1730 次判断（每条目 2 名评分者，人均约 865 次）。第 8 项已完成，因此**现在就可以开始标**，目标产物是 `exp-197f6aacc171`。
- “Raw run bundles 已完整保存”“Checksums 和 replay 已验证”：**已在 `exp-197f6aacc171` 的 210 个 bundle 上完成**（verify OK ×2、replay 210/210 零差异、从 bundle 重算的 17 张指标表逐字节一致）。hybrid 的 378 个 bundle 已落盘、已归档、checksums verify OK，**replay 亦已实测**（378 runs / 0 differences，其中 376 个命中录制的远程响应，见第 16 项）。
- Chapter 3 的 scenario-level 统计与消融一致性描述已有可对齐的代码与报告文字，属于纯写作工作。

---

## 十、最小优先执行顺序

1. [x] 修复 clarification task-success。
2. [x] 修复 clarification precision/recall。
3. [x] 接入 clarification efficiency。
4. [x] 修正 scenario-level 报告描述。
5. [x] 接入 framework deltas。
6. [x] 接入配置一致性检查。
7. [x] 重跑 deterministic 实验。→ `exp-197f6aacc171`，210 runs，0 system_error，0 invalid_runs。
8. [ ] 完成人工标注——**工具侧已不再是阻塞项**：标注 UI（`python -m jobrec_eval.annotation_ui build / serve / export`）与人工标签指标通路均已就绪，本步就是标注这一趟本身；`--experiment-dir` 请指向 `evaluation/outputs/_runs/exp-197f6aacc171`。
9. [x] 重跑 hybrid 实验。→ `exp-06cc34defe39`，378 runs（3 变体 × 42 场景 × 3 重复），134.7 分钟，0 system_error，`invalid_runs = 1`（那个待决策的缺陷）。与主实验**同 `source_fingerprint`**。
10. [x] 运行 PostgreSQL 和完整测试套件（含 perf 与 mypy）。
11. [x] 验证 checksums 与 replay（已在 `exp-197f6aacc171` 产物上复验：verify OK ×2、replay 210/210 零差异、从 bundle 重算的 17 张指标表逐字节一致；hybrid 两棵树 checksums 亦 OK，replay 未实测）。
12. [ ] 执行 code freeze。
13. [ ] 定稿 Chapter 5–7。

> 第 1–7、9、10、11 步已完成（`ruff` 干净、683 passed / 2 skipped / 22 deselected、coverage 92%、mypy 14 = 门槛、perf 22 passed、PostgreSQL 5 passed / 0 skipped）。**下一个动作是第 8 步（人工标注）——它现在是唯一的实质阻塞，缺的是人工时间**，目标产物已确定为 `exp-197f6aacc171`。与此并行需要用户就第 9 项末尾那个 `AttributeError` 缺陷拍板（修并重跑 hybrid，还是如实记录为已知限制），因为该决定会影响 code freeze 的时点与两组产物的同指纹一致性。

---

## 最终判断

代码层面的评测修正**已经完成**：clarification 计分与 clarification 指标现在基于完整 dialogue trace，clarification efficiency 与 failure-path 指标已真正接入流水线，统计表述已改为 scenario-level 并附配对来源表，framework deltas 以独立的 secondary family 呈现，消融配置一致性在报告写出前强制校验，数据质量注解在不改变 `catalog_hash` 的前提下落地。

**人工评测一侧现在同样已经完成机器部分**：标注工具（`src/jobrec_eval/annotation_ui/`，localhost FastAPI + 服务端渲染 UI + headless 数据层）已建成并测试通过，结构性盲评、评分者隔离、每条目恰好 2 名评分者的种子化负载均衡分配、claim 内容寻址去重、带必填理由的裁定记录均已落地；人工标签的指标通路（`pipeline --relevance-source human`）也已实现——从裁定表重算 NDCG@5 / P@5 / mean graded relevance，无裁定标签时大声失败，产出 `metrics/relevance_source_comparison.csv`，把来源与标签文件 sha256 记入 `manifests/analysis_plan.yaml`，报告文字条件化，retrieval recall 有意继续用 oracle 标签。端到端验证已在真实的 24-run 产物上跑通（`full` 的 NDCG@5 0.963 → 0.759、P@5 1.000 → 0.330）。此外，run bundle 现在持久化 `evidence_items.jsonl`（evidence id 因此可离线解析），run-bundle 目录树的 checksums 失效缺陷已修复，两棵目录树的 `cli verify` 都返回 OK。

**实验与验证一侧也已完成**：deterministic 主实验已重跑定稿（`exp-197f6aacc171`，210 runs，0 system_error，0 invalid_runs），并在此之前先修掉四件会让结论站不住的事——`one_shot` 与 `no_memory` 曾行为完全相同（死 flag）、experiment id 跨代码版本碰撞并静默覆盖产物、`clarification_efficiency` 不惩罚“问了却放弃”、error taxonomy 类别名张冠李戴。真实 PostgreSQL 套件已真跑通过（5 passed / 0 skipped），perf（22 passed）与 mypy（14 errors = CI 门槛）已补齐，replay 在正式产物上 210/210 零差异，从已保存 bundle 重算的 17 张指标表与正式分析逐字节相同，两棵目录树 `cli verify` 均 OK。报告本身也补上了分母披露：§1 headline、§5 逐指标分母表与警示段、§5.x 的配对数、§5.2 的 `n=applicable` 与 `unk`、§7 的 `n` 释义，并修掉两处静默丢变体的渲染缺陷。

**本轮还补齐了 LLM 调用记账并跑完了 hybrid 补充鲁棒性验证**：`RemoteLLMProvider` 此前从不填充 `metadata`（token usage / request params / retry 痕迹全部丢失）、`save_raw_responses` 对 run bundle 无效、`ReplayProvider` 的索引键在 hybrid bundle 上必然失配——这三处补齐之后，两个实验从**同一份代码**重跑，因此 `exp-197f6aacc171` 与 `exp-06cc34defe39` 共享同一个 `source_fingerprint`。hybrid 实测 378 runs / 134.7 分钟 / 341,074 tokens（其中 reasoning 52,333）/ 单次调用中位 11.6 s、p95 20.4 s，fallback 与 retry 均为 0；full 的 task_success 0.9206 低于 deterministic 的 1.0000，主因是 SC-G-01 / SC-G-02 两个澄清场景在真实 LLM 下 3/3 系统性失败。同时新发现一个真实缺陷（真实 LLM 返回列表包裹值 → `validate_field` 判 ok=True 却保留列表 → `canonical_role` 对 list 调 `.strip()` 崩溃），影响面 1/378、零数据污染、只占 hybrid–deterministic 差距的约 10%，**已定位、已量化、尚未修复，是否修复并重跑仍待用户拍板**。

剩余路径已经不再是代码问题，只剩人工时间、一个待拍板的决定与最后的收尾：

```text
（已完成）重跑 deterministic 主实验 → exp-197f6aacc171
（已完成）补齐 LLM 调用记账并重跑 hybrid 补充鲁棒性验证 → exp-06cc34defe39（同指纹）
（已完成）真实 PostgreSQL integration suite
（已完成）在两组正式产物上复验 checksums 与 replay（deterministic 210/210、hybrid 378 runs 0 differences）
（已完成）两组产物归档 + 两个 zip 打包
→ 在标注 UI 中采集人工 relevance / grounding 标签（缺人工时间；目标产物 exp-197f6aacc171）
⏸ 待用户决定：那个 AttributeError 缺陷是否修复并重跑 hybrid
→ 执行 code freeze（冻结前先把工作树提交干净）
→ 定稿 Chapter 5–7
```

Chapter 3、4 与 Chapter 5 中不依赖人工标注的部分（含 deterministic vs hybrid 对比与真实延迟/成本）**现在就可以动笔**。

完成本清单后，可形成完整链路：

```text
Framework Design
→ Prototype Instantiation
→ Controlled Ablation
→ Human and Automatic Evaluation
→ Statistical Analysis
→ Reproducible Final Results
→ Chapter 5–7 Finalisation
```

**引用纪律（最后再强调一次）**：可引用的正式产物**只有两个**——**主实验 `exp-197f6aacc171`**（deterministic，五变体，210 runs）与**补充鲁棒性验证 `exp-06cc34defe39`**（hybrid，真实 LLM，378 runs），两者 `source_fingerprint` 相同（`8eba8f8106dc...`）。中间/历史产物 `exp-301060a1899d`、`exp-515b63d6a656`、**`exp-f90573008bdb`**（已被 `exp-197f6aacc171` 取代）与修复前产物 `evaluation/outputs/exp-8793b18de5b2/`、`test_results/`（含其中的 `analysis_report.md` 与 hybrid 报告）**都不得引用，也不得手工改写**——它们仍带修复前的标签、旧的 “run-level discordant pairs” 表述与旧的 “68 tests” 说法。要用就整体重跑替换，否则只能另附 erratum。同理，任何早前引用过 `exp-301060a1899d` 或 `exp-f90573008bdb` 的文档都需要改指到 `exp-197f6aacc171`（deterministic 结论）或 `exp-06cc34defe39`（hybrid 结论）。

`exp-f90573008bdb` 唯一仍可提及的场合是**第 16 项的可复现性证据 B**：它与 `exp-197f6aacc171` 的 `variant_summary` 逐项一致，这本身就是"LLM 记账改动未触及指标计算路径"的证据。除此之外，它不得作为任何结果数字的来源。
