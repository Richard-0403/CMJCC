# CMJCC 项目代码全面审计与“一次正式重跑”方案

审计对象：`CMJCC-main (2)(2).zip`  
配套论文：`PD_draft_Ruizhuo_v13.docx`  
目标：先消除会使正式实验失效的系统性问题；所有门禁通过后，只执行一次新的正式主实验，随后以结果替换和解释更新为主。

## 1. 结论先行

当前代码**不建议直接开始正式重跑**。项目不是要全部推翻：场景、实验框架、统计分析、发布校验和论文主体结构都可以保留；但有六类 P0 问题会改变状态、候选集、排序、解释或评价结论，必须先修。

最关键的实测证据：

- 对 42 个正式场景用实际 `ExperimentRunner` 做了一次 deterministic/full 审计运行，0 崩溃，但最终状态有 **12 个场景不符合声明 reference**：10 个 hard-field 不一致，另有 2 个 work-mode 值漏抽取。
- 当前 claim validator 只验证 evidence ID 是否存在，并不验证命题是否由证据推出。在 11,197 条 claim 中，程序判定 11,197 条全部 supported，而人工标注有 **2,349 条 unsupported**。
- 薪资阈值的实现把“岗位薪资区间跨过候选人底线”判为满足；人工 rubric 要求的是“岗位保证的最低薪资不低于候选人底线”。现有人工与 oracle 分歧中，有 43 个 salary-crossing pair，其中 39 个同时是“人工比 oracle 更严格”。
- 多轮对话会重新解析旧 utterance，丢失原始抽取结果及强弱/provenance，并把旧证据重新标成当前 turn；`DialogueTurn.evidence_ids` 目前没有被填充。
- no-match 原因只在 retrieval pool 内计数，没有记录“全目录→角色范围→硬约束→最终候选”的完整因果轨迹。
- 正式实验 ID 没有包含 catalog、prompt、实际远端 model/base URL；相同 ID 可能对应不同实际实验。

所以合理承诺是：**不能保证导师“绝对一次过”，但可以把昂贵的 588 次正式运行控制为一次**。规则是：任何 P0 门禁未通过都不启动正式实验；先用自动测试、42 场景本地审计和小型 Hybrid pilot 把问题挡在正式运行之前。

## 2. 哪些保留，哪些重做

| 组件 | 结论 | 处理 |
|---|---|---|
| 42 个正式场景及其人工声明 reference | 可保留 | 冻结为权威输入，补完整性校验 |
| 200 条 synthetic job catalog | 可保留 | 冻结并纳入实验 identity/hash |
| 实验矩阵 | 可保留 | Deterministic 210 + Hybrid 378 = 588 |
| scenario-level 统计、paired bootstrap、McNemar、Holm | 未发现致命实现问题 | 保留；重算新数据 |
| relevance 双人标注表结构 | 可保留 | 稳定 `(scenario_id, job_id)` 可复用，新增 pair 增量标注 |
| CandidateUnderstanding/状态合并 | 必须修 | 字段局部 cue、替代值、多轮事件、value-level strength |
| 薪资 eligibility/ranking/oracle | 必须统一 | 采用 guaranteed-minimum 语义 |
| claim grounding/validator | 必须重构 | 结构可追踪与语义支持分开 |
| no-match 诊断 | 必须增强 | 保存完整阶段过滤 trace |
| oracle | 必须升版 | 按人工 rubric 对齐为 oracle v4 |
| 实验 identity/provenance | 必须补齐 | catalog/prompt/model/endpoint 等入 hash |
| 旧实验及旧论文 | 不删除、不覆盖 | 标为 `v3/pre_fix`；新结果为 `v4/post_fix` |

论文不会“全部推翻”。Introduction、related work、总体方法框架和统计设计大体保留；方法细节、评价定义和第 5 章所有数字必须同步更新。预计代码改动集中在约 15–25 个源码/测试/脚本文件，论文需要实质修改约 8–12 个小节，但不是从零重写。

## 3. P0 修复清单：文件、错误和验收条件

### P0-1：约束抽取、强弱绑定和多值表达

涉及：

- `src/jobrec/agents/candidate_understanding.py`
- `src/jobrec/domain/extraction.py`
- `src/jobrec/domain/job.py`
- 对应 unit/contract/scenario tests

现状：

- `_clause(text, start, end)` 截止到实体末尾，读不到实体后的 cue，因此 `Hybrid only.`、`onsite only.` 被判成 SOFT。
- `_strength_for` 扫描过大的前置文本；前面薪资的 “at least” 会污染后面 location 或 work mode 的强弱。
- work-mode 提取循环遇到第一个值就 `break`，`remote or hybrid` 只保留 remote。

实际 42 场景不一致：

- 漏值：`SC-A-03`、`SC-D-08` 只得到 remote，声明为 remote + hybrid。
- 少 hard：`SC-D-02` 少 work_modes。
- 多 hard：`SC-B-04`、`SC-D-09`、`SC-D-11` 多 preferred_locations；`SC-D-10`、`SC-D-12` 多 work_modes；`SC-E-01`、`SC-E-03`、`SC-H-01`、`SC-H-03` 多 target_roles。

修改方案：

1. cue 与最近的字段/value span 做局部绑定，同时支持前置和后置 cue；不要扫描整个 clause。
2. 对 `only / must / at least / minimum / required / cannot` 与 `prefer / ideally / fine / acceptable` 建立明确、可测试的优先级。
3. 支持 `or / either / also / as well` 的多值集合。
4. 引入显式操作：`add | replace | remove | relax | confirm`。
5. strength 放到**每个值**上，而不只放到字段上；旧的 `hard_constraint_fields`/`soft_preference_fields` 只作为派生兼容字段。
6. 不允许针对 `SC-D-02` 或任一 scenario ID 写特例。

验收：

- 42/42 正式场景在 full deterministic 下的最终 values、hard fields、soft fields 与 declared reference 相符。
- 另建至少 24 条未进入正式场景的 paraphrase/边界测试，覆盖前置 cue、后置 cue、多个字段同句、多值、否定、放宽和替换。
- 在运行时代码中搜索不到任何正式 scenario ID。

### P0-2：多轮状态与 evidence provenance

涉及：

- `src/jobrec/orchestration/orchestrator.py`
- `src/jobrec/cmjcc.py`
- `src/jobrec/agents/memory_agent.py`
- `src/jobrec/domain/dialogue.py`
- `src/jobrec/storage/repositories.py`

现状：

- `_merge_prior_dialogue` 每一轮重新解析全部旧 utterance；Hybrid 的原始模型抽取可能被规则抽取替换。
- `build_dialogue_evidence` 对合并后的新旧抽取统一使用当前 `turn_id`，导致旧证据被复制并错误归属当前轮。
- `DialogueTurn.evidence_ids` 为空。
- 服务重启后 repository 没有完整 evidence/state rehydration 接口；当前“重解析旧文本”掩盖了这个缺陷。

修改方案：

1. 每轮只抽取当前 utterance，并保存不可变的 `PreferenceEvent`：字段、值、强弱、置信度、confirmation、operation、evidence_id、原始 turn_id。
2. ActiveSearchState 由 typed event history reducer 构建；永不重解析历史自然语言。
3. `DialogueTurn.evidence_ids` 写入本轮真实 evidence。
4. repository 增加 `get_session_evidence`/event-history 读取，支持进程重启后的确定性恢复。
5. 明确冲突和 revision 规则：只有明确的 `relax` 才能让 HARD 降为 SOFT；`add` 新允许值不应静默改变旧值强弱。

验收：

- 同一条 evidence 的 ID、turn_id、source span 在后续轮次保持不变，不能生成语义相同但 turn 错误的副本。
- 进程内连续运行和“每轮后重启服务再继续”得到完全相同的最终状态、决策和 evidence graph。
- SC-D-06 等多轮场景中每个 `DialogueTurn.evidence_ids` 非空且只指向本轮证据。
- deterministic replay 输出字节级一致或通过项目声明的 canonical-normalization 后一致。

### P0-3：薪资语义统一

涉及：

- `src/jobrec/agents/job_context_agent.py::_check_salary`
- `src/jobrec/ranking/features.py::salary_preference`
- `src/jobrec_eval/relevance.py`
- oracle/reference 生成代码与配置

推荐正式规则：

> 候选人表达 “at least RM X/month” 时，岗位只有在 `job.salary_min_monthly_myr >= X` 时才满足；区间上限跨过 X 不算保证满足。缺失岗位最低薪资为 UNKNOWN，不能用最高薪资代替。

修改方案：

1. 把 `salary_threshold_policy: guaranteed_minimum` 写入类型化配置和版本化规则说明。
2. eligibility、ranking feature、oracle、claim 文本共用同一**规则规范**，但 oracle 保持独立实现/测试，避免 production bug 与 oracle 同源复制。
3. salary period/currency 不确定时返回 UNKNOWN；若该条件为 HARD，则按明确的 unknown policy 触发 clarification 或 fail-closed。
4. 删除/替换 `salary_range_crosses_min` 的 PASS 语义。
5. 升级 scorer/oracle version；旧 v3 不覆盖。

验收：

- 边界测试：岗位最低薪资 `X-1` 失败、`X` 通过、`X+1` 通过、`[X-500, X+1000]` 失败、缺失最低值为 UNKNOWN。
- eligibility、ranking reason、oracle grade、explanation claim 对同一 pair 不出现相互矛盾。
- 所有 43 个旧 salary-crossing pair 生成可审阅 before/after diff。

### P0-4：claim 的语义 grounding

涉及：

- `src/jobrec/domain/recommendation.py`
- `src/jobrec/agents/explanation_agent.py`
- `src/jobrec_eval/metrics.py`
- `src/jobrec_eval/metrics_extra.py`
- annotation/export scripts

现状：

- `ResponseClaim.support_status` 默认是 `"supported"`。
- `validate_claims` 只验证 evidence ID 非空且可解析。
- 旧人工标注的 2,349 条 unsupported 主要来自：
  - 1,883 条 skill-gap：只引用岗位 required skills，没有候选人能力/当前档案缺失证据；
  - 247 条 “Salary meets minimum”：岗位最低薪资低于候选人底线；
  - 156 条 no-match：只引用约束值，没有该约束实际过滤候选的因果证据；
  - 35 条使用未确认 evidence；
  - 28 条使用冲突值。

修改方案：

1. `support_status` 不得正向默认；初始为 UNKNOWN。
2. 分开两个维度：
   - `trace_status`：evidence IDs 是否存在且类型正确；
   - `semantic_status`：证据是否推出该原子命题。
3. claim 使用类型化原子 predicate，例如：
   - `candidate_preference(field, value)`
   - `ranking_match(job_id, field, value)`
   - `skill_not_recorded(job_id, skill, profile_snapshot_id)`
   - `constraint_blocked(field, blocked_count, pool_size)`
4. assertive claim 不能使用 unconfirmed/conflicting evidence。
5. skill-gap 改为“该技能为岗位要求，但当前档案未记录”，并同时引用岗位技能和候选人技能快照；不能写成“你不会/缺乏该技能”。
6. no-match claim 只允许引用 `decision.no_match_reason_codes` 中真实造成过滤的字段及计数。
7. 旧指标重命名为 `evidence_resolution_rate`；新增 semantic claim support/precision、unsupported recall/F1、abstention rate，并按 claim type 分层。
8. task success 不应只要求 `grounded_claim_count > 0`；建议要求“无已交付 semantic-unsupported assertive claim，且至少有一条 supported rationale”。

验收：

- 用旧 11,197 条 claim 作为回归集，validator 不再 constant-positive。
- 上述五类已知失败均能返回 UNSUPPORTED 或 ABSTAIN，而不是 SUPPORTED。
- 构造 wrong-field、conflict、unconfirmed、缺少 candidate-side evidence、缺少 causal trace 测试。
- 对新运行报告 confusion matrix 和按 claim-type 指标；不能只报告覆盖率。

### P0-5：no-match 因果、role scope 与 UNKNOWN

涉及：

- `src/jobrec/agents/job_context_agent.py::diagnose_no_match`
- `RecommendationDecision`
- orchestration/explanation/metrics

修改方案：

1. 保存完整阶段 trace：
   `catalog → retrieved/role-scoped pool → hard-filtered pool → eligible → ranked`。
2. 每阶段记录输入 job IDs、输出 job IDs、各字段排除数和 reason code。
3. `retrieved_job_ids` 在 fallback/full-catalog 情况也必须记录实际评估集合。
4. no-match 文案必须限定范围，例如“在目标角色范围内，应用已确认的硬约束后无合格岗位”，不能误写成“这些条件在全目录不可同时满足”。
5. `SC-E-02`、`SC-E-04` 已触发数据质量警告：目录中目标角色以外存在满足硬约束的岗位。因此它们尤其需要 role-scope + hard-constraint 的联合说明。
6. 解决 `UnknownPolicy.CLARIFY` 的死路径：区分“明确无授权要求”和“授权信息未知”。若论文继续声称支持 unknown/clarification，就实现并测试；否则删除该能力主张。

验收：

- 每个 no-match 决策能从 trace 重新计算，且原因只包含实际减少候选数的字段。
- 目标角色外存在满足项时，系统不再声称硬约束全局不可满足。
- UNKNOWN + HARD + CLARIFY 会在 ranking 前短路，不得当成 eligible。

### P0-6：oracle、实验身份与权威场景

涉及：

- `src/jobrec_eval/relevance.py`
- `src/jobrec_eval/oracle_reference.py`
- `src/jobrec_eval/experiment_identity.py`
- `src/jobrec_eval/experiment_runner.py`
- remote provider
- `scripts/build_eval_scenarios.py`
- `Makefile`、README、CI

Oracle：

- 现有 oracle grade 主要看 hard eligibility、role score、required-skill coverage；人工 rubric 的 grade 3 还要求 location、mode、salary、experience 全部合适，grade 2/1 按软问题数量区分。
- 当前 oracle-human weighted κ 约 0.7517；rater-rater weighted κ 约 0.9364，说明人工协议稳定，主要需要修 oracle 构念。

修改方案：

1. 写一份先于代码的 `oracle_v4` graded relevance specification，与人工 rubric 逐字段一致。
2. 使用固定 scenario-level dev/holdout 校准，不能在全部 368 条已标 pair 上反复调到最好。
3. v4 canonical oracle 与 v3 并存，生成机器可读 diff。
4. 正式运行只接受人工 declared reference，禁止运行时从系统输出生成 reference。

实验 identity/provenance：

1. ID 增加 catalog hash、scenario/reference hash、prompt hash、provider/model、非秘密 endpoint fingerprint、generation parameters、code commit。
2. model/base URL 作为非秘密配置显式固定；API key 继续仅从环境读取。
3. 远端响应若提供 `response_id`/`system_fingerprint`，写入 manifest。
4. 正式运行拒绝 dirty/unversioned tree，拒绝覆盖既有 output root。
5. 当前压缩包不含 `.git`，正式重跑应从原始 Git 仓库的 clean commit 开始；若原仓库确实丢失，先将当前基线和修复建立可追踪的新仓库。

权威场景：

- 项目同时有 legacy `data/scenarios/scenarios.jsonl` 和正式 `evaluation/data/scenarios.jsonl`。
- `scripts/build_eval_scenarios.py` 默认可写正式路径，但生成内容缺少 `reference`，误执行可能清空人工声明。

修改方案：

1. 正式文件只读/审阅后冻结；builder 默认写 `scenarios_draft.jsonl`。
2. 覆盖正式文件必须显式危险 flag，并在 CI 中禁止。
3. 完整性检查：42 个唯一 ID、reference 全部存在且 provenance 为 declared、clarification answers 完整、子集同步、hash 固定。
4. README/Makefile 将旧路径标成 demo/legacy，正式 target 只引用 evaluation 路径。

验收：

- 分别只改变 code、scenario、catalog、prompt、model 时，experiment ID 都必须变化。
- API key/secret 不进入 manifest、hash 输入或日志。
- 同一 clean commit + 同一冻结输入可重放。
- v4 oracle 在冻结 holdout 上至少明显优于 0.7517，建议预注册目标 weighted κ ≥ 0.80；若未达到，不得通过调测试集掩盖，需报告原因或继续修构念。

## 4. P1 工程门禁

### 4.1 统一本地与 CI

现有 CI 有 ruff、coverage ≥85%、data quality、smoke evaluation 和 mypy ceiling；但本地 `Makefile` 的 `mypy src || true` 会把失败当成功，local coverage target 也没有同等阈值。

需要：

- 删除 `|| true`。
- 建立单一 `make gate-local` 或 `scripts/preflight.py`，本地与 CI 调用同一组命令。
- coverage 未达 85%、ruff/mypy/data-quality/reference integrity/replay 任一失败均返回非零。
- 如果暂时保留 mypy error ceiling，必须由共享脚本显式 ratchet，不能静默忽略；优先把现存错误清零。
- 固定依赖并保存 lock/freeze 文件。

### 4.2 本次只读审计已通过/未能验证

已通过：

- ZIP 路径安全检查：550 个 entries，无绝对路径、父目录穿越和 symlink。
- `python -m compileall -q src tests scripts`。
- 42 场景 deterministic/full 审计运行：42 runs、0 crashes。
- final slim release manifest：100 个文件，missing/changed/unrecorded 均为 0。
- data-quality validate：0 errors；2 warnings；另有 27 个已确认的 expired fixtures。

当前环境未安装 pytest、hypothesis、mypy、ruff，因此本次无法声称完整 test/lint/typecheck 已通过。这不是忽略项：修复工作第一步应在项目锁定环境中跑完整基线，并把失败清单保存下来。

## 5. 推荐实施顺序

### Phase 0：冻结规格与旧结果

1. 将当前两个正式实验标为只读：
   - Deterministic：`exp-e748800507ef`
   - Hybrid：`exp-6db1e87daed5`
2. 保存 v3 code/config/scenario/catalog/prompt/oracle/annotation/checksum。
3. 先写 salary、constraint revision、UNKNOWN、claim predicates、oracle v4 规格。
4. 为所有已知 bug 先添加 failing tests；不要先改实现再反推测试。

### Phase 1：状态与 provenance

1. 引入 `PreferenceEvent`、value-level strength、operation。
2. 当前轮抽取、event reducer、历史 rehydration。
3. 修 cue binding、多值与 revision。
4. 42 场景状态 reference gate 通过后再进入下一阶段。

### Phase 2：决策与解释

1. strict salary minimum。
2. 完整 eligibility/no-match stage trace。
3. UNKNOWN/clarification。
4. atomic claim + semantic validator。
5. 修改 task-success/grounding metrics。

### Phase 3：评价与可复现性

1. oracle v4 及固定 dev/holdout。
2. scenario authority 防误覆盖。
3. experiment identity/provenance。
4. remote model fingerprint、依赖锁定、release bundle。

### Phase 4：全量测试与 pilot

顺序建议：

1. `make gate-local`：unit/integration/scenario tests、ruff、mypy、coverage、data quality、reference integrity。
2. 42 场景 deterministic/full 审计：不作为论文数据，只看状态、候选、claim 和 trace。
3. 选择 12 个 sentinel scenarios 做 Hybrid pilot，覆盖 hard/soft、多值、salary、no-match、skill-gap、clarification 和正常成功；建议运行 full/no_memory/no_context × 3 repeats，共 108 pilot runs。
4. 重放 pilot 并核对 manifest/checksum/model-call coverage。
5. 人工抽查每类至少 10 个输出；任何 P0 问题复现即回到对应阶段。

pilot 数据与正式输出目录严格分离，不进入论文统计。

### Phase 5：冻结后只跑一次正式主实验

正式冻结项：

- clean code commit；
- dependency lock；
- 42 个场景及 declared reference hash；
- catalog hash；
- prompt hash；
- oracle/scorer/metric version；
- deterministic 与 Hybrid config；
- 固定远端 model 和 endpoint fingerprint；
- annotation schema；
- 空的新 output root。

运行矩阵保持可比性：

| Backend | Variants | Repeats | Runs |
|---|---:|---:|---:|
| Deterministic | full、no_memory、no_context、one_shot、profile_only | 1 | 42 × 5 = 210 |
| Hybrid | full、no_memory、no_context | 3 | 42 × 3 × 3 = 378 |
| 合计 |  |  | **588** |

正式运行期间：

- 不使用 `--allow-overwrite`。
- 任一 run failure、manifest mismatch、model fallback、scenario/reference drift 立即停止整批；修复后重新创建新的版本化实验，而不是在原目录补跑并伪装成一次。
- 两个 backend 必须共享 scenario/catalog/prompt/code fingerprints；provider/model 差异按设计记录。

### Phase 6：发布与注释

1. 导出完整 replay archives，而不只是 slim release；保存 SHA-256。
2. 当前压缩包中的 slim release 校验通过，但 manifest 声明的两个完整归档未包含在附件中：
   - `CMJCC_deterministic_exp-e748800507ef.zip`
   - `CMJCC_hybrid_exp-6db1e87daed5.zip`
   新发布必须包含完整包或提供稳定下载位置和 hash。
3. relevance 标注按 `(scenario_id, job_id)` 复用旧标签；scenario/catalog 不变且 pair 相同的无需重复标，只有新增 pair 做双人增量标注与 adjudication。
4. claim 的 run_id/claim_id 会变化，不能直接声称旧 occurrence 标签继续适用。建议按稳定的“claim type + normalized predicate/arguments + evidence signature”去重后双人标注。
5. 旧数据中有 11,197 个 claim occurrences，但只有 553 个 `(claim_type, text, evidence projection)` 唯一组合；新版本预计仍可把实际人工量控制在数百级，而非一万多条。论文同时报告 unique-signature 和 occurrence-weighted 结果。

## 6. 正式运行前的“一票否决”验收表

| 门禁 | 必须达到 |
|---|---|
| 完整测试 | pytest 全绿；ruff/mypy 全绿或显式且不增长的批准 ceiling；coverage ≥85% |
| 42 场景状态 | 42/42 values/strength/reference 一致 |
| 隐藏边界测试 | ≥24 条，全部通过，且无 scenario-ID 特例 |
| provenance | 原证据不复制、不改 turn；重启前后状态/决策一致 |
| salary | eligibility/ranking/oracle/claim 对所有边界一致 |
| no-match | 原因可由 stage trace 重算；role scope 表述准确 |
| claim validator | 能检出旧的 unsupported 家族，不再 constant-positive |
| oracle v4 | rubric 对齐；冻结 holdout 上达到预注册门槛或给出审阅结论 |
| 实验 identity | code/scenario/catalog/prompt/model 任一改变都会改 ID |
| pilot | 0 crash、0 fallback、0 manifest drift、100% 预期 model-call coverage |
| replay/release | replay 一致；完整包与 slim 包 checksum 全部通过 |

只有整表全部通过，才启动 588 次正式运行。

## 7. 重跑后论文具体改哪里

不需要从头重写，但不能只替换一张总表。

必须更新：

- **Abstract**：新数据、主要结论和限制。
- **3.3.2**：权威场景/reference 的生成与冻结方式。
- **3.4**：实验 identity、实际 model、prompt/catalog hash、版本。
- **3.5**：oracle v4 规则及其与人工 rubric 的关系。
- **3.6.1–3.6.3**：relevance、constraint/task success、structural vs semantic grounding 指标。
- **3.8**：relevance 增量标注与 claim-signature 双人标注。
- **3.9、3.11**：reproducibility、full archive、threats。
- **4.3–4.7**：typed event、provenance、revision、salary、UNKNOWN、eligibility/no-match trace、atomic claims。
- **4.10–4.11**：版本、发布物和仍存在的限制。
- **第 5 章全部数值结果**：主结果、SC-D-02、多轮、消融、no-match、grounding、human relevance、oracle-human agreement、salary 分析、图表和置信区间。
- **第 6 章**：RQ 回答、结论及不过度外推的边界。

大体可保留：

- 研究动机和 RQ 框架；
- 大部分 related work；
- 总体 CMJCC 模块结构；
- scenario 为统计单位、paired analysis 和 multiple-comparison correction 的总体设计；
- 42 场景和 200 jobs 的基本描述（前提是冻结内容不变）。

重跑后的工作不是纯粹“复制新数字”：需要检查结论方向是否变化并重写解释。但如果上述代码和评价门禁先完成，届时不应再发生架构级返工，剩余主要是表格/图更新、增量标注、结果解释和措辞校准。

## 8. 推荐的提交结构

```text
release_v4/
  code/
  configs/
  inputs/
    scenarios_with_declared_references.jsonl
    job_catalog.jsonl
    prompt_manifest.json
  oracle_v4/
  deterministic_full_archive/
  hybrid_full_archive/
  slim_release/
  annotations/
    relevance_source.csv
    claim_signature_source.csv
    adjudication_log.csv
  reports/
    before_after_diff.md
    regression_gate_report.json
    experiment_report.md
  checksums.sha256
  environment.lock
  provenance.json
```

## 9. 最终判断

这套项目有可用的实验骨架，不需要全部推翻。真正危险的是在现有 validator、salary 规则、历史重解析和不完整 experiment identity 上直接重跑：那样即使程序没有崩溃，结果仍可能在导师追问时失去可信度。

最稳妥的路线是：**先固定规则 → 修状态/provenance → 修决策/claims → 升级 oracle 与 identity → 全门禁 + pilot → 冻结 commit → 一次跑完 588 → 增量人工标注 → 统一更新论文。**

如果严格执行这条路线，正式重跑之后出现“还要再改核心代码、整批再跑”的概率会显著降低，后续工作将主要收敛为报告更新和小范围人工核验。
