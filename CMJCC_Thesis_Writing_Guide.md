# CMJCC 论文写作指导（第 5–7 章）

> 目的：把"写错就站不住"的表述约束集中到一处。本文档记录的是**已决定的口径**与**已核实的事实**，不是建议清单。
> 依据产物：**主实验 `exp-197f6aacc171`**（deterministic，五变体，210 runs，0 system_error，0 invalid_runs）与**补充鲁棒性验证 `exp-06cc34defe39`**（hybrid，真实 LLM，378 runs，`invalid_runs = 1`）。两者 `source_fingerprint` 相同，四棵目录树 `cli verify` 均 OK。
> 适用范围：Chapter 3 方法描述、Chapter 5 结果、Chapter 6 讨论、Chapter 7 结论。

---

## 0. 引用纪律（先看这一条）

**可引用的正式产物只有两个，且各有明确的论文角色。**

| 角色 | experiment id | 目录 | 规模 |
|---|---|---|---|
| **主实验**（deterministic，五变体） | **`exp-197f6aacc171`** | `evaluation/outputs/exp-197f6aacc171/` + `evaluation/outputs/_runs/exp-197f6aacc171/` | 210 runs（5 × 42 × 1） |
| **补充鲁棒性验证**（hybrid，真实 LLM） | **`exp-06cc34defe39`** | `evaluation/outputs_hybrid/exp-06cc34defe39/` + `evaluation/outputs_hybrid/_runs/exp-06cc34defe39/` | 378 runs（3 × 42 × 3） |

**两者的 `source_fingerprint` 完全相同**：`8eba8f8106dcf9201a19bd6e701851f025c9fc51969c0d238853cb682d7a03c8`（`commit_hash 9768116417...`，`git_dirty: true`）。也就是说**两组产物出自同一份代码**——这是刻意付出重跑 deterministic 的代价换来的，论文可以据此说 deterministic 与 hybrid 的差异**只来自后端（mock vs 真实 LLM）**，不来自代码版本差异。

归档副本（与源逐文件一致，四棵树 `cli verify` 均 OK）：`final_release/deterministic_runs/`、`final_release/hybrid_runs/`；对外分发为 `dist/CMJCC_deterministic_runs_exp-197f6aacc171.zip`（7.2 MB）与 `dist/CMJCC_hybrid_runs_exp-06cc34defe39.zip`（13.9 MB）。✅ 均可引用。

**不可引用**：

| 目录 / id | 性质 | 能否引用 |
|---|---|---|
| **`exp-f90573008bdb`** | 上一份 deterministic 产物，**已被 `exp-197f6aacc171` 取代** | ❌ |
| `exp-301060a1899d`、`exp-515b63d6a656` | 修复过程中的**中间产物**（已删除） | ❌ |
| `evaluation/outputs/exp-8793b18de5b2/` | **修复前**产物 | ❌ |
| `test_results/`（含其中的 hybrid 报告） | **修复前**产物 | ❌ |

不可引用的那几个目录里仍带修复前的标签与数字（旧的 "run-level discordant pairs" 表述、旧的 "68 tests" 说法、`one_shot == no_memory` 的结果）。**它们不得手工改写**——手改会产出一份与任何真实运行都不对应的报告。要用就整体重跑替换，否则只能另附 erratum。

`exp-f90573008bdb` 有**唯一一个**仍可提及的用途（见 §10）：它与 `exp-197f6aacc171` 的 `variant_summary` **逐项一致**，可作为"LLM 调用记账的代码改动未触及任何指标计算路径"的可复现性证据。除此之外，**不得**把它当作任何结果数字的来源。

experiment id 现在把**源码指纹**计入哈希（`jobrec/` + `jobrec_eval/` 全部 `*.py`，LF 归一化后 sha256），因此**任何代码改动都会同时换掉两个 id**，并破坏当前的同指纹一致性。若在定稿前又改了代码，必须重跑并把全文的 id 一并改掉，不能沿用旧 id。

---

## 1. no-match 的定义（SC-E-02 / SC-E-04）——**已决定：保留现状，在论文中写清楚**

### 决定

保留这两个场景现有的 turn 文本，**不**收紧约束。数据质量报告中那 2 个 `no_match_scenario_constraint_satisfiable` warning 是**已知且已接受**的，不是待修缺陷。

理由：这不是数据缺陷，而是"no-match"这一构念的定义问题。系统实现的定义本来就是"角色匹配 ∧ 硬约束"联合不可行；收紧文本会改动 `scenarios.jsonl`，进而改变 scenario hash 与 experiment id，使已定稿的 `exp-197f6aacc171` **与 `exp-06cc34defe39` 双双作废**（hybrid 重跑还要再花约 135 分钟与约 34 万 tokens）——为一个措辞问题重跑两个实验不成比例。

### 论文中必须怎么写

在 Chapter 3 定义 no-match 时（以及 Chapter 5 §5.3 报告 no-match precision/recall 时），必须显式写出：

> A no-match outcome is defined as the **joint** infeasibility of role fit and the hard constraints, not the infeasibility of the hard constraints alone. A posting that satisfies every stated hard constraint but lies outside the candidate's requested role family is correctly excluded, and a scenario in which only such postings remain is correctly a no-match.

然后如实说明这一定义的可观测后果（这正是那 2 个 warning 的内容）：

- **SC-E-02**（Business Analyst，仅吉隆坡、≥ RM4500、仅 onsite）：目录中有 **5** 个职位（`job-0021`、`job-0086`、`job-0089`、`job-0094`、`job-0169`）满足全部硬约束，但**全部落在所请求的角色族之外**。因此 no-match 成立于"角色 ∧ 约束"，而非"仅约束在全目录内不可行"。
- **SC-E-04**（Software Engineer，仅吉隆坡、hybrid、≥ RM6000）：同理，有 **1** 个职位（`job-0012`）满足硬约束但在角色族之外。

场景自带的 `notes` 字段已经写明了角色族内的联合不可行性（"no KL onsite BA posting pays >= RM4500" / "no KL hybrid SE posting pays >= RM6000"），可直接作为论文中的依据引用。

**不要**这样写：把这 2 个 warning 说成"已修复"或"无影响"。正确的写法是：数据质量检查按更严格的口径（仅约束不可行）发出提示，而本研究采用的是联合不可行口径，两者的差异已在此说明——这是一处**有意的、已记录的**构念选择。

---

## 2. ⚠️ 生存者偏差——涉及排序类指标时的三条硬约束

这是本论文最容易被答辩攻击的一处。**根因**：排序类指标（NDCG@5、P@5、HCSR、mean graded relevance）只在"该变体确实返回了排序列表"的场景上取平均。放弃对话或拒答的变体在那些场景上没有返回列表，因此**退出了分母**，而退出的恰好是最难的场景。

实测分母（**主实验 `exp-197f6aacc171`**；hybrid 的分母另见 §6.2，同一套警告同样适用）：

| variant | n(NDCG@5) | n(P@5) | n(HCSR) | n(Grounding) | n(TaskSucc) |
|---|---|---|---|---|---|
| full | 37 | 37 | 37 | 42 | 42 |
| profile_only | 24 | 28 | 28 | 28 | 42 |
| one_shot | **21** | **21** | **21** | **26** | 42 |
| no_memory | 28 | 28 | 28 | 33 | 42 |
| no_context | 38 | 42 | 42 | 42 | 42 |

后果：`one_shot` 的 P@5 与 HCSR 都读作 **1.000**，比 `full` 的 0.974 **更高**——但这纯粹是因为它放弃的 7 个最难场景没进分母。

### 约束 1：跨变体比较只能用 `task_success`

`task_success` 在全部 42 个场景上都有定义（放弃即记失败，不会退出分母），是唯一可跨变体直接比较的主指标。排序类指标只在 `_n` 相当时才可比。

引用变体表时**必须连 `_n` 一起给**。报告已经替你印好了：§1 headline 每个指标带 `n`，§5 表下有逐指标分母表和 "Read the denominators before the means" 警示段。

### 约束 2：`full vs one_shot` 的近零 Δ 不是"无差异"

NDCG@5 Δ=+0.005，95% CI [+0.000, +0.014]，n=**21** 配对场景。

- ✅ 正确：**"在 `one_shot` 放弃的那些场景上不可估计"**（not estimable on the scenarios that variant abandoned）。配对只在两个变体都返回了可比值的场景上成立，被放弃的场景不构成配对、不进入 Δ。
- ❌ 错误："两者排序质量无差异"、"one_shot 与 full 相当"、"消融未显示影响"。

### 约束 3：`grounding` 带同样的警告

`grounding` 只在实际产出了事实性 claim 的 run 上取平均；没有推荐就没有 claim 可核，于是退出分母。因此**在缩小的分母上得到 1.000，意味着被检查的 claim 更少，而不是解释更可靠**。`one_shot` 的 n(Grounding) 只有 26。

同理，§5.2 的每格合规率现在是 `rate (n=applicable)` 形式：`one_shot` 的 `work_modes` 合规率 1.000 只建立在 **6** 个可判定 pair 上，而 `no_context` 的 0.457 建立在 35 个之上、其中 37% 是未知值（`unk 37%`）。**不要**把前者写成"one_shot 完美遵守工作模式约束"。

---

## 3. grounding = 1.000 与 handoff = 1.000 怎么解释

主实验里这两个数都是 1.000。这**不是**缺陷，也**不是**"系统永不出错"的证据。按报告 §10 的口径写：

- well-formed 输入下，每条 claim 都能解析到已登记的 evidence，因为 claim validator 会把解析不到的丢弃——因此不存在"剩下的未被支撑的 claim"。这是**由构造保证**的。
- 鲁棒性证据来自**独立的 fault-injection 套件**（11 类故障，见报告 §10.2），在含故障的集合上这两个 rate 严格小于 1.000。
- 故障样本**刻意不混入主实验**：那样做会为了一个鲁棒性论点而污染主测量。

**不要**把主实验的 1.000 写成"在对抗性/异常输入下的表现"。

另外，报告 §10.1 里 `failure_detection_rate` 与 `recovery_success_rate` 显示 **N/A（空分母）**，因为主实验没有注入故障。这是刻意的：在零观测上印 0.0 或 1.000 都是误导。论文里如需这两个数，只能引 fault-injection 套件。

---

## 4. 必须改写的措辞：ActiveSearchState 持久化

**数据库 schema 里没有 ActiveSearchState 表。** 持久化的只有 decision 上的 `active_search_id` 这个标识，状态本身每次检索重新推导。

- ❌ 不能写："ActiveSearchState 被完整持久化到 PostgreSQL"。
- ✅ 应写：active search 的**身份**（`active_search_id`）被持久化并可追溯，**状态本身按需重建**（re-derived per search）。

注意区分两个层面：run bundle 里**确实**有 `active_search_state.json`（因此**归档/可审计**层面是完整的，逐 run 可离线检查），但**数据库持久化**层面不成立。Chapter 4 讲 persistence 时按上面的口径写；讲 reproducibility/archival 时可以说 bundle 完整保存了该状态。

---

## 5. 延迟与成本：两个数据源，绝不可混用（Chapter 5 Latency 节）

Latency 一节现在有**两个互不替代**的数据源。写错的唯一方式就是把它们混在一起——两者相差**三个数量级**。

### 5.1 真实 LLM 成本与延迟 → 引 hybrid 实验 `exp-06cc34defe39`

这是**唯一**可以称为"真实 API 成本"的数据。全部来自 378 次真实调用（`gpt-5.5` @ `https://api.vectorengine.ai/v1`，OpenAI 兼容接口）。

**延迟**

- 单次 LLM 调用：中位 **11,624 ms** / p95 **20,420 ms** / 最大 **57,518 ms**。
- LLM 等待合计 **81.7 分钟**，占实验总耗时 **134.7 分钟**的约 **61%**——这个比例本身就是可引用的结论：真实后端下，端到端延迟由模型往返支配，编排开销可忽略。
- 变体级 `total_latency_ms` 均值：full **12,353** / no_memory **17,237** / no_context **14,642**。

**token 成本**

- 378 次调用，全部 `purpose = intent_extraction`，每 run 恰好 **1.00** 次。
- prompt **227,105** + completion **113,969**（**其中 reasoning 52,333**）= **total 341,074**，每 run 平均 **902** tokens。
- **reasoning token 必须单独报告**：`gpt-5.5` 是推理模型，把 reasoning 折进 completion 会低估真实计费量，分开写才对得上账。
- `usage` 字段缺失 **0** 次——378/378 都是模型返回的真实用量，没有一处是补零推算的（代码刻意在 `usage` 缺失时记为**缺失**而非 0）。

**可靠性**：fallback **0** 次，retry（去掉 `response_format` 重试）**0** 次。

### 5.2 编排开销的规模行为 → 引 perf 套件

perf 实测（`artifacts/reports/perf_latency.json`，catalog 100/200/300）：

- e2e 中位数：deterministic 15.3 / 18.3 / 18.6 ms；hybrid 17.1 / 20.1 / 19.8 ms
- retrieval：2.9 → 4.8 → 6.2 ms（随目录规模增长）
- LLM 中位数：deterministic 0.0；hybrid 0.558 / 0.701 / 0.620 ms

**这一区分仍然成立且必须写明**：perf 套件用的**还是 mock provider**，`provider_reported_llm_ms` 在**所有**单元格（含标着 hybrid 的那几列）都是 0.0，因为 mock 不发生网络往返。所以 perf 表里的 "LLM latency" 是**mock 的计算耗时，不是 API 成本**。

- ❌ 不能写："hybrid 模式下 LLM 调用仅增加约 0.6 ms 延迟"——这句话会把 mock 开销当成真实模型延迟，而真实值是它的两万倍。
- ✅ 应写：perf 表度量的是**编排开销（orchestration overhead）在固定后端下的规模行为**；真实 LLM 网络延迟不在其中，见 §5.1。

`retrieval` 随目录规模近线性增长这一点是真实可引用的结论。

### 5.3 一句话口径

> **报告真实成本与延迟 → 引 `exp-06cc34defe39`（hybrid）。报告编排开销随目录规模的行为 → 引 perf 套件。两者不得放进同一张表。**

---

## 6. deterministic 主实验 与 hybrid 补充鲁棒性验证：关系与写法

### 6.1 两者的角色是**不对称**的，写法也必须不对称

| | 主实验 | 补充鲁棒性验证 |
|---|---|---|
| id | `exp-197f6aacc171` | `exp-06cc34defe39` |
| 后端 | deterministic（mock provider） | hybrid（真实 `gpt-5.5`） |
| 变体 | full / profile_only / one_shot / no_memory / no_context | full / no_memory / no_context |
| 规模 | 210 runs（5 × 42 × 1） | 378 runs（3 × 42 × 3） |
| 承担的论证 | 全部消融结论、Δmemory / Δcontext、统计检验、error taxonomy | "把 mock 换成真实 LLM 后，结论是否还站得住" |

**所有主要结果（消融 Δ、McNemar、Holm 校正、变体总览、错误分类）一律引主实验。** hybrid 只用来回答一个问题：换成真实模型后，机制层面的结论方向是否保持。

- ❌ **绝对不要**把两组数字放进同一张表求平均、求总和，或算"跨两个实验的均值"。它们的后端不同、重复次数不同（1 次 vs 3 次）、变体集不同，任何合并数都没有对应的实验设计。
- ❌ **不要用 hybrid 讨论 `profile_only` 或 `one_shot`**——hybrid 根本没跑这两个变体，**没有 hybrid 对照**。这两个基线的任何论述只能引主实验。
- ✅ 正确的呈现方式：主实验一张完整表；hybrid 单独一小节，只对 full / no_memory / no_context **逐变体并列**，并写明"3 变体 × 42 场景 × 3 重复"。

**一个可以放心写的强论点**：两组产物的 `source_fingerprint` 相同（`8eba8f8106dc...`），所以两者的差异**只来自后端**，不来自代码版本。这在多数同类研究里做不到。

### 6.2 hybrid vs deterministic 的实测差异（同场景集、同变体）

| variant | 指标 | hybrid | deterministic |
|---|---|---|---|
| full | task_success | **0.9206** | 1.0000 |
| full | ndcg@5 | 0.9584 | 0.9585 |
| full | hcsr | 0.9910 | 1.0000 |
| full | grounding | 1.000 | 1.000 |
| full | handoff | 0.996 | 1.000 |
| no_memory | task_success | 0.7381 | 0.7619 |
| no_memory | ndcg@5 | 0.9271 | 0.9452 |
| no_context | task_success | 0.1667 | 0.1667 |
| no_context | ndcg@5 | 比 deterministic **高 0.0173** | （主实验表为 0.710，n=38） |
| no_context | hcsr | 比 deterministic **高 0.0286** | （主实验表为 0.600，n=42） |

hybrid 分母（**§2 的生存者偏差同样适用，引用排序类指标必须带 `_n`**）：

| variant | n(ndcg) | n(task) | n(grounding) |
|---|---|---|---|
| full | 37 | 42 | 42 |
| no_memory | 30 | 42 | 35 |
| no_context | 38 | 42 | 42 |

### 6.3 `full` 的 task_success 0.9206 < 1.0000 是**真实发现**，要正面写

deterministic 下 full 是 42/42 全对；真实 LLM 下降到 0.9206。这不是缺陷掩盖，而是本研究最有价值的鲁棒性观察之一。**主因必须写对**：

full 的 10/126 个 run 级失败分布：

- **SC-G-01：3/3 全败**
- **SC-G-02：3/3 全败**
- SC-D-11：2 次、SC-D-12：1 次
- SC-A-04：1 次（=§6.4 的那个崩溃）

即 deterministic 与 hybrid 之间 **0.079** 的差距中，**约 90% 是真实 LLM 的行为**——尤其 SC-G-01 / SC-G-02 这两个**澄清场景在真实 LLM 下系统性失败**（不是偶发，是 3/3），只有约 **10%** 来自那个崩溃。

- ✅ 应写：在真实 LLM 后端下，澄清编排在两个场景上系统性失败，说明该机制的有效性依赖抽取质量；mock 后端会掩盖这一依赖。
- ❌ 不要写：hybrid 的下降"主要由一个实现 bug 造成"——事实相反，bug 只占约十分之一。

**抽取来源可作为解释依据**：full rule 0.204 / llm 0.796；no_context rule 0.205 / llm 0.795（schema_failure_rate 0.0019，1 个字段回落规则）；**no_memory rule 0.009 / llm 0.991**（schema_failure_rate 0.0094，4 个字段回落规则）。`no_memory` 几乎完全依赖模型抽取（没有记忆可复用），因此 schema 失败率最高——这与它 task_success 的下降方向一致。fallback 0 次、retry 0 次。

### 6.4 可引用的失败案例（Chapter 6 Discussion / Failure Cases）

hybrid 的 `invalid_runs = 1`：`full` / **SC-A-04** / repeat 1，`failure_code = INTERNAL_ERROR`，`AttributeError: 'list' object has no attribute 'strip'`。

**根因链路**（已完整定位并复现，可直接写进论文）：

1. 真实 LLM 把 `normalized_value` 返回成**单元素列表**（mock provider 从不产生这种形状）；
2. `validate_field("target_roles", ["software engineer"])` 返回 **ok=True 且原样保留该列表**——因为 `target_roles` 本身就是列表型字段，形状"合法"；
3. orchestrator 的 schema 修复循环**只处理 ok=False**，于是跳过了它；
4. 该列表流入下游 `canonical_role(r) for r in ...`，`canonical_role` 对字符串调 `.strip()` → 崩溃。

**同一条响应里的对照组说明修复机制本身是对的**：`preferred_locations` 与 `work_modes` 被判 ok=False，因此被"单元素列表 → 标量"的修复逻辑**正确处理**。漏掉的是"**验证通过但形状仍错**"这一类，不是修复机制失效。这正是可以写进 Discussion 的教训：**校验通过不等于形状可用**，修复循环以 `ok=False` 为触发条件时，会漏掉合法但不可用的形状。

**引用这个案例时必须连带说明以下三点，缺一不可**（否则会被读成系统性缺陷）：

1. **影响面 1/378**：378 个 run 中只有 **2 个**收到列表包裹值，其中 1 个崩溃、1 个被正确修复；
2. **无静默数据污染**：检查全部 378 个 `active_search_state.json`，**零个**出现嵌套列表——错误以崩溃形式暴露，而不是悄悄写进状态；
3. **只占差距的约 10%**：该崩溃对 full 场景级 task_success 的影响仅 **0.9206 → 0.9286**（0.008），hybrid 与 deterministic 之间 0.079 的差距约 90% 来自真实 LLM 行为（见 §6.3）。

**表述纪律**：

- ❌ 不要写"LLM 返回了错误格式导致失败"——**这是流水线未处理的输出形状**，模型返回的是一个语义正确的单元素列表，责任在验证/修复的衔接处。
- ❌ 不要写"已修复"。**该缺陷尚未修复，是否修复并重跑 hybrid 仍待决定**（修复会改变 `source_fingerprint`、破坏两组产物的同指纹一致性，且重跑需约 135 分钟与约 34 万 tokens，收益只有 0.008）。论文若在此决定前定稿，应把它写成**已定位、已量化的已知限制**。
- ✅ 可以写：该缺陷由真实 LLM 后端暴露，而 deterministic 后端**无法**暴露它——这本身就是"补充 hybrid 验证有价值"的证据。

### 6.5 `no_context` 在 hybrid 下 ndcg/hcsr 略高：不得当作机制结论

`no_context` 的 ndcg@5 在 hybrid 下比 deterministic 高 **+0.0173**、hcsr 高 **+0.0286**，而 task_success 两侧**完全相同**（0.1667）。

- ✅ 应写：这属于**小样本波动与 LLM 抽取差异**造成的排序层面扰动；决定性的 task_success 没有变化，因此不改变 Δcontext 的结论。
- ❌ **绝对不要**写成"去掉 context 反而更好"、"context 机制在真实 LLM 下无益"或任何机制层面的结论。0.1667 的 task_success 说明该变体在真实 LLM 下依然**失败得一样彻底**，排序指标的小幅上移只是在它返回的那些列表上取平均的结果。

---

## 7. 统计表述（Chapter 3 方法 + Chapter 5 结果）

以下口径与代码、与自动生成的报告 §8 三者一致，照抄即可：

- **分析单位是 `scenario_id`**。`repeat_index` **仅**用于稳定性与方差分析，**从不**当作独立样本；配对之前先在场景内把重复折叠。deterministic 运行默认每场景 1 次；重复运行**不能**扩大样本或压低 p 值。
- **连续指标**：先在 scenario × variant 上对重复取平均，再按 `scenario_id` 配对。配对 bootstrap（**5000** 次迭代，seed **2026**）给出 scenario-mean 差的 95% CI；p 值用 Wilcoxon signed-rank。
- **二值 task success**：McNemar 作用于**场景级**配对二值结果（场景内按多数票折叠，偶数次平票**保守**记为 not-success），再对不一致的场景对做精确二项检验。因此 `n_pairs` 是**有效配对场景数，不是运行数**。报告 §8 有一张 pairing-provenance 表把这一点摊开。
- **多重比较**：Holm 校正在每个消融内、每个场景子集内、且**每个 outcome family 内**独立进行。预注册的 primary family（§6.1/§6.2）与 secondary 的过程指标 family（§6.3，即 `response_turns` 与 `clarification_efficiency`）**分开校正**——把过程指标追加进 primary 会让 Holm 从 6 个变 8 个，从而抬高每一个预注册 p 值。两个 family 都记录在 `manifests/analysis_plan.yaml`。
- **小样本口径**：CI 含 0 时写 **"direction observed, uncertain"（方向已观察到，但不确定）**，**绝不**写 "no effect"。

### 可引用的主要结果

- Δcontext（context-dependent 子集，n=15/10）：task_success Δ=**1.000**（15/15 全不一致）、HCSR Δ=**0.520**、NDCG@5 Δ=**0.259**、mean_violation_count Δ=**−0.740**，CI 全部不含 0。
- Δmemory（memory-dependent 子集）：task_success Δ=**0.5625**，p=0.004，**p_holm=0.023**，n=16。同子集 NDCG Δ=0.014（CI 含 0，n=7）→ 按"方向已观察到，不确定"写。
- full vs profile_only：task_success Δ=**0.833**，CI [+0.714, +0.929]，n=42。

---

## 8. 消融条件的有效性：`one_shot` 是真正的单轮基线

Chapter 3 描述变体时需要知道：`one_shot` 与 `no_memory` 曾经**行为完全相同**（42 场景 × 43 列，0 处差异），因为 `use_multi_turn_continuation` 这个 flag 在语义上是**死的**。该缺陷已修复——runner 的澄清循环现在真正受该 flag 门控，`one_shot` 因此是真正的单轮条件。

可引用的分离证据：两者在 **7** 个澄清场景（SC-B-01..05、SC-G-01、SC-G-02）上各有 **21** 列实质差异（turns 1 vs 2；termination `continuation_disabled` vs `recommendation`；returned_count 0 vs 5；task_success 在 6/7 处 0 vs 1），其余 35 个场景仅变体标签不同——正是单一机制消融应有的形态。

**方法学上要写明**：场景脚本的轮次**刻意未被截断**。它们是实验刺激（stimulus）；截断会让不同变体收到不同输入，破坏"同输入配对"的基础。`one_shot` 的单轮性体现在**系统不续话**，而非输入被削减。

---

## 9. clarification 相关指标的正确读法

- **`clarification_efficiency` 是惩罚量表**，所有变体都是负数，**数值越高越有效率**。量级不是概率。实测：full 与 no_context −1.50，no_memory −216.00，profile_only −335.00，one_shot **−382.50**（最差）。
- 该分数实现的序关系是：**问了并解决 > 问了但放弃 > 跳过必要澄清**。因此"少问问题但答错"永远不会被评为高效率，"更短的对话"也不能让变体在这个序上前移。
- 报告 §5.4 把 `Abandoned`（`asked_unresolved`）与 `AnsweredRate` 紧挨着 `EffScore` 呈现：one_shot 放弃 16、AnsweredRate 0.125；no_memory 放弃 9、0.562。**`Abandoned` 非零的变体不得仅凭 EffScore 被称为高效**。
- `response_turns` **必须**与 task success 联合解释，不得单独呈现。
- **缺口（如需引用 reason code 分布必须说明）**：`clarification_reason_code` 只有在"对话结束时仍停留在未回答的提问上"才有值，因为 trace 记录本身不携带 reason code。取值范围受此限制。

### 错误分类法（报告 §9）

96 个 task-unsuccessful runs 的分布：missing_constraint_enforcement 35（no_context）、missing_dialogue_evidence 35（profile_only）、stale_or_missing_memory 18（no_memory）、**missing_dialogue_continuation 7**（one_shot）、other 1。合计等于各变体失败数之和（no_context 35 + profile_only 35 + one_shot 16 + no_memory 10 = 96）。

类别键控在**已记录的列**上，而非变体名。`cannot_answer` / `max_turns` / `repeated_slot` **刻意排除**在 `missing_dialogue_continuation` 之外——那些场景的续话机制是工作的，折进来会把 profile_only 的 7 个 `repeated_slot` 失败从 baseline 类别里偷走。

---

## 10. 可复现性：replay 证明了什么、没证明什么

- ✅ **已证明（主实验）**：`replay_experiment()` 在 `exp-197f6aacc171` 上重放全部 **210/210** runs，**0 differences，0 errors**；五个 key-state 哈希（extracted slots、state version、filtered jobs、ranking output、explanation claims）逐一相同。从已保存 bundle 重算统计与报告（`pipeline --experiment-dir`）产出的 **17 张 metrics CSV 与正式分析逐字节相同**。四棵目录树（主实验 2 棵 + hybrid 2 棵）`cli verify` 均 OK。
- ⚠️ **这条 210/210 证明的是什么**：deterministic bundle 的 `model_calls.jsonl` 是 **0 字节**（mock provider 不产生模型调用），所以重放时 `ReplayProvider` 无记录可服务、回落到规则抽取器。因此这条证据支持的是**流水线的确定性**，**不是**"录制的模型输出可回放"。
- ✅ **已证明（hybrid）：录制的真实模型输出可回放**。在 `exp-06cc34defe39` 上跑完整 replay：**378 runs、0 differences**。其中 **376 个 run 真正命中了录制的远程响应**并重算出逐一相同的五个 key-state 哈希——`ReplayProvider` 按 `call_id` 命中确实成立，**prompt 在任何模式下都不落盘**（只存 `prompt_hash`），因此这条回放不依赖持久化 prompt。

  引用这一条时**必须连带给出三类例外的确切计数**，否则会被读成 378/378 全部命中：

  | 类别 | 计数 | 原因 |
  |---|---|---|
  | 命中录制响应并哈希一致 | **376** | 正常路径 |
  | 回落到规则抽取器 | **2** | 这两个 bundle 的 `model_calls.jsonl` **本就为空**（`no_context/SC-D-03/2`、`no_memory/SC-A-02/2`，纯规则抽取，没有发生模型调用）——不是查表失败 |
  | 无法重放 | **1** | 即 §6.4 那个崩溃的 `full/SC-A-04/1`：它在写出 `candidate_state_before.json` 之前就失败，bundle 不完整，`ReplayInputError` 被如实报出而非静默跳过 |

  每 bundle 的模型调用数分布为 **{0 次: 2, 1 次: 374, 2 次: 2}**，合计 378 次，与 §5.1 的 token 口径一致。

- ❌ 仍**不可写**："378/378 全部由录制输出重放"（有 2 个 bundle 没有录制可服务）、或"replay 无任何例外"（有 1 个 bundle 因崩溃而不可重放）。

**跨运行证据的精确措辞**——有两条，性质不同：

- **证据 A（历史，产物已不可复核）**：连续两次完整运行（`exp-515b63d6a656` 与 `exp-f90573008bdb`，二者均已不是当前产物）之间**唯一的代码改动是报告渲染层**，产生指标的代码路径未变。逐列比对后唯一差异是 `run_id` 与 wall-clock 延迟列；所有指标、统计量、分类法与配对比较逐字节相同。
- **证据 B（本轮新增，✅ 仍可复核）**：`exp-f90573008bdb` → `exp-197f6aacc171` 之间的代码改动只发生在 **LLM 调用记账**（token usage / request parameters / retry 痕迹 / `raw_response` 落盘、`ReplayProvider` 索引），逐项比对后**两份 `variant_summary` 的全部指标一致**，error taxonomy 分布亦不变。这条比证据 A 更有力：改动真的触及了 provider 与 bundle 写出路径，指标却一个都没变。`exp-f90573008bdb` 的两棵目录树**目前仍在磁盘上**（虽然已不可作为结果来源），因此这条比对随时可以重新复核。

两条证据的共同结论应写作：**在固定输入与固定种子下流水线的数值输出可复现；experiment id 因源码指纹而改变，并不意味着数字会改变**——不能写成"完全相同的代码跑两次"。这也顺带解释了为什么主实验的 headline 数字与上一份产物一致而 id 却不同。

---

## 11. 人工评测：当前必须按"未采集"来写

标注工具（`src/jobrec_eval/annotation_ui/`）与人工标签指标通路（`pipeline --relevance-source human`）都已建成并端到端验证，但**真人标签尚未采集**。因此：

- Chapter 5 现在只能报告 **automatic oracle**（version 1.0.0）作为 relevance 来源，并把 inter-rater agreement 列为**未报告**的构念效度威胁（报告 §4、§12 已按此条件化生成）。
- ❌ 不能写"两位评分者标注并计算了 kappa"——Chapter 3 若有此承诺，在标注完成前必须写成计划/未来工作，或等标注完成后再写。
- 一处**有意保留的不对称**需说明：retrieval recall 在两种模式下都使用 oracle 标签，因为人工判断只覆盖被返回的 pair，换成人工标签会让 recall@pool 平凡地趋近 1.000。这记录在 `manifests/analysis_plan.yaml` 的 `retrieval_recall_relevance_source`。
- 标注完成后，人工与 oracle 的对比另有一处 caveat 需写明：人工标签只覆盖被返回的 pair，因此人工模式下 NDCG@5 的 ideal DCG 是在**已判定池**上计算的，而 oracle 对全目录评分——Δ 因此**同时**混入了"标签来源变化"与"标签宇宙变化"，只能当作一致性诊断，不是无偏效应估计。
- 标注的**目标产物是主实验 `exp-197f6aacc171`**（`--experiment-dir evaluation/outputs/_runs/exp-197f6aacc171`）。人工标签不针对 hybrid 采集。

---

## 12. 数字口径

- 测试数量一律报 **683 tests**（683 passed / 2 skipped / 22 deselected / 0 failed，`pytest -m "not postgres and not perf"`）。**不要**引用 "68 tests"（修复前旧报告）或过程中出现过的 483 / 594 / 599 / 635 / 647 / **661**。
  - 那 2 个 skip 只是该次运行的 shell 未设 `DATABASE_URL`；带真实库运行时为 **5 passed / 0 skipped**。
  - 22 个 deselected 是 perf 测试，单独执行为 **22 passed**。
- coverage **92%**（CI 门槛 85%）。
- mypy **14 errors**，等于 CI 门槛 `MYPY_MAX_ERRORS: "14"`，因此门禁通过。若论文提到静态检查，应如实写"存在 14 个已记录的余量告警"，不要写"零告警"。
- 检查器已钉版（`ruff==0.16.0`、`mypy==2.3.0`）——这是冻结后门禁判定可复现的前提，Chapter 3 讲 reproducibility 时值得一提。
- 数据质量：**0 error / 2 warning / 27 条已确认 fixture**。那 27 条是**故意保留**的过期职位，带 `is_test_fixture` / `expected_ineligible_reason` 注解，且这些注解被排除在 `raw_payload_hash` 之外，因此 `catalog_hash` 可证明未变（`145dfa05...`）。
- 目录与场景：200 个职位、42 个场景；memory-dependent（≥ medium）16 个，context-dependent（high）15 个。
- **主实验规模**：5 个变体 × 42 场景 × 1 次重复 = **210 runs**（0 system_error，0 invalid_runs）。
- **hybrid 补充鲁棒性验证规模**：3 个变体（full / no_memory / no_context）× 42 场景 × **3 次重复** = **378 runs**，耗时 **134.7 分钟**，0 system_error，**invalid_runs = 1**（见 §6.4）。
- **hybrid token 总量**：**341,074** tokens（prompt 227,105 + completion 113,969，其中 reasoning **52,333**），每 run 平均 **902**；378 次 LLM 调用，每 run 1.00 次；`usage` 缺失 0 次。
- **hybrid 真实延迟**：单次调用中位 **11,624 ms** / p95 **20,420 ms** / 最大 **57,518 ms**；LLM 等待 **81.7 分钟** / 总 **134.7 分钟**。
- 两个实验的 `source_fingerprint` 均为 `8eba8f8106dc...`，`commit_hash 9768116417...`，`git_dirty: true`。

---

## 13. 禁止表述清单

| ❌ 不要写 | ✅ 改成 |
|---|---|
| 本框架全面优于现有框架 / 优于 LangChain 等 | 观察到的差异归因于**受控原型实例化下**的特定框架机制（framework mechanism contribution under the controlled prototype instantiation） |
| one_shot 与 full 排序质量无差异 | 在 one_shot 放弃的场景上不可估计 |
| one_shot 的 P@5 达到 1.000（优于 full） | 该值建立在 21 个场景的缩小分母上，属生存者偏差，不可与 full 的 37 直接比较 |
| 系统 grounding 率 100%，解释完全可靠 | well-formed 输入下由构造保证；鲁棒性证据来自独立的 fault-injection 套件 |
| ActiveSearchState 完整持久化 | active search 的身份被持久化，状态按需重建 |
| hybrid 模式 LLM 延迟约 0.6 ms | 那是 perf 套件里 mock provider 的计算耗时；真实 API 延迟中位 11.6 s、p95 20.4 s，见 §5.1 |
| 消融证明 memory 机制普遍有效 | 在本原型实例化与本场景集下，memory 机制的贡献集中于 memory-dependent 场景 |
| CI 含 0 说明无效应 | 方向已观察到，但不确定（小样本） |
| 结果可推广到真实招聘结果 | 合成目录与合成候选人、场景数有限，不外推到真实招聘结果 |
| 把 deterministic 与 hybrid 的数字放进同一张表、求平均或求总和 | 主实验与补充鲁棒性验证分开呈现；两者变体集、重复次数与后端都不同，合并数没有对应的实验设计（§6.1） |
| 用 hybrid 结果讨论 `profile_only` 或 `one_shot` | hybrid 只跑了 full / no_memory / no_context，**这两个基线没有 hybrid 对照**；相关论述只能引主实验（§6.1） |
| 去掉 context 在真实 LLM 下反而更好（因为 no_context 的 ndcg/hcsr 略高） | 小样本波动与 LLM 抽取差异造成的排序层面扰动；task_success 两侧同为 0.1667，Δcontext 的结论不变（§6.5） |
| 那次 `AttributeError` 崩溃是"LLM 返回了错误格式" | **是流水线未处理的输出形状**——模型返回的是语义正确的单元素列表，`validate_field` 判 ok=True 而修复循环只处理 ok=False（§6.4） |
| hybrid 与 deterministic 的差距主要由那个 bug 造成 | 该崩溃只占约 10%（0.008/0.079）；约 90% 是真实 LLM 行为，尤其 SC-G-01 / SC-G-02 的 3/3 系统性失败（§6.3） |
| 该崩溃已修复 / 已知问题已解决 | **尚未修复，是否修复并重跑仍待决定**；按已定位、已量化的已知限制来写（§6.4） |
| 378/378 全部由录制的模型输出重放 / replay 无任何例外 | 378 runs、0 differences，其中 **376** 命中录制响应、**2** 个 bundle 无录制可服务（纯规则抽取）、**1** 个因崩溃导致 bundle 不完整而无法重放（§10） |

Chapter 6/7 收尾时，Threats to Validity 四类（construct / internal / external / conclusion）已在报告 §12 生成，可直接对齐：construct = oracle 非人工判断、grounding 度量证据支撑而非感知质量；internal = deterministic mock 去除了 LLM 随机性但也未真正驱动模型，变体行为由单一代码路径上的 feature flags 控制；external = 小规模合成目录、场景数有限；conclusion = 小样本限制统计功效，重点在效应量、CI 与逐场景图。

---

## 14. 现在能写 / 还不能写

**现在就能写（数据已定稿）**：Chapter 3 全部、Chapter 4 全部、Chapter 5 的 dataset / scenario taxonomy / variants / metrics 定义 / statistical method / overall results / memory 与 context 子集 / clarification 子集 / fault-injection robustness / retrieval vs ranking error / error analysis / framework mechanism deltas，以及——**本轮新增可写**——Chapter 5 的 **deterministic vs hybrid 配置对比**与 **Latency 中的真实 LLM 成本**、Chapter 6 的**"deterministic 与 hybrid 差异"与 failure cases**。写这几部分前先读 §5 与 §6。

**必须等人工标注**：Chapter 5 的 Human annotation 与 Human-vs-oracle comparison、Chapter 3 中关于两位评分者与 agreement 的承诺。

**一处仍未定的事**：§6.4 那个 `AttributeError` 缺陷是否修复并重跑 hybrid **尚未决定**。若在定稿前决定修复，两个 experiment id 都会变，全文 id 与 headline 数字须重新核对；若决定不修，按 §6.4 的口径写成已知限制。**不要在决定之前把它写成任何一种既成事实。**
