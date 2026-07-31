# CMJCC 论文正式结果 — 唯一数字来源

> 本文件是论文引用数字的**唯一**来源。任何其他文档（含 `docs/legacy/` 下的两份）中的
> 数值与实验 id 均已作废。所有数字直接读自 `final_release/`，未经转录。

## 1. 正式实验对

| | deterministic（主实验） | hybrid（补充鲁棒性验证） |
|---|---|---|
| experiment id | **`exp-e748800507ef`** | **`exp-6db1e87daed5`** |
| config | `configs/experiment_full.yaml` | `configs/hybrid_vectorengine.yaml` |
| 变体 × 场景 × 重复 | 5 × 42 × 1 = **210** | 3 × 42 × 3 = **378** |
| 完成 / 计划 / 崩溃 | 210 / 210 / **0** | 378 / 378 / **0** |
| verify | 两棵树 **OK** | 两棵树 **OK** |
| replay | **210/210，0 differences** | **378/378，0 differences** |
| 模型调用 | 0（mock，按设计） | **622**，0 失败，3,031,744 tokens |
| 墙钟耗时 | ≈2 min | **88.9 min**（其中 86.6 min 为 API 等待，97.4%） |

**共同身份**（两者逐字符相同，所以差异只来自后端）：

- `commit_hash` = **`f7970b81f653`**
- `execution_fingerprint` = **`f3ef9775f6b6d08a`**
- canonical oracle = **v3.0.0，declared，42/42 declared，0 system-derived**
  （`inputs_fingerprint b66a395e…`，`reference_fingerprint be950e06…`）

`git_dirty`：deterministic 为 `false`，hybrid 为 **`true`**。该标记**如实保留、未篡改**，
且**不表示** hybrid 用了被修改的代码 —— 见 `final_release/provenance.json` 的
`git_dirty_note`（成因是 deterministic 刚写出当时未被追踪的输出目录）。

## 2. 正式指标（variant summary）

`n` 是该指标实际取到值的场景数，**各指标不同**。

### deterministic `exp-e748800507ef`

| variant | NDCG@5 | P@5 | MGR | HCSR | Task Success | Grounding | n(NDCG) |
|---|---|---|---|---|---|---|---|
| **full** | **0.9115** | **0.9581** | 2.6959 | **0.9838** | **0.9762** | 1.0000 | 37 |
| no_memory | 0.9381 | 0.9714 | 2.7071 | 0.9714 | 0.7619 | 1.0000 | 28 |
| one_shot | 0.9174 | 0.9619 | 2.7048 | 0.9619 | 0.5952 | 1.0000 | 21 |
| no_context | 0.7191 | 0.6842 | 1.9842 | 0.6238 | 0.1667 | 1.0000 | 38 |
| profile_only | 0.5605 | 0.5583 | 1.6250 | 0.4786 | 0.1667 | 1.0000 | 24 |

### hybrid `exp-6db1e87daed5`

| variant | NDCG@5 | P@5 | MGR | HCSR | Task Success | Grounding | n(NDCG) |
|---|---|---|---|---|---|---|---|
| full | 0.9145 | 0.9730 | 2.7171 | 0.9579 | 0.8889 | 1.0000 | 37 |
| no_memory | 0.8891 | 0.9644 | 2.6489 | 0.9711 | 0.7381 | 1.0000 | 30 |
| no_context | 0.7012 | 0.6842 | 1.9632 | 0.6286 | 0.1667 | 1.0000 | 38 |

**排序指标的分母不同**，所以 `no_memory` 的 NDCG 高于 `full` **不能**读作"去掉记忆更好"：
那是**生存者偏差** —— 放弃了场景的变体不进入该指标的分母。跨变体比较请用 `task_success`，
它在每个场景上都有定义。

## 3. 消融分析（primary 家族，Holm 在家族内校正）

估计量：`task_success` 为**场景级二值**（重复按多数票折叠），其余为**重复平均后的场景均值**。

### 记忆机制（full − no_memory）

| 实验 | 子集 | Δ task_success | p | **p(Holm)** | n |
|---|---|---|---|---|---|
| deterministic | all | 0.2143 | 0.00391 | **0.02344** | 42 |
| deterministic | memory_dependent | **0.5625** | 0.00391 | **0.02344** | 16 |
| hybrid | all | 0.1429 | 0.03125 | **0.1875** | 42 |
| hybrid | memory_dependent | **0.3750** | 0.03125 | **0.1875** | 16 |

⚠️ **两个实验方向一致但显著性不同，必须分别表述：**

- **deterministic：Holm 校正后显著**（adjusted p = 0.0234）
- **hybrid：Holm 校正后不显著**（adjusted p = 0.188）

记忆机制在 NDCG@5 / HCSR 上均**不显著**（Holm 1.0），两个实验皆然。

### job-context 机制（full − no_context）

| 实验 | 子集 | 指标 | Δ | p | **p(Holm)** | n |
|---|---|---|---|---|---|---|
| deterministic | all | task_success | 0.8095 | <1e-5 | **<0.001** | 42 |
| deterministic | context_dependent | HCSR | 0.5000 | 0.00195 | **0.00977** | 10 |
| deterministic | context_dependent | task_success | 1.0000 | 6e-5 | **0.00037** | 15 |
| hybrid | all | task_success | 0.7143 | <1e-5 | **<0.001** | 42 |
| hybrid | context_dependent | HCSR | 0.4545 | 0.00195 | **0.00977** | 11 |
| hybrid | context_dependent | task_success | 1.0000 | 6e-5 | **0.00037** | 15 |

job-context 机制的贡献在**两个实验、多个指标上都稳健显著**，是全套结果中最强的一条。

## 4. 主要失败案例：SC-D-02

`full` **不再是满分**，原因集中于此。

- 话语：`"Business analyst in Penang, onsite only."` + `"Hybrid is fine too, at least RM4000."`
- 声明式参考答案：`work_modes = {onsite, hybrid}`，**hard**（依据 `only`）
- 系统自身将 `work_modes` 判为 **soft**，返回 5 个 job，**仅 2 个满足** → HCSR **0.4**，task_success **0**

**这类缺陷是自派生 oracle 结构上无法发现的**：旧 oracle 继承了系统自己的 soft 判断，
违规因此不可见。这是声明式 ground truth 带来的直接收益，也是论文可引用的具体局限。

## 5. 错误分类法

| deterministic（98 个 task 失败） | 数量 | hybrid（152 个） | 数量 |
|---|---|---|---|
| missing_constraint_enforcement | 35 | missing_constraint_enforcement | 102 |
| missing_dialogue_evidence | 35 | stale_or_missing_memory | 21 |
| missing_dialogue_continuation | 16 | other | 20 |
| stale_or_missing_memory | 9 | no_match_misclassification | 6 |
| other | 3 | no_context_other | 3 |

deterministic 的分母是 **98**,不是本文件早先写的 96。冻结报告本来就是对的
(`task-unsuccessful runs: 98`,分变体 full 1 / no_memory 10 / one_shot 17 /
no_context 35 / profile_only 35),`error_taxonomy.csv` 的百分比列也以 98 为分母
(35 / 98 = 35.7%)。这是本文件的转录错误,已记入 `final_release/ERRATA.md` E-1。

## 5.1 no-match 场景的表述范围

`no_match_metrics.csv` 中 full / no_memory / one_shot 三个变体的 no-match
precision / recall / F1 均为 **1.000**,分母 `no_match_expected = 5`。这个算术结论成立,
但**不能**据此说这 5 个场景都是硬约束联合不可满足。其中 2 个不是:

| 场景 | data-quality 告警 | 满足硬约束的 job |
|---|---|---|
| `SC-E-02` | `no_match_scenario_constraint_satisfiable` | 5 个(job-0021、job-0086、job-0089、job-0094、job-0169) |
| `SC-E-04` | 同上 | 1 个(job-0012) |

这些 job 全部落在请求的角色族之外。两个场景的类型标签是 `multiple_hard` 而非 `no_match` ——
只有 3 个场景是 `no_match` 类型,这就是报告的场景类型计数写 `no_match 3`、而 no-match 指标
分母是 5 的原因。报告的 data-quality 段落已如实列出 `no_match_scenario_constraint_satisfiable 2`。

**正确表述**:同时应用目标角色范围和硬约束后,不存在合格且相关的职位。

详见 `final_release/ERRATA.md` E-2。

## 6. 表格与产物路径

均在 `final_release/` 下，`<exp>` = `deterministic/exp-e748800507ef` 或 `hybrid/exp-6db1e87daed5`：

| 内容 | 路径 |
|---|---|
| 分析报告 | `<exp>/report/analysis_report.md` |
| variant summary | `<exp>/metrics/variant_summary.csv` |
| 消融贡献 | `<exp>/metrics/memory_contribution.csv`、`context_contribution.csv` |
| 配对统计 | `<exp>/statistics/paired_comparisons.csv` |
| 错误分类 | `<exp>/metrics/error_taxonomy.csv` |
| 澄清效率 | `<exp>/metrics/clarification_efficiency.csv` |
| 检索 / top-k / 抽取来源 | `<exp>/metrics/retrieval_metrics.csv` 等（报告 §5.6） |
| 图 | `<exp>/plots/*.png` |
| 冻结 oracle | `<exp>/manifests/canonical_oracle.json` |
| 分析计划 | `<exp>/manifests/analysis_plan.yaml` |
| 数据血缘 | `<exp>/audit/data_lineage.csv` |
| 身份与归档溯源 | `final_release/provenance.json` |

完整 bundle 归档（含 `normalized/`、model-call 记录、脱敏 raw responses）在 `dist/`，
**不入普通 git**；文件名、大小、SHA-256 与包含范围记录在 `provenance.json` 的 `bundle_archives`。

正式勘误记录在 `final_release/ERRATA.md`。冻结的 `analysis_report.md` **不作修改**，
保持生成时的逐字节原状，勘误另行记录而非事后改写。

## 6.1 两级验证方法

slim release 与完整 bundle 用的是不同机制,不要混用。

**slim release** —— 对 `final_release/checksums.json` 验证:

```
python scripts/verify_final_release.py
```

覆盖 **100** 个文件（含每个实验各自的 `checksums.json`，以及 `ERRATA.md`），双向检查:
记录的文件是否都在且未变，磁盘上的文件是否都被记录。`final_release/` 的文本由
`.gitattributes` 锁定为 LF，所以记录的哈希在任何平台检出后都成立，而不只在构建它的机器上成立。

**完整 bundle** 不在上述 manifest 覆盖范围内，也不入 git。分两步验证 —— 先验归档本身，再验其内容:

```
# 1. 归档本身，对 provenance.json 的 bundle_archives[].sha256
Get-FileHash dist\CMJCC_deterministic_exp-e748800507ef.zip -Algorithm SHA256
Get-FileHash dist\CMJCC_hybrid_exp-6db1e87daed5.zip        -Algorithm SHA256

# 2. 解压后的树，对 bundle 自带的 checksums.json，用项目自身的 verifier
python -m jobrec_eval.cli verify <解压后的 analysis 目录>
python -m jobrec_eval.cli replay <解压后的 run bundle 目录>
```

第 2 步分别覆盖 **5305**（deterministic）和 **9505**（hybrid）个受校验文件。

## 7. 允许的表述

- job-context 机制的贡献在两个实验上**稳健显著**（Holm 校正后仍显著）。
- 记忆机制：**deterministic 上 Holm 校正后显著**（adjusted p = 0.0234，memory-dependent 子集 Δ = 0.5625）。
- 记忆机制：**hybrid 上方向一致、效应量可观（memory-dependent 子集 Δ = 0.375），但 Holm 校正后不显著（adjusted p = 0.188）**。
- 相关性由**声明式 canonical oracle v3.0.0** 评分，参考答案是场景文件的一部分、
  独立于变体与随机重复；但它**仍非人工标注**，是透明代理。
- 差异归因于**受控原型实例化下的特定机制**，不主张对任何外部框架的普遍优越性。
- deterministic 与 hybrid 的差异**只来自后端**（相同 commit 与 execution fingerprint）。
- `full` 在 deterministic 上**不是满分**（HCSR 0.9838、task_success 0.9762），SC-D-02 是主要失败案例。
- no-match 场景可写为：**同时应用目标角色范围和硬约束后，不存在合格且相关的职位**。

## 8. 禁止的表述

- ❌ 记忆效应"统计显著"**而不指明实验** —— hybrid 上不显著。
- ❌ 引用任何旧数字（如 full NDCG 0.9585、HCSR 1.000、task_success 1.000、Holm 0.0469）。
- ❌ 用 `no_memory` 的 NDCG 高于 `full` 论证"记忆无用" —— 那是生存者偏差，分母不同。
- ❌ 把 `grounding = 1.000` 说成"解释质量好" —— 解释文本**从不由模型生成**，
  claim 由证据装配，所以验证器无可拒绝之物。这是**构造使然**，两个后端皆然。
- ❌ 声称 oracle 是人工标注或独立 ground truth。
- ❌ 把 hybrid 的 `git_dirty=true` 解释为代码被修改。
- ❌ 引用长期记忆验证集 `exp-1fe49fedbd22` 的 NDCG / HCSR / 任何排序质量数字（见 §10）。
- ❌ 引用 §9 列出的任何被取代实验。
- ❌ 把 5 个 no-match 场景**整体**概括为"硬约束联合不可满足" —— `SC-E-02` 与 `SC-E-04`
  的 no-match 来自角色族不匹配,不是约束不可满足（见 §5.1）。引用冻结报告中
  "Correct no-match (SC-E-02)" 案例时必须带上这一限定。
- ❌ 引用 deterministic task 失败数为 96 —— 正确值是 **98**（见 §5）。

## 9. 被取代、不可引用的实验

7 个,与 `final_release/provenance.json` 的 `superseded_and_not_citable` 和
`superseded_details` 逐项一致。

| experiment id | 原因 | 当前存储状态 |
|---|---|---|
| `exp-87aec1bc99dc` | 早于 v3 declared oracle | release v1 后已清理;报告目录可从 tag `cmjcc-thesis-release-v1` 取回,run 树不可 |
| `exp-197f6aacc171` | 早于 v3 declared oracle | 同上;其 runs zip 未被跟踪,不可恢复 |
| `exp-06cc34defe39` | 早于 v3 declared oracle | release v1 后已清理;run 树与 runs zip 均未跟踪,**不可恢复** |
| `exp-f90573008bdb` | 仅可作为可复现性证据，**不得用于结果** | **保留在盘**,replay diff 在 `artifacts/reports/` |
| `exp-8793b18de5b2` | 修复前产物 | release v1 后已清理,连同 `test_results/`;两者均可从 tag `cmjcc-thesis-release-v1` 取回 |
| `exp-515b63d6a656` | 过程中间产物 | 不在盘上,未保留任何树或归档 |
| `exp-301060a1899d` | 过程中间产物 | 不在盘上,未保留任何树或归档 |

这三个早于 v3 oracle 的实验**在结构上没有对比价值**,不只是"旧":它们的分数来自继承了被评估系统
自身 soft/hard 判断的旧 oracle,而这正是 v3 要消除的混淆。拿它们与正式实验对比等于比较两套不同的
ground truth。

## 10. 长期记忆验证集 `exp-1fe49fedbd22`

**仅作补充工程验证（supplementary engineering validation）。**
其 oracle 仍为 `system_derived_pass`（0/5 declared），因此：

- ❌ **不得引用**其 NDCG、HCSR 或任何排序质量数字。
- ✅ **可引用**的工程验证：长期写回（persistence）、作用域隔离（`this time only` 不被继承）、
  跨会话继承与累积（cross-session recovery）、R4.11 冲突守卫的实际行为。

已归档为 10 runs，两棵 checksum 树 verify OK。关键观测：`SC-LT-01` 达到 version 2 且第二会话
的检索带入 `hybrid`（其话语从未提及工作模式）；`SC-LT-03` 达到 version 3 且两个持久值都生效；
`full` 与 `no_memory` **恰好只在 SC-LT-01 与 SC-LT-03 上不同**，说明该验证集对被验证机制敏感、
在别处不敏感。

## 11. 人工标注结果

标注在正式实验对上完成,**未重跑任何实验** —— oracle 与 claim validator 都是分析时施加的。
分析输出在 `evaluation/outputs_human/`,封存的 `evaluation/outputs*/` 与两个归档未被写入。

完整可审计证据包:`evaluation/annotation_workspace/`(rater 级标签、adjudicated gold、
实际施加的 rubric、claim 正文与其证据、provenance、checksums)。

### 11.1 标注规模与信度

| | relevance | claim |
|---|---|---|
| 判定单元 | `(scenario_id, job_id)` | `(run_id, claim_id)` |
| 行数 | **368**(覆盖 42 场景、82 职位) | **11197** |
| 两位标注者原始一致 | 0.7962 | 0.9804 |
| 两位标注者 kappa | **0.9364**(quadratic weighted) | **0.9389**(Cohen) |
| 分歧数 | 75 | 219 |
| 已裁决 / 遗留 | 75 / **0** | 219 / **0** |
| 可用 gold | 368(100%) | 11197(100%) |

`adjudication_source = adjudicated_column`。分歧一律显式裁决,**未使用**取平均的
legacy fallback。

### 11.2 自动 oracle vs 人工(relevance)

> **本节数字对应 pre-fix 的 oracle(封存实验对所用)。** guaranteed-minimum 薪资修正之后
> 的口径见 §11.5,两者不可混用。

oracle-vs-human quadratic weighted kappa = **0.7517**;精确同分率 0.7283;
平均绝对差 0.4457。方向明确:**oracle 比人工宽松** —— 更宽 95 例、更严 5 例、相同 268 例。

```
oracle(行) × 人工(列)
            human 0   1   2   3
oracle 0        137   2   1   0
oracle 1          0   1   1   0
oracle 2         11  10  32   1
oracle 3         26   0  48  98
```

`full` 变体在两个后端上的指标变化:

| | deterministic | | | hybrid | | |
|---|---|---|---|---|---|---|
| 指标 | oracle | 人工 | Δ | oracle | 人工 | Δ |
| NDCG@5 | 0.9115 | 0.8745 | **−0.0369** | 0.9145 | 0.8942 | **−0.0203** |
| P@5 | 0.9581 | 0.7703 | **−0.1878** | 0.9730 | 0.8000 | **−0.1730** |
| MGR | 2.6959 | 2.0896 | **−0.6063** | 2.7171 | 2.1414 | **−0.5757** |

**NDCG 几乎不动而 P@5 与 MGR 大幅下降**,这是一致的可解释模式:排序**次序**基本正确,
但人工认为 top-5 中真正相关的更少 —— oracle 高估的是绝对相关性,不是排序能力。
`no_context` 的 NDCG 变化仅 −0.002(deterministic)/ +0.013(hybrid),
说明该消融的排序确实差,人工判断予以确认。

分歧主因记录在 `annotation_workspace/RUBRIC.md`:话语中的 "at least RM4000",人工要求职位
**起薪** `salary_min_monthly_myr ≥ 4000`,而自动 oracle 接受薪资区间与阈值**重叠**即通过。

### 11.3 claim validator:κ = 0 是退化值,不是随机水平一致

**这一条必须按下述方式表述。**

`validator` 列在全部 **11197** 条上取值恒为 `1`(supported),`distinct values = [1]`。
Cohen's kappa 要求双边都有方差,常量预测器的 kappa 在实质上**无定义**,实现返回
`0.000`。冻结报告 §4 会照打这个 `0.000`,**不得**据此解读为"一致度不优于随机"。

真实情况:

| 量 | 值 |
|---|---|
| validator vs 人工 **原始一致度** | **0.7902** |
| 人工判定 unsupported | **2349 / 11197 = 20.98%** |
| validator 标记 unsupported | **0** |
| **unsupported detection rate** | **0.0** |

```
validator(行) × 人工(列)
              human 0    human 1
validator 0         0          0
validator 1      2349       8848
```

validator 的实质缺陷是**检出为零**,而不是一致度低。分类型看,缺陷是结构性的:

| claim_type | n | supported rate(人工) | unsupported |
|---|---|---|---|
| `skill_gap` | 1883 | **0.0000** | 1883 |
| `no_match_reason` | 156 | **0.0000** | 156 |
| `ranking_reason` | 7904 | 0.9615 | 304 |
| `candidate_preference` | 1254 | 0.9952 | 6 |

**2349 条中有 275 条的性质与其余不同,不可合并陈述。** 那 275 条文本是
`Salary meets your stated minimum.`,引用的证据是**未归一化**的原始字段
(如 `job_posting:salary_min=1350.0`,单位为 SGD/月),而系统比较用的是归一化后的
`salary_min_monthly_myr = 4725.0`。候选人要求 4000 MYR,**结论为真**;标注者依据所引证据
判定不支持,依规程也**正确**。所以这一类是「结论正确,但其自身引用的证据与之矛盾」,
与 `skill_gap` / `no_match_reason` 那种「结论无法从任何证据推出」不是同一类失败。

因此可引用的构成是:

| 性质 | 数量 |
|---|---|
| 结论无法由证据推出(`skill_gap` 1883 + `no_match_reason` 156) | **2039** |
| 结论正确但证据引用了错误字段(薪资) | **275** |
| 使用未确认或冲突证据 | 35 |

只有后两类可由证据层修复而不改变系统结论;第一类需要改 claim 模板本身。区分这两者对
论文很重要:「系统得出错误结论」与「系统得出正确结论却无法为其举证」只有前者影响排序质量。

前两类整类判 unsupported,原因是**结构性证据缺口**,不是逐条例外:

- `skill_gap` 文本形如 `Gap: the role requires excel, which is not in your listed skills.`,
  唯一证据是 `job_posting:required_skills=[...]` —— 只能证明职位要求该技能,
  **不能**证明它不在候选人技能列表中。支撑该否定性断言所需的候选人技能证据缺失。
- `no_match_reason` 文本形如 `Your hard requirement on target roles limits the results.`,
  证据仅证明约束存在,**不能**证明它导致结果受限 —— 因果断言,证据只到相关性。

这为 §8 中"`grounding = 1.000` 是构造使然"提供了量化人工证据。

### 11.4 本节的允许与禁止表述

允许:

- 人工标注者间信度高(relevance weighted κ = 0.9364,claim κ = 0.9389),分歧全部裁决,
  无遗留。
- 自动 oracle 与人工的一致度为 weighted κ = **0.7517**,方向是 oracle 系统性更宽松。
- 排序次序稳健(ΔNDCG@5 = −0.0369 / −0.0203),绝对相关性被高估
  (ΔP@5 ≈ −0.18,ΔMGR ≈ −0.58)。
- claim validator 对人工认定的 unsupported claim **检出率为 0**,人工 unsupported 率
  **20.98%**,原始一致度 0.7902。

禁止:

- ❌ 把 `validator_vs_human_kappa = 0.000` 解释为 chance-level agreement 或
  "validator 与人工毫无关联" —— 它是**全 1 常量预测**导致的退化值,必须同时给出
  原始一致度 0.7902、unsupported 率 20.98% 与 detection rate 0.0。
- ❌ 把人工列的排序指标当作**无偏效应估计**。人工标签只覆盖实际返回的 368 对,
  而 oracle 评分覆盖全目录,NDCG 的 ideal DCG 因此在不同池上计算;该 Δ 同时混入了
  标签来源变化与**标签宇宙变化**,只能作为一致性诊断。
- ❌ 用人工标签解读 retrieval recall。`retrieval_metrics.csv` 在两种模式下**均**使用
  自动 oracle,因为 recall 需要全目录标签宇宙;这一点记录在
  `manifests/analysis_plan.yaml` 的 `retrieval_recall_relevance_source`。
- ❌ 声称人工标注消除了 oracle 的构念效度威胁 —— 它**量化**了该威胁,标注者仍是作者。

### 11.5 薪资口径修正后的 oracle 一致度

正式实验对使用的 oracle 接受**薪资区间重叠**即满足最低薪要求;修正为 **guaranteed
minimum**(职位自身下限须达到阈值)后,在同一批 368 个人工裁决对上:

| 划分 | 场景 | 判定对 | κ(修正前) | κ(修正后) | 95% CI | 等级变动 |
|---|---|---|---|---|---|---|
| dev | 29 | 246 | 0.7398 | **0.9556** | [0.9409, 0.9676] | 27 |
| holdout | 13 | 122 | 0.7777 | **0.9426** | [0.9089, 0.9669] | 10 |
| 全部 | 42 | 368 | 0.7517 | **0.9516** | [0.9390, 0.9630] | 37 |

划分是**场景级**(判定单元不是 pair —— 同一场景内的 pair 共享候选人档案与声明 reference,
按 pair 划分会让近重复样本出现在两侧),seed `20260731` 已冻结并提交,记录在
`evaluation/annotation_workspace/agreement/agreement.json`。

**这不是预注册留出估计。** 薪资修正是在检查**全部** 368 对的分歧之后做的,两侧都已被看过,
事后划分无法追认为留出集。可以据此说的是两点:

- 该修正**没有任何拟合参数** —— 它把一条比较规则换成 rubric 里写明的另一条。dev 与 holdout
  同步移动、CI 大幅重叠,说明改进来自**规则本身**而非个别对子;两侧分离才是过拟合的警号。
- 划分已冻结,**下一次** oracle 改动可以在未见过的一侧评估。

**剩余分歧 63 / 368,其中 58 条只差一个等级**(人工 2 / oracle 3),零硬约束违反。原因是
已知的 rubric 差异:oracle 的 grade 3 阈值只看 role fit 与 required-skill coverage,而人工
rubric 的 grade 3 还要求 location / work mode / experience 也合适。这是 **3 与 2 的边界校准**,
不是效度失败 —— 它从不把相关翻转成不相关,这也解释了为什么 ΔNDCG@5 很小而 ΔMGR 很大:
系统性的 3→2 位移改变分级幅度,几乎不改变排序次序。
