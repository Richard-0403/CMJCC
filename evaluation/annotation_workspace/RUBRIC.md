# 标注规程(rubric)

本文件记录**实际施加**的判定标准,供复核与论文方法学章节引用。

## 通用规则

- `rater_1`、`rater_2` 两列必须都有值。缺任一列的行会被流程整行丢弃。
- 两位标注者不一致时,`adjudicated` 必填。留空的分歧行被排除,**不取平均** ——
  未裁决的分歧不允许当作人工结论发表。
- `oracle_grade`(relevance)与 `validator`(claim)是自动基线,标注过程中不得修改。

## relevance:0-3 分级

判定对象是 `(scenario_id, job_id)`:在该场景所述需求下,这个职位对候选人的相关性。

依据:
- 场景 `turns`(候选人实际说的话)与 `reference`(声明式权威答案)。
  `reference.hard` 列出硬约束,`reference.unknown` 规定字段缺失时算 pass 还是 fail。
- 职位记录的 `role_family`、`required_skills`、`city`、`work_mode`、
  `salary_min_monthly_myr` / `salary_max_monthly_myr`、`min_years_experience`、
  `application_deadline`、`is_active`。

分级含义:
- `3` 完全符合:角色、地点、工作模式、薪资、经验均满足,无硬约束违反。
- `2` 基本符合:硬约束满足,soft 偏好有一项不理想。
- `1` 勉强相关:角色族相关但多项偏好不符。
- `0` 不相关或违反硬约束。

**薪资口径**:话语中的"at least RM4000"按 `salary_min_monthly_myr >= 4000` 判定,
即要求职位的**起薪**达标,而非薪资区间与阈值重叠即可。这与自动 oracle 的口径不同,
是两者分歧的主要来源之一,已在结果中如实报告。

## claim:1 = supported / 0 = unsupported

判定对象是 `(run_id, claim_id)`:该 claim 陈述的内容,是否**被它自己列出的证据支持**。

- 证据不足以支撑陈述 → `0`。
- 证据与陈述不符 → `0`。
- 引用的证据无法解析(`claim_evidence.csv` 中 `resolved = False`)→ `0`。

两类系统性判 `0` 的情形,理由记录如下,不是逐条例外而是结构性缺陷:

**`skill_gap`** 文本形如
`Gap: the role requires excel, which is not in your listed skills.`
唯一证据是 `job_posting:required_skills=[...]`。该证据只能证明职位要求某技能,
**不能**证明它不在候选人技能列表中。支撑这个否定性断言需要候选人技能的证据,而它缺失。
该类型全部结构相同,因此一致判 `0`。

**`no_match_reason`** 文本形如
`Your hard requirement on target roles limits the results.`
证据仅证明该约束存在,**不能**证明它导致了结果受限 —— 这是因果断言,证据只到相关性。
该类型全部结构相同,因此一致判 `0`。

`ranking_reason` 与 `candidate_preference` 逐条判断,判定结果随证据充分程度变化,
不是一刀切。
