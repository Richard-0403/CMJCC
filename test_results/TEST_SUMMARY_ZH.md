# CMJCC — 测试与评测总结（中文）

本文件夹汇总了 CMJCC 对话式职位推荐原型的测试结果与总结：（1）软件测试结果；
（2）由 `jobrec_eval` 评测流水线产生的 RQ4 评测实验（现已包含 per-constraint 合规、
no-match/澄清 的 precision-recall、根因错误分类、以及代表性案例分析）。

- 原型代码版本：`main` 分支（阶段 A–G + 评测增强）。
- 评测实验编号：`exp-8793b18de5b2`（确定性运行模式）。
- 本文件夹中的表/图/报告为 `evaluation/outputs/exp-8793b18de5b2/` 的副本，完全可复现。

---

## 1. 软件测试结果

| 测试套件 | 结果 |
|---|---|
| 单元（抽取、约束、记忆冲突、排序、claim 校验、config hash） | 通过 |
| 契约（schema 校验拒绝非法输入 / 未知枚举 / 多余字段） | 通过 |
| 集成（完整流水线、LLM 失败回退、多轮记忆） | 通过 |
| 端到端（FastAPI 接口） | 通过 |
| 黄金场景（10 个场景 + 消融差异断言） | 通过 |
| 性质测试（full 不选违反硬约束的职位；总分==贡献之和；claim 可解析；不超过 top-k） | 通过 |
| 评测单元（NDCG 手算、McNemar、Holm、bootstrap 种子） | 通过 |

- **合计：** 68 个测试通过；1 个 PostgreSQL 标记测试默认跳过
  （已单独在真实 PostgreSQL 15 上验证）。
- **覆盖率：** 整体约 83%；核心逻辑更高（CMJCC 91%、排序 98%、编排器 89%）。
  **规范：** `ruff` 全通过。**确定性：** 重复运行完全一致（方差 0）。
- **评测过程中修复的一个缺陷：** 当币种已由前一轮确定时（如先说"RM8000"、后说
  "4000 也可以"），澄清器不再重复追问币种；此前会触发一次多余的澄清。

---

## 2. 评测实验（RQ4）

- 设计：42 个带标签场景 × 5 种版本 × 3 次重复 = **630 次运行**，**0 次系统失败**。
  固定目录快照、提示词、随机种子；基准日期 2026-01-01；top-k = 5。场景期望已对照
  目录做质检（两个"多硬约束但联合不可行"的场景被重新标注为正确的 no-match）。
- 场景分布：完整 6、澄清 5、资料-对话冲突 5、偏好变化 12、多硬约束 5、软偏好权衡 4、
  模糊角色 2、无匹配 3。记忆依赖（≥中）16 个；上下文依赖（高）15 个。

### 2.1 各版本总体结果（场景均值）

| 版本 | NDCG@5 | P@5 | HCSR | 任务成功率 | grounding | Handoff |
|---|---|---|---|---|---|---|
| **full** | 0.951 | 0.973 | **1.000** | **1.000** | 1.000 | 1.000 |
| no_memory | 0.949 | 1.000 | 1.000 | 0.786 | 1.000 | 1.000 |
| one_shot | 0.949 | 1.000 | 1.000 | 0.786 | 1.000 | 1.000 |
| no_context | 0.700 | 0.558 | **0.571** | 0.310 | 1.000 | 1.000 |
| profile_only | 0.587 | 0.500 | 0.500 | 0.333 | 1.000 | 1.000 |

来源：`tables/variant_summary.csv`。

### 2.2 逐约束合规（推荐职位 vs 权威硬约束）

| 约束字段 | full | no_context | profile_only |
|---|---|---|---|
| 地点 | 1.000 | 0.600 | 0.073 |
| 最低薪资 | 1.000 | 0.800 | 0.642 |
| 工作模式 | 1.000 | 0.457 | 0.160 |
| 未过期 | 1.000 | 0.846 | 1.000 |

`no_context` 还通过"未知"放过了约 37% 的工作模式检查。来源：`tables/constraint_compliance.csv`。

### 2.3 职位上下文贡献（full vs no_context，上下文依赖场景）

| 指标 | full | no_context | Δ | 95% CI | p | n |
|---|---|---|---|---|---|---|
| HCSR | 1.000 | 0.480 | **+0.520** | [0.400, 0.660] | 0.002 | 10 |
| 任务成功率 | 1.000 | 0.000 | **+1.000** | [1.000, 1.000] | <0.001 | 15 |
| 每职位平均违规数 | 0.000 | 0.740 | **−0.740** | [−1.120, −0.460] | 0.002 | 10 |
| NDCG@5 | 0.914 | 0.655 | +0.259 | [0.156, 0.418] | 0.002 | 10 |

来源：`tables/context_contribution.csv`。

### 2.4 候选人记忆贡献（full vs no_memory，记忆依赖场景）

| 指标 | full | no_memory | Δ | 95% CI | p | n |
|---|---|---|---|---|---|---|
| 任务成功率 | 1.000 | 0.438 | **+0.562** | [0.312, 0.812] | <0.001（McNemar） | 16 |
| NDCG@5 | 0.939 | 0.925 | +0.014 | [0.000, 0.042] | 1.000 | 7 |

来源：`tables/memory_contribution.csv`。

### 2.5 无匹配与澄清正确性

- 无匹配：**full 的 precision = recall = F1 = 1.00**；`no_context` 与 `profile_only`
  的 recall = 0（因不执行硬约束过滤，无法正确判定 no-match）。来源：`tables/no_match_metrics.csv`。
- 澄清：`full` 在缺角色/模糊角色场景中正确澄清；`no_memory`/`one_shot` 还会（正确地）
  在"角色只在前一轮说明"的多轮场景中触发澄清。

### 2.6 根因错误分类（任务未成功的运行）

| 类别 | 占比 | 最受影响版本 |
|---|---|---|
| 缺少约束执行（消融） | 38.7% | no_context |
| 缺少对话证据（基线） | 37.3% | profile_only |
| 记忆缺失/过期（消融） | 24.0% | no_memory |

本次运行 `full` 没有任务失败。来源：`tables/error_taxonomy.csv`。
5 个代表性案例（记忆有用、上下文有用、正确 no-match、full 最难案例、claim 校验器）
见 `analysis_report.md` 第 9.1 节。

---

## 3. 主要发现

1. 完整架构满足全部指定硬约束（HCSR = 1.00），无未被证据支持的事实性 claim
   （grounding = 1.00），正确判定无匹配（F1 = 1.00），且所有 handoff 通过。
2. **职位上下文编排贡献最大**：移除后 HCSR 掉到 0.57，每个推荐职位平均新增约 0.74 个
   硬约束违规；在上下文依赖场景上任务成功率从 1.00 掉到 0.00。
3. **候选人记忆在多轮场景中明显有效**：记忆依赖场景任务成功率 +0.56（McNemar p < 0.001）。
4. 效应集中出现在预期的子集（依赖子集），与各组件承担其设计职责的解释一致。

---

## 4. 三项扩展的完成状态

- **(A) 可由数据直接推导的补充 —— 已完成。** 场景质检/重标、逐约束合规、
  no-match/澄清 precision-recall、案例分析、错误分类均已实现并写入报告。
- **(C) 人工标注支持 —— 已就绪，等待标签。** 流水线会导出标注模板
  （`evaluation/outputs/<exp>/annotation/`）。放入
  `relevance_labels_human.csv` / `claim_annotations_human.csv` 后，流水线会自动计算
  加权 Cohen's κ（相关性）、Cohen's κ（claim）与 oracle-vs-人工一致性。不伪造任何人工标签。
- **(B) 真实 LLM（hybrid）运行 —— 已接好，等待 API key。** 已提供接 Vector Engine
  （gpt-5.5，OpenAI 兼容）的 hybrid 配置（`configs/hybrid_vectorengine.yaml`）。设置
  `JOBREC_LLM_API_KEY`、`JOBREC_LLM_BASE_URL=https://api.vectorengine.ai/v1`、
  `JOBREC_LLM_MODEL=gpt-5.5`，用 `--config configs/hybrid_vectorengine.yaml` 运行即可让
  grounding / 抽取 / 延迟变成真实数字。

---

## 5. 局限性（论文需说明）

- 本次运行中**相关性由透明的自动 oracle 打分，而非人工评分者**；NDCG/P@5/MGR 衡量的是
  与确定性参考的一致性，尚未报告评分者一致性（构念效度威胁）——见 (C)。
- **grounding 与 handoff 在确定性后端下为 1.00 是"构造上必然"**（模板化解释只输出
  已校验的 claim）。它们是正确性保证，而非实验变量；在真实 LLM（B）下才有意义。
- 延迟仅为确定性计算开销（真实成本在 LLM）。响应轮次在此无区分度。
- 合成目录规模小、场景数量适中；结果不能外推到真实招聘效果。

---

## 6. 复现方式

```bash
pip install -e ".[dev,eval]"
python scripts/generate_raw_catalog.py --output data/raw/jobs.csv --count 200
python scripts/prepare_catalog.py --input data/raw/jobs.csv --out-dir data/processed
python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
python -m jobrec_eval.cli pipeline --repeats 3 --bootstrap-iters 5000   # 确定性
# 真实 LLM（需要 key）：
JOBREC_LLM_API_KEY=... JOBREC_LLM_BASE_URL=https://api.vectorengine.ai/v1 \
JOBREC_LLM_MODEL=gpt-5.5 \
  python -m jobrec_eval.cli pipeline --config configs/hybrid_vectorengine.yaml --repeats 3
pytest -m "not postgres"
```
