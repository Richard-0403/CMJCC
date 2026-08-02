# CMJCC 论文正式结果 — 唯一数字来源（release v2）

> 本文件是论文引用数字的**唯一**来源。所有数字直接读自 `final_release_v2/`，未经手工转录。
>
> **`final_release/`（v1）连同 `docs/legacy/` 下的两份文档，以及本文件的历史版本，其数值与
> 实验 id 均已作废。** v1 由代码 0.1.0 对 canonical oracle **1.0.0** 产出，本轮是代码 0.2.0 对
> canonical oracle **4.0.0**，所有由分级标签推导的指标（NDCG@5、P@5、MGR、retrieval recall）
> 测的是不同的 ground truth，**不可与 v1 并列引用**。v1 保留在原处，因为既往发布是记录而不是
> 需要抹除的错误 —— 但它不是本论文的结果。

引用前请先读 `final_release_v2/provenance.json` 的 `known_limitations`（本文件第 6 节转述）。

## 1. 正式实验对

| | deterministic | hybrid |
|---|---|---|
| experiment id | **`exp-40a9cd647575`** | **`exp-2b33b808a0f8`** |
| config | `configs/experiment_full.yaml` | `configs/hybrid_vectorengine.yaml` |
| 变体 × 场景 × 重复 | 5 × 42 × 1 = **210** | 3 × 42 × 3 = **378** |
| 完成 / 计划 / 崩溃 | 210 / 210 / **0** | 378 / 378 / **0** |
| replay | **210/210 identical，0 differences** | **378/378 identical，0 differences** |
| checksums | **OK** | **OK** |
| 并发度 | 1 | **20** |
| 模型调用 | 0（mock，按设计） | **621 logical calls / 885 HTTP 尝试** |

合计 **588 runs**。

**共同身份**（两臂逐字符相同，所以差异只来自后端）：

- `commit_hash` = **`40ded11a8222`**，tag `cmjcc-thesis-execution-v2`
- `code_version` = **0.2.0**
- `git_dirty` = **false（两臂皆是）**
- `execution_fingerprint` 与 `source_fingerprint` 两臂一致
- canonical oracle = **v4.0.0**

## 2. 正式指标（自动 oracle 分级，主结果）

`n` 是该指标实际取到值的场景数，各指标不同；下表 `n` 为 NDCG@5 的。

### deterministic `exp-40a9cd647575`

| variant | NDCG@5 | P@5 | MGR | HCSR | Task Success | Grounding | n |
|---|---|---|---|---|---|---|---|
| **full** | **0.9570** | **0.9946** | **2.7459** | **1.0000** | **1.0000** | 1.0000 | 37 |
| no_memory | 0.9018 | 0.9429 | 2.6000 | 0.9429 | 0.6905 | 1.0000 | 28 |
| one_shot | 0.9055 | 0.9524 | 2.6381 | 0.9524 | 0.5714 | 1.0000 | 21 |
| no_context | 0.6417 | 0.5842 | 1.6947 | 0.5333 | 0.1190 | 1.0000 | 38 |
| profile_only | 0.4107 | 0.4250 | 1.2583 | 0.3643 | 0.0952 | 1.0000 | 24 |

### hybrid `exp-2b33b808a0f8`

| variant | NDCG@5 | P@5 | MGR | HCSR | Task Success | Grounding | n |
|---|---|---|---|---|---|---|---|
| **full** | **0.9233** | **0.9856** | **2.6883** | **1.0000** | **0.9365** | 0.7028 | 37 |
| no_memory | 0.8714 | 0.9402 | 2.5724 | 0.9471 | 0.6667 | 0.8711 | 29 |
| no_context | 0.6275 | 0.5895 | 1.6895 | 0.5429 | 0.1190 | 0.7114 | 38 |

两臂在 no_context 上几乎重合（NDCG 0.6417 vs 0.6275，task_success 两者皆 **0.1190**），而
deterministic 臂**没有任何模型调用**，因此不可能带端点混杂 —— 这是 no_context 的弱表现属于
消融本身、而非运行期扰动的独立佐证。

## 3. 人工标注与一致性

覆盖率均达到预注册的 **100%**，因此 κ 与人工排名指标**已解除抑制**
（`human_metrics_withheld_reason` 为 None）。

| | deterministic | hybrid |
|---|---|---|
| claim 标注单位 | annotation_signature（命题） | 同 |
| claim universe | **694** | **588**（574 delivered + 147 中抽样 14 dropped） |
| claim 覆盖率 | **100%**（694/694） | **100%**（588/588） |
| claim Cohen's κ | **0.8154** | **0.7950** |
| claim 原始一致率 | 99.42% | 98.98% |
| relevance n | 390 | 396 |
| relevance 两标注者加权 κ | **0.9308** | **0.9280** |
| oracle vs 人工加权 κ | **0.9197** | **0.9134** |
| 分歧 / 已裁定 | 12 / **12** | 18 / **18** |

标注单位是 `annotation_signature` 而非 `claim_id`：后者是**渲染后句子**的摘要，同一句话在不同
数值下会共用一个 id。本轮 claim_id 数分别为 416 与 366，若按其去重会把 694/588 个命题压成
416/366，即约 40% 的命题永远不会被作为自身判断。

裁定规则：两人一致时以其共同标签为 gold；分歧仅接受**已记录**的第三方裁定，未裁定的分歧被
**丢弃并计数**，绝不折中平均。30 项分歧全部由第三方（`ADJUDICATOR-C`）裁定。

relevance 标签复用了 v1 的可用部分（判断对象是场景与岗位，与系统版本无关），仅对覆盖缺口
增量补标 22（deterministic）与 28（hybrid）项。**claim 标签未复用** —— 它与当前 signature
universe 重合为 0，因为 P0-4/P0-5 改动了 claim 的 predicate 与文本。

### 自动 oracle 与人工分级的差异（`metrics/relevance_source_comparison.csv`）

| hybrid variant | 指标 | oracle | 人工 | 差值 |
|---|---|---|---|---|
| full | NDCG@5 | 0.9233 | 0.9418 | **+0.0185** |
| full | P@5 | 0.9856 | 0.8811 | **−0.1045** |
| full | MGR | 2.6883 | 2.3730 | **−0.3153** |
| no_memory | NDCG@5 | 0.8714 | 0.8670 | −0.0044 |
| no_memory | MGR | 2.5724 | 2.2575 | −0.3149 |
| no_context | NDCG@5 | 0.6275 | 0.6946 | +0.0671 |
| no_context | MGR | 1.6895 | 1.6396 | −0.0498 |

跨两臂、跨所有变体的一致模式：**人工的 NDCG@5 略高，但 P@5 与 MGR 明显低**。人工给的绝对
分数更严（MGR 低约 0.30），但对"把好岗位排在前面"这件事的评价高于 oracle。

## 4. Hybrid 端点行为（对照预注册阈值）

阈值在批次开始**之前**固定，通过 126-run pilot 验证后才批准正式实验。

| 指标 | 阈值 | 实测 |
|---|---|---|
| final fallback call rate | ≤ 1% | **0.97%**（6 / 621 logical calls） |
| 受影响 run 率 | ≤ 2% | **1.85%**（7 / 378） |
| logical call 成功率 | — | **99.03%** |
| retry 恢复率 | — | **96.32%**（157 / 163） |
| 崩溃 | 0 | **0** |
| legacy rule reparse | 0 | **0** |
| evidence 重复 / turn 漂移 | 0 | **0** |

621 logical calls、885 HTTP 尝试、264 次重试尝试；265 次失败尝试**全部为 `transport_error`**，
**0 次 JSON 解析错误**。结构化输出使用 OpenAI 兼容的 **JSON object mode**
（`response_format: {"type":"json_object"}`），185/185 次抽取调用均携带该约束、**0 次被端点拒绝**；
未采用 strict `json_schema`。

`system_fingerprint`：该端点从不返回（885/885 条记录均缺失），如实记录为
`system_fingerprint_available: false`，未伪造。

retry 策略写入 manifest：`max_retries: 4`，固定退避 `[2, 5, 10, 20]` 秒，单次超时 90 秒。
观测到的 retry/fallback 汇总也写入 manifest（`llm_call_summary`），与
`scripts/diagnose_llm_fallbacks.py` 共用同一实现，两者不可能给出不同的数。

## 5. 延迟

**并发运行所得延迟不得直接解释为单请求延迟。** Hybrid 臂以并发 20 运行，其
`total_latency_ms` 与各分位数是墙钟且叠加了重试。

单请求延迟的可引用来源是串行子实验 **`exp-e63f05ad75bb`**（42 runs，full 变体，并发 1）。

| | 串行（并发 1） | 并发 20 |
|---|---|---|
| 单次成功 HTTP 尝试延迟中位 | **23,438 ms** | 22,479 ms |
| 同 p90 | 54,172 ms | 49,872 ms |
| 每 run `total` 中位（full） | 35,080 ms | 23,957 ms |

**并发 20 并未抬高单请求延迟**（中位数还略低），瓶颈是端点自身约 23 秒的处理时间而非排队。
每 run 总延迟串行反而更高，是因为串行子实验正好跑在端点恶化窗口（66 logical calls 中 19 次
重试恢复），叠加了退避等待 —— 两次测量都带端点抖动，**都不是干净基线**。

本地各组件延迟极小且稳定（explanation 1.6 ms、filtering 1.8 ms、ranking 6–15 ms、
retrieval 6–7 ms、memory_merge 0.5 ms），端到端延迟几乎全部由 `intent_extraction` 的远程调用
构成。

## 6. 必须与数字一同引用的局限

完整表述见 `final_release_v2/provenance.json` → `known_limitations`。

1. **Hybrid fallback 在变体间分布不均，属执行顺序假象。** no_context 6/126（4.76%）、
   full 1/126（0.79%）、no_memory 0/126。run plan 是变体优先，三个变体占据批次时间轴的不同
   窗口（full 位置 0–169、no_memory 115–288、no_context 240–377），而端点在这 30 分钟内单调
   恶化（按批次六等分的失败尝试数：2、14、38、53、88、70），6 个受影响的 no_context run 全部
   落在位置 280–317。**对结论影响可忽略**：剔除这 7 个 run 后 no_context 的 ndcg_at_5
   +0.0065、task_success −0.0107、hcsr +0.0038、precision_at_5 +0.0068，全部 <0.011 且方向
   有正有负。deterministic 臂无模型调用却给出几乎相同的 no_context 数值，独立佐证该弱表现
   为真。后续批次应把 plan 改为交错提交。
2. **Hybrid 臂延迟为并发墙钟**，见第 5 节。
3. **`validator_vs_human_kappa` 近零系构造所致，非不一致。** 0.0（deterministic）与
   −0.014（hybrid）。校验器只交付它判过的内容，delivered 集合被待测对象本身过滤，负例极少；
   deterministic 的混淆矩阵有整整两格为 0（校验器=支撑：人工=1 为 681、人工=0 为 9；
   校验器=不支撑：两者皆 0），原始一致率 99.4%。可引用的是比率：**校验器假阳性率
   9/690 = 1.30%（deterministic）、12/568 = 2.11%（hybrid）**。hybrid 之所以有负例，是因为
   抽样框纳入了 14 个被撤回（dropped）的 claim，而两位标注者**一致认为这 14 项证据都是支撑
   的** —— 即 14/14 的假阴性，样本仅 14 项，是"校验器偏保守"的方向性证据，不是总体比率。
4. **标注耗时未记录。** 标签通过表格文件收集而非 web UI，`annotation_manifest.json` 记录
   `timed_annotations: 0`、`median_duration_ms: null`。本 release 不支持任何关于每项标注耗时
   的论断。
5. **标注者量表未用满。** 0–3 分的 relevance 量表上两位标注者都没用过 3，标注者 2 也从未
   用过 0。这是新增 50 个 delta 对的未加权 κ 偏低（0.31–0.38）而全量 390/396 对的加权 κ 达
   0.93 的原因：分歧几乎全是相邻分值，30 项分歧的差距**全部为 1**。

## 7. 产物与核验

```
final_release_v2/
├─ deterministic/exp-40a9cd647575/   metrics statistics plots report manifests audit
│                                    audit_evidence/  human_annotations/
│                                    run_bundle_provenance/
├─ hybrid/exp-2b33b808a0f8/          同上
├─ latency_serial/exp-e63f05ad75bb/  串行延迟子实验
├─ inputs/                           scenarios、canonical oracle、configs、合并后人工标签
├─ provenance.json / README.md / checksums.json
```

152 个文件，16.8 MB。核验：

```
python scripts/verify_release_v2.py     # 151 recorded | missing 0 | changed 0 | unrecorded 0
python scripts/preflight.py             # 8/8
```

未纳入（与 v1 同一原则）：`normalized/` 表与 run bundles（约 290 MB，可从 bundle 重建），
以及标注工作库 `annotation.sqlite3`（其归档形式 `human_annotations.jsonl` 已包含在内）。
