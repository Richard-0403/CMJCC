# 人工标注证据包

CMJCC 人工相关性标注与 claim 标注的完整可审计记录。所有输入均读自已封存的正式实验对
`exp-e748800507ef`(deterministic)与 `exp-6db1e87daed5`(hybrid),
**没有重跑任何实验,也没有写入任何封存产物**。

由 `scripts/build_annotation_workspace.py --write` 生成,可重建。

## 布局

```
labels/
  relevance_raters.csv        368 行：两位标注者原始标签 + adjudicated + gold + 与 oracle 的差
  relevance_adjudicated.csv   最终 gold，可直接作为标签表消费的形状
  claim_raters.csv          11197 行：同上，附 gold 是否与 validator 一致
  claim_adjudicated.csv       最终 gold
reference/
  claim_texts.csv           11197 行：claim 正文 + 类型 + 展开后的证据摘要
  claim_evidence.csv          每条 (claim, evidence) 一行，含证据全字段与是否可解析
  scenarios.jsonl             42 场景：turns 与声明式 reference
  canonical_oracle_scenarios.json  冻结 oracle 产物
  jobs.jsonl / jobs.csv       200 个职位
  relevance_labels_oracle.csv 自动 oracle 打分，含 role_fit / skill_fit / 硬约束违反
  claims_*.csv                claim 元数据（不含正文）
  runs_index_*.csv            run_id -> run_dir 映射
RUBRIC.md                     实际施加的判定标准
provenance.json               来源、计数、校验结论
checksums.json                本目录的 SHA-256 清单
```

## gold 的确定规则

`adjudicated` 有值即为 gold;否则两位标注者一致时以其共同值为 gold;否则该行
**unadjudicated**,不参与任何统计。分歧从不取平均。`labels/*_raters.csv` 的
`gold_source` 列逐行记录走的是哪一条。

## 校验

`provenance.json` 的 `verification` 块记录构建时的结论:gold 是否完整覆盖、
未裁决数是否为零、取值是否越界、每条 claim 是否都有正文。
`checksums.json` 覆盖本目录除自身以外的全部文件。

## 已知的口径差异

人工判定比自动 oracle 严格:368 对中人工更严 95 例、更宽 5 例。主要来源是薪资口径
(见 `RUBRIC.md`)。claim validator 对全部 11197 条判 supported,人工判定其中
约 21% 不被支持,主要集中在 `skill_gap` 与 `no_match_reason` 两个结构性缺陷类型。
