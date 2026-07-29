# CMJCC 论文写作指导（第 5–7 章）

> **数字一律见 [`THESIS_OFFICIAL_RESULTS.md`](THESIS_OFFICIAL_RESULTS.md)。**
> 本文件只管"怎么写才成立"，不重复任何数值 —— 数值只有一处来源，避免两份文档漂移。
>
> 旧版（写于 canonical oracle v3.0.0 之前，含已作废的 id 与数字）：
> `docs/legacy/CMJCC_Thesis_Writing_Guide.SUPERSEDED.md`。

---

## 0. 引用纪律

只有两个正式产物可引用：deterministic **`exp-e748800507ef`** 与 hybrid
**`exp-6db1e87daed5`**。被取代的实验清单见正式结果文档 §9。长期记忆验证集
`exp-1fe49fedbd22` 只能引用工程验证，不能引用排序质量数字（§10）。

引用数字时**指明来源文件**（例如 `final_release/deterministic/exp-e748800507ef/metrics/variant_summary.csv`），
不要写"经计算"。完整 bundle 归档在 `dist/`，其 SHA-256 记录在 `final_release/provenance.json`。

---

## 1. 生存者偏差 —— 最容易写错的一条

排序指标（NDCG@5、P@5、MGR）**只在变体返回了可评分结果的场景上有值**，所以各变体的
分母 `n` 不同。一个放弃了困难场景的变体，其排序均值可能**高于** `full`。

**这不是"更好"，是分母更小。**

- 跨变体比较必须用 `task_success`：它在每个场景上都有定义，放弃或答错都记为失败。
- 呈现排序指标时**必须同时给出 `n`**。
- `grounding` 有同样的问题：只在真正产出事实性 claim 的运行上取均值，没有推荐就没有 claim。
- 禁止表述："去掉记忆后排序反而更好"。

---

## 2. `grounding = 1.000` 的正确解释

**这是构造使然，不是解释质量的证据，两个后端皆然。**

解释文本**从不由模型生成**：claim 由证据记录装配而成。因此 claim 验证器没有任何
未被证据支持的内容可以拒绝。写作时必须说明这一点，否则读者会把它读成"模型解释很可靠"。

正确表述：grounding 衡量的是**证据支持度**，不是被感知的解释质量或用户信任。

---

## 3. 相关性 oracle 的定位

canonical oracle **v3.0.0** 是**声明式**的：每个场景在场景文件里声明自己的权威参考答案
（字段值、hard/soft 强度、unknown 处理），oracle 只做机制性映射与算术比较，**不咨询系统的抽取器**。
因此它独立于变体、独立于随机重复，两个后端共用同一把尺子。

但**它仍然不是人工标注**。参考答案由作者从话语人工判定，比较机制与系统共享。写作时：

- ✅ "由声明式 canonical oracle 评分，参考答案是冻结输入的一部分"
- ✅ 作为构念效度威胁写入 §12
- ❌ "人工标注" / "独立 ground truth" / "客观相关性"

**可引用的收益**：SC-D-02 暴露了系统未强制执行话语中带 `only` 的工作模式约束 ——
这类缺陷自派生 oracle 结构上无法发现。

---

## 4. ActiveSearchState 的措辞

`ActiveSearchState` 是 CMJCC 合并候选人长期档案与当前对话证据后产出的**"本次检索"视图**，
并把字段分类为 hard / soft / unknown / 待澄清。不要把它写成"用户画像"或"查询"。
`hard_constraint_fields` 是**系统的判断**；参考答案里的 hard 是**声明的事实** —— 两者不可混称。

---

## 5. 延迟的 caveat

hybrid 的耗时**绝大部分是 API 等待**，不是框架计算（具体比例见正式结果文档 §1）。
所以：

- ❌ 不要用 hybrid 的延迟论证框架开销。
- ✅ 报告 deterministic 的延迟作为框架自身成本，hybrid 的延迟单独说明其构成。
- ✅ `run_totals.json` 提供整段对话的延迟汇总；`component_latency.json` 是最后一轮的分解。

---

## 6. no-match 的定义

SC-E-02 与 SC-E-04 的 no-match 是**多个硬约束联合不可满足**的结果，不是单一约束过严。
写作时必须说明是"联合不可行"，并可引用数据质量报告里
`no_match_scenario_constraint_satisfiable` 的 2 条警示 —— 那是**有意保留**的已知张力，
不是缺陷。

---

## 7. 统计口径

- **分析单元是 `scenario_id`。** `repeat_index` 只用于稳定性与方差分析，
  **永不**当作独立样本；重复在配对前先在场景内折叠。重复不能扩大样本或缩小 p 值。
- `task_success` 用**场景级二值**（多数票折叠，偶数重复平局取 0）+ McNemar；
  其余指标用重复平均后的场景均值 + 配对 bootstrap CI + Wilcoxon。
  **每一行只报告一个估计量**，不要把二值检验的 p 与小数均值的 Δ 并排。
- **Holm 在每个结局家族内独立校正**：预注册的 primary 家族与 secondary
  过程指标家族分开，所以加入过程指标不改变任何 primary p 值。
- 小样本下，包含 0 的 CI 写作"方向已观察、不确定"，**不写**"无效应"。
- §5.3 / §5.4 的计数单位是 **run**，不是场景。

---

## 8. 澄清效率分数的读法

该分数是**分层惩罚尺度**，不是概率也不是幅度。跳过必要澄清的惩罚为 1e6，
放弃已提问的对话为 1e3，所以**五位数的均值编码的是"最差层运行的占比"**。

- ✅ 主读法是**层级计数**（asked-and-resolved / asked-then-abandoned / skipped）与**中位数**。
- ✅ 均值只用于它保证的序关系：resolved > abandoned > skipped。
- ❌ 不要把均值当严重程度，不要跨变体比较均值的绝对值。
- 必须与 `Abandoned` 和 `AnsweredRate` 并读：`Abandoned` 非零的变体不得仅凭分数被称为高效。

---

## 9. 禁止表述清单

- ❌ 记忆效应"统计显著"而不指明实验（hybrid 上不显著）
- ❌ 任何旧数字或旧 experiment id
- ❌ 用排序指标跨变体论证优劣而不给 `n`
- ❌ 把 `grounding = 1.000` 当解释质量
- ❌ 称 oracle 为人工标注 / 独立 ground truth
- ❌ 把 hybrid 的 `git_dirty=true` 解释为代码被修改
- ❌ 引用 `exp-1fe49fedbd22` 的排序质量数字
- ❌ 主张对外部框架的普遍优越性 —— 一律限定为"受控原型实例化下的机制贡献"

---

## 10. 必须写入 §12 的构念与内部效度威胁

1. 相关性由声明式 oracle 评分，非人工判断；比较机制与系统共享。
2. `grounding = 1.000` 是构造使然。
3. deterministic 用 mock provider，不检验真实模型；hybrid 检验了真实模型，
   但响应随机，重复只衡量方差、不增加独立样本。
4. 目录与候选人为合成数据，场景数适中，结果不外推到真实招聘结果。
5. 小样本限制统计功效，重点在效应量、CI 与逐场景图，而非单个 p 值。
