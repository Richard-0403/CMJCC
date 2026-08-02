"""Write the BLANK per-rater annotation workload as spreadsheets, and read it back.

Why this exists
---------------
``evaluation/annotation_workspace/`` is a post-hoc AUDIT package: it materialises labels that
were already collected so a reader can check a verdict. It is not something a rater can work
from. The annotation store holds the workload, but it is a SQLite file, and the web UI needs a
server running. This script produces the third thing: flat files a rater opens, fills in and
sends back.

Blinding is not re-implemented here. Every row is rendered from
:meth:`jobrec_eval.annotation_ui.store.AnnotationStore.queue`, which is the same rater-facing
read the web UI uses -- it returns :class:`~jobrec_eval.annotation_ui.store.RaterItem`, whose
dataclass has no field for the analysis side at all. So the validator verdict and the oracle
grade cannot reach these files through this path, and the exporter re-checks anyway.

Rater isolation: one file per rater, containing only that rater's assigned queue, in that
rater's own seeded shuffle order (two raters get the same items in different orders, so fatigue
or drift cannot line up between them and inflate agreement). No file contains another rater's
label.

Two workloads, and they unlock different things:

``claim_<rater>.csv``      one row per annotation_signature. Needed for Cohen's kappa on claim
                          grounding. Large, because the old claim labels overlap the current
                          signature universe by ZERO and cannot be reused.
``relevance_<rater>.csv``  one row per returned (scenario, job) pair that has NO human label
                          yet -- the coverage DELTA. Small, because relevance is a judgement
                          about a scenario and a posting, not about the system version, so the
                          existing labels stay valid and only the gap needs filling.

Usage
-----
    python scripts/build_annotation_workload.py --annotation-dir artifacts/annotation_official/hybrid \
        --delta artifacts/main_hybrid/analysis/exp-2b33b808a0f8/annotation/relevance_delta_annotation.csv \
        --out-dir artifacts/annotation_workload/hybrid --write

    # after the raters fill the label column in
    python scripts/build_annotation_workload.py --annotation-dir ... --import artifacts/annotation_workload/hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jobrec_eval.annotation_ui.store import (  # noqa: E402
    BLINDED_FIELD_NAMES,
    KIND_CLAIM,
    KIND_RELEVANCE,
    LABEL_RANGES,
    open_store,
)

#: The column the rater fills. Left EMPTY on export; a row with it blank is simply not
#: imported, so a partly finished file is usable and no blank ever becomes a 0.
LABEL_COLUMN = "label_YOUR_ANSWER"

CLAIM_COLUMNS = [
    "item_key", "queue_position", LABEL_COLUMN, "notes", "flags",
    "claim_text", "claim_type", "predicate", "field", "job_id",
    "expected_value", "observed_value", "claim_args",
    "delivery_status", "cited_evidence_count", "occurrence_count",
    "has_unresolvable_evidence", "unresolvable_evidence_ids",
    "evidence", "referenced_jobs", "scenario_ids",
]

RELEVANCE_COLUMNS = [
    "item_key", "queue_position", LABEL_COLUMN, "notes", "flags",
    "scenario_id", "job_id", "scenario_type",
    "candidate_profile", "conversation",
    "job_title", "job_company", "job_location", "job_work_mode",
    "job_employment_type", "job_salary", "job_required_skills",
    "job_preferred_skills", "job_min_years_experience", "job_experience_level",
    "job_description",
]


def _render(value: Any) -> str:
    """A cell a human can read in a spreadsheet, and that survives a round trip."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _evidence_text(evidence: list[dict]) -> str:
    """The cited evidence as numbered lines: what field of what object said what."""
    lines = []
    for i, item in enumerate(evidence, 1):
        lines.append(
            f"[{i}] source={item.get('source') or '?'}"
            f" object={item.get('source_object_id') or '?'}"
            f" field={item.get('field_name') or '?'}"
            f" value={_render(item.get('normalized_value'))}"
            f" | raw: {item.get('raw_text') or ''}")
    return "\n".join(lines)


def _jobs_text(jobs: list[dict]) -> str:
    lines = []
    for job in jobs:
        if job.get("missing_from_catalog"):
            lines.append(f"{job.get('job_id')}: NOT IN CATALOG SNAPSHOT")
            continue
        lines.append(
            f"{job.get('job_id')}: {job.get('title')} @ {job.get('company')}"
            f" | {_render(job.get('location'))} | work_mode={job.get('work_mode')}"
            f" | salary={_render(job.get('salary'))}"
            f" | required={_render(job.get('required_skills'))}"
            f" | preferred={_render(job.get('preferred_skills'))}"
            f" | min_years={job.get('min_years_experience')}"
            f" | level={job.get('experience_level')}"
            f" | active={job.get('is_active')}")
    return "\n".join(lines)


def _conversation_text(scenario: dict) -> str:
    turns = scenario.get("conversation") or []
    return "\n".join(f"turn {t.get('turn_index')}: {t.get('candidate_utterance')}"
                     for t in turns)


def _claim_row(item) -> dict[str, str]:
    p = item.payload
    return {
        "item_key": item.item_key,
        "queue_position": str(item.position),
        LABEL_COLUMN: "",
        "notes": "",
        "flags": "",
        "claim_text": _render(p.get("claim_text")),
        "claim_type": _render(p.get("claim_type")),
        "predicate": _render(p.get("predicate")),
        "field": _render(p.get("claim_field")),
        "job_id": _render(p.get("claim_job_id")),
        "expected_value": _render(p.get("expected_value")),
        "observed_value": _render(p.get("observed_value")),
        "claim_args": _render(p.get("claim_args")),
        "delivery_status": _render(p.get("delivery_status")),
        "cited_evidence_count": _render(p.get("cited_evidence_count")),
        "occurrence_count": _render(p.get("occurrence_count")),
        "has_unresolvable_evidence": _render(p.get("has_unresolvable_evidence")),
        "unresolvable_evidence_ids": _render(p.get("unresolvable_evidence_ids")),
        "evidence": _evidence_text(p.get("evidence") or []),
        "referenced_jobs": _jobs_text(p.get("referenced_jobs") or []),
        "scenario_ids": _render(p.get("scenario_ids")),
    }


def _relevance_row(item) -> dict[str, str]:
    p = item.payload
    scenario = p.get("scenario") or {}
    job = p.get("job") or {}
    return {
        "item_key": item.item_key,
        "queue_position": str(item.position),
        LABEL_COLUMN: "",
        "notes": "",
        "flags": "",
        "scenario_id": _render(scenario.get("scenario_id")),
        "job_id": _render(job.get("job_id")),
        "scenario_type": _render(scenario.get("scenario_type")),
        "candidate_profile": _render(scenario.get("candidate_profile")),
        "conversation": _conversation_text(scenario),
        "job_title": _render(job.get("title")),
        "job_company": _render(job.get("company")),
        "job_location": _render(job.get("location")),
        "job_work_mode": _render(job.get("work_mode")),
        "job_employment_type": _render(job.get("employment_type")),
        "job_salary": _render(job.get("salary")),
        "job_required_skills": _render(job.get("required_skills")),
        "job_preferred_skills": _render(job.get("preferred_skills")),
        "job_min_years_experience": _render(job.get("min_years_experience")),
        "job_experience_level": _render(job.get("experience_level")),
        "job_description": _render(job.get("description")),
    }


def _assert_blind(rows: list[dict[str, str]], what: str) -> None:
    """Refuse to write a file carrying the machine's own answer.

    The rater-facing read cannot supply those fields, so this is a second lock rather than the
    only one -- but a workload file is the one artifact that leaves the repo and goes to a
    person, and an unblinded one silently turns an independent judgement into agreement with
    the system under test.
    """
    blob = json.dumps(rows, ensure_ascii=False).lower()
    leaked = sorted(n for n in BLINDED_FIELD_NAMES if f'"{n}"' in blob)
    if leaked:
        raise SystemExit(f"refusing to write {what}: it carries {leaked}")


def _write_csv(rows: list[dict[str, str]], columns: list[str], path: Path,
               write: bool) -> int:
    if not write:
        return len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows opens the file with the right encoding.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


_RUBRIC = """# 标注说明 / Annotation guide

请只填写 `label_YOUR_ANSWER` 一列（以及可选的 `notes` / `flags`）。**不要改动 `item_key`**，
回填时要靠它定位。留空的行不会被导入，所以可以分批做完再交。

## 本目录的文件

| 文件 | 用途 |
|---|---|
| `claim_RATER-*.csv` | **要填**：判断证据是否支撑该命题（0/1） |
| `relevance_RATER-*.csv` | **要填**：判断岗位与需求的匹配度（0–3） |
| `GLOSSARY.md` | 所有 id 和枚举取值的中文解释 —— **先看这个** |
| `scenarios_reference.csv` | `scenario_id` 查表：候选人档案、逐轮对话 |
| `jobs_reference.csv` | `job_id` 查表：岗位全部字段 |
| `workload.json` | 溯源记录，不用管 |

`claim_*.csv` 里没有内嵌场景对话，因为同一个命题可能出现在多个场景中（`scenario_ids` 列会列出
全部）。需要了解候选人说了什么时，用该列的 id 去 `scenarios_reference.csv` 查。

## claim_*.csv —— 判断"引用的证据是否支撑这句话"（0 / 1）

每一行是一个**命题**，不是一句话。同一句话在不同数值下是不同的行，请分别判断。

- `1` = 引用的证据支撑这句话
- `0` = 不支撑

判断要点：

1. 只看 `evidence` 列里列出的证据。系统有没有别的理由不算。
2. `expected_value` 与 `observed_value` 是这条命题断言的值，请核对它们与证据、与
   `referenced_jobs` 里的岗位字段是否一致。
3. `has_unresolvable_evidence = yes` 表示有引用指向不存在的证据。**指向空处的引用不能支撑
   任何结论**——这类情况通常应判 `0`。
4. `delivery_status` 会改变问题：
   - `delivered` = 用户真的看到了这句话，判断它是否被支撑；
   - `dropped` = 系统内部生成后撤回了，判断它**本来是否应该**被支撑（用于估计校验器的漏判）。

## relevance_*.csv —— 判断岗位与候选人需求的匹配度（0–3）

- `3` = 强匹配
- `2` = 部分匹配
- `1` = 弱匹配
- `0` = 不相关

请按 `conversation` 里的**轮次顺序**阅读：后面的轮次可能修正前面说过的偏好。

## 两位标注者请独立完成

同一批条目在两个人的文件里顺序不同，这是刻意的。请不要交换文件或讨论具体条目——
两人独立是 Cohen's kappa 成立的前提。分歧会在之后的裁定环节单独处理。
"""


SCENARIO_COLUMNS = ["scenario_id", "scenario_type", "difficulty", "memory_dependency",
                    "context_dependency", "candidate_profile", "conversation",
                    "acceptable_clarification_slots"]

JOB_COLUMNS = ["job_id", "title", "company", "city", "region", "country", "work_mode",
               "employment_type", "salary_min", "salary_max", "salary_currency",
               "salary_period", "required_skills", "preferred_skills",
               "min_years_experience", "experience_level", "application_deadline",
               "is_active", "responsibilities", "description"]

_GLOSSARY = """# 字段与取值对照表 / Glossary

标注文件里出现的每一种 id 和枚举值的含义。**判断时只依据 `evidence` 列列出的证据**，
本表用于查对照，不是新增的判断依据。

## id 怎么查

| id | 在哪查 | 说明 |
|---|---|---|
| `scenario_id`（如 `SC-D-10`） | `scenarios_reference.csv` | 候选人档案、逐轮对话、场景类型 |
| `job_id`（如 `job-0110`） | `jobs_reference.csv` | 岗位的全部字段 |
| `item_key`（如 `clm::sig-d9b1...`） | 不用查 | 回填时的定位键，**请勿修改** |
| `queue_position` | 不用查 | 你的队列顺序，两位标注者刻意不同 |

`claim_*.csv` 的 `referenced_jobs` 列已经内嵌了该命题涉及岗位的主要字段，通常不必再查
`jobs_reference.csv`；需要看岗位描述全文时才查。

`scenario_ids` 列可能有多个值，表示同一个命题在多个场景中出现过（588 行里有 263 行如此）。
这是正常的：命题是按内容去重的，不是按场景。

## claim_type —— 这句话属于哪一类

| 取值 | 含义 |
|---|---|
| `ranking_reason` | 推荐理由：解释为什么这个岗位排进了结果 |
| `skill_gap` | 技能差距：指出候选人缺少或未记录某项技能 |
| `candidate_preference` | 复述候选人自己表达过的偏好 |
| `no_match_reason` | 无匹配结果时给出的原因 |
| `no_match_cause` | 无匹配的具体成因（哪条约束导致） |

## predicate —— 这句话断言的关系

| 取值 | 含义 | 判断时核对什么 |
|---|---|---|
| `ranking_match` | 某字段与候选人偏好一致 | `expected_value`（偏好）与 `observed_value`（岗位实际值）是否真的一致 |
| `salary_meets_min` | 薪资达到候选人下限 | 岗位薪资是否 ≥ `expected_value` |
| `experience_in_range` | 经验年限落在岗位要求内 | 候选人年限与岗位 `min_years_experience` |
| `skill_covered` | 候选人具备该技能 | 技能是否真在候选人档案/对话中 |
| `skill_not_recorded` | 候选人未记录该技能 | 该技能是否**确实**没有出现在证据里 |
| `candidate_preference` | 候选人确实表达过该偏好 | 对话或档案里是否真有 |
| `constraint_applied` | 某条硬约束被应用了 | 证据是否显示该约束生效 |
| `no_match_cause` | 该约束是无匹配的成因 | 证据是否支持这个归因 |

## field —— 命题针对的字段

| 取值 | 含义 |
|---|---|
| `target_roles` | 目标岗位/职位方向 |
| `preferred_locations` | 期望工作地点 |
| `work_modes` | 工作模式（remote / hybrid / onsite） |
| `salary_min` | 薪资下限 |
| `skills_have` | 候选人已具备的技能 |
| `years_experience` | 工作年限 |

## evidence 列的 source —— 这条证据来自哪里

| 取值 | 含义 |
|---|---|
| `profile` | 候选人档案（对话之前就已知的信息） |
| `dialogue` | 候选人在对话中说的话 |
| `job_posting` | 岗位公告的字段 |
| `system_rule` | 系统内部规则（不是候选人或岗位提供的事实） |

`source=system_rule` 值得留意：规则本身不能证明关于候选人或岗位的事实陈述。

## 特殊情况

- **`job_id` 与 `referenced_jobs` 都为空**（deterministic / hybrid 各约 60 行）：
  这类命题不针对具体岗位，通常是 `no_match_reason` / `no_match_cause`，即"为什么没有结果"。
  按证据判断该归因是否成立即可。
- **`has_unresolvable_evidence = yes`**：有引用指向不存在的证据。
  指向空处的引用不能支撑任何结论，这类通常判 `0`。
- **`delivery_status = dropped`**（仅 hybrid，14 行）：系统生成后被自动校验器撤回，用户没有看到。
  请判断它**本来是否应该**被支撑（用于估计校验器的漏判率），而不是判断用户看到了什么。
"""


def _write_reference(annotation_dir: Path, out_dir: Path, meta: dict,
                     write: bool) -> list[str]:
    """The lookup tables the workload's ids point at, plus the glossary.

    The claim rows name a ``scenario_id`` and a ``job_id`` but cannot carry the scenario inline:
    263 of the 588 hybrid signatures occur in MORE than one scenario, because a proposition is
    deduplicated by content rather than by where it appeared. A row-level scenario column would
    therefore be empty or ambiguous for nearly half the file, so the scenarios are shipped as a
    table the rater looks up instead.
    """
    from jobrec.catalog import load_catalog
    from jobrec_eval.scenarios import load_scenarios

    scenarios_path = meta.get("scenarios_path") or "evaluation/data/scenarios.jsonl"
    catalog_path = meta.get("catalog_path") or "data/processed/jobs.jsonl"

    scen_rows = []
    for sid, s in sorted(load_scenarios(scenarios_path).items()):
        scen_rows.append({
            "scenario_id": sid,
            "scenario_type": _render(s.scenario_type),
            "difficulty": _render(getattr(s, "difficulty", None)),
            "memory_dependency": _render(getattr(s, "memory_dependency", None)),
            "context_dependency": _render(getattr(s, "context_dependency", None)),
            "candidate_profile": _render(dict(s.profile)),
            "conversation": "\n".join(
                f"turn {i}: {t}" for i, t in enumerate(s.turns)),
            "acceptable_clarification_slots": _render(list(s.acceptable_slots)),
        })

    job_rows = []
    for job in load_catalog(catalog_path):
        job_rows.append({
            "job_id": job.job_id, "title": _render(job.title),
            "company": _render(job.company), "city": _render(job.city),
            "region": _render(job.region), "country": _render(job.country),
            "work_mode": _render(job.work_mode),
            "employment_type": _render(job.employment_type),
            "salary_min": _render(job.salary_min), "salary_max": _render(job.salary_max),
            "salary_currency": _render(job.salary_currency),
            "salary_period": _render(job.salary_period),
            "required_skills": _render(list(job.required_skills)),
            "preferred_skills": _render(list(job.preferred_skills)),
            "min_years_experience": _render(job.min_years_experience),
            "experience_level": _render(job.experience_level),
            "application_deadline": _render(
                job.application_deadline.isoformat() if job.application_deadline else None),
            "is_active": _render(bool(job.is_active)),
            "responsibilities": _render(list(job.responsibilities)),
            "description": _render(job.description),
        })

    _write_csv(scen_rows, SCENARIO_COLUMNS, out_dir / "scenarios_reference.csv", write)
    _write_csv(job_rows, JOB_COLUMNS, out_dir / "jobs_reference.csv", write)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "GLOSSARY.md").write_text(_GLOSSARY, encoding="utf-8", newline="\n")
    return [f"  scenarios_reference.csv: {len(scen_rows)} 个场景",
            f"  jobs_reference.csv: {len(job_rows)} 个岗位",
            "  GLOSSARY.md: claim_type / predicate / field / evidence source 对照表"]


def export(annotation_dir: Path, out_dir: Path, delta_path: Path | None,
           write: bool) -> int:
    delta: set[tuple[str, str]] | None = None
    if delta_path is not None and delta_path.exists():
        with delta_path.open(encoding="utf-8-sig", newline="") as handle:
            delta = {(r["scenario_id"], r["job_id"]) for r in csv.DictReader(handle)}

    with open_store(annotation_dir, create=False) as store:
        meta = store.meta()
        raters = store.raters()
        if not raters:
            raise SystemExit(f"no raters registered in {annotation_dir}")
        total = 0
        summary: list[str] = []
        for rater in raters:
            claims = [_claim_row(i) for i in store.queue(rater, kind=KIND_CLAIM)]
            rel_items = store.queue(rater, kind=KIND_RELEVANCE)
            rel_rows = []
            for item in rel_items:
                row = _relevance_row(item)
                if delta is not None and (row["scenario_id"], row["job_id"]) not in delta:
                    continue
                rel_rows.append(row)
            _assert_blind(claims, f"claim workload for {rater}")
            _assert_blind(rel_rows, f"relevance workload for {rater}")
            n_c = _write_csv(claims, CLAIM_COLUMNS, out_dir / f"claim_{rater}.csv", write)
            n_r = _write_csv(rel_rows, RELEVANCE_COLUMNS,
                             out_dir / f"relevance_{rater}.csv", write)
            total += n_c + n_r
            summary.append(f"  {rater}: claim {n_c} 行, relevance {n_r} 行"
                           + ("" if delta is None else " (仅覆盖缺口)"))

    summary += _write_reference(annotation_dir, out_dir, meta, write)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.md").write_text(_RUBRIC, encoding="utf-8", newline="\n")
        (out_dir / "workload.json").write_text(json.dumps({
            "annotation_dir": str(annotation_dir),
            "experiment_id": meta.get("experiment_id"),
            "schema_version": meta.get("schema_version"),
            "annotation_universe": meta.get("annotation_universe"),
            "sampling_seed": meta.get("sampling_seed"),
            "raters": list(raters),
            "label_column": LABEL_COLUMN,
            "label_ranges": {k: list(v) for k, v in LABEL_RANGES.items()},
            "relevance_scope": ("coverage delta only" if delta is not None
                                else "every returned pair"),
        }, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    print(f"experiment {meta.get('experiment_id')}  ->  {out_dir}")
    print("\n".join(summary))
    if not write:
        print("\n(dry run; pass --write to create the files)")
    return 0


def import_labels(annotation_dir: Path, workload_dir: Path, write: bool) -> int:
    """Read filled workload files back through the store's own write path.

    Every row goes through :meth:`AnnotationStore.upsert_annotation`, which refuses an item the
    rater was not assigned and refuses a label outside the kind's range. A blank label column is
    SKIPPED, never written as 0: an unanswered item is unanswered.
    """
    with open_store(annotation_dir, create=False) as store:
        raters = store.raters()
        applied = skipped = 0
        problems: list[str] = []
        for rater in raters:
            for kind, prefix in ((KIND_CLAIM, "claim"), (KIND_RELEVANCE, "relevance")):
                path = workload_dir / f"{prefix}_{rater}.csv"
                if not path.exists():
                    continue
                valid = LABEL_RANGES[kind]
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for line_no, row in enumerate(csv.DictReader(handle), start=2):
                        raw = (row.get(LABEL_COLUMN) or "").strip()
                        if not raw:
                            skipped += 1
                            continue
                        try:
                            label = int(float(raw))
                        except ValueError:
                            problems.append(
                                f"{path.name}:{line_no} label={raw!r} 不是数字")
                            continue
                        if label not in valid:
                            problems.append(
                                f"{path.name}:{line_no} label={label} 不在 {valid} 内")
                            continue
                        if not write:
                            applied += 1
                            continue
                        try:
                            store.upsert_annotation(
                                row["item_key"], rater, label,
                                notes=(row.get("notes") or ""),
                                flags=(row.get("flags") or ""))
                            applied += 1
                        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                            problems.append(
                                f"{path.name}:{line_no} {type(exc).__name__}: {exc}")
        print(f"标签{'已写入' if write else '可写入'}: {applied} | 空白跳过: {skipped}")
        if problems:
            print(f"\n{len(problems)} 个问题（这些行未写入）：")
            for p in problems[:20]:
                print(f"  - {p}")
            if len(problems) > 20:
                print(f"  ... 另有 {len(problems) - 20} 个")
        if not write:
            print("\n(dry run; pass --write to apply)")
        return 1 if problems else 0


#: Files every rater needs a copy of: the guide, the glossary and the two lookup tables.
_SHARED = ("README.md", "GLOSSARY.md", "scenarios_reference.csv", "jobs_reference.csv")


def package(workload_dirs: list[Path], out_dir: Path, write: bool) -> int:
    """One self-contained folder and zip PER RATER, across every arm.

    Structural rater isolation. The workload directory holds both raters' files, so sending it
    whole would let each rater read the other's queue -- and two label sets that saw each other
    are not independent, which is a precondition of Cohen's kappa, not a nicety. Each package
    therefore carries that rater's files only.
    """
    import shutil

    raters: dict[str, list[tuple[Path, str]]] = {}
    for wd in workload_dirs:
        arm = wd.name
        for path in sorted(wd.glob("*_RATER-*.csv")):
            rater = path.stem.split("_", 1)[1]
            raters.setdefault(rater, []).append((path, f"{arm}/{path.name}"))
        for shared in _SHARED:
            src = wd / shared
            if src.exists():
                for rater in raters:
                    raters[rater].append((src, f"{arm}/{shared}"))

    lines = []
    for rater, files in sorted(raters.items()):
        target = out_dir / rater
        if write:
            if target.exists():
                shutil.rmtree(target)
            for src, rel in files:
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            archive = shutil.make_archive(str(out_dir / rater), "zip", root_dir=target)
            size = Path(archive).stat().st_size / 1024
            lines.append(f"  {rater}: {len(files)} 个文件 -> {rater}.zip ({size:,.0f} KB)")
        else:
            lines.append(f"  {rater}: {len(files)} 个文件")
        # No package may contain another rater's file.
        foreign = [rel for _s, rel in files if "RATER-" in rel and rater not in rel]
        if foreign:
            raise SystemExit(f"{rater} 的包里混入了别人的文件: {foreign}")

    print(f"每位标注者一个包 -> {out_dir}")
    print("\n".join(lines))
    if not write:
        print("\n(dry run; pass --write to create them)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default=None,
                        help="directory holding the annotation SQLite store")
    parser.add_argument("--package", nargs="+", default=None,
                        help="workload directories to assemble into per-rater packages")
    parser.add_argument("--out-dir", default=None, help="where the workload files go")
    parser.add_argument("--delta", default=None,
                        help="relevance_delta_annotation.csv; restricts the relevance "
                             "workload to the pairs that have no human label yet")
    parser.add_argument("--import-dir", default=None,
                        help="read filled workload files back into the store")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.package:
        if not args.out_dir:
            parser.error("--out-dir is required with --package")
        return package([Path(p) for p in args.package], Path(args.out_dir), args.write)
    if not args.annotation_dir:
        parser.error("--annotation-dir is required")
    annotation_dir = Path(args.annotation_dir)
    if args.import_dir:
        return import_labels(annotation_dir, Path(args.import_dir), args.write)
    if not args.out_dir:
        parser.error("--out-dir is required unless --import-dir is given")
    return export(annotation_dir, Path(args.out_dir),
                  Path(args.delta) if args.delta else None, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
