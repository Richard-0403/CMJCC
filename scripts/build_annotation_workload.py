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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True,
                        help="directory holding the annotation SQLite store")
    parser.add_argument("--out-dir", default=None, help="where the workload files go")
    parser.add_argument("--delta", default=None,
                        help="relevance_delta_annotation.csv; restricts the relevance "
                             "workload to the pairs that have no human label yet")
    parser.add_argument("--import-dir", default=None,
                        help="read filled workload files back into the store")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    if args.import_dir:
        return import_labels(annotation_dir, Path(args.import_dir), args.write)
    if not args.out_dir:
        parser.error("--out-dir is required unless --import-dir is given")
    return export(annotation_dir, Path(args.out_dir),
                  Path(args.delta) if args.delta else None, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
