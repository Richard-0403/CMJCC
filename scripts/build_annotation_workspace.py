"""Assemble the persisted human-annotation evidence package.

Everything a reader needs to audit the human relevance and claim labels: the claim text
with its evidence lifted out of the official run bundles, the rater-level labels, the
adjudicated gold, the rubric that was applied, and a checksum manifest over the lot.

Why the claim text has to be materialised here: the annotation templates carry only
``claim_id``, while the text lives in each bundle's ``response_claims.json`` and the
evidence it cites in ``evidence_items.jsonl``. Auditing a verdict from the template alone
would mean opening two files per row across 588 bundles, so the join is done once and
written down.

Nothing is re-run. Every input is read from the sealed official pair.

    python scripts/build_annotation_workspace.py            # report what would be built
    python scripts/build_annotation_workspace.py --write
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Reuse the release builder's newline normalisation rather than keeping a second copy that
# can drift. Same reason it exists there: this directory carries its own checksum manifest
# over the bytes on disk, so text copied out of a Windows-produced tree must be pinned to
# LF before hashing or the manifest only holds on the machine that built it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_final_release import _normalize_newlines  # noqa: E402

WS = Path("evaluation/annotation_workspace")

#: The official pair. Both are read-only here.
SOURCES = {
    "deterministic": {
        "experiment_id": "exp-e748800507ef",
        "runs": Path("evaluation/outputs/_runs/exp-e748800507ef"),
        "analysis": Path("evaluation/outputs/exp-e748800507ef"),
    },
    "hybrid": {
        "experiment_id": "exp-6db1e87daed5",
        "runs": Path("evaluation/outputs_hybrid/_runs/exp-6db1e87daed5"),
        "analysis": Path("evaluation/outputs_hybrid/exp-6db1e87daed5"),
    },
}

HUMAN_RELEVANCE = Path("evaluation/data/relevance_labels_human.csv")
HUMAN_CLAIMS = Path("evaluation/data/claim_annotations_human.csv")

RELEVANCE_VALUES = {"0", "1", "2", "3"}
CLAIM_VALUES = {"0", "1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cell(row: dict[str, str], name: str) -> str:
    return (row.get(name) or "").strip()


def gold_of(row: dict[str, str]) -> tuple[str, str]:
    """The human gold for one row, and where it came from.

    This is the project's documented rule, restated: an ``adjudicated`` value IS the
    gold; otherwise two raters that agree are their own gold; otherwise the row is
    unadjudicated and contributes nothing. It is never the mean of a disagreement.
    """
    a, b, adj = cell(row, "rater_1"), cell(row, "rater_2"), cell(row, "adjudicated")
    if adj:
        return adj, "adjudicated"
    if a and b and a == b:
        return a, "rater_concordant"
    return "", "unadjudicated"


# --------------------------------------------------------------- claim text + evidence
def collect_claims() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Join every claim in the official bundles with the evidence it cites."""
    claims: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    bundles = 0
    for label, spec in SOURCES.items():
        _, index = read_csv(spec["runs"] / "runs_index.csv")
        for entry in index:
            bundle = Path(entry["run_dir"])
            claims_path = bundle / "response_claims.json"
            if not claims_path.exists():
                continue
            bundles += 1
            evidence: dict[str, dict] = {}
            ev_path = bundle / "evidence_items.jsonl"
            if ev_path.exists():
                for line in ev_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        evidence[item["evidence_id"]] = item
            for claim in json.loads(claims_path.read_text(encoding="utf-8")):
                eids = claim.get("evidence_ids") or []
                summary = []
                for eid in eids:
                    item = evidence.get(eid)
                    if item is None:
                        summary.append(f"{eid}: MISSING")
                        evidence_rows.append({
                            "run_id": entry["run_id"], "claim_id": claim["claim_id"],
                            "evidence_id": eid, "resolved": False, "source": "",
                            "source_object_id": "", "field_name": "",
                            "normalized_value": "", "raw_text": "", "confidence": "",
                            "confirmation_status": "", "persistence_scope": "",
                            "turn_id": "",
                        })
                        continue
                    summary.append(
                        f"{item.get('source')}:{item.get('field_name')}="
                        f"{item.get('normalized_value')} "
                        f"({item.get('confirmation_status')}, conf {item.get('confidence')})")
                    evidence_rows.append({
                        "run_id": entry["run_id"], "claim_id": claim["claim_id"],
                        "evidence_id": eid, "resolved": True,
                        "source": item.get("source"),
                        "source_object_id": item.get("source_object_id"),
                        "field_name": item.get("field_name"),
                        "normalized_value": item.get("normalized_value"),
                        "raw_text": item.get("raw_text"),
                        "confidence": item.get("confidence"),
                        "confirmation_status": item.get("confirmation_status"),
                        "persistence_scope": item.get("persistence_scope"),
                        "turn_id": item.get("turn_id"),
                    })
                claims.append({
                    "experiment": label,
                    "experiment_id": spec["experiment_id"],
                    "run_id": entry["run_id"],
                    "scenario_id": entry["scenario_id"],
                    "variant": entry["experiment_variant"],
                    "run_index": entry["run_index"],
                    "claim_id": claim["claim_id"],
                    "claim_type": claim.get("claim_type"),
                    "claim_text": claim.get("text"),
                    "auto_support_status": claim.get("support_status"),
                    "evidence_count": len(eids),
                    "evidence": " | ".join(summary),
                })
    return claims, evidence_rows, bundles


CLAIM_TEXT_COLUMNS = ["experiment", "experiment_id", "run_id", "scenario_id", "variant",
                      "run_index", "claim_id", "claim_type", "claim_text",
                      "auto_support_status", "evidence_count", "evidence"]
EVIDENCE_COLUMNS = ["run_id", "claim_id", "evidence_id", "resolved", "source",
                    "source_object_id", "field_name", "normalized_value", "raw_text",
                    "confidence", "confirmation_status", "persistence_scope", "turn_id"]

RELEVANCE_LABEL_COLUMNS = ["scenario_id", "job_id", "oracle_grade", "rater_1", "rater_2",
                           "adjudicated", "gold", "gold_source", "delta_human_minus_oracle",
                           "notes"]
CLAIM_LABEL_COLUMNS = ["run_id", "claim_id", "claim_type", "validator", "rater_1",
                       "rater_2", "adjudicated", "gold", "gold_source",
                       "agrees_with_validator", "notes"]


def audit(rows: list[dict[str, Any]], valid: set[str]) -> dict[str, Any]:
    counts = {"rows": len(rows), "both_raters": 0, "one_rater": 0, "no_rater": 0,
              "agree": 0, "disagree": 0, "disagree_adjudicated": 0,
              "unadjudicated": 0, "out_of_range": 0, "gold_rows": 0}
    for row in rows:
        a, b, adj = cell(row, "rater_1"), cell(row, "rater_2"), cell(row, "adjudicated")
        for value in (a, b, adj):
            if value and value not in valid:
                counts["out_of_range"] += 1
        if a and b:
            counts["both_raters"] += 1
            if a == b:
                counts["agree"] += 1
            else:
                counts["disagree"] += 1
                if adj:
                    counts["disagree_adjudicated"] += 1
        elif a or b:
            counts["one_rater"] += 1
        else:
            counts["no_rater"] += 1
        gold, source = gold_of(row)
        if source == "unadjudicated":
            counts["unadjudicated"] += 1
        if gold:
            counts["gold_rows"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rel_cols, rel_rows = read_csv(HUMAN_RELEVANCE)
    clm_cols, clm_rows = read_csv(HUMAN_CLAIMS)
    claims, evidence_rows, bundles = collect_claims()

    rel_audit = audit(rel_rows, RELEVANCE_VALUES)
    clm_audit = audit(clm_rows, CLAIM_VALUES)
    claim_index = {(c["run_id"], c["claim_id"]): c for c in claims}
    missing_text = [k for k in ((r["run_id"], r["claim_id"]) for r in clm_rows)
                    if k not in claim_index]

    print(f"bundles scanned            : {bundles}")
    print(f"claims joined with evidence: {len(claims)} "
          f"({len(evidence_rows)} claim-evidence pairs)")
    print(f"relevance rows             : {rel_audit['rows']} "
          f"(gold {rel_audit['gold_rows']}, unadjudicated {rel_audit['unadjudicated']})")
    print(f"claim rows                 : {clm_audit['rows']} "
          f"(gold {clm_audit['gold_rows']}, unadjudicated {clm_audit['unadjudicated']})")
    print(f"claim rows without text    : {len(missing_text)}")

    if not args.write:
        print("\npass --write to build")
        return 0

    if WS.exists():
        shutil.rmtree(WS)
    (WS / "labels").mkdir(parents=True)
    (WS / "reference").mkdir(parents=True)

    # ---- reference: claim text + evidence ----------------------------------
    write_csv(WS / "reference" / "claim_texts.csv", CLAIM_TEXT_COLUMNS,
              sorted(claims, key=lambda r: (r["run_id"], r["claim_id"])))
    write_csv(WS / "reference" / "claim_evidence.csv", EVIDENCE_COLUMNS,
              sorted(evidence_rows,
                     key=lambda r: (r["run_id"], r["claim_id"], r["evidence_id"])))

    # ---- reference: the inputs a rater judges against ----------------------
    for src, dst in (
        (Path("evaluation/data/scenarios.jsonl"), "scenarios.jsonl"),
        (Path("evaluation/data/canonical_oracle_scenarios.json"),
         "canonical_oracle_scenarios.json"),
        (Path("data/processed/jobs.jsonl"), "jobs.jsonl"),
        (Path("data/processed/jobs.csv"), "jobs.csv"),
    ):
        shutil.copy2(src, WS / "reference" / dst)
    # One copy: both experiments' oracle label tables are byte-identical, because the
    # same declared canonical oracle v3.0.0 graded both.
    shutil.copy2(SOURCES["deterministic"]["analysis"] / "normalized" / "relevance_labels.csv",
                 WS / "reference" / "relevance_labels_oracle.csv")
    for label, spec in SOURCES.items():
        shutil.copy2(spec["analysis"] / "normalized" / "claims.csv",
                     WS / "reference" / f"claims_{label}.csv")
        shutil.copy2(spec["runs"] / "runs_index.csv",
                     WS / "reference" / f"runs_index_{label}.csv")

    # ---- labels: rater level, with the gold made explicit ------------------
    rel_out = []
    for row in rel_rows:
        gold, source = gold_of(row)
        delta = ""
        if gold and cell(row, "oracle_grade"):
            delta = str(int(gold) - int(cell(row, "oracle_grade")))
        rel_out.append({**{c: cell(row, c) for c in RELEVANCE_LABEL_COLUMNS
                           if c in row},
                        "gold": gold, "gold_source": source,
                        "delta_human_minus_oracle": delta})
    write_csv(WS / "labels" / "relevance_raters.csv", RELEVANCE_LABEL_COLUMNS,
              sorted(rel_out, key=lambda r: (r["scenario_id"], r["job_id"])))
    write_csv(WS / "labels" / "relevance_adjudicated.csv",
              ["scenario_id", "job_id", "rater_id", "relevance_grade"],
              [{"scenario_id": r["scenario_id"], "job_id": r["job_id"],
                "rater_id": "human_adjudicated", "relevance_grade": r["gold"]}
               for r in sorted(rel_out, key=lambda r: (r["scenario_id"], r["job_id"]))
               if r["gold"]])

    clm_out = []
    for row in clm_rows:
        gold, source = gold_of(row)
        validator = cell(row, "validator")
        clm_out.append({**{c: cell(row, c) for c in CLAIM_LABEL_COLUMNS if c in row},
                        "gold": gold, "gold_source": source,
                        "agrees_with_validator":
                            "" if not gold else str(gold == validator)})
    write_csv(WS / "labels" / "claim_raters.csv", CLAIM_LABEL_COLUMNS,
              sorted(clm_out, key=lambda r: (r["run_id"], r["claim_id"])))
    write_csv(WS / "labels" / "claim_adjudicated.csv",
              ["run_id", "claim_id", "rater_id", "supported"],
              [{"run_id": r["run_id"], "claim_id": r["claim_id"],
                "rater_id": "human_adjudicated", "supported": r["gold"]}
               for r in sorted(clm_out, key=lambda r: (r["run_id"], r["claim_id"]))
               if r["gold"]])

    (WS / "RUBRIC.md").write_text(_RUBRIC, encoding="utf-8", newline="\n")
    (WS / "README.md").write_text(_README, encoding="utf-8", newline="\n")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    provenance = {
        "generated_from_commit": commit or "unknown",
        "experiments_read": {label: spec["experiment_id"] for label, spec in SOURCES.items()},
        "experiment_re_run": False,
        "note": ("Every input was read from the sealed official pair. No experiment was "
                 "executed and no sealed artifact was written to."),
        "bundles_scanned": bundles,
        "claim_evidence_pairs": len(evidence_rows),
        "source_label_files": {
            str(HUMAN_RELEVANCE).replace("\\", "/"): {
                "sha256": sha256(HUMAN_RELEVANCE), "bytes": HUMAN_RELEVANCE.stat().st_size},
            str(HUMAN_CLAIMS).replace("\\", "/"): {
                "sha256": sha256(HUMAN_CLAIMS), "bytes": HUMAN_CLAIMS.stat().st_size},
        },
        "relevance": rel_audit,
        "claims": clm_audit,
        "claim_rows_without_text": len(missing_text),
        "verification": {
            "relevance_gold_complete": rel_audit["gold_rows"] == rel_audit["rows"],
            "claims_gold_complete": clm_audit["gold_rows"] == clm_audit["rows"],
            "relevance_unadjudicated_zero": rel_audit["unadjudicated"] == 0,
            "claims_unadjudicated_zero": clm_audit["unadjudicated"] == 0,
            "no_out_of_range_values":
                rel_audit["out_of_range"] == 0 and clm_audit["out_of_range"] == 0,
            "every_claim_row_has_text": not missing_text,
        },
    }
    (WS / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    # Pin the copied text to LF before hashing, so the manifest below reproduces on an LF
    # checkout too. The files written above are already LF; the copies are not.
    print(f"normalised {_normalize_newlines(WS)} copied text file(s) to LF")

    manifest_path = WS / "checksums.json"
    files = sorted(p for p in WS.rglob("*") if p.is_file() and p != manifest_path)
    manifest_path.write_text(json.dumps({
        "algorithm": "sha256",
        "file_count": len(files),
        "files": {p.relative_to(WS).as_posix(): sha256(p) for p in files},
    }, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    size = sum(p.stat().st_size for p in WS.rglob("*") if p.is_file())
    print(f"\nwrote {len(files) + 1} files, {size / 1024 / 1024:.2f} MB to {WS}")
    for key, value in provenance["verification"].items():
        print(f"  {key:<32} {value}")
    return 0 if all(provenance["verification"].values()) else 1


_RUBRIC = """# 标注规程(rubric)

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
"""


_README = """# 人工标注证据包

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
"""


if __name__ == "__main__":
    raise SystemExit(main())
