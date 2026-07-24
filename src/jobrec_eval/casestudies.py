"""Auto-extract representative case studies and a root-cause error taxonomy.

Case studies (evaluation guide 25.4): full-beats-no_memory, full-beats-no_context,
a full difficulty/failure case, a correct no-match, and a claim-validator block
(the latter only materialises under a real LLM; deterministic runs ground all
claims by construction).
"""

from __future__ import annotations

import pandas as pd


def _bundle_index(bundles):
    return {(b.variant, b.scenario_id, b.run_index): b for b in bundles}


def _trace(bundle, grade) -> dict:
    a = bundle.active_search or {}
    d = bundle.decision or {}
    selected = d.get("selected_job_ids", [])
    ranked = {r["job_id"]: r for r in d.get("ranked_jobs", [])}
    top = []
    for jid in selected[:3]:
        rj = ranked.get(jid, {})
        top.append({
            "job_id": jid, "score": rj.get("total_score"),
            "grade": grade.get((bundle.scenario_id, jid)),
            "skill_gaps": rj.get("skill_gaps", []),
        })
    return {
        "variant": bundle.variant,
        "turns": [t.get("text") for t in (bundle.dialogue_state or {}).get("turns", [])
                  if t.get("speaker") == "candidate"],
        "response_type": (bundle.response or {}).get("response_type"),
        "active_roles": a.get("target_roles"), "active_skills": a.get("skills_have"),
        "active_locations": a.get("preferred_locations"),
        "active_salary_min": a.get("salary_min"),
        "hard_fields": a.get("hard_constraint_fields"),
        "no_match_reasons": d.get("no_match_reason_codes", []),
        "selected_top": top,
        "message": ((bundle.response or {}).get("message") or "")[:400],
    }


def extract_cases(bundles, run_metrics: pd.DataFrame, grade: dict) -> dict:
    idx = _bundle_index(bundles)
    rm = run_metrics
    cases: dict[str, dict] = {}

    def pair(scenario_id, base="full", other=None):
        b = idx.get((base, scenario_id, 0))
        o = idx.get((other, scenario_id, 0)) if other else None
        return b, o

    # 1) full beats no_memory (memory-dependent)
    cand = rm[(rm.variant == "full") & (rm.task_success == 1) &
              (rm.memory_dependency.isin(["medium", "high"]))]
    for sid in cand.scenario_id:
        nm = rm[(rm.variant == "no_memory") & (rm.scenario_id == sid) & (rm.task_success == 0)]
        if len(nm):
            b, o = pair(sid, "full", "no_memory")
            cases["full_beats_no_memory"] = {"scenario_id": sid, "full": _trace(b, grade),
                                             "no_memory": _trace(o, grade)}
            break

    # 2) full beats no_context (context-dependent)
    cand = rm[(rm.variant == "full") & (rm.task_success == 1) & (rm.context_dependency == "high")]
    for sid in cand.scenario_id:
        nc = rm[(rm.variant == "no_context") & (rm.scenario_id == sid) & (rm.task_success == 0)]
        if len(nc):
            b, o = pair(sid, "full", "no_context")
            cases["full_beats_no_context"] = {"scenario_id": sid, "full": _trace(b, grade),
                                              "no_context": _trace(o, grade)}
            break

    # 3) correct no-match by full
    cand = rm[(rm.variant == "full") & (rm.no_match_expected) & (rm.response_type == "no_match")]
    if len(cand):
        sid = cand.iloc[0].scenario_id
        cases["correct_no_match"] = {"scenario_id": sid, "full": _trace(idx[("full", sid, 0)], grade)}

    # 4) a full failure / hardest case
    fails = rm[(rm.variant == "full") & (rm.task_success == 0)]
    if len(fails):
        sid = fails.iloc[0].scenario_id
        cases["full_failure"] = {"scenario_id": sid, "full": _trace(idx[("full", sid, 0)], grade),
                                 "note": "full did not meet the task-success rule for this scenario"}
    else:
        hardest = rm[(rm.variant == "full")].sort_values("partial_task_score").head(1)
        if len(hardest):
            sid = hardest.iloc[0].scenario_id
            cases["full_hardest"] = {"scenario_id": sid, "full": _trace(idx[("full", sid, 0)], grade),
                                     "note": "no full task failures; showing the lowest partial-score case"}

    # 5) claim-validator block (dropped claims) — only under a real LLM
    dropped = rm[rm.get("grounding", 1.0) < 1.0] if "grounding" in rm.columns else pd.DataFrame()
    if len(dropped):
        sid = dropped.iloc[0].scenario_id
        v = dropped.iloc[0].variant
        cases["claim_block"] = {"scenario_id": sid, "trace": _trace(idx[(v, sid, 0)], grade)}
    else:
        cases["claim_block_note"] = (
            "Under the deterministic backend every emitted claim is grounded by "
            "construction, so the claim validator drops nothing (grounding = 1.0). "
            "This case becomes informative only under a real LLM backend.")
    return cases


def render_cases_md(cases: dict) -> str:
    lines = []

    def fmt(trace, label):
        top = "; ".join(f"{t['job_id']}(g={t['grade']},s={t['score']})" for t in trace["selected_top"])
        return (f"- **{label}** [{trace['variant']}] — turns={trace['turns']}; "
                f"response={trace['response_type']}; active roles={trace['active_roles']}, "
                f"loc={trace['active_locations']}, salary_min={trace['active_salary_min']}, "
                f"hard={trace['hard_fields']}; top=[{top}]"
                + (f"; no_match_reasons={trace['no_match_reasons']}" if trace['no_match_reasons'] else ""))

    if "full_beats_no_memory" in cases:
        c = cases["full_beats_no_memory"]
        lines += [f"### Case 1 — Memory helps ({c['scenario_id']})", fmt(c["full"], "full ✓"),
                  fmt(c["no_memory"], "no_memory ✗"), ""]
    if "full_beats_no_context" in cases:
        c = cases["full_beats_no_context"]
        lines += [f"### Case 2 — Job-context helps ({c['scenario_id']})", fmt(c["full"], "full ✓"),
                  fmt(c["no_context"], "no_context ✗"), ""]
    if "correct_no_match" in cases:
        c = cases["correct_no_match"]
        lines += [f"### Case 3 — Correct no-match ({c['scenario_id']})", fmt(c["full"], "full ✓"), ""]
    if "full_failure" in cases:
        c = cases["full_failure"]
        lines += [f"### Case 4 — Full failure ({c['scenario_id']})", fmt(c["full"], "full ✗"),
                  f"_{c['note']}_", ""]
    elif "full_hardest" in cases:
        c = cases["full_hardest"]
        lines += [f"### Case 4 — Hardest full case ({c['scenario_id']})", fmt(c["full"], "full"),
                  f"_{c['note']}_", ""]
    if "claim_block" in cases:
        c = cases["claim_block"]
        lines += [f"### Case 5 — Claim validator block ({c['scenario_id']})", fmt(c["trace"], "dropped claim"), ""]
    elif "claim_block_note" in cases:
        lines += ["### Case 5 — Claim validator", cases["claim_block_note"], ""]
    return "\n".join(lines)


def error_taxonomy(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Assign a primary root cause to each task-failure run and tabulate."""
    fails = run_metrics[run_metrics.task_success == 0].copy()

    def root_cause(row) -> str:
        v = row["variant"]
        rt = row["response_type"]
        if v == "no_context" and (row.get("hcsr") is not None and row.get("hcsr", 1) < 1):
            return "missing_constraint_enforcement (ablation)"
        if v == "no_context":
            return "no_context_other"
        if v in ("no_memory", "one_shot") and rt == "clarification":
            return "stale_or_missing_memory (ablation)"
        if v == "profile_only":
            return "missing_dialogue_evidence (baseline)"
        if rt == "clarification":
            return "under/over_clarification"
        if rt == "no_match":
            return "no_match_misclassification"
        return "other"

    if fails.empty:
        return pd.DataFrame(columns=["error_category", "count", "percentage", "most_affected_variant"])
    fails["error_category"] = fails.apply(root_cause, axis=1)
    total = len(fails)
    rows = []
    for cat, sub in fails.groupby("error_category"):
        rows.append({
            "error_category": cat, "count": len(sub),
            "percentage": round(100 * len(sub) / total, 1),
            "most_affected_variant": sub["variant"].value_counts().idxmax(),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)
