"""Auto-extract representative case studies and a root-cause error taxonomy.

Case studies (evaluation guide 25.4): full-beats-no_memory, full-beats-no_context,
a full difficulty/failure case, a correct no-match, and a claim-validator block
(the latter only materialises under a real LLM; deterministic runs ground all
claims by construction).

Error taxonomy (:func:`error_taxonomy`) assigns each task-unsuccessful run ONE primary
root cause, read off the RECORDED run-metric columns rather than off the variant name
wherever the columns can carry the attribution. The categories, in the precedence order
the classifier applies them:

1. ``missing_dialogue_continuation (ablation)`` -- the run ended an *unresolved*
   clarification dialogue: it asked a question, the answer was never fed back
   (``clarification_answered`` false), it returned nothing, and its recorded terminal
   state says the dialogue was not continued (``termination_reason`` in
   :data:`DIALOGUE_NOT_CONTINUED_REASONS`). The failure is single-turn truncation, i.e.
   the multi-turn continuation mechanism, not memory and not constraint handling.
2. ``missing_constraint_enforcement (ablation)`` -- ``no_context`` returned a
   recommendation that violates a hard constraint (``hcsr < 1``).
3. ``no_context_other`` -- any other ``no_context`` failure.
4. ``stale_or_missing_memory (ablation)`` -- a memory-ablated variant asked a
   clarification for evidence it should have carried over from earlier turns or from
   persistent memory (the ask itself is the symptom of the missing memory).
5. ``missing_dialogue_evidence (baseline)`` -- ``profile_only`` ignores the current
   turn, so its failures (including re-asking an already answered slot) are caused by
   the dialogue evidence it never consumes.
6. ``under/over_clarification`` -- any other failure whose final response is a
   clarification, e.g. a dialogue that ran out of turns (``max_turns``), asked a slot
   the user cannot answer (``cannot_answer``) or re-asked an answered slot
   (``repeated_slot``) while continuation WAS available.
7. ``no_match_misclassification`` -- an unexpected no-match.
8. ``other`` -- everything else.

Category 1 is checked first because a truncated dialogue produced no recommendation at
all, so none of the downstream evidence (ranking, HCSR, no-match) exists to attribute
the failure to; and because a recorded terminal state is a direct observation of the
mechanism that ended the run, which outranks the variant-name inferences below it. It
cannot steal from category 2: that one only fires on a recommendation response, while
category 1 only fires on a clarification response.
"""

from __future__ import annotations

import pandas as pd

#: Terminal states in which the dialogue was never continued past the clarification the
#: system asked, so the pending question is still open when the run ends. Today only the
#: experiment condition can produce that (``FeatureFlags.use_multi_turn_continuation``
#: off, i.e. the ``one_shot`` variant or ``memory.use_multi_turn_continuation: false``),
#: recorded by ``jobrec.evaluation.experiment_runner.TERMINATION_CONTINUATION_DISABLED``.
#:
#: The loop's other exits are deliberately NOT in this family. ``cannot_answer``,
#: ``max_turns`` and ``repeated_slot`` are only reachable by a variant that IS allowed to
#: continue: there the dialogue mechanism worked and the failure is what the system asked
#: for (an unanswerable slot, an exhausted turn budget, a re-asked answered slot), which
#: the clarification-quality / baseline categories already own. Folding them in here
#: would relabel those defects as a truncation that did not happen -- and would pull
#: ``profile_only``'s ``repeated_slot`` failures out of the baseline category that
#: legitimately owns them.
DIALOGUE_NOT_CONTINUED_REASONS = frozenset({"continuation_disabled"})


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


def _flag(value) -> bool:
    """Read a recorded boolean column that may arrive as bool, string or NaN."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _text(value) -> str:
    """Read a recorded string column, mapping missing/NaN to the empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _returned(row) -> int:
    """Number of jobs the run delivered; a missing column counts as none delivered."""
    value = row.get("returned_count")
    try:
        if value is None or pd.isna(value):
            return 0
    except (TypeError, ValueError):
        return 0
    return int(value)


def _is_unresolved_clarification(row) -> bool:
    """The run ended with the question it asked still open and nothing delivered.

    Read purely from the recorded columns (``response_type``, ``returned_count``,
    ``clarification_asked``, ``clarification_answered``), so the state is recognised for
    whichever variant produced it.
    """
    return (_text(row.get("response_type")) == "clarification"
            and _returned(row) == 0
            and _flag(row.get("clarification_asked"))
            and not _flag(row.get("clarification_answered")))


def _dialogue_was_not_continued(row) -> bool:
    """Whether the run's terminal state says the dialogue was never continued.

    Keyed on the recorded ``termination_reason`` family
    (:data:`DIALOGUE_NOT_CONTINUED_REASONS`). When no terminal state was recorded --
    an older metrics frame, or a scenario that never entered the clarification loop
    because it did not expect a clarification -- fall back to the scenario's own
    expectation: an *expected* clarification that was asked and never answered is a
    dialogue that stopped one turn early, whereas an *unexpected* clarification means
    asking was itself the defect (the evidence-driven categories own that case).
    """
    reason = _text(row.get("termination_reason"))
    if reason:
        return reason in DIALOGUE_NOT_CONTINUED_REASONS
    return _flag(row.get("clarification_expected"))


def error_taxonomy(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Assign a primary root cause to each task-failure run and tabulate.

    Categories and their precedence are documented in the module docstring. The
    truncated-dialogue rule is applied first and is the only rule derived entirely from
    the recorded dialogue state rather than from the variant name.
    """
    fails = run_metrics[run_metrics.task_success == 0].copy()

    def root_cause(row) -> str:
        v = row["variant"]
        rt = row["response_type"]
        # The run never got to answer its own question: the dialogue was cut off, so
        # the truncation -- not memory, constraints or ranking -- is the root cause.
        if _is_unresolved_clarification(row) and _dialogue_was_not_continued(row):
            return "missing_dialogue_continuation (ablation)"
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
