"""Additional diagnostic metrics: per-constraint compliance, no-match and
clarification precision/recall (evaluation guide sections 10.2, 11.4, 11.5)."""

from __future__ import annotations

import pandas as pd

from jobrec.domain.enums import ConstraintOutcome, ConstraintStrength


def per_constraint_compliance(computer, bundles) -> pd.DataFrame:
    """Compliance of recommended jobs per hard-constraint field, per variant.

    Evaluated against the authoritative (full-variant) constraints, so an
    ablation that skips filtering is measured against the true constraints.
    Compliance_c = pass / applicable; `unknown` is tallied separately (never a
    pass).
    """
    tally: dict[tuple[str, str], dict[str, int]] = {}
    for b in bundles:
        ctx = computer._reference_context(b.scenario_id)
        if ctx is None or not b.decision:
            continue
        strength_by_id = {c.constraint_id: c.strength for c in ctx.constraints}
        for jid in b.decision.get("selected_job_ids", []):
            job = computer.jobs_by_id.get(jid)
            if job is None:
                continue
            res = computer.agent.evaluate(job, ctx)
            for c in res.checks:
                if strength_by_id.get(c.constraint_id) != ConstraintStrength.HARD:
                    continue
                key = (b.variant, c.field_name)
                t = tally.setdefault(key, {"pass": 0, "fail": 0, "unknown": 0})
                if c.outcome == ConstraintOutcome.PASS:
                    t["pass"] += 1
                elif c.outcome == ConstraintOutcome.FAIL:
                    t["fail"] += 1
                elif c.outcome == ConstraintOutcome.UNKNOWN:
                    t["unknown"] += 1
    rows = []
    for (variant, field), t in sorted(tally.items()):
        applicable = t["pass"] + t["fail"] + t["unknown"]
        rows.append({
            "variant": variant, "constraint_field": field,
            "pass": t["pass"], "fail": t["fail"], "unknown": t["unknown"],
            "applicable": applicable,
            "compliance": (t["pass"] / applicable) if applicable else None,
            "unknown_rate": (t["unknown"] / applicable) if applicable else None,
        })
    return pd.DataFrame(rows)


def no_match_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """No-match precision / recall / F1 per variant."""
    rows = []
    for variant, sub in run_metrics.groupby("variant"):
        returned_nm = sub[sub["response_type"] == "no_match"]
        expected_nm = sub[sub["no_match_expected"]]
        tp = len(returned_nm[returned_nm["no_match_expected"]])
        precision = tp / len(returned_nm) if len(returned_nm) else None
        recall = tp / len(expected_nm) if len(expected_nm) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else None)
        rows.append({
            "variant": variant, "no_match_returned": len(returned_nm),
            "no_match_expected": len(expected_nm), "true_no_match": tp,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def clarification_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Clarification precision / recall per variant.

    - Precision = clarifications on scenarios that expected clarification AND
      targeting an acceptable slot / all clarification responses.
    - Recall = scenarios expecting clarification that got a (useful) one /
      scenarios expecting clarification.
    """
    rows = []
    for variant, sub in run_metrics.groupby("variant"):
        clar = sub[sub["response_type"] == "clarification"]
        expected = sub[sub["clarification_expected"]]

        def useful(row) -> bool:
            if not row["clarification_expected"]:
                return False
            slots = set(str(row.get("acceptable_slots", "")).split(";")) - {""}
            targets = set(str(row.get("clarification_target", "")).split(";")) - {""}
            return (not slots) or bool(slots & targets)

        useful_count = int(sub.apply(useful, axis=1).sum()) if len(sub) else 0
        precision = (useful_count / len(clar)) if len(clar) else None
        recall = (useful_count / len(expected)) if len(expected) else None
        rows.append({
            "variant": variant, "clarifications": len(clar),
            "expected_clarification": len(expected), "useful": useful_count,
            "precision": precision, "recall": recall,
        })
    return pd.DataFrame(rows)
