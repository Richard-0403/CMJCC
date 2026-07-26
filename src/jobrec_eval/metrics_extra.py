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



# A penalty large enough to dominate any turn/unnecessary-ask contribution for
# realistic runs (turns are bounded by ExperimentConfig.max_dialogue_turns, and
# the number of asked slots is small). This guarantees the R7.5 monotonicity
# invariant: a run that SKIPS a necessary clarification can never receive a
# higher efficiency score than one that asked it, regardless of turn counts.
_SKIP_PENALTY = 1_000_000.0
_UNNECESSARY_PENALTY = 1.0


def _slot_set(value) -> set[str]:
    """Parse a ``;``-joined slot string into a set (dropping empties)."""
    return set(str(value or "").split(";")) - {""}


def clarification_efficiency(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Necessary / unnecessary clarification classification and efficiency per variant.

    Each asked clarification slot (``clarification_target``) is classified against
    the scenario's ``acceptable_slots``:

    - **Necessary** — the slot IS an acceptable slot (the reference says the
      answer is required to reach the correct outcome).
    - **Unnecessary** — the slot was asked but is NOT an acceptable slot (a wasted
      turn).

    A run "misses" a necessary clarification when the scenario expected a
    clarification (``clarification_expected``) over a non-empty ``acceptable_slots``
    set, yet none of the asked slots was acceptable (including the case where the
    run asked nothing and guessed instead).

    Efficiency (higher = more efficient) is computed per run as
    ``-turns - _UNNECESSARY_PENALTY * unnecessary_asked``, so fewer turns and fewer
    wasted asks score better. A run that missed a necessary clarification has
    ``_SKIP_PENALTY`` subtracted, which — because turn counts and ask counts are
    bounded well below ``_SKIP_PENALTY`` — guarantees the R7.5 monotonicity rule:
    skipping a necessary clarification is NEVER scored as more efficient than
    asking it. The per-variant ``efficiency_score`` is the mean of the per-run
    scores.

    ``turns`` is taken from ``response_turns`` when present (added by tasks
    10.2/10.3); otherwise it falls back to ``turn_count`` and finally to ``1`` when
    neither is available, so existing callers/columns are never broken.
    """
    has_response_turns = "response_turns" in run_metrics.columns
    has_turn_count = "turn_count" in run_metrics.columns

    def _turns(row) -> float:
        if has_response_turns and pd.notna(row.get("response_turns")):
            return float(row["response_turns"])
        if has_turn_count and pd.notna(row.get("turn_count")):
            return float(row["turn_count"])
        return 1.0

    rows = []
    for variant, sub in run_metrics.groupby("variant"):
        necessary_asked = necessary_missed = unnecessary_asked = 0
        efficiencies: list[float] = []
        for _, row in sub.iterrows():
            acceptable = _slot_set(row.get("acceptable_slots"))
            asked = _slot_set(row.get("clarification_target"))
            n_necessary = len(asked & acceptable)
            n_unnecessary = len(asked - acceptable)
            missed = bool(row.get("clarification_expected")) and bool(acceptable) \
                and n_necessary == 0

            necessary_asked += n_necessary
            unnecessary_asked += n_unnecessary
            if missed:
                necessary_missed += 1

            eff = -_turns(row) - _UNNECESSARY_PENALTY * n_unnecessary
            if missed:
                eff -= _SKIP_PENALTY
            efficiencies.append(eff)

        rows.append({
            "variant": variant,
            "runs": len(sub),
            "necessary_asked": necessary_asked,
            "necessary_missed": necessary_missed,
            "unnecessary_asked": unnecessary_asked,
            "efficiency_score": (sum(efficiencies) / len(efficiencies))
            if efficiencies else None,
        })
    return pd.DataFrame(rows)
