"""Additional diagnostic metrics: per-constraint compliance, no-match and
clarification precision/recall (evaluation guide sections 10.2, 11.4, 11.5)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from jobrec.domain.enums import ConstraintOutcome, ConstraintStrength

from .relevance import grade_lookup

# --------------------------------------------------------------------- shared helpers
_TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}


def _as_bool(value) -> bool:
    """Coerce a cell (bool / int / float / str / NaN) to a strict boolean.

    Strings are matched case-insensitively against a small truthy set so a
    ``run_metrics`` frame reloaded from CSV (where booleans round-trip as text)
    behaves the same as one built in-memory. Missing values (NaN/None) are False.
    """
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value)


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


# ------------------------------------------------- R7 clarification (dialogue-level)
# Three penalty tiers, each an order of magnitude apart so the tier ALWAYS decides the
# ranking before turn counts can (turns are bounded by ExperimentConfig.max_dialogue_turns
# and the number of asked slots is small, so the wasted-ask/turn term of one run stays in
# the single digits):
#
#   _UNNECESSARY_PENALTY (1e0)  -- one wasted ask, comparable to one extra turn.
#   _UNRESOLVED_PENALTY  (1e3)  -- the dialogue never reached a terminal answer.
#   _SKIP_PENALTY        (1e6)  -- a necessary clarification was never asked.
#
# Together they give the per-run ordering
#
#   asked & resolved  >  asked & abandoned  >  necessary clarification skipped
#
# regardless of turn counts. The first inequality is why _UNRESOLVED_PENALTY exists: a
# variant that asks the right question and then abandons the dialogue (the dialogue loop
# terminating on ``continuation_disabled``, ``max_turns``, ``cannot_answer`` or
# ``repeated_slot``) used to look MORE efficient than one that asked and answered, simply
# because its dialogue was one turn shorter. Abandoning a dialogue is not efficiency, so it
# can no longer be rewarded. The second inequality is the R7.5 monotonicity invariant: a run
# that SKIPS a necessary clarification can never receive a higher efficiency score than one
# that asked it, regardless of turn counts.
_SKIP_PENALTY = 1_000_000.0
_UNRESOLVED_PENALTY = 1_000.0
_UNNECESSARY_PENALTY = 1.0

#: Per-run columns naming the slots a run asked about. ``clarification_asked_slots`` is
#: the trace-derived set of ALL asks across the dialogue (added by ``metrics._one``);
#: ``clarification_target`` is the final pending ask, kept for backward compatibility.
#: Their union is what the dialogue asked about, so a frame carrying either column (or a
#: frame reloaded from an older CSV) is scored the same way.
_ASKED_SLOT_COLUMNS = ("clarification_asked_slots", "clarification_target")

#: Per-run boolean columns describing the dialogue, all optional so pre-trace frames
#: still work: whether the system asked, whether the user answered, whether a slot was
#: re-asked.
_ASKED_COLUMN = "clarification_asked"
_ANSWERED_COLUMN = "clarification_answered"
_REPEATED_COLUMN = "clarification_repeated_slot"

#: Per-run column carrying the FINAL response of the dialogue, and the value that means the
#: run ended while still asking. ``jobrec.evaluation.experiment_runner`` only leaves a
#: clarification as the final response when the dialogue never reached a terminal answer, so
#: this one value covers every unresolved termination (``continuation_disabled``,
#: ``max_turns``, ``cannot_answer``, ``repeated_slot``) without naming any of them.
_RESPONSE_TYPE_COLUMN = "response_type"
_CLARIFICATION_RESPONSE = "clarification"


def _is_clarification_response(row) -> bool:
    """Is the run's FINAL recorded response a clarification (a still-pending question)?

    Compared case-insensitively on the text so a frame reloaded from CSV behaves like one
    built in memory; a missing column or value is False (no evidence either way).
    """
    value = row.get(_RESPONSE_TYPE_COLUMN)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() == _CLARIFICATION_RESPONSE

_CLARIFICATION_COLUMNS = [
    "variant", "runs", "clarifications", "expected_clarification", "useful",
    "asked_clarifications", "necessary_asked", "unnecessary_asked", "answered", "repeated",
    "precision", "recall", "necessary_recall", "useful_rate", "unnecessary_rate",
    "repeated_question_rate", "answered_rate",
]


def _slot_set(value) -> set[str]:
    """Parse a ``;``-joined slot string into a set (dropping empties).

    Missing values are empty sets: an empty slot column round-trips through CSV as
    ``NaN``, which must never be read as a slot literally named ``nan``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return set(str(value).split(";")) - {""}


def _asked_slots(row) -> set[str]:
    """Every slot the run asked about, from whichever slot columns the frame carries."""
    slots: set[str] = set()
    for column in _ASKED_SLOT_COLUMNS:
        slots |= _slot_set(row.get(column))
    return slots


def _asked_clarification(row) -> bool:
    """Did this run ask a clarification anywhere in its dialogue? (R7.3)

    Any recorded signal counts: the trace-derived ``clarification_asked`` flag, a
    recorded asked slot, or a final response of ``clarification`` (the only signal older
    frames carry).
    """
    return bool(_as_bool(row.get(_ASKED_COLUMN)) or _asked_slots(row)
                or _is_clarification_response(row))


def _dialogue_unresolved(row) -> bool:
    """Did this run's dialogue end without ever reaching a terminal answer? (R7.4/R7.5)

    The signal is the FINAL response still being a clarification. The dialogue loop in
    ``jobrec.evaluation.experiment_runner`` only stops on a terminal response
    (recommendation / no-match / error) or on a guard firing while the pending question is
    still open, so "the last thing the system produced is a question" is exactly the set of
    unresolved dialogues -- ``continuation_disabled``, ``max_turns``, ``cannot_answer`` and
    ``repeated_slot`` alike, plus any future guard, with no termination reason enumerated
    here.

    It is deliberately NOT ``clarification_answered``: the simulated user does answer in the
    ``repeated_slot`` case, and the dialogue is unresolved all the same. It is also not
    ``task_success``, which additionally folds in ranking/grounding quality and would
    conflate "abandoned the dialogue" with "answered badly" (measured elsewhere).

    It is likewise NOT ``casestudies.DIALOGUE_NOT_CONTINUED_REASONS``: that constant
    deliberately names one reason (``continuation_disabled``) to attribute a case study to
    the continuation mechanism, which is the opposite of what is needed here -- scoring must
    treat every abandoned dialogue alike, so this predicate diverges from it on purpose
    rather than reusing an enumeration that would special-case a single variant's reason.

    A frame with no ``response_type`` column carries no evidence either way and is treated
    as resolved, so pre-trace frames score exactly as before.
    """
    return _is_clarification_response(row)


def _classify_asks(row) -> tuple[int, int, bool]:
    """Classify one run's asks: ``(necessary, unnecessary, missed_necessary)`` (R7.4).

    Each asked slot is Necessary when it IS one of the scenario's ``acceptable_slots``
    (the reference says the answer is required to reach the correct outcome) and
    Unnecessary otherwise (a wasted turn). A run *misses* a necessary clarification when
    the scenario expected one over a non-empty ``acceptable_slots`` set yet no acceptable
    slot was asked -- including the case where it asked nothing and guessed instead.
    """
    acceptable = _slot_set(row.get("acceptable_slots"))
    asked = _asked_slots(row)
    necessary = len(asked & acceptable)
    unnecessary = len(asked - acceptable)
    missed = (_as_bool(row.get("clarification_expected")) and bool(acceptable)
              and necessary == 0)
    return necessary, unnecessary, missed


def _useful_ask(row) -> bool:
    """Was this run's clarification useful? (R7.4)

    Useful means the run actually asked, the scenario needed a clarification, and the
    target was an acceptable slot (an empty ``acceptable_slots`` set leaves the target
    unconstrained). A clarification asked where none was needed is never useful.
    """
    if not _asked_clarification(row) or not _as_bool(row.get("clarification_expected")):
        return False
    necessary, _unnecessary, missed = _classify_asks(row)
    return bool(necessary) or not missed


def clarification_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Clarification precision / recall and dialogue-level rates per variant (R7.3/R7.4).

    Everything is derived from the whole dialogue rather than the final response: the
    per-run columns ``clarification_asked`` / ``clarification_asked_slots`` /
    ``clarification_answered`` / ``clarification_repeated_slot`` come from
    ``dialogue_trace.jsonl`` (see :func:`jobrec_eval.metrics.dialogue_view`), so a run that
    asked the right question and then recommended is credited instead of scoring 0.
    Frames that predate those columns fall back to the final response and
    ``clarification_target``, so older CSVs still tabulate.

    Run-level columns:

    - ``clarifications`` — runs that asked a clarification at any point (the precision
      denominator), ``expected_clarification`` — runs whose scenario expected one.
    - ``useful`` — runs whose ask was necessary; ``precision`` = ``useful`` over runs that
      asked; ``recall`` / ``necessary_recall`` — the same necessary-clarification recall
      over the scenarios that expected one (``recall`` is the legacy column name).
    - ``answered_rate`` — asking runs whose clarification the simulated user answered;
      ``repeated_question_rate`` — asking runs that re-asked a slot (or tripped the
      repeated-slot guard). ``repeated_question_rate`` is ``None`` when the frame carries
      no repeat column at all, rather than an unfounded 0.0.

    Ask-level columns count individual asked slots, so a run that asked two slots
    contributes two: ``asked_clarifications``, ``necessary_asked``, ``unnecessary_asked``
    and their shares ``useful_rate`` / ``unnecessary_rate``.

    Every rate is ``None`` on an empty denominator, matching the convention of the other
    metrics in this module (never a misleading 0.0 or 1.0).
    """
    has_repeated = _REPEATED_COLUMN in run_metrics.columns
    rows = []
    for variant, sub in run_metrics.groupby("variant"):
        asked_runs = useful_runs = answered_runs = repeated_runs = 0
        necessary = unnecessary = 0
        expected = int(sub["clarification_expected"].map(_as_bool).sum())
        for _, row in sub.iterrows():
            n_necessary, n_unnecessary, _missed = _classify_asks(row)
            if not _asked_clarification(row):
                continue
            asked_runs += 1
            necessary += n_necessary
            unnecessary += n_unnecessary
            useful_runs += int(_useful_ask(row))
            answered_runs += int(_as_bool(row.get(_ANSWERED_COLUMN)))
            repeated_runs += int(_as_bool(row.get(_REPEATED_COLUMN)))

        asks = necessary + unnecessary
        recall = (useful_runs / expected) if expected else None
        rows.append({
            "variant": variant,
            "runs": len(sub),
            "clarifications": asked_runs,
            "expected_clarification": expected,
            "useful": useful_runs,
            "asked_clarifications": asks,
            "necessary_asked": necessary,
            "unnecessary_asked": unnecessary,
            "answered": answered_runs,
            "repeated": repeated_runs if has_repeated else None,
            "precision": (useful_runs / asked_runs) if asked_runs else None,
            "recall": recall,
            "necessary_recall": recall,
            "useful_rate": (necessary / asks) if asks else None,
            "unnecessary_rate": (unnecessary / asks) if asks else None,
            "repeated_question_rate": ((repeated_runs / asked_runs)
                                       if has_repeated and asked_runs else None),
            "answered_rate": (answered_runs / asked_runs) if asked_runs else None,
        })
    return pd.DataFrame(rows, columns=_CLARIFICATION_COLUMNS)


def clarification_efficiency_per_run(run_metrics: pd.DataFrame) -> pd.Series:
    """Per-run clarification efficiency score (higher = more efficient) (R7.4/R7.5).

    The single definition of the score, shared by :func:`clarification_efficiency` (which
    averages it per variant) and by the ``clarification_efficiency`` column of
    ``run_metrics``, so the two can never drift apart. One run scores

    ``-turns - _UNNECESSARY_PENALTY * unnecessary_asked``

    minus :data:`_UNRESOLVED_PENALTY` when the dialogue never reached a terminal answer
    (:func:`_dialogue_unresolved`) and minus :data:`_SKIP_PENALTY` when the run missed a
    necessary clarification. Because turn counts and ask counts are bounded far below both
    penalties, and the two penalties are three orders of magnitude apart, the tier always
    decides the ranking before turn counts can:

    ``asked & resolved  >  asked & abandoned  >  necessary clarification skipped``

    The right-hand inequality is the R7.5 monotonicity rule (skipping a necessary
    clarification is NEVER scored as more efficient than asking it). The left-hand one says
    a shorter dialogue cannot buy efficiency by abandoning the question it asked: a run that
    asked the necessary slot but ended still asking (continuation disabled, turn cap,
    unanswerable, repeated slot) never out-scores one that asked and then answered, however
    many turns the answering run needed.

    ``turns`` is taken from ``response_turns`` when present (the dialogue-trace turn
    count), otherwise from ``turn_count``, and finally ``1`` when neither is available, so
    frames written before the trace columns existed still score.
    """
    has_response_turns = "response_turns" in run_metrics.columns
    has_turn_count = "turn_count" in run_metrics.columns

    def _turns(row) -> float:
        if has_response_turns and pd.notna(row.get("response_turns")):
            return float(row["response_turns"])
        if has_turn_count and pd.notna(row.get("turn_count")):
            return float(row["turn_count"])
        return 1.0

    def _score(row) -> float:
        _necessary, unnecessary, missed = _classify_asks(row)
        score = -_turns(row) - _UNNECESSARY_PENALTY * unnecessary
        if _dialogue_unresolved(row):
            score -= _UNRESOLVED_PENALTY
        return score - _SKIP_PENALTY if missed else score

    if run_metrics.empty:
        return pd.Series(dtype=float)
    return pd.Series([_score(row) for _, row in run_metrics.iterrows()],
                     index=run_metrics.index, dtype=float)


#: Columns of the :func:`clarification_efficiency` table. The first five are the original
#: classification/score columns plus ``asked_unresolved``, which sits next to
#: ``efficiency_score`` so the score is always read beside the number of dialogues the
#: variant abandoned; the ``*_response_turns`` block is the response-turn distribution the
#: report reads for its median/IQR figures.
_CLARIFICATION_EFFICIENCY_COLUMNS = [
    "variant", "runs", "necessary_asked", "necessary_missed", "unnecessary_asked",
    "asked_unresolved", "efficiency_score", "response_turns_n", "median_response_turns",
    "q1_response_turns", "q3_response_turns", "iqr_response_turns",
]

#: Per-run dialogue turn count the response-turn distribution is computed from. Deliberately
#: NOT falling back to ``turn_count``: the distribution describes how many turns the
#: dialogue needed, which only the trace-derived column measures, and a bundle without a
#: trace must read N/A rather than borrow another column's number.
_RESPONSE_TURNS_COLUMN = "response_turns"


def _quartiles(values: list[float]) -> tuple[float | None, float | None]:
    """``(Q1, Q3)`` of ``values``, or ``(None, None)`` when empty.

    Each hinge is the :func:`_median` of one half of the sorted sample, with the overall
    median excluded from both halves for an odd sample size (the exclusive convention). A
    single observation is its own Q1 and Q3, so the IQR of one run is 0.0 rather than
    undefined.
    """
    if not values:
        return None, None
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    half = n // 2
    lower = ordered[:half]
    upper = ordered[half:] if n % 2 == 0 else ordered[half + 1:]
    q1 = _median(lower) if lower else float(ordered[0])
    q3 = _median(upper) if upper else float(ordered[-1])
    return q1, q3


def _response_turn_values(run_metrics: pd.DataFrame) -> list[float]:
    """Every recorded ``response_turns`` value in ``run_metrics`` (missing ones dropped).

    Bundles without a dialogue trace carry ``response_turns = None``, and a frame written
    before the column existed carries no column at all; both yield an empty list, which the
    callers turn into ``None`` rather than a fabricated 0.
    """
    if _RESPONSE_TURNS_COLUMN not in run_metrics.columns:
        return []
    return [float(v) for v in run_metrics[_RESPONSE_TURNS_COLUMN].dropna().tolist()]


def clarification_efficiency(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Necessary / unnecessary clarification classification and efficiency per variant.

    Asks are classified by :func:`_classify_asks` over every slot the dialogue asked about
    (``clarification_asked_slots`` when the trace-derived column is present, else the final
    ``clarification_target``):

    - **Necessary** — the slot IS an acceptable slot (the reference says the
      answer is required to reach the correct outcome).
    - **Unnecessary** — the slot was asked but is NOT an acceptable slot (a wasted
      turn).

    A run "misses" a necessary clarification when the scenario expected a
    clarification (``clarification_expected``) over a non-empty ``acceptable_slots``
    set, yet none of the asked slots was acceptable (including the case where the
    run asked nothing and guessed instead).

    The per-variant ``efficiency_score`` is the mean of the per-run scores produced by
    :func:`clarification_efficiency_per_run` -- the single definition of the score, which
    keeps the R7.5 monotonicity invariant (skipping a necessary clarification is never
    scored more efficient than asking it) and the companion rule that abandoning a dialogue
    after asking is never scored more efficient than asking and answering.
    ``None`` when the variant has no runs.

    ``asked_unresolved`` makes that second rule auditable: it counts the runs that asked a
    clarification and then ended with the question still pending (:func:`_dialogue_unresolved`
    -- continuation disabled, turn cap, unanswerable, repeated slot). It is reported NEXT TO
    the score so a variant whose score is depressed by abandoned dialogues can be read as
    such, and so a high score can never be mistaken for good behaviour while this count is
    non-zero.

    The ``*_response_turns`` columns carry the response-turn DISTRIBUTION of the variant,
    which is what makes the score interpretable jointly with task success (R7.4): asking
    fewer questions is only efficiency when the outcome is still correct, and the score
    already encodes that through its skip penalty.

    - ``response_turns_n`` — runs with a recorded ``response_turns`` value (the
      denominator of the distribution; runs whose bundle has no dialogue trace are
      excluded).
    - ``median_response_turns`` / ``q1_response_turns`` / ``q3_response_turns`` /
      ``iqr_response_turns`` — the median, the two hinges from :func:`_quartiles` and
      ``Q3 − Q1``. All ``None`` when the variant recorded no turn counts, matching the
      None-on-empty convention of this module.
    """
    rows = []
    for variant, sub in run_metrics.groupby("variant"):
        necessary_asked = necessary_missed = unnecessary_asked = asked_unresolved = 0
        for _, row in sub.iterrows():
            n_necessary, n_unnecessary, missed = _classify_asks(row)
            necessary_asked += n_necessary
            unnecessary_asked += n_unnecessary
            necessary_missed += int(missed)
            asked_unresolved += int(_asked_clarification(row) and _dialogue_unresolved(row))

        efficiencies = clarification_efficiency_per_run(sub)
        turns = _response_turn_values(sub)
        q1, q3 = _quartiles(turns)
        rows.append({
            "variant": variant,
            "runs": len(sub),
            "necessary_asked": necessary_asked,
            "necessary_missed": necessary_missed,
            "unnecessary_asked": unnecessary_asked,
            "asked_unresolved": asked_unresolved,
            "efficiency_score": float(efficiencies.mean()) if len(efficiencies) else None,
            "response_turns_n": len(turns),
            "median_response_turns": _median(turns),
            "q1_response_turns": q1,
            "q3_response_turns": q3,
            "iqr_response_turns": (q3 - q1) if (q1 is not None and q3 is not None) else None,
        })
    return pd.DataFrame(rows, columns=_CLARIFICATION_EFFICIENCY_COLUMNS)


# --------------------------------------------------------------- R10 failure-path rates
# These four rates back Requirement 10.8/10.9: over a scenario set that contains genuine
# failure paths, grounding/handoff numbers must NOT be trivially fixed at 1.000, and the
# pipeline must report how many injected failures were detected and how many recoverable
# failures actually recovered. Each rate returns ``None`` (never a misleading 0.0/1.0) when
# it has no data to average over, matching the None-on-empty convention used by
# ``no_match_metrics`` / ``per_constraint_compliance`` above.

def _bool_mask(run_metrics: pd.DataFrame, column: str) -> pd.Series | None:
    """Return a boolean Series for ``column`` or ``None`` when it is absent."""
    if column not in run_metrics.columns:
        return None
    return run_metrics[column].map(_as_bool)


def _flag_counts(run_metrics: pd.DataFrame, condition: str, outcome: str,
                 ) -> tuple[int, int] | None:
    """``(numerator, denominator)`` for ``outcome`` among rows where ``condition`` holds.

    ``None`` when either column is absent, so a frame without the instrumentation reads
    as N/A instead of reporting a rate over an assumed-false column. The numerator
    requires BOTH flags, so an outcome recorded without its condition is never
    over-counted.
    """
    cond = _bool_mask(run_metrics, condition)
    out = _bool_mask(run_metrics, outcome)
    if cond is None or out is None:
        return None
    return int((cond & out).sum()), int(cond.sum())


def _claim_counts(bundles) -> tuple[int, int]:
    """``(supported, factual)`` claim counts across ``bundles`` (the grounding fraction).

    ``non_factual`` claims are excluded from the denominator and a claim counts as
    supported only when its ``support_status`` is exactly ``"supported"``, mirroring the
    per-run ``grounding`` definition in ``metrics.py``.
    """
    total = supported = 0
    for b in bundles:
        for c in b.claims:
            if c.get("claim_type") == "non_factual":
                continue
            total += 1
            if c.get("support_status") == "supported":
                supported += 1
    return supported, total


def _handoff_counts(bundles) -> tuple[int, int]:
    """``(valid_completed, attempted)`` handoff counts across ``bundles``.

    A handoff succeeds only when it both passed validation and reached
    ``status == "completed"``, the same rule as the per-run ``handoff_success`` metric.
    """
    total = valid = 0
    for b in bundles:
        for h in b.handoffs:
            total += 1
            if h.get("validation_passed") and h.get("status") == "completed":
                valid += 1
    return valid, total


def failure_detection_rate(run_metrics: pd.DataFrame) -> float | None:
    """Fraction of injected failures the system detected: ``detected / injected`` (R10.8).

    Reads two boolean columns from ``run_metrics``:

    - ``failure_injected`` — a fault was injected into this run (dangling evidence
      id, unsupported claim, invalid handoff, agent exception/timeout, ...).
    - ``failure_detected`` — the system flagged/handled that fault (dropped the
      claim, rejected the handoff, emitted a ``failure_code``, ...).

    The numerator counts runs that were BOTH injected and detected, so a run that
    reports a detection without an injected fault is never over-counted. Returns
    ``None`` when the required columns are missing or no failures were injected
    (empty denominator), so a happy-path-only set reads as N/A rather than 1.000.
    """
    counts = _flag_counts(run_metrics, "failure_injected", "failure_detected")
    if counts is None or counts[1] == 0:
        return None
    return counts[0] / counts[1]


def recovery_success_rate(run_metrics: pd.DataFrame) -> float | None:
    """Fraction of recoverable failures that recovered: ``recovered / recoverable`` (R10.8).

    Reads two boolean columns from ``run_metrics``:

    - ``recoverable`` — the injected failure was designed to be recoverable
      (timeout-with-retry, partial failure with rule fallback).
    - ``recovered`` — the run actually recovered and completed.

    The numerator counts runs that were BOTH recoverable and recovered. Returns
    ``None`` when the columns are missing or nothing recoverable occurred, so the
    rate is well-defined and never silently reported as a perfect 1.000.
    """
    counts = _flag_counts(run_metrics, "recoverable", "recovered")
    if counts is None or counts[1] == 0:
        return None
    return counts[0] / counts[1]


def grounding_rate(bundles) -> float | None:
    """Supported factual claims / all factual claims across ``bundles`` (R10.8/10.9).

    Mirrors the per-run ``grounding`` definition in ``metrics.py``: ``non_factual``
    claims are excluded from the denominator, and a claim counts as grounded only
    when its ``support_status`` is exactly ``"supported"`` (``unsupported`` /
    ``unknown`` never count). Over a failure-containing set some claims are dropped
    or flagged, so the rate is strictly ``< 1.000`` (R10.9). Returns ``None`` when
    there are no factual claims to score.
    """
    supported, total = _claim_counts(bundles)
    return (supported / total) if total else None


def handoff_success_rate(bundles) -> float | None:
    """Valid completed handoffs / all handoffs across ``bundles`` (R10.8/10.9).

    A handoff succeeds only when it both passed validation and reached
    ``status == "completed"`` (same rule as the per-run ``handoff_success`` metric
    in ``metrics.py``). Schema-invalid or missing-field handoffs therefore drag the
    rate strictly below ``1.000`` over a failure-containing set (R10.9). Returns
    ``None`` when no handoffs were recorded.
    """
    valid, total = _handoff_counts(bundles)
    return (valid / total) if total else None


#: Tidy long-form shape of :func:`failure_metrics`: one row per rate, so the CSV stays
#: readable as more failure-path rates are added and an absent rate is a blank ``value``
#: cell rather than a missing column.
_FAILURE_METRIC_COLUMNS = ["metric", "source", "numerator", "denominator", "value"]

#: Row order and provenance of the four R10 rates, kept explicit so the CSV is stable.
_FAILURE_METRIC_ROWS = (
    ("failure_detection_rate", "run_metrics"),
    ("recovery_success_rate", "run_metrics"),
    ("grounding_rate", "run_bundles"),
    ("handoff_success_rate", "run_bundles"),
)


def failure_metrics(run_metrics: pd.DataFrame, bundles) -> pd.DataFrame:
    """The four R10 failure-path rates as one tidy table (R10.8/10.9).

    One row per rate with columns ``metric``, ``source`` (``run_metrics`` for the
    injection/recovery flags, ``run_bundles`` for the claim/handoff artifacts),
    ``numerator``, ``denominator`` and ``value``:

    - ``failure_detection_rate`` — detected over injected faults.
    - ``recovery_success_rate`` — recovered over recoverable faults.
    - ``grounding_rate`` — supported over factual claims.
    - ``handoff_success_rate`` — validated-and-completed over attempted handoffs.

    ``value`` comes from the four public rate functions above, so the published number
    has exactly one definition; the numerator/denominator columns are what let the report
    say *why* a rate is N/A. A ``denominator`` of 0 means the loaded runs contained
    nothing to measure (no fault was injected, no claim was made), and ``value`` is then
    ``None`` — never a misleading 0.0 or 1.000. A ``denominator`` of ``None`` means the
    frame does not carry the instrumentation columns at all.
    """
    detection = _flag_counts(run_metrics, "failure_injected", "failure_detected")
    recovery = _flag_counts(run_metrics, "recoverable", "recovered")
    supported, factual = _claim_counts(bundles)
    valid_handoffs, attempted_handoffs = _handoff_counts(bundles)

    counts: dict[str, tuple[int | None, int | None]] = {
        "failure_detection_rate": detection or (None, None),
        "recovery_success_rate": recovery or (None, None),
        "grounding_rate": (supported, factual),
        "handoff_success_rate": (valid_handoffs, attempted_handoffs),
    }
    values: dict[str, float | None] = {
        "failure_detection_rate": failure_detection_rate(run_metrics),
        "recovery_success_rate": recovery_success_rate(run_metrics),
        "grounding_rate": grounding_rate(bundles),
        "handoff_success_rate": handoff_success_rate(bundles),
    }
    rows = []
    for metric, source in _FAILURE_METRIC_ROWS:
        numerator, denominator = counts[metric]
        rows.append({
            "metric": metric, "source": source,
            "numerator": numerator, "denominator": denominator,
            "value": values[metric],
        })
    return pd.DataFrame(rows, columns=_FAILURE_METRIC_COLUMNS)


# --------------------------------------------------- R13 extraction-source statistics
# Requirement 13 asks how much of the extracted structure the LLM actually produced:
# per variant AND per scenario type, how many fields came from the rule extractor vs
# the model, and how often the validation ladder had to repair, fall back to rules, or
# give up. Requirement 8.12 additionally asks for the hybrid-mode schema-failure and
# fallback rates. The inputs are the per-field provenance tags written by
# ``orchestrator._extract`` onto ``ExtractedPreference.metadata`` and persisted in each
# bundle's ``extracted_preferences.json``.

_METHOD_RULE = "rule"
_METHOD_LLM = "llm"

#: Ladder rungs recorded as ``extraction_source``. Anything other than ``normalized``
#: means the model's raw output did not validate on the first attempt.
_SOURCE_NORMALIZED = "normalized"
_SOURCE_REPAIRED = "repaired"
_SOURCE_RULE_FALLBACK = "rule_fallback"
_SOURCE_UNRESOLVED = "unresolved"

_SCOPE_VARIANT = "variant"
_SCOPE_SCENARIO_TYPE = "variant_scenario_type"

_EXTRACTION_COLUMNS = [
    "scope", "variant", "scenario_type", "runs", "fields",
    "rule_fields", "llm_fields", "rule_share", "llm_share",
    "normalized_fields", "repaired_fields", "rule_fallback_fields", "unresolved_fields",
    "hybrid_runs", "hybrid_fields", "schema_failure_rate", "fallback_rate",
]


def _run_mode(bundle) -> str | None:
    """Run mode of a bundle (``deterministic`` / ``hybrid`` / ``replay``).

    Read from the provider manifest recorded on the run record, so hybrid-only
    rates (R8.12) are never computed over deterministic runs.
    """
    manifest = (bundle.run_record or {}).get("model_manifest") or {}
    mode = manifest.get("mode")
    return str(mode) if mode is not None else None


def _preference_provenance(bundle) -> list[tuple[str | None, str | None]]:
    """``(extraction_method, extraction_source)`` for every extracted field of a run."""
    dump = getattr(bundle, "extracted_preferences", None) or {}
    out: list[tuple[str | None, str | None]] = []
    for pref in dump.get("preferences", []) or []:
        metadata = pref.get("metadata") or {}
        method = metadata.get("extraction_method")
        # ``source`` is accepted as an alias so a bundle written by an older
        # exporter still contributes its rung information.
        source = metadata.get("extraction_source", metadata.get("source"))
        out.append((method, source))
    return out


def _blank_tally() -> dict[str, int]:
    return {
        "runs": 0, "fields": 0,
        _METHOD_RULE: 0, _METHOD_LLM: 0,
        _SOURCE_NORMALIZED: 0, _SOURCE_REPAIRED: 0,
        _SOURCE_RULE_FALLBACK: 0, _SOURCE_UNRESOLVED: 0,
        "hybrid_runs": 0, "hybrid_fields": 0,
        "hybrid_schema_failures": 0, "hybrid_fallbacks": 0,
    }


def _tally_row(scope: str, variant: str, scenario_type: str, t: dict[str, int]) -> dict:
    """Render one tally as a report row, using ``None`` for undefined rates."""
    fields = t["fields"]
    hybrid_fields = t["hybrid_fields"]
    return {
        "scope": scope,
        "variant": variant,
        "scenario_type": scenario_type,
        "runs": t["runs"],
        "fields": fields,
        "rule_fields": t[_METHOD_RULE],
        "llm_fields": t[_METHOD_LLM],
        "rule_share": (t[_METHOD_RULE] / fields) if fields else None,
        "llm_share": (t[_METHOD_LLM] / fields) if fields else None,
        "normalized_fields": t[_SOURCE_NORMALIZED],
        "repaired_fields": t[_SOURCE_REPAIRED],
        "rule_fallback_fields": t[_SOURCE_RULE_FALLBACK],
        "unresolved_fields": t[_SOURCE_UNRESOLVED],
        "hybrid_runs": t["hybrid_runs"],
        "hybrid_fields": hybrid_fields,
        "schema_failure_rate": (t["hybrid_schema_failures"] / hybrid_fields)
        if hybrid_fields else None,
        "fallback_rate": (t["hybrid_fallbacks"] / hybrid_fields) if hybrid_fields else None,
    }


def extraction_source_metrics(bundles, scenarios=None) -> pd.DataFrame:
    """Rule-vs-LLM extraction and fallback counts per variant and scenario type (R13.2).

    Reads the per-field provenance tags persisted in each run bundle's
    ``extracted_preferences.json``:

    - ``extraction_method`` — ``rule`` (deterministic rule extractor) or ``llm``
      (model output that survived validation), giving ``rule_fields`` /
      ``llm_fields`` and their shares.
    - ``extraction_source`` — which rung of the validation ladder produced the
      value: ``normalized`` (validated as emitted), ``repaired`` (schema repair or
      the bounded retry fixed it), ``rule_fallback`` (the rule extractor supplied
      the value), or ``unresolved`` (a stated value preserved as an unconfirmed
      constraint because no rung could normalize it, R8.9).

    Two rate columns answer R8.12 and are restricted to **hybrid** runs, identified
    from the provider manifest on the run record:

    - ``schema_failure_rate`` — fields whose source is not ``normalized`` (i.e.
      needed repair/retry/fallback or stayed unresolved) over all hybrid fields.
    - ``fallback_rate`` — fields whose source is ``rule_fallback`` over all hybrid
      fields.

    Both are ``None`` when there are no hybrid fields, so a deterministic-only
    experiment reads as N/A instead of a misleading 0.0 (matching the convention of
    the other metrics in this module).

    Rows come in two scopes, tagged in the ``scope`` column so totals are never
    double-counted: ``variant`` (one row per variant, ``scenario_type == "(all)"``)
    and ``variant_scenario_type`` (one row per variant x scenario type). Scenario
    types are taken from ``scenarios`` (a ``{scenario_id: Scenario}`` mapping as
    returned by :func:`jobrec_eval.scenarios.load_scenarios`); an unmapped scenario
    is reported as ``unknown``.
    """
    by_variant: dict[str, dict[str, int]] = {}
    by_type: dict[tuple[str, str], dict[str, int]] = {}

    for b in bundles:
        scen = (scenarios or {}).get(b.scenario_id)
        scenario_type = getattr(scen, "scenario_type", None) or "unknown"
        is_hybrid = _run_mode(b) == "hybrid"
        provenance = _preference_provenance(b)

        tallies = [
            by_variant.setdefault(b.variant, _blank_tally()),
            by_type.setdefault((b.variant, scenario_type), _blank_tally()),
        ]
        for t in tallies:
            t["runs"] += 1
            if is_hybrid:
                t["hybrid_runs"] += 1
            for method, source in provenance:
                t["fields"] += 1
                if method in (_METHOD_RULE, _METHOD_LLM):
                    t[method] += 1
                if source in (_SOURCE_NORMALIZED, _SOURCE_REPAIRED,
                              _SOURCE_RULE_FALLBACK, _SOURCE_UNRESOLVED):
                    t[source] += 1
                if is_hybrid:
                    t["hybrid_fields"] += 1
                    if source is not None and source != _SOURCE_NORMALIZED:
                        t["hybrid_schema_failures"] += 1
                    if source == _SOURCE_RULE_FALLBACK:
                        t["hybrid_fallbacks"] += 1

    rows = [_tally_row(_SCOPE_VARIANT, variant, "(all)", t)
            for variant, t in sorted(by_variant.items())]
    rows += [_tally_row(_SCOPE_SCENARIO_TYPE, variant, stype, t)
             for (variant, stype), t in sorted(by_type.items())]
    return pd.DataFrame(rows, columns=_EXTRACTION_COLUMNS)


# ------------------------------------------------------ R14 retrieval-layer evaluation
# Requirement 14 asks for retrieval quality to be assessable independently of ranking
# quality: what the retriever recalled, how often it had to fall back to the whole
# catalog, how much of the relevant material it actually reached (Recall@pool), and how
# long it took -- with retrieval errors reported SEPARATELY from ranking errors (R14.2).
# The inputs are each bundle's ``retrieval_results.json`` (written by
# ``jobrec.evaluation.exporters``) and the relevance oracle in ``relevance.py``.

#: Handoff contract emitted by the retrieval stage; a failed one is a retrieval error.
_RETRIEVAL_CONTRACT = "RetrievalOutcome"
#: Component that produces the ranking; a failed handoff out of it is a ranking error.
_RANKING_COMPONENT = "ranking_agent"

_RETRIEVAL_COLUMNS = [
    "variant", "runs", "retrieval_runs",
    "mean_initial_pool_size", "mean_pool_size", "mean_retrieval_score",
    "full_catalog_fallbacks", "fallback_rate",
    "median_retrieval_latency_ms", "mean_retrieval_latency_ms",
    "scored_runs", "recall_at_pool", "relevant_job_coverage",
    "retrieval_errors", "retrieval_error_rate",
    "ranking_errors", "ranking_error_rate",
]


def _relevant_job_ids(grade: dict[tuple[str, str], int], scenario_id: str,
                      threshold: int) -> set[str]:
    """Job ids the oracle grades relevant (``grade >= threshold``) for a scenario."""
    return {jid for (sid, jid), g in grade.items() if sid == scenario_id and g >= threshold}


def _pool_job_ids(bundle) -> list[str]:
    """Job ids in the pool the retrieval layer handed downstream.

    Prefers ``pool_job_ids`` (the pool after the empty-recall full-catalog fallback)
    and falls back to ``retrieved_job_ids`` for bundles written before that field
    existed.
    """
    retrieval = getattr(bundle, "retrieval", None) or {}
    pool = retrieval.get("pool_job_ids")
    if pool is None:
        pool = retrieval.get("retrieved_job_ids")
    return list(pool or [])


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def retrieval_metrics(bundles, relevance_labels=None, relevance_threshold: int = 2,
                      ) -> pd.DataFrame:
    """Retrieval-layer metrics per variant, separated from ranking (R14.1/14.2).

    Reads each bundle's ``retrieval_results.json`` and reports, per variant:

    - ``mean_initial_pool_size`` — jobs the retriever matched before truncation to
      ``retrieval_pool_size``; ``mean_pool_size`` — the pool handed downstream.
    - ``mean_retrieval_score`` — mean retrieval score over all recalled jobs.
    - ``full_catalog_fallbacks`` / ``fallback_rate`` — how often an empty recall
      forced the fallback to the whole catalog, over runs that reached retrieval.
    - ``median_retrieval_latency_ms`` / ``mean_retrieval_latency_ms`` — the
      retrieval stage's own latency, never mixed with ranking latency.

    When ``relevance_labels`` (the oracle table from :mod:`jobrec_eval.relevance`) is
    supplied, two recall columns are added over the runs that have at least one
    relevant job (``scored_runs``):

    - ``recall_at_pool`` — mean over runs of ``|relevant ∩ pool| / |relevant|``.
    - ``relevant_job_coverage`` — share of those runs whose pool contained at least
      one relevant job.

    Errors are attributed to the layer that caused them (R14.2). A run that reached
    retrieval is a **retrieval error** when its retrieval handoff failed validation,
    or its pool is empty, or the scenario has relevant jobs and none of them reached
    the pool. It is a **ranking error** — never both — when retrieval did deliver a
    relevant job into the pool but no relevant job survived into
    ``selected_job_ids``, or when a handoff out of the ranking agent failed. Rates
    are over ``retrieval_runs``.

    Every rate/mean is ``None`` rather than 0.0/1.0 when its denominator is empty
    (no runs reached retrieval, no labels supplied, no relevant jobs), matching the
    convention of the other metrics in this module.
    """
    grade = grade_lookup(relevance_labels) if relevance_labels is not None else {}
    relevant_cache: dict[str, set[str]] = {}

    tallies: dict[str, dict] = {}
    for b in bundles:
        t = tallies.setdefault(b.variant, {
            "runs": 0, "retrieval_runs": 0, "fallbacks": 0,
            "initial_pool_sizes": [], "pool_sizes": [], "scores": [], "latencies": [],
            "recalls": [], "covered": 0, "scored_runs": 0,
            "retrieval_errors": 0, "ranking_errors": 0,
        })
        t["runs"] += 1

        retrieval = getattr(b, "retrieval", None) or {}
        if not retrieval.get("executed"):
            continue
        t["retrieval_runs"] += 1

        if retrieval.get("initial_pool_size") is not None:
            t["initial_pool_sizes"].append(float(retrieval["initial_pool_size"]))
        pool = _pool_job_ids(b)
        t["pool_sizes"].append(float(len(pool)))
        t["fallbacks"] += int(retrieval.get("full_catalog_fallback_count") or 0)
        for row in retrieval.get("initial_pool") or []:
            if row.get("score") is not None:
                t["scores"].append(float(row["score"]))
        if retrieval.get("retrieval_latency_ms") is not None:
            t["latencies"].append(float(retrieval["retrieval_latency_ms"]))

        if b.scenario_id not in relevant_cache:
            relevant_cache[b.scenario_id] = _relevant_job_ids(
                grade, b.scenario_id, relevance_threshold)
        relevant = relevant_cache[b.scenario_id]
        pool_set = set(pool)
        retrieved_relevant = relevant & pool_set
        if relevant:
            t["scored_runs"] += 1
            t["recalls"].append(len(retrieved_relevant) / len(relevant))
            if retrieved_relevant:
                t["covered"] += 1

        handoff_failed = any(
            h.get("contract_name") == _RETRIEVAL_CONTRACT and not h.get("validation_passed")
            for h in b.handoffs)
        if handoff_failed or not pool or (relevant and not retrieved_relevant):
            t["retrieval_errors"] += 1
            continue

        selected = set(((b.decision or {}).get("selected_job_ids")) or [])
        ranking_handoff_failed = any(
            h.get("from_component") == _RANKING_COMPONENT and not h.get("validation_passed")
            for h in b.handoffs)
        if ranking_handoff_failed or (retrieved_relevant and not (retrieved_relevant & selected)):
            t["ranking_errors"] += 1

    rows = []
    for variant, t in sorted(tallies.items()):
        n_retrieval = t["retrieval_runs"]
        rows.append({
            "variant": variant,
            "runs": t["runs"],
            "retrieval_runs": n_retrieval,
            "mean_initial_pool_size": _mean(t["initial_pool_sizes"]),
            "mean_pool_size": _mean(t["pool_sizes"]),
            "mean_retrieval_score": _mean(t["scores"]),
            "full_catalog_fallbacks": t["fallbacks"],
            "fallback_rate": (t["fallbacks"] / n_retrieval) if n_retrieval else None,
            "median_retrieval_latency_ms": _median(t["latencies"]),
            "mean_retrieval_latency_ms": _mean(t["latencies"]),
            "scored_runs": t["scored_runs"],
            "recall_at_pool": _mean(t["recalls"]),
            "relevant_job_coverage": (t["covered"] / t["scored_runs"])
            if t["scored_runs"] else None,
            "retrieval_errors": t["retrieval_errors"],
            "retrieval_error_rate": (t["retrieval_errors"] / n_retrieval)
            if n_retrieval else None,
            "ranking_errors": t["ranking_errors"],
            "ranking_error_rate": (t["ranking_errors"] / n_retrieval) if n_retrieval else None,
        })
    return pd.DataFrame(rows, columns=_RETRIEVAL_COLUMNS)


# --------------------------------------------- R25 top-k score-breakdown contribution
# Requirement 25 has two halves. R25.1 is persistence: every ranked job's per-feature
# breakdown (``RankedJob.features`` -> name / normalized_score / weight /
# weighted_contribution / explanation_code) is written into each bundle's
# ``recommendation_decision.json`` by ``jobrec.evaluation.exporters.write_run_bundle``,
# so nothing has to be recomputed here. R25.2 is this table: it reads those persisted
# breakdowns back and reports, for the top-k recommended jobs, how much each ranking
# feature actually contributed to the score -- i.e. why a job was ranked where it was.

_SCOPE_TOPK_VARIANT = "variant"
_SCOPE_TOPK_RANK = "variant_rank"

_TOPK_CONTRIBUTION_COLUMNS = [
    "scope", "variant", "rank", "feature", "jobs",
    "mean_total_score", "mean_normalized_score", "mean_weight", "mean_contribution",
    "contribution_share", "top_driver_jobs", "top_driver_share",
    "inactive_jobs", "dominant_explanation_code",
]


def _topk_ranked_jobs(bundle, top_k: int | None = None) -> list[tuple[int, dict]]:
    """``(rank, ranked_job)`` for the jobs a run actually recommended, in rank order.

    The recommended set is ``selected_job_ids``, which the orchestrator fills with
    the first ``top_k`` entries of the ranked list, so it already IS the top-k in
    rank order. ``rank`` is taken from the persisted ``RankedJob.rank`` and falls
    back to the 1-based position when a bundle predates that field. A run that
    recommended nothing (no-match or a clarification short-circuit) yields no rows.
    """
    decision = bundle.decision or {}
    ranked_by_id = {rj.get("job_id"): rj for rj in decision.get("ranked_jobs") or []}
    out: list[tuple[int, dict]] = []
    for position, job_id in enumerate(decision.get("selected_job_ids") or [], start=1):
        ranked_job = ranked_by_id.get(job_id)
        if ranked_job is None:
            continue
        rank = ranked_job.get("rank")
        out.append((int(rank) if rank else position, ranked_job))
        if top_k is not None and len(out) >= top_k:
            break
    return out


def _blank_contribution_tally() -> dict:
    return {
        "jobs": 0, "total_score": 0.0, "normalized": 0.0, "weight": 0.0,
        "contribution": 0.0, "top_driver": 0, "inactive": 0,
        "codes": Counter(),
    }


def _top_driver_feature(features: list[dict]) -> str | None:
    """The feature that contributed most to one job's score, or ``None``.

    Ties are broken by feature name so exactly one feature is ever credited per
    job, and a job whose contributions are all zero or negative credits nobody --
    there is no dominant reason to report.
    """
    best_name, best_value = None, 0.0
    for feature in features:
        value = float(feature.get("weighted_contribution") or 0.0)
        name = feature.get("name")
        if name is None or value <= 0.0:
            continue
        if value > best_value or (value == best_value and name < (best_name or "")):
            best_name, best_value = name, value
    return best_name


def _contribution_row(scope: str, variant: str, rank: int | None, feature: str,
                      t: dict) -> dict:
    """Render one contribution tally as a report row, ``None`` for undefined rates."""
    jobs = t["jobs"]
    score_sum = t["total_score"]
    codes = t["codes"]
    return {
        "scope": scope,
        "variant": variant,
        "rank": rank,
        "feature": feature,
        "jobs": jobs,
        "mean_total_score": (score_sum / jobs) if jobs else None,
        "mean_normalized_score": (t["normalized"] / jobs) if jobs else None,
        "mean_weight": (t["weight"] / jobs) if jobs else None,
        "mean_contribution": (t["contribution"] / jobs) if jobs else None,
        "contribution_share": (t["contribution"] / score_sum) if score_sum else None,
        "top_driver_jobs": t["top_driver"],
        "top_driver_share": (t["top_driver"] / jobs) if jobs else None,
        "inactive_jobs": t["inactive"],
        "dominant_explanation_code": (
            min(codes.items(), key=lambda kv: (-kv[1], kv[0]))[0] if codes else None),
    }


def topk_contribution_table(bundles, top_k: int | None = None) -> pd.DataFrame:
    """Per-feature contribution table over the top-k recommended jobs (R25.2).

    Reads the score breakdown persisted with every ranked job in each bundle's
    ``recommendation_decision.json`` (R25.1) -- no scores are recomputed -- and
    aggregates one row per ranking feature:

    - ``mean_total_score`` — mean ``RankedJob.total_score`` of the jobs in the group
      (identical across the features of a group, so each feature's contribution can
      be read against the score it fed).
    - ``mean_normalized_score`` / ``mean_weight`` / ``mean_contribution`` — the mean
      of the feature's persisted ``normalized_score``, its effective ``weight``
      (already renormalized over applicable features by the ranking agent) and its
      ``weighted_contribution``.
    - ``contribution_share`` — the feature's summed contribution over the summed
      total score of the group: the fraction of the top-k score the feature is
      responsible for. This is the column the paper table is built on.
    - ``top_driver_jobs`` / ``top_driver_share`` — how often the feature was the
      single largest contributor to a job's score, i.e. the dominant reason that job
      ranked where it did. Ties are broken by feature name so exactly one feature is
      credited per job; a job with no positive contribution credits none.
    - ``inactive_jobs`` — jobs where the feature contributed exactly 0 (not
      applicable, or applicable but scored 0), which is how an unstated preference
      shows up.
    - ``dominant_explanation_code`` — the feature's most frequent
      ``explanation_code`` (ties by code name), naming the rule behind the number.

    Rows come in two scopes, tagged in the ``scope`` column so totals are never
    double-counted:

    - ``variant`` — one row per variant x feature over all recommended jobs, with
      ``rank`` left ``None``.
    - ``variant_rank`` — one row per variant x rank x feature, which is what shows
      how the reasons differ between the top recommendation and the k-th one.

    ``top_k`` optionally truncates further; by default the whole recommended set of
    each run is used, which the orchestrator already caps at
    ``ExperimentConfig.top_k``. Runs that recommended nothing (no-match,
    clarification) contribute no rows, and every mean/share is ``None`` rather than
    0.0 when its denominator is empty, matching the convention of the other metrics
    in this module.
    """
    by_variant: dict[tuple[str, str], dict] = {}
    by_rank: dict[tuple[str, int, str], dict] = {}

    for b in bundles:
        for rank, ranked_job in _topk_ranked_jobs(b, top_k):
            features = list(ranked_job.get("features") or [])
            total_score = float(ranked_job.get("total_score") or 0.0)
            driver = _top_driver_feature(features)
            for feature in features:
                name = feature.get("name")
                if name is None:
                    continue
                contribution = float(feature.get("weighted_contribution") or 0.0)
                tallies = [
                    by_variant.setdefault((b.variant, name), _blank_contribution_tally()),
                    by_rank.setdefault((b.variant, rank, name), _blank_contribution_tally()),
                ]
                for t in tallies:
                    t["jobs"] += 1
                    t["total_score"] += total_score
                    t["normalized"] += float(feature.get("normalized_score") or 0.0)
                    t["weight"] += float(feature.get("weight") or 0.0)
                    t["contribution"] += contribution
                    if contribution == 0.0:
                        t["inactive"] += 1
                    if name == driver:
                        t["top_driver"] += 1
                    code = feature.get("explanation_code")
                    if code is not None:
                        t["codes"][code] += 1

    rows = [_contribution_row(_SCOPE_TOPK_VARIANT, variant, None, feature, t)
            for (variant, feature), t in sorted(by_variant.items())]
    rows += [_contribution_row(_SCOPE_TOPK_RANK, variant, rank, feature, t)
             for (variant, rank, feature), t in sorted(by_rank.items())]
    return pd.DataFrame(rows, columns=_TOPK_CONTRIBUTION_COLUMNS)
