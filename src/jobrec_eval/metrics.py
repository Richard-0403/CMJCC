"""Metric computations (RQ4).

Design choices per the evaluation guide:
- Analysis unit is the scenario; run-level metrics aggregate to
  scenario x variant, then to variant.
- HCSR / violations are recomputed against the *authoritative* hard constraints
  (the full variant's JobContextState for that scenario) so an ablation that
  skips filtering (no_context) is scored against the true constraints, not its
  own pass-through eligibility.
- Explicit policies: correct no-match => ranking metrics N/A (not 0); empty
  recommendation => HCSR N/A; unknown hard checks are NOT counted as pass.
- Grounding uses the system's claim validator output (supported/total factual).
- Clarification-dependent scenarios are scored over the WHOLE dialogue (the
  per-turn ``dialogue_trace.jsonl``), not the final response: asking a necessary
  clarification, having it answered and then recommending is the correct
  behaviour, so that is what task success rewards (R7.3, R7.4, R7.8).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.config import AppConfig
from jobrec.domain.constraints import JobContextState
from jobrec.domain.enums import ConstraintOutcome, ConstraintStrength
from jobrec.domain.job import JobPosting

from .metrics_extra import clarification_efficiency_per_run
from .relevance import grade_lookup, ideal_grades
from .scenarios import Scenario


# --------------------------------------------------------------- ranking maths
def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ranking_is_defined(all_grades_desc: list[int], k: int = 5) -> bool:
    """Whether a ranking metric is DEFINED for this scenario's label universe.

    It is not when the oracle grades nothing in the whole catalog relevant to the
    scenario: there is no attainable ideal ranking, so a system cannot rank well or
    badly. :func:`ndcg_at_k` has always returned ``None`` in that case, but
    :func:`precision_at_k` and :func:`mean_graded_relevance` used to return 0 -- which
    is not a measurement, it is arithmetic on an empty target. The three metrics were
    therefore averaged over DIFFERENT sets of scenarios while being presented side by
    side as if they described the same runs. This predicate is the one definition all
    three now share.
    """
    return dcg(sorted(all_grades_desc, reverse=True)[:k]) > 0


def ndcg_at_k(ranked_grades: list[int], all_grades_desc: list[int], k: int = 5) -> float | None:
    if not ranking_is_defined(all_grades_desc, k):
        return None  # no relevant items exist -> N/A (do not silently set 0)
    return dcg(ranked_grades[:k]) / dcg(sorted(all_grades_desc, reverse=True)[:k])


def precision_at_k(ranked_grades: list[int], k: int, threshold: int, returned: int) -> float | None:
    """Precision@k over the returned list.

    Callers must gate this on :func:`ranking_is_defined`: nothing in the arguments can
    tell whether a 0 means "ranked badly" or "there was nothing to rank".
    """
    if returned == 0:
        return None
    denom = min(returned, k)
    rel = sum(1 for g in ranked_grades[:k] if g >= threshold)
    return rel / denom


def mean_graded_relevance(ranked_grades: list[int]) -> float | None:
    """Mean graded relevance of the returned list. Gate as for :func:`precision_at_k`."""
    if not ranked_grades:
        return None
    return float(np.mean(ranked_grades))


# ------------------------------------------------------ dialogue trace (R7.3/R7.8)
#: ``system_action`` values recorded per turn by
#: :func:`jobrec.evaluation.exporters.trace_record` (the turn's response type).
_ACTION_CLARIFICATION = "clarification"
_ACTION_RECOMMENDATION = "recommendation"
_ACTION_NO_MATCH = "no_match"

#: Termination reason the clarification loop records when it stopped because a slot
#: would have been re-asked (``experiment_runner._run_clarification_loop``).
_REASON_REPEATED_SLOT = "repeated_slot"

#: A slot legitimately occupies at most two trace records: the turn on which the system
#: ASKED it, and the turn whose simulated answer the runner records against it. More
#: occurrences than that mean the slot was asked again.
_MAX_SLOT_RECORDS = 2


@dataclass(frozen=True)
class DialogueView:
    """Dialogue-level facts about one run, derived from its per-turn trace (R7.3/R7.8)."""

    #: The trace carries dialogue-level evidence (see :func:`dialogue_view`). When
    #: False, callers fall back to scoring the final response alone.
    scored: bool
    #: The system asked at least one clarification somewhere in the dialogue.
    asked: bool
    #: Every slot the dialogue asked about, in first-asked order.
    asked_slots: tuple[str, ...]
    #: The user answered at least one ask (the dialogue continued past it).
    answered: bool
    #: A slot was asked again (or the repeated-slot guard fired).
    repeated_slot: bool
    #: Number of processed turns (one trace record per turn).
    turns: int
    #: Terminal outcome recorded on the final trace record.
    termination_reason: str | None


def dialogue_view(bundle) -> DialogueView:
    """Summarize a run's whole dialogue from ``dialogue_trace.jsonl`` (R7.3/R7.8).

    The trace holds one record per processed turn
    (:func:`jobrec.evaluation.exporters.trace_record`): ``system_action`` is that turn's
    response type and ``clarification_slot`` is the slot the turn is about -- the slot the
    SYSTEM asked on a scripted turn, or the slot the simulated user ANSWERED on a loop
    turn. Either way the recorded slot is one the system asked at some point, so the union
    of the recorded slots plus the still-pending ask in ``clarification.json`` is the set
    of slots the dialogue asked about.

    ``scored`` reports whether the trace says anything about the dialogue: a multi-record
    trace, or a single record stamped with the loop's termination reason. A lone record
    with no termination reason is the fallback trace ``write_run_bundle`` derives from a
    final turn result (non-loop callers, older bundles); it carries no dialogue-level
    evidence, so callers score those runs by the final response instead.

    A repeat is recorded either by the loop's ``repeated_slot`` termination reason or by a
    slot occupying more than :data:`_MAX_SLOT_RECORDS` trace records.
    """
    trace = list(getattr(bundle, "dialogue_trace", None) or [])
    pending = [str(f) for f in ((bundle.clarification or {}).get("target_fields") or [])]

    slots: list[str] = []
    asked = answered = False
    for index, record in enumerate(trace):
        if str(record.get("system_action") or "") == _ACTION_CLARIFICATION:
            asked = True
            # The dialogue continued past the ask, so the user supplied an answer.
            answered = answered or index < len(trace) - 1
        slot = record.get("clarification_slot")
        if slot:
            slots.append(str(slot))

    ordered: list[str] = []
    for slot in [*slots, *pending]:
        if slot not in ordered:
            ordered.append(slot)

    termination = bundle.termination_reason
    counts = Counter(slots)
    return DialogueView(
        scored=bool(trace) and (len(trace) > 1 or termination is not None),
        asked=asked or bool(pending),
        asked_slots=tuple(ordered),
        answered=answered,
        repeated_slot=(termination == _REASON_REPEATED_SLOT
                       or any(n > _MAX_SLOT_RECORDS for n in counts.values())),
        turns=bundle.response_turns,
        termination_reason=termination,
    )


# ----------------------------------------------- fault injection / recovery (R10.8)
#: Provider-manifest block that records an injected fault (written by the
#: fault-injecting provider and mirrored onto ``RunRecord.model_manifest``). It is the
#: only place the pipeline records that a fault was injected at all.
_FAULT_INJECTION_KEY = "fault_injection"
#: Status a handoff / evidence-log record carries when a fault was absorbed by a retry.
_RECOVERED_STATUS = "recovered"
#: Statuses that record a caught fault rather than a clean step.
_FAILED_HANDOFF_STATUS = "failed"
_FAILED_LOG_STATUS = "failure"


def failure_flags(bundle) -> dict[str, bool]:
    """Fault-injection / recovery booleans for one run (R10.8).

    Read strictly from what the run artifacts RECORD; nothing is inferred about faults
    that were never instrumented:

    - ``failure_injected`` — the run's provider manifest carries a ``fault_injection``
      block. The main deterministic experiment injects nothing, so the column is False
      for every one of its runs and :func:`metrics_extra.failure_detection_rate` reports
      N/A instead of a misleading 1.000.
    - ``failure_detected`` — the run recorded a caught fault: a ``failure_code`` on the
      run record, a handoff that failed validation or ended ``failed``, or an
      evidence-log record with a ``failure`` / ``recovered`` status or an error code.
      Reported independently of injection; the detection rate intersects the two, so a
      detection without an injected fault is never counted as one.
    - ``recoverable`` — a recovery was recorded at all: a handoff or evidence-log record
      with ``recovered`` status, which is what the retry path writes when it absorbs a
      fault.
    - ``recovered`` — that recovery carried the run through: a recovery marker plus a run
      record flagged successful.
    """
    run_record = bundle.run_record or {}
    manifest = run_record.get("model_manifest") or {}
    handoffs = bundle.handoffs or []
    logs = bundle.evidence_log or []

    detected = bool(run_record.get("failure_code"))
    detected = detected or any(
        (not h.get("validation_passed")) or h.get("status") == _FAILED_HANDOFF_STATUS
        for h in handoffs)
    detected = detected or any(
        log.get("error_code") or log.get("status") in {_FAILED_LOG_STATUS, _RECOVERED_STATUS}
        for log in logs)

    recoverable = any(h.get("status") == _RECOVERED_STATUS for h in handoffs) or \
        any(log.get("status") == _RECOVERED_STATUS for log in logs)

    return {
        "failure_injected": bool(manifest.get(_FAULT_INJECTION_KEY)),
        "failure_detected": detected,
        "recoverable": recoverable,
        "recovered": recoverable and bool(run_record.get("success")),
    }


class MetricsComputer:
    """Per-run metrics for one label universe.

    ``relevance_labels`` is a relevance label table in the shape
    :func:`~jobrec_eval.relevance.grade_lookup` / :func:`~jobrec_eval.relevance.ideal_grades`
    consume (``scenario_id``, ``job_id``, ``relevance_grade``). It is deliberately
    source-agnostic: the automatic oracle table from
    :func:`~jobrec_eval.relevance.grade_catalog` and an adjudicated human table from
    :func:`~jobrec_eval.annotation.load_adjudicated_relevance_labels` are interchangeable
    here, and every grade-derived metric (NDCG@5, Precision@5, mean graded relevance) is
    computed from whichever table it was given. Which one that is belongs to the pipeline
    wiring (``cli.run_pipeline``), not to this class, so there is exactly one metric
    implementation for both sources.
    """

    def __init__(
        self,
        config: AppConfig,
        catalog: list[JobPosting],
        references: dict[str, dict],
        relevance_labels: pd.DataFrame,
        scenarios: dict[str, Scenario],
        relevance_threshold: int = 2,
        top_k: int = 5,
    ) -> None:
        self.config = config
        self.jobs_by_id = {j.job_id: j for j in catalog}
        self.references = references
        self.labels = relevance_labels
        self.grade = grade_lookup(relevance_labels)
        self.scenarios = scenarios
        self.threshold = relevance_threshold
        self.top_k = top_k
        self.agent = JobContextAgent(config)
        self._ideal_cache: dict[str, list[int]] = {}
        self._ctx_cache: dict[str, JobContextState] = {}

    def _ideal(self, scenario_id: str) -> list[int]:
        if scenario_id not in self._ideal_cache:
            self._ideal_cache[scenario_id] = ideal_grades(self.labels, scenario_id)
        return self._ideal_cache[scenario_id]

    def _reference_context(self, scenario_id: str) -> JobContextState | None:
        if scenario_id in self._ctx_cache:
            return self._ctx_cache[scenario_id]
        ref = self.references.get(scenario_id)
        ctx = JobContextState.model_validate(ref["job_context"]) if ref and ref.get("job_context") else None
        self._ctx_cache[scenario_id] = ctx
        return ctx

    # ---------------------------------------------------------------- per run
    def run_metrics(self, bundles) -> pd.DataFrame:
        rows = []
        for b in bundles:
            rows.append(self._one(b))
        frame = pd.DataFrame(rows)
        # Per-run clarification efficiency, from the single definition shared with
        # ``metrics_extra.clarification_efficiency`` so the per-run column and the
        # per-variant score can never drift apart (R7.4/R7.5).
        frame["clarification_efficiency"] = clarification_efficiency_per_run(frame)
        return frame

    def _one(self, b) -> dict:
        scen = self.scenarios.get(b.scenario_id)
        d = b.decision or {}
        selected = d.get("selected_job_ids", [])
        response_type = (b.response or {}).get("response_type")
        grades = [self.grade.get((b.scenario_id, jid), 0) for jid in selected]
        returned = len(selected)

        # ranking metrics (N/A for correct no-match / no returns, and N/A when the
        # scenario has no relevant item in the whole label universe -- see
        # ``ranking_is_defined``. All THREE share that gate: previously only NDCG did,
        # so the three columns silently had different denominators.)
        if returned == 0 or not ranking_is_defined(self._ideal(b.scenario_id), self.top_k):
            ndcg = prec = mgr = None
        else:
            ndcg = ndcg_at_k(grades, self._ideal(b.scenario_id), self.top_k)
            prec = precision_at_k(grades, self.top_k, self.threshold, returned)
            mgr = mean_graded_relevance(grades)

        # HCSR / violations against TRUE constraints
        ctx = self._reference_context(b.scenario_id)
        hcsr = mvc = unknown_hard_rate = expired_rate = None
        trace_complete = None
        if returned and ctx is not None:
            eligible_flags, violations, unknown_hard, applicable_hard, expired = [], [], 0, 0, 0
            ranked = {r["job_id"]: r for r in d.get("ranked_jobs", [])}
            traced = 0
            for jid in selected:
                job = self.jobs_by_id.get(jid)
                if job is None:
                    continue
                res = self.agent.evaluate(job, ctx)
                eligible_flags.append(1 if res.eligible else 0)
                violations.append(res.hard_violation_count)
                if not job.is_active or (job.application_deadline is not None
                                         and str(job.application_deadline) < self.config.project.reference_date):
                    expired += 1
                for c in res.checks:
                    strength = next((cc.strength for cc in ctx.constraints
                                     if cc.constraint_id == c.constraint_id), None)
                    if strength == ConstraintStrength.HARD:
                        applicable_hard += 1
                        if c.outcome == ConstraintOutcome.UNKNOWN:
                            unknown_hard += 1
                rj = ranked.get(jid, {})
                if rj.get("features") and rj.get("eligibility_result_id") and \
                        any(f.get("evidence_ids") for f in rj.get("features", [])):
                    traced += 1
            hcsr = float(np.mean(eligible_flags)) if eligible_flags else None
            mvc = float(np.mean(violations)) if violations else None
            # N/A, NOT 0.0, when no hard constraint was applicable: with an empty
            # denominator there is no share to report, and 0.0 reads as "nothing was
            # unknown" -- the strongest possible claim -- on the basis of no observation
            # at all. It also drags the variant mean towards 0 for free. Every other rate
            # in this method already follows the None-on-empty-denominator convention.
            unknown_hard_rate = (unknown_hard / applicable_hard) if applicable_hard else None
            expired_rate = expired / returned
            trace_complete = traced / returned

        # grounding (from claim validator output)
        factual = [c for c in b.claims if c.get("claim_type") != "non_factual"]
        supported = [c for c in factual if c.get("support_status") == "supported"]
        grounding = (len(supported) / len(factual)) if factual else None
        grounded_claim_count = len(supported)

        # handoffs
        att = len(b.handoffs)
        valid = sum(1 for h in b.handoffs if h.get("validation_passed") and h.get("status") == "completed")
        handoff_success = (valid / att) if att else None

        # Decision-log completeness: every logged stage succeeded. An EMPTY log counts as
        # complete only for an error response, where there was nothing to log.
        #
        # Three dead lines used to sit here: a `core_stages.issubset(logged | core_stages)`
        # test that is true for any input by construction, and the two locals feeding it.
        # Its result was overwritten on the next line, so it never reached a number -- it
        # only made the metric look as though stage coverage was being checked.
        all_stages_ok = (bool(b.evidence_log)
                         and all(log.get("status") == "success" for log in b.evidence_log))
        dlc = 1.0 if (all_stages_ok
                      or (not b.evidence_log and response_type == "error")) else 0.0

        turn_count = len([t for t in ((b.dialogue_state or {}).get("turns", []))
                          if t.get("speaker") == "candidate"]) or None

        dialogue = dialogue_view(b)
        task = self._task_success(scen, response_type, returned, hcsr, grounded_claim_count,
                                  d.get("no_match_reason_codes", []), b.clarification, dialogue)
        partial = self._partial(scen, response_type, returned, hcsr, grounded_claim_count, d,
                                b.clarification, dialogue)

        return {
            "run_id": b.run_id, "scenario_id": b.scenario_id, "variant": b.variant,
            "repeat_index": b.run_index,
            "scenario_type": scen.scenario_type if scen else None,
            "difficulty": scen.difficulty if scen else None,
            "memory_dependency": scen.memory_dependency if scen else None,
            "context_dependency": scen.context_dependency if scen else None,
            "response_type": response_type,
            "success_run": bool(b.run_record.get("success")),
            "returned_count": returned,
            "ndcg_at_5": ndcg, "precision_at_5": prec, "mean_graded_relevance": mgr,
            "hcsr": hcsr, "mean_violation_count": mvc, "unknown_hard_rate": unknown_hard_rate,
            "expired_rate": expired_rate, "trace_completeness": trace_complete,
            "grounding": grounding, "grounded_claim_count": grounded_claim_count,
            "handoff_success": handoff_success, "decision_log_completeness": dlc,
            "turn_count": turn_count,
            "total_latency_ms": b.run_record.get("total_latency_ms"),
            "task_success": task, "partial_task_score": partial,
            "no_match_expected": scen.no_match_expected if scen else False,
            "no_match_returned": bool(d.get("no_match", False)),
            "clarification_expected": scen.clarification_expected if scen else False,
            "clarification_target": ";".join((b.clarification or {}).get("target_fields", []) if b.clarification else []),
            # Dialogue-level clarification view (R7.3/R7.8). ``clarification_target`` above
            # stays the FINAL pending ask for backward compatibility; the columns below
            # describe the whole dialogue.
            "clarification_asked": dialogue.asked,
            "clarification_asked_slots": ";".join(dialogue.asked_slots),
            "clarification_answered": dialogue.answered,
            "clarification_repeated_slot": dialogue.repeated_slot,
            "clarification_reason_code": (b.clarification or {}).get("reason_code"),
            # ``None`` rather than 0 when the bundle carries no trace, so turn-based
            # metrics fall back to ``turn_count`` instead of reading a phantom zero.
            "response_turns": dialogue.turns or None,
            "termination_reason": dialogue.termination_reason,
            "acceptable_slots": ";".join(scen.acceptable_slots) if scen else "",
            **failure_flags(b),
        }

    # ------------------------------------------------------------ success rules
    @staticmethod
    def _slots_ok(scen, slots) -> bool:
        """Did the run ask about an acceptable slot? (R7.4)

        An empty ``acceptable_slots`` set means the reference does not constrain the
        target, so any asked slot counts.
        """
        if not scen.acceptable_slots:
            return True
        return any(slot in scen.acceptable_slots for slot in slots)

    @staticmethod
    def _recommendation_ok(response_type, returned, hcsr, grounded) -> bool:
        """The recommendation quality bar: returns, no hard violation, grounded rationale.

        HCSR is recomputed against the authoritative constraints, so ``None`` (no
        reference context) never passes; grounding requires at least one supported
        factual claim.
        """
        return bool(response_type == _ACTION_RECOMMENDATION and returned > 0
                    and hcsr is not None and hcsr >= 1.0 and grounded > 0)

    @staticmethod
    def _no_match_ok(response_type, no_match_codes) -> bool:
        """A correct no-match: the no-match response plus at least one reason code."""
        return bool(response_type == _ACTION_NO_MATCH and len(no_match_codes) > 0)

    def _task_success(self, scen, response_type, returned, hcsr, grounded, no_match_codes,
                      clar, dialogue: DialogueView | None = None) -> int:
        """Binary task success for one run.

        Clarification-dependent scenarios are scored over the WHOLE dialogue (R7.4): the
        run succeeds only when the system asked a necessary clarification, the target was
        an acceptable slot, the simulated user answered it, and the dialogue then reached
        a correct terminal outcome that still clears the usual quality bars (HCSR 1.0 and
        a grounded rationale for a recommendation; reason codes for a no-match). A run
        that skipped the necessary question and guessed straight to a recommendation, or
        one still sitting on a clarification when a guard fired (``max_turns``,
        ``cannot_answer``, ``repeated_slot``), is never a success.

        Bundles whose trace carries no dialogue-level evidence (see :func:`dialogue_view`)
        keep the previous final-response rule, so older bundles and non-loop runs are
        unaffected. Recommendation- and no-match-expected scenarios are unchanged.
        """
        if scen is None:
            return 0
        if scen.clarification_expected:
            if dialogue is None or not dialogue.scored:
                # No dialogue evidence: score the final response, as before.
                fields = (clar or {}).get("target_fields", []) if clar else []
                return 1 if (response_type == _ACTION_CLARIFICATION
                             and self._slots_ok(scen, fields)) else 0
            if not (dialogue.asked and self._slots_ok(scen, dialogue.asked_slots)
                    and dialogue.answered):
                return 0
            terminal_ok = (self._no_match_ok(response_type, no_match_codes)
                           if scen.no_match_expected
                           else self._recommendation_ok(response_type, returned, hcsr, grounded))
            return 1 if terminal_ok else 0
        if scen.no_match_expected:
            return 1 if self._no_match_ok(response_type, no_match_codes) else 0
        # recommendation expected
        return 1 if self._recommendation_ok(response_type, returned, hcsr, grounded) else 0

    def _partial(self, scen, response_type, returned, hcsr, grounded, d, clar,
                 dialogue: DialogueView | None = None) -> float:
        """Partial credit over four components, on the same dialogue-level view as
        :meth:`_task_success` (R7.4).

        For a clarification-dependent run the components are: the necessary question was
        asked on an acceptable slot, the answer was consumed and the dialogue moved on,
        the terminal outcome is the expected one and clears the hard-constraint bar, and
        the terminal outcome is grounded. Runs without dialogue evidence keep the previous
        final-response components.
        """
        if scen is None:
            return 0.0
        slot = correctness = grounding_ok = rec_ok = 0
        no_match_codes = d.get("no_match_reason_codes", [])
        if scen.clarification_expected:
            if dialogue is None or not dialogue.scored:
                asked_ok = 1 if response_type == _ACTION_CLARIFICATION else 0
                slot = rec_ok = correctness = asked_ok
                grounding_ok = 1
            else:
                slot = 1 if (dialogue.asked
                             and self._slots_ok(scen, dialogue.asked_slots)) else 0
                correctness = 1 if (slot and dialogue.answered) else 0
                if scen.no_match_expected:
                    rec_ok = 1 if self._no_match_ok(response_type, no_match_codes) else 0
                    grounding_ok = rec_ok
                else:
                    rec_ok = 1 if (response_type == _ACTION_RECOMMENDATION and returned > 0
                                   and hcsr is not None and hcsr >= 1.0) else 0
                    grounding_ok = 1 if grounded > 0 else 0
        elif scen.no_match_expected:
            slot = 1
            rec_ok = 1 if response_type == _ACTION_NO_MATCH else 0
            correctness = 1 if self._no_match_ok(response_type, no_match_codes) else 0
            grounding_ok = 1 if grounded > 0 or response_type == _ACTION_NO_MATCH else 0
        else:
            slot = 1 if response_type == _ACTION_RECOMMENDATION else 0
            rec_ok = 1 if (response_type == _ACTION_RECOMMENDATION and returned > 0) else 0
            correctness = 1 if (hcsr is not None and hcsr >= 1.0) else 0
            grounding_ok = 1 if grounded > 0 else 0
        return (slot + correctness + rec_ok + grounding_ok) / 4.0


# --------------------------------------------------------------- aggregation
def aggregate_scenario_variant(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeats within each scenario x variant (paired analysis unit)."""
    metric_cols = ["ndcg_at_5", "precision_at_5", "mean_graded_relevance", "hcsr",
                   "mean_violation_count", "unknown_hard_rate", "expired_rate",
                   "trace_completeness", "grounding", "handoff_success",
                   "decision_log_completeness", "turn_count", "response_turns",
                   "total_latency_ms", "task_success", "partial_task_score",
                   "grounded_claim_count", "clarification_efficiency"]
    agg = (run_metrics.groupby(["scenario_id", "variant", "scenario_type", "difficulty",
                                "memory_dependency", "context_dependency"], dropna=False)[metric_cols]
           .mean().reset_index())
    return agg


def variant_summary(scenario_variant: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["ndcg_at_5", "precision_at_5", "mean_graded_relevance", "hcsr",
                   "mean_violation_count", "unknown_hard_rate", "trace_completeness",
                   "grounding", "handoff_success", "decision_log_completeness",
                   "turn_count", "response_turns", "total_latency_ms", "task_success",
                   "clarification_efficiency"]
    rows = []
    for variant, sub in scenario_variant.groupby("variant"):
        row = {"variant": variant, "n_scenarios": sub["scenario_id"].nunique()}
        for m in metric_cols:
            vals = sub[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else None
            row[f"{m}_median"] = float(vals.median()) if len(vals) else None
            row[f"{m}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def latency_percentiles(component_latency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, comp), sub in component_latency.groupby(["variant", "component"]):
        vals = sub["latency_ms"].dropna()
        if not len(vals):
            continue
        rows.append({
            "variant": variant, "component": comp,
            "median_ms": float(np.median(vals)), "p95_ms": float(np.percentile(vals, 95)),
            "mean_ms": float(np.mean(vals)), "n": int(len(vals)),
        })
    return pd.DataFrame(rows)
