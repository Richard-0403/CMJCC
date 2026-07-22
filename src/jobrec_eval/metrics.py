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
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.config import AppConfig
from jobrec.domain.constraints import JobContextState
from jobrec.domain.enums import ConstraintOutcome, ConstraintStrength
from jobrec.domain.job import JobPosting

from .relevance import grade_lookup, ideal_grades
from .scenarios import Scenario


# --------------------------------------------------------------- ranking maths
def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades: list[int], all_grades_desc: list[int], k: int = 5) -> float | None:
    ideal = dcg(sorted(all_grades_desc, reverse=True)[:k])
    if ideal == 0:
        return None  # no relevant items exist -> N/A (do not silently set 0)
    return dcg(ranked_grades[:k]) / ideal


def precision_at_k(ranked_grades: list[int], k: int, threshold: int, returned: int) -> float | None:
    if returned == 0:
        return None
    denom = min(returned, k)
    rel = sum(1 for g in ranked_grades[:k] if g >= threshold)
    return rel / denom


def mean_graded_relevance(ranked_grades: list[int]) -> float | None:
    if not ranked_grades:
        return None
    return float(np.mean(ranked_grades))


class MetricsComputer:
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
        return pd.DataFrame(rows)

    def _one(self, b) -> dict:
        scen = self.scenarios.get(b.scenario_id)
        d = b.decision or {}
        selected = d.get("selected_job_ids", [])
        response_type = (b.response or {}).get("response_type")
        grades = [self.grade.get((b.scenario_id, jid), 0) for jid in selected]
        returned = len(selected)

        # ranking metrics (N/A for correct no-match / no returns)
        if returned == 0:
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
            unknown_hard_rate = (unknown_hard / applicable_hard) if applicable_hard else 0.0
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

        # decision-log completeness (successful runs should log all core stages)
        core_stages = {"understanding", "context_built"}
        logged_stages = {log.get("stage") for log in b.evidence_log}
        dlc = 1.0 if core_stages.issubset(logged_stages | {"understanding", "context_built"}) else 0.0
        dlc = len(b.evidence_log) > 0 and all(log.get("status") == "success" for log in b.evidence_log)
        dlc = 1.0 if dlc else (1.0 if not b.evidence_log and response_type == "error" else 0.0)

        turn_count = len([t for t in ((b.dialogue_state or {}).get("turns", []))
                          if t.get("speaker") == "candidate"]) or None

        task = self._task_success(scen, response_type, returned, hcsr, grounded_claim_count,
                                  d.get("no_match_reason_codes", []), b.clarification)
        partial = self._partial(scen, response_type, returned, hcsr, grounded_claim_count, d, b.clarification)

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
        }

    def _task_success(self, scen, response_type, returned, hcsr, grounded, no_match_codes, clar) -> int:
        if scen is None:
            return 0
        if scen.no_match_expected:
            return 1 if (response_type == "no_match" and len(no_match_codes) > 0) else 0
        if scen.clarification_expected:
            if response_type != "clarification":
                return 0
            fields = (clar or {}).get("target_fields", []) if clar else []
            if not scen.acceptable_slots:
                return 1
            return 1 if any(f in scen.acceptable_slots for f in fields) else 0
        # recommendation expected
        if response_type != "recommendation" or returned == 0:
            return 0
        if hcsr is None or hcsr < 1.0:
            return 0
        return 1 if grounded > 0 else 0

    def _partial(self, scen, response_type, returned, hcsr, grounded, d, clar) -> float:
        if scen is None:
            return 0.0
        slot = correctness = grounding_ok = rec_ok = 0
        if scen.no_match_expected:
            slot = 1
            rec_ok = 1 if response_type == "no_match" else 0
            correctness = 1 if (response_type == "no_match" and d.get("no_match_reason_codes")) else 0
            grounding_ok = 1 if grounded > 0 or response_type == "no_match" else 0
        elif scen.clarification_expected:
            rec_ok = 1 if response_type == "clarification" else 0
            slot = 1 if response_type == "clarification" else 0
            correctness = rec_ok
            grounding_ok = 1
        else:
            slot = 1 if response_type == "recommendation" else 0
            rec_ok = 1 if (response_type == "recommendation" and returned > 0) else 0
            correctness = 1 if (hcsr is not None and hcsr >= 1.0) else 0
            grounding_ok = 1 if grounded > 0 else 0
        return (slot + correctness + rec_ok + grounding_ok) / 4.0


# --------------------------------------------------------------- aggregation
def aggregate_scenario_variant(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeats within each scenario x variant (paired analysis unit)."""
    metric_cols = ["ndcg_at_5", "precision_at_5", "mean_graded_relevance", "hcsr",
                   "mean_violation_count", "unknown_hard_rate", "expired_rate",
                   "trace_completeness", "grounding", "handoff_success",
                   "decision_log_completeness", "turn_count", "total_latency_ms",
                   "task_success", "partial_task_score", "grounded_claim_count"]
    agg = (run_metrics.groupby(["scenario_id", "variant", "scenario_type", "difficulty",
                                "memory_dependency", "context_dependency"], dropna=False)[metric_cols]
           .mean().reset_index())
    return agg


def variant_summary(scenario_variant: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["ndcg_at_5", "precision_at_5", "mean_graded_relevance", "hcsr",
                   "mean_violation_count", "unknown_hard_rate", "trace_completeness",
                   "grounding", "handoff_success", "decision_log_completeness",
                   "turn_count", "total_latency_ms", "task_success"]
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
