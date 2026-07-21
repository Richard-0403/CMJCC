"""Conversation Orchestrator.

Drives the workflow state machine, calls each authorised component in order,
captures handoffs and failures, and produces a unified RunRecord. It never ranks
jobs itself, never mutates CandidateState implicitly, and never bypasses the
CMJCC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .. import CODE_VERSION
from ..agents.candidate_understanding import CandidateUnderstandingAgent
from ..agents.explanation_agent import ExplanationAgent
from ..agents.job_context_agent import JobContextAgent, diagnose_no_match
from ..agents.memory_agent import MemoryAgent
from ..config import AppConfig
from ..domain.candidate import CandidateState
from ..domain.dialogue import DialogueState
from ..domain.enums import ErrorCode, ResponseType, RunMode
from ..domain.enums import WorkflowState as S
from ..domain.extraction import ExtractedPreferenceSet
from ..domain.handoff import AgentHandoff, EvidenceLogEntry
from ..domain.job import JobPosting
from ..domain.recommendation import RecommendationDecision, Response
from ..domain.run_record import RunRecord
from ..evidence_store import EvidenceStore
from ..prompts import prompt_hash, render_intent_extraction
from ..ranking.scoring import SCORER_VERSION, RankingAgent
from ..retrieval.base import QuerySpec
from ..retrieval.hybrid import make_retriever
from ..utils.hashing import content_id
from ..utils.time import utcnow
from .cmjcc import CMJCC, CMJCCInput
from .feature_flags import FeatureFlags
from .state_machine import StateMachine


@dataclass
class TurnResult:
    run_record: RunRecord
    response: Response
    decision: RecommendationDecision | None
    candidate_state: CandidateState
    dialogue_state: DialogueState
    active_search_state: object | None
    handoffs: list[AgentHandoff]
    evidence_log: list[EvidenceLogEntry]
    dropped_claims: list = field(default_factory=list)
    clarification: object | None = None
    model_calls: list = field(default_factory=list)
    # extra artifacts for run-bundle export (optional)
    extracted_preferences: object | None = None
    candidate_state_before: object | None = None
    job_context_state: object | None = None
    retrieval_outcome: object | None = None


class ConversationOrchestrator:
    """End-to-end turn processor with full decision logging."""

    name = "orchestrator"

    def __init__(
        self,
        config: AppConfig,
        jobs: list[JobPosting],
        catalog_snapshot_id: str,
        catalog_hash: str,
        provider=None,
        store: EvidenceStore | None = None,
    ) -> None:
        self.config = config
        self.jobs = jobs
        self.jobs_by_id = {j.job_id: j for j in jobs}
        self.catalog_snapshot_id = catalog_snapshot_id
        self.catalog_hash = catalog_hash
        self.flags = FeatureFlags.from_config(config)
        self.store = store or EvidenceStore()
        self.retriever = make_retriever(jobs, config)
        self.rule_extractor = CandidateUnderstandingAgent()
        self.provider = provider
        self.memory = MemoryAgent(self.store, config)
        self.job_context = JobContextAgent(config)
        self.ranking = RankingAgent(self.store, config)
        self.explainer = ExplanationAgent(self.store, config)

    # ---------------------------------------------------------------- turn
    def process_turn(
        self,
        candidate_state: CandidateState,
        dialogue_state: DialogueState,
        text: str,
        scenario_id: str | None = None,
    ) -> TurnResult:
        run_id = content_id("run", dialogue_state.session_id, dialogue_state.version, text)
        sm = StateMachine()
        handoffs: list[AgentHandoff] = []
        evidence_log: list[EvidenceLogEntry] = []
        model_calls: list = []
        latencies: dict[str, float] = {}
        started = utcnow()
        candidate_before = candidate_state
        extraction = None

        def handoff(frm: str, to: str, contract: str, ok: bool, err: str | None = None) -> None:
            handoffs.append(AgentHandoff(
                handoff_id=content_id("ho", run_id, frm, to, str(len(handoffs))),
                run_id=run_id, from_component=frm, to_component=to, contract_name=contract,
                input_schema_version="1.0.0", output_schema_version="1.0.0" if ok else None,
                attempted_at=utcnow(), completed_at=utcnow() if ok else None,
                validation_passed=ok, status="completed" if ok else "failed",
                error_code=err,
            ))

        def timed(label: str, fn):
            t0 = time.perf_counter()
            out = fn()
            latencies[label] = round((time.perf_counter() - t0) * 1000, 3)
            return out

        try:
            # 1) UNDERSTANDING: append turn + extract intent.
            sm.to(S.UNDERSTANDING)
            dialogue_state = self.memory.append_turn(dialogue_state, "candidate", text)
            extraction, calls = timed("intent_extraction", lambda: self._extract(text))
            model_calls.extend(calls)
            # Fold in prior-turn dialogue evidence when the variant permits memory.
            if self.flags.use_prior_dialogue:
                extraction = self._merge_prior_dialogue(dialogue_state, extraction)
            handoff(self.rule_extractor.name, self.memory.name, "ExtractedPreferenceSet", True)

            # 2) VALIDATING: CMJCC merge + conflicts + constraints.
            sm.to(S.VALIDATING)
            cmjcc = CMJCC(self.store, self.config)
            cmjcc_out = timed("memory_merge", lambda: cmjcc.run(CMJCCInput(
                candidate_state, dialogue_state, extraction,
                self.catalog_snapshot_id, self.config, run_id,
            )))
            evidence_log.extend(cmjcc_out.evidence_log_entries)
            dialogue_state = cmjcc_out.dialogue_state
            candidate_state = cmjcc_out.candidate_state
            active = cmjcc_out.active_search_state
            handoff(self.memory.name, cmjcc.name, "CMJCCOutput", True)

            # 2b) Clarification short-circuit.
            if cmjcc_out.clarification_action is not None:
                sm.to(S.CLARIFICATION_REQUIRED)
                response = self._clarification_response(dialogue_state, cmjcc_out.clarification_action)
                sm.to(S.EXPLAINED)
                sm.to(S.COMPLETED)
                result = self._finish(
                    run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
                    model_calls, candidate_state, dialogue_state, active, None, response,
                    [], cmjcc_out.clarification_action, success=True,
                )
                result.extracted_preferences = extraction
                result.candidate_state_before = candidate_before
                result.job_context_state = cmjcc_out.job_context_state
                return result

            sm.to(S.MEMORY_UPDATED)

            # 3) CONTEXT_BUILT already done inside CMJCC (constraint bundle).
            sm.to(S.CONTEXT_BUILT)
            context = cmjcc_out.job_context_state

            # 4) RETRIEVED
            sm.to(S.RETRIEVED)
            query = QuerySpec.from_active_search(active)
            outcome = timed("retrieval", lambda: self.retriever.retrieve(
                query, self.jobs, self.config.experiment.retrieval_pool_size))
            pool = [self.jobs_by_id[r.job_id] for r in outcome.retrieved]
            # Fallback: if nothing recalled lexically, use whole catalog once.
            if not pool:
                pool = list(self.jobs)
                outcome.expanded = True
                outcome.expansion_reason = "empty_recall_fallback"
            handoff(self.retriever.name, self.job_context.name, "RetrievalOutcome", True)

            # 5) FILTERED (eligibility). In no_context, skip explicit hard filtering.
            sm.to(S.FILTERED)
            if self.flags.explicit_constraint_orchestration and context is not None:
                eligibility = timed("filtering", lambda: [
                    self.job_context.evaluate(j, context) for j in pool])
            else:
                # no_context: treat all recalled jobs as "eligible" (no hard filter).
                eligibility = timed("filtering", lambda: [
                    self._passthrough_eligibility(j) for j in pool])
            handoff(self.job_context.name, self.ranking.name, "EligibilityResults", True)

            # 6) RANKED or NO_MATCH
            ranked = timed("ranking", lambda: self.ranking.rank(active, self.jobs_by_id, eligibility))
            no_match = len(ranked) == 0
            no_match_codes: list[str] = []
            if no_match:
                diag = diagnose_no_match(eligibility, context) if context else {"blocking_constraints": []}
                no_match_codes = [b["field"] for b in diag.get("blocking_constraints", [])]

            decision = RecommendationDecision(
                decision_id=content_id("dec", run_id),
                session_id=dialogue_state.session_id,
                active_search_id=active.active_search_id,
                context_id=context.context_id if context else None,
                experiment_variant=self.config.experiment.variant.value,
                retrieved_job_ids=[r.job_id for r in outcome.retrieved],
                eligibility_results=eligibility,
                ranked_jobs=ranked,
                selected_job_ids=[rj.job_id for rj in ranked[: self.config.experiment.top_k]],
                no_match=no_match,
                no_match_reason_codes=no_match_codes,
                created_at=utcnow(),
                scorer_version=SCORER_VERSION,
                config_hash=self.config.config_hash(),
            )

            if no_match:
                sm.to(S.NO_MATCH)
            else:
                sm.to(S.RANKED)

            # 7) EXPLAINED
            sm.to(S.EXPLAINED)
            response, dropped = timed(
                "explanation", lambda: self.explainer.explain(decision, active, self.jobs_by_id))
            handoff(self.ranking.name, self.explainer.name, "RecommendationDecision", True)

            sm.to(S.COMPLETED)
            result = self._finish(
                run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
                model_calls, candidate_state, dialogue_state, active, decision, response,
                dropped, None, success=True,
            )
            result.extracted_preferences = extraction
            result.candidate_state_before = candidate_before
            result.job_context_state = context
            result.retrieval_outcome = outcome
            return result

        except Exception as exc:  # noqa: BLE001 - convert to explicit failed run
            sm.fail()
            handoff(self.name, self.name, "run", False, err=ErrorCode.INTERNAL_ERROR.value)
            response = Response(
                response_id=content_id("resp", run_id),
                session_id=dialogue_state.session_id,
                response_type=ResponseType.ERROR.value,
                message=f"The request could not be completed: {type(exc).__name__}: {exc}",
                claims=[], created_at=utcnow(),
            )
            return self._finish(
                run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
                model_calls, candidate_state, dialogue_state, None, None, response,
                [], None, success=False, failure_code=ErrorCode.INTERNAL_ERROR.value,
            )

    # ------------------------------------------------------------ helpers
    def _extract(self, text: str) -> tuple[ExtractedPreferenceSet, list]:
        """Extract intent according to the run mode, with rule fallback."""
        mode = self.config.llm.mode
        if mode == RunMode.DETERMINISTIC or self.provider is None:
            return self.rule_extractor.extract(text), []

        from ..llm.provider import LLMError
        from ..llm.retry import retry_call
        from ..llm.structured_output import parse_extraction

        prompt = render_intent_extraction(text)
        calls: list = []

        def once() -> ExtractedPreferenceSet:
            payload, record = self.provider.complete_json(prompt, purpose="intent_extraction")
            calls.append(record)
            return parse_extraction(payload)

        try:
            return retry_call(once, self.config.llm.max_retries), calls
        except LLMError:
            # Explicit fallback to the deterministic rule extractor (no fabrication).
            return self.rule_extractor.extract(text), calls

    def _merge_prior_dialogue(
        self, dialogue_state: DialogueState, current: ExtractedPreferenceSet
    ) -> ExtractedPreferenceSet:
        """Merge earlier candidate-turn preferences (memory) with the current turn.

        Prior preferences come first so that current-turn statements take
        precedence for scalar overrides (salary, location, level). This is what
        distinguishes ``full`` from ``no_memory`` / ``one_shot`` across turns.
        """
        prior_texts = [t.text for t in dialogue_state.turns[:-1] if t.speaker == "candidate"]
        if not prior_texts:
            return current
        prior_prefs = []
        for text in prior_texts:
            prior_prefs.extend(self.rule_extractor.extract(text).preferences)
        return current.model_copy(update={"preferences": prior_prefs + list(current.preferences)})

    def _passthrough_eligibility(self, job: JobPosting):
        from ..domain.constraints import EligibilityResult
        return EligibilityResult(
            eligibility_result_id=content_id("elig", "nocontext", job.job_id),
            job_id=job.job_id, eligible=True, checks=[], hard_violation_count=0,
            unknown_hard_constraint_count=0, filtered_reason_codes=[],
        )

    def _clarification_response(self, dialogue_state, clar) -> Response:
        # Optionally rephrase via provider (hybrid); otherwise use policy text.
        text = clar.question_text
        if self.provider is not None and self.config.llm.mode != RunMode.DETERMINISTIC:
            from ..prompts import prompt_templates
            tmpl = prompt_templates()["clarification"]
            prompt = (tmpl.replace("{field}", ",".join(clar.target_fields))
                          .replace("{reason}", clar.reason_code)
                          .replace("{options}", ", ".join(clar.options)))
            text, _ = self.provider.complete_text(prompt, purpose="clarification", fallback=clar.question_text)
        return Response(
            response_id=content_id("resp", clar.clarification_id, dialogue_state.version),
            session_id=dialogue_state.session_id,
            response_type=ResponseType.CLARIFICATION.value,
            message=text, claims=[], created_at=utcnow(),
        )

    def _finish(
        self, run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
        model_calls, candidate_state, dialogue_state, active, decision, response,
        dropped, clarification, success, failure_code=None,
    ) -> TurnResult:
        completed = utcnow()
        total = round((completed - started).total_seconds() * 1000, 3)
        latencies["total"] = total
        state_ids = {
            "candidate_state": f"{candidate_state.candidate_id}:v{candidate_state.version}",
            "dialogue_state": f"{dialogue_state.session_id}:v{dialogue_state.version}",
        }
        if active is not None:
            state_ids["active_search_state"] = active.active_search_id
        run_record = RunRecord(
            run_id=run_id, scenario_id=scenario_id, session_id=dialogue_state.session_id,
            candidate_id=candidate_state.candidate_id,
            experiment_variant=self.config.experiment.variant.value,
            workflow_states=sm.as_str_list(), state_object_ids=state_ids,
            handoff_ids=[h.handoff_id for h in handoffs],
            evidence_log_ids=[e.log_id for e in evidence_log],
            final_decision_id=decision.decision_id if decision else None,
            final_response_id=response.response_id if response else None,
            started_at=started, completed_at=completed,
            component_latency_ms=latencies, total_latency_ms=total,
            success=success, failure_code=failure_code,
            config_hash=self.config.config_hash(), catalog_hash=self.catalog_hash,
            prompt_hash=prompt_hash(),
            model_manifest=(self.provider.manifest() if self.provider else {"provider": "none"}),
            code_version=CODE_VERSION,
        )
        return TurnResult(
            run_record=run_record, response=response, decision=decision,
            candidate_state=candidate_state, dialogue_state=dialogue_state,
            active_search_state=active, handoffs=handoffs, evidence_log=evidence_log,
            dropped_claims=dropped, clarification=clarification, model_calls=model_calls,
        )


def make_provider(config: AppConfig, replay_path: str | None = None):
    """Provider factory keyed by run mode."""
    mode = config.llm.mode
    if mode == RunMode.DETERMINISTIC:
        from ..llm.mock_provider import MockLLMProvider
        return MockLLMProvider()
    if mode == RunMode.REPLAY:
        from ..llm.replay import ReplayProvider
        return ReplayProvider(replay_path or "model_calls.jsonl")
    # hybrid
    if config.llm.provider == "remote":
        from ..llm.remote_provider import RemoteLLMProvider
        return RemoteLLMProvider(
            timeout_seconds=config.llm.timeout_seconds,
            extraction_temperature=config.llm.extraction_temperature,
            response_temperature=config.llm.response_temperature,
        )
    from ..llm.mock_provider import MockLLMProvider
    return MockLLMProvider()
