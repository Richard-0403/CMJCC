"""Conversation Orchestrator.

Drives the workflow state machine, calls each authorised component in order,
captures handoffs and failures, and produces a unified RunRecord. It never ranks
jobs itself, never mutates CandidateState implicitly, and never bypasses the
CMJCC.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

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
from ..domain.extraction import ExtractedPreference, ExtractedPreferenceSet
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
from ..utils.observability import RunTrace, run_trace
from ..utils.time import utcnow
from .cmjcc import CMJCC, CMJCCInput
from .feature_flags import FeatureFlags
from .state_machine import StateMachine

logger = logging.getLogger(__name__)

# How a preference's value was produced, recorded on ExtractedPreference.metadata.
#: Marks a turn whose prior-dialogue preferences had to be recovered by re-parsing its
#: text, because it carries no ``extraction_snapshot``. Only a dialogue persisted before
#: snapshots existed can produce it. A fresh official run must never contain it -- re-parsing
#: substitutes the rule extractor's reading for the model's -- so ``ExperimentRunner``
#: treats its presence as a failed experiment rather than a warning to skim.
LEGACY_REPARSE_WARNING = "legacy_rule_reparse"

_METHOD_RULE = "rule"
_METHOD_LLM = "llm"

# Which rung of the validation ladder produced the value, mirroring
# ``FieldResult.source`` and recorded on ExtractedPreference.metadata so the
# evaluation pipeline can report schema-failure and fallback rates (R8.12/R13.1).
# ``unresolved`` is the fourth, orchestrator-only state: a stated value that no
# rung could normalize and that is preserved as an UNCONFIRMED constraint (R8.9).
_SOURCE_NORMALIZED = "normalized"
_SOURCE_REPAIRED = "repaired"


def _failed_call_record(purpose: str, prompt: str, exc: BaseException,
                        provider: object, attempt: int):
    """A call record for a model attempt that raised (R11.1).

    Carries the exception's CLASS NAME only. A transport error message can quote the
    request -- and therefore the credential -- so the message is never recorded.

    The ``call_id`` is deliberately suffixed with the attempt number so a failed
    attempt can never occupy the replay key of the successful call that shares its
    (purpose, prompt): :class:`~jobrec.llm.replay.ReplayProvider` indexes records by
    ``call_id``, and a body-less failure row must not shadow a real recording.
    """
    from ..llm.provider import LLMCallRecord
    from ..utils.hashing import content_id

    return LLMCallRecord(
        call_id=f"{content_id('call', purpose, prompt)}#failed{attempt}",
        purpose=purpose,
        prompt=prompt,
        raw_response="",
        parsed_ok=False,
        latency_ms=0.0,
        provider=getattr(provider, "name", "unknown"),
        model=getattr(provider, "model", "unknown"),
        metadata={"failed": True, "error": type(exc).__name__, "attempts": attempt},
    )
_SOURCE_RULE_FALLBACK = "rule_fallback"
_SOURCE_UNRESOLVED = "unresolved"


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
    #: Structured log records emitted while processing this turn (R27.1-27.3);
    #: exported as ``log_trace.jsonl`` in the run bundle.
    log_trace: list[dict] = field(default_factory=list)
    #: Every :class:`~jobrec.domain.evidence.EvidenceItem` registered in this
    #: session's ``EvidenceStore`` as of the end of this turn; exported as
    #: ``evidence_items.jsonl`` in the run bundle so a claim's ``evidence_ids``
    #: can be resolved to "field X of object Y = value Z" offline.
    #:
    #: The store is SESSION-scoped and accumulates across turns, which is what a
    #: claim needs: a claim made on the final turn may legitimately cite evidence
    #: registered several turns earlier, so the dump covers the whole session
    #: rather than only the turn that produced it.
    evidence_items: list = field(default_factory=list)
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
        versions: dict | None = None,
    ) -> TurnResult:
        run_id = content_id("run", dialogue_state.session_id, dialogue_state.version, text)
        # Structured trace for this turn (R27): every record carries the run,
        # session, scenario and variant so logs are filterable, and the collected
        # records are exported as ``log_trace.jsonl`` in the run bundle.
        trace = run_trace(
            self.config,
            run_id=run_id,
            session_id=dialogue_state.session_id,
            scenario_id=scenario_id,
        )
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
            trace.info(self.name, "turn_started", turn_index=len(dialogue_state.turns) + 1)
            # 1) UNDERSTANDING: append turn + extract intent.
            sm.to(S.UNDERSTANDING)
            dialogue_state = self.memory.append_turn(dialogue_state, "candidate", text)
            extraction, calls = timed(
                "intent_extraction", lambda: self._extract(text, trace=trace))
            model_calls.extend(calls)
            # Fold in prior-turn dialogue evidence when the variant permits memory.
            # The continuation gate keeps this on the SAME shared code path rather than
            # forking a pipeline. Both conjuncts are load-bearing and neither is
            # redundant: within the five-variant matrix no variant has
            # use_prior_dialogue=True while use_multi_turn_continuation=False, but
            # ``memory.use_multi_turn_continuation: false`` in config resolves exactly
            # that combination for any variant (see FeatureFlags.from_config), and this
            # is where it takes effect -- turning continuation off then stops prior-turn
            # evidence from being carried forward even under ``full``. The other half of
            # the flag (never continuing a dialogue past a clarification at all) is
            # enforced in ExperimentRunner._continues_dialogue.
            prior_preferences: list[ExtractedPreference] = []
            if self.flags.use_prior_dialogue and self.flags.use_multi_turn_continuation:
                prior_preferences, prior_warnings = self._prior_dialogue_preferences(
                    dialogue_state)
                if prior_warnings:
                    # Surfaced on the CURRENT extraction because that is what the run
                    # bundle exports; the run is then auditable for having taken the
                    # legacy path without inspecting every turn.
                    extraction = extraction.model_copy(update={
                        "extraction_warnings": [*extraction.extraction_warnings,
                                                *prior_warnings],
                    })
            handoff(self.rule_extractor.name, self.memory.name, "ExtractedPreferenceSet", True)

            # 2) VALIDATING: CMJCC merge + conflicts + constraints.
            sm.to(S.VALIDATING)
            cmjcc = CMJCC(self.store, self.config)
            cmjcc_out = timed("memory_merge", lambda: cmjcc.run(CMJCCInput(
                candidate_state, dialogue_state, extraction,
                self.catalog_snapshot_id, self.config, run_id,
                prior_preferences=prior_preferences,
            )))
            evidence_log.extend(cmjcc_out.evidence_log_entries)
            dialogue_state = cmjcc_out.dialogue_state
            candidate_state = cmjcc_out.candidate_state
            active = cmjcc_out.active_search_state
            handoff(self.memory.name, cmjcc.name, "CMJCCOutput", True)

            # 2b) Clarification short-circuit.
            if cmjcc_out.clarification_action is not None:
                sm.to(S.CLARIFICATION_REQUIRED)
                clar_action = cmjcc_out.clarification_action
                trace.info(
                    cmjcc.name, "clarification_requested",
                    target_fields=list(getattr(clar_action, "target_fields", []) or []),
                    reason_code=getattr(clar_action, "reason_code", None),
                )
                response = self._clarification_response(
                    dialogue_state, cmjcc_out.clarification_action, model_calls)
                sm.to(S.EXPLAINED)
                sm.to(S.COMPLETED)
                result = self._finish(
                    run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
                    model_calls, candidate_state, dialogue_state, active, None, response,
                    [], cmjcc_out.clarification_action, success=True, versions=versions,
                    trace=trace,
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
                trace.warning(
                    self.retriever.name, "empty_recall_fallback",
                    "lexical recall was empty; falling back to the full catalog",
                    catalog_size=len(pool),
                    requested_pool_size=self.config.experiment.retrieval_pool_size,
                )
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
                diag = (diagnose_no_match(
                    eligibility, context,
                    catalog_size=len(self.jobs_by_id), pool_size=len(pool),
                    ranked_size=len(ranked))
                    if context else {"blocking_constraints": []})
                no_match_codes = [b["field"] for b in diag.get("blocking_constraints", [])]
                trace.warning(
                    self.job_context.name, "no_match",
                    "no job survived the constraint and ranking layers",
                    pool_size=len(pool), reason_codes=no_match_codes,
                )

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
                # The diagnosis travels with the decision, so the run bundle records what
                # each filtering stage removed instead of only the final verdict.
                no_match_diagnosis=(diag if no_match else None),
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
                dropped, None, success=True, versions=versions, trace=trace,
            )
            result.extracted_preferences = extraction
            result.candidate_state_before = candidate_before
            result.job_context_state = context
            result.retrieval_outcome = outcome
            return result

        except Exception as exc:  # noqa: BLE001 - convert to explicit failed run
            sm.fail()
            handoff(self.name, self.name, "run", False, err=ErrorCode.INTERNAL_ERROR.value)
            trace.system_failure(
                self.name, "run_failed", f"{type(exc).__name__}: {exc}",
                failure_code=ErrorCode.INTERNAL_ERROR.value,
                error_type=type(exc).__name__,
                workflow_state=sm.as_str_list()[-1] if sm.as_str_list() else None,
            )
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
                versions=versions, trace=trace,
            )

    # ------------------------------------------------------------ helpers
    def _extract(
        self, text: str, trace: RunTrace | None = None
    ) -> tuple[ExtractedPreferenceSet, list]:
        """Extract intent according to the run mode, with field validation and
        a bounded repair -> retry -> rule-fallback recovery ladder (R8.8/8.9).

        Deterministic mode uses the rule extractor directly (its output is already
        normalized and must stay byte-stable), tagging every field
        ``extraction_method="rule"``. Hybrid mode calls the model, validates every
        field via :func:`validate_extraction`, and — only if some field fails —
        attempts schema repair, then a single bounded model retry, then a rule-based
        fallback, logging at each step. A stated constraint is never dropped: an
        unrecoverable value is preserved as ``UNCONFIRMED`` with a warning.

        Every returned preference carries ``extraction_method`` (``rule``|``llm``)
        and ``extraction_source`` (the ladder rung that produced the value) on its
        metadata, so ``extracted_preferences.json`` is self-describing (R13.1).
        """
        # Structured emission (R27) happens alongside the existing human-readable
        # logging on this module's logger; a detached trace keeps direct unit-level
        # calls into ``_extract`` working unchanged.
        trace = trace if trace is not None else RunTrace()
        component = self.rule_extractor.name
        mode = self.config.llm.mode
        if mode == RunMode.DETERMINISTIC or self.provider is None:
            return self._tag_all(self.rule_extractor.extract(text), _METHOD_RULE), []

        from ..llm.field_validation import validate_extraction
        from ..llm.provider import LLMError
        from ..llm.retry import retry_call
        from ..llm.structured_output import parse_extraction_lenient

        prompt = render_intent_extraction(text)
        calls: list = []

        def once() -> ExtractedPreferenceSet:
            try:
                payload, record = self.provider.complete_json(
                    prompt, purpose="intent_extraction")
            except Exception as exc:
                # A raising attempt used to leave NO record at all, because the append
                # below only ran on success. An empty ``model_calls.jsonl`` therefore
                # could not be distinguished from "the call died" -- two real runs were
                # silently in that state. Record the attempt, then re-raise so the
                # bounded retry and the rule fallback behave exactly as before.
                calls.append(_failed_call_record(
                    "intent_extraction", prompt, exc, self.provider, len(calls) + 1))
                raise
            calls.append(record)
            # Lenient parse: model reliably emits field/value/strength/polarity;
            # provenance fields are defaulted so a valid response is actually used.
            return parse_extraction_lenient(payload, utterance=text)

        # ---- model call (bounded retry on transient LLM errors) --------------
        try:
            pref_set = retry_call(once, self.config.llm.max_retries)
        except LLMError:
            # Explicit fallback to the deterministic rule extractor (no fabrication).
            logger.warning("extraction: model call failed; falling back to rule extractor")
            trace.warning(
                component, "extraction_model_call_failed",
                "model call failed; falling back to the rule extractor",
                max_retries=self.config.llm.max_retries,
            )
            return self._tag_all(
                self.rule_extractor.extract(text), _METHOD_RULE, _SOURCE_RULE_FALLBACK
            ), calls

        # ---- 1) field validation right after lenient parse (R8.8) ------------
        pref_set, results = validate_extraction(pref_set)
        pref_set = self._tag_llm(pref_set, results)
        if all(r.ok for r in results):
            return pref_set, calls

        # ---- 2a) schema repair (coerce obvious shapes) -----------------------
        failed = sum(1 for r in results if not r.ok)
        logger.warning(
            "extraction: %d field(s) failed validation; attempting schema repair", failed
        )
        trace.validation_error(
            component, "extraction_field_validation_failed",
            "field validation failed; attempting schema repair",
            failed_fields=failed,
            fields=[p.field_name for p, r in zip(pref_set.preferences, results, strict=False)
                    if not r.ok],
        )
        pref_set, results = self._repair_fields(pref_set, results, trace=trace)
        if all(r.ok for r in results):
            return pref_set, calls

        # ---- 2b) single bounded model retry (HYBRID only) --------------------
        logger.warning("extraction: repair incomplete; attempting one bounded model retry")
        trace.warning(
            component, "extraction_bounded_retry",
            "schema repair incomplete; attempting one bounded model retry",
        )
        try:
            retried = once()
            retried, retried_results = validate_extraction(retried)
            retried = self._tag_llm(retried, retried_results)
            retried, retried_results = self._repair_fields(
                retried, retried_results, trace=trace)
            # Prefer the retry only if it recovers at least as many fields.
            if sum(1 for r in retried_results if r.ok) >= sum(1 for r in results if r.ok):
                pref_set, results = retried, retried_results
        except LLMError:
            logger.warning("extraction: bounded model retry failed")
            trace.warning(component, "extraction_bounded_retry_failed",
                          "the bounded model retry failed")
        if all(r.ok for r in results):
            return pref_set, calls

        # ---- 2c) rule fallback for still-failing fields (never drop) ---------
        unresolved = sum(1 for r in results if not r.ok)
        logger.error(
            "extraction: %d field(s) unresolved after repair/retry; applying rule fallback",
            unresolved,
        )
        trace.validation_error(
            component, "extraction_unresolved_fields",
            "fields unresolved after repair and retry; applying the rule fallback",
            unresolved_fields=unresolved,
        )
        pref_set = self._rule_fallback(pref_set, results, text, trace=trace)
        return pref_set, calls

    # -- extraction-method tagging & recovery helpers ----------------------
    @staticmethod
    def _tag_all(
        pref_set: ExtractedPreferenceSet,
        method: str,
        source: str = _SOURCE_NORMALIZED,
    ) -> ExtractedPreferenceSet:
        """Return a copy tagging every preference with method + source provenance."""
        tagged = [
            p.model_copy(update={"metadata": {
                **p.metadata, "extraction_method": method, "extraction_source": source,
            }})
            for p in pref_set.preferences
        ]
        return pref_set.model_copy(update={"preferences": tagged})

    @staticmethod
    def _tag_llm(pref_set: ExtractedPreferenceSet, results: list) -> ExtractedPreferenceSet:
        """Tag model-derived preferences with their per-field ``FieldResult.source``.

        A field that validated cleanly carries that result's source; a field that
        failed validation is provisionally ``unresolved`` and is overwritten by the
        rung that eventually resolves it (repair or rule fallback).
        """
        by_index = {i: r for i, r in enumerate(results)}
        tagged = []
        for i, p in enumerate(pref_set.preferences):
            res = by_index.get(i)
            source = res.source if (res is not None and res.ok) else _SOURCE_UNRESOLVED
            tagged.append(p.model_copy(update={"metadata": {
                **p.metadata, "extraction_method": _METHOD_LLM, "extraction_source": source,
            }}))
        return pref_set.model_copy(update={"preferences": tagged})

    def _repair_fields(
        self, pref_set: ExtractedPreferenceSet, results: list, trace: RunTrace | None = None
    ) -> tuple[ExtractedPreferenceSet, list]:
        """Attempt to coerce obviously-wrong shapes for each failing field and
        re-validate. Repaired values keep ``extraction_method="llm"`` (still
        model-derived) and record ``source="repaired"`` on their FieldResult.
        """
        from ..llm.field_validation import validate_field

        new_prefs = list(pref_set.preferences)
        new_results = list(results)
        extra_warnings: list[str] = []
        for i, (pref, res) in enumerate(zip(pref_set.preferences, results, strict=False)):
            if res.ok:
                continue
            repaired_raw = _repair_raw(pref.field_name, pref.normalized_value)
            if repaired_raw is None:
                continue
            new_res = validate_field(pref.field_name, repaired_raw)
            if new_res.ok:
                new_res.source = _SOURCE_REPAIRED
                new_prefs[i] = pref.model_copy(update={
                    "normalized_value": new_res.value,
                    "metadata": {**pref.metadata, "extraction_source": _SOURCE_REPAIRED},
                })
                new_results[i] = new_res
                msg = f"{pref.field_name}: repaired via schema coercion"
                extra_warnings.append(msg)
                logger.warning("extraction: %s", msg)
                if trace is not None:
                    trace.warning(
                        self.rule_extractor.name, "extraction_field_repaired", msg,
                        field=pref.field_name,
                    )
        new_set = pref_set.model_copy(update={
            "preferences": new_prefs,
            "extraction_warnings": [*pref_set.extraction_warnings, *extra_warnings],
        })
        return new_set, new_results

    def _rule_fallback(
        self,
        pref_set: ExtractedPreferenceSet,
        results: list,
        text: str,
        trace: RunTrace | None = None,
    ) -> ExtractedPreferenceSet:
        """For each still-failing field, substitute the rule-extracted value when
        available (tagged ``extraction_method="rule"``); otherwise preserve the
        stated constraint as ``UNCONFIRMED`` with a warning. A stated constraint is
        never silently dropped (R8.9).
        """
        from ..domain.enums import ConfirmationStatus

        rule_by_field: dict[str, object] = {}
        for rp in self.rule_extractor.extract(text).preferences:
            rule_by_field.setdefault(rp.field_name, rp)

        new_prefs = list(pref_set.preferences)
        extra_warnings: list[str] = []
        for i, (pref, res) in enumerate(zip(pref_set.preferences, results, strict=False)):
            if res.ok:
                continue
            rule_pref = rule_by_field.get(pref.field_name)
            if rule_pref is not None:
                new_prefs[i] = pref.model_copy(update={
                    "normalized_value": rule_pref.normalized_value,
                    "confirmation_status": ConfirmationStatus.UNCONFIRMED,
                    "metadata": {
                        **pref.metadata,
                        "extraction_method": _METHOD_RULE,
                        "extraction_source": _SOURCE_RULE_FALLBACK,
                    },
                })
                msg = (f"{pref.field_name}: LLM value unrecoverable; used rule-based "
                       "fallback (unconfirmed)")
                logger.error("extraction: %s", msg)
                if trace is not None:
                    trace.validation_error(
                        self.rule_extractor.name, "extraction_field_rule_fallback", msg,
                        field=pref.field_name,
                    )
            else:
                # Never drop the stated constraint: keep the raw value, mark unconfirmed.
                new_prefs[i] = pref.model_copy(update={
                    "confirmation_status": ConfirmationStatus.UNCONFIRMED,
                    "metadata": {**pref.metadata, "extraction_source": _SOURCE_UNRESOLVED},
                })
                msg = (f"{pref.field_name}: value could not be normalized; preserved as "
                       "unconfirmed constraint")
                logger.warning("extraction: %s", msg)
                if trace is not None:
                    trace.warning(
                        self.rule_extractor.name, "extraction_field_unconfirmed", msg,
                        field=pref.field_name,
                    )
            extra_warnings.append(msg)
        return pref_set.model_copy(update={
            "preferences": new_prefs,
            "extraction_warnings": [*pref_set.extraction_warnings, *extra_warnings],
        })

    def _prior_dialogue_preferences(
        self, dialogue_state: DialogueState
    ) -> tuple[list[ExtractedPreference], list[str]]:
        """Earlier candidate turns' preferences, oldest first, as they were understood.

        Returns ``(preferences, warnings)``. Read from each turn's stored
        ``extraction_snapshot``, so nothing is re-extracted and every entry keeps the
        strength, confirmation, provenance metadata, evidence id and turn id the turn that
        stated it produced.

        This replaced re-parsing every earlier utterance with the RULE extractor on every
        turn. That was not just redundant work: it threw away the original extraction, so
        in hybrid mode the model's reading of turn 1 was replaced by the rule extractor's
        from turn 2 onwards -- a two-turn hybrid state was ``rule(turn1) + llm(turn2)``,
        and the strength assigned to an earlier field was recomputed rather than
        remembered. On the 42-scenario authoritative set that affected the 12 multi-turn
        scenarios.

        The fallback for a turn with no snapshot is kept so a dialogue persisted before
        this field existed can still be processed, but it re-introduces exactly the defect
        above, so it is LABELLED rather than silent: the warning rides on the extraction
        into the run bundle, and ``ExperimentRunner`` refuses to complete an experiment
        whose runs carry it.
        """
        prior_turns = [t for t in dialogue_state.turns[:-1] if t.speaker == "candidate"]
        prior_prefs: list[ExtractedPreference] = []
        warnings: list[str] = []
        for turn in prior_turns:
            snapshot = turn.extraction_snapshot
            if snapshot is not None:
                prior_prefs.extend(snapshot.preferences)
                continue
            # Legacy dialogue only: no snapshot was ever recorded for this turn.
            if LEGACY_REPARSE_WARNING not in warnings:
                warnings.append(LEGACY_REPARSE_WARNING)
            logger.warning(
                "prior dialogue: turn %s has no extraction snapshot; falling back to "
                "re-parsing its text with the rule extractor", turn.turn_id)
            tagged = self._tag_all(self.rule_extractor.extract(turn.text), _METHOD_RULE)
            prior_prefs.extend(
                pref.model_copy(update={"origin_turn_id": turn.turn_id,
                                        "metadata": {**pref.metadata,
                                                     "extraction_source":
                                                         LEGACY_REPARSE_WARNING}})
                for pref in tagged.preferences
            )
        return prior_prefs, warnings

    def _passthrough_eligibility(self, job: JobPosting):
        from ..domain.constraints import EligibilityResult
        return EligibilityResult(
            eligibility_result_id=content_id("elig", "nocontext", job.job_id),
            job_id=job.job_id, eligible=True, checks=[], hard_violation_count=0,
            unknown_hard_constraint_count=0, filtered_reason_codes=[],
        )

    def _clarification_response(self, dialogue_state, clar, model_calls=None) -> Response:
        """Build the clarification response, optionally rephrased by the provider.

        ``model_calls`` receives the phrasing call's record. Threading it is not
        cosmetic: this call is a real, billed model call in hybrid mode (it fired 21+
        times in a single experiment) and its record used to be discarded at the call
        site, so it appeared in no artifact -- breaking call accounting, token/cost
        totals and replay completeness for every clarification turn.
        """
        text = clar.question_text
        if self.provider is not None and self.config.llm.mode != RunMode.DETERMINISTIC:
            from ..prompts import prompt_templates
            tmpl = prompt_templates()["clarification"]
            prompt = (tmpl.replace("{field}", ",".join(clar.target_fields))
                          .replace("{reason}", clar.reason_code)
                          .replace("{options}", ", ".join(clar.options)))
            text, record = self.provider.complete_text(
                prompt, purpose="clarification", fallback=clar.question_text)
            if model_calls is not None:
                model_calls.append(record)
        return Response(
            response_id=content_id("resp", clar.clarification_id, dialogue_state.version),
            session_id=dialogue_state.session_id,
            response_type=ResponseType.CLARIFICATION.value,
            message=text, claims=[], created_at=utcnow(),
        )

    def _finish(
        self, run_id, scenario_id, sm, started, latencies, handoffs, evidence_log,
        model_calls, candidate_state, dialogue_state, active, decision, response,
        dropped, clarification, success, failure_code=None, versions=None, trace=None,
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
        # JSON-safe snapshot of the resolved flags (R5.4): convert the ``variant`` enum to
        # its string value so the record serializes cleanly under model_dump(mode="json").
        feature_flags = asdict(self.flags)
        feature_flags["variant"] = self.flags.variant.value
        run_record = RunRecord(
            run_id=run_id, scenario_id=scenario_id, session_id=dialogue_state.session_id,
            candidate_id=candidate_state.candidate_id,
            experiment_variant=self.config.experiment.variant.value,
            feature_flags=feature_flags,
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
            db_version=(versions or {}).get("db_version"),
            migration_version=(versions or {}).get("migration_version"),
        )
        if trace is not None:
            # Closing lifecycle record, emitted before the trace is snapshotted so
            # every bundle's ``log_trace.jsonl`` ends on the turn's outcome (R27.3).
            trace.info(
                self.name, "turn_completed",
                success=success, failure_code=failure_code,
                response_type=getattr(response, "response_type", None),
                total_latency_ms=total,
            )
        return TurnResult(
            run_record=run_record, response=response, decision=decision,
            candidate_state=candidate_state, dialogue_state=dialogue_state,
            active_search_state=active, handoffs=handoffs, evidence_log=evidence_log,
            dropped_claims=dropped, clarification=clarification, model_calls=model_calls,
            log_trace=(trace.records if trace is not None else []),
            # Snapshot of the session-scoped evidence registry: every id a claim
            # of this run can cite resolves inside this list (R10, R11).
            evidence_items=self.store.all(),
        )


def _repair_raw(field_name: str, raw: object) -> object | None:
    """Coerce an obviously-wrong shape into a plausible scalar for re-validation.

    Handles the common ways a model mis-shapes a field: a single-element list
    (``["remote"]`` -> ``"remote"``), a wrapper object (``{"value": "remote"}`` ->
    ``"remote"``), or a padded string. Returns ``None`` when no unambiguous
    coercion is possible, leaving the field for the rule fallback.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        non_empty = [x for x in raw if x not in (None, "")]
        return non_empty[0] if len(non_empty) == 1 else None
    if isinstance(raw, dict):
        for key in ("value", field_name, "name", "label", "text"):
            candidate = raw.get(key)
            if candidate not in (None, ""):
                return candidate
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        # Only worth re-validating if trimming actually changed the string.
        return stripped if stripped != raw and stripped else None
    return None


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
