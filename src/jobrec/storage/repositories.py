"""Repositories: persist and retrieve states, decisions, runs and evidence.

Two implementations share one Protocol:
- ``SqlRepository``     : PostgreSQL-backed (production/default store).
- ``InMemoryRepository``: no database required (tests, offline demos). Selecting
  it is always explicit; the system never silently loses persistence.

Both implement the run-detail levels of ``GET /v1/runs/{run_id}`` (R12): a run
bundle carries the run record, response and decision, while handoffs, evidence,
state versions and raw model outputs are attached only when requested. Raw model
outputs and state payloads are redacted on the way out (R12.3) via
``utils.redaction``, honouring ``config.logging.redact_candidate_text``.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import AppConfig
from ..domain.candidate import CandidateState
from ..domain.dialogue import DialogueState
from ..orchestration.orchestrator import TurnResult
from ..utils.redaction import redact, redact_payload
from ..utils.time import utcnow


class Repository(Protocol):
    def upsert_candidate_state(self, state: CandidateState) -> None: ...
    def get_candidate_state(self, candidate_id: str, version: int | None = None) -> CandidateState | None: ...
    def create_session(self, session_id: str, candidate_id: str, variant: str) -> None: ...
    def get_session(self, session_id: str) -> dict | None: ...
    def get_latest_dialogue_state(self, session_id: str) -> DialogueState | None: ...
    def save_turn(self, turn: TurnResult, evidence_items: list, model_calls: list) -> None: ...
    def get_run(self, run_id: str, **flags: bool) -> dict | None: ...


# --------------------------------------------------------------------------- mem
class InMemoryRepository:
    """A dict-backed repository for tests and offline runs."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config
        self.candidates: dict[tuple[str, int], CandidateState] = {}
        self.sessions: dict[str, dict] = {}
        self.dialogues: dict[tuple[str, int], DialogueState] = {}
        self.runs: dict[str, dict] = {}

    def upsert_candidate_state(self, state: CandidateState) -> None:
        self.candidates[(state.candidate_id, state.version)] = state

    def get_candidate_state(self, candidate_id: str, version: int | None = None) -> CandidateState | None:
        versions = [v for (cid, v) in self.candidates if cid == candidate_id]
        if not versions:
            return None
        v = version if version is not None else max(versions)
        return self.candidates.get((candidate_id, v))

    def create_session(self, session_id: str, candidate_id: str, variant: str) -> None:
        self.sessions[session_id] = {
            "session_id": session_id, "candidate_id": candidate_id,
            "experiment_variant": variant, "created_at": utcnow().isoformat(),
        }

    def get_session(self, session_id: str) -> dict | None:
        return self.sessions.get(session_id)

    def get_latest_dialogue_state(self, session_id: str) -> DialogueState | None:
        versions = [v for (sid, v) in self.dialogues if sid == session_id]
        if not versions:
            return None
        return self.dialogues.get((session_id, max(versions)))

    def save_turn(self, turn: TurnResult, evidence_items: list, model_calls: list) -> None:
        self.upsert_candidate_state(turn.candidate_state)
        self.dialogues[(turn.dialogue_state.session_id, turn.dialogue_state.version)] = turn.dialogue_state
        self.runs[turn.run_record.run_id] = _bundle(turn, evidence_items, model_calls)

    def get_run(self, run_id: str, include_states: bool = False, include_evidence: bool = False,
                include_handoffs: bool = False, include_raw_model_outputs: bool = False) -> dict | None:
        bundle = self.runs.get(run_id)
        if bundle is None:
            return None
        redact_text = _redact_candidate_text(self.config)
        out: dict[str, Any] = {
            "run_record": bundle["run_record"],
            "response": bundle["response"],
            "decision": bundle["decision"],
        }
        if include_handoffs:
            out["handoffs"] = bundle["handoffs"]
        if include_evidence:
            out["evidence_log"] = bundle["evidence_log"]
            out["evidence_items"] = bundle["evidence_items"]
        if include_states:
            out["states"] = redact_payload(bundle["states"], redact_candidate_text=redact_text)
        if include_raw_model_outputs:
            out["raw_model_outputs"] = [
                _model_call_view(call, redact_candidate_text=redact_text)
                for call in bundle["model_calls"]
            ]
        return out


# --------------------------------------------------------------------------- sql
class SqlRepository:
    """PostgreSQL-backed repository."""

    def __init__(self, session_factory, config: AppConfig | None = None) -> None:
        self._sf = session_factory
        #: Resolved config, used for redaction (R12.3) and raw-response retention.
        #: ``AppService`` injects it when the repository was built without one.
        self.config = config

    def versions(self) -> dict:
        """Return the server DB version and the current schema/migration version.

        ``db_version`` is the server's ``SELECT version()`` string (PostgreSQL-
        specific), captured best-effort: on a non-PostgreSQL or unavailable DB the
        call is wrapped in try/except and yields ``None``. ``migration_version`` is
        the recorded ``SchemaVersion`` row's version, falling back to the current
        target ``migrations.CURRENT_SCHEMA_VERSION`` when no row is present.
        """
        from sqlalchemy import text

        from . import migrations
        from .models import SchemaVersion

        db_version: str | None = None
        migration_version: int | None = migrations.CURRENT_SCHEMA_VERSION
        with self._sf() as s:
            try:
                db_version = s.execute(text("SELECT version()")).scalar_one_or_none()
            except Exception:  # noqa: BLE001 - non-PG / unavailable DB yields None
                db_version = None
            row = s.get(SchemaVersion, 1)
            if row is not None:
                migration_version = row.version
        return {"db_version": db_version, "migration_version": migration_version}

    def upsert_candidate_state(self, state: CandidateState) -> None:
        from .models import Candidate, CandidateStateVersion

        with self._sf() as s:
            cand = s.get(Candidate, state.candidate_id)
            if cand is None:
                s.add(Candidate(candidate_id=state.candidate_id, created_at=utcnow(),
                                latest_version=state.version))
            else:
                cand.latest_version = max(cand.latest_version, state.version)
            if s.get(CandidateStateVersion, (state.candidate_id, state.version)) is None:
                s.add(CandidateStateVersion(
                    candidate_id=state.candidate_id, version=state.version,
                    updated_at=state.updated_at, payload=state.model_dump(mode="json")))
            s.commit()

    def get_candidate_state(self, candidate_id: str, version: int | None = None) -> CandidateState | None:

        from .models import Candidate, CandidateStateVersion

        with self._sf() as s:
            if version is None:
                cand = s.get(Candidate, candidate_id)
                if cand is None:
                    return None
                version = cand.latest_version
            row = s.get(CandidateStateVersion, (candidate_id, version))
            return CandidateState.model_validate(row.payload) if row else None

    def create_session(self, session_id: str, candidate_id: str, variant: str) -> None:
        from .models import Session

        with self._sf() as s:
            if s.get(Session, session_id) is None:
                s.add(Session(session_id=session_id, candidate_id=candidate_id,
                              experiment_variant=variant, created_at=utcnow()))
                s.commit()

    def get_session(self, session_id: str) -> dict | None:
        from .models import Session

        with self._sf() as s:
            row = s.get(Session, session_id)
            if row is None:
                return None
            return {"session_id": row.session_id, "candidate_id": row.candidate_id,
                    "experiment_variant": row.experiment_variant,
                    "created_at": row.created_at.isoformat()}

    def get_latest_dialogue_state(self, session_id: str) -> DialogueState | None:
        from sqlalchemy import desc, select

        from .models import DialogueStateVersion

        with self._sf() as s:
            row = s.execute(
                select(DialogueStateVersion).where(DialogueStateVersion.session_id == session_id)
                .order_by(desc(DialogueStateVersion.version)).limit(1)
            ).scalar_one_or_none()
            return DialogueState.model_validate(row.payload) if row else None

    def save_turn(self, turn: TurnResult, evidence_items: list, model_calls: list) -> None:
        from .models import (
            AgentHandoffRow,
            DialogueStateVersion,
            EvidenceItemRow,
            EvidenceLogRow,
            ModelCallRow,
            RecommendationDecisionRow,
            ResponseClaimRow,
            ResponseRow,
            RunRecordRow,
        )

        self.upsert_candidate_state(turn.candidate_state)
        ds = turn.dialogue_state
        rr = turn.run_record
        with self._sf() as s:
            if s.get(DialogueStateVersion, (ds.session_id, ds.version)) is None:
                s.add(DialogueStateVersion(session_id=ds.session_id, version=ds.version,
                                           candidate_id=ds.candidate_id, payload=ds.model_dump(mode="json")))
            for item in evidence_items:
                if s.get(EvidenceItemRow, item.evidence_id) is None:
                    s.add(EvidenceItemRow(evidence_id=item.evidence_id, source=item.source.value,
                                          field_name=item.field_name, payload=item.model_dump(mode="json")))
            if turn.decision is not None:
                d = turn.decision
                s.merge(RecommendationDecisionRow(
                    decision_id=d.decision_id, run_id=rr.run_id, session_id=d.session_id,
                    experiment_variant=d.experiment_variant, no_match=d.no_match,
                    selected_count=len(d.selected_job_ids), payload=d.model_dump(mode="json")))
            for h in turn.handoffs:
                s.merge(AgentHandoffRow(handoff_id=h.handoff_id, run_id=h.run_id,
                        from_component=h.from_component, to_component=h.to_component,
                        validation_passed=h.validation_passed, status=h.status,
                        payload=h.model_dump(mode="json")))
            for log in turn.evidence_log:
                s.merge(EvidenceLogRow(log_id=log.log_id, run_id=log.run_id, stage=log.stage,
                        event_type=log.event_type, status=log.status, payload=log.model_dump(mode="json")))
            seen_claims: set[str] = set()
            for claim in turn.response.claims:
                if claim.claim_id in seen_claims:
                    continue
                seen_claims.add(claim.claim_id)
                s.merge(ResponseClaimRow(claim_id=claim.claim_id, run_id=rr.run_id,
                        claim_type=claim.claim_type, support_status=claim.support_status,
                        payload=claim.model_dump(mode="json")))
            resp = turn.response
            s.merge(ResponseRow(response_id=resp.response_id, run_id=rr.run_id, session_id=resp.session_id,
                    response_type=resp.response_type, message=resp.message, payload=resp.model_dump(mode="json")))
            for call in model_calls:
                s.merge(ModelCallRow(call_id=call.call_id, run_id=rr.run_id, purpose=call.purpose,
                        provider=call.provider, model=call.model,
                        payload=self._model_call_payload(call)))
            s.merge(RunRecordRow(
                run_id=rr.run_id, scenario_id=rr.scenario_id, session_id=rr.session_id,
                candidate_id=rr.candidate_id, experiment_variant=rr.experiment_variant,
                success=rr.success, failure_code=rr.failure_code, total_latency_ms=rr.total_latency_ms,
                config_hash=rr.config_hash, catalog_hash=rr.catalog_hash, prompt_hash=rr.prompt_hash,
                payload=rr.model_dump(mode="json")))
            s.commit()

    def _model_call_payload(self, call) -> dict[str, Any]:
        """The stored payload for one model call.

        Prompts are never persisted. The raw response is stored only while
        ``config.llm.save_raw_responses`` is on, and is redacted on the way in so
        the database never holds credentials, PII or (when
        ``config.logging.redact_candidate_text`` is on) candidate free text.
        """
        payload: dict[str, Any] = {
            "call_id": call.call_id,
            "purpose": call.purpose,
            "provider": call.provider,
            "model": call.model,
            "parsed_ok": getattr(call, "parsed_ok", None),
            "latency_ms": call.latency_ms,
        }
        if self._save_raw_responses:
            payload["raw_response"] = redact(
                getattr(call, "raw_response", "") or "",
                redact_candidate_text=_redact_candidate_text(self.config),
            )
        return payload

    @property
    def _save_raw_responses(self) -> bool:
        """Whether raw model responses are retained (``config.llm.save_raw_responses``)."""
        return bool(self.config.llm.save_raw_responses) if self.config is not None else True

    def get_run(self, run_id: str, include_states: bool = False, include_evidence: bool = False,
                include_handoffs: bool = False, include_raw_model_outputs: bool = False) -> dict | None:
        from sqlalchemy import select

        from .models import (
            AgentHandoffRow,
            EvidenceLogRow,
            ModelCallRow,
            RecommendationDecisionRow,
            ResponseRow,
            RunRecordRow,
        )

        redact_text = _redact_candidate_text(self.config)
        with self._sf() as s:
            rr = s.get(RunRecordRow, run_id)
            if rr is None:
                return None
            out: dict[str, Any] = {"run_record": rr.payload}
            resp = s.execute(select(ResponseRow).where(ResponseRow.run_id == run_id)).scalars().first()
            if resp:
                out["response"] = resp.payload
            dec = s.execute(select(RecommendationDecisionRow).where(
                RecommendationDecisionRow.run_id == run_id)).scalars().first()
            if dec:
                out["decision"] = dec.payload
            if include_handoffs:
                out["handoffs"] = [h.payload for h in s.execute(
                    select(AgentHandoffRow).where(AgentHandoffRow.run_id == run_id)).scalars()]
            if include_evidence:
                out["evidence_log"] = [e.payload for e in s.execute(
                    select(EvidenceLogRow).where(EvidenceLogRow.run_id == run_id)).scalars()]
            if include_states:
                out["states"] = redact_payload(
                    self._load_states(s, rr), redact_candidate_text=redact_text)
            if include_raw_model_outputs:
                out["raw_model_outputs"] = [
                    _model_call_view(_model_call_row_fields(m), redact_candidate_text=redact_text)
                    for m in s.execute(
                        select(ModelCallRow).where(ModelCallRow.run_id == run_id)).scalars()
                ]
            return out

    def _load_states(self, s, rr) -> dict[str, Any]:
        """Load the CandidateState/DialogueState versions this run was produced from (R12.1).

        ``RunRecord.state_object_ids`` pins the exact versions (``"<id>:v<N>"``);
        when that pin is missing or its row was never written, the latest version
        for the run's candidate/session is used instead.
        """
        from .models import CandidateStateVersion, DialogueStateVersion

        state_ids = rr.payload.get("state_object_ids") or {} if isinstance(rr.payload, dict) else {}
        candidate = self._state_row(
            s, CandidateStateVersion, CandidateStateVersion.candidate_id, rr.candidate_id,
            _state_version(state_ids.get("candidate_state")))
        dialogue = self._state_row(
            s, DialogueStateVersion, DialogueStateVersion.session_id, rr.session_id,
            _state_version(state_ids.get("dialogue_state")))
        states: dict[str, Any] = {
            "candidate_state": candidate.payload if candidate is not None else None,
            "dialogue_state": dialogue.payload if dialogue is not None else None,
        }
        active_search_id = state_ids.get("active_search_state")
        if active_search_id:
            # ActiveSearchState is per-search and re-derived; only its identity persists.
            states["active_search_id"] = active_search_id
        return states

    @staticmethod
    def _state_row(s, model, key_column, key_value: str, version: int | None):
        """Fetch one state-version row: the pinned version, else the latest one."""
        from sqlalchemy import desc, select

        if version is not None:
            row = s.get(model, (key_value, version))
            if row is not None:
                return row
        return s.execute(
            select(model).where(key_column == key_value)
            .order_by(desc(model.version)).limit(1)
        ).scalars().first()


def _bundle(turn: TurnResult, evidence_items: list, model_calls: list) -> dict:
    return {
        "run_record": turn.run_record.model_dump(mode="json"),
        "response": turn.response.model_dump(mode="json"),
        "decision": turn.decision.model_dump(mode="json") if turn.decision else None,
        "handoffs": [h.model_dump(mode="json") for h in turn.handoffs],
        "evidence_log": [e.model_dump(mode="json") for e in turn.evidence_log],
        "evidence_items": [e.model_dump(mode="json") for e in evidence_items],
        "states": {
            "candidate_state": turn.candidate_state.model_dump(mode="json"),
            "dialogue_state": turn.dialogue_state.model_dump(mode="json"),
        },
        "model_calls": [c.__dict__ for c in model_calls],
    }


def _redact_candidate_text(config: AppConfig | None) -> bool:
    """Resolve ``config.logging.redact_candidate_text`` (defaults to off)."""
    return bool(config.logging.redact_candidate_text) if config is not None else False


def _model_call_row_fields(row) -> dict:
    """Flatten a ``ModelCallRow`` into its payload overlaid with its columns."""
    payload = row.payload if isinstance(row.payload, dict) else {}
    fields = dict(payload)
    fields.update({"call_id": row.call_id, "purpose": row.purpose,
                   "provider": row.provider, "model": row.model})
    return fields


def _model_call_view(call: dict, *, redact_candidate_text: bool) -> dict:
    """Build the redacted raw-model-output row returned by ``get_run`` (R12.2, R12.3).

    Keeps the identifying/structural fields plus the model's raw response, and
    NEVER returns the prompt (prompts are excluded from every export path).
    """
    view = {key: value for key, value in call.items() if key != "prompt"}
    return redact_payload(view, redact_candidate_text=redact_candidate_text)


def _state_version(state_object_id: str | None) -> int | None:
    """Parse the version out of a ``"<id>:v<N>"`` state object id."""
    if not state_object_id or ":v" not in state_object_id:
        return None
    try:
        return int(state_object_id.rsplit(":v", 1)[1])
    except ValueError:
        return None
