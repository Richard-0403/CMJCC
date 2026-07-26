"""Repositories: persist and retrieve states, decisions, runs and evidence.

Two implementations share one Protocol:
- ``SqlRepository``     : PostgreSQL-backed (production/default store).
- ``InMemoryRepository``: no database required (tests, offline demos). Selecting
  it is always explicit; the system never silently loses persistence.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..domain.candidate import CandidateState
from ..domain.dialogue import DialogueState
from ..orchestration.orchestrator import TurnResult
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

    def __init__(self) -> None:
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

    def get_run(self, run_id: str, **flags: bool) -> dict | None:
        return self.runs.get(run_id)


# --------------------------------------------------------------------------- sql
class SqlRepository:
    """PostgreSQL-backed repository."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

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
                        payload={"purpose": call.purpose, "latency_ms": call.latency_ms}))
            s.merge(RunRecordRow(
                run_id=rr.run_id, scenario_id=rr.scenario_id, session_id=rr.session_id,
                candidate_id=rr.candidate_id, experiment_variant=rr.experiment_variant,
                success=rr.success, failure_code=rr.failure_code, total_latency_ms=rr.total_latency_ms,
                config_hash=rr.config_hash, catalog_hash=rr.catalog_hash, prompt_hash=rr.prompt_hash,
                payload=rr.model_dump(mode="json")))
            s.commit()

    def get_run(self, run_id: str, include_states: bool = False, include_evidence: bool = False,
                include_handoffs: bool = False, include_raw_model_outputs: bool = False) -> dict | None:
        from sqlalchemy import select

        from .models import (
            AgentHandoffRow,
            EvidenceLogRow,
            RecommendationDecisionRow,
            ResponseRow,
            RunRecordRow,
        )

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
            return out


def _bundle(turn: TurnResult, evidence_items: list, model_calls: list) -> dict:
    return {
        "run_record": turn.run_record.model_dump(mode="json"),
        "response": turn.response.model_dump(mode="json"),
        "decision": turn.decision.model_dump(mode="json") if turn.decision else None,
        "handoffs": [h.model_dump(mode="json") for h in turn.handoffs],
        "evidence_log": [e.model_dump(mode="json") for e in turn.evidence_log],
        "evidence_items": [e.model_dump(mode="json") for e in evidence_items],
        "model_calls": [c.__dict__ for c in model_calls],
    }
