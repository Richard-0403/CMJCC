"""FastAPI application factory and routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from ..app_service import AppService
from .dependencies import get_service
from .schemas import (
    CreateCandidateRequest,
    CreateSessionRequest,
    TurnRequest,
    TurnResponse,
)


def create_app(service: AppService | None = None) -> FastAPI:
    app = FastAPI(title="CMJCC Conversational Job Recommendation", version="0.1.0")

    def svc() -> AppService:
        return service or get_service()

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> dict:
        s = svc()
        ready = len(s.jobs) > 0 and s.retriever_ready()
        if not ready:
            raise HTTPException(status_code=503, detail="catalog or index not ready")
        return {"status": "ready", "catalog_records": len(s.jobs)}

    @app.post("/v1/candidates")
    def create_candidate(req: CreateCandidateRequest) -> dict:
        state = svc().create_candidate(req.model_dump())
        return {"candidate_id": state.candidate_id, "version": state.version,
                "evidence_ids": _candidate_evidence_ids(state)}

    @app.post("/v1/sessions")
    def create_session(req: CreateSessionRequest) -> dict:
        if svc().get_candidate(req.candidate_id) is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        session_id = svc().create_session(req.candidate_id, req.experiment_variant)
        return {"session_id": session_id, "experiment_variant": req.experiment_variant}

    @app.post("/v1/sessions/{session_id}/turns", response_model=TurnResponse)
    def create_turn(session_id: str, req: TurnRequest) -> TurnResponse:
        try:
            result = svc().process_turn(session_id, req.text)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _to_turn_response(result)

    @app.get("/v1/runs/{run_id}")
    def get_run(
        run_id: str,
        include_states: bool = Query(False),
        include_evidence: bool = Query(False),
        include_handoffs: bool = Query(False),
        include_raw_model_outputs: bool = Query(False),
    ) -> dict:
        run = svc().get_run(run_id, include_states=include_states, include_evidence=include_evidence,
                            include_handoffs=include_handoffs,
                            include_raw_model_outputs=include_raw_model_outputs)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.post("/v1/runs/{run_id}/replay")
    def replay_run(run_id: str) -> dict:
        run = svc().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        # Replay uses stored decision/response; no remote model is called.
        return {"run_id": run_id, "replayed": True,
                "response": run.get("response"), "decision_no_match":
                (run.get("decision") or {}).get("no_match")}

    return app


def _candidate_evidence_ids(state) -> list[str]:
    ids: list[str] = []
    for pv in state.skills + state.target_roles + state.preferred_locations:
        ids.extend(pv.evidence_ids)
    for scalar in [state.salary_min, state.years_experience]:
        if scalar is not None:
            ids.extend(scalar.evidence_ids)
    return ids


def _to_turn_response(result) -> TurnResponse:
    decision = result.decision
    recs = []
    if decision is not None:
        by_rank = {rj.job_id: rj for rj in decision.ranked_jobs}
        for jid in decision.selected_job_ids:
            rj = by_rank.get(jid)
            if rj:
                recs.append({
                    "job_id": rj.job_id, "rank": rj.rank, "total_score": rj.total_score,
                    "skill_gaps": rj.skill_gaps,
                    "features": [{"name": f.name, "score": f.normalized_score,
                                  "weight": f.weight, "contribution": f.weighted_contribution,
                                  "code": f.explanation_code} for f in rj.features],
                })
    trace = {}
    if decision is not None:
        trace = {"retrieved": len(decision.retrieved_job_ids),
                 "eligible": sum(1 for e in decision.eligibility_results if e.eligible),
                 "returned": len(decision.selected_job_ids)}
    return TurnResponse(
        run_id=result.run_record.run_id,
        response_type=result.response.response_type,
        message=result.response.message,
        claims=[c.model_dump(mode="json") for c in result.response.claims],
        recommendations=recs,
        state_versions={"candidate": result.candidate_state.version,
                        "dialogue": result.dialogue_state.version},
        trace_summary=trace,
    )


app = create_app()
