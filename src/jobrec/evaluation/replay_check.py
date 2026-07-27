"""Artifact replay and deterministic recomputation (R18).

A saved run bundle is only *provably* reproducible if re-running it from its own
recorded inputs lands on exactly the same intermediate decisions. This module
does that: it takes a run bundle written by
:func:`jobrec.evaluation.exporters.write_run_bundle`, re-executes the recorded
turn in :attr:`~jobrec.domain.enums.RunMode.REPLAY` mode against the saved
``model_calls.jsonl`` (through the existing
:class:`~jobrec.llm.replay.ReplayProvider`), and compares five key-state hashes
against the ones recomputed from the original artifacts (R18.1, R18.2):

==================== ====================================================
key state            recomputed from
==================== ====================================================
``extracted_slots``  ``extracted_preferences.json`` -- per-field value,
                     polarity, strength, confirmation, scope, plus the
                     ambiguous fields
``state_versions``   ``run_record.json`` ``state_object_ids`` -- the
                     candidate / dialogue / active-search state versions
``filtered_jobs``    ``eligibility_results.json`` -- per-job eligibility,
                     hard-violation and unknown counts, reason codes
``ranking_output``   ``recommendation_decision.json`` -- ranked order,
                     scores, per-feature contributions, selection, no-match
``explanation_claims`` ``response_claims.json`` -- claim id, type, text,
                     evidence ids and support status
==================== ====================================================

Every comparison is written to ``replay_diff.json`` (R18.3), which records each
key state whose recomputed hash differs from the original, naming the run and the
key state (R18.4). The report is deterministic: runs and differences are sorted
by path and no timestamp or host detail is recorded, so replaying an unchanged
tree twice produces byte-identical output.

Regenerating statistics and reports from saved artifacts (the other half of
R18.1) already works through the ``jobrec_eval`` loaders, which read the same
bundles; this module adds the recomputation check on top of them.

**How a turn is replayed.** A bundle records the inputs of the turn it describes:
``candidate_state_before.json`` is that turn's incoming candidate state and
``dialogue_state.json`` is the dialogue state *after* the turn. The pre-turn
dialogue state is therefore reconstructed by dropping the final (current) turn and
rewinding the version by the two bumps a turn applies (one when the turn is
appended, one when the CMJCC persists its merge). The recorded ``session_id`` is
reused so every content-addressed id -- turn ids, evidence ids, the run id --
reproduces exactly. Prior-turn utterances are still present in the reconstructed
history, so a multi-turn run's memory continuation is recomputed rather than
assumed.

**Deterministic runs record no model calls**, so their ``model_calls.jsonl`` is
empty and the replay provider has nothing to serve. Extraction then follows the
documented fallback to the same rule extractor the original run used, which
reproduces identical slots; the "falling back to rule extractor" warnings logged
during such a replay are expected, not a divergence.

**What is intentionally excluded from the hashes.** Wall-clock fields
(``created_at``, latencies), the provider manifest and the per-preference
extraction provenance (``metadata.extraction_method`` / ``extraction_source``)
are not hashed: they describe *how* a value was produced, which legitimately
differs between an original call and its replay, not *what* the value is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..catalog import catalog_hash, load_catalog
from ..config import AppConfig
from ..domain.candidate import CandidateState
from ..domain.dialogue import DialogueState
from ..domain.enums import EvidenceSource, PersistenceScope, RunMode
from ..evidence_store import EvidenceStore
from ..utils.hashing import stable_hash

__all__ = [
    "KEY_STATES",
    "REPLAY_DIFF_FILENAME",
    "KeyStateDifference",
    "ReplayReport",
    "ReplayRunResult",
    "key_state_hashes",
    "key_state_views",
    "recorded_key_states",
    "replay_experiment",
    "replay_run",
    "write_replay_diff",
]

#: The five key states compared between the original run and its replay (R18.2).
KEY_STATES: tuple[str, ...] = (
    "extracted_slots",
    "state_versions",
    "filtered_jobs",
    "ranking_output",
    "explanation_claims",
)

#: Name of the replay diff report (R18.3).
REPLAY_DIFF_FILENAME = "replay_diff.json"

#: Catalog snapshot copied into an experiment directory by the runner.
_CATALOG_SNAPSHOT = "catalog.jsonl"

#: Dialogue-state version bumps applied by a single turn: one by
#: ``MemoryAgent.append_turn``, one by the CMJCC when it persists its merge.
_VERSION_BUMPS_PER_TURN = 2

#: Decimal places used when hashing floating-point scores, so a hash difference
#: means a real scoring difference rather than last-bit representation noise.
_SCORE_PRECISION = 6


# ------------------------------------------------------------------- findings
@dataclass(frozen=True)
class KeyStateDifference:
    """One key state whose recomputed hash differs from the original (R18.4)."""

    run_dir: str
    run_id: str
    key_state: str
    original: str | None
    recomputed: str | None

    def describe(self) -> str:
        """Human-readable one-liner naming the run and the key state."""
        return (
            f"{self.run_dir} [{self.run_id}]: {self.key_state} differs "
            f"(original {self.original}, replay {self.recomputed})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "key_state": self.key_state,
            "original": self.original,
            "recomputed": self.recomputed,
        }


@dataclass(frozen=True)
class ReplayRunResult:
    """Outcome of replaying one run bundle."""

    run_dir: str
    run_id: str
    variant: str | None
    scenario_id: str | None
    status: str
    original: dict[str, str]
    recomputed: dict[str, str]
    differences: tuple[KeyStateDifference, ...] = ()
    error: str | None = None

    @property
    def identical(self) -> bool:
        """True when the run replayed and every key-state hash matched."""
        return self.status == "ok" and not self.differences

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "variant": self.variant,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "identical": self.identical,
            "error": self.error,
            "original": dict(self.original),
            "recomputed": dict(self.recomputed),
            "differences": [d.to_dict() for d in self.differences],
        }


@dataclass(frozen=True)
class ReplayReport:
    """The replay diff report for a set of run bundles (R18.3)."""

    root: str
    runs: tuple[ReplayRunResult, ...]

    @property
    def identical(self) -> bool:
        """True when every run replayed and reproduced identical key states."""
        return bool(self.runs) and all(run.identical for run in self.runs)

    @property
    def differences(self) -> tuple[KeyStateDifference, ...]:
        """Every key-state difference across all replayed runs."""
        return tuple(diff for run in self.runs for diff in run.differences)

    @property
    def errors(self) -> tuple[ReplayRunResult, ...]:
        """Runs that could not be replayed at all."""
        return tuple(run for run in self.runs if run.status != "ok")

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-serialisable payload (no timestamps or host detail)."""
        return {
            "root": self.root,
            "key_states": list(KEY_STATES),
            "run_count": len(self.runs),
            "replayed_count": sum(1 for run in self.runs if run.status == "ok"),
            "identical": self.identical,
            "difference_count": len(self.differences),
            "runs": [run.to_dict() for run in self.runs],
            "differences": [diff.to_dict() for diff in self.differences],
            "errors": [
                {"run_dir": run.run_dir, "run_id": run.run_id, "error": run.error}
                for run in self.errors
            ],
        }

    def summary(self) -> str:
        """Multi-line summary listing every difference and unreplayable run."""
        head = (
            f"replay {'reproduced' if self.identical else 'DIVERGED from'} "
            f"{len(self.runs)} run(s) at {self.root}"
        )
        lines = [head]
        lines += [f"  difference: {diff.describe()}" for diff in self.differences]
        lines += [f"  not replayed: {run.run_dir}: {run.error}" for run in self.errors]
        return "\n".join(lines)


# ------------------------------------------------------------------ artifacts
def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _round(value: Any) -> Any:
    """Round floats to :data:`_SCORE_PRECISION`, pass anything else through."""
    return round(value, _SCORE_PRECISION) if isinstance(value, float) else value


def _slot_view(extracted: Any) -> dict[str, Any]:
    """Canonical view of the extracted slots (values, not provenance)."""
    payload = extracted if isinstance(extracted, dict) else {}
    preferences = payload.get("preferences") or []
    return {
        "slots": [
            {
                "field": pref.get("field_name"),
                "value": pref.get("normalized_value"),
                "polarity": pref.get("polarity"),
                "strength": pref.get("proposed_strength"),
                "confirmation": pref.get("confirmation_status"),
                "scope": pref.get("persistence_scope"),
                "temporal_scope": pref.get("temporal_scope"),
            }
            for pref in preferences
        ],
        "ambiguous_fields": sorted(payload.get("ambiguous_fields") or []),
    }


def _state_version_view(run_record: Any) -> dict[str, Any]:
    """Canonical view of the state versions carried by the run record."""
    payload = run_record if isinstance(run_record, dict) else {}
    return dict(payload.get("state_object_ids") or {})


def _filtered_view(eligibility: Any) -> list[dict[str, Any]]:
    """Canonical view of the filtering stage, in the order the jobs were checked."""
    rows = eligibility if isinstance(eligibility, list) else []
    return [
        {
            "job_id": row.get("job_id"),
            "eligible": row.get("eligible"),
            "hard_violation_count": row.get("hard_violation_count"),
            "unknown_hard_constraint_count": row.get("unknown_hard_constraint_count"),
            "filtered_reason_codes": sorted(row.get("filtered_reason_codes") or []),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _ranking_view(decision: Any) -> dict[str, Any]:
    """Canonical view of the ranking output, including feature contributions."""
    payload = decision if isinstance(decision, dict) else {}
    ranked = []
    for job in payload.get("ranked_jobs") or []:
        ranked.append({
            "job_id": job.get("job_id"),
            "rank": job.get("rank"),
            "total_score": _round(job.get("total_score")),
            "skill_gaps": list(job.get("skill_gaps") or []),
            "features": [
                {
                    "name": feature.get("name"),
                    "normalized_score": _round(feature.get("normalized_score")),
                    "weight": _round(feature.get("weight")),
                    "weighted_contribution": _round(feature.get("weighted_contribution")),
                    "explanation_code": feature.get("explanation_code"),
                }
                for feature in job.get("features") or []
            ],
        })
    return {
        "ranked_jobs": ranked,
        "selected_job_ids": list(payload.get("selected_job_ids") or []),
        "no_match": bool(payload.get("no_match", False)),
        "no_match_reason_codes": sorted(payload.get("no_match_reason_codes") or []),
    }


def _claims_view(claims: Any) -> list[dict[str, Any]]:
    """Canonical view of the grounded explanation claims."""
    rows = claims if isinstance(claims, list) else []
    return [
        {
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "text": claim.get("text"),
            "evidence_ids": list(claim.get("evidence_ids") or []),
            "support_status": claim.get("support_status"),
        }
        for claim in rows
        if isinstance(claim, dict)
    ]


def key_state_views(
    *,
    extracted_preferences: Any,
    run_record: Any,
    eligibility_results: Any,
    decision: Any,
    claims: Any,
) -> dict[str, Any]:
    """Build the canonical view of each key state from artifact payloads.

    Both sides of a comparison go through this one function -- the original from
    the saved JSON artifacts and the replay from the recomputed turn dumped to the
    same shape -- so the two hashes can only differ if the recomputed decision
    itself differs.
    """
    return {
        "extracted_slots": _slot_view(extracted_preferences),
        "state_versions": _state_version_view(run_record),
        "filtered_jobs": _filtered_view(eligibility_results),
        "ranking_output": _ranking_view(decision),
        "explanation_claims": _claims_view(claims),
    }


def key_state_hashes(views: dict[str, Any]) -> dict[str, str]:
    """Hash each canonical key-state view with the shared stable hasher."""
    return {name: stable_hash(views[name]) for name in KEY_STATES}


def recorded_key_states(run_dir: str | Path) -> dict[str, str]:
    """Key-state hashes of the ORIGINAL run, read from its saved bundle."""
    path = Path(run_dir)
    return key_state_hashes(key_state_views(
        extracted_preferences=_read_json(path / "extracted_preferences.json"),
        run_record=_read_json(path / "run_record.json"),
        eligibility_results=_read_json(path / "eligibility_results.json"),
        decision=_read_json(path / "recommendation_decision.json"),
        claims=_read_json(path / "response_claims.json"),
    ))


def _result_key_states(result: Any) -> dict[str, str]:
    """Key-state hashes of a freshly recomputed turn result."""
    decision = result.decision
    extracted = result.extracted_preferences
    return key_state_hashes(key_state_views(
        extracted_preferences=(
            extracted.model_dump(mode="json") if extracted is not None else None
        ),
        run_record=result.run_record.model_dump(mode="json"),
        eligibility_results=(
            [e.model_dump(mode="json") for e in decision.eligibility_results]
            if decision is not None else []
        ),
        decision=decision.model_dump(mode="json") if decision is not None else None,
        claims=[c.model_dump(mode="json") for c in result.response.claims],
    ))


# --------------------------------------------------------------- replay inputs
class ReplayInputError(ValueError):
    """Raised when a bundle does not carry the inputs needed to replay it."""


def _load_config(run_dir: Path) -> AppConfig:
    """Load the bundle's resolved config and switch it to replay mode."""
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        raise ReplayInputError(f"no resolved_config.yaml in {run_dir}")
    config = AppConfig.model_validate(yaml.safe_load(path.read_text()) or {})
    config.llm.mode = RunMode.REPLAY
    return config


def _find_catalog(run_dir: Path) -> Path:
    """Locate the catalog snapshot for a bundle by walking up to the experiment root."""
    for parent in [run_dir, *run_dir.parents]:
        candidate = parent / _CATALOG_SNAPSHOT
        if candidate.is_file():
            return candidate
    raise ReplayInputError(
        f"no {_CATALOG_SNAPSHOT} found at or above {run_dir}; pass catalog_path explicitly"
    )


def _catalog_snapshot_id(run_dir: Path, catalog_path: Path) -> str:
    """Recover the catalog snapshot id the original run used."""
    context = _read_json(run_dir / "job_context_state.json")
    if isinstance(context, dict) and context.get("catalog_snapshot_id"):
        return str(context["catalog_snapshot_id"])
    manifest = _read_json(catalog_path.parent / "catalog_manifest.json")
    if isinstance(manifest, dict) and manifest.get("catalog_snapshot_id"):
        return str(manifest["catalog_snapshot_id"])
    return "catalog-unknown"


def _pre_turn_dialogue(recorded: DialogueState) -> tuple[DialogueState, str]:
    """Rewind a recorded dialogue state to just before its final turn.

    Returns the pre-turn state and the utterance that produced the recorded one.
    Conflicts, unresolved slots and the active-search link are cleared because
    they are OUTPUTS the turn re-derives, not inputs it reads.
    """
    if not recorded.turns:
        raise ReplayInputError("recorded dialogue state has no turns to replay")
    version = recorded.version - _VERSION_BUMPS_PER_TURN
    if version < 1:
        raise ReplayInputError(
            f"recorded dialogue version {recorded.version} is too low to rewind one turn"
        )
    pre = recorded.model_copy(update={
        "version": version,
        "turns": list(recorded.turns[:-1]),
        "conflicts": [],
        "unresolved_slots": [],
        "active_search_id": None,
    })
    return pre, recorded.turns[-1].text


def _rehydrate_candidate_evidence(
    store: EvidenceStore, candidate: CandidateState, config: AppConfig
) -> None:
    """Re-register the evidence backing a candidate state into a fresh store.

    Evidence ids are content-addressed, so re-registering the same field/value
    reproduces the identical id. Profile values are restored through the memory
    agent; long-term values written back from an EARLIER turn's dialogue carry
    dialogue-sourced ids that the profile pass cannot reproduce, so those are
    re-registered too. Without this, claims grounded in remembered values would
    be dropped by the claim validator during replay and report a false difference.
    """
    from ..agents.memory_agent import MemoryAgent

    MemoryAgent(store, config).register_profile_evidence(candidate)

    for field, value in _written_back_values(candidate):
        if all(store.exists(eid) for eid in value.evidence_ids):
            continue
        store.register_field(
            EvidenceSource.DIALOGUE, candidate.candidate_id, field, value.value,
            confidence=value.confidence, confirmation=value.confirmation_status,
            scope=PersistenceScope.LONG_TERM,
        )


def _written_back_values(candidate: CandidateState) -> list[tuple[str, Any]]:
    """Candidate values keyed by the EXTRACTION field name that wrote them back.

    ``MemoryAgent.apply_confirmed_updates`` registers write-back evidence under the
    extracted preference's field name (e.g. ``skills_have``), while the state stores
    it under the corresponding attribute (``skills``). The agent's own field maps are
    inverted here so the re-registered id matches the recorded one exactly.
    """
    from ..agents.memory_agent import _LIST_FIELDS, _SCALAR_FIELDS

    pairs: list[tuple[str, Any]] = []
    for extraction_field, attribute in {**_LIST_FIELDS, **_SCALAR_FIELDS}.items():
        value = getattr(candidate, attribute, None)
        if isinstance(value, list):
            pairs.extend((extraction_field, item) for item in value)
        elif value is not None:
            pairs.append((extraction_field, value))
    return pairs


def _recompute(run_dir: Path, catalog_path: Path | None) -> Any:
    """Re-execute the bundle's turn in replay mode and return the turn result."""
    from ..orchestration.orchestrator import ConversationOrchestrator

    config = _load_config(run_dir)
    catalog = Path(catalog_path) if catalog_path else _find_catalog(run_dir)
    jobs = load_catalog(str(catalog))
    computed_catalog_hash = catalog_hash(jobs)

    run_record = _read_json(run_dir / "run_record.json")
    if not isinstance(run_record, dict):
        raise ReplayInputError(f"no run_record.json in {run_dir}")
    recorded_hash = run_record.get("catalog_hash")
    if recorded_hash and recorded_hash != computed_catalog_hash:
        raise ReplayInputError(
            f"catalog mismatch: run recorded {recorded_hash} but {catalog} hashes to "
            f"{computed_catalog_hash}"
        )

    dialogue_payload = _read_json(run_dir / "dialogue_state.json")
    if not isinstance(dialogue_payload, dict):
        raise ReplayInputError(f"no dialogue_state.json in {run_dir}")
    candidate_payload = _read_json(run_dir / "candidate_state_before.json")
    if not isinstance(candidate_payload, dict):
        raise ReplayInputError(f"no candidate_state_before.json in {run_dir}")

    pre_dialogue, utterance = _pre_turn_dialogue(
        DialogueState.model_validate(dialogue_payload)
    )
    candidate = CandidateState.model_validate(candidate_payload)

    from ..llm.replay import ReplayProvider

    store = EvidenceStore()
    orchestrator = ConversationOrchestrator(
        config, jobs, _catalog_snapshot_id(run_dir, catalog), computed_catalog_hash,
        provider=ReplayProvider(run_dir / "model_calls.jsonl"), store=store,
    )
    _rehydrate_candidate_evidence(store, candidate, config)
    return orchestrator.process_turn(
        candidate, pre_dialogue, utterance,
        scenario_id=run_record.get("scenario_id"),
    )


# ------------------------------------------------------------------ public API
def replay_run(
    run_dir: str | Path,
    *,
    catalog_path: str | Path | None = None,
    label: str | None = None,
) -> ReplayRunResult:
    """Replay one run bundle and compare its key-state hashes (R18.1, R18.2).

    Never raises for a bundle that cannot be replayed: the reason is returned on
    the result as ``status="error"`` so a batch report can record it.
    """
    path = Path(run_dir)
    name = label or path.as_posix()
    run_record = _read_json(path / "run_record.json") or {}
    run_id = str(run_record.get("run_id", ""))
    variant = run_record.get("experiment_variant")
    scenario_id = run_record.get("scenario_id")
    original = recorded_key_states(path)

    try:
        result = _recompute(path, Path(catalog_path) if catalog_path else None)
    except (ReplayInputError, OSError, ValueError) as exc:
        return ReplayRunResult(
            run_dir=name, run_id=run_id, variant=variant, scenario_id=scenario_id,
            status="error", original=original, recomputed={}, error=f"{type(exc).__name__}: {exc}",
        )

    recomputed = _result_key_states(result)
    differences = tuple(
        KeyStateDifference(
            run_dir=name, run_id=run_id, key_state=key,
            original=original.get(key), recomputed=recomputed.get(key),
        )
        for key in KEY_STATES
        if original.get(key) != recomputed.get(key)
    )
    return ReplayRunResult(
        run_dir=name, run_id=run_id, variant=variant, scenario_id=scenario_id,
        status="ok", original=original, recomputed=recomputed, differences=differences,
    )


def iter_run_dirs(root: str | Path) -> list[Path]:
    """Every run-bundle directory under ``root``, sorted for deterministic output."""
    base = Path(root)
    if (base / "run_record.json").is_file():
        return [base]
    return sorted(
        (path.parent for path in base.rglob("run_record.json")),
        key=lambda path: path.as_posix(),
    )


def replay_experiment(
    root: str | Path, *, catalog_path: str | Path | None = None
) -> ReplayReport:
    """Replay every run bundle under ``root`` and return the diff report (R18.3)."""
    base = Path(root)
    runs = tuple(
        replay_run(
            run_dir,
            catalog_path=catalog_path,
            label=(
                run_dir.relative_to(base).as_posix()
                if run_dir != base and base in run_dir.parents else run_dir.name
            ),
        )
        for run_dir in iter_run_dirs(base)
    )
    return ReplayReport(root=base.as_posix(), runs=runs)


def write_replay_diff(
    root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    out_path: str | Path | None = None,
    report: ReplayReport | None = None,
) -> Path:
    """Replay ``root`` and write ``replay_diff.json``, returning its path (R18.3).

    Pass an already-computed ``report`` to serialise it without replaying again.
    The payload records every differing key state with its run and both hashes
    (R18.4), and is byte-stable for an unchanged tree.
    """
    base = Path(root)
    report = report or replay_experiment(base, catalog_path=catalog_path)
    target = Path(out_path) if out_path else base / REPLAY_DIFF_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return target
