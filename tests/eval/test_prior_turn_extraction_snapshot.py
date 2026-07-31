"""P0-2 (1): prior turns must be carried forward, not re-parsed.

The defect
----------
``Orchestrator._merge_prior_dialogue`` re-extracted EVERY earlier candidate utterance on
every turn, with ``self.rule_extractor``, tagged ``extraction_method="rule"``. In hybrid
mode that means only the CURRENT turn is answered by the model: from turn 2 onwards the
history is rule-derived, so a two-turn hybrid run's state is ``rule(turn1) + llm(turn2)``
rather than ``llm(turn1) + llm(turn2)``. The Hybrid-vs-deterministic comparison is supposed
to measure what model extraction adds, and on the 12 multi-turn scenarios of the
authoritative set it was measuring a mixture.

Re-parsing also fed the merged set straight into evidence construction, conflict detection
and the durable write-back, all of which are current-turn operations. So an utterance from
turn 1 was re-registered as evidence stamped with turn 2's id, re-checked for conflicts and
re-considered for write-back on every subsequent turn.

What is asserted here
---------------------
The fix separates two things that used to be one flat set:

* ``current_turn_extraction`` -- what this turn actually said. The only thing that may
  create evidence, raise conflicts or be written back to the profile.
* ``effective_dialogue_preferences`` -- the prior turns' stored snapshots plus the current
  turn, used only to build the search state. Prior entries keep their ORIGINAL strength,
  confirmation, metadata, evidence id and turn id.

The discriminator is a scripted provider whose turn-1 answer deliberately disagrees with
the rule extractor on three axes at once, so no single accidental match can make these
tests pass: the rule extractor reads "I prefer onsite work." as ``work_modes=onsite``,
SOFT, method ``rule``; the model is scripted to return ``remote``, HARD, method ``llm``.
No real LLM is involved.

Deliberately NOT asserted: a generic event reducer or state rebuilt after a process
restart. Both are out of scope for this batch and neither is exercised by the experiment --
0 of the 42 authoritative scenarios declare ``session_breaks``, and a session break creates
a new session inside the same process rather than restarting one.
"""

from __future__ import annotations

import pytest

from jobrec import app_service as app_service_module
from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import EvidenceSource, RunMode
from jobrec.llm.provider import LLMCallRecord

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"

#: Turn 1 text. The rule extractor reads this as work_modes=onsite, SOFT.
TURN1 = "I prefer onsite work."
#: Turn 2 text. Names a different field, so nothing about turn 1 is legitimately restated
#: and any second work_modes evidence item is a duplicate rather than a new statement.
TURN2 = "Data analyst please."
TURN3 = "At least RM4000."

#: What the scripted model returns for turn 1: a different VALUE and a stronger STRENGTH
#: than the rule extractor would produce for the same sentence.
LLM_TURN1_VALUE = "remote"
LLM_TURN1_STRENGTH = "hard"
#: What the rule extractor produces for TURN1, i.e. what re-parsing would substitute.
RULE_TURN1_VALUE = "onsite"


class ScriptedProvider:
    """Replays fixed extraction payloads; records every call for assertions."""

    name = "scripted"
    model = "scripted-v1"

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self.json_calls = 0
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        payload = self._payloads[min(self.json_calls, len(self._payloads) - 1)]
        self.json_calls += 1
        self.prompts.append(prompt)
        return payload, LLMCallRecord(
            call_id=f"call-{self.json_calls}", purpose=purpose, prompt=prompt,
            raw_response="<scripted>", parsed_ok=True, latency_ms=0.0,
            provider=self.name, model=self.model,
        )

    def complete_text(self, prompt: str, *, purpose: str,
                      fallback: str = "") -> tuple[str, LLMCallRecord]:
        return fallback, LLMCallRecord(
            call_id="call-text", purpose=purpose, prompt=prompt, raw_response=fallback,
            parsed_ok=True, latency_ms=0.0, provider=self.name, model=self.model,
        )

    def manifest(self) -> dict:
        return {"provider": self.name, "model": self.model, "mode": "scripted"}


def _pref(field: str, value, strength: str = "soft", raw: str = "scripted") -> dict:
    return {"field_name": field, "normalized_value": value, "raw_text": raw,
            "proposed_strength": strength, "polarity": "positive"}


def _payload(*prefs: dict) -> dict:
    return {"preferences": list(prefs)}


#: Turn 1 disagrees with the rule extractor; turn 2 and 3 add unrelated fields.
SCRIPT = [
    _payload(_pref("work_modes", LLM_TURN1_VALUE, LLM_TURN1_STRENGTH, TURN1)),
    _payload(_pref("target_roles", "data analyst", "soft", TURN2)),
    _payload(_pref("salary_min", 4000, "hard", TURN3)),
]


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, base_dir="configs")


def _hybrid(monkeypatch, config, turns: list[str], variant: str = "full",
            script: list[dict] | None = None):
    """Drive ``turns`` on one hybrid session backed by the scripted provider.

    Returns ``(results, store, provider)``. The store is the session's own evidence store,
    so every registered EvidenceItem is inspectable.
    """
    provider = ScriptedProvider(script if script is not None else SCRIPT)
    monkeypatch.setattr(app_service_module, "make_provider",
                        lambda cfg, replay_path=None: provider)
    cfg = config.model_copy(deep=True)
    cfg.llm.mode = RunMode.HYBRID
    # Not "remote": AppService refuses hybrid+remote without an API key, and this test
    # must never depend on a credential being present.
    cfg.llm.provider = "mock"
    service = AppService(cfg, CATALOG)
    candidate = service.create_candidate({"candidate_id": "snap-cand", "skills": []})
    session = service.create_session(candidate.candidate_id, variant)
    results = [service.process_turn(session, text, scenario_id="snap") for text in turns]
    _orch, store = service._sessions[session]
    return results, store, provider


def _modes(result) -> list[str]:
    return [str(v).casefold() for v in (result.active_search_state.work_modes or [])]


def _hard(result) -> set[str]:
    return set(result.active_search_state.hard_constraint_fields or [])


def _candidate_turns(result):
    return [t for t in result.dialogue_state.turns if t.speaker == "candidate"]


# ------------------------------------------------- the model's turn-1 result survives
def test_the_models_turn_one_value_is_not_replaced_by_the_rule_extractor(monkeypatch,
                                                                        config):
    """The core defect. Turn 2 must not re-read turn 1 with the rule extractor."""
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])

    assert LLM_TURN1_VALUE in _modes(results[0]), "turn 1 did not use the model's answer"
    assert LLM_TURN1_VALUE in _modes(results[1]), (
        "turn 1's work mode was re-derived by the rule extractor on turn 2")
    assert RULE_TURN1_VALUE not in _modes(results[1]), (
        "the rule extractor's reading of turn 1 leaked into the turn-2 state")


def test_the_models_turn_one_strength_survives_the_next_turn(monkeypatch, config):
    """Strength is carried forward from the snapshot, not recomputed from the text."""
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])

    assert "work_modes" in _hard(results[0])
    assert "work_modes" in _hard(results[1]), (
        "a hard work mode from turn 1 was downgraded by re-parsing on turn 2")


def test_the_models_turn_one_provenance_stays_llm(monkeypatch, config):
    """``extraction_method`` must still attribute turn 1 to the model on turn 2.

    Without this the run bundle claims a hybrid multi-turn state was rule-derived, which is
    what made the Hybrid arm's provenance untrustworthy on multi-turn scenarios.
    """
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])

    snapshot = _candidate_turns(results[1])[0].extraction_snapshot
    assert snapshot is not None, "turn 1 stored no extraction snapshot"
    methods = {p.field_name: p.metadata.get("extraction_method")
               for p in snapshot.preferences}
    assert methods.get("work_modes") == "llm", methods


def test_the_model_is_asked_once_per_turn_and_never_re_asked_for_history(monkeypatch,
                                                                        config):
    """Carrying a snapshot forward must not be implemented by re-calling the model."""
    _results, _store, provider = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3])
    assert provider.json_calls == 3, provider.json_calls


# --------------------------------------------------------------- evidence provenance
def test_turn_one_evidence_keeps_pointing_at_turn_one(monkeypatch, config):
    results, store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])
    turns = _candidate_turns(results[1])
    turn1_id, turn2_id = turns[0].turn_id, turns[1].turn_id

    work_mode_items = [i for i in store.all() if i.field_name == "work_modes"]
    assert work_mode_items, "no work_modes evidence was registered at all"
    assert {i.turn_id for i in work_mode_items} == {turn1_id}, (
        "turn 1's work mode evidence was re-stamped onto a later turn")
    assert all(i.turn_id != turn2_id for i in work_mode_items)


def test_the_second_turn_does_not_duplicate_the_first_turns_evidence(monkeypatch, config):
    """One statement, one EvidenceItem -- however many turns follow it."""
    results, store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3])

    work_mode_items = [i for i in store.all() if i.field_name == "work_modes"]
    assert len(work_mode_items) == 1, (
        f"turn 1's single work_modes statement produced {len(work_mode_items)} evidence "
        "items; re-parsing registered it again on every later turn")
    # Nor did anything else the DIALOGUE produced get duplicated. Scoped to dialogue
    # evidence on purpose: the same store also holds catalog evidence, where many jobs
    # legitimately share a title or a role family.
    seen: dict[tuple, int] = {}
    for item in store.all():
        if item.source != EvidenceSource.DIALOGUE:
            continue
        key = (item.field_name, str(item.normalized_value))
        seen[key] = seen.get(key, 0) + 1
    assert not [k for k, n in seen.items() if n > 1], seen
    # Each of the three turns stated exactly one distinct field, so three items is the
    # complete, non-duplicated total.
    assert sum(seen.values()) == 3, seen
    assert results  # the run really did complete


def test_each_turn_records_only_its_own_evidence(monkeypatch, config):
    results, store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3])
    turns = _candidate_turns(results[2])
    assert len(turns) == 3

    by_id = {i.evidence_id: i for i in store.all()}
    for turn in turns:
        assert turn.evidence_ids, f"turn {turn.turn_index} recorded no evidence"
        for eid in turn.evidence_ids:
            assert eid in by_id, f"turn {turn.turn_index} cites unknown evidence {eid}"
            assert by_id[eid].turn_id == turn.turn_id, (
                f"turn {turn.turn_index} claims evidence belonging to "
                f"{by_id[eid].turn_id}")
    # No evidence id appears on two different turns.
    all_ids = [eid for t in turns for eid in t.evidence_ids]
    assert len(all_ids) == len(set(all_ids)), all_ids


def test_the_search_state_cites_the_original_evidence_id_for_a_prior_turn(monkeypatch,
                                                                         config):
    """Prior fields must resolve to the evidence the ORIGINAL turn created."""
    results, store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])
    turns = _candidate_turns(results[1])
    turn1_evidence = set(turns[0].evidence_ids)

    cited = set(results[1].active_search_state.field_evidence_map.get("work_modes", []))
    assert cited, "the turn-2 search state cites no evidence for the prior work mode"
    assert cited <= {i.evidence_id for i in store.all()}
    assert cited & turn1_evidence, (
        "the prior work mode cites evidence that turn 1 never created")


# ------------------------------------------- history is not reprocessed as current
def test_conflict_detection_does_not_reprocess_the_first_turn(monkeypatch, config):
    """A conflict is a property of what was just said, so it must not recur each turn."""
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3])

    # Conflicts accumulate on the DialogueState; the same field/turn pair must not be
    # raised twice just because a later turn re-walked history.
    conflicts = results[2].dialogue_state.conflicts
    ids = [c.conflict_id for c in conflicts]
    assert len(ids) == len(set(ids)), ids
    per_field = {}
    for c in conflicts:
        per_field[c.field_name] = per_field.get(c.field_name, 0) + 1
    assert not [f for f, n in per_field.items() if n > 1], per_field


def test_the_durable_write_back_does_not_reapply_the_first_turn(monkeypatch, config):
    """The profile version must not advance once per turn per historical preference.

    Re-parsing offered turn 1's preferences to ``apply_confirmed_updates`` again on every
    later turn, so a single durable statement could bump the candidate state repeatedly.
    """
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3])
    versions = [r.candidate_state.version for r in results]
    # Monotone, and never advanced by a turn that restated nothing durable.
    assert versions == sorted(versions), versions
    assert versions[2] - versions[1] <= 1, versions


# --------------------------------------------------------- ordering and overrides
def test_three_turns_keep_their_order_and_the_latest_scalar_wins(monkeypatch, config):
    """Prior-then-current ordering is what makes the current turn override a scalar."""
    script = [
        _payload(_pref("salary_min", 3000, "soft", "first")),
        _payload(_pref("target_roles", "data analyst", "soft", TURN2)),
        _payload(_pref("salary_min", 4000, "hard", TURN3)),
    ]
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2, TURN3],
                                  script=script)

    assert results[0].active_search_state.salary_min == 3000
    assert results[1].active_search_state.salary_min == 3000, (
        "an unrelated turn dropped the salary stated in turn 1")
    assert results[2].active_search_state.salary_min == 4000, (
        "the current turn failed to override the earlier scalar")
    assert "salary_min" in _hard(results[2])
    assert "data analyst" in [str(v).casefold()
                              for v in (results[2].active_search_state.target_roles or [])]


# ------------------------------------------------------------- variants and modes
def test_deterministic_multi_turn_state_is_unchanged(config):
    """The deterministic arm must be byte-comparable before and after this change.

    It never had the defect -- re-parsing used the rule extractor, which is the same
    extractor the deterministic path uses -- so its state is the regression guard.
    """
    service = AppService(config, CATALOG)
    candidate = service.create_candidate({"candidate_id": "det-cand", "skills": []})
    session = service.create_session(candidate.candidate_id, "full")
    results = [service.process_turn(session, t, scenario_id="det")
               for t in ["Onsite only.", TURN2, TURN3]]

    assert _modes(results[0]) == ["onsite"]
    assert _modes(results[2]) == ["onsite"]
    assert "work_modes" in _hard(results[2])
    assert "salary_min" in _hard(results[2])
    assert results[2].active_search_state.salary_min == 4000


@pytest.mark.parametrize("variant", ["no_memory", "one_shot"])
def test_variants_that_do_not_inherit_history_still_do_not(monkeypatch, config, variant):
    """The fix must not turn a non-inheriting variant into an inheriting one."""
    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2], variant=variant)

    assert LLM_TURN1_VALUE in _modes(results[0])
    assert not _modes(results[1]), (
        f"{variant} inherited turn 1's work mode, which is the memory condition it "
        "is defined by not having")


# ------------------------------------------------------- serialisation and legacy
def test_the_snapshot_survives_a_dialogue_state_round_trip(monkeypatch, config):
    """Persisted and re-read state must still carry the per-turn extraction.

    Not process-restart rehydration (out of scope): this is the narrower requirement that
    the snapshot is part of the serialised model rather than in-memory-only, so it is not
    lost by the repository round-trip the pipeline already performs.
    """
    from jobrec.domain.dialogue import DialogueState

    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])
    restored = DialogueState.model_validate(
        results[1].dialogue_state.model_dump(mode="json"))

    original = _candidate_turns(results[1])[0].extraction_snapshot
    round_tripped = [t for t in restored.turns if t.speaker == "candidate"][0]
    assert round_tripped.extraction_snapshot is not None, "the snapshot was lost"
    assert original is not None
    assert round_tripped.extraction_snapshot.model_dump(mode="json") == \
        original.model_dump(mode="json")


def test_a_prior_turn_without_a_snapshot_is_marked_not_silently_reparsed(monkeypatch,
                                                                        config):
    """A legacy DialogueState has no snapshots. Falling back must be auditable.

    The fallback exists so old dialogue can still be processed, but a fresh official run
    must never take it, so it is labelled rather than silent -- and
    ``tests/eval/test_no_legacy_reparse_in_official_runs.py`` is what turns the label into
    a gate failure.
    """
    from jobrec.orchestration.orchestrator import LEGACY_REPARSE_WARNING

    results, _store, _p = _hybrid(monkeypatch, config, [TURN1, TURN2])
    legacy = results[1].dialogue_state.model_copy(update={
        "turns": [t.model_copy(update={"extraction_snapshot": None})
                  for t in results[1].dialogue_state.turns],
    })

    service = AppService(config.model_copy(deep=True), CATALOG)
    orch, _store2 = service._orchestrator_for("legacy-session", "full")
    out = orch.process_turn(results[1].candidate_state, legacy, TURN3,
                            scenario_id="legacy")

    assert out.extracted_preferences is not None
    warnings = list(out.extracted_preferences.extraction_warnings)
    assert LEGACY_REPARSE_WARNING in warnings, warnings


# --------------------------------------------- the label is a gate, not a footnote
def test_an_official_run_refuses_to_complete_if_it_re_parsed_history(tmp_path,
                                                                    monkeypatch):
    """The legacy fallback must fail the batch, not be recorded and moved past.

    Simulated by forcing the fallback for every prior turn, which is what a run pipeline
    that had stopped carrying snapshots would look like. The check is that no
    ``experiment_manifest.json`` is produced: an incomplete directory can be re-run freely
    once the cause is fixed, whereas a manifest would make the batch look citable.
    """
    import json

    from jobrec.evaluation import experiment_runner as runner_module
    from jobrec.evaluation.experiment_identity import EXPERIMENT_MANIFEST_FILENAME
    from jobrec.evaluation.experiment_runner import ExperimentRunner, LegacyReparseError
    from jobrec.orchestration.orchestrator import ConversationOrchestrator

    scenario = {
        "scenario_id": "SC-LEGACY-01",
        "scenario_type": "multi_turn",
        "profile": {"candidate_id": "SC-LEGACY-01-cand", "skills": ["Python"],
                    "years_experience": 3},
        "turns": ["Onsite only.", TURN2],
        "expects": {"response_type": "recommendation"},
    }
    scenarios_path = tmp_path / "scenarios.jsonl"
    scenarios_path.write_text(json.dumps(scenario) + "\n", encoding="utf-8")

    # Drop every snapshot as the state is read back, which is exactly the legacy shape.
    original = ConversationOrchestrator._prior_dialogue_preferences

    def forced_legacy(self, dialogue_state):
        stripped = dialogue_state.model_copy(update={
            "turns": [t.model_copy(update={"extraction_snapshot": None})
                      for t in dialogue_state.turns],
        })
        return original(self, stripped)

    monkeypatch.setattr(ConversationOrchestrator, "_prior_dialogue_preferences",
                        forced_legacy)

    cfg = load_config(CONFIG, base_dir="configs")
    cfg.experiment.repeat_count = 1
    runner = ExperimentRunner(cfg, CATALOG, str(scenarios_path),
                              out_dir=str(tmp_path / "runs"))

    with pytest.raises(LegacyReparseError, match=runner_module.LEGACY_REPARSE_WARNING):
        runner.run(["full"])

    manifests = list((tmp_path / "runs").rglob(EXPERIMENT_MANIFEST_FILENAME))
    assert not manifests, "a batch that re-parsed history was recorded as complete"
