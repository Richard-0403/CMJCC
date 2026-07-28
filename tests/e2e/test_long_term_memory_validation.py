"""Experiment-level validation of long-term memory write-back and inheritance (R19.1).

Why this set exists
-------------------
The 210-run deterministic experiment recorded **zero** long-term-eligible preferences:
all 168 extracted preferences across its 56 turns resolved to ``active_search``, and no
scenario utterance carried a durable cue ("from now on ..."). The write-back mechanism
therefore had no archived evidence whatsoever, and it could not have had any: every run
builds its own :class:`~jobrec.app_service.AppService`, so a scenario could not span two
sessions, and INHERITANCE is only observable across a session boundary.

``tests/unit/test_memory_writeback.py`` already proves inheritance end-to-end through
``AppService``. What was missing is evidence in the experiment ARTIFACTS -- run bundles a
reader can inspect -- which is what this scenario set and these assertions produce.

The set is deliberately a SEPARATE file (``evaluation/data/scenarios_longterm.jsonl``).
Mixing it into the 42 main scenarios would change every reported denominator and make the
main experiment's numbers non-comparable with the ones already archived.

Design of the set
-----------------
Each positive case is paired with a control, so a difference cannot be an artifact of the
session-boundary mechanism itself:

* ``SC-LT-01`` durable statement, then a new session -> must be inherited.
* ``SC-LT-02`` the SAME shape with "this time only" -> must NOT be inherited (scope).
* ``SC-LT-03`` three sessions, two different fields -> must accumulate.
* ``SC-LT-04`` durable statement contradicting a profile value -> the R4.11 conflict
  guard, observed across a boundary.
* ``SC-LT-05`` two identical non-durable turns either side of a break -> the boundary
  alone must change nothing.

The ``full`` / ``no_memory`` contrast is the second control: the set must be SENSITIVE to
the mechanism it validates (the two inheritance cases collapse without persistent memory)
and INSENSITIVE elsewhere.

A known, characterised limitation
--------------------------------
:mod:`jobrec.evaluation.replay_check` replays each bundle in isolation, reconstructing the
turn from ``candidate_state_before.json`` plus that session's dialogue. For a run that
INHERITED long-term values, the inherited records cite evidence registered in an earlier
session, which the bundle does not contain. The consequence is pinned by
:func:`test_cross_session_runs_replay_identically_except_for_claim_evidence`: constraints,
filtering, ranking and state versions replay identically, and only ``explanation_claims``
differs. This is a property of session-scoped replay, not of the recommendation pipeline,
and it does not touch the main 42-scenario experiment, which contains no cross-session run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner

CONFIG = "configs/experiment_full.yaml"
CATALOG = "data/processed/jobs.jsonl"
SCENARIOS = "evaluation/data/scenarios_longterm.jsonl"

#: Variants the set is run under. ``no_memory`` resolves ``use_persistent_memory`` and
#: ``persist_confirmed_updates`` to False, so it is the ablation the set must detect.
_VARIANTS = ("full", "no_memory")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _active(state: dict, field: str) -> list:
    """Active (non-retired) values held for ``field`` in a candidate-state dump."""
    return [p["value"] for p in (state.get(field) or []) if p.get("is_active")]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict]:
    return {s["scenario_id"]: s
            for s in _read_jsonl(Path(SCENARIOS))}


@pytest.fixture(scope="module")
def runs(tmp_path_factory, scenarios) -> dict[tuple[str, str], dict]:
    """Every scenario run once per variant; returns ``{(variant, scenario_id): view}``."""
    root = tmp_path_factory.mktemp("long-term-memory")
    cfg = load_config(CONFIG, base_dir="configs")
    out: dict[tuple[str, str], dict] = {}
    for variant in _VARIANTS:
        runner = ExperimentRunner(cfg, CATALOG, SCENARIOS,
                                  out_dir=str(root / variant / "_runs"))
        for scenario_id, scenario in scenarios.items():
            row, failure = runner._run_one(
                variant, scenario, 0, root / variant / "exp")
            assert failure is None, (variant, scenario_id, failure)
            run_dir = Path(row["run_dir"])
            out[(variant, scenario_id)] = {
                "row": row,
                "turns": _read_jsonl(run_dir / "turn_records.jsonl"),
                "candidate_after": json.loads(
                    (run_dir / "candidate_state_after.json").read_text()),
                "active_search": json.loads(
                    (run_dir / "active_search_state.json").read_text()),
                "run_dir": run_dir,
            }
    return out


# --------------------------------------------------------- the runner mechanism
def test_session_breaks_start_new_sessions_and_are_archived(runs, scenarios) -> None:
    """Each declared break starts a new session, and the archive records which is which.

    Without the per-turn ``session_id`` there is no way to tell from a bundle where the
    boundary fell, so "this turn could only have known that through long-term memory"
    would be unverifiable offline.

    **Validates: Requirements 19.1, 7.3**
    """
    for scenario_id, scenario in scenarios.items():
        expected_sessions = len({0, *scenario.get("session_breaks", [])})
        for variant in _VARIANTS:
            view = runs[(variant, scenario_id)]
            assert view["row"]["session_count"] == expected_sessions, (
                variant, scenario_id)
            sessions = [t["session_id"] for t in view["turns"]]
            assert all(sessions), (variant, scenario_id)
            assert len(set(sessions)) == expected_sessions, (variant, scenario_id)
            # The boundary falls exactly where the scenario declared it.
            for index in scenario.get("session_breaks", []):
                assert sessions[index] != sessions[index - 1], (
                    variant, scenario_id, index)


def test_a_scenario_without_breaks_stays_single_session() -> None:
    """No declaration -> one session, and index 0 is ignored rather than wasted.

    This is what makes the feature safe for the 42-scenario main set: a scenario that
    does not declare breaks behaves exactly as before, so no archived result can move.

    **Validates: Requirements 19.1**
    """
    assert ExperimentRunner._session_breaks({}) == frozenset()
    assert ExperimentRunner._session_breaks({"session_breaks": []}) == frozenset()
    # Turn 0 already starts a fresh session; breaking there would only create an unused one.
    assert ExperimentRunner._session_breaks({"session_breaks": [0]}) == frozenset()
    assert ExperimentRunner._session_breaks({"session_breaks": [0, 2, 1]}) == frozenset({1, 2})


# ------------------------------------------------------- inheritance, positive case
def test_durable_statement_is_written_and_inherited_by_the_next_session(runs) -> None:
    """SC-LT-01: "from now on" survives a session boundary and shapes the new search.

    The second session has a fresh dialogue and its utterance never mentions a work
    mode, so long-term candidate memory is the ONLY channel through which ``hybrid``
    can reach its active search. This is the archived evidence the main experiment
    could not produce.

    **Validates: Requirements 19.1, 4.2, 4.5**
    """
    view = runs[("full", "SC-LT-01")]
    row, turns = view["row"], view["turns"]

    # A write-back happened: the candidate state advanced past its initial version.
    assert row["candidate_state_version"] == 2
    assert [t["candidate_state_version"] for t in turns] == [2, 2]
    assert "hybrid" in _active(view["candidate_after"], "work_modes")

    # And it was INHERITED: the second session's search carries it unprompted.
    assert "hybrid" in (view["active_search"].get("work_modes") or [])
    assert turns[0]["session_id"] != turns[1]["session_id"]
    # The run still produced a usable answer, so the inheritance is not incidental to a
    # degenerate outcome.
    assert row["response_type"] == "recommendation"
    assert row["returned"] > 0


def test_inheritance_collapses_without_persistent_memory(runs) -> None:
    """The same scenario under ``no_memory`` inherits nothing.

    The sensitivity control: if the set passed under both variants it would not be
    testing the mechanism at all.

    **Validates: Requirements 19.1, 5.3**
    """
    full = runs[("full", "SC-LT-01")]
    ablated = runs[("no_memory", "SC-LT-01")]

    assert ablated["row"]["candidate_state_version"] == 1
    assert _active(ablated["candidate_after"], "work_modes") == []
    assert "hybrid" not in (ablated["active_search"].get("work_modes") or [])
    # The two conditions genuinely differ on the mechanism under test.
    assert (ablated["row"]["candidate_state_version"]
            != full["row"]["candidate_state_version"])


# ------------------------------------------------------- inheritance, scope control
def test_current_search_scope_is_not_inherited(runs) -> None:
    """SC-LT-02: "this time only" leaves long-term memory untouched.

    Same shape as SC-LT-01, same boundary, opposite temporal cue -- so a difference
    between the two isolates scope resolution rather than the session mechanism.

    **Validates: Requirements 19.1, 4.5, 4.7**
    """
    view = runs[("full", "SC-LT-02")]
    assert view["row"]["candidate_state_version"] == 1
    assert _active(view["candidate_after"], "work_modes") == []
    assert "remote" not in (view["active_search"].get("work_modes") or [])
    assert view["active_search"].get("salary_min") != 8000.0
    # It is not that the run failed to do anything: it still answered.
    assert view["row"]["response_type"] == "recommendation"


def test_a_session_boundary_alone_changes_nothing(runs) -> None:
    """SC-LT-05: two identical non-durable turns either side of a break.

    The mechanism control. If merely opening a second session bumped the version or
    altered the outcome, every result above would be confounded.

    **Validates: Requirements 19.1**
    """
    for variant in _VARIANTS:
        view = runs[(variant, "SC-LT-05")]
        assert view["row"]["session_count"] == 2, variant
        assert view["row"]["candidate_state_version"] == 1, variant
        assert _active(view["candidate_after"], "work_modes") == [], variant
        assert view["row"]["response_type"] == "recommendation", variant
        assert view["row"]["returned"] > 0, variant


# ----------------------------------------------------------------- accumulation
def test_long_term_memory_accumulates_across_three_sessions(runs) -> None:
    """SC-LT-03: each write builds on the version it inherited, not on the profile.

    Two different fields written in two different sessions must BOTH be active in the
    third, which fails if a new session reloads the original profile instead of the
    latest persisted version.

    **Validates: Requirements 19.1, 4.2, 4.3**
    """
    view = runs[("full", "SC-LT-03")]
    row, turns = view["row"], view["turns"]

    assert row["session_count"] == 3
    # One version per durable write, each built on the previous one.
    assert [t["candidate_state_version"] for t in turns] == [2, 3, 3]
    assert row["candidate_state_version"] == 3

    assert "hybrid" in _active(view["candidate_after"], "work_modes")
    assert "Kuala Lumpur" in _active(view["candidate_after"], "preferred_locations")
    # The third session sees both, though its utterance mentions neither.
    search = view["active_search"]
    assert "hybrid" in (search.get("work_modes") or [])
    assert "Kuala Lumpur" in (search.get("preferred_locations") or [])

    ablated = runs[("no_memory", "SC-LT-03")]
    assert ablated["row"]["candidate_state_version"] == 1


# ------------------------------------------------------------- R4.11 conflict guard
def test_a_durable_correction_of_a_profile_value_is_refused_and_stays_refused(runs) -> None:
    """SC-LT-04: the conflict guard blocks the write, and the next session still sees the
    OLD value.

    This pins observed behaviour that is worth reporting rather than a success:
    ``PreferenceConflict.resolution`` has no ``override`` member, so R4.11's guard is
    total -- a candidate who says "from now on I want Kuala Lumpur instead" of the
    ``Penang`` on their profile gets NEITHER a long-term update NOR a carried-over
    search override. The correction is simply lost at the session boundary. The unit
    suite proves the guard fires; this is what that means for a user, in an artifact.

    **Validates: Requirements 4.11, 19.1**
    """
    view = runs[("full", "SC-LT-04")]

    # No write-back: the guard refused it.
    assert view["row"]["candidate_state_version"] == 1
    # The profile value is still the active long-term one ...
    assert _active(view["candidate_after"], "preferred_locations") == ["Penang"]
    # ... and it is what the NEXT session searches on, so the stated correction did not
    # survive in any form.
    assert view["active_search"].get("preferred_locations") == ["Penang"]
    assert "Kuala Lumpur" not in (view["active_search"].get("preferred_locations") or [])

    # Ablating memory changes nothing here: there was no write to ablate.
    ablated = runs[("no_memory", "SC-LT-04")]
    assert ablated["row"]["candidate_state_version"] == 1


# --------------------------------------------------------------- set-level property
def test_the_set_is_sensitive_only_where_the_mechanism_applies(runs, scenarios) -> None:
    """Exactly the two inheritance scenarios differ between ``full`` and ``no_memory``.

    A validation set that differed everywhere would be measuring something broader than
    write-back; one that differed nowhere would be measuring nothing.

    **Validates: Requirements 19.1, 5.3**
    """
    differing = {
        scenario_id for scenario_id in scenarios
        if (runs[("full", scenario_id)]["row"]["candidate_state_version"]
            != runs[("no_memory", scenario_id)]["row"]["candidate_state_version"])
    }
    assert differing == {"SC-LT-01", "SC-LT-03"}, differing

    # And every scenario in the set does cross a boundary, so none of them is silently
    # a single-session scenario that could not have tested inheritance at all.
    for scenario_id, scenario in scenarios.items():
        assert scenario.get("session_breaks"), scenario_id
        assert runs[("full", scenario_id)]["row"]["session_count"] > 1, scenario_id


# ------------------------------------------------------------- replay characterisation
#: Key states that must replay identically even for a cross-session run: everything that
#: determines WHAT was recommended. Excludes ``explanation_claims``, whose evidence ids
#: are minted in the session that registered the evidence.
_REPLAY_STABLE_KEY_STATES = ("extracted_slots", "state_versions", "filtered_jobs",
                             "ranking_output")


def test_cross_session_runs_replay_identically_except_for_claim_evidence(runs) -> None:
    """A replayed inheritance run reproduces the recommendation but not the claim ids.

    Pinned rather than fixed. Replay reconstructs a turn from its own bundle, and an
    inherited long-term value cites evidence registered in a session that bundle does not
    contain, so the recomputed claims carry different evidence ids. What matters for the
    thesis's reproducibility claim is that everything determining the ANSWER --
    extraction, state versions, constraint filtering and ranking -- still matches
    exactly; leaving this uncharacterised would turn a session-scoped replay limitation
    into an apparent non-reproducible pipeline.

    **Validates: Requirements 18.2, 18.4, 19.1**
    """
    from jobrec.evaluation.replay_check import replay_run

    for scenario_id in ("SC-LT-01", "SC-LT-03"):
        # ``catalog_path`` is explicit because these runs come from ``_run_one``, which
        # writes no ``catalog.jsonl`` snapshot (only a full ``runner.run()`` does).
        result = replay_run(runs[("full", scenario_id)]["run_dir"], catalog_path=CATALOG)
        assert result.status == "ok", (scenario_id, result.status, result.error)
        differing = {d.key_state for d in result.differences}
        assert differing == {"explanation_claims"}, (scenario_id, differing)
        for key in _REPLAY_STABLE_KEY_STATES:
            assert result.original[key] == result.recomputed[key], (scenario_id, key)


def test_single_session_runs_in_the_set_replay_identically(runs) -> None:
    """The scenarios that inherit nothing replay bit-identically, so the limitation is
    confined to inheritance rather than to the session-break feature.

    **Validates: Requirements 18.2, 19.1**
    """
    from jobrec.evaluation.replay_check import replay_run

    for scenario_id in ("SC-LT-02", "SC-LT-04", "SC-LT-05"):
        result = replay_run(runs[("full", scenario_id)]["run_dir"], catalog_path=CATALOG)
        assert result.status == "ok", (scenario_id, result.status, result.error)
        assert result.identical, (scenario_id, [d.key_state for d in result.differences])
