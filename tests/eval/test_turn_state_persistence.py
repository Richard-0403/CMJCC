"""Multi-turn state: what must survive P0-2, and what P0-2 still has to fix.

Two kinds of test live here on purpose.

The GUARDS pin OUTCOMES, not the mechanism that produces them. They were written while
``_merge_prior_dialogue`` still re-parsed every earlier utterance on each turn, to make sure
that whatever replaced it carried constraint STRENGTH forward: a change that stopped
re-parsing without preserving strength would let an earlier hard constraint quietly go
absent, which widens the candidate pool and surfaces as a metric shift rather than an error.
They did their job -- re-parsing is gone (``_prior_dialogue_preferences`` reads each turn's
stored ``extraction_snapshot``) and every guard here still passes unchanged.

The second group was the P0-2 entry criterion and failed when it was written: an explicit
relaxation could not downgrade a hard constraint, and ``DialogueTurn.evidence_ids`` was
empty on every turn. Both are fixed.

Current P0-2 status, because the previous version of this docstring outlived its facts and
was later cited as evidence that the re-parsing was still in place:

* DONE -- per-turn extraction snapshots replace re-parsing history; evidence, conflict
  detection and the durable write-back are current-turn only; explicit relaxation is the
  only path from hard to soft. Asserted in
  ``tests/eval/test_prior_turn_extraction_snapshot.py``.
* NOT DONE, and deliberately out of scope -- a generic typed event reducer, and
  deterministic state recovery after a process restart. Neither is exercised by the
  experiment: 0 of the 42 authoritative scenarios declare ``session_breaks``, and a session
  break opens a new session inside the same process. Nothing may claim support for either.
"""

from __future__ import annotations

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, base_dir="configs")


def _drive(config, turns: list[str]):
    """Run ``turns`` on a single session and return every per-turn result."""
    service = AppService(config, CATALOG)
    candidate = service.create_candidate({"candidate_id": "probe-cand", "skills": []})
    session = service.create_session(candidate.candidate_id, "full")
    return [service.process_turn(session, text, scenario_id="probe") for text in turns]


def _hard(result) -> set[str]:
    return set(result.active_search_state.hard_constraint_fields or [])


def _modes(result) -> set[str]:
    return {str(v).casefold() for v in (result.active_search_state.work_modes or [])}


# ------------------------------------------------------------------------- guards
def test_a_hard_constraint_survives_turns_that_do_not_restate_it(config) -> None:
    """Stated once in turn 0, still binding two turns later.

    The mechanism that makes this work today (re-parsing history) is scheduled for
    removal, so this asserts the OUTCOME rather than the mechanism: whatever carries state
    forward afterwards must carry strength with it.
    """
    results = _drive(config, ["Onsite only.", "Data analyst please.", "At least RM4000."])
    assert "work_modes" in _hard(results[0])
    assert "work_modes" in _hard(results[1]), "a hard work mode was lost by turn 1"
    assert "work_modes" in _hard(results[2]), "a hard work mode was lost by turn 2"
    assert "salary_min" in _hard(results[2]), "the new turn's own hard field is missing"


def test_a_hard_location_survives_an_unrelated_turn(config) -> None:
    results = _drive(config, ["Kuala Lumpur only.", "I have Python and SQL."])
    assert "preferred_locations" in _hard(results[0])
    assert "preferred_locations" in _hard(results[1])


def test_broadening_widens_the_allowed_set_and_keeps_it_binding(config) -> None:
    """"onsite only" then "hybrid is also fine" allows BOTH and stays hard.

    Widening an allowed set is not the same as dropping the requirement. This is the shape
    of SC-D-02, and getting it wrong in either direction is a distinct failure: losing
    hybrid asserts the candidate ruled it out, and dropping the strength stops enforcing a
    constraint the candidate never withdrew.
    """
    results = _drive(config, ["Onsite only.", "Hybrid is also fine."])
    assert _modes(results[-1]) == {"onsite", "hybrid"}, _modes(results[-1])
    assert "work_modes" in _hard(results[-1])


def test_a_later_turn_can_upgrade_a_soft_preference_to_hard(config) -> None:
    results = _drive(config, ["I would prefer hybrid.", "Actually hybrid only."])
    assert "work_modes" not in _hard(results[0])
    assert "work_modes" in _hard(results[1])


# --------------------------------------------------------------------------- gate
@pytest.mark.parametrize("relaxation", [
    "Actually onsite is just a preference.",
    "Actually I am flexible on work mode.",
    "Onsite is no longer a requirement.",
])
def test_an_explicit_relaxation_downgrades_a_hard_constraint(config, relaxation) -> None:
    """A candidate who withdraws a requirement must be believed.

    ``_stronger()`` is monotone -- SOFT never wins over HARD -- which is the right default
    for merging two statements inside one turn, and wrong across turns: it makes a hard
    constraint permanent for the rest of the session. The candidate says the requirement is
    now a preference and the system keeps filtering on it, so results stay narrower than
    what was asked for and nothing in the output says why.

    Only an EXPLICIT relaxation may downgrade. Restating a value with no cue must not, or
    every later mention would erode a constraint the candidate still holds -- which is why
    the guards above assert the opposite case.
    """
    results = _drive(config, ["Onsite only.", relaxation])
    assert "work_modes" in _hard(results[0]), "precondition: turn 0 must be hard"
    assert "work_modes" not in _hard(results[-1]), (
        f"{relaxation!r} did not relax the constraint; "
        f"hard={sorted(_hard(results[-1]))}"
    )


def test_every_dialogue_turn_records_its_own_evidence(config) -> None:
    """``DialogueTurn.evidence_ids`` is empty on every turn.

    The field exists and nothing fills it, so there is no per-turn provenance: the run
    bundle cannot answer "which evidence did THIS turn produce", and the only surviving
    attribution is ``field_evidence_map``, which is keyed by field and rebuilt each turn.
    That is the gap that lets re-parsed history be re-stamped with the current turn id
    without anything noticing.
    """
    results = _drive(config, ["Data analyst in Penang.", "At least RM4000, onsite only."])
    turns = list(results[-1].dialogue_state.turns or [])
    assert len(turns) == 2, len(turns)
    empty = [getattr(turn, "turn_index", index)
             for index, turn in enumerate(turns)
             if not (getattr(turn, "evidence_ids", None) or [])]
    assert not empty, f"turns with no recorded evidence: {empty}"
