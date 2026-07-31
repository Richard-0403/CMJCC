"""Hidden paraphrase suite for constraint extraction (P0-1).

None of these utterances appears in the 42 official scenarios -- there is a test at the
bottom that enforces it. That is the point: the 42/42 reference gate can in principle be
satisfied by tuning until those particular sentences pass, and this suite is what makes
that not enough. It states the extraction contract in terms of the linguistic phenomenon
rather than the example.

Several of these FAIL on the current implementation, deliberately, and they are the
specification for the fix:

* POST-positioned cues are invisible. ``_clause(lower, span[0], span[1])`` ends the window
  at the end of the matched value, so "onsite only" yields the clause "onsite" and the
  hard cue never reaches ``_strength_for``. Pre-positioned cues work, which is why
  "at least RM4000" is correctly hard in the same sentence.
* Disjunctions lose every alternative after the first: the work-mode loop ``break``s on
  its first match, so "remote or hybrid" becomes ["remote"].
* Cue scope bleeds across fields. ``_strength_for`` matches a cue anywhere in the clause
  it is given, so one field's "at least"/"only" can decide another field's strength when
  both appear in the same clause.

Cases that already pass are kept rather than trimmed: they are the regression half, and a
fix that makes post-positioned cues work by widening the window until it swallows the
whole utterance would break them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.agents.candidate_understanding import CandidateUnderstandingAgent
from jobrec.domain.enums import ConstraintStrength

#: Marked as a gate so the 12 currently-failing cases do not sit inside the default suite,
#: where the next genuine regression would arrive in an already-red run and be
#: indistinguishable from the expected failures. The 19 that pass today are still executed
#: -- by the gate check rather than the default one. Remove this marker once the suite is
#: green so the whole module returns to the default run.
pytestmark = pytest.mark.extraction_gate

SCENARIOS = Path("evaluation/data/scenarios.jsonl")

HARD = ConstraintStrength.HARD
SOFT = ConstraintStrength.SOFT
UNKNOWN = ConstraintStrength.UNKNOWN


@pytest.fixture(scope="module")
def agent() -> CandidateUnderstandingAgent:
    return CandidateUnderstandingAgent()


def _by_field(agent: CandidateUnderstandingAgent, text: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for pref in agent.extract(text).preferences:
        out.setdefault(pref.field_name, []).append(pref)
    return out


def _values(prefs: list) -> set[str]:
    return {str(p.normalized_value).casefold() for p in prefs}


def _strengths(prefs: list) -> set[ConstraintStrength]:
    return {p.proposed_strength for p in prefs}


# ------------------------------------------------------- 1. pre-positioned hard cues
@pytest.mark.parametrize("text,field", [
    ("I must be in Cyberjaya.", "preferred_locations"),
    ("It has to pay a minimum of RM7200.", "salary_min"),
    ("Salary no less than RM8100.", "salary_min"),
])
def test_pre_positioned_hard_cue_is_hard(agent, text, field) -> None:
    prefs = _by_field(agent, text).get(field)
    assert prefs, f"{field} not extracted from {text!r}"
    assert HARD in _strengths(prefs), (text, field, _strengths(prefs))


# ------------------------------------------------------ 2. post-positioned hard cues
@pytest.mark.parametrize("text,field,value", [
    ("Remote only.", "work_modes", "remote"),
    ("Onsite only, nothing else.", "work_modes", "onsite"),
    ("Cyberjaya only.", "preferred_locations", "Cyberjaya"),
    ("I want hybrid, and that is mandatory.", "work_modes", "hybrid"),
])
def test_post_positioned_hard_cue_is_hard(agent, text, field, value) -> None:
    """The cue follows the value, which is the ordinary English word order for "only"."""
    prefs = _by_field(agent, text).get(field)
    assert prefs, f"{field} not extracted from {text!r}"
    assert value.casefold() in _values(prefs), (text, _values(prefs))
    assert HARD in _strengths(prefs), (
        f"{text!r}: {field} should be HARD -- the cue sits after the value, "
        f"got {_strengths(prefs)}"
    )


# ------------------------------------------------------------------ 3. soft/flex cues
@pytest.mark.parametrize("text,field", [
    ("I would prefer Cyberjaya.", "preferred_locations"),
    ("Ideally remote.", "work_modes"),
    ("Hybrid is fine.", "work_modes"),
    ("I am open to onsite.", "work_modes"),
])
def test_soft_cue_stays_soft(agent, text, field) -> None:
    prefs = _by_field(agent, text).get(field)
    assert prefs, f"{field} not extracted from {text!r}"
    assert HARD not in _strengths(prefs), (text, _strengths(prefs))


# ----------------------------------------------------------------- 4. unsure cues
@pytest.mark.parametrize("text,field", [
    ("Maybe Cyberjaya.", "preferred_locations"),
    ("I am not sure about remote.", "work_modes"),
])
def test_unsure_cue_is_not_hard(agent, text, field) -> None:
    prefs = _by_field(agent, text).get(field)
    assert prefs, f"{field} not extracted from {text!r}"
    assert HARD not in _strengths(prefs), (text, _strengths(prefs))


# ------------------------------------------------------------------ 5. multi-value sets
@pytest.mark.parametrize("text,expected", [
    ("Remote or hybrid.", {"remote", "hybrid"}),
    ("Either remote or onsite works.", {"remote", "onsite"}),
    ("Hybrid, and remote as well.", {"hybrid", "remote"}),
    ("I can do onsite or hybrid.", {"onsite", "hybrid"}),
])
def test_a_disjunction_keeps_every_alternative(agent, text, expected) -> None:
    """Dropping an alternative asserts the candidate ruled it out, which is not what was said."""
    prefs = _by_field(agent, text).get("work_modes")
    assert prefs, f"work_modes not extracted from {text!r}"
    assert _values(prefs) == expected, (text, _values(prefs), expected)


def test_multiple_locations_are_both_kept(agent) -> None:
    prefs = _by_field(agent, "Cyberjaya or Selangor.").get("preferred_locations")
    assert prefs, "preferred_locations not extracted"
    assert _values(prefs) == {"cyberjaya", "selangor"}, _values(prefs)


# -------------------------------------------------- 6. cue scope must not cross fields
def test_a_salary_threshold_cue_does_not_harden_a_work_mode(agent) -> None:
    """"at least" belongs to the salary, not to the work mode in the same sentence."""
    fields = _by_field(agent, "Hybrid is fine, at least RM7300.")
    assert fields.get("salary_min"), "salary_min not extracted"
    assert HARD in _strengths(fields["salary_min"])
    assert fields.get("work_modes"), "work_modes not extracted"
    assert HARD not in _strengths(fields["work_modes"]), (
        "the salary's 'at least' leaked onto work_modes: "
        f"{_strengths(fields['work_modes'])}"
    )


def test_a_location_only_cue_does_not_harden_a_salary(agent) -> None:
    fields = _by_field(agent, "Cyberjaya only, ideally around RM7400.")
    assert fields.get("preferred_locations"), "preferred_locations not extracted"
    assert HARD in _strengths(fields["preferred_locations"])
    assert fields.get("salary_min"), "salary_min not extracted"
    assert HARD not in _strengths(fields["salary_min"]), (
        f"the location's 'only' leaked onto salary_min: {_strengths(fields['salary_min'])}"
    )


def test_two_fields_keep_their_own_strengths(agent) -> None:
    fields = _by_field(agent, "Remote only, and I would prefer Selangor.")
    assert fields.get("work_modes") and fields.get("preferred_locations")
    assert HARD in _strengths(fields["work_modes"]), _strengths(fields["work_modes"])
    assert HARD not in _strengths(fields["preferred_locations"]), (
        _strengths(fields["preferred_locations"]))


# ----------------------------------------------------------------------- 7. negation
@pytest.mark.parametrize("text,field", [
    ("I do not want Singapore.", "preferred_locations"),
    ("No onsite please.", "work_modes"),
])
def test_a_negated_value_is_not_a_positive_preference(agent, text, field) -> None:
    prefs = _by_field(agent, text).get(field) or []
    positives = [p for p in prefs if p.polarity == "positive"]
    assert not positives, (
        f"{text!r}: a negated {field} became a positive preference: "
        f"{[p.normalized_value for p in positives]}"
    )


# --------------------------------------------------------------------- 8. relaxation
def test_relaxing_a_value_is_not_stated_as_hard(agent) -> None:
    """"I can be flexible on X" must not arrive as a hard constraint on X."""
    prefs = _by_field(agent, "I can be flexible on remote.").get("work_modes")
    assert prefs, "work_modes not extracted"
    assert HARD not in _strengths(prefs), _strengths(prefs)


def test_a_broadened_set_is_still_a_set(agent) -> None:
    """Adding an alternative widens the allowed values; it does not replace them."""
    prefs = _by_field(agent, "Onsite is required, though hybrid would also work.").get(
        "work_modes")
    assert prefs, "work_modes not extracted"
    assert _values(prefs) == {"onsite", "hybrid"}, _values(prefs)


# -------------------------------------------------------------------- 9. replacement
def test_a_replacement_states_only_the_new_value(agent) -> None:
    """A single utterance naming one location yields that location, not a merge."""
    prefs = _by_field(agent, "Actually make it Selangor instead.").get(
        "preferred_locations")
    assert prefs, "preferred_locations not extracted"
    assert _values(prefs) == {"selangor"}, _values(prefs)


# ------------------------------------------------------- 10. thresholds keep an amount
@pytest.mark.parametrize("text,amount", [
    ("At least RM7500 a month.", 7500.0),
    ("Minimum RM7600.", 7600.0),
    ("I need more than RM7700.", 7700.0),
])
def test_a_threshold_keeps_its_amount_and_is_hard(agent, text, amount) -> None:
    prefs = _by_field(agent, text).get("salary_min")
    assert prefs, f"salary_min not extracted from {text!r}"
    # The project's single salary parser, so this asserts the amount the rest of the
    # pipeline will see rather than a second interpretation of the same field.
    from jobrec.llm.field_validation import salary_amount

    assert any(salary_amount(p.normalized_value) == amount for p in prefs), (
        text, [p.normalized_value for p in prefs])
    assert HARD in _strengths(prefs), (text, _strengths(prefs))


# ------------------------------------------------------------------ suite self-check
def _utterances_used() -> set[str]:
    """Every literal utterance this module feeds to the extractor."""
    source = Path(__file__).read_text(encoding="utf-8")
    found = set()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(('("', '"')) and '",' in stripped:
            candidate = stripped.split('"')[1]
            if len(candidate) > 12 and " " in candidate:
                found.add(candidate)
    return found


def test_no_paraphrase_reuses_an_official_scenario_utterance() -> None:
    """These have to be hidden cases, not the 42 scenarios spelled out again.

    Reusing scenario text would let a fix tuned to the reference gate pass here too, and
    the suite would stop being independent evidence.
    """
    official: set[str] = set()
    for line in SCENARIOS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            official.update(str(t).casefold() for t in json.loads(line).get("turns", []))
    reused = sorted(u for u in _utterances_used() if u.casefold() in official)
    assert not reused, f"paraphrases copied from the official scenarios: {reused}"


def test_the_suite_has_at_least_24_cases(request) -> None:
    """The plan's acceptance criterion is >= 24 hidden boundary cases."""
    collected = [item for item in request.session.items
                 if item.nodeid.startswith(f"{Path(__file__).name.split('.')[0]}")
                 or "test_constraint_cue_paraphrases" in item.nodeid]
    # -1 so this bookkeeping test does not count itself.
    assert len(collected) - 1 >= 24, len(collected) - 1
