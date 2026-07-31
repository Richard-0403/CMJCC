"""42/42 gate: the system's final state must match every declared reference.

**THIS GATE IS RED ON PURPOSE.** It was written before the extraction fix so that the fix
is proven by the gate turning green, not asserted. Its baseline failure set, measured on
commit 49eacbf, is 12 of 42 scenarios:

* value-level, a disjunction losing its second alternative (2):
  ``SC-A-03`` and ``SC-D-08`` declare ``work_modes = [remote, hybrid]`` and the extractor
  produces ``[remote]`` -- the work-mode loop ``break``s on the first match.
* strength-level, hard where the reference declares soft (7):
  ``SC-B-04``, ``SC-D-09``, ``SC-D-11`` on ``preferred_locations``;
  ``SC-D-10``, ``SC-D-12`` on ``work_modes``;
  ``SC-E-01``, ``SC-E-03``, ``SC-H-01``, ``SC-H-03`` on ``target_roles``.
  (9 scenarios over 3 fields.)
* strength-level, soft where the reference declares hard (1):
  ``SC-D-02`` on ``work_modes`` -- "onsite only" is classified SOFT because the clause
  window is truncated at the end of the matched value, so a POST-positioned cue like
  "only" never reaches the strength classifier. "at least RM4000" works in the same
  utterance because "at least" precedes its number.

What this gate is not: a list of the scenarios above. It runs all 42 and compares each
against its own declaration, so a fix that special-cases an id fails just as loudly as no
fix at all. The expectations are derived from declared fields (``reference.hard``,
``reference.role_scope``), never from a scenario identifier -- there is a test below that
enforces that property on this module's own source.

Why it drives the real runner rather than reconstructing the turn loop: multi-turn and
clarification scenarios only reach their final state through the runner's session
threading and its simulated-user clarification loop. Re-implementing that here would be a
second pipeline free to drift from the one the official experiment uses, and the gate would
then be measuring the test instead of the system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner, load_scenarios

pytestmark = pytest.mark.extraction_gate

CATALOG = "data/processed/jobs.jsonl"
SCENARIOS = "evaluation/data/scenarios.jsonl"
CONFIG = "configs/experiment_full.yaml"

EXPECTED_SCENARIO_COUNT = 42

#: Declared reference fields compared value-for-value against the final state.
_LIST_FIELDS = ("target_roles", "skills_have", "preferred_locations", "work_modes")
_SCALAR_FIELDS = ("salary_min", "salary_currency", "years_experience")

#: Hard-constraint fields the SYSTEM injects and a declaration never names, so they are
#: outside the comparison: ``exclusions`` appears whenever the profile or a turn excludes
#: something, and ``not_expired`` is an always-on system constraint.
_SYSTEM_HARD_FIELDS = frozenset({"exclusions", "not_expired"})


def _declared_scenarios() -> dict[str, dict]:
    return {s["scenario_id"]: s for s in load_scenarios(SCENARIOS)}


@pytest.fixture(scope="module")
def declared() -> dict[str, dict]:
    return _declared_scenarios()


@pytest.fixture(scope="module")
def final_states(tmp_path_factory) -> dict[str, dict[str, Any]]:
    """Final ``ActiveSearchState`` per scenario, from one full-variant audit run.

    One variant and one repeat: this gate is about what the extractor produces from the
    scenario text, and ``full`` is the condition with every mechanism enabled, so a
    mismatch here is an extraction defect rather than an ablation effect.
    """
    config = load_config(CONFIG, base_dir="configs")
    config = config.model_copy(deep=True)
    config.experiment.repeat_count = 1

    out_dir = tmp_path_factory.mktemp("reference_gate")
    runner = ExperimentRunner(config, CATALOG, SCENARIOS, out_dir=str(out_dir))
    result = runner.run(["full"])

    assert result["crashed_run_count"] == 0, result["crashed_runs"]
    assert result["run_count"] == EXPECTED_SCENARIO_COUNT, result["run_count"]

    states: dict[str, dict[str, Any]] = {}
    exp_dir = Path(result["experiment_dir"])
    for path in exp_dir.rglob("active_search_state.json"):
        # <exp>/full/<scenario_id>/<repeat>/active_search_state.json
        scenario_id = path.parent.parent.name
        states[scenario_id] = json.loads(path.read_text(encoding="utf-8"))
    assert len(states) == EXPECTED_SCENARIO_COUNT, sorted(states)
    return states


def _expected_values(reference: dict) -> dict[str, Any]:
    """The values the final state should carry, read off the declaration.

    ``role_scope == "unspecified_until_clarified"`` means the opening turn does NOT state
    the role and the dialogue establishes it, so the role to expect at the END of the run
    is the declared clarification answer. Keying on that declared field rather than on a
    scenario id is what keeps this rule general.
    """
    expected: dict[str, Any] = {}
    for field in _LIST_FIELDS:
        if reference.get(field) is not None:
            expected[field] = [str(v).casefold() for v in reference[field]]
    if reference.get("role_scope") == "unspecified_until_clarified":
        answer = (reference.get("clarification_answer") or {}).get("target_roles")
        if answer:
            values = answer if isinstance(answer, list) else [answer]
            expected["target_roles"] = [str(v).casefold() for v in values]
    for field in _SCALAR_FIELDS:
        if reference.get(field) is not None:
            expected[field] = reference[field]
    return expected


def _actual_values(state: dict) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    for field in _LIST_FIELDS:
        actual[field] = [str(v).casefold() for v in (state.get(field) or [])]
    for field in _SCALAR_FIELDS:
        actual[field] = state.get(field)
    return actual


def _hard_fields(state: dict) -> set[str]:
    return set(state.get("hard_constraint_fields") or []) - _SYSTEM_HARD_FIELDS


def _scenario_ids() -> list[str]:
    return sorted(_declared_scenarios())


# ------------------------------------------------------------------- value-level gate
@pytest.mark.parametrize("scenario_id", _scenario_ids())
def test_declared_values_reach_the_final_state(scenario_id, declared, final_states) -> None:
    reference = declared[scenario_id]["reference"]
    expected = _expected_values(reference)
    actual = _actual_values(final_states[scenario_id])

    wrong = {field: {"declared": value, "actual": actual.get(field)}
             for field, value in expected.items()
             if _differs(field, value, actual.get(field))}
    assert not wrong, f"{scenario_id}: declared values missing from the final state: {wrong}"


def _differs(field: str, declared: Any, actual: Any) -> bool:
    if field in _LIST_FIELDS:
        return set(declared or []) != set(actual or [])
    if isinstance(declared, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(declared) - float(actual)) > 1e-9
    return declared != actual


# ---------------------------------------------------------------- strength-level gate
@pytest.mark.parametrize("scenario_id", _scenario_ids())
def test_declared_hard_fields_are_exactly_the_hard_fields(
    scenario_id, declared, final_states
) -> None:
    reference = declared[scenario_id]["reference"]
    expected = set(reference.get("hard") or [])
    actual = _hard_fields(final_states[scenario_id])
    assert actual == expected, (
        f"{scenario_id}: hard-constraint fields disagree with the declaration. "
        f"declared={sorted(expected)} actual={sorted(actual)} "
        f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    )


@pytest.mark.parametrize("scenario_id", _scenario_ids())
def test_a_declared_soft_field_is_not_treated_as_hard(
    scenario_id, declared, final_states
) -> None:
    """A stated field the declaration leaves out of ``hard`` must land in the soft set.

    Separate from the test above because the failure means something different: the value
    was extracted correctly and then over-constrained, which silently shrinks the candidate
    pool instead of mis-ranking it.
    """
    reference = declared[scenario_id]["reference"]
    state = final_states[scenario_id]
    hard = set(reference.get("hard") or [])
    stated = {field for field in (*_LIST_FIELDS, *_SCALAR_FIELDS)
              if reference.get(field) not in (None, [], {})}
    soft_expected = stated - hard
    actual_hard = _hard_fields(state)
    over_constrained = sorted(soft_expected & actual_hard)
    assert not over_constrained, (
        f"{scenario_id}: declared soft but enforced as hard: {over_constrained}"
    )


# ------------------------------------------------------------------ gate self-checks
def test_the_gate_covers_every_scenario(declared) -> None:
    assert len(declared) == EXPECTED_SCENARIO_COUNT
    assert len(_scenario_ids()) == EXPECTED_SCENARIO_COUNT


def test_the_gate_contains_no_scenario_specific_expectation() -> None:
    """No scenario id may appear in executable code in this module.

    A fix that special-cases an id must not be able to satisfy the gate, and a gate that
    hard-codes the currently-failing ids would stop being a gate the moment the list
    changed. Scenario ids appear only in the module docstring, which records the baseline.
    """
    # Assembled rather than written literally: spelled out, the needle would appear on
    # this very line and the check would report itself.
    needle = "SC" + "-"
    source = Path(__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2] if source.count('"""') >= 2 else source
    offenders = [line.strip() for line in body.splitlines()
                 if needle in line and not line.lstrip().startswith("#")]
    assert not offenders, f"scenario ids in executable code: {offenders}"
