"""The frozen scenario reference must be complete, self-consistent and non-leaking.

The relevance oracle grades every run against ``scenario["reference"]``, so that block IS
the ground truth. These tests assert the properties that make it usable as such:

* it exists for every scenario, and identically in every scenario file that contains that
  scenario -- one scenario cannot have two notions of what the candidate asked for;
* the unknown-handling policy is SERIALISED into it rather than inherited from config at
  grading time, because whether an unstated field fails, passes or clarifies changes
  eligibility and therefore changes grades;
* a clarification-dependent scenario declares the answer, so the simulated user and the
  oracle read the same value instead of a global default table;
* and the declared answer does NOT reach the active search or the ranking before the
  system has actually asked for it -- otherwise an ambiguous-role scenario would be
  scored as though the ambiguity had never existed, and a variant that failed to clarify
  would be rewarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import UnknownPolicy
from jobrec_eval.oracle_reference import reference_from_declaration

CATALOG = "data/processed/jobs.jsonl"
MAIN = Path("evaluation/data/scenarios.jsonl")
SUBSET = Path("evaluation/data/scenarios_subset.jsonl")

#: Scenarios whose utterance offers two acceptable work modes. The rule extractor keeps
#: only the first alternative of a disjunction, so the reference had to be corrected; a
#: reference asserting ``["remote"]`` would say the candidate ruled hybrid out.
_DISJUNCTION_WORK_MODES = {"SC-A-03", "SC-D-08"}


def _rows(path: Path) -> dict[str, dict]:
    return {json.loads(line)["scenario_id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


@pytest.fixture(scope="module")
def main_rows() -> dict[str, dict]:
    return _rows(MAIN)


@pytest.fixture(scope="module")
def config():
    return load_config("configs/experiment_full.yaml", base_dir="configs")


def test_every_scenario_declares_a_reference(main_rows) -> None:
    """42/42 declared: nothing falls back to reading the system's own extraction.

    **Validates: Requirements 33.1**
    """
    assert main_rows
    undeclared = [sid for sid, row in main_rows.items() if not row.get("reference")]
    assert undeclared == [], undeclared
    for sid, row in main_rows.items():
        assert isinstance(row["reference"].get("hard"), list), sid


def test_the_unknown_policy_is_serialised_into_the_reference(main_rows) -> None:
    """Unknown handling is declared per constraint field, not inherited at grading time.

    It decides whether a job that does not state a value is eligible, so leaving it to
    config would keep a system policy choice inside the ground truth while the reference
    claimed to be declared.

    **Validates: Requirements 33.1, 15.1**
    """
    valid = {policy.value for policy in UnknownPolicy}
    for sid, row in main_rows.items():
        declared = row["reference"].get("unknown")
        assert isinstance(declared, dict) and declared, f"{sid}: no unknown policy declared"
        for field, policy in declared.items():
            assert policy in valid, (sid, field, policy)
        # Every hard field must be covered: that is where the policy actually bites.
        for field in row["reference"]["hard"]:
            assert field in declared, (sid, field)


def test_the_declared_unknown_policy_reaches_the_built_constraints(main_rows, config) -> None:
    """The declared policy is what the graded constraint bundle ends up carrying.

    Serialising it would be decoration if the builder ignored it, so this drives the real
    construction and reads the result back.

    **Validates: Requirements 33.1, 15.1**
    """
    scenario = main_rows["SC-H-01"]
    flipped = json.loads(json.dumps(scenario))
    # ``salary_min`` is hard here and declared "fail"; flip it and require the change to
    # survive into the constraint the oracle grades with.
    flipped["reference"]["unknown"]["salary_min"] = UnknownPolicy.PASS.value

    for source, expected in ((scenario, UnknownPolicy.FAIL), (flipped, UnknownPolicy.PASS)):
        context, _active = reference_from_declaration(source, config, "cat-test")
        salary = [c for c in context["constraints"] if c["field_name"] == "salary_min"]
        assert salary, "salary_min did not become a constraint"
        assert salary[0]["unknown_policy"] == expected.value


def test_a_disjunction_keeps_both_alternatives(main_rows) -> None:
    """"remote or hybrid" declares BOTH, and neither is binding.

    **Validates: Requirements 33.1**
    """
    for sid in _DISJUNCTION_WORK_MODES:
        reference = main_rows[sid]["reference"]
        assert set(reference["work_modes"]) == {"remote", "hybrid"}, sid
        assert "work_modes" not in reference["hard"], sid


def test_the_subset_declares_the_same_reference_as_the_main_set(main_rows) -> None:
    """A scenario present in two files has ONE ground truth.

    **Validates: Requirements 33.1**
    """
    for sid, row in _rows(SUBSET).items():
        assert sid in main_rows, sid
        assert row.get("reference") == main_rows[sid]["reference"], sid


def test_clarification_dependent_scenarios_declare_their_answer(main_rows) -> None:
    """Every acceptable slot of a clarification-expected scenario has a declared answer.

    **Validates: Requirements 33.1**
    """
    for sid, row in main_rows.items():
        if not row.get("clarification_expected"):
            continue
        slots = row.get("acceptable_slots") or []
        declared = (row["reference"].get("clarification_answer") or {})
        assert slots, sid
        for slot in slots:
            assert declared.get(slot), (sid, slot)


# --------------------------------------------------------------- no pre-clarification leak
#: The scenarios whose role is deliberately vague. Their declared answer is what the
#: dialogue eventually establishes, NOT what the first turn states.
_AMBIGUOUS = ("SC-G-01", "SC-G-02")


@pytest.mark.parametrize("scenario_id", _AMBIGUOUS)
def test_the_declared_answer_does_not_leak_before_the_system_asks(
    scenario_id, main_rows, config
) -> None:
    """After the opening turn the clarified role is absent from the active search.

    This is the property that makes grading against the clarified role honest. If the
    declared answer reached the active state (or the ranking) before the system asked for
    it, the ambiguity would be scored as though it had never existed and a variant that
    never clarified would be rewarded for guessing -- which is the failure mode a broad
    "any data role counts" reference would have had.

    **Validates: Requirements 33.1, 7.8**
    """
    scenario = main_rows[scenario_id]
    declared = scenario["reference"]["clarification_answer"]["target_roles"]

    service = AppService(config, CATALOG)
    candidate = service.create_candidate(dict(scenario["profile"]))
    session = service.create_session(candidate.candidate_id, "full")
    result = service.process_turn(session, scenario["turns"][0], scenario_id=scenario_id)

    active_roles = list(getattr(result.active_search_state, "target_roles", []) or [])
    assert declared not in active_roles, (scenario_id, active_roles)
    assert active_roles == [], (scenario_id, active_roles)

    # Nothing was ranked or returned on the strength of a role the user never stated.
    assert result.response.response_type == "clarification"
    assert not (result.decision and result.decision.selected_job_ids)
    # And the profile does not supply the role either, so the reference cannot be
    # satisfied by the profile alone -- the dialogue has to establish it.
    assert not scenario["profile"].get("target_roles")


@pytest.mark.parametrize("scenario_id", _AMBIGUOUS)
def test_the_reference_marks_the_role_unspecified_until_clarified(
    scenario_id, main_rows
) -> None:
    """The reference records that the opening role is unspecified, not merely absent.

    **Validates: Requirements 33.1**
    """
    reference = main_rows[scenario_id]["reference"]
    assert reference.get("role_scope") == "unspecified_until_clarified"
    assert reference["clarification_answer"]["target_roles"]
