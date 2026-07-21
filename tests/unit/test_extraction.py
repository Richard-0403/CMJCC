"""Unit tests for the rule-based extractor."""

from __future__ import annotations

from jobrec.agents.candidate_understanding import CandidateUnderstandingAgent
from jobrec.domain.enums import ConstraintStrength


def _by_field(result):
    out = {}
    for p in result.preferences:
        out.setdefault(p.field_name, []).append(p)
    return out


def test_extracts_core_slots():
    r = CandidateUnderstandingAgent().extract(
        "I want a junior data analyst role in Kuala Lumpur. I know Python and SQL. "
        "I prefer hybrid work and salary above RM4000."
    )
    f = _by_field(r)
    assert f["target_roles"][0].normalized_value == "data analyst"
    assert f["experience_level"][0].normalized_value == "junior"
    assert f["preferred_locations"][0].normalized_value == "Kuala Lumpur"
    assert {p.normalized_value for p in f["skills_have"]} == {"python", "sql"}
    assert f["salary_min"][0].normalized_value == 4000.0
    assert f["salary_currency"][0].normalized_value == "MYR"


def test_threshold_makes_salary_hard():
    r = CandidateUnderstandingAgent().extract("data analyst, at least RM5000 per month")
    sal = [p for p in r.preferences if p.field_name == "salary_min"][0]
    assert sal.proposed_strength == ConstraintStrength.HARD


def test_only_makes_hard():
    r = CandidateUnderstandingAgent().extract("I only want remote roles in Penang")
    loc = [p for p in r.preferences if p.field_name == "preferred_locations"][0]
    assert loc.proposed_strength == ConstraintStrength.HARD


def test_negation_creates_exclusion():
    r = CandidateUnderstandingAgent().extract("I do not want sales analyst jobs")
    excl = [p for p in r.preferences if p.polarity == "negative"]
    assert excl and excl[0].normalized_value == "sales analyst"


def test_no_spurious_single_char_skill():
    r = CandidateUnderstandingAgent().extract("I want a role in a great company")
    skills = [p for p in r.preferences if p.field_name == "skills_have"]
    assert all(p.normalized_value != "r" for p in skills)
