"""Tests for the dataset data-quality validator (R17).

Each defect class the requirement names is injected into an otherwise clean
dataset and must come back as a finding that carries the offending identifier and
the violation type (R17.1/R17.2/R17.4), and the emitted report must be
machine-readable (R17.3).

Everything runs against real data structures: the catalog rows are real records
from ``data/processed/jobs.jsonl``, the scenarios have the shape the scenario
loader produces, and the no-match check drives the real orchestration path with
the deterministic rule extractor -- no mocks, no stubbed eligibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jobrec.catalog import load_catalog
from jobrec.config import load_config
from jobrec.domain.job import JobPosting
from jobrec_eval.data_quality import (
    DATA_QUALITY_REPORT_FILENAME,
    read_data_quality_report,
    validate_dataset,
    write_data_quality_report,
)
from jobrec_eval.scenarios import load_scenarios

CATALOG_PATH = "data/processed/jobs.jsonl"

#: A real, active, non-expired Kuala Lumpur data-analyst posting: eligible for the
#: recommendation scenario below and used as the "clean" catalog record.
_CLEAN_JOB_ID = "job-0012"


@pytest.fixture(scope="module")
def raw_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(CATALOG_PATH).read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def clean_row(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = next(r for r in raw_rows if r["job_id"] == _CLEAN_JOB_ID)
    return row


@pytest.fixture(scope="module")
def clean_job(clean_row: dict[str, Any]) -> JobPosting:
    return JobPosting.model_validate(clean_row)


@pytest.fixture()
def config():
    return load_config("configs/experiment_full.yaml", base_dir="configs")


def _recommendation_scenario(scenario_id: str = "SC-DQ-01") -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "scenario_type": "complete",
        "profile": {"skills": ["Python", "SQL"], "years_experience": 3,
                    "target_roles": ["Data Analyst"]},
        "turns": ["I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."],
        "no_match_expected": False,
        "clarification_expected": False,
        "expects": {"response_type": "recommendation"},
    }


def _no_match_scenario(turn: str, scenario_id: str = "SC-DQ-NM") -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "scenario_type": "no_match",
        "profile": {"skills": ["Python", "SQL"], "years_experience": 3,
                    "target_roles": ["Data Analyst"]},
        "turns": [turn],
        "no_match_expected": True,
        "clarification_expected": False,
        "expects": {"response_type": "no_match", "blocking": "salary_min"},
    }


def _types(report) -> set[str]:
    return {finding.violation_type for finding in report}


# --------------------------------------------------------------------- baseline
def test_clean_dataset_produces_no_findings(clean_job, config):
    """A well-formed catalog + labelled scenario yields an empty, OK report."""
    scenario = _recommendation_scenario()
    report = validate_dataset(
        [clean_job], [scenario], config=config,
        relevance_labels=[scenario["scenario_id"]],
    )

    assert report.ok
    assert list(report) == []
    assert report.job_count == 1
    assert report.scenario_count == 1
    assert report.reference_date == config.project.reference_date
    assert report.checks_skipped == {}


def test_real_shipped_dataset_has_no_errors(config):
    """The dataset the experiment actually runs on is free of error-level defects."""
    catalog = load_catalog(CATALOG_PATH)
    scenarios = load_scenarios("data/scenarios/scenarios.jsonl")

    report = validate_dataset(
        catalog, scenarios, config=config,
        relevance_labels="data/scenarios/relevance_labels.csv",
    )

    assert report.errors == (), report.summary()
    # Every expired posting in the catalog is accounted for, and each one is
    # correctly flagged inactive (so it is a warning, not contradictory data).
    assert report.by_type("expired_deadline_active") == ()
    assert report.by_type("expired_deadline")


# -------------------------------------------------------------- catalog defects
def _mutate(row: dict[str, Any], **changes: Any) -> dict[str, Any]:
    mutated = dict(row)
    mutated.update(changes)
    return mutated


@pytest.mark.parametrize(
    ("violation_type", "changes", "field_name"),
    [
        ("salary_min_exceeds_max", {"salary_min": 9000.0, "salary_max": 4000.0},
         "salary_min"),
        ("unknown_currency", {"salary_currency": "XYZ"}, "salary_currency"),
        ("invalid_work_mode", {"work_mode": "teleport"}, "work_mode"),
        ("invalid_experience_level", {"experience_level": "wizard"},
         "experience_level"),
        ("expired_deadline_active", {"application_deadline": "2025-11-30", "is_active": True},
         "application_deadline"),
        ("invalid_deadline", {"application_deadline": "not-a-date"}, "application_deadline"),
        ("empty_title", {"title": "  "}, "title"),
        ("empty_skills", {"required_skills": [], "preferred_skills": []},
         "required_skills"),
        ("empty_location", {"city": None, "region": None, "country": None}, "city"),
    ],
)
def test_each_catalog_defect_class_is_flagged(
    clean_row, config, violation_type, changes, field_name
):
    """Every R17.1 catalog defect is reported with its job id and violation type."""
    defective = _mutate(clean_row, job_id="job-defect", **changes)

    report = validate_dataset([clean_row, defective], [], config=config)

    findings = report.by_type(violation_type)
    assert len(findings) == 1, report.summary()
    assert findings[0].identifier == "job-defect"
    assert findings[0].field_name == field_name
    assert findings[0].severity == "error"
    # The clean record next to it is never implicated.
    assert all(f.identifier == "job-defect" for f in report)


def test_duplicate_job_id_is_flagged(clean_row, config):
    """A repeated job id is reported once, naming the duplicated identifier."""
    report = validate_dataset([clean_row, dict(clean_row)], [], config=config)

    findings = report.by_type("duplicate_job_id")
    assert [f.identifier for f in findings] == [_CLEAN_JOB_ID]
    assert findings[0].observed == 2


def test_missing_job_id_is_flagged(clean_row, config):
    """A record with no job id is still reportable, addressed by its row."""
    report = validate_dataset([_mutate(clean_row, job_id="")], [], config=config)

    findings = report.by_type("empty_job_id")
    assert len(findings) == 1
    assert "row 0" in findings[0].identifier


def test_expired_but_inactive_posting_is_a_warning_not_an_error(clean_row, config):
    """Deliberately expired fixtures are recorded without failing the dataset."""
    expired = _mutate(clean_row, job_id="job-expired",
                      application_deadline="2025-10-01", is_active=False)

    report = validate_dataset([expired], [], config=config)

    findings = report.by_type("expired_deadline")
    assert [f.identifier for f in findings] == ["job-expired"]
    assert findings[0].severity == "warning"
    assert report.ok


# ------------------------------------------------------------- scenario defects
def test_duplicate_scenario_id_is_flagged(config):
    scenario = _recommendation_scenario()
    report = validate_dataset([], [scenario, dict(scenario)], config=config)

    findings = report.by_type("duplicate_scenario_id")
    assert [f.identifier for f in findings] == [scenario["scenario_id"]]


def test_scenario_without_turns_is_flagged(config):
    scenario = _mutate(_recommendation_scenario(), turns=[])
    report = validate_dataset([], [scenario], config=config)

    findings = report.by_type("empty_scenario_turns")
    assert [f.identifier for f in findings] == [scenario["scenario_id"]]


def test_missing_relevance_label_is_flagged_only_where_required(config):
    """A recommendation scenario needs a label; clarification/no-match ones do not."""
    recommendation = _recommendation_scenario("SC-DQ-REC")
    clarification = {**_recommendation_scenario("SC-DQ-CLAR"),
                     "clarification_expected": True,
                     "expects": {"response_type": "clarification"}}
    no_match = _no_match_scenario("Data analyst in Kuala Lumpur, at least RM50000.",
                                  "SC-DQ-NOLABEL")

    report = validate_dataset(
        [], [recommendation, clarification, no_match], config=config,
        relevance_labels=[],
    )

    findings = report.by_type("missing_relevance_label")
    assert [f.identifier for f in findings] == ["SC-DQ-REC"]
    assert findings[0].severity == "warning"


def test_relevance_label_check_is_skipped_when_no_labels_supplied(config):
    """Without a label source the check is recorded as not run, not as failures."""
    report = validate_dataset([], [_recommendation_scenario()], config=config)

    assert report.by_type("missing_relevance_label") == ()
    assert "scenario_relevance_labels" in report.checks_skipped


def test_missing_hard_constraint_reference_is_flagged(config):
    """A no-match scenario that names no blocking constraint is reported (R17.2)."""
    scenario = _no_match_scenario("Data analyst in Kuala Lumpur, at least RM50000.")
    scenario["expects"] = {"response_type": "no_match"}

    report = validate_dataset([], [scenario], config=config)

    findings = report.by_type("missing_hard_constraint_reference")
    assert [f.identifier for f in findings] == [scenario["scenario_id"]]


def test_contradictory_expectation_labels_are_flagged(config):
    scenario = _mutate(_recommendation_scenario(), no_match_expected=True)
    report = validate_dataset([], [scenario], config=config)

    findings = report.by_type("inconsistent_no_match_expectation")
    assert [f.identifier for f in findings] == [scenario["scenario_id"]]
    assert findings[0].observed == "recommendation"


# ------------------------------------------------------- no-match verification
def test_mislabelled_no_match_scenario_is_detected(clean_job, config):
    """A satisfiable scenario labelled no-match is an error naming the eligible jobs."""
    scenario = _no_match_scenario(
        "I only want a data analyst role in Kuala Lumpur, at least RM4000.",
        "SC-DQ-FAKE-NOMATCH",
    )

    report = validate_dataset([clean_job], [scenario], config=config)

    findings = report.by_type("no_match_scenario_satisfiable")
    assert len(findings) == 1, report.summary()
    assert findings[0].identifier == "SC-DQ-FAKE-NOMATCH"
    assert findings[0].severity == "error"
    assert _CLEAN_JOB_ID in findings[0].observed
    assert not report.ok


def test_genuine_no_match_scenario_passes(clean_job, config):
    """An unsatisfiable scenario produces no no-match finding."""
    scenario = _no_match_scenario(
        "I only want a data analyst role in Kuala Lumpur with salary at least "
        "RM50000 per month.",
        "SC-DQ-REAL-NOMATCH",
    )

    report = validate_dataset([clean_job], [scenario], config=config)

    assert _types(report) & {
        "no_match_scenario_satisfiable", "no_match_scenario_constraint_satisfiable",
    } == set()
    assert "no_match_scenarios_unsatisfiable" in report.checks_run
    assert report.ok


def test_no_match_check_can_be_skipped(clean_job, config):
    """Disabling the replay records the skip instead of silently passing."""
    scenario = _no_match_scenario(
        "I only want a data analyst role in Kuala Lumpur, at least RM4000.")

    report = validate_dataset([clean_job], [scenario], config=config,
                              verify_no_match=False)

    assert report.by_type("no_match_scenario_satisfiable") == ()
    assert "no_match_scenarios_unsatisfiable" in report.checks_skipped


# --------------------------------------------------------------------- reporting
def test_report_is_machine_readable_and_names_every_offender(clean_row, config, tmp_path):
    """The emitted report records identifier + violation type per finding (R17.3/17.4)."""
    defective = _mutate(clean_row, job_id="job-bad", salary_min=9000.0, salary_max=1000.0)
    report = validate_dataset([defective], [], config=config)

    path = write_data_quality_report(report, tmp_path)
    assert path.name == DATA_QUALITY_REPORT_FILENAME

    payload = read_data_quality_report(tmp_path)
    assert payload["ok"] is False
    assert payload["error_count"] == 1
    assert payload["job_count"] == 1
    assert payload["reference_date"] == config.project.reference_date
    assert payload["counts_by_violation_type"] == {"salary_min_exceeds_max": 1}
    assert payload["findings"] == [{
        "identifier": "job-bad",
        "entity": "job",
        "violation_type": "salary_min_exceeds_max",
        "severity": "error",
        "field": "salary_min",
        "observed": {"salary_min": 9000.0, "salary_max": 1000.0},
        "detail": "salary_min 9000.0 exceeds salary_max 1000.0",
    }]


def test_report_output_is_stable_across_writes(clean_row, config, tmp_path):
    """Validating unchanged inputs twice yields byte-identical reports."""
    report = validate_dataset([clean_row], [_recommendation_scenario()], config=config)

    first = write_data_quality_report(report, tmp_path).read_bytes()
    again = validate_dataset([clean_row], [_recommendation_scenario()], config=config)
    second = write_data_quality_report(again, tmp_path).read_bytes()

    assert first == second

# ------------------------------------------------------------------- property 24
#: Catalog defects expressed as field changes applied to one otherwise clean row.
_CATALOG_FIELD_DEFECTS: dict[str, dict[str, Any]] = {
    "salary_min_exceeds_max": {"salary_min": 9000.0, "salary_max": 4000.0},
    "unknown_currency": {"salary_currency": "XYZ"},
    "invalid_work_mode": {"work_mode": "teleport"},
    "invalid_experience_level": {"experience_level": "wizard"},
    "invalid_deadline": {"application_deadline": "not-a-date"},
    "expired_deadline_active": {"application_deadline": "2025-11-30", "is_active": True},
    "expired_deadline": {"application_deadline": "2025-10-01", "is_active": False},
    "empty_title": {"title": "   "},
    "empty_skills": {"required_skills": [], "preferred_skills": []},
    "empty_location": {"city": None, "region": None, "country": None},
}

#: Defects that need more than a single mutated row/record to express.
_STRUCTURAL_DEFECTS: tuple[str, ...] = (
    "duplicate_job_id",
    "empty_job_id",
    "duplicate_scenario_id",
    "empty_scenario_id",
    "empty_scenario_turns",
    "inconsistent_no_match_expectation",
    "missing_hard_constraint_reference",
    "missing_relevance_label",
)

#: The full defect catalogue the property draws combinations from.
DEFECT_CATALOGUE: tuple[str, ...] = (*_CATALOG_FIELD_DEFECTS, *_STRUCTURAL_DEFECTS)

#: Defects the validator records as ``warning`` by design: an expired posting that is
#: correctly flagged inactive, and missing-but-optional references.
_WARNING_DEFECTS: frozenset[str] = frozenset({
    "expired_deadline", "missing_hard_constraint_reference", "missing_relevance_label",
})


def _clarification_scenario(scenario_id: str) -> dict[str, Any]:
    """A scenario that needs neither a relevance label nor a constraint reference."""
    return {**_recommendation_scenario(scenario_id), "clarification_expected": True,
            "expects": {"response_type": "clarification"}}


def _inject_defects(
    defects: list[str], clean_row: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], set[tuple[str, str]]]:
    """Build a dataset that is clean apart from exactly ``defects``.

    Returns ``(jobs, scenarios, relevance_labels, expected)`` where ``expected`` is the
    set of ``(identifier, violation_type)`` pairs the validator must report -- one per
    injected defect, each carried by its own record so the offending identifier is
    unambiguous.
    """
    jobs: list[dict[str, Any]] = [dict(clean_row)]
    scenarios: list[dict[str, Any]] = [_recommendation_scenario("SC-DQ-CLEAN")]
    labelled: list[str] = ["SC-DQ-CLEAN"]
    expected: set[tuple[str, str]] = set()

    for n, defect in enumerate(defects):
        job_id, sid = f"job-dq-{n}", f"SC-DQ-{n}"
        if defect in _CATALOG_FIELD_DEFECTS:
            jobs.append(_mutate(clean_row, job_id=job_id, **_CATALOG_FIELD_DEFECTS[defect]))
            expected.add((job_id, defect))
        elif defect == "duplicate_job_id":
            jobs.append(_mutate(clean_row, job_id=job_id))
            jobs.append(_mutate(clean_row, job_id=job_id))
            expected.add((job_id, defect))
        elif defect == "empty_job_id":
            jobs.append(_mutate(clean_row, job_id=""))
            expected.add((f"<catalog row {len(jobs) - 1}>", defect))
        elif defect == "duplicate_scenario_id":
            scenarios.append(_recommendation_scenario(sid))
            scenarios.append(_recommendation_scenario(sid))
            labelled.append(sid)
            expected.add((sid, defect))
        elif defect == "empty_scenario_id":
            scenarios.append(_clarification_scenario(""))
            expected.add((f"<scenario row {len(scenarios) - 1}>", defect))
        elif defect == "empty_scenario_turns":
            scenarios.append(_mutate(_recommendation_scenario(sid), turns=[]))
            labelled.append(sid)
            expected.add((sid, defect))
        elif defect == "inconsistent_no_match_expectation":
            scenarios.append(_mutate(
                _recommendation_scenario(sid), no_match_expected=True,
                expects={"response_type": "recommendation", "blocking": "salary_min"},
            ))
            expected.add((sid, defect))
        elif defect == "missing_hard_constraint_reference":
            scenario = _no_match_scenario("Data analyst in Kuala Lumpur, at least RM50000.",
                                          sid)
            scenario["expects"] = {"response_type": "no_match"}
            scenarios.append(scenario)
            expected.add((sid, defect))
        else:  # missing_relevance_label -- the one scenario deliberately left unlabelled
            scenarios.append(_recommendation_scenario(sid))
            expected.add((sid, defect))
    return jobs, scenarios, labelled, expected


# Feature: cmjcc-experiment-readiness, Property 24: Data-quality validation flags every
# injected defect
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(defects=st.lists(st.sampled_from(DEFECT_CATALOGUE), min_size=1,
                        max_size=len(DEFECT_CATALOGUE), unique=True))
def test_property_every_injected_defect_is_flagged(clean_row, config, defects) -> None:
    """Any combination of injected defects is reported, and nothing else is.

    Each defect rides its own record inside an otherwise clean catalog/scenario set, so
    the validator must return exactly one finding per defect, keyed by that record's
    identifier and violation type -- nothing missed, nothing invented against the clean
    records.

    **Validates: Requirements 17.1, 17.2, 17.4**
    """
    jobs, scenarios, labelled, expected = _inject_defects(defects, clean_row)

    # verify_no_match=False: the no-match replay drives the orchestration path and is
    # covered by the dedicated tests above; the defect catalogue here is static data.
    report = validate_dataset(jobs, scenarios, config=config,
                              relevance_labels=labelled, verify_no_match=False)

    observed = {(f.identifier, f.violation_type) for f in report}
    assert expected - observed == set(), f"missed defects; got {report.summary()}"
    assert observed - expected == set(), f"spurious findings; got {report.summary()}"

    severities = {f.violation_type: f.severity for f in report}
    for defect in defects:
        want = "warning" if defect in _WARNING_DEFECTS else "error"
        assert severities[defect] == want
    assert report.ok == all(defect in _WARNING_DEFECTS for defect in defects)
    # The clean baseline records are never implicated.
    assert not {_CLEAN_JOB_ID, "SC-DQ-CLEAN"} & {f.identifier for f in report}
