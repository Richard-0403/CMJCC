"""Report sections for the process measures, the failure paths and the pairing wording.

These tests cover the three report-side changes that the readiness checklist asks for and
that nothing previously exercised:

- **§5.4 / clarification efficiency (checklist item 3).** The efficiency table carries the
  response-turn distribution (median, IQR) and reaches the rendered report, and the prose
  states that response turns are read jointly with task success.
- **§6.3 / two-family Holm (checklist item 6).** ``_contrib_table`` filters by outcome
  family, so the secondary process-measure Δ rows render in their own sub-table without
  appearing in -- or perturbing -- §6.1/§6.2.
- **§8 / scenario-level pairing (checklist item 4).** The statistical section no longer
  claims run-level discordant pairs and surfaces the pairing provenance.
- **§10 / fault injection (checklist item 5).** The failure-metric table reads N/A (never
  0.0 or 1.000) when the loaded runs injected nothing, and the fault-class enumeration
  names only classes the repo genuinely covers.

``generate_markdown`` reads nothing but the assembled data dict, so everything here is a
pure render over a small fixture.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from jobrec_eval.metrics_extra import clarification_efficiency, failure_metrics
from jobrec_eval.report import generate_markdown

_PRIMARY = ["ndcg_at_5", "hcsr", "task_success", "grounding", "mean_violation_count",
            "turn_count"]
_SECONDARY = ["response_turns", "clarification_efficiency"]


# --------------------------------------------------------------------- fixtures
def _contrib_rows(subsets: tuple[str, ...]) -> list[dict]:
    """Contribution rows for both families, tagged with a ``family`` column."""
    rows = []
    for subset in subsets:
        for family, metrics in (("primary", _PRIMARY), ("secondary", _SECONDARY)):
            for i, metric in enumerate(metrics):
                rows.append({
                    "subset": subset, "family": family, "metric": metric,
                    "base_mean": 0.9, "other_mean": 0.7, "delta": 0.2,
                    "ci_low": 0.1, "ci_high": 0.3,
                    "p_value": 0.01 * (i + 1), "p_value_holm": 0.02 * (i + 1),
                    "effect_size": 0.5, "effect_type": "cohens_dz", "n_pairs": 6,
                })
    return rows


def _report_data(**overrides) -> dict:
    metric_keys = ["ndcg_at_5", "precision_at_5", "hcsr", "task_success", "grounding",
                   "handoff_success", "turn_count", "total_latency_ms"]
    data = {
        "experiment": {
            "experiment_id": "exp-sections", "reference_date": "2026-01-01",
            "catalog_snapshot_id": "catalog-2026-01", "catalog_hash": "abc123def456789",
            "variants": ["full", "no_memory", "no_context"],
            "scenario_count": 2, "repeat_count": 1, "run_count": 6,
            "bootstrap_iterations": 200, "bootstrap_seed": 2026, "eval_version": "1.0.0",
        },
        "oracle_version": "1.0.0",
        "scenario_type_counts": {"multi_turn": 2},
        "n_memory_dependent": 2,
        "n_context_dependent": 1,
        "variant_summary": [{"variant": v, **{f"{k}_mean": 0.5 for k in metric_keys}}
                            for v in ["full", "no_memory", "no_context"]],
        "scenario_variant": [
            {"scenario_id": f"s{i}", "scenario_type": "multi_turn", "variant": v,
             "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9,
             "response_turns": 2.0, "clarification_efficiency": -2.0}
            for i in range(2) for v in ["full", "no_memory", "no_context"]
        ],
        "memory_contribution": _contrib_rows(("all", "memory_dependent")),
        "context_contribution": _contrib_rows(("all", "context_dependent")),
        "overall_comparisons": [
            {"metric": "task_success", "base": "full", "other": "no_memory",
             "n_pairs": 12, "scenario_count": 12, "total_run_count": 24,
             "repeats_per_scenario": 1, "valid_pairs": 12, "discordant_pairs": 3,
             "p_value": 0.25, "delta": 0.25, "ci_low": 0.08, "ci_high": 0.5},
            {"metric": "ndcg_at_5", "base": "full", "other": "no_memory",
             "n_pairs": 12, "delta": 0.1, "ci_low": 0.02, "ci_high": 0.18},
        ],
        "error_summary": "Runs: 6; system failures: 0; task-unsuccessful runs: 1.",
    }
    data.update(overrides)
    return data


def _section(md: str, heading: str) -> str:
    """Text of one Markdown section, up to the next heading of the same or higher level."""
    start = md.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = md[start + len(heading):]
    ends = [m.start() for lv in range(2, level + 1)
            if (m := re.search(rf"^{'#' * lv} ", rest, re.MULTILINE))]
    return rest[:min(ends)] if ends else rest


def _table_rows(section: str) -> dict[str, list[str]]:
    """Body of the first Markdown table of a section as ``{first cell: remaining cells}``."""
    lines = section.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    i += 1  # header row
    if i < len(lines) and set(lines[i].replace("|", "").strip()) <= {"-"}:
        i += 1  # separator row
    rows: dict[str, list[str]] = {}
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip("|").split("|")]
        rows[cells[0]] = cells[1:]
        i += 1
    return rows


# ------------------------------------------- checklist item 3: response-turn distribution
def _efficiency_frame(turns: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame([{
        "run_id": f"r{i}", "scenario_id": f"s{i}", "variant": "full",
        "acceptable_slots": "", "clarification_target": "",
        "clarification_expected": False, "response_turns": t,
    } for i, t in enumerate(turns)])


def test_efficiency_table_reports_median_and_iqr_of_response_turns():
    """The distribution columns are computed from ``response_turns`` (checklist item 3)."""
    row = clarification_efficiency(_efficiency_frame([1.0, 2.0, 3.0, 4.0])).iloc[0]

    assert row["response_turns_n"] == 4
    assert row["median_response_turns"] == pytest.approx(2.5)
    assert row["q1_response_turns"] == pytest.approx(1.5)
    assert row["q3_response_turns"] == pytest.approx(3.5)
    assert row["iqr_response_turns"] == pytest.approx(2.0)
    # The pre-existing classification/score columns are untouched.
    assert row["runs"] == 4
    assert row["efficiency_score"] == pytest.approx(-2.5)


def test_efficiency_table_reports_na_when_no_run_recorded_response_turns():
    """A trace-less bundle set reads N/A, never a fabricated 0 (module-wide convention)."""
    no_values = clarification_efficiency(_efficiency_frame([None, None])).iloc[0]
    assert no_values["response_turns_n"] == 0
    for column in ("median_response_turns", "q1_response_turns", "q3_response_turns",
                   "iqr_response_turns"):
        assert pd.isna(no_values[column])

    no_column = _efficiency_frame([1.0]).drop(columns=["response_turns"])
    assert pd.isna(clarification_efficiency(no_column).iloc[0]["median_response_turns"])


def test_single_run_has_a_zero_iqr_rather_than_an_undefined_one():
    row = clarification_efficiency(_efficiency_frame([3.0])).iloc[0]
    assert row["median_response_turns"] == pytest.approx(3.0)
    assert row["iqr_response_turns"] == pytest.approx(0.0)


def test_report_renders_the_clarification_efficiency_table_and_joint_reading():
    """§5.4 carries the six checklist figures and states the joint interpretation rule."""
    efficiency = clarification_efficiency(_efficiency_frame([1.0, 2.0, 3.0]))
    data = _report_data(
        clarification_efficiency=efficiency.to_dict(orient="records"),
        clarification_metrics=[{"variant": "full", "necessary_recall": 1.0, "repeated": 2,
                                "answered_rate": 1.0}],
    )
    section = _section(generate_markdown(data), "### 5.4")

    # Asking fewer questions with a wrong answer must be denied explicitly (item 3).
    squeezed = " ".join(section.split())
    assert "jointly with task success" in squeezed
    assert "asking" in squeezed and "fewer questions" in squeezed
    assert "NOT efficiency" in squeezed
    assert "skip penalty" in squeezed
    # ... and so must abandoning a dialogue after asking (R7.4/R7.5).
    assert "Abandoning a dialogue" in squeezed
    assert "unresolved-dialogue penalty" in squeezed

    row = _table_rows(section)["full"]
    # The tier composition and the median are the primary reading; the mean is kept but
    # explicitly labelled a penalty scale rather than a magnitude.
    assert "penalty scale, not a rate" in squeezed
    header = ["MedTurns", "IQR(Q1-Q3)", "NecRecall", "NecAsked", "UnnecAsked",
              "NecMissed", "RepeatGuard", "Abandoned", "AnsweredRate",
              "Tier res/aband/skip", "MedEff", "EffScore", "n"]
    cells = dict(zip(header, row, strict=True))
    assert cells["MedTurns"] == "2.00"
    assert cells["IQR(Q1-Q3)"] == "2.00 (1.00-3.00)"
    assert cells["NecRecall"] == "1.000"      # necessary clarification recall
    assert cells["UnnecAsked"] == "0"         # unnecessary clarification count
    assert cells["RepeatGuard"] == "2"        # repeated-slot guard activations
    assert cells["Abandoned"] == "0"          # asked-then-unresolved dialogues
    assert cells["AnsweredRate"] == "1.000"   # share of asks the user answered
    assert cells["EffScore"] == "-2.00"       # clarification efficiency score
    assert cells["n"] == "3"


def test_scenario_type_table_carries_the_process_measures():
    """§7 aggregates response turns and clarification efficiency per scenario type."""
    section = _section(generate_markdown(_report_data()), "## 7.")
    assert "| Turns | ClarEff |" in section
    for line in section.splitlines():
        if line.startswith("| multi_turn | full |"):
            assert line.rstrip().endswith("| 2.00 | -2.00 | 2 |")
            break
    else:
        pytest.fail("no scenario-type row rendered for full")


def test_scenario_type_table_still_renders_without_the_process_columns():
    """A frame predating the process measures renders them as N/A instead of raising."""
    data = _report_data(scenario_variant=[
        {"scenario_id": "s0", "scenario_type": "multi_turn", "variant": "full",
         "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9},
    ])
    section = _section(generate_markdown(data), "## 7.")
    assert "| multi_turn | full |" in section
    assert "| N/A | N/A | 1 |" in section


# --------------------------------------------- checklist item 6: two outcome families
def test_secondary_family_rows_render_apart_from_the_primary_tables():
    """§6.1/§6.2 show primary metrics only; §6.3 shows the process measures only."""
    md = generate_markdown(_report_data())

    for heading in ("### 6.1", "### 6.2"):
        metrics = set(_table_rows(_section(md, heading)))
        assert metrics == set(_PRIMARY), heading
        assert not metrics & set(_SECONDARY), heading

    secondary = _section(md, "### 6.3")
    assert set(_table_rows(secondary)) == set(_SECONDARY)
    # All four sub-tables (memory/context x subset/all) are rendered.
    assert secondary.count("| metric | full | other |") == 4


def test_secondary_family_does_not_disturb_the_primary_holm_values():
    """The rendered primary p(Holm) cells are exactly the ones in the primary rows.

    The guard for the methodological constraint behind checklist item 6: the secondary
    family is corrected separately, so its presence must not change a single primary cell.
    Rendering with and without the secondary rows must produce identical §6.1 tables.
    """
    both = _report_data()
    primary_only = _report_data(
        memory_contribution=[r for r in both["memory_contribution"]
                             if r["family"] == "primary"],
        context_contribution=[r for r in both["context_contribution"]
                              if r["family"] == "primary"],
    )
    for heading in ("### 6.1", "### 6.2"):
        assert (_table_rows(_section(generate_markdown(both), heading))
                == _table_rows(_section(generate_markdown(primary_only), heading)))


def test_report_states_that_holm_is_applied_within_each_family():
    md = " ".join(generate_markdown(_report_data()).split())
    assert "within each outcome family independently" in md
    assert "secondary_outcomes" in md
    assert "leaves every primary p-value unchanged" in md


def test_secondary_family_keeps_the_mechanism_contribution_framing():
    """§6.3 extends the existing framing rather than inventing a new claim (R32.5/32.6)."""
    section = " ".join(_section(generate_markdown(_report_data()), "### 6.3").split())
    assert ("contribution of that framework mechanism under the controlled prototype "
            "instantiation") in section
    assert "superior" not in section.lower()


# ------------------------------------------ checklist item 4: scenario-level pairing
def test_statistical_section_no_longer_claims_run_level_pairing():
    md = generate_markdown(_report_data())
    section = _section(md, "## 8.")
    squeezed = " ".join(section.split())

    assert "run-level discordant pairs" not in md
    assert "McNemar on run-level" not in md
    assert "scenario-level paired binary outcomes" in squeezed
    assert "`scenario_id` is the independent analysis unit" in squeezed
    assert "never** treated as an independent sample" in squeezed
    assert "majority vote" in squeezed and "ties" in squeezed
    assert "number of **validly paired scenarios**, not the number of runs" in squeezed
    assert "Deterministic runs default to one repeat per scenario" in squeezed


def test_statistical_section_surfaces_the_pairing_provenance():
    """The reader can see n_pairs is a scenario count, from compare()'s own bookkeeping."""
    rows = _table_rows(_section(generate_markdown(_report_data()), "## 8."))
    header = ["scenarios", "runs", "repeats/scenario", "valid pairs", "discordant",
              "n_pairs", "p"]
    cells = dict(zip(header, rows["full vs no_memory"], strict=True))

    assert cells["scenarios"] == "12"
    assert cells["runs"] == "24"
    assert cells["repeats/scenario"] == "1"
    assert cells["valid pairs"] == cells["n_pairs"] == "12"
    assert cells["discordant"] == "3"
    # Only the binary metric is pairing-relevant; continuous rows are not listed.
    assert set(rows) == {"full vs no_memory"}


# ----------------------------------------- checklist item 5: fault-injection section
def _run_metrics(**flags) -> pd.DataFrame:
    base = {"failure_injected": False, "failure_detected": False,
            "recoverable": False, "recovered": False}
    base.update(flags)
    return pd.DataFrame([{"run_id": "r1", "variant": "full", **base}])


class _Bundle:
    def __init__(self, claims=None, handoffs=None):
        self.claims = claims or []
        self.handoffs = handoffs or []


def test_failure_metrics_table_shape_and_na_when_nothing_was_injected():
    """Four tidy rows; the injection rates carry an empty denominator and a None value."""
    bundles = [_Bundle(
        claims=[{"claim_type": "job_attribute", "support_status": "supported"}],
        handoffs=[{"validation_passed": True, "status": "completed"}])]
    table = failure_metrics(_run_metrics(), bundles)

    assert list(table.columns) == ["metric", "source", "numerator", "denominator", "value"]
    assert list(table["metric"]) == ["failure_detection_rate", "recovery_success_rate",
                                     "grounding_rate", "handoff_success_rate"]
    assert list(table["source"]) == ["run_metrics", "run_metrics",
                                     "run_bundles", "run_bundles"]
    by_metric = table.set_index("metric")
    for rate in ("failure_detection_rate", "recovery_success_rate"):
        assert by_metric.loc[rate, "denominator"] == 0
        # None is the in-frame spelling of "no value"; pandas stores it as NaN, which is
        # what the CSV writes as an empty cell.
        assert pd.isna(by_metric.loc[rate, "value"])
    # A normal experiment legitimately grounds everything -- that is not a defect.
    assert by_metric.loc["grounding_rate", "value"] == pytest.approx(1.0)
    assert by_metric.loc["handoff_success_rate", "value"] == pytest.approx(1.0)


def test_failure_metrics_table_reports_measured_rates_when_faults_were_injected():
    table = failure_metrics(
        pd.concat([_run_metrics(failure_injected=True, failure_detected=True,
                                recoverable=True, recovered=True),
                   _run_metrics(failure_injected=True)], ignore_index=True),
        [_Bundle(claims=[{"claim_type": "job_attribute", "support_status": "unsupported"}])],
    ).set_index("metric")

    assert table.loc["failure_detection_rate", ["numerator", "denominator"]].tolist() == [1, 2]
    assert table.loc["failure_detection_rate", "value"] == pytest.approx(0.5)
    assert table.loc["recovery_success_rate", "value"] == pytest.approx(1.0)
    assert table.loc["grounding_rate", "value"] == pytest.approx(0.0)
    assert table.loc["handoff_success_rate", "denominator"] == 0
    assert pd.isna(table.loc["handoff_success_rate", "value"])


def test_failure_metrics_table_reports_none_denominators_without_instrumentation():
    """A frame with no failure columns reads N/A rather than a rate over assumed falses."""
    table = failure_metrics(pd.DataFrame([{"run_id": "r1", "variant": "full"}]), []
                            ).set_index("metric")
    for rate in ("failure_detection_rate", "recovery_success_rate"):
        assert pd.isna(table.loc[rate, "denominator"])
        assert pd.isna(table.loc[rate, "value"])


def test_fault_injection_section_is_separate_and_reads_na_when_nothing_was_injected():
    """§10 is its own section, defends grounding = 1.000 and never prints 0.0/1.000 for N/A."""
    bundles = [_Bundle(
        claims=[{"claim_type": "job_attribute", "support_status": "supported"}],
        handoffs=[{"validation_passed": True, "status": "completed"}])]
    table = failure_metrics(_run_metrics(), bundles)
    md = generate_markdown(_report_data(
        failure_metrics=table.to_dict(orient="records")))
    section = _section(md, "## 10. Fault-Injection Robustness")
    squeezed = " ".join(section.split())

    # Separate robustness experiment, not part of the main results.
    assert "separate robustness experiment" in md
    assert "kept separate from the main experiment results" in squeezed
    assert "grounding rate of 1.000 in the main experiment is legitimate, not a defect" in squeezed
    assert "Failure samples are NOT mixed into the main experiment" in squeezed

    # N/A, spelled out -- not 0.0 and not 1.000.
    assert "N/A (empty denominator)" in squeezed
    assert "injected no faults, so the detection and recovery rates read N/A, not 0.0 and not 1.000" in squeezed
    for rate in ("failure_detection_rate", "recovery_success_rate"):
        row = next(line for line in section.splitlines() if line.startswith(f"| {rate} "))
        assert "0.000" not in row and "1.000" not in row


def test_fault_injection_section_states_measured_counts_when_faults_were_injected():
    table = failure_metrics(
        _run_metrics(failure_injected=True, failure_detected=True), [])
    section = _section(generate_markdown(_report_data(
        failure_metrics=table.to_dict(orient="records"))), "### 10.1")
    assert "The loaded runs injected 1 fault(s), of which 1 were detected." in " ".join(
        section.split())
    assert "recovery rate reads N/A" in " ".join(section.split())


def test_fault_class_enumeration_names_only_classes_the_repo_covers():
    """Every checklist fault class appears, and each one exists in the repo's suite."""
    section = _section(generate_markdown(_report_data()), "### 10.2")
    for fault_class in ("invalid evidence id", "missing evidence source",
                        "wrong-field evidence", "unsupported salary claim",
                        "unsupported location claim", "unsupported skill claim",
                        "schema-invalid handoff", "missing-field handoff",
                        "agent exception", "timeout + retry",
                        "partial failure + recovery"):
        assert f"| {fault_class} |" in section, fault_class
    # The suite that backs the table is named, so the claim is checkable.
    assert "tests/support/fault_injection.py" in section
    assert "tests/unit/test_failure_paths.py" in section
    assert "tests/integration/test_failure_metrics.py" in section


# ------------------------------------- checklist item 14: data quality in the artifacts
def test_data_quality_counts_are_reported_in_the_dataset_section():
    data = _report_data(data_quality={
        "reference_date": "2026-01-01", "job_count": 200, "scenario_count": 12,
        "ok": True, "error_count": 0, "warning_count": 4, "info_count": 27,
        "checks_run": ["catalog_records", "no_match_scenarios_unsatisfiable"],
        "checks_skipped": {"scenario_relevance_labels": "no relevance labels supplied"},
        "counts_by_violation_type": {"expired_deadline": 27},
    })
    section = " ".join(_section(generate_markdown(data), "### 3.1").split())

    assert "200 catalog job(s) and 12 scenario(s)" in section
    assert "**0 error(s), 4 warning(s), 27 acknowledged test fixture(s)**" in section
    assert "no_match_scenarios_unsatisfiable" in section
    assert "scenario_relevance_labels (no relevance labels supplied)" in section
    assert "data_quality_report.json" in section


def test_dataset_section_says_so_when_no_data_quality_report_exists():
    section = _section(generate_markdown(_report_data()), "### 3.1")
    assert "No data-quality report" in section
