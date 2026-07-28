"""Reporting-honesty guards for the §5 denominators and the §7 variant coverage.

Two rendering defects motivated this module. Neither touched a metric, a statistic or a
CSV schema -- both hid information the metric tables already carried, which is exactly
the kind of defect a reader cannot detect from the report alone:

- **§5 printed only the ``<metric>_mean`` columns.** A variant that abandons a dialogue
  returns no ranking list, so its ranking metrics are averaged over fewer scenarios --
  and the scenarios that drop out are the hardest ones. Printed without its denominator,
  such a variant can show a *higher* ranking mean than ``full`` while having answered a
  strictly easier subset. The ``<metric>_n`` columns are now rendered under the table,
  with prose saying that ``task_success`` is the metric defined on all scenarios.
- **§7 filtered to ``full``/``no_memory``/``no_context``.** Both dialogue baselines were
  dropped from the per-scenario-type breakdown with no note, including the clarification
  rows where their behaviour actually shows. The table now renders every variant the
  frame carries, in the canonical order, so a hardcoded filter cannot come back.

``generate_markdown`` reads nothing but the assembled data dict, so everything here is a
pure render over a small fixture.
"""

from __future__ import annotations

import re

from jobrec_eval.report import generate_markdown

_METRIC_KEYS = ["ndcg_at_5", "precision_at_5", "hcsr", "task_success", "grounding",
                "handoff_success", "turn_count", "total_latency_ms"]
_VARIANTS = ["full", "profile_only", "one_shot", "no_memory", "no_context"]
_SCENARIO_TYPES = ["clarification", "preference_change"]

#: Ranking denominators that differ per variant, mirroring the shape of the official run:
#: ``full`` answers the most scenarios, the dialogue baselines fewer. ``task_success`` is
#: defined on every scenario, so its denominator is the full scenario count everywhere.
_RANKING_N = {"full": 37, "profile_only": 24, "one_shot": 21, "no_memory": 28,
              "no_context": 38}
_SCENARIO_COUNT = 42


def _contrib_rows(subsets: tuple[str, ...]) -> list[dict]:
    return [{
        "subset": subset, "metric": metric, "base_mean": 0.8, "other_mean": 0.6,
        "delta": 0.2, "ci_low": 0.05, "ci_high": 0.35, "p_value": 0.04,
        "p_value_holm": 0.08, "effect_size": 0.6, "effect_type": "cohens_dz",
        "n_pairs": 6,
    } for subset in subsets for metric in _METRIC_KEYS]


def _variant_summary(with_n: bool = True) -> list[dict]:
    rows = []
    for v in _VARIANTS:
        row: dict[str, object] = {"variant": v}
        for key in _METRIC_KEYS:
            # one_shot scores highest on P@5 -- on the smallest denominator, which is the
            # reading the disclosure exists to prevent.
            row[f"{key}_mean"] = 1.0 if (v == "one_shot" and key == "precision_at_5") else 0.5
            if with_n:
                ranking = key in ("ndcg_at_5", "precision_at_5", "hcsr")
                row[f"{key}_n"] = _RANKING_N[v] if ranking else _SCENARIO_COUNT
        rows.append(row)
    return rows


def _report_data(**overrides) -> dict:
    data = {
        "experiment": {
            "experiment_id": "exp-denominators", "reference_date": "2026-01-01",
            "catalog_snapshot_id": "catalog-2026-01", "catalog_hash": "abc123def456789",
            "variants": list(_VARIANTS), "scenario_count": _SCENARIO_COUNT,
            "repeat_count": 1, "run_count": _SCENARIO_COUNT * len(_VARIANTS),
            "bootstrap_iterations": 200, "bootstrap_seed": 2026, "eval_version": "1.0.0",
        },
        "oracle_version": "1.0.0",
        "scenario_type_counts": {t: 2 for t in _SCENARIO_TYPES},
        "n_memory_dependent": 2,
        "n_context_dependent": 1,
        "variant_summary": _variant_summary(),
        "scenario_variant": [
            {"scenario_id": f"s{i}", "scenario_type": t, "variant": v,
             "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9,
             "response_turns": 2.0, "clarification_efficiency": -2.0}
            for i, t in enumerate(_SCENARIO_TYPES) for v in _VARIANTS
        ],
        "memory_contribution": _contrib_rows(("all", "memory_dependent")),
        "context_contribution": _contrib_rows(("all", "context_dependent")),
        "overall_comparisons": [
            {"metric": "ndcg_at_5", "base": "full", "other": "profile_only",
             "delta": 0.345, "ci_low": 0.214, "ci_high": 0.484, "n_pairs": 23},
            {"metric": "ndcg_at_5", "base": "full", "other": "one_shot",
             "delta": 0.005, "ci_low": 0.0, "ci_high": 0.014, "n_pairs": 21},
            {"metric": "task_success", "base": "full", "other": "profile_only",
             "delta": 0.833, "ci_low": 0.714, "ci_high": 0.929, "n_pairs": 42},
        ],
        "error_summary": "Runs: 210; system failures: 0; task-unsuccessful runs: 40.",
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


def _tables(section: str) -> list[dict[str, list[str]]]:
    """Markdown tables of a section as ``[{first cell: remaining cells}]``, headers kept."""
    tables: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            if current:
                tables.append(current)
                current = {}
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= {"-"}:
            continue  # separator row
        current[cells[0]] = cells[1:]
    if current:
        tables.append(current)
    return tables


def _rows(section: str, prefix: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip("|").split("|")]
            for line in section.splitlines() if line.startswith(prefix)]


# ------------------------------------------------- defect 1: §5 per-metric denominators
def test_variant_summary_keeps_its_nine_compact_columns():
    """The means table is unchanged: variant + the eight metric columns, nothing added."""
    means = _tables(_section(generate_markdown(_report_data()), "## 5. Overall Results"))[0]

    assert means["variant"] == ["NDCG@5", "P@5", "HCSR", "TaskSucc", "Grounding",
                                "Handoff", "Turns", "Lat(ms)"]
    assert list(means)[1:] == _VARIANTS
    assert means["one_shot"][1] == "1.000"  # the flattering P@5 mean is still printed


def test_variant_summary_discloses_the_denominator_behind_every_printed_metric():
    """Each mean in §5 gets its ``<metric>_n`` printed under it, per variant."""
    section = _section(generate_markdown(_report_data()), "## 5. Overall Results")
    means, denominators = _tables(section)[:2]

    assert denominators["variant"] == [f"n({c})" for c in means["variant"]]
    assert list(denominators)[1:] == _VARIANTS
    for v in _VARIANTS:
        ndcg_n, p_at_5_n, hcsr_n, task_n = denominators[v][:4]
        assert ndcg_n == p_at_5_n == hcsr_n == str(_RANKING_N[v])
        assert task_n == str(_SCENARIO_COUNT)


def test_a_reduced_ranking_denominator_is_visible_next_to_the_higher_mean():
    """The survivorship case is disclosed: best P@5, smallest n, both on the page."""
    section = _section(generate_markdown(_report_data()), "## 5. Overall Results")
    means, denominators = _tables(section)[:2]
    p_at_5 = means["variant"].index("P@5")

    assert means["one_shot"][p_at_5] > means["full"][p_at_5]           # 1.000 vs 0.500
    assert int(denominators["one_shot"][p_at_5]) < int(denominators["full"][p_at_5])
    # Every variant whose ranking n differs from full's is disclosed, not just one.
    differing = [v for v in _VARIANTS if _RANKING_N[v] != _RANKING_N["full"]]
    assert differing
    for v in differing:
        assert denominators[v][p_at_5] != denominators["full"][p_at_5]


def test_missing_denominator_columns_render_empty_cells_instead_of_raising():
    """A frame predating the ``_n`` columns still renders (defensive as ``_col_mean``)."""
    section = _section(
        generate_markdown(_report_data(variant_summary=_variant_summary(with_n=False))),
        "## 5. Overall Results")
    denominators = _tables(section)[1]

    assert list(denominators)[1:] == _VARIANTS
    assert all(cell == "" for cell in denominators["full"])
    # No fabricated count anywhere in the block.
    assert "n(NDCG@5)" in denominators["variant"][0]


def test_section_5_warns_that_a_smaller_denominator_is_survivorship_not_ranking():
    """The prose names the affected metrics and points the comparison at task success."""
    squeezed = " ".join(
        _section(generate_markdown(_report_data()), "## 5. Overall Results").split())

    assert "NDCG@5, P@5, HCSR and the graded-relevance mean" in squeezed
    assert "averaged over the scenarios where the variant actually returned a ranked list" \
        in squeezed
    assert "SMALLER denominator" in squeezed
    assert "hardest" in squeezed
    assert "survivorship, NOT better ranking" in squeezed
    assert "must never be read as that variant out-ranking `full`" in squeezed
    assert "`task_success` is the metric defined on every scenario" in squeezed
    assert "`task_success` is the column to compare across variants" in squeezed


def test_denominator_warning_carries_no_run_specific_values():
    """The prose is regenerated for other runs, so it names no count and no baseline."""
    prose = _section(generate_markdown(_report_data()), "## 5. Overall Results")
    prose = prose[prose.index("**Read the denominators"):prose.index("### 5.x")]

    assert "one_shot" not in prose
    assert "profile_only" not in prose
    assert not re.search(r"\b\d+\b", prose.replace("NDCG@5", "").replace("P@5", ""))


def test_dialogue_baseline_bullets_report_the_pair_count_behind_each_delta():
    """§5.x prints ``n_pairs`` next to Δ and CI, and denies the equivalence reading."""
    section = _section(generate_markdown(_report_data()), "### 5.x")
    squeezed = " ".join(section.split())

    assert ("NDCG@5, full vs profile_only: Δ=+0.345, 95% CI [+0.214, +0.484] "
            "(CI excludes 0), n=23 paired scenarios.") in squeezed
    assert ("NDCG@5, full vs one_shot: Δ=+0.005, 95% CI [+0.000, +0.014] "
            "(CI includes 0), n=21 paired scenarios.") in squeezed
    assert ("Task success, full vs profile_only: Δ=+0.833, 95% CI [+0.714, +0.929] "
            "(CI excludes 0), n=42 paired scenarios.") in squeezed
    assert ('"not estimable on the scenarios that variant abandoned", not evidence of '
            "equivalence") in squeezed


def test_dialogue_baseline_bullets_survive_a_comparison_without_a_pair_count():
    """A row lacking ``n_pairs`` keeps its Δ phrase rather than printing a wrong count."""
    data = _report_data(overall_comparisons=[
        {"metric": "ndcg_at_5", "base": "full", "other": "one_shot",
         "delta": 0.005, "ci_low": 0.0, "ci_high": 0.014}])
    section = _section(generate_markdown(data), "### 5.x")

    assert "NDCG@5, full vs one_shot: Δ=+0.005" in section
    assert "n=" not in section
    assert "NDCG@5, full vs profile_only: N/A." in section


# ------------------------------------------------------ defect 2: §7 variant coverage
def test_scenario_type_table_renders_every_variant_in_the_frame():
    """Regression guard: no variant may be filtered out of the per-scenario-type table."""
    section = _section(generate_markdown(_report_data()), "## 7. Results by Scenario Type")

    for stype in _SCENARIO_TYPES:
        rendered = [r[1] for r in _rows(section, f"| {stype} |")]
        assert rendered == _VARIANTS, stype  # all five, canonical order, none dropped
    assert len(_rows(section, "| ")) == len(_SCENARIO_TYPES) * len(_VARIANTS) + 1  # + header


def test_scenario_type_table_groups_by_scenario_type_in_canonical_variant_order():
    """scenario_type is the outer key, variant the inner one -- not alphabetical."""
    section = _section(generate_markdown(_report_data()), "## 7. Results by Scenario Type")
    body = [(r[0], r[1]) for r in _rows(section, "| ")[1:]]

    assert body == [(t, v) for t in sorted(_SCENARIO_TYPES) for v in _VARIANTS]
    assert body != sorted(body)  # alphabetical variant order would put full third


def test_scenario_type_table_keeps_the_process_columns_for_the_added_variants():
    """Turns, ClarEff and n are rendered for every variant, not only the ablation three."""
    section = _section(generate_markdown(_report_data()), "## 7. Results by Scenario Type")

    assert "| Turns | ClarEff | n |" in section
    for row in _rows(section, "| clarification |"):
        assert len(row) == 9
        assert row[-3:] == ["2.00", "-2.00", "1"]


def test_an_unrecognised_variant_is_appended_rather_than_dropped():
    """A variant outside the canonical list still reaches §5 and §7, ordered last."""
    data = _report_data(
        variant_summary=[*_variant_summary(), {"variant": "zz_probe",
                                               **{f"{k}_mean": 0.4 for k in _METRIC_KEYS}}],
        scenario_variant=[*_report_data()["scenario_variant"],
                          {"scenario_id": "s9", "scenario_type": "clarification",
                           "variant": "zz_probe", "ndcg_at_5": 0.1, "hcsr": 0.2,
                           "task_success": 0.0, "grounding": 0.3}],
    )
    md = generate_markdown(data)

    assert list(_tables(_section(md, "## 5. Overall Results"))[0])[1:] == \
        [*_VARIANTS, "zz_probe"]
    section_7 = _section(md, "## 7. Results by Scenario Type")
    assert [r[1] for r in _rows(section_7, "| clarification |")] == [*_VARIANTS, "zz_probe"]


# ------------------------------- defect 3: §5.2 compliance coverage / §5.3 variant order
#: The four authoritative hard-constraint fields of ``metrics/constraint_compliance.csv``.
_CONSTRAINT_FIELDS = ["not_expired", "preferred_locations", "salary_min", "work_modes"]


#: ``applicable`` counts that differ by an order of magnitude between variants, the shape
#: the official run has: ``full`` offers the most checkable pairs, the abandoning
#: baselines far fewer.
_APPLICABLE_N = {"full": 12, "profile_only": 25, "one_shot": 6, "no_memory": 6,
                 "no_context": 35}


def _compliance_rows(unknown_rate: float = 0.0) -> list[dict]:
    """One row per (constraint field, variant), the shape of ``constraint_compliance.csv``."""
    return [{"variant": v, "constraint_field": f, "pass": 1, "fail": 0, "unknown": 0,
             "applicable": _APPLICABLE_N[v], "compliance": 0.5,
             "unknown_rate": unknown_rate}
            for v in _VARIANTS for f in _CONSTRAINT_FIELDS]


def test_per_constraint_compliance_renders_every_variant_in_the_frame():
    """Regression guard: §5.2 filtered to three variants and dropped the other two.

    The frame carries a row for every (constraint field, variant) pair, so a hardcoded
    column list silently omitted `one_shot` and `no_memory` — the two variants whose
    constraint handling the section exists to compare.
    """
    section = _section(generate_markdown(_report_data(
        constraint_compliance=_compliance_rows())), "### 5.2")
    table = _tables(section)[0]

    assert list(table)[1:] == _CONSTRAINT_FIELDS          # one row per constraint field
    assert table["constraint field"] == _VARIANTS          # all five columns, canonical order
    for field in _CONSTRAINT_FIELDS:
        assert table[field] == [f"0.500 (n={_APPLICABLE_N[v]})" for v in _VARIANTS]


def test_per_constraint_compliance_appends_an_unrecognised_variant():
    """A variant outside the canonical list gets a column too, ordered last."""
    rows = [*_compliance_rows(),
            {"variant": "zz_probe", "constraint_field": "salary_min", "compliance": 0.25}]
    section = _section(generate_markdown(_report_data(constraint_compliance=rows)),
                       "### 5.2")
    table = _tables(section)[0]

    assert table["constraint field"] == [*_VARIANTS, "zz_probe"]
    assert table["salary_min"][-1] == "0.250"
    assert table["work_modes"][-1] == "N/A"   # unmeasured stays unmeasured, never 0


def _pr_rows(**by_variant: float) -> list[dict]:
    return [{"variant": v, "precision": p, "recall": p, "f1": p, "true_no_match": 1,
             "no_match_expected": 2, "useful": 1, "expected_clarification": 2}
            for v, p in by_variant.items()]


def test_no_match_and_clarification_tables_use_the_canonical_variant_order():
    """§5.3 had its own row order, which disagreed with §5, §5.2 and §7."""
    rows = _pr_rows(**{v: 0.5 for v in reversed(_VARIANTS)})
    section = _section(generate_markdown(
        _report_data(no_match_metrics=rows, clarification_metrics=rows)), "### 5.3")
    no_match, clarification = _tables(section)[:2]

    assert list(no_match)[1:] == _VARIANTS
    assert list(clarification)[1:] == _VARIANTS
    assert no_match["variant"] == ["Precision", "Recall", "F1", "TP", "Expected"]
    assert clarification["variant"] == ["Precision", "Recall", "Useful", "Expected"]


def test_no_match_table_appends_an_unrecognised_variant_rather_than_dropping_it():
    """The order list also filtered: a variant it did not name never reached the table."""
    section = _section(generate_markdown(_report_data(
        no_match_metrics=_pr_rows(full=0.5, zz_probe=0.25))), "### 5.3")

    assert [r[0] for r in _rows(section, "| ")][1:] == ["full", "zz_probe"]


# --------------------------------------- defect 4: grounding, §1 denominators, §7's n
def test_section_5_warning_also_covers_the_grounding_denominator():
    """`grounding` has a per-variant denominator too, and the warning now says so."""
    prose = _section(generate_markdown(_report_data()), "## 5. Overall Results")
    prose = prose[prose.index("**Read the denominators"):prose.index("### 5.x")]
    squeezed = " ".join(prose.split())

    assert "`grounding` carries the same caveat" in squeezed
    assert "averaged only over the runs that actually emitted a factual claim" in squeezed
    assert "returns no recommendation registers no claims" in squeezed
    assert "FEWER claims were checked, NOT better-grounded explanations" in squeezed
    # Still one warning block, not two, and still free of run-specific values.
    assert squeezed.count("**Read the denominators before the means.**") == 1
    assert not re.search(r"\b\d+\b", prose.replace("NDCG@5", "").replace("P@5", ""))


def test_executive_summary_headline_carries_the_denominator_of_every_metric():
    """§1 mixed denominators (ranking vs task success) with no `n` on any of them."""
    section = _section(generate_markdown(_report_data()), "## 1. Executive Summary")
    headline = " ".join(
        section[section.index("- Headline"):section.index("- Ablation direction")].split())

    assert f"NDCG@5 0.500 (n={_RANKING_N['full']})" in headline
    assert f"HCSR 0.500 (n={_RANKING_N['full']})" in headline
    assert f"Task Success 0.500 (n={_SCENARIO_COUNT})" in headline
    assert f"Grounding 0.500 (n={_SCENARIO_COUNT})" in headline
    assert f"Handoff 0.500 (n={_SCENARIO_COUNT})" in headline
    assert "|" not in headline          # a bullet, not a table


def test_headline_omits_the_denominator_when_the_frame_carries_none():
    """A frame predating the ``_n`` columns prints the means without inventing a count."""
    section = _section(
        generate_markdown(_report_data(variant_summary=_variant_summary(with_n=False))),
        "## 1. Executive Summary")
    headline = section[section.index("- Headline"):section.index("- Ablation direction")]

    assert "NDCG@5 0.500," in " ".join(headline.split())
    assert "n=" not in headline


def test_headline_reads_na_when_the_full_variant_is_absent():
    variants = [r for r in _variant_summary() if r["variant"] != "full"]
    section = _section(generate_markdown(_report_data(variant_summary=variants)),
                       "## 1. Executive Summary")

    assert "scenario-mean" in section
    assert "N/A." in section[section.index("- Headline"):]


def test_section_7_states_that_its_n_is_a_scenario_count_not_a_denominator():
    """§7's `n` beside an `N/A` metric cell was ambiguous; the clause disambiguates it."""
    section = _section(generate_markdown(_report_data()), "## 7. Results by Scenario Type")
    squeezed = " ".join(section.split())

    assert "number of scenarios OF THAT TYPE that the variant ran" in squeezed
    assert "it is NOT the per-metric denominator" in squeezed
    assert "A metric cell reading `N/A` therefore means that variant returned nothing to " \
        "score" in squeezed


# ------------- §5.2 compliance denominators, unknown share, §5.4 canonical row order
def test_per_constraint_compliance_prints_the_applicable_count_behind_every_rate():
    """A compliance rate over 6 checkable pairs must not read like one over 35.

    ``applicable`` is the number of (recommended job, constraint) pairs the check could be
    applied to, and a variant that returns fewer recommendations offers fewer of them - the
    same survivorship shape as the §5 ranking denominators, on a different table.
    """
    section = _section(generate_markdown(_report_data(
        constraint_compliance=_compliance_rows())), "### 5.2")
    table = _tables(section)[0]
    squeezed = " ".join(section.split())

    for field in _CONSTRAINT_FIELDS:
        for variant, cell in zip(_VARIANTS, table[field], strict=True):
            assert cell == f"0.500 (n={_APPLICABLE_N[variant]})"
    assert "The denominators differ by an order of magnitude between variants" in squeezed
    assert "a perfect rate on a small `n` is not evidence of constraint enforcement" in \
        squeezed


def test_per_constraint_compliance_flags_an_unknown_driven_rate():
    """A rate mostly made of undeterminable values says so; unknowns are in the denominator."""
    section = _section(generate_markdown(_report_data(
        constraint_compliance=_compliance_rows(unknown_rate=0.371))), "### 5.2")
    table = _tables(section)[0]
    squeezed = " ".join(section.split())

    assert table["work_modes"][0] == f"0.500 (n={_APPLICABLE_N['full']}, unk 37%)"
    assert "unknowns are counted as non-compliant in the denominator" in squeezed
    assert "driven by missing data rather than by observed violations" in squeezed


def test_per_constraint_compliance_omits_the_count_when_the_frame_has_none():
    """A frame without ``applicable`` prints the bare rate rather than inventing a count."""
    rows = [{"variant": "full", "constraint_field": "salary_min", "compliance": 0.25}]
    table = _tables(_section(generate_markdown(
        _report_data(constraint_compliance=rows)), "### 5.2"))[0]

    assert table["salary_min"] == ["0.250"]


def _efficiency_rows() -> list[dict]:
    return [{"variant": v, "runs": 42, "necessary_asked": 7, "necessary_missed": 0,
             "unnecessary_asked": 0, "asked_unresolved": 0, "efficiency_score": -1.5,
             "response_turns_n": 42, "median_response_turns": 1.0,
             "q1_response_turns": 1.0, "q3_response_turns": 2.0,
             "iqr_response_turns": 1.0}
            for v in reversed(_VARIANTS)]


def test_clarification_efficiency_table_uses_the_canonical_variant_order():
    """§5.4 kept its own row order, which disagreed with §5, §5.2, §5.3 and §7."""
    section = _section(generate_markdown(_report_data(
        clarification_efficiency=_efficiency_rows())), "### 5.4")

    assert [r[0] for r in _rows(section, "| ")][1:] == _VARIANTS


def test_clarification_efficiency_table_appends_an_unrecognised_variant():
    rows = [*_efficiency_rows(), {"variant": "zz_probe", "runs": 1,
                                  "efficiency_score": -9.0}]
    section = _section(generate_markdown(_report_data(clarification_efficiency=rows)),
                       "### 5.4")

    assert [r[0] for r in _rows(section, "| ")][1:] == [*_VARIANTS, "zz_probe"]
