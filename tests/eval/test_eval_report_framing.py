"""Unit tests for ablation delta rendering and report framing (R32.4-32.6).

The report template only reads the assembled data dict, so these tests render
`generate_markdown` over a minimal-but-complete fixture and inspect the Markdown:

- the Δ definitions and the Δ cells of the contribution tables (R32.4),
- the "framework mechanism contribution under the controlled prototype
  instantiation" framing on every Δ (R32.5),
- the absence of any superiority claim over external frameworks (R32.6). The
  detector is negation-aware, because the disclaimers legitimately mention
  "superiority" while denying it.
"""

from __future__ import annotations

import re

from jobrec_eval.report import generate_markdown

# Δ is defined as full − other, so the fixture stores the two means and derives the
# delta from them; the tables must render exactly that difference.
_MEMORY_CELLS = {
    ("memory_dependent", "ndcg_at_5"): (0.812, 0.640),
    ("memory_dependent", "task_success"): (0.900, 0.700),
    ("all", "ndcg_at_5"): (0.781, 0.702),
    ("all", "task_success"): (0.850, 0.775),
}
_CONTEXT_CELLS = {
    ("context_dependent", "hcsr"): (0.940, 0.615),
    ("context_dependent", "task_success"): (0.880, 0.660),
    ("all", "hcsr"): (0.910, 0.804),
    ("all", "task_success"): (0.845, 0.769),
}

_METRIC_KEYS = ["ndcg_at_5", "precision_at_5", "hcsr", "task_success", "grounding",
                "handoff_success", "turn_count", "total_latency_ms"]

# Claim vocabulary that Requirement 32.6 forbids the report from asserting.
_SUPERIORITY_TERMS = re.compile(
    r"\b(superior(?:ity)?|outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|beats|"
    r"better\s+than|state[\s-]of[\s-]the[\s-]art|sota|best\s+framework)\b",
    re.IGNORECASE)
_NEGATION = re.compile(r"\b(not|never|nor|neither|without|no)\b", re.IGNORECASE)


def _contrib_rows(cells: dict[tuple[str, str], tuple[float, float]]) -> list[dict]:
    rows = []
    for (subset, metric), (base, other) in cells.items():
        rows.append({
            "subset": subset, "metric": metric,
            "base_mean": base, "other_mean": other, "delta": base - other,
            "ci_low": base - other - 0.05, "ci_high": base - other + 0.05,
            "p_value": 0.04, "p_value_holm": 0.08,
            "effect_size": 0.6, "effect_type": "cohens_dz", "n_pairs": 6,
        })
    return rows


def _report_data() -> dict:
    variant_summary = [
        {"variant": v, **{f"{k}_mean": 0.5 for k in _METRIC_KEYS}}
        for v in ["full", "profile_only", "one_shot", "no_memory", "no_context"]
    ]
    scenario_variant = [
        {"scenario_id": f"s{i}", "scenario_type": "multi_turn", "variant": v,
         "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9}
        for i in range(2) for v in ["full", "no_memory", "no_context"]
    ]
    overall = [
        {"metric": m, "other": o, "delta": 0.1, "ci_low": 0.02, "ci_high": 0.18}
        for m in ["ndcg_at_5", "task_success"] for o in ["profile_only", "one_shot"]
    ]
    return {
        "experiment": {
            "experiment_id": "exp-test-0001", "reference_date": "2026-01-01",
            "catalog_snapshot_id": "catalog-2026-01", "catalog_hash": "abc123def456789",
            "variants": ["full", "profile_only", "one_shot", "no_memory", "no_context"],
            "scenario_count": 2, "repeat_count": 1, "run_count": 10,
            "bootstrap_iterations": 5000, "bootstrap_seed": 2026, "eval_version": "1.0.0",
        },
        "oracle_version": "1.0.0",
        "scenario_type_counts": {"multi_turn": 2},
        "n_memory_dependent": 2,
        "n_context_dependent": 1,
        "variant_summary": variant_summary,
        "scenario_variant": scenario_variant,
        "memory_contribution": _contrib_rows(_MEMORY_CELLS),
        "context_contribution": _contrib_rows(_CONTEXT_CELLS),
        "overall_comparisons": overall,
        "error_summary": "Runs: 10; system failures: 0; task-unsuccessful runs: 1.",
    }


def _section(md: str, heading: str) -> str:
    """Text of one Markdown section, up to the next heading of the same or higher level."""
    start = md.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = md[start + len(heading):]
    ends = [m.start() for lv in range(2, level + 1)
            if (m := re.search(rf"^{'#' * lv} ", rest, re.MULTILINE))]
    return rest[:min(ends)] if ends else rest


def _tables(section: str) -> list[dict[str, list[str]]]:
    """Parse the Markdown tables of a section into {first cell: remaining cells}."""
    tables, current = [], {}
    for line in section.splitlines():
        if not line.startswith("|"):
            if current:
                tables.append(current)
                current = {}
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= {"-"} or cells[0] in ("metric", "variant"):
            continue  # header / separator row
        current[cells[0]] = cells[1:]
    if current:
        tables.append(current)
    return tables


def _sentences(md: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(md.split()))


def _superiority_claims(md: str) -> list[str]:
    """Sentences that assert a superiority term without negating it first.

    A disclaimer ("not a claim of superiority over any external framework") carries a
    negation cue ahead of the term; a real claim ("the framework is superior to X")
    does not, so only the latter is reported.
    """
    claims = []
    for sentence in _sentences(md):
        for match in _SUPERIORITY_TERMS.finditer(sentence):
            if not _NEGATION.search(sentence[:match.start()]):
                claims.append(sentence)
                break
    return claims


# --------------------------------------------------------------------- R32.4

def test_ablation_deltas_are_defined_and_rendered_from_the_input_data():
    """Δmemory / Δcontext are defined as full − other and the tables carry those values."""
    md = generate_markdown(_report_data())

    # Δ definitions (R32.4).
    assert "Δmemory(M) = M_full \u2212 M_no_memory" in md
    assert "Δcontext(M) = M_full \u2212 M_no_context" in md

    for heading, cells, primary_subset in (
        ("### 6.1 Memory Contribution: Full vs No-Memory", _MEMORY_CELLS, "memory_dependent"),
        ("### 6.2 Job-Context Contribution: Full vs No-Context", _CONTEXT_CELLS,
         "context_dependent"),
    ):
        # Two tables per section: the primary subset first, then all scenarios.
        primary, overall = _tables(_section(md, heading))
        for subset, table in ((primary_subset, primary), ("all", overall)):
            expected = {m: v for (s, m), v in cells.items() if s == subset}
            assert set(table) == set(expected)
            for metric, (base, other) in expected.items():
                full_cell, other_cell, delta_cell = table[metric][:3]
                assert full_cell == f"{base:.3f}"
                assert other_cell == f"{other:.3f}"
                assert delta_cell == f"{base - other:.3f}"


# --------------------------------------------------------------------- R32.5

def test_report_frames_each_delta_as_a_mechanism_contribution():
    """Every Δ is framed as a mechanism contribution under this prototype instantiation."""
    md = " ".join(generate_markdown(_report_data()).split())
    framing = "framework mechanism under the controlled prototype instantiation"

    # Section 6 intro frames all Δ values; ** markers survive the whitespace squeeze.
    assert framing in md.replace("**", "")
    # Δmemory (6.1), Δcontext (6.2) and the conclusion repeat the framing.
    assert md.count("contribution under the controlled prototype instantiation") >= 2
    assert ("attribute observed differences to specific framework mechanisms under the "
            "controlled prototype instantiation") in md


# --------------------------------------------------------------------- R32.6

def test_report_makes_no_superiority_claim_over_external_frameworks():
    md = generate_markdown(_report_data())

    assert _superiority_claims(md) == []
    # The disclaimers do mention superiority -- while denying it.
    assert "superiority over" in md


def test_superiority_detector_flags_an_actual_claim():
    """Guard on the assertion above: an asserted claim is caught, a denial is not."""
    md = generate_markdown(_report_data())
    claim = "This framework is superior to all existing external frameworks."

    flagged = _superiority_claims(md + "\n\n" + claim)
    assert len(flagged) == 1
    assert claim in flagged[0]
    assert _superiority_claims("We do not claim superiority over external frameworks.") == []
