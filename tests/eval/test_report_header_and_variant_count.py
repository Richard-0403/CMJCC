"""Report prose that must stay true for any run shape: variant count and bundle pointer.

Two defects motivated this module, both of which would have been read as fact in the
thesis:

- §2 hardcoded the word "Five", so a three-variant run rendered "Five variants (full,
  no_memory, no_context)" and contradicted itself in the same sentence. The count is now
  derived from the experiment manifest, so the sentence is true for 1, 3, 5 or any other
  variant list, and the one-variant case stays grammatical.
- the header claimed every number is reproducible from "the run bundles under `raw/`", a
  directory the pipeline never creates. Bundles land under
  ``<out_root>/_runs/<experiment_id>/<variant>/<scenario_id>/<run_index>/`` while the
  analysis tables land under ``<out_root>/<experiment_id>/``, i.e. the two are siblings.

Rendering is a pure function of the data dict, so these tests drive
``generate_markdown`` directly. That the pointed-at directory really exists on disk is
asserted against a real pipeline run in ``test_pipeline_artifacts.py``.
"""

from __future__ import annotations

import re

import pytest

from jobrec_eval.report import generate_markdown

_METRIC_KEYS = ["ndcg_at_5", "precision_at_5", "hcsr", "task_success", "grounding",
                "handoff_success", "turn_count", "total_latency_ms"]


def _contrib_rows(other: str, subsets: tuple[str, ...]) -> list[dict]:
    return [{
        "subset": subset, "metric": metric, "base_mean": 0.8, "other_mean": 0.6,
        "delta": 0.2, "ci_low": 0.05, "ci_high": 0.35, "p_value": 0.04,
        "p_value_holm": 0.08, "effect_size": 0.6, "effect_type": "cohens_dz",
        "n_pairs": 2, "other": other,
    } for subset in subsets for metric in _METRIC_KEYS]


def _report_data(variants: list[str], experiment_id: str = "exp-8793b18de5b2") -> dict:
    """Minimal-but-complete data dict for the template, over an arbitrary variant list."""
    return {
        "experiment": {
            "experiment_id": experiment_id, "reference_date": "2026-01-01",
            "catalog_snapshot_id": "catalog-2026-01", "catalog_hash": "145dfa05aaaa454509",
            "variants": variants, "scenario_count": 2, "repeat_count": 1,
            "run_count": 2 * len(variants), "bootstrap_iterations": 5000,
            "bootstrap_seed": 2026, "eval_version": "1.0.0",
        },
        "oracle_version": "1.0.0",
        "scenario_type_counts": {"multi_turn": 2},
        "n_memory_dependent": 2,
        "n_context_dependent": 1,
        "variant_summary": [{"variant": v, **{f"{k}_mean": 0.5 for k in _METRIC_KEYS}}
                            for v in variants],
        "scenario_variant": [
            {"scenario_id": f"s{i}", "scenario_type": "multi_turn", "variant": v,
             "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9}
            for i in range(2) for v in variants],
        "memory_contribution": _contrib_rows("no_memory", ("all", "memory_dependent")),
        "context_contribution": _contrib_rows("no_context", ("all", "context_dependent")),
        "overall_comparisons": [
            {"metric": m, "other": o, "delta": 0.1, "ci_low": 0.02, "ci_high": 0.18}
            for m in ["ndcg_at_5", "task_success"] for o in variants[1:]],
        "error_summary": "Runs: 6; system failures: 0; task-unsuccessful runs: 1.",
    }


def _squeeze(md: str) -> str:
    """Whitespace-normalised text with Markdown blockquote markers stripped.

    The header is a blockquote, so its sentences carry a leading ``> `` on every
    continuation line; dropping the markers lets a sentence be matched as a sentence.
    """
    lines = [re.sub(r"^>\s?", "", line) for line in md.splitlines()]
    return " ".join(" ".join(lines).split())


@pytest.mark.parametrize(("variants", "expected"), [
    (["full"], "One variant (full) is run over a frozen scenario set"),
    (["full", "no_memory", "no_context"],
     "Three variants (full, no_memory, no_context) are run over a frozen scenario set"),
    (["full", "profile_only", "one_shot", "no_memory", "no_context"],
     "Five variants (full, profile_only, one_shot, no_memory, no_context) are run "
     "over a frozen scenario set"),
])
def test_section_2_states_the_variant_count_it_actually_ran(variants, expected):
    """The §2 sentence names the real count and reads correctly in the singular."""
    md = _squeeze(generate_markdown(_report_data(variants)))

    assert expected in md
    # No stale count survives: the only spelled-out count is the true one.
    stale = {"One", "Two", "Three", "Four", "Five"} - {expected.split()[0]}
    for word in stale:
        assert f"{word} variant" not in md


def test_single_variant_sentence_is_not_pluralised():
    md = _squeeze(generate_markdown(_report_data(["full"])))

    assert "One variant (full) is run" in md
    assert "One variants" not in md
    assert "variant (full) are run" not in md


def test_variant_count_sentence_is_consistent_with_the_executive_summary():
    """§1 counts variants from the same list, so the two sections cannot disagree."""
    variants = ["full", "no_memory", "no_context"]
    md = _squeeze(generate_markdown(_report_data(variants)))

    assert f"2 scenarios × {len(variants)} variants × 1 repeat(s)" in md
    assert f"Three variants ({', '.join(variants)}) are run" in md


def test_header_points_at_the_run_bundle_layout_the_pipeline_writes():
    """The reproducibility pointer names ``_runs/<experiment_id>/...``, not ``raw/``."""
    experiment_id = "exp-8793b18de5b2"
    md = generate_markdown(_report_data(["full", "no_memory", "no_context"], experiment_id))

    assert f"`_runs/{experiment_id}/<variant>/<scenario_id>/<run_index>/`" in md
    # The directory that never existed is no longer named anywhere.
    assert "`raw/`" not in md
    assert "under `raw" not in md
    # The relationship between the two directories is stated, not implied.
    squeezed = _squeeze(md)
    assert ("The run-bundle tree is a sibling of this analysis output directory "
            f"`{experiment_id}/`: both live under the pipeline's `--out-root`.") in squeezed
    assert "the tables under `metrics/` and `statistics/` inside this directory" in squeezed


def test_header_pointer_carries_every_run_bundle_path_component():
    """Every level of the on-disk hierarchy is addressable from the header alone."""
    md = generate_markdown(_report_data(["full", "no_memory"], "exp-abc123abc123"))

    pointer = re.search(r"`(_runs/[^`]+)`", md)
    assert pointer is not None
    assert pointer.group(1).split("/") == [
        "_runs", "exp-abc123abc123", "<variant>", "<scenario_id>", "<run_index>", ""]
