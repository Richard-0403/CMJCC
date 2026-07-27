"""Tests for the R13 extraction-source aggregation (`extraction_source_metrics`).

Covers the two halves of the requirement: the per-field provenance tags the
orchestrator persists into ``extracted_preferences.json`` (``extraction_method`` +
``extraction_source``), and the per-variant / per-scenario-type aggregation of
rule-vs-LLM and fallback counts, including the hybrid-only schema-failure and
fallback rates (R8.12, R13.1, R13.2).
"""

from __future__ import annotations

from pathlib import Path

from jobrec.config import AppConfig
from jobrec.domain.enums import RunMode
from jobrec.orchestration.orchestrator import ConversationOrchestrator
from jobrec_eval.loaders import RunBundle
from jobrec_eval.metrics_extra import extraction_source_metrics
from jobrec_eval.scenarios import Scenario

UTTERANCE = "I want a remote data analyst role in Kuala Lumpur paying RM6000 per month"


def _scenario(scenario_id: str, scenario_type: str) -> Scenario:
    return Scenario(
        scenario_id=scenario_id, scenario_type=scenario_type, difficulty="medium",
        memory_dependency="none", context_dependency="low",
        no_match_expected=False, clarification_expected=False,
    )


def _preference(field_name: str, method: str, source: str) -> dict:
    """One persisted ``ExtractedPreference`` dump carrying its provenance tags."""
    return {
        "field_name": field_name, "normalized_value": "remote", "raw_text": "remote",
        "metadata": {"extraction_method": method, "extraction_source": source},
    }


def _bundle(variant: str, scenario_id: str, mode: str, preferences: list[dict]) -> RunBundle:
    b = RunBundle(
        variant=variant, scenario_id=scenario_id, run_index=0, path=Path("."),
        run_record={"run_id": f"{variant}-{scenario_id}",
                    "model_manifest": {"provider": "p", "model": "m", "mode": mode}},
        decision=None, response=None, claims=[], handoffs=[], evidence_log=[],
        latency={}, active_search=None, job_context=None,
    )
    b.extracted_preferences = {"utterance_id": "u1", "preferences": preferences}
    return b


def _rows(df, scope: str) -> dict:
    """Index the rows of one scope by ``(variant, scenario_type)``."""
    sub = df[df["scope"] == scope]
    return {(r["variant"], r["scenario_type"]): r for _, r in sub.iterrows()}


def test_rule_and_llm_fields_are_counted_per_variant_and_per_scenario_type():
    """Counts and shares are reported for both scopes without double counting (R13.2)."""
    bundles = [
        _bundle("full", "s1", "hybrid", [
            _preference("work_modes", "llm", "normalized"),
            _preference("location", "llm", "normalized"),
            _preference("salary", "rule", "rule_fallback"),
        ]),
        _bundle("full", "s2", "hybrid", [
            _preference("work_modes", "rule", "normalized"),
        ]),
        _bundle("no_memory", "s1", "hybrid", [
            _preference("work_modes", "llm", "repaired"),
        ]),
    ]
    scenarios = {"s1": _scenario("s1", "multi_turn"), "s2": _scenario("s2", "single_turn")}

    df = extraction_source_metrics(bundles, scenarios)

    per_variant = _rows(df, "variant")
    full = per_variant[("full", "(all)")]
    assert full["runs"] == 2
    assert full["fields"] == 4
    assert full["rule_fields"] == 2
    assert full["llm_fields"] == 2
    assert full["rule_share"] == 0.5
    assert full["llm_share"] == 0.5
    assert full["normalized_fields"] == 3
    assert full["rule_fallback_fields"] == 1

    # Per-scenario-type rows partition the same runs, so their field counts sum back
    # to the per-variant total rather than duplicating it.
    per_type = _rows(df, "variant_scenario_type")
    assert per_type[("full", "multi_turn")]["fields"] == 3
    assert per_type[("full", "single_turn")]["fields"] == 1
    assert per_type[("no_memory", "multi_turn")]["repaired_fields"] == 1
    assert (per_type[("full", "multi_turn")]["fields"]
            + per_type[("full", "single_turn")]["fields"]) == full["fields"]


def test_hybrid_schema_failure_and_fallback_rates_use_only_hybrid_runs():
    """R8.12 rates count non-``normalized`` sources over hybrid fields only."""
    bundles = [
        _bundle("full", "s1", "hybrid", [
            _preference("work_modes", "llm", "normalized"),
            _preference("location", "llm", "repaired"),
            _preference("salary", "rule", "rule_fallback"),
            _preference("deadline", "llm", "unresolved"),
        ]),
        # A deterministic run in the same variant contributes fields but no hybrid
        # fields, so it cannot dilute the hybrid-only rates.
        _bundle("full", "s2", "deterministic", [
            _preference("work_modes", "rule", "normalized"),
        ]),
    ]

    row = _rows(extraction_source_metrics(bundles, {"s1": _scenario("s1", "multi_turn")}),
                "variant")[("full", "(all)")]

    assert row["fields"] == 5
    assert row["hybrid_runs"] == 1
    assert row["hybrid_fields"] == 4
    # repaired + rule_fallback + unresolved over the 4 hybrid fields.
    assert row["schema_failure_rate"] == 0.75
    assert row["fallback_rate"] == 0.25
    assert row["unresolved_fields"] == 1


def test_hybrid_rates_are_none_when_no_hybrid_runs_exist():
    """A deterministic-only experiment reads N/A, never a misleading 0.0."""
    bundles = [_bundle("full", "s1", "deterministic",
                       [_preference("work_modes", "rule", "normalized")])]

    row = _rows(extraction_source_metrics(bundles), "variant")[("full", "(all)")]

    assert row["hybrid_fields"] == 0
    assert row["schema_failure_rate"] is None
    assert row["fallback_rate"] is None
    # An unmapped scenario is reported as ``unknown`` rather than dropped.
    assert ("full", "unknown") in _rows(extraction_source_metrics(bundles),
                                        "variant_scenario_type")


def test_empty_bundle_list_returns_the_documented_columns():
    df = extraction_source_metrics([])
    assert df.empty
    assert "schema_failure_rate" in df.columns
    assert "rule_fields" in df.columns


def test_persisted_extraction_carries_method_and_source_tags():
    """The extraction the exporter dumps is self-describing and readable by the metric.

    Runs the real deterministic extraction path, dumps the resulting
    ``ExtractedPreferenceSet`` exactly as ``write_run_bundle`` does, and feeds it back
    through the aggregation so the persisted metadata contract is verified end to end
    (R13.1).
    """
    config = AppConfig()
    config.llm.mode = RunMode.DETERMINISTIC
    orchestrator = ConversationOrchestrator(
        config, [], "snapshot-test", "catalog-hash-test", provider=None
    )

    pref_set, _calls = orchestrator._extract(UTTERANCE)
    dump = pref_set.model_dump(mode="json")

    assert dump["preferences"], "rule extractor produced nothing for the utterance"
    for pref in dump["preferences"]:
        assert pref["metadata"]["extraction_method"] == "rule"
        assert pref["metadata"]["extraction_source"] == "normalized"

    bundle = _bundle("full", "s1", "deterministic", dump["preferences"])
    row = _rows(extraction_source_metrics([bundle]), "variant")[("full", "(all)")]
    assert row["fields"] == len(dump["preferences"])
    assert row["rule_fields"] == row["fields"]
    assert row["llm_fields"] == 0
