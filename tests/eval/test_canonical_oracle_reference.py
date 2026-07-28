"""The relevance oracle's reference is CANONICAL, not lifted from the graded experiment.

Before this, :func:`jobrec_eval.relevance.build_references` built the label universe from
the analysed experiment's own run bundles, keeping the first ``full``-variant bundle it
met -- repeat 0 of the best-equipped condition. Three consequences, all of which reached
the reported numbers:

* every variant was graded against ``full``'s own understanding of the scenario;
* with repeats > 1 the labels depended on which repeat was enumerated first (in the
  hybrid experiment repeat 0 is the minority outcome on two scenarios);
* the deterministic and hybrid experiments grew separate label universes, so their
  ranking numbers were never measured against a common yardstick.

What is asserted here:

- **Independence**: an analysis containing NO ``full`` runs at all is still fully
  labelled. Under the old derivation that produced an empty reference and no ranking
  metrics whatsoever, which is the sharpest available demonstration that the labels no
  longer come from the runs being scored.
- **Frozen and shared**: the artifact is written beside the scenario file on first use,
  reused unchanged afterwards, and a scenario/catalog change makes it fail loudly instead
  of being silently re-derived under a "frozen" heading.
- **Model-free**: the derivation is pinned to deterministic mode, so a hybrid experiment
  and a deterministic one are graded on identical labels.

Every pipeline run here writes only into ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from jobrec.config import load_config
from jobrec.domain.enums import RunMode
from jobrec_eval.cli import run_pipeline
from jobrec_eval.oracle_reference import (
    CANONICAL_ORACLE_FILENAME,
    CANONICAL_ORACLE_VERSION,
    CANONICAL_REPEATS,
    CANONICAL_VARIANT,
    DERIVATION_DECLARED,
    StaleCanonicalOracleError,
    build_canonical_references,
    canonical_config,
    frozen_artifact_path,
    grading_projection,
    inputs_fingerprint,
    load_or_build_canonical_references,
)
from jobrec_eval.relevance import grade_catalog

CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios_subset.jsonl"
CATALOG = "data/processed/jobs.jsonl"

#: Deliberately EXCLUDES ``full``. The reference must not depend on the analysed
#: conditions, so an analysis of two ablations has to be labelled just as completely.
_ABLATIONS_ONLY = ["profile_only", "no_memory"]


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG, base_dir="configs")


@pytest.fixture(scope="module")
def scenario_ids() -> list[str]:
    return [
        json.loads(line)["scenario_id"]
        for line in Path(SCENARIOS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def isolated_scenarios(tmp_path_factory) -> Path:
    """A private copy of the scenario file, so the frozen artifact lands in tmp_path."""
    root = tmp_path_factory.mktemp("canonical-oracle") / "data"
    root.mkdir(parents=True)
    path = root / Path(SCENARIOS).name
    path.write_text(Path(SCENARIOS).read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def canonical(isolated_scenarios, cfg):
    """The canonical reference, derived once and frozen beside the scenario copy."""
    return load_or_build_canonical_references(isolated_scenarios, CATALOG, cfg)


def test_every_scenario_gets_a_reference(canonical, scenario_ids) -> None:
    """No scenario is silently left unlabelled.

    A scenario with no reference gets no labels at all, so its ranking metrics quietly
    disappear from the aggregate rather than being reported as missing.

    **Validates: Requirements 15.1, 33.1**
    """
    assert sorted(canonical.references) == sorted(scenario_ids)
    assert canonical.provenance["scenarios_without_reference"] == []
    assert canonical.provenance["referenced_scenario_count"] == len(scenario_ids)
    for scenario_id, ref in canonical.references.items():
        assert ref["job_context"] and ref["active_search"], scenario_id


def test_clarification_scenario_reference_carries_the_answered_role(canonical) -> None:
    """The derivation drives the clarification loop, so a role is actually stated.

    ``SC-B-01`` states no role in its opening turn. A reference built from the scripted
    turns alone would have no target role, every job would score 0, and the scenario's
    NDCG/P@5 would be undefined instead of merely different.

    **Validates: Requirements 15.1, 33.1**
    """
    active = canonical.references["SC-B-01"]["active_search"]
    assert active.get("target_roles"), active


def test_derivation_is_deterministic_and_model_free(isolated_scenarios, cfg) -> None:
    """Two derivations agree, and the condition is pinned regardless of the caller's config.

    Pinning ``llm.mode`` is what lets the deterministic and the hybrid experiment share a
    yardstick: a model sample must never be able to move a label.

    **Validates: Requirements 15.1, 33.1**
    """
    hybrid = cfg.model_copy(deep=True)
    hybrid.llm.mode = RunMode.HYBRID
    hybrid.experiment.repeat_count = 3

    pinned = canonical_config(hybrid)
    assert pinned.llm.mode == RunMode.DETERMINISTIC
    assert pinned.experiment.repeat_count == CANONICAL_REPEATS
    # The caller's config is not mutated in passing.
    assert hybrid.llm.mode == RunMode.HYBRID

    # The input fingerprint is a function of the INPUTS only, so a hybrid run and a
    # deterministic run over the same scenarios/catalog reuse the same frozen artifact.
    assert (inputs_fingerprint(isolated_scenarios, CATALOG)
            == inputs_fingerprint(isolated_scenarios, CATALOG))

    first = build_canonical_references(isolated_scenarios, CATALOG, cfg)
    second = build_canonical_references(isolated_scenarios, CATALOG, hybrid)
    # The label-determining content is identical -- which is the property that matters.
    # The raw payload is not compared: it carries ``normalized_at`` and ids derived from
    # timestamped objects, so it differs between any two derivations by construction.
    assert grading_projection(first.references) == grading_projection(second.references)
    assert first.reference_fingerprint == second.reference_fingerprint
    assert first.inputs_fingerprint == second.inputs_fingerprint
    assert first.provenance["llm_mode"] == RunMode.DETERMINISTIC.value
    assert first.provenance["variant"] == CANONICAL_VARIANT


def test_frozen_artifact_is_written_once_and_then_reused(isolated_scenarios, canonical) -> None:
    """First use derives and freezes; later uses load the same bytes.

    **Validates: Requirements 15.1, 16.1**
    """
    frozen = frozen_artifact_path(isolated_scenarios)
    assert frozen.exists(), "the canonical reference was not frozen as an input artifact"
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert payload["canonical_oracle_version"] == CANONICAL_ORACLE_VERSION
    assert payload["inputs_fingerprint"] == canonical.inputs_fingerprint
    assert payload["reference_fingerprint"] == canonical.reference_fingerprint

    before = frozen.read_bytes()
    reloaded = load_or_build_canonical_references(
        isolated_scenarios, CATALOG, load_config(CONFIG, base_dir="configs"))
    assert frozen.read_bytes() == before, "a reused artifact must not be rewritten"
    assert reloaded.reference_fingerprint == canonical.reference_fingerprint
    assert reloaded.provenance["loaded_from"] == str(frozen)


def test_two_scenario_files_in_one_directory_do_not_share_an_artifact(tmp_path) -> None:
    """Each scenario file gets its OWN frozen reference.

    ``evaluation/data/`` holds both the 42-scenario set and the 12-scenario subset. With
    a single shared filename the two overwrite each other's reference, and because a
    mismatched artifact fails loudly, whichever ran second aborted instead of analysing.

    **Validates: Requirements 15.1, 16.1**
    """
    full = tmp_path / "scenarios.jsonl"
    subset = tmp_path / "scenarios_subset.jsonl"
    assert frozen_artifact_path(full) != frozen_artifact_path(subset)
    assert frozen_artifact_path(full).parent == full.parent
    assert full.stem in frozen_artifact_path(full).name
    assert subset.stem in frozen_artifact_path(subset).name


def test_a_stale_frozen_artifact_fails_loudly(tmp_path, cfg, canonical) -> None:
    """Changing the scenarios invalidates the artifact instead of silently re-deriving.

    Either alternative is a reporting error: reusing it grades jobs against a scenario
    that no longer exists, and regenerating it changes every grade-derived number under a
    heading that claims the labels were frozen.

    **Validates: Requirements 15.1, 16.1**
    """
    data = tmp_path / "data"
    data.mkdir()
    scenarios = data / Path(SCENARIOS).name
    lines = [
        line for line in Path(SCENARIOS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenarios.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    # A frozen artifact derived for the FULL scenario set, placed beside a shorter one.
    frozen_artifact_path(scenarios).write_text(
        json.dumps(canonical.as_artifact(), indent=2, default=str), encoding="utf-8")

    with pytest.raises(StaleCanonicalOracleError, match="different"):
        load_or_build_canonical_references(scenarios, CATALOG, cfg)


def test_reference_does_not_come_from_the_analysed_runs(canonical, cfg) -> None:
    """The reference is a function of the inputs, not of any variant's output.

    The old derivation read the analysed bundles, so it could only ever describe what the
    ``full`` variant happened to understand. The canonical reference is built without any
    reference to the experiment being scored, which this asserts by grading the catalog
    from it in complete isolation from any run directory.

    **Validates: Requirements 15.1, 33.1**
    """
    from jobrec.catalog import load_catalog

    labels = grade_catalog(load_catalog(CATALOG), canonical.references, cfg)
    assert not labels.empty
    assert set(labels["scenario_id"]) == set(canonical.references)
    # Non-degenerate: the oracle actually separates jobs rather than grading everything 0.
    assert labels["relevance_grade"].max() > 0
    assert (labels["relevance_grade"] > 0).any() and (labels["relevance_grade"] == 0).any()


# ------------------------------------------------------------------ pipeline wiring
@pytest.fixture(scope="module")
def ablation_only_run(tmp_path_factory, scenario_ids) -> SimpleNamespace:
    """A full analysis whose variant set contains no ``full`` runs at all."""
    root = tmp_path_factory.mktemp("canonical-oracle-pipeline")
    data = root / "data"
    data.mkdir()
    scenarios = data / Path(SCENARIOS).name
    scenarios.write_text(Path(SCENARIOS).read_text(encoding="utf-8"), encoding="utf-8")
    result = run_pipeline(CONFIG, str(scenarios), CATALOG, str(root / "out"), 1, None,
                          200, 2026, variants=_ABLATIONS_ONLY)
    return SimpleNamespace(out=Path(result["out_dir"]), scenarios=scenarios,
                           scenario_ids=scenario_ids)


def test_analysis_without_full_runs_is_still_fully_labelled(ablation_only_run) -> None:
    """Labels exist for every scenario even though no ``full`` run was analysed.

    This is the regression: with bundle-derived references this analysis produced an
    EMPTY label table, so NDCG@5 / P@5 / mean graded relevance silently ceased to exist.

    **Validates: Requirements 15.1, 33.1**
    """
    labels = pd.read_csv(ablation_only_run.out / "normalized" / "relevance_labels.csv")
    assert not labels.empty
    assert set(labels["scenario_id"].astype(str)) == set(ablation_only_run.scenario_ids)

    runs = pd.read_csv(ablation_only_run.out / "normalized" / "runs.csv")
    assert CANONICAL_VARIANT not in set(runs["variant"].astype(str)), (
        "the fixture is meant to contain no full runs")

    summary = pd.read_csv(ablation_only_run.out / "metrics" / "variant_summary.csv")
    assert summary["ndcg_at_5_mean"].notna().any(), "ranking metrics vanished"


def test_analysis_records_the_canonical_reference_provenance(ablation_only_run) -> None:
    """The plan and the manifests state which yardstick produced the grades.

    Without the fingerprints a reader cannot tell whether two experiments were graded on
    the same labels, which is exactly the claim the thesis makes when it compares them.

    **Validates: Requirements 15.1, 16.1**
    """
    out = ablation_only_run.out
    artifact = json.loads(
        (out / "manifests" / CANONICAL_ORACLE_FILENAME).read_text(encoding="utf-8"))
    plan = yaml.safe_load((out / "manifests" / "analysis_plan.yaml").read_text())

    assert artifact["canonical_oracle_version"] == CANONICAL_ORACLE_VERSION
    assert plan["canonical_reference_version"] == CANONICAL_ORACLE_VERSION
    assert plan["canonical_reference_inputs_fingerprint"] == artifact["inputs_fingerprint"]
    assert plan["canonical_reference_fingerprint"] == artifact["reference_fingerprint"]
    assert plan["canonical_reference_condition"] == {
        "variant": CANONICAL_VARIANT,
        "repeats": CANONICAL_REPEATS,
        "llm_mode": RunMode.DETERMINISTIC.value,
    }
    assert plan["canonical_reference_artifact"] == f"manifests/{CANONICAL_ORACLE_FILENAME}"

    data = json.loads(
        (out / "report" / "analysis_report_data.json").read_text(encoding="utf-8"))
    canon = data["relevance_source"]["canonical_reference"]
    assert canon["inputs_fingerprint"] == artifact["inputs_fingerprint"]
    # Every scenario in the subset declares its authoritative reference, so nothing falls
    # back to reading the system's own extraction. This is the property the whole exercise
    # is for, so it is asserted rather than merely reported.
    assert canon["derivation"] == DERIVATION_DECLARED
    assert canon["system_derived_scenarios"] == []
    assert canon["declared_scenario_count"] == canon["scenario_count"]
    assert canon["scenarios_without_reference"] == []


def test_the_frozen_artifact_is_shared_by_a_second_analysis(ablation_only_run, tmp_path) -> None:
    """A second analysis over the same inputs reuses the artifact rather than re-deriving.

    That reuse is what makes the deterministic and hybrid experiments comparable: both
    read the same file, so a ranking difference between them cannot be a label difference.

    **Validates: Requirements 15.1, 16.1**
    """
    frozen = frozen_artifact_path(ablation_only_run.scenarios)
    assert frozen.exists()
    before = frozen.read_bytes()

    result = run_pipeline(CONFIG, str(ablation_only_run.scenarios), CATALOG,
                          str(tmp_path / "second"), 1, None, 200, 2026,
                          variants=_ABLATIONS_ONLY)
    assert frozen.read_bytes() == before

    first = json.loads((ablation_only_run.out / "manifests" / CANONICAL_ORACLE_FILENAME)
                       .read_text(encoding="utf-8"))
    second = json.loads((Path(result["out_dir"]) / "manifests" / CANONICAL_ORACLE_FILENAME)
                        .read_text(encoding="utf-8"))
    assert first["reference_fingerprint"] == second["reference_fingerprint"]
    assert first["references"] == second["references"]
