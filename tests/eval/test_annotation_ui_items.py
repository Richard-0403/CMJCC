"""Item building from real run bundles: dedup, evidence resolution and blinding.

Drives :func:`jobrec_eval.annotation_ui.loader.build_items` over a real deterministic
experiment (see ``conftest.annotation_experiment``) and checks the three properties the human
annotation pass rests on (checklist items 10/11):

- one relevance item per unique ``(scenario_id, job_id)`` pair that was actually RETURNED,
  carrying the candidate profile, the conversation turns IN ORDER and the posting's readable
  fields;
- one claim item per content-addressed ``claim_id``, with every ``(run_id, claim_id)``
  occurrence recorded so the export can expand the single judgement back to one row per run;
- an unresolvable evidence id stays VISIBLE on the rater's screen beside the citations that
  did resolve, instead of being silently dropped;
- the oracle grade and the validator verdict are absent from every rater-facing payload and
  present on the analysis side.

The oracle grades used here are an obviously-synthetic table (``SYNTHETIC-ORACLE``); no human
label is fabricated anywhere in this module.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from jobrec.catalog import load_catalog
from jobrec_eval.annotation_ui.loader import (
    CLAIM_JOB_FIELDS,
    RELEVANCE_GRADE_SCALE,
    build_items,
    claim_item_key,
    relevance_item_key,
)
from jobrec_eval.annotation_ui.store import (
    BLINDED_FIELD_NAMES,
    KIND_CLAIM,
    KIND_RELEVANCE,
)
from jobrec_eval.loaders import normalize
from jobrec_eval.scenarios import load_scenarios

SYNTHETIC_ORACLE_RATER = "SYNTHETIC-ORACLE"
MISSING_EVIDENCE_ID = "ev-SYNTHETIC-MISSING-0001"


@pytest.fixture(scope="module")
def built(annotation_experiment):
    """Items built from the real experiment, with a synthetic oracle table attached."""
    returned = normalize(annotation_experiment.bundles)["recommendations"]
    pairs = returned[["scenario_id", "job_id"]].drop_duplicates()
    # Obviously-synthetic oracle grades: the loader only has to keep them OUT of the payload,
    # so their values are irrelevant and are marked as machine-side.
    oracle = pd.DataFrame({
        "scenario_id": pairs["scenario_id"].astype(str),
        "job_id": pairs["job_id"].astype(str),
        "rater_id": SYNTHETIC_ORACLE_RATER,
        "relevance_grade": [i % 4 for i in range(len(pairs))],
    })
    return build_items(annotation_experiment.experiment_dir,
                       annotation_experiment.scenarios_path,
                       annotation_experiment.catalog_path,
                       oracle_labels=oracle)


def test_relevance_items_cover_every_unique_returned_pair(built, annotation_experiment):
    """One item per unique returned pair -- no more, no fewer."""
    returned = normalize(annotation_experiment.bundles)["recommendations"]
    expected = {(str(r.scenario_id), str(r.job_id)) for r in returned.itertuples()}
    assert expected, "the real run returned no recommendations at all"

    keys = {item.item_key for item in built.relevance_items}
    assert keys == {relevance_item_key(s, j) for s, j in expected}
    assert len(built.relevance_items) == len(expected)
    assert built.stats.relevance_items == len(expected)
    # More rows were returned than unique pairs: the same pair is returned under both
    # variants, and a human grades it once.
    assert built.stats.returned_pairs >= built.stats.relevance_items
    assert all(item.kind == KIND_RELEVANCE for item in built.relevance_items)


def test_relevance_payload_carries_profile_turns_in_order_and_job_fields(built,
                                                                        annotation_experiment):
    """The rater sees the candidate, the conversation in sequence and the posting."""
    scenarios = load_scenarios(annotation_experiment.scenarios_path)
    catalog = {job.job_id: job for job in load_catalog(annotation_experiment.catalog_path)}

    item = next(i for i in built.relevance_items
                if i.scenario_id in scenarios and i.job_id in catalog)
    scenario, job = scenarios[item.scenario_id], catalog[item.job_id]
    payload = item.payload

    assert payload["grade_scale"] == RELEVANCE_GRADE_SCALE
    assert payload["scenario"]["candidate_profile"] == scenario.profile
    # Turn ORDER matters: a later turn can revise an earlier preference.
    assert [turn["candidate_utterance"] for turn in payload["scenario"]["conversation"]] == list(
        scenario.turns)
    assert [turn["turn_index"] for turn in payload["scenario"]["conversation"]] == list(
        range(len(scenario.turns)))
    assert payload["job"]["title"] == job.title
    assert payload["job"]["company"] == job.company
    assert payload["job"]["work_mode"] == job.work_mode
    assert payload["job"]["required_skills"] == list(job.required_skills)
    assert payload["job"]["salary"]["min"] == job.salary_min
    assert payload["job"]["location"] == {"city": job.city, "region": job.region,
                                          "country": job.country}
    assert payload["job"]["is_active"] == job.is_active


def test_claim_items_are_deduplicated_by_content_addressed_claim_id(built,
                                                                    annotation_experiment):
    """Identical claims across variants collapse to ONE judgement, occurrences preserved."""
    occurrences_in_runs = [(b.run_id, str(c["claim_id"]))
                           for b in annotation_experiment.bundles for c in b.claims
                           if c.get("claim_id")]
    unique_claim_ids = {claim_id for _, claim_id in occurrences_in_runs}

    assert len(built.claim_items) == len(unique_claim_ids)
    assert built.stats.claim_occurrences == len(occurrences_in_runs)
    assert built.stats.claim_items == len(unique_claim_ids)
    # The whole point of the dedup key: fewer judgements than CSV rows.
    assert built.stats.claim_items < built.stats.claim_occurrences
    assert built.stats.claim_dedup_ratio > 1.0

    # Every recorded occurrence maps back to a real (run_id, claim_id) pair, once.
    recorded = [(o.run_id, o.claim_id) for item in built.claim_items for o in item.occurrences]
    assert sorted(recorded) == sorted(occurrences_in_runs)
    assert len(set(recorded)) == len(recorded)

    collapsed = [i for i in built.claim_items if len(i.occurrences) > 1]
    assert collapsed, "no claim was shared across runs, so dedup proved nothing"
    cross_variant = [i for i in collapsed if len({o.variant for o in i.occurrences}) > 1]
    assert cross_variant, "no claim was shared ACROSS variants"
    item = cross_variant[0]
    assert item.item_key == claim_item_key(item.claim_id)
    assert set(item.analysis["variants"]) == {o.variant for o in item.occurrences}


def test_claim_payload_shows_resolved_evidence_and_the_referenced_posting(built):
    """A claim screen carries the cited evidence records and the posting's field values."""
    item = next(i for i in built.claim_items if i.payload["referenced_jobs"])
    payload = item.payload

    assert payload["claim_text"]
    assert payload["claim_type"]
    assert payload["cited_evidence_count"] >= len(payload["evidence"])
    for evidence in payload["evidence"]:
        assert set(evidence) == {"field_name", "normalized_value", "raw_text", "source",
                                 "source_object_id"}
        assert evidence["field_name"]
    for job in payload["referenced_jobs"]:
        assert set(job) <= set(CLAIM_JOB_FIELDS) | {"missing_from_catalog"}
        assert job["job_id"]
    # The clean deterministic run resolves every citation; the dangling-id case is covered by
    # the injected-fault test below.
    assert all(not i.payload["has_unresolvable_evidence"] for i in built.claim_items)


def test_unresolvable_evidence_id_stays_visible_beside_resolved_ones(annotation_experiment):
    """A dangling citation is flagged, not dropped (checklist item 11 asks a rater to check).

    The real deterministic run resolves every id, so ONE synthetic dangling id is injected
    into a copy of the bundles -- an obviously fake ``ev-SYNTHETIC-MISSING-0001``. Nothing
    about the human labels is synthesised; only the fault is.
    """
    bundles = [copy.deepcopy(b) for b in annotation_experiment.bundles]
    target_bundle = next(b for b in bundles
                         if any(c.get("evidence_ids") for c in b.claims))
    target_claim = next(c for c in target_bundle.claims if c.get("evidence_ids"))
    target_claim["evidence_ids"] = [*target_claim["evidence_ids"], MISSING_EVIDENCE_ID]
    claim_id = str(target_claim["claim_id"])

    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path, bundles=bundles)

    item = next(i for i in built.claim_items if i.claim_id == claim_id)
    assert item.payload["unresolvable_evidence_ids"] == [MISSING_EVIDENCE_ID]
    assert item.payload["has_unresolvable_evidence"] is True
    # The resolved citations are still there: the rater sees BOTH sides and can judge that a
    # claim resting partly on a citation that goes nowhere is not fully supported.
    assert item.payload["evidence"], "resolved evidence was lost with the dangling id"
    assert MISSING_EVIDENCE_ID not in {e["source_object_id"] for e in item.payload["evidence"]}
    assert item.payload["cited_evidence_count"] > len(item.payload["evidence"])
    assert item.analysis["unresolved_id_counts"][MISSING_EVIDENCE_ID] >= 1
    assert built.stats.claims_with_unresolved_evidence >= 1


def test_no_payload_carries_the_oracle_grade_or_the_validator_verdict(built):
    """Blinding: the machine's answers are on the analysis side only."""
    for item in built.all_items:
        serialised = json.dumps(item.payload, default=str)
        for blinded in BLINDED_FIELD_NAMES:
            assert f'"{blinded}"' not in serialised, (
                f"{item.item_key} payload leaks {blinded}")

    graded = [i for i in built.relevance_items if i.analysis["oracle_grade"] is not None]
    assert graded, "the synthetic oracle table reached no item"
    assert built.stats.relevance_items_with_oracle_grade == len(graded)

    for item in built.claim_items:
        assert item.analysis["validator_supported_binary"]
        assert set(item.analysis["validator_supported_binary"]) == {
            o.run_id for o in item.occurrences}
        assert all(o.validator_label in (0, 1) for o in item.occurrences)


def test_build_stats_describe_the_workload(built, annotation_experiment):
    """The counts the CLI reports are the real ones, not estimates."""
    stats = built.stats.to_dict()
    assert stats["experiment_id"] == annotation_experiment.experiment_dir.name
    assert stats["runs"] == len(annotation_experiment.bundles)
    assert set(stats["variants"]) == {b.variant for b in annotation_experiment.bundles}
    assert stats["scenarios"] == len({b.scenario_id for b in annotation_experiment.bundles})
    assert stats["relevance_items"] + stats["claim_items"] == len(built.all_items)
    assert {i.kind for i in built.all_items} == {KIND_RELEVANCE, KIND_CLAIM}
