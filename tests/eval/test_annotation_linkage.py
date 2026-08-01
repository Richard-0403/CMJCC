"""Human annotations must describe the experiment they are reported against.

Two failure families, both of which produce a confident-looking number from nothing.

Merging on ``claim_id``. That id is ``content_id("claim", claim_type, text, key_extra)`` -- a
digest of the RENDERED SENTENCE, with ``key_extra`` at most a job id. Two claims whose
sentences read alike therefore share an id even when they assert different things, so a rating
made for one threshold gets reused for another. The three collisions asserted here are the ones
that actually occur: the same salary conclusion at different minimums, the same work-mode
conclusion with different expected modes, and the same skill-gap conclusion against different
candidate skill sets.

Reporting agreement without linkage. ``claim_agreement`` read a CSV and computed kappa with no
connection to the runs being analysed, so labels from an earlier batch produced a figure for
runs they had never seen. Kappa over zero overlapping items is not a low agreement score, it is
not a measurement.

And the quiet one: a returned pair with no human relevance label is UNKNOWN, not irrelevant.
Scoring it 0 converts "nobody judged this" into "the system was wrong" and biases every
human-scored ranking metric downward by exactly the amount of missing annotation.
"""

from __future__ import annotations

import pytest

from jobrec_eval.annotation_linkage import (
    DELIVERY_DELIVERED,
    DELIVERY_DROPPED,
    MissingHumanLabelsError,
    StaleAnnotationError,
    annotation_signature,
    claim_occurrences,
    evidence_projection,
    link_claim_labels,
    relevance_coverage,
    require_linked,
    require_relevance_coverage,
)


def _claim(**over) -> dict:
    base = {
        "claim_id": "claim-same-id",
        "claim_type": "ranking_reason",
        "text": "Salary meets your stated minimum.",
        "predicate": "salary_meets_min",
        "field_name": "salary_min",
        "job_id": "job-1",
        "expected_value": 4000,
        "observed_value": 4500,
        "claim_args": {"feature": "salary_meets_min"},
        "evidence_ids": [],
    }
    base.update(over)
    return base


# ------------------------------------------------------- the three collisions
def test_the_same_salary_sentence_at_different_minimums_does_not_merge():
    """Identical text and identical claim_id, different threshold."""
    low = _claim(expected_value=4000)
    high = _claim(expected_value=6000)

    assert low["claim_id"] == high["claim_id"], "the fixture must share a claim_id"
    assert low["text"] == high["text"]
    assert annotation_signature(low) != annotation_signature(high)


def test_the_same_work_mode_sentence_with_different_expected_modes_does_not_merge():
    onsite = _claim(predicate="ranking_match", field_name="work_modes",
                    text="Work mode is onsite, matching your preference.",
                    expected_value=["onsite"], observed_value="onsite")
    hybrid = _claim(predicate="ranking_match", field_name="work_modes",
                    text="Work mode is onsite, matching your preference.",
                    expected_value=["onsite", "hybrid"], observed_value="onsite")

    assert annotation_signature(onsite) != annotation_signature(hybrid)


def test_the_same_skill_gap_sentence_against_different_skill_sets_does_not_merge():
    text = "Gap: the role requires excel, which is not recorded in your profile skills."
    few = _claim(claim_type="skill_gap", predicate="skill_not_recorded", text=text,
                 claim_args={"skill": "excel"}, observed_value=["python"])
    many = _claim(claim_type="skill_gap", predicate="skill_not_recorded", text=text,
                  claim_args={"skill": "excel"}, observed_value=["python", "sql", "r"])

    assert annotation_signature(few) != annotation_signature(many)


def test_the_same_sentence_resting_on_different_evidence_does_not_merge():
    """The evidence projection is part of the proposition."""
    store_a = {"ev-1": {"source": "dialogue", "field_name": "salary_min",
                        "normalized_value": 4000}}
    store_b = {"ev-1": {"source": "dialogue", "field_name": "salary_min",
                        "normalized_value": 6000}}
    claim = _claim(evidence_ids=["ev-1"])

    assert annotation_signature(claim, store_a) != annotation_signature(claim, store_b)


def test_evidence_ids_themselves_do_not_split_one_proposition():
    """Ids are content-addressed over the TURN, so the same statement in two runs has two ids."""
    run_a = {"ev-a": {"source": "dialogue", "field_name": "salary_min",
                      "normalized_value": 4000}}
    run_b = {"ev-b": {"source": "dialogue", "field_name": "salary_min",
                      "normalized_value": 4000}}
    assert annotation_signature(_claim(evidence_ids=["ev-a"]), run_a) == \
        annotation_signature(_claim(evidence_ids=["ev-b"]), run_b)


def test_not_looking_at_evidence_differs_from_citing_none():
    assert evidence_projection(_claim(evidence_ids=[]), None) is None
    assert evidence_projection(_claim(evidence_ids=[]), {}) == []
    assert annotation_signature(_claim(), None) != annotation_signature(_claim(), {})


def test_value_formatting_does_not_split_one_proposition():
    """4000 and 4000.0, and list order, are the same assertion."""
    assert annotation_signature(_claim(expected_value=4000)) == \
        annotation_signature(_claim(expected_value=4000.0))
    assert annotation_signature(_claim(expected_value=["onsite", "hybrid"])) == \
        annotation_signature(_claim(expected_value=["hybrid", "onsite"]))


# ------------------------------------------------------ occurrences and delivery
def test_occurrences_keep_dropped_claims_with_an_explicit_status():
    """Withheld claims are needed for a false-negative estimate and must not read as shown."""
    rows = claim_occurrences("exp-1", [{
        "run_id": "run-1", "scenario_id": "SC-A-01", "variant": "full", "repeat_index": 0,
        "claims": [_claim()],
        "dropped_claims": [_claim(text="Withheld.", expected_value=1)],
        "evidence_by_id": {},
    }])

    assert {r["delivery_status"] for r in rows} == {DELIVERY_DELIVERED, DELIVERY_DROPPED}
    for row in rows:
        for column in ("experiment_id", "run_id", "scenario_id", "variant",
                       "repeat_index", "claim_id", "annotation_signature"):
            assert row[column] is not None, column
    # Different propositions, so different signatures even though both are one occurrence set.
    assert len({r["annotation_signature"] for r in rows}) == 2


# ----------------------------------------------------------- linkage and staleness
def test_labels_for_another_experiment_are_stale_and_raise():
    occurrences = claim_occurrences("exp-new", [{
        "run_id": "r", "scenario_id": "SC-A-01", "variant": "full", "repeat_index": 0,
        "claims": [_claim()], "evidence_by_id": {}}])
    old_labels = [{"experiment_id": "exp-old", "annotation_signature": "sig-somethingelse",
                   "rater_1": 1, "rater_2": 1}]

    report = link_claim_labels("exp-new", occurrences, old_labels)
    assert report.is_stale
    assert report.overlapping_signatures == 0
    with pytest.raises(StaleAnnotationError, match="not a measurement"):
        require_linked(report, min_coverage=0.8)


def test_partial_coverage_is_refused_rather_than_quietly_reported():
    occurrences = claim_occurrences("exp-1", [{
        "run_id": "r", "scenario_id": "SC-A-01", "variant": "full", "repeat_index": 0,
        "claims": [_claim(expected_value=v) for v in (4000, 5000, 6000, 7000)],
        "evidence_by_id": {}}])
    labelled = [{"experiment_id": "exp-1",
                 "annotation_signature": occurrences[0]["annotation_signature"],
                 "rater_1": 1, "rater_2": 1}]

    report = link_claim_labels("exp-1", occurrences, labelled)
    assert report.current_signatures == 4
    assert report.overlapping_signatures == 1
    assert report.coverage == pytest.approx(0.25)
    with pytest.raises(MissingHumanLabelsError, match="below the required"):
        require_linked(report, min_coverage=0.8)
    # Under a requirement it does meet, it passes and reports the fraction.
    require_linked(report, min_coverage=0.2)


def test_an_experiment_with_no_claims_has_not_been_fully_annotated():
    """Coverage is None, not 1.0: nothing was measured."""
    report = link_claim_labels("exp-1", [], [])
    assert report.coverage is None
    assert not report.is_stale
    with pytest.raises(MissingHumanLabelsError, match="no claims"):
        require_linked(report, min_coverage=0.0)


def test_obsolete_labels_are_counted_not_silently_dropped():
    occurrences = claim_occurrences("exp-1", [{
        "run_id": "r", "scenario_id": "SC-A-01", "variant": "full", "repeat_index": 0,
        "claims": [_claim()], "evidence_by_id": {}}])
    labels = [{"experiment_id": "exp-1",
               "annotation_signature": occurrences[0]["annotation_signature"],
               "rater_1": 1, "rater_2": 1},
              {"experiment_id": "exp-1", "annotation_signature": "sig-gone",
               "rater_1": 0, "rater_2": 0}]

    report = link_claim_labels("exp-1", occurrences, labels)
    assert report.obsolete_signatures == 1
    assert report.coverage == 1.0


# -------------------------------------------------- relevance coverage, and not-zero
def test_an_unlabelled_returned_pair_is_reported_as_delta_not_scored_zero():
    returned = {("SC-A-01", "job-1"), ("SC-A-01", "job-2"), ("SC-A-02", "job-3")}
    labelled = {("SC-A-01", "job-1"), ("SC-Z-99", "job-9")}

    cov = relevance_coverage(returned, labelled)
    assert cov.reused == [("SC-A-01", "job-1")]
    assert set(cov.delta) == {("SC-A-01", "job-2"), ("SC-A-02", "job-3")}
    assert cov.obsolete == [("SC-Z-99", "job-9")]
    assert cov.coverage == pytest.approx(1 / 3)

    payload = cov.as_dict()
    assert payload["delta_pairs_requiring_annotation"] == 2
    assert payload["reused_overlapping_labels"] == 1
    assert payload["obsolete_extra_labels"] == 1


def test_insufficient_relevance_coverage_refuses_to_report():
    cov = relevance_coverage({("SC-A-01", "job-1"), ("SC-A-01", "job-2")},
                             {("SC-A-01", "job-1")})
    with pytest.raises(MissingHumanLabelsError, match="UNKNOWN, not irrelevant"):
        require_relevance_coverage(cov, min_coverage=0.9)
    require_relevance_coverage(cov, min_coverage=0.5)


def test_full_relevance_coverage_passes():
    pairs = {("SC-A-01", "job-1"), ("SC-A-01", "job-2")}
    cov = relevance_coverage(pairs, pairs)
    assert cov.coverage == 1.0
    assert cov.delta == []
    require_relevance_coverage(cov, min_coverage=1.0)


def test_no_returned_jobs_is_not_perfect_coverage():
    cov = relevance_coverage(set(), {("SC-A-01", "job-1")})
    assert cov.coverage is None
    with pytest.raises(MissingHumanLabelsError, match="nothing to score"):
        require_relevance_coverage(cov, min_coverage=0.0)


# --------------------------------------------------- the guard reaches claim_agreement
def test_claim_agreement_refuses_stale_labels(tmp_path):
    import pandas as pd

    from jobrec_eval.annotation import claim_agreement

    csv = tmp_path / "claims_human.csv"
    pd.DataFrame([{"experiment_id": "exp-old", "annotation_signature": "sig-old",
                   "claim_id": "c1", "validator": 1, "rater_1": 1, "rater_2": 1}]
                 ).to_csv(csv, index=False)

    occurrences = claim_occurrences("exp-new", [{
        "run_id": "r", "scenario_id": "SC-A-01", "variant": "full", "repeat_index": 0,
        "claims": [_claim()], "evidence_by_id": {}}])

    # Without linkage it still reports, which is the legacy behaviour this guard fixes.
    assert claim_agreement(csv) is not None
    # strict=True raises; the default reports N/A with the reason and NO kappa.
    with pytest.raises(StaleAnnotationError):
        claim_agreement(csv, occurrences=occurrences, experiment_id="exp-new",
                        min_coverage=0.8, strict=True)

    reported = claim_agreement(csv, occurrences=occurrences, experiment_id="exp-new",
                               min_coverage=0.8)
    assert reported["cohens_kappa"] is None
    assert reported["validator_vs_human_kappa"] is None
    assert reported["raw_agreement"] is None
    assert "not a measurement" in reported["unusable_reason"]
    assert reported["linkage"]["overlapping_signatures"] == 0
