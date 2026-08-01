"""Build deduplicated, blinded annotation items from real run bundles.

Two kinds of item, one per checklist requirement:

- **relevance** (checklist item 10): one item per unique ``(scenario_id, job_id)`` pair that
  was actually RETURNED to a candidate. Only returned pairs matter, because every reported
  ranking metric (NDCG@5, P@5, mean graded relevance) is computed over what the system
  returned; grading the rest of the catalog by hand would be thousands of judgements that no
  reported number reads.
- **claim** (checklist item 11): one item per unique ``annotation_signature`` -- per
  PROPOSITION. The signature digests what the claim asserts (type, text, predicate, field, job,
  expected and observed values, normalised arguments, evidence projection), so an identical
  proposition produced under several variants needs ONE human judgement while two propositions
  that merely read alike stay apart. Every occurrence is recorded with its
  ``experiment_id``, ``run_id``, ``repeat_index``, ``claim_id`` and ``delivery_status``, so the
  export can expand the single judgement back to one CSV row per occurrence and a label can be
  traced to the run it was made from.

  NOT keyed on ``claim_id``: that digests the rendered SENTENCE, so it merges propositions that
  format identically at different values. Measured on the 210-run deterministic pilot, 416
  claim_ids spanned 694 signatures and 278 propositions -- 40% -- would never have been judged
  as themselves.

Blinding (enforced again in the store, see
:data:`~jobrec_eval.annotation_ui.store.BLINDED_FIELD_NAMES`): the automatic oracle's grade
and the claim validator's ``support_status``/``supported_binary`` are built into
:attr:`~jobrec_eval.annotation_ui.store.AnnotationItem.analysis`, never into ``payload``. The
payload builders below take only rater-facing inputs, so there is no place in the call graph
where a blinded value could reach a payload. The analysis side is still kept, because the
export needs the ``validator`` column and the analysis compares oracle against human.

Evidence is resolved through :meth:`jobrec_eval.loaders.RunBundle.resolve_claim_evidence`,
which reports the ids that do NOT resolve. An unresolvable citation is exactly what checklist
item 11 asks a rater to check ("检查 claim 对应 evidence ID 是否可解析"), so it is carried into
the payload as a flagged list rather than dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jobrec.catalog import load_catalog
from jobrec.domain.job import JobPosting

from ..annotation_linkage import (
    DELIVERY_DELIVERED,
    DELIVERY_DROPPED,
    annotation_signature,
)
from ..loaders import RunBundle, load_bundles, normalize
from ..scenarios import Scenario, load_scenarios
from .store import KIND_CLAIM, KIND_RELEVANCE, AnnotationItem, ClaimOccurrence

#: ``item_key`` prefixes, so relevance and claim keys can never collide in one table.
RELEVANCE_KEY_PREFIX = "rel"
CLAIM_KEY_PREFIX = "clm"

#: Evidence source value whose ``source_object_id`` is a catalog ``job_id``. Used to work out
#: which posting a claim is about: ``ResponseClaim`` carries no ``job_id``, so the posting is
#: identified from the evidence the claim itself cites.
JOB_EVIDENCE_SOURCE = "job_posting"

#: The 0-3 relevance scale from checklist item 10, shipped inside every relevance payload so
#: the rater's screen and the exported grades cannot drift apart.
RELEVANCE_GRADE_SCALE = {
    "0": "irrelevant",
    "1": "weak fit",
    "2": "partial fit",
    "3": "strong fit",
}

#: The binary claim scale from checklist item 11.
CLAIM_LABEL_SCALE = {
    "1": "supported by the cited evidence",
    "0": "not supported by the cited evidence",
}

#: Evidence-item fields a rater needs to judge a citation: what field of what object, the
#: normalized value, the raw text it came from and which side produced it.
EVIDENCE_FIELDS = ("field_name", "normalized_value", "raw_text", "source", "source_object_id")

#: Posting fields put on a CLAIM screen. Deliberately narrower than the relevance payload:
#: the task is "does the posting support this sentence", so the comparable field values are
#: shown and the prose description is left out to keep the comparison on one screen.
CLAIM_JOB_FIELDS = ("job_id", "title", "company", "location", "work_mode", "employment_type",
                    "salary", "required_skills", "preferred_skills", "min_years_experience",
                    "experience_level", "application_deadline", "is_active")


def relevance_item_key(scenario_id: str, job_id: str) -> str:
    """Stable key for a relevance item (one unique returned ``(scenario, job)`` pair)."""
    return f"{RELEVANCE_KEY_PREFIX}::{scenario_id}::{job_id}"


def claim_item_key(annotation_signature: str) -> str:
    """Stable key for a claim item, derived from the PROPOSITION it stands for.

    Deterministic in the signature, so rebuilding the store from the same bundles reproduces
    the same keys and the rebuild is idempotent.

    Not derived from ``claim_id``: that digests the rendered sentence, so it merged 278 of the
    694 propositions in the 210-run pilot into 416 items.
    """
    return f"{CLAIM_KEY_PREFIX}::{annotation_signature}"


def _claims_with_delivery(bundle: RunBundle):
    """Every claim in a bundle, paired with whether the user saw it.

    Dropped claims are included because they are the only sample a validator false-NEGATIVE
    estimate can be measured on. The status is carried explicitly rather than inferred, so a
    withheld claim can never be counted later as a delivered explanation.
    """
    for claim in bundle.claims:
        yield claim, DELIVERY_DELIVERED
    for claim in bundle.dropped_claims:
        yield claim, DELIVERY_DROPPED


def _in_scope(signature: str, delivery: str,
              dropped_sample: frozenset[str] | None) -> bool:
    """Whether this occurrence belongs in the annotation workload.

    Delivered claims are always in scope: they are the explanations under test. Withheld ones
    are in scope only when the PRE-REGISTERED sample names them, because there can be far more
    of them than a rater can judge and a frame chosen after the fact makes coverage circular --
    it reports 100% for whatever happened to get annotated.

    ``dropped_sample is None`` means no frame was registered, and then every withheld claim is
    kept. That is the stricter reading, and the right default when the caller has not decided.
    """
    if delivery == DELIVERY_DELIVERED:
        return True
    return dropped_sample is None or signature in dropped_sample


def job_payload(job: JobPosting | None, fields: Sequence[str] | None = None) -> dict[str, Any]:
    """A posting's readable fields for a rater's screen.

    ``None`` for a job id that is not in the catalog snapshot: reported as
    ``{"job_id": ..., "missing_from_catalog": True}`` rather than omitted, so a rater sees why
    there is nothing to compare against instead of a blank panel.
    """
    if job is None:
        return {}
    full: dict[str, Any] = {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company,
        "description": job.description,
        "responsibilities": list(job.responsibilities),
        "location": {"city": job.city, "region": job.region, "country": job.country},
        "work_mode": job.work_mode,
        "employment_type": job.employment_type,
        "salary": {
            "min": job.salary_min, "max": job.salary_max,
            "currency": job.salary_currency, "period": job.salary_period,
        },
        "required_skills": list(job.required_skills),
        "preferred_skills": list(job.preferred_skills),
        "min_years_experience": job.min_years_experience,
        "experience_level": job.experience_level,
        "application_deadline": (job.application_deadline.isoformat()
                                 if job.application_deadline else None),
        "is_active": bool(job.is_active),
    }
    if fields is None:
        return full
    return {name: full[name] for name in fields if name in full}


def _scenario_payload(scenario: Scenario | None, scenario_id: str) -> dict[str, Any]:
    """Candidate profile plus the conversation turns IN ORDER.

    Turn order is the point: a later turn can revise an earlier preference, so a rater judging
    fit has to read them in sequence, not as a set.
    """
    if scenario is None:
        return {"scenario_id": scenario_id, "missing_from_scenario_file": True,
                "candidate_profile": {}, "conversation": []}
    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario.scenario_type,
        "candidate_profile": dict(scenario.profile),
        "conversation": [{"turn_index": i, "candidate_utterance": text}
                         for i, text in enumerate(scenario.turns)],
        "acceptable_clarification_slots": list(scenario.acceptable_slots),
    }


@dataclass(frozen=True)
class BuildStats:
    """Counts behind the annotation workload, including the dedup saving.

    Reported by the CLI so the size of the human pass is a measured number, not an estimate:
    ``claim_occurrences`` is how many rows the CSV will hold, ``claim_items`` is how many
    judgements a rater actually has to make.
    """

    experiment_id: str
    runs: int
    variants: tuple[str, ...]
    scenarios: int
    returned_pairs: int
    relevance_items: int
    claim_occurrences: int
    claim_items: int
    claims_with_unresolved_evidence: int
    claims_with_multiple_occurrences: int
    relevance_items_with_oracle_grade: int
    #: Items split by whether the user saw the proposition. Reported because they answer
    #: different questions -- delivered claims measure the explanations under test, withheld
    #: ones estimate the validator's false-negative rate -- and pooling them would conflate the
    #: two. Also the number that shows a withheld sample was actually drawn.
    delivered_claim_items: int = 0
    dropped_claim_items: int = 0
    #: How many distinct ``claim_id`` values the claim items span. Reported beside
    #: ``claim_items`` so the gap between the two -- 416 against 694 on the 210-run pilot -- is
    #: visible in the build output rather than only in an audit script.
    claim_ids: int = 0

    @property
    def claim_dedup_ratio(self) -> float:
        """Occurrences per unique claim: how much the content-addressed id saves."""
        return (self.claim_occurrences / self.claim_items) if self.claim_items else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "runs": self.runs,
            "variants": list(self.variants),
            "scenarios": self.scenarios,
            "returned_pairs": self.returned_pairs,
            "relevance_items": self.relevance_items,
            "claim_occurrences": self.claim_occurrences,
            "claim_items": self.claim_items,
            "claim_ids": self.claim_ids,
            "delivered_claim_items": self.delivered_claim_items,
            "dropped_claim_items": self.dropped_claim_items,
            "claim_dedup_ratio": round(self.claim_dedup_ratio, 4),
            "claims_with_unresolved_evidence": self.claims_with_unresolved_evidence,
            "claims_with_multiple_occurrences": self.claims_with_multiple_occurrences,
            "relevance_items_with_oracle_grade": self.relevance_items_with_oracle_grade,
        }


@dataclass(frozen=True)
class ItemBuildResult:
    """The built items plus the counts that describe the annotation workload."""

    relevance_items: tuple[AnnotationItem, ...] = ()
    claim_items: tuple[AnnotationItem, ...] = ()
    stats: BuildStats | None = None
    #: Inputs the items were built from, written into the store's ``meta``.
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def all_items(self) -> tuple[AnnotationItem, ...]:
        """Relevance items followed by claim items, both in key order."""
        return self.relevance_items + self.claim_items


def _oracle_grade_lookup(oracle_labels: pd.DataFrame | str | Path | None
                         ) -> dict[tuple[str, str], int]:
    """``(scenario_id, job_id) -> oracle grade``, for the ANALYSIS side only.

    Accepts the in-memory table from :func:`jobrec_eval.relevance.grade_catalog` or the
    ``normalized/relevance_labels.csv`` the pipeline writes. Absent labels are fine: the
    analysis-side grade is simply ``None`` and the oracle-vs-human comparison skips the pair.
    """
    if oracle_labels is None:
        return {}
    frame = (oracle_labels if isinstance(oracle_labels, pd.DataFrame)
             else pd.read_csv(oracle_labels))
    if frame.empty:
        return {}
    return {(str(row.scenario_id), str(row.job_id)): int(row.relevance_grade)
            for row in frame.itertuples()}


def _returned_pairs(bundles: Sequence[RunBundle]) -> pd.DataFrame:
    """The ``(scenario_id, job_id, rank, run_id, variant)`` rows actually returned."""
    recommendations = normalize(list(bundles))["recommendations"]
    if recommendations.empty:
        return pd.DataFrame(columns=["scenario_id", "job_id", "rank", "run_id", "variant"])
    return recommendations


def _build_relevance_items(bundles: Sequence[RunBundle], scenarios: dict[str, Scenario],
                           catalog: dict[str, JobPosting],
                           oracle: dict[tuple[str, str], int]) -> tuple[
                               tuple[AnnotationItem, ...], int]:
    """One item per unique returned pair; returns the items and the raw returned-row count."""
    returned = _returned_pairs(bundles)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in returned.itertuples():
        key = (str(row.scenario_id), str(row.job_id))
        grouped.setdefault(key, []).append({
            "run_id": str(row.run_id), "variant": str(row.variant),
            "rank": (None if pd.isna(row.rank) else int(row.rank)),
        })

    items: list[AnnotationItem] = []
    for (scenario_id, job_id) in sorted(grouped):
        occurrences = grouped[(scenario_id, job_id)]
        payload = {
            "item_kind": KIND_RELEVANCE,
            "task": "Grade how well this job posting fits this candidate's stated need.",
            "grade_scale": RELEVANCE_GRADE_SCALE,
            "scenario": _scenario_payload(scenarios.get(scenario_id), scenario_id),
            "job": job_payload(catalog.get(job_id)) or {"job_id": job_id,
                                                        "missing_from_catalog": True},
        }
        analysis = {
            # Blinded: the oracle's own answer, kept for the oracle-vs-human comparison.
            "oracle_grade": oracle.get((scenario_id, job_id)),
            # Rank is the system's opinion about this pair, so it stays out of the payload
            # too: showing it would anchor the rater on the ranking under test.
            "returned_by": sorted(occurrences, key=lambda o: (o["variant"], o["run_id"])),
        }
        items.append(AnnotationItem(
            item_key=relevance_item_key(scenario_id, job_id), kind=KIND_RELEVANCE,
            payload=payload, analysis=analysis, scenario_id=scenario_id, job_id=job_id))
    return tuple(items), int(len(returned))


def _evidence_entry(item: dict[str, Any]) -> dict[str, Any]:
    """One resolved evidence record, reduced to the fields a rater judges it by."""
    return {name: item.get(name) for name in EVIDENCE_FIELDS}


def _build_claim_items(bundles: Sequence[RunBundle], scenarios: dict[str, Scenario],
                       catalog: dict[str, JobPosting],
                       experiment_id: str = "",
                       dropped_sample: frozenset[str] | None = None) -> tuple[
                           tuple[AnnotationItem, ...], int]:
    """One item per unique ``annotation_signature`` -- per PROPOSITION, not per sentence.

    Keyed on ``claim_id`` until schema v2, which was wrong in a way that only showed up when
    measured: ``claim_id`` digests the rendered SENTENCE, so one id covers several propositions
    whenever a sentence formats identically at different values. On the 210-run deterministic
    pilot, 416 claim_ids spanned 694 signatures; 178 ids covered more than one proposition and
    278 propositions -- 40% of the total -- would never have been judged as themselves. Their
    labels would have been inherited from whichever occurrence a rater happened to see, over
    evidence unioned across up to five different propositions.

    Merge rules, chosen so nothing a rater needs is lost and nothing they must not see is added:

    - occurrences of ONE signature merge. They assert the same thing about the same field, job
      and values, and the signature is computed over the evidence projection, so their evidence
      is equivalent in content even where the per-session ids differ.
    - occurrences of DIFFERENT signatures never merge and never share evidence. That is the
      whole point of the key change.
    - an id counts as UNRESOLVABLE only when it resolves in no occurrence OF THAT SIGNATURE.
      Evidence ids are content-addressed per session, so an id that resolves in one run is not
      a dangling citation; one that resolves nowhere is, and it stays visible to the rater.
    - per-occurrence ``support_status``, ``validator_label`` and resolution completeness stay on
      the analysis side, because the validator ran per run and a rater must not see its verdict.

    Both delivered and dropped claims become items. A withheld claim is the only sample a
    validator false-NEGATIVE estimate can be measured on, and ``delivery_status`` keeps it
    distinguishable from what the user actually saw. ``dropped_sample`` restricts the withheld
    ones to a pre-registered frame; see :func:`_in_scope`.
    """
    merged: dict[str, dict[str, Any]] = {}
    occurrence_count = 0
    for bundle in bundles:
        evidence_by_id = {str(i.get("evidence_id")): i for i in bundle.evidence_items}
        for claim, delivery in _claims_with_delivery(bundle):
            claim_id = str(claim.get("claim_id") or "")
            if not claim_id:
                continue
            signature = annotation_signature(claim, evidence_by_id)
            if not _in_scope(signature, delivery, dropped_sample):
                continue
            occurrence_count += 1
            resolved = bundle.resolve_claim_evidence(claim)
            entry = merged.setdefault(signature, {
                "annotation_signature": signature,
                "claim_id": claim_id,
                "claim_type": str(claim.get("claim_type") or ""),
                "text": str(claim.get("text") or ""),
                # The structured proposition, shown to the rater. The sentence alone does not
                # say which field, which job or which values are being asserted, and a rater
                # judging "is this supported" needs exactly that.
                "predicate": claim.get("predicate"),
                "field_name": claim.get("field_name"),
                "job_id": claim.get("job_id"),
                "expected_value": claim.get("expected_value"),
                "observed_value": claim.get("observed_value"),
                "claim_args": dict(claim.get("claim_args") or {}),
                "delivery_status": delivery,
                "evidence": {},
                "cited_ids": [],
                "unresolved": {},
                "job_ids": [],
                "scenario_ids": [],
                "occurrences": [],
            })
            for cited in claim.get("evidence_ids") or []:
                if str(cited) not in entry["cited_ids"]:
                    entry["cited_ids"].append(str(cited))
            for evidence in resolved.items:
                evidence_id = str(evidence.get("evidence_id"))
                entry["evidence"].setdefault(evidence_id, _evidence_entry(evidence))
                if (evidence.get("source") == JOB_EVIDENCE_SOURCE
                        and evidence.get("source_object_id")
                        and evidence["source_object_id"] not in entry["job_ids"]):
                    entry["job_ids"].append(str(evidence["source_object_id"]))
            for unresolved_id in resolved.unresolved_ids:
                entry["unresolved"].setdefault(str(unresolved_id), 0)
                entry["unresolved"][str(unresolved_id)] += 1
            if bundle.scenario_id not in entry["scenario_ids"]:
                entry["scenario_ids"].append(bundle.scenario_id)
            status = str(claim.get("support_status") or "")
            if delivery == DELIVERY_DELIVERED:
                # Seen at least once, so the item is a delivered one; the per-occurrence split
                # stays visible in item_occurrences.
                entry["delivery_status"] = DELIVERY_DELIVERED
            entry["occurrences"].append(ClaimOccurrence(
                run_id=bundle.run_id, claim_id=claim_id, variant=bundle.variant,
                scenario_id=bundle.scenario_id,
                validator_label=1 if status == "supported" else 0,
                support_status=status, fully_resolved=resolved.fully_resolved,
                experiment_id=experiment_id, repeat_index=bundle.run_index,
                annotation_signature=signature, delivery_status=delivery))

    items: list[AnnotationItem] = []
    for signature in sorted(merged):
        entry = merged[signature]
        claim_id = entry["claim_id"]
        resolved_ids = set(entry["evidence"])
        unresolvable = [cited for cited in entry["cited_ids"]
                        if cited in entry["unresolved"] and cited not in resolved_ids]
        scenario_ids = entry["scenario_ids"]
        representative = (_scenario_payload(scenarios.get(scenario_ids[0]), scenario_ids[0])
                          if len(scenario_ids) == 1 else None)
        payload = {
            "item_kind": KIND_CLAIM,
            "task": ("Decide whether the cited evidence supports this sentence. "
                     "A citation that resolves to nothing cannot support anything."),
            "label_scale": CLAIM_LABEL_SCALE,
            "claim_id": claim_id,
            "annotation_signature": signature,
            "claim_type": entry["claim_type"],
            "claim_text": entry["text"],
            # The structured proposition, so the rater judges what was ASSERTED rather than
            # inferring it from the sentence. These are this signature's values and no other's.
            "predicate": entry["predicate"],
            "claim_field": entry["field_name"],
            "claim_job_id": entry["job_id"],
            "expected_value": entry["expected_value"],
            "observed_value": entry["observed_value"],
            "claim_args": entry["claim_args"],
            "cited_evidence_count": len(entry["cited_ids"]),
            # Merged across the runs that produced THIS proposition, and across no others. The
            # signature is computed over the evidence projection, so these occurrences assert
            # the same thing about the same values -- the citations differ only in per-session
            # ids. Under the old claim_id key this union spanned up to five distinct
            # propositions, which is what the key change fixes.
            "occurrence_count": len(entry["occurrences"]),
            "evidence_merged_across_runs": len(entry["occurrences"]) > 1,
            "evidence": [entry["evidence"][key] for key in entry["evidence"]],
            "unresolvable_evidence_ids": unresolvable,
            "has_unresolvable_evidence": bool(unresolvable),
            "referenced_jobs": [job_payload(catalog.get(job_id), CLAIM_JOB_FIELDS)
                                or {"job_id": job_id, "missing_from_catalog": True}
                                for job_id in entry["job_ids"]],
            "scenario_ids": list(scenario_ids),
            # Whether the user saw this sentence. Shown because it changes the question: a
            # withheld claim is being judged for what the validator SHOULD have concluded, not
            # for what was presented. The validator's own verdict stays analysis-side.
            "delivery_status": entry["delivery_status"],
        }
        if representative is not None:
            payload["scenario"] = representative
        occurrences = tuple(sorted(entry["occurrences"], key=lambda o: (o.run_id, o.claim_id)))
        analysis = {
            # Blinded: the claim validator's verdict, per run, plus how the citations resolved.
            "validator_support_status": {occ.run_id: occ.support_status for occ in occurrences},
            "validator_supported_binary": {occ.run_id: occ.validator_label
                                           for occ in occurrences},
            "occurrence_count": len(occurrences),
            "variants": sorted({occ.variant for occ in occurrences}),
            "unresolved_id_counts": dict(sorted(entry["unresolved"].items())),
            "fully_resolved_occurrences": sum(1 for occ in occurrences if occ.fully_resolved),
        }
        items.append(AnnotationItem(
            item_key=claim_item_key(signature), kind=KIND_CLAIM, payload=payload,
            analysis=analysis, claim_id=claim_id, annotation_signature=signature,
            scenario_id=scenario_ids[0] if len(scenario_ids) == 1 else None,
            occurrences=occurrences))
    return tuple(items), occurrence_count


def build_items(experiment_dir: str | Path, scenarios_path: str | Path,
                catalog_path: str | Path, *,
                oracle_labels: pd.DataFrame | str | Path | None = None,
                bundles: Iterable[RunBundle] | None = None,
                dropped_sample: Iterable[str] | None = None) -> ItemBuildResult:
    """Build every annotation item for one experiment directory.

    Args:
        experiment_dir: ``<out_root>/_runs/<experiment_id>``, the directory
            :func:`jobrec_eval.loaders.load_bundles` walks.
        scenarios_path: The scenario JSONL the run used; supplies the candidate profile and
            the conversation turns a rater reads.
        catalog_path: The catalog JSONL the run used; supplies the posting fields.
        oracle_labels: Optional automatic-oracle relevance table (or the path to
            ``normalized/relevance_labels.csv``). Used for the ANALYSIS side only -- it is
            never put in a payload, so passing it cannot unblind a rater.
        bundles: Pre-loaded bundles, for tests that construct bundles directly instead of
            reading a directory.
        dropped_sample: The pre-registered withheld signatures, from
            :meth:`jobrec_eval.annotation_linkage.AnnotationUniverse.dropped_sampled`. When
            given, only these withheld claims become items; delivered claims are unaffected.
            ``None`` keeps every withheld claim.

    Returns:
        ItemBuildResult: items plus :class:`BuildStats`.
    """
    experiment_dir = Path(experiment_dir)
    loaded = list(bundles) if bundles is not None else load_bundles(experiment_dir)
    scenarios = load_scenarios(scenarios_path)
    catalog = {job.job_id: job for job in load_catalog(catalog_path)}
    oracle = _oracle_grade_lookup(oracle_labels)

    relevance_items, returned_pairs = _build_relevance_items(loaded, scenarios, catalog, oracle)
    claim_items, claim_occurrences = _build_claim_items(
        loaded, scenarios, catalog, experiment_id=Path(experiment_dir).name,
        dropped_sample=(None if dropped_sample is None else frozenset(dropped_sample)))

    stats = BuildStats(
        experiment_id=experiment_dir.name,
        runs=len(loaded),
        variants=tuple(sorted({b.variant for b in loaded})),
        scenarios=len({b.scenario_id for b in loaded}),
        returned_pairs=returned_pairs,
        relevance_items=len(relevance_items),
        claim_occurrences=claim_occurrences,
        claim_items=len(claim_items),
        claims_with_unresolved_evidence=sum(
            1 for item in claim_items if item.payload["has_unresolvable_evidence"]),
        claims_with_multiple_occurrences=sum(
            1 for item in claim_items if len(item.occurrences) > 1),
        relevance_items_with_oracle_grade=sum(
            1 for item in relevance_items if item.analysis.get("oracle_grade") is not None),
        delivered_claim_items=sum(
            1 for item in claim_items
            if item.payload["delivery_status"] == DELIVERY_DELIVERED),
        dropped_claim_items=sum(
            1 for item in claim_items
            if item.payload["delivery_status"] == DELIVERY_DROPPED),
        claim_ids=len({item.claim_id for item in claim_items}),
    )
    return ItemBuildResult(
        relevance_items=relevance_items, claim_items=claim_items, stats=stats,
        sources={
            "experiment_dir": str(experiment_dir),
            "scenarios_path": str(scenarios_path),
            "catalog_path": str(catalog_path),
        })
