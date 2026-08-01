"""Tie human annotations to the experiment they actually describe.

Why a signature, and why ``claim_id`` is not one
-----------------------------------------------
``claim_id`` is ``content_id("claim", claim_type, text, key_extra)`` -- a digest of the
RENDERED SENTENCE. Two claims whose sentences read alike therefore share an id even when they
assert different things, because the numbers a sentence quotes are formatted the same way at
different values, and because ``key_extra`` is only the job id. Merging annotations on that id
pools ratings across propositions: a rater who judged "salary meets your minimum" for a
RM4000 threshold has their label reused for an RM6000 one.

The signature digests the PROPOSITION instead: claim type, text, predicate, field, job,
expected and observed values, the normalised arguments, and a projection of the evidence the
claim cites. Anything that changes what is being asserted changes the signature, so different
thresholds, different expected values, different arguments and different evidence cannot merge.

Why linkage has to be checked at all
------------------------------------
``claim_agreement`` used to read a CSV and report kappa, with no connection to the experiment
being analysed. Old labels therefore produced a confident-looking kappa for runs they had never
seen. Kappa over zero overlapping items is not a low agreement score; it is not a measurement,
and it must fail or report N/A rather than be published.

The same applies to relevance: a missing human label for a returned pair is UNKNOWN, not
irrelevant. Defaulting it to 0 silently converts "nobody judged this" into "the system was
wrong", which biases every human-scored ranking metric downward by exactly the amount of
missing annotation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobrec.utils.hashing import stable_hash

#: Columns that identify one OCCURRENCE of a claim: which batch, which run, which delivery.
#: Kept alongside the signature so a rating can be traced to the exact run it was made from,
#: and so the same proposition appearing in several runs is counted as several occurrences.
OCCURRENCE_FIELDS: tuple[str, ...] = (
    "experiment_id", "run_id", "scenario_id", "variant", "repeat_index",
    "claim_id", "annotation_signature", "delivery_status",
)

#: Delivery states a claim can be annotated in. ``delivered`` is what the user saw; the others
#: were built and withheld. Sampling the withheld ones is the only way to estimate the
#: validator's false NEGATIVE rate, but they must never be counted as shown explanations.
DELIVERY_DELIVERED = "delivered"
DELIVERY_DROPPED = "dropped"


class StaleAnnotationError(RuntimeError):
    """Raised when human labels do not describe the experiment being analysed."""


class MissingHumanLabelsError(RuntimeError):
    """Raised when a human-scored metric is asked for without the labels to compute it."""


def _normalise(value: Any) -> Any:
    """A comparable form of a claim argument: order-insensitive, number-normalised.

    Lists are sorted because ``["onsite", "hybrid"]`` and ``["hybrid", "onsite"]`` state the
    same thing, and numbers pass through ``float`` so ``4000`` and ``4000.0`` do not split one
    proposition into two signatures. Text is stripped and case-folded for the same reason.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple | set | frozenset):
        return sorted((_normalise(v) for v in value), key=repr)
    return str(value).strip().casefold()


def evidence_projection(
    claim: dict, evidence_by_id: dict[str, dict] | None
) -> list[dict[str, Any]] | None:
    """What the cited evidence SAYS, as a sorted, id-free projection.

    Ids are not used: they are content-addressed over the turn, so the same statement made in
    two runs has two ids and would split one proposition into two signatures. The projection
    keeps source, field and value -- what a rater actually reads -- so a claim citing a
    different salary or a different skill set gets a different signature even when its sentence
    is unchanged. That is the collision the loader must never merge.

    When the evidence is not available the projection is ``None``, which is distinct from an
    empty projection: "we did not look" must not compare equal to "it cited nothing".
    """
    if evidence_by_id is None:
        return None
    out: list[dict[str, Any]] = []
    for eid in claim.get("evidence_ids") or []:
        item = evidence_by_id.get(eid)
        if item is None:
            # A cited id that does not resolve is itself part of what the claim asserts, so it
            # is projected rather than skipped: a claim that lost its evidence is not the same
            # proposition as one that never cited any.
            out.append({"missing": True})
            continue
        out.append({"source": str(item.get("source") or ""),
                    "field_name": str(item.get("field_name") or ""),
                    "value": _normalise(item.get("normalized_value"))})
    return sorted(out, key=repr)


def annotation_signature(claim: dict, evidence_by_id: dict[str, dict] | None = None) -> str:
    """A digest of WHAT a claim asserts, for merging human ratings safely.

    Deliberately includes ``text``: two propositions that differ only in wording are different
    things to a human rater, and pooling their labels would hide a rendering change. Equally
    deliberately includes the evidence projection, so the same sentence resting on different
    evidence does not merge.
    """
    return "sig-" + stable_hash({
        "claim_type": claim.get("claim_type"),
        "text": str(claim.get("text") or "").strip(),
        "predicate": claim.get("predicate"),
        "field_name": claim.get("field_name"),
        "job_id": claim.get("job_id"),
        "expected_value": _normalise(claim.get("expected_value")),
        "observed_value": _normalise(claim.get("observed_value")),
        "claim_args": _normalise(claim.get("claim_args") or {}),
        "evidence": evidence_projection(claim, evidence_by_id),
    })[:16]


def claim_occurrences(
    experiment_id: str,
    runs: list[dict],
) -> list[dict]:
    """One row per claim occurrence, delivered and dropped alike.

    ``runs`` items carry ``run_id``, ``scenario_id``, ``variant``, ``repeat_index``,
    ``claims``, ``dropped_claims`` and optionally ``evidence_by_id``.

    Dropped claims are included with ``delivery_status="dropped"``. They are what a validator
    false-negative estimate has to be measured on, and keeping them in the same table with an
    explicit status is what stops them being mistaken later for explanations the user saw.
    """
    rows: list[dict] = []
    for run in runs:
        evidence = run.get("evidence_by_id")
        for status, key in ((DELIVERY_DELIVERED, "claims"),
                            (DELIVERY_DROPPED, "dropped_claims")):
            for claim in run.get(key) or []:
                rows.append({
                    "experiment_id": experiment_id,
                    "run_id": run.get("run_id"),
                    "scenario_id": run.get("scenario_id"),
                    "variant": run.get("variant"),
                    "repeat_index": run.get("repeat_index"),
                    "claim_id": claim.get("claim_id"),
                    "annotation_signature": annotation_signature(claim, evidence),
                    "delivery_status": status,
                    "claim_type": claim.get("claim_type"),
                    "predicate": claim.get("predicate"),
                    "support_status": claim.get("support_status"),
                })
    return rows


@dataclass
class LinkageReport:
    """How much of the current experiment a set of human labels actually covers."""

    experiment_id: str
    label_experiment_ids: list[str] = field(default_factory=list)
    current_signatures: int = 0
    labelled_signatures: int = 0
    overlapping_signatures: int = 0
    missing_signatures: int = 0
    obsolete_signatures: int = 0

    @property
    def coverage(self) -> float | None:
        """Fraction of current signatures that carry a label, or ``None`` when there are none.

        ``None`` rather than 1.0 for an empty experiment: a batch with no claims has not been
        fully annotated, it has not been measured.
        """
        if not self.current_signatures:
            return None
        return self.overlapping_signatures / self.current_signatures

    @property
    def is_stale(self) -> bool:
        """True when the labels describe none of the current experiment."""
        return self.current_signatures > 0 and self.overlapping_signatures == 0

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "label_experiment_ids": sorted(self.label_experiment_ids),
            "current_signatures": self.current_signatures,
            "labelled_signatures": self.labelled_signatures,
            "overlapping_signatures": self.overlapping_signatures,
            "missing_signatures": self.missing_signatures,
            "obsolete_signatures": self.obsolete_signatures,
            "coverage": self.coverage,
            "is_stale": self.is_stale,
        }


def link_claim_labels(
    experiment_id: str,
    occurrences: list[dict],
    labels: list[dict],
) -> LinkageReport:
    """Compare the signatures the experiment produced with the ones that were annotated."""
    current = {row["annotation_signature"] for row in occurrences}
    labelled = {row.get("annotation_signature") for row in labels
                if row.get("annotation_signature")}
    return LinkageReport(
        experiment_id=experiment_id,
        label_experiment_ids=sorted({str(r.get("experiment_id")) for r in labels
                                     if r.get("experiment_id")}),
        current_signatures=len(current),
        labelled_signatures=len(labelled),
        overlapping_signatures=len(current & labelled),
        missing_signatures=len(current - labelled),
        obsolete_signatures=len(labelled - current),
    )


def require_linked(report: LinkageReport, *, min_coverage: float) -> None:
    """Raise unless the labels describe enough of THIS experiment to be reported.

    ``min_coverage`` is the pre-registered requirement. Below it the metric is not computed:
    reporting kappa or a human-scored ranking figure over a fraction of the batch, without
    saying which fraction, is the failure this guard exists for.
    """
    if report.is_stale:
        raise StaleAnnotationError(
            f"human labels describe none of experiment {report.experiment_id}: "
            f"{report.current_signatures} current signature(s), "
            f"{report.labelled_signatures} labelled, 0 overlapping. "
            f"Labels reference {report.label_experiment_ids or ['no experiment']}. "
            f"Agreement over zero overlapping items is not a low score, it is not a "
            f"measurement."
        )
    coverage = report.coverage
    if coverage is None:
        raise MissingHumanLabelsError(
            f"experiment {report.experiment_id} produced no claims to annotate")
    if coverage < min_coverage:
        raise MissingHumanLabelsError(
            f"human claim coverage {coverage:.1%} is below the required {min_coverage:.1%} "
            f"({report.overlapping_signatures}/{report.current_signatures} signatures "
            f"annotated, {report.missing_signatures} missing)"
        )


# --------------------------------------------------------------------- relevance
@dataclass
class RelevanceCoverage:
    """Which returned pairs a human relevance label set covers."""

    reused: list[tuple[str, str]] = field(default_factory=list)
    delta: list[tuple[str, str]] = field(default_factory=list)
    obsolete: list[tuple[str, str]] = field(default_factory=list)

    @property
    def coverage(self) -> float | None:
        total = len(self.reused) + len(self.delta)
        return len(self.reused) / total if total else None

    def as_dict(self) -> dict:
        return {
            "returned_pairs": len(self.reused) + len(self.delta),
            "reused_overlapping_labels": len(self.reused),
            "delta_pairs_requiring_annotation": len(self.delta),
            "obsolete_extra_labels": len(self.obsolete),
            "coverage": self.coverage,
            "delta_sample": sorted(self.delta)[:20],
            "obsolete_sample": sorted(self.obsolete)[:20],
        }


def relevance_coverage(
    returned_pairs: set[tuple[str, str]],
    labelled_pairs: set[tuple[str, str]],
) -> RelevanceCoverage:
    """Split the returned ``(scenario_id, job_id)`` pairs by whether a label exists.

    ``delta`` is the set a human still has to judge. It is NOT a set of zeros: an unlabelled
    pair is unknown, and scoring it as irrelevant biases every human ranking metric downward by
    exactly the amount of missing annotation.
    """
    return RelevanceCoverage(
        reused=sorted(returned_pairs & labelled_pairs),
        delta=sorted(returned_pairs - labelled_pairs),
        obsolete=sorted(labelled_pairs - returned_pairs),
    )


def require_relevance_coverage(cov: RelevanceCoverage, *, min_coverage: float) -> None:
    """Raise unless enough returned pairs carry a human label."""
    coverage = cov.coverage
    if coverage is None:
        raise MissingHumanLabelsError("no jobs were returned, so there is nothing to score")
    if coverage < min_coverage:
        raise MissingHumanLabelsError(
            f"human relevance coverage {coverage:.1%} is below the required "
            f"{min_coverage:.1%}: {len(cov.delta)} of "
            f"{len(cov.reused) + len(cov.delta)} returned pair(s) are unlabelled. "
            f"They are UNKNOWN, not irrelevant -- scoring them 0 would bias every "
            f"human-scored ranking metric downward by that amount."
        )


def write_coverage_report(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    return out
