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
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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


def annotation_items(occurrences: list[dict]) -> list[dict]:
    """One row per SIGNATURE -- the unit a rater judges -- with its occurrences counted.

    Deduplication is by ``annotation_signature``, never by ``claim_id``: that id digests the
    rendered sentence, so it merges propositions that read alike at different values. One item
    per signature is therefore both the smallest honest unit of work and the largest safe one.

    Evidence is NOT merged across signatures. Each item carries the projection of its own
    signature only, because a rater shown the union of two propositions' evidence is being
    asked a question neither claim makes.

    Dropped claims get their own items. They are the only way to estimate the validator's
    false-negative rate, and ``delivery_status`` keeps them distinguishable from what the user
    actually saw.
    """
    items: dict[str, dict] = {}
    for row in occurrences:
        signature = row["annotation_signature"]
        item = items.get(signature)
        if item is None:
            items[signature] = {
                "annotation_signature": signature,
                "experiment_id": row.get("experiment_id"),
                "claim_type": row.get("claim_type"),
                "predicate": row.get("predicate"),
                "text": row.get("text"),
                "claim_field": row.get("claim_field"),
                "claim_job_id": row.get("claim_job_id"),
                "delivery_status": row.get("delivery_status"),
                "validator": (1 if row.get("support_status") == "supported" else 0),
                "occurrence_count": 1,
                "example_run_id": row.get("run_id"),
                "example_claim_id": row.get("claim_id"),
            }
            continue
        item["occurrence_count"] += 1
        # A signature seen both delivered and withheld is reported as delivered, because the
        # user did see it at least once; the split stays visible in the occurrence table.
        if row.get("delivery_status") == DELIVERY_DELIVERED:
            item["delivery_status"] = DELIVERY_DELIVERED
    return sorted(items.values(), key=lambda r: str(r["annotation_signature"]))


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
                    # The rendered sentence, so an annotation task can be built from this
                    # table alone rather than joined back to the bundles.
                    "text": claim.get("text"),
                    "claim_field": claim.get("field_name"),
                    "claim_job_id": claim.get("job_id"),
                })
    return rows


# ------------------------------------------------------- pre-registered annotation universe
#: Fields a dropped claim is stratified on before sampling. Chosen because they are what a
#: validator false-negative rate is expected to vary along: the KIND of proposition, the
#: PREDICATE that decides it, and the verdict that got it withheld. Sampling without strata
#: would let one populous claim type swallow the sample and leave the rest unestimated.
DROPPED_STRATA_FIELDS: tuple[str, ...] = ("claim_type", "predicate", "support_status")

#: Default withheld signatures drawn per stratum. Small on purpose: the withheld sample exists
#: to ESTIMATE a false-negative rate, and every drawn item is human work.
DEFAULT_DROPPED_PER_STRATUM = 5

#: Seed for the stratified draw. Fixed and recorded, so the frame is reproducible and was
#: demonstrably not chosen after seeing which claims were easy to label. Defined here rather
#: than in :mod:`jobrec_eval.cli` so the annotation tool and the analysis pipeline read ONE
#: number -- the frame a rater works from has to be the frame coverage is measured against, and
#: the annotation package deliberately does not import the CLI (and with it the plotting stack).
DEFAULT_SAMPLING_SEED = 20260801


def stratum_of(row: dict, fields: Sequence[str] = DROPPED_STRATA_FIELDS) -> tuple[str, ...]:
    """The stratum key of one occurrence row."""
    return tuple(str(row.get(name) or "") for name in fields)


@dataclass(frozen=True)
class AnnotationUniverse:
    """The signatures that were PRE-REGISTERED for annotation, and how they were chosen.

    This is the denominator of claim coverage. Coverage against "whatever got annotated" is
    circular -- it reports 100% for any sample -- so the frame is fixed first, written to a
    manifest with its seed and strata rules, and coverage is measured against it afterwards.

    Delivered signatures are taken WHOLE: they are what the user saw, so leaving any of them
    out would mean reporting agreement over a subset of the explanations under test. Withheld
    (``dropped``) signatures are stratified and sampled, because they exist only to estimate the
    validator's false-negative rate and there can be far more of them than a rater can judge.
    """

    experiment_id: str
    delivered: tuple[str, ...]
    dropped_sampled: tuple[str, ...]
    dropped_population: tuple[str, ...]
    seed: int
    dropped_per_stratum: int
    strata_fields: tuple[str, ...]
    #: ``stratum -> (population size, sampled size)``, so a reader can see which strata were
    #: exhausted and which were subsampled.
    strata: tuple[tuple[tuple[str, ...], int, int], ...] = ()
    created_at: str = ""

    @property
    def signatures(self) -> frozenset[str]:
        """Every signature in the frame: all delivered, plus the sampled withheld ones."""
        return frozenset(self.delivered) | frozenset(self.dropped_sampled)

    @property
    def size(self) -> int:
        return len(self.signatures)

    def manifest(self) -> dict:
        """The pre-registration record: seed, rules, per-stratum counts and the frame itself.

        The signature lists are included in full rather than digested. A digest proves the
        frame did not change; the list is what lets coverage be RECOMPUTED later, which is the
        point of pre-registering a denominator.
        """
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "sampling_seed": self.seed,
            "strata_fields": list(self.strata_fields),
            "dropped_per_stratum": self.dropped_per_stratum,
            "sampling_rule": (
                "every delivered signature is included; withheld signatures are grouped by "
                f"{'+'.join(self.strata_fields)} and up to {self.dropped_per_stratum} are drawn "
                "per stratum with a per-stratum seeded RNG, so adding a stratum cannot reshuffle "
                "the draw of any other"),
            "counts": {
                "universe_signatures": self.size,
                "delivered_signatures": len(self.delivered),
                "dropped_population_signatures": len(self.dropped_population),
                "dropped_sampled_signatures": len(self.dropped_sampled),
                "strata": len(self.strata),
            },
            "strata": [{"stratum": dict(zip(self.strata_fields, key, strict=True)),
                        "population": population, "sampled": sampled}
                       for key, population, sampled in self.strata],
            "delivered": list(self.delivered),
            "dropped_sampled": list(self.dropped_sampled),
        }


def build_annotation_universe(
    experiment_id: str,
    occurrences: list[dict],
    *,
    seed: int = DEFAULT_SAMPLING_SEED,
    dropped_per_stratum: int = DEFAULT_DROPPED_PER_STRATUM,
    strata_fields: Sequence[str] = DROPPED_STRATA_FIELDS,
) -> AnnotationUniverse:
    """Fix the annotation frame for one experiment, before any labelling happens.

    Deterministic in ``(seed, occurrences)``: the draw is made from the SORTED signatures of
    each stratum with an RNG seeded on ``(seed, stratum)``, so rebuilding the frame from the
    same bundles reproduces it exactly, and a stratum appearing or growing cannot change which
    signatures were drawn from any other stratum.

    A signature that appears both delivered and withheld counts as delivered: the user saw it
    at least once, so it is in the frame whole rather than by sample.
    """
    delivered: set[str] = set()
    dropped: dict[tuple[str, ...], set[str]] = {}
    dropped_rows: dict[str, tuple[str, ...]] = {}
    for row in occurrences:
        signature = str(row["annotation_signature"])
        if row.get("delivery_status") == DELIVERY_DELIVERED:
            delivered.add(signature)
        else:
            key = stratum_of(row, strata_fields)
            dropped.setdefault(key, set()).add(signature)
            dropped_rows.setdefault(signature, key)

    sampled: set[str] = set()
    strata: list[tuple[tuple[str, ...], int, int]] = []
    for key in sorted(dropped):
        # Delivered wins, so a signature already in the frame is not also sampled.
        population = sorted(dropped[key] - delivered)
        if not population:
            strata.append((key, 0, 0))
            continue
        take = min(dropped_per_stratum, len(population))
        rng = random.Random(f"{seed}|{'|'.join(key)}")  # noqa: S311 - sampling, not crypto
        drawn = sorted(rng.sample(population, take))
        sampled.update(drawn)
        strata.append((key, len(population), len(drawn)))

    return AnnotationUniverse(
        experiment_id=experiment_id,
        delivered=tuple(sorted(delivered)),
        dropped_sampled=tuple(sorted(sampled)),
        dropped_population=tuple(sorted(set(dropped_rows) - delivered)),
        seed=seed, dropped_per_stratum=dropped_per_stratum,
        strata_fields=tuple(strata_fields), strata=tuple(strata),
        created_at=datetime.now(UTC).isoformat(),
    )


#: Why a human label row was refused entry to kappa.
EXCLUDED_OTHER_EXPERIMENT = "other_experiment"
EXCLUDED_NO_SIGNATURE = "no_signature"
EXCLUDED_OBSOLETE = "obsolete_signature"
EXCLUDED_OUT_OF_UNIVERSE = "outside_pre_registered_universe"


@dataclass(frozen=True)
class LabelFilterResult:
    """Which label rows may enter kappa, and why the rest may not."""

    #: Row indices to keep, in input order.
    kept: tuple[int, ...]
    #: ``reason -> count`` for the refused rows.
    excluded: dict[str, int]

    @property
    def n_excluded(self) -> int:
        return sum(self.excluded.values())

    def as_dict(self) -> dict:
        return {"kept_rows": len(self.kept), "excluded_rows": self.n_excluded,
                "excluded_by_reason": dict(sorted(self.excluded.items()))}


def filter_labels_to_universe(
    labels: list[dict],
    *,
    experiment_id: str,
    universe: frozenset[str] | set[str],
    produced: frozenset[str] | set[str] | None = None,
) -> LabelFilterResult:
    """Keep only the label rows that describe THIS experiment's pre-registered frame.

    Four kinds of row are refused, each counted separately so the exclusion is auditable
    rather than a silent drop:

    - a row from another ``experiment_id`` -- it was made from runs this analysis is not about;
    - a row with no ``annotation_signature`` -- every v1 file is like this, and there is no way
      to tell which proposition it judged, because ``claim_id`` covered several;
    - a row whose signature no longer exists (obsolete): the proposition changed or went away;
    - a row inside the experiment but outside the pre-registered frame -- a withheld claim that
      was not sampled, say. Counting it would make coverage's denominator depend on what
      happened to get annotated, which reports 100% for any sample.

    ``experiment_id`` is only enforced where the row states one. A file without the column is
    not assumed to be foreign; its rows still have to match on signature, which is the check
    that actually ties a label to these runs.

    Args:
        produced: Every signature the experiment produced, used only to tell the two
            out-of-frame cases apart: a signature that does not exist at all is OBSOLETE, while
            one that exists but was left out of the frame is out of scope. Without it both are
            reported as out of scope, which is true but less useful.
    """
    kept: list[int] = []
    excluded: dict[str, int] = {}

    def refuse(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for index, row in enumerate(labels):
        row_experiment = str(row.get("experiment_id") or "")
        if row_experiment and experiment_id and row_experiment != experiment_id:
            refuse(EXCLUDED_OTHER_EXPERIMENT)
            continue
        signature = str(row.get("annotation_signature") or "")
        if not signature:
            refuse(EXCLUDED_NO_SIGNATURE)
            continue
        if signature not in universe:
            refuse(EXCLUDED_OBSOLETE if produced is not None and signature not in produced
                   else EXCLUDED_OUT_OF_UNIVERSE)
            continue
        kept.append(index)
    return LabelFilterResult(kept=tuple(kept), excluded=excluded)


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
    #: Set when a pre-registered universe supplied the denominator, so a reader can tell a
    #: coverage measured against a fixed frame from one measured against whatever was produced.
    universe_signatures: int | None = None
    #: Signatures the experiment produced that the frame deliberately left out (unsampled
    #: withheld claims). Not missing annotation -- out of scope by pre-registration.
    out_of_universe_signatures: int = 0
    #: Label rows refused entry to kappa, by reason.
    excluded_label_rows: dict[str, int] = field(default_factory=dict)

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
            "universe_signatures": self.universe_signatures,
            "out_of_universe_signatures": self.out_of_universe_signatures,
            "excluded_label_rows": dict(sorted(self.excluded_label_rows.items())),
            "coverage": self.coverage,
            "coverage_denominator": (
                "pre_registered_universe" if self.universe_signatures is not None
                else "signatures_produced"),
            "is_stale": self.is_stale,
        }


def link_claim_labels(
    experiment_id: str,
    occurrences: list[dict],
    labels: list[dict],
    *,
    universe: AnnotationUniverse | None = None,
) -> LinkageReport:
    """Compare the signatures the experiment produced with the ones that were annotated.

    With a ``universe``, the PRE-REGISTERED frame is the denominator and labels are filtered to
    it first: rows from another experiment, rows with no signature, obsolete signatures and
    signatures deliberately left out of the frame are all excluded from the overlap, each
    counted under its own reason. Without one the denominator is every signature produced,
    which is the stricter reading and the right default when no frame was registered.
    """
    produced = {str(row["annotation_signature"]) for row in occurrences}
    denominator = set(universe.signatures) if universe is not None else produced

    if universe is not None:
        kept = filter_labels_to_universe(
            labels, experiment_id=experiment_id, universe=universe.signatures,
            produced=frozenset(produced))
        usable = [labels[i] for i in kept.kept]
        excluded = dict(kept.excluded)
    else:
        usable = list(labels)
        excluded = {}

    labelled = {str(row.get("annotation_signature")) for row in usable
                if row.get("annotation_signature")}
    all_labelled = {str(row.get("annotation_signature")) for row in labels
                    if row.get("annotation_signature")}
    return LinkageReport(
        experiment_id=experiment_id,
        label_experiment_ids=sorted({str(r.get("experiment_id")) for r in labels
                                     if r.get("experiment_id")}),
        current_signatures=len(denominator),
        labelled_signatures=len(labelled),
        overlapping_signatures=len(denominator & labelled),
        missing_signatures=len(denominator - labelled),
        obsolete_signatures=len(all_labelled - produced),
        universe_signatures=(universe.size if universe is not None else None),
        out_of_universe_signatures=len(produced - denominator),
        excluded_label_rows=excluded,
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
