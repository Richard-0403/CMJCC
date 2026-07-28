"""Deterministic, load-balanced allocation of exactly two raters per annotation item.

Why exactly two, never three: the export contract
(:mod:`jobrec_eval.annotation`) is ``rater_1, rater_2, adjudicated`` and the reported
agreement statistic is PAIRWISE -- weighted Cohen's kappa for relevance, Cohen's kappa for
claims. Three raters per item would need Fleiss' kappa (a different estimator, reported on a
different scale) and a wider CSV that
:func:`jobrec_eval.annotation.load_adjudicated_relevance_labels` does not accept. Two is
also what Chapter 3 promises. A pool of fewer than two raters is rejected outright rather
than degraded to single-rating, because a single rater yields no agreement at all.

Determinism: the allocation is a pure function of ``(item_keys, rater_pool, seed)``. Tie
breaking uses a BLAKE2b digest of ``seed|item_key|rater_id`` rather than a running RNG, so
the choice for one item does not depend on how many items were processed before it -- adding
an item to the set leaves the other items' rater pairs reproducible from the same seed.

Each rater also gets a seeded, RECORDED presentation order (``position``). Two raters walking
the same queue in the same order would drift together (fatigue, learning, anchoring on the
previous item), which inflates agreement; independent seeded orders remove that shared
sequence while keeping the whole allocation replayable from the seed.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Exactly two raters per item -- see the module docstring for why this is not configurable.
RATERS_PER_ITEM = 2


class InsufficientRaterPoolError(ValueError):
    """Fewer than :data:`RATERS_PER_ITEM` distinct raters were supplied.

    A clear error rather than a silent single-rating fallback: with one rater there is no
    inter-rater agreement to report, so the annotation pass would not satisfy checklist
    items 10/11 at all.
    """


@dataclass(frozen=True)
class Assignment:
    """One rater's slot on one item, plus that rater's queue position for it."""

    item_key: str
    rater_id: str
    slot: int
    position: int


@dataclass(frozen=True)
class AssignmentPlan:
    """A complete, reproducible allocation. Persist with ``store.save_assignment_plan``."""

    seed: int
    rater_pool: tuple[str, ...]
    assignments: tuple[Assignment, ...]

    def for_rater(self, rater_id: str) -> tuple[Assignment, ...]:
        """This rater's assignments in their presentation order."""
        return tuple(sorted((a for a in self.assignments if a.rater_id == rater_id),
                            key=lambda a: (a.position, a.item_key)))

    def load(self) -> dict[str, int]:
        """Items per rater, in pool order."""
        counts = dict.fromkeys(self.rater_pool, 0)
        for assignment in self.assignments:
            counts[assignment.rater_id] += 1
        return counts

    def raters_for(self, item_key: str) -> dict[int, str]:
        """``slot -> rater_id`` for one item."""
        return {a.slot: a.rater_id for a in self.assignments if a.item_key == item_key}

    @property
    def item_keys(self) -> tuple[str, ...]:
        """The items covered, sorted."""
        return tuple(sorted({a.item_key for a in self.assignments}))

    @property
    def max_load_imbalance(self) -> int:
        """Largest minus smallest per-rater load; 0 or 1 for a balanced plan."""
        load = self.load()
        return (max(load.values()) - min(load.values())) if load else 0


def _tiebreak(seed: int, item_key: str, rater_id: str) -> str:
    """Stable pseudo-random tiebreaker for (item, rater).

    A digest, not an RNG draw: it depends only on the triple, so the pair chosen for an item
    is independent of the processing order and of how many other items exist.
    """
    payload = f"{seed}|{item_key}|{rater_id}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def assign_two_raters(item_keys: Iterable[str], rater_pool: Sequence[str],
                      seed: int) -> AssignmentPlan:
    """Allocate exactly two distinct raters to every item, load-balanced and reproducibly.

    Greedy minimum-load selection: for each item (processed in sorted key order) the two
    raters with the smallest current load are chosen, ties broken by :func:`_tiebreak`. Always
    taking the two globally smallest loads keeps the per-rater totals within one item of each
    other for any pool size, so no rater silently carries the pass.

    Slot 1 / slot 2 is the order of that same comparison, which is what makes the export's
    ``rater_1``/``rater_2`` columns positional and stable rather than tied to a rater's
    identity.

    Args:
        item_keys: Items to allocate; duplicates are collapsed.
        rater_pool: At least two distinct rater ids. Pool order is part of the input, so
            recording it (the store's ``meta``/manifest does) is required to replay a plan.
        seed: Assignment seed. The same seed and pool always produce the identical plan.

    Raises:
        InsufficientRaterPoolError: Fewer than two distinct raters.
    """
    pool = tuple(dict.fromkeys(str(r) for r in rater_pool))
    if len(pool) < RATERS_PER_ITEM:
        raise InsufficientRaterPoolError(
            f"need at least {RATERS_PER_ITEM} distinct raters to keep the rater_1/rater_2 "
            f"contract and a pairwise Cohen's kappa valid; got {len(pool)} "
            f"({', '.join(pool) or 'none'})")

    keys = sorted(dict.fromkeys(str(k) for k in item_keys))
    load = dict.fromkeys(pool, 0)
    chosen: list[tuple[str, tuple[str, ...]]] = []
    for key in keys:
        ranked = sorted(pool, key=lambda rater, k=key: (load[rater], _tiebreak(seed, k, rater)))
        picked = tuple(ranked[:RATERS_PER_ITEM])
        for rater in picked:
            load[rater] += 1
        chosen.append((key, picked))

    positions = _presentation_order(chosen, seed)
    assignments = tuple(
        Assignment(item_key=key, rater_id=rater, slot=slot,
                   position=positions[(rater, key)])
        for key, picked in chosen
        for slot, rater in enumerate(picked, start=1)
    )
    return AssignmentPlan(seed=seed, rater_pool=pool, assignments=assignments)


def _presentation_order(chosen: Sequence[tuple[str, tuple[str, ...]]],
                        seed: int) -> dict[tuple[str, str], int]:
    """Seeded per-rater shuffle of each rater's queue, as ``(rater, item) -> position``.

    Seeded from ``seed`` and the rater id, so the order is recorded implicitly by the seed and
    reproducible, while differing between the two raters of any item.
    """
    per_rater: dict[str, list[str]] = {}
    for key, picked in chosen:
        for rater in picked:
            per_rater.setdefault(rater, []).append(key)
    positions: dict[tuple[str, str], int] = {}
    for rater, keys in per_rater.items():
        ordered = sorted(keys)
        random.Random(f"{seed}|{rater}").shuffle(ordered)
        for index, key in enumerate(ordered):
            positions[(rater, key)] = index
    return positions
