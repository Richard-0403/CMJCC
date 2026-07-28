"""Assignment: exactly two distinct raters per item, balanced, and reproducible from a seed.

Two raters per item is not a tunable: the export contract is ``rater_1``/``rater_2`` and the
reported statistic is a PAIRWISE Cohen's kappa, so a third rater would need Fleiss' kappa and
a wider CSV that :mod:`jobrec_eval.annotation` does not read. These tests pin that, the
load balance, the reproducibility from ``(items, pool, seed)`` and the per-rater presentation
order that keeps ordering effects from lining up between the two raters.

Rater ids are synthetic (``SYNTHETIC-RATER-*``); no annotation data is involved here at all.
"""

from __future__ import annotations

import pytest

from jobrec_eval.annotation_ui.assignment import (
    RATERS_PER_ITEM,
    InsufficientRaterPoolError,
    assign_two_raters,
)

SEED = 2026


def pool(size: int) -> list[str]:
    return [f"SYNTHETIC-RATER-{index:02d}" for index in range(size)]


def keys(count: int) -> list[str]:
    return [f"rel::SYN-SC-{index:03d}::SYN-job-{index:03d}" for index in range(count)]


@pytest.mark.parametrize("pool_size", [2, 3, 4, 5, 7])
def test_every_item_gets_exactly_two_distinct_raters(pool_size):
    plan = assign_two_raters(keys(37), pool(pool_size), SEED)

    assert len(plan.assignments) == 37 * RATERS_PER_ITEM
    for item_key in plan.item_keys:
        slots = plan.raters_for(item_key)
        assert sorted(slots) == [1, 2]
        assert len(set(slots.values())) == 2, "the same rater was given both slots"


@pytest.mark.parametrize("pool_size", [2, 3, 4, 5, 7])
def test_load_is_balanced_within_one_item(pool_size):
    """Greedy minimum-load selection keeps totals within one, so nobody carries the pass."""
    plan = assign_two_raters(keys(37), pool(pool_size), SEED)
    load = plan.load()

    assert sum(load.values()) == 37 * RATERS_PER_ITEM
    assert set(load) == set(pool(pool_size))
    assert plan.max_load_imbalance <= 1


def test_the_same_seed_and_pool_reproduce_the_identical_plan():
    """Reproducibility is the whole reason the seed is recorded in the store's meta."""
    first = assign_two_raters(keys(23), pool(4), SEED)
    second = assign_two_raters(keys(23), pool(4), SEED)
    assert first.assignments == second.assignments

    # Input order does not matter, only the set: the plan is keyed on sorted item keys.
    shuffled = assign_two_raters(list(reversed(keys(23))), pool(4), SEED)
    assert shuffled.assignments == first.assignments

    other_seed = assign_two_raters(keys(23), pool(4), SEED + 1)
    assert other_seed.assignments != first.assignments


def test_adding_an_item_leaves_the_other_items_pairs_untouched():
    """Tie breaking is a digest of (seed, item, rater), not a running RNG.

    So a late addition to the item set cannot reshuffle the pairs already handed out, which is
    what makes a partially annotated pass extensible.
    """
    base = assign_two_raters(keys(20), pool(2), SEED)
    extended = assign_two_raters(keys(21), pool(2), SEED)

    for item_key in base.item_keys:
        assert base.raters_for(item_key) == extended.raters_for(item_key)


def test_a_pool_smaller_than_two_is_rejected_with_a_clear_error():
    """No silent single-rating fallback: one rater yields no agreement to report."""
    with pytest.raises(InsufficientRaterPoolError, match="at least 2 distinct raters"):
        assign_two_raters(keys(5), ["SYNTHETIC-RATER-00"], SEED)
    with pytest.raises(InsufficientRaterPoolError):
        assign_two_raters(keys(5), [], SEED)
    # Duplicates do not make a pool of two.
    with pytest.raises(InsufficientRaterPoolError):
        assign_two_raters(keys(5), ["SYNTHETIC-RATER-00", "SYNTHETIC-RATER-00"], SEED)


def test_each_rater_gets_a_recorded_seeded_shuffle_order():
    """Two raters walk the same items in different orders, reproducibly."""
    plan = assign_two_raters(keys(30), pool(2), SEED)
    a, b = pool(2)

    order_a = [assignment.item_key for assignment in plan.for_rater(a)]
    order_b = [assignment.item_key for assignment in plan.for_rater(b)]

    assert sorted(order_a) == sorted(order_b) == sorted(keys(30))
    assert order_a != order_b, "both raters would meet the items in the same sequence"
    assert order_a != sorted(order_a), "the order was not shuffled at all"
    # Positions are a dense 0..n-1 ranking per rater, so a UI can page through them.
    assert sorted(assignment.position for assignment in plan.for_rater(a)) == list(range(30))
    assert assign_two_raters(keys(30), pool(2), SEED).for_rater(a) == plan.for_rater(a)


def test_slots_are_positions_not_identities():
    """Slot 1 is not one fixed person: that is what keeps the CSV columns balanced."""
    plan = assign_two_raters(keys(40), pool(3), SEED)
    slot_1_raters = {plan.raters_for(key)[1] for key in plan.item_keys}
    assert len(slot_1_raters) > 1
