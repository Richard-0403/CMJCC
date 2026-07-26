"""Variant-isolation unit tests for ablation attribution (R32.1/2/7)."""

from __future__ import annotations

from jobrec.config import load_config
from jobrec.domain.enums import ExperimentVariant
from jobrec.orchestration.feature_flags import (
    CONTEXT_FLAGS,
    MEMORY_FLAGS,
    FeatureFlags,
    flag_diff,
)


def _flags(variant: str) -> FeatureFlags:
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    cfg.experiment.variant = ExperimentVariant(variant)
    return FeatureFlags.from_config(cfg)


def test_flag_diff_full_vs_no_memory_isolates_memory_flags():
    diff = flag_diff(_flags("full"), _flags("no_memory"))
    assert diff, "full and no_memory must differ on at least one flag"
    assert diff <= MEMORY_FLAGS


def test_flag_diff_full_vs_no_context_isolates_context_flags():
    diff = flag_diff(_flags("full"), _flags("no_context"))
    assert diff, "full and no_context must differ on at least one flag"
    assert diff <= CONTEXT_FLAGS


def test_one_shot_vs_no_memory_differ_only_by_multi_turn_continuation():
    diff = flag_diff(_flags("one_shot"), _flags("no_memory"))
    assert diff == {"use_multi_turn_continuation"}


def test_flag_diff_excludes_variant_field():
    # Same behaviour flags but different variant label -> no diff.
    full = _flags("full")
    relabelled = FeatureFlags(
        variant=ExperimentVariant.NO_CONTEXT,
        use_profile=full.use_profile,
        use_current_turn=full.use_current_turn,
        use_multi_turn_continuation=full.use_multi_turn_continuation,
        use_prior_dialogue=full.use_prior_dialogue,
        use_persistent_memory=full.use_persistent_memory,
        persist_confirmed_updates=full.persist_confirmed_updates,
        explicit_constraint_orchestration=full.explicit_constraint_orchestration,
    )
    assert flag_diff(full, relabelled) == set()
