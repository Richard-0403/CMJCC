"""Variant-isolation unit tests for ablation attribution (R32.1/2/7)."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.app_service import AppService
from jobrec.config import AppConfig, load_config
from jobrec.domain.enums import ExperimentVariant
from jobrec.orchestration.feature_flags import (
    CONTEXT_FLAGS,
    MEMORY_FLAGS,
    FeatureFlags,
    flag_diff,
)

CATALOG_PATH = "data/processed/jobs.jsonl"


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
    """The pair is isolated to one flag, so any Δ between them attributes to it (R5.3).

    Flag-level only: this says nothing about whether the flag is READ anywhere, and it
    passed while the flag was semantically dead and the two conditions produced identical
    runs. The behavioural counterpart is
    ``tests/e2e/test_clarification_loop.py::test_one_shot_and_no_memory_diverge_behaviourally``
    together with ``test_multi_turn_continuation_flag_is_not_dead``; both guarantees are
    needed and neither replaces the other.
    """
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

# ---------------------------------------------------------------------------
# Property-based tests (Properties 6 and 7)
# ---------------------------------------------------------------------------

#: Behaviour switches only; ``variant`` is a label, not a switch (see ``flag_diff``).
BEHAVIOUR_FLAGS = {f.name for f in dataclass_fields(FeatureFlags) if f.name != "variant"}
VARIANTS = list(ExperimentVariant)
#: Ordered pairs of *distinct* variants - a smart generator domain with no filtering.
DISTINCT_VARIANT_PAIRS = [(a, b) for a in VARIANTS for b in VARIANTS if a != b]


def _resolve(
    variant: ExperimentVariant,
    *,
    use_prior_dialogue: bool = True,
    use_multi_turn_continuation: bool = True,
    random_seed: int = 42,
    top_k: int = 5,
) -> FeatureFlags:
    """Resolve ``FeatureFlags`` for ``variant`` under a mechanism-enabled config.

    ``memory.enabled`` and ``context.explicit_constraint_orchestration`` are held True
    because ``from_config`` may only *restrict* behaviour, never expand it: switching a
    mechanism off globally collapses the variants that differ only in that mechanism, which
    is a configuration error the consistency gate rejects (R15/R32.7) rather than a
    violation of these properties. Non-mechanism knobs (seed, top-k) are varied to show
    they never leak into flag resolution.
    """
    cfg = AppConfig()
    cfg.experiment.variant = variant
    cfg.experiment.random_seed = random_seed
    cfg.experiment.top_k = top_k
    cfg.memory.enabled = True
    cfg.memory.use_prior_dialogue = use_prior_dialogue
    cfg.memory.use_multi_turn_continuation = use_multi_turn_continuation
    cfg.context.explicit_constraint_orchestration = True
    return FeatureFlags.from_config(cfg)


# Feature: cmjcc-experiment-readiness, Property 6: All distinct variants resolve to distinct
# FeatureFlags (one_shot != no_memory)
@settings(max_examples=100)
@given(
    pair=st.sampled_from(DISTINCT_VARIANT_PAIRS),
    use_prior_dialogue=st.booleans(),
    random_seed=st.integers(min_value=0, max_value=65535),
    top_k=st.integers(min_value=1, max_value=20),
)
def test_property_distinct_variants_resolve_to_distinct_feature_flags(
    pair: tuple[ExperimentVariant, ExperimentVariant],
    use_prior_dialogue: bool,
    random_seed: int,
    top_k: int,
) -> None:
    """Distinct variants never resolve to the same behaviour-flag set.

    **Validates: Requirements 5.3, 5.6**
    """
    first, second = pair
    flags_a = _resolve(
        first,
        use_prior_dialogue=use_prior_dialogue,
        random_seed=random_seed,
        top_k=top_k,
    )
    flags_b = _resolve(
        second,
        use_prior_dialogue=use_prior_dialogue,
        random_seed=random_seed,
        top_k=top_k,
    )

    diff = flag_diff(flags_a, flags_b)
    assert diff, f"{first} and {second} resolve to identical behaviour flags"
    assert diff <= BEHAVIOUR_FLAGS
    # The variant label is never the only thing that distinguishes two conditions.
    assert {f: getattr(flags_a, f) for f in BEHAVIOUR_FLAGS} != {
        f: getattr(flags_b, f) for f in BEHAVIOUR_FLAGS
    }
    if {first, second} == {ExperimentVariant.ONE_SHOT, ExperimentVariant.NO_MEMORY}:
        assert diff == {"use_multi_turn_continuation"}


# Feature: cmjcc-experiment-readiness, Property 7: Ablation pairs differ only in their
# target-mechanism flags
@settings(max_examples=100)
@given(
    use_prior_dialogue=st.booleans(),
    use_multi_turn_continuation=st.booleans(),
    random_seed=st.integers(min_value=0, max_value=65535),
    top_k=st.integers(min_value=1, max_value=20),
)
def test_property_ablation_pairs_differ_only_in_target_mechanism_flags(
    use_prior_dialogue: bool,
    use_multi_turn_continuation: bool,
    random_seed: int,
    top_k: int,
) -> None:
    """Each ablation pair isolates exactly one mechanism, so Δ is attributable.

    **Validates: Requirements 32.1, 32.2, 32.7**
    """
    kwargs = dict(
        use_prior_dialogue=use_prior_dialogue,
        use_multi_turn_continuation=use_multi_turn_continuation,
        random_seed=random_seed,
        top_k=top_k,
    )
    full = _resolve(ExperimentVariant.FULL, **kwargs)
    no_memory = _resolve(ExperimentVariant.NO_MEMORY, **kwargs)
    no_context = _resolve(ExperimentVariant.NO_CONTEXT, **kwargs)

    memory_diff = flag_diff(full, no_memory)
    context_diff = flag_diff(full, no_context)

    assert memory_diff, "full and no_memory must differ on at least one memory flag"
    assert memory_diff <= MEMORY_FLAGS
    assert not memory_diff & (BEHAVIOUR_FLAGS - MEMORY_FLAGS)

    assert context_diff, "full and no_context must differ on at least one context flag"
    assert context_diff <= CONTEXT_FLAGS
    assert not context_diff & (BEHAVIOUR_FLAGS - CONTEXT_FLAGS)

    # The two ablations target disjoint mechanisms.
    assert not memory_diff & context_diff


# ---------------------------------------------------------------------------
# R5.6 guard: the shipped one_shot / no_memory configs must never resolve identically
# ---------------------------------------------------------------------------


def test_shipped_one_shot_and_no_memory_configs_resolve_to_different_flags():
    """Fails if the resolved `one_shot` and `no_memory` flag sets become identical (R5.6).

    This resolves the *shipped* per-variant config files rather than swapping a label on a
    single config, so it also catches a config file that silently collapses the two
    conditions.
    """
    one_shot = FeatureFlags.from_config(
        load_config("configs/experiment_one_shot.yaml", base_dir="configs")
    )
    no_memory = FeatureFlags.from_config(
        load_config("configs/experiment_no_memory.yaml", base_dir="configs")
    )

    assert one_shot.variant is ExperimentVariant.ONE_SHOT
    assert no_memory.variant is ExperimentVariant.NO_MEMORY

    one_shot_behaviour = {f: getattr(one_shot, f) for f in BEHAVIOUR_FLAGS}
    no_memory_behaviour = {f: getattr(no_memory, f) for f in BEHAVIOUR_FLAGS}
    assert one_shot_behaviour != no_memory_behaviour, (
        "one_shot and no_memory resolved to identical FeatureFlags: the two experimental "
        "conditions are no longer distinguishable by configuration alone (R5.3/R5.6)"
    )

    diff = flag_diff(one_shot, no_memory)
    assert diff == {"use_multi_turn_continuation"}
    # one_shot is the genuine single-turn condition.
    assert one_shot.use_multi_turn_continuation is False
    assert no_memory.use_multi_turn_continuation is True

# ---------------------------------------------------------------------------
# Behavioural isolation (R24.1): distinct flags must produce distinct BEHAVIOUR
# ---------------------------------------------------------------------------
#
# Everything above checks flag *resolution* and attribution. R24.1 also requires each
# variant to behave differently, which only the running system can show. These tests
# drive the SAME profile and the SAME utterances through ``AppService`` (deterministic
# mode, real catalog, real agents) under each variant and assert the observable
# difference each mechanism implies:
#
#   profile_only  -> the current turn is ignored (use_current_turn)
#   no_context    -> no explicit constraint orchestration (no hard filter, no bundle)
#   no_memory     -> prior dialogue and long-term write-back are both off
#   one_shot      -> as no_memory, plus the multi-turn continuation gate
#
# The per-turn dialogue/turn-count view of this is covered end-to-end in
# tests/e2e/test_clarification_loop.py; here the assertions are per mechanism on a
# single turn's ActiveSearchState / JobContextState / CandidateState.

#: One utterance that states a location, a work mode and a hard salary threshold, none of
#: which are in the profile below - so "was the current turn used?" is directly visible.
_TURN = "I want a remote role in Penang paying at least RM9000."

#: Long-term profile facts that contradict (location) or extend (work mode) the utterance.
_PROFILE_FIELDS = {
    "skills": ["Python", "SQL"],
    "years_experience": 2,
    "target_roles": ["data analyst"],
    "preferred_locations": ["Kuala Lumpur"],
    "work_modes": ["hybrid"],
}

_VARIANT_NAMES = ("full", "profile_only", "one_shot", "no_memory", "no_context")
_ABLATIONS = ("profile_only", "one_shot", "no_memory", "no_context")


def _profile(tag: str, variant: str, **fields) -> dict:
    """A profile dict with a candidate id unique per (test, variant).

    Each variant gets its own candidate so that a variant which writes back to long-term
    memory cannot change the profile a later variant reads.
    """
    return {"candidate_id": f"{tag}-{variant}", **(fields or _PROFILE_FIELDS)}


def _drive(service, variant: str, profile: dict, utterances: list[str]) -> list:
    """Run ``utterances`` through one session of ``variant`` and return every TurnResult."""
    service.create_candidate(profile)
    session_id = service.create_session(profile["candidate_id"], variant)
    return [service.process_turn(session_id, text) for text in utterances]


def _long_term(state, field: str) -> list:
    """Active (non-retired) long-term values held for ``field`` on a CandidateState."""
    return [pv.value for pv in getattr(state, field) if pv.is_active]


def _signature(result) -> tuple:
    """Observable behaviour of one turn, reduced to a comparable tuple."""
    active = result.active_search_state
    return (
        tuple(active.preferred_locations),
        tuple(active.work_modes),
        active.salary_min,
        tuple(active.hard_constraint_fields),
        result.job_context_state is not None,
        tuple(sorted({c.field_name for c in result.dialogue_state.conflicts})),
        result.candidate_state.version,
    )


def test_each_variant_behaves_differently_on_the_same_turn(service):
    """The same profile + utterance produces variant-specific observable behaviour.

    Every ablation differs from ``full`` on the single shared code path, so a measured
    delta is attributable to the ablated mechanism rather than to input variation
    (R24.1). ``one_shot`` and ``no_memory`` coincide *within one turn* - they differ only
    by the multi-turn continuation gate, which needs a second turn to observe and is
    covered by ``test_multi_turn_continuation_gate_changes_behaviour``.
    """
    signatures = {
        variant: _signature(
            _drive(service, variant, _profile("sig", variant), [_TURN])[0]
        )
        for variant in _VARIANT_NAMES
    }

    for variant in _ABLATIONS:
        assert signatures[variant] != signatures["full"], (
            f"{variant} behaves identically to full on the same input: the ablation is "
            "not observable, so no measured delta could be attributed to it"
        )
    # full, profile_only, no_context and the memory-less pair are four distinct behaviours.
    assert len(set(signatures.values())) >= 4, signatures


def test_profile_only_ignores_the_current_turn(service):
    """``profile_only`` searches on long-term profile alone; ``full`` uses the utterance.

    Isolates ``use_current_turn``: the location, work mode and salary stated in the turn
    reach the active search under ``full`` and are absent under ``profile_only``, whose
    search still carries the profile values (R24.1).
    """
    full = _drive(service, "full", _profile("cur", "full"), [_TURN])[0].active_search_state
    only = _drive(
        service, "profile_only", _profile("cur", "profile_only"), [_TURN]
    )[0].active_search_state

    # full: the current turn overrides the profile location and adds the stated values.
    assert full.preferred_locations == ["Penang"]
    assert "remote" in full.work_modes
    assert full.salary_min == 9000.0

    # profile_only: none of the turn's constraints are in scope ...
    assert only.preferred_locations == ["Kuala Lumpur"]
    assert "remote" not in only.work_modes
    assert only.salary_min is None
    # ... while the long-term profile still drives the search.
    assert only.work_modes == ["hybrid"]
    assert only.target_roles == ["data analyst"]


def test_no_context_skips_explicit_constraint_orchestration(service):
    """``no_context`` builds no constraint bundle and applies no hard filter.

    Isolates ``explicit_constraint_orchestration``: the merged search view is the same as
    ``full``'s (same locations, work modes, salary), but nothing is classified as a hard
    constraint, no ``JobContextState`` is built, and eligibility is a pass-through with no
    recorded checks (R24.1).
    """
    full = _drive(service, "full", _profile("ctx", "full"), [_TURN])[0]
    none = _drive(service, "no_context", _profile("ctx", "no_context"), [_TURN])[0]

    # The candidate-side merge is untouched: only the context mechanism is ablated.
    assert none.active_search_state.preferred_locations == full.active_search_state.preferred_locations
    assert none.active_search_state.work_modes == full.active_search_state.work_modes
    assert none.active_search_state.salary_min == full.active_search_state.salary_min

    # full: hard constraints are declared, a bundle is built and filtering records checks.
    assert full.job_context_state is not None
    assert full.active_search_state.hard_constraint_fields
    assert full.decision is not None and full.decision.context_id is not None
    assert any(e.checks for e in full.decision.eligibility_results)

    # no_context: no bundle, no hard constraints, and every recalled job passes unchecked.
    assert none.job_context_state is None
    assert none.active_search_state.hard_constraint_fields == []
    assert none.decision is not None and none.decision.context_id is None
    assert none.decision.eligibility_results
    assert all(e.eligible and e.checks == [] for e in none.decision.eligibility_results)
    # The stated constraints are still present, as soft relevance features.
    assert "salary_min" in none.active_search_state.soft_preference_fields


def test_memoryless_variants_ignore_prior_dialogue(service):
    """A constraint stated in turn 1 survives into turn 2 only where memory is on.

    Isolates the prior-dialogue fold: the role is stated once and never restated, so on
    turn 2 ``full`` / ``no_context`` still search for it while ``no_memory`` / ``one_shot``
    have lost it and have to ask for it again. ``profile_only`` never picked it up at all
    (R24.1).
    """
    turns = ["I want a data analyst role.", "In Kuala Lumpur, hybrid please."]
    # A profile with no target role, so the role can only come from the dialogue.
    fields = {"skills": ["Python", "SQL"], "years_experience": 2}
    second = {
        variant: _drive(service, variant, _profile("prior", variant, **fields), turns)[1]
        for variant in _VARIANT_NAMES
    }

    for variant in ("full", "no_context"):
        assert second[variant].active_search_state.target_roles == ["data analyst"], variant
        assert second[variant].response.response_type == "recommendation", variant

    for variant in ("no_memory", "one_shot"):
        assert second[variant].active_search_state.target_roles == [], variant
        # Without the role the search is under-specified, so the system asks instead.
        assert second[variant].response.response_type == "clarification", variant
        # The current turn is still in scope - only the earlier turn was dropped.
        assert second[variant].active_search_state.preferred_locations == ["Kuala Lumpur"], variant

    # profile_only ignores both turns.
    assert second["profile_only"].active_search_state.target_roles == []
    assert second["profile_only"].active_search_state.preferred_locations == []


def test_memoryless_variants_ignore_persistent_memory(service):
    """A durable statement updates long-term memory only where persistence is on.

    Isolates ``use_persistent_memory`` / ``persist_confirmed_updates``: "from now on"
    writes a new ``CandidateState`` version under ``full`` / ``no_context`` and writes
    nothing under ``no_memory`` / ``one_shot`` / ``profile_only``. For the memory-less
    pair the value still shapes the CURRENT search, which separates *persisting* the turn
    from *using* it (R24.1).
    """
    utterance = "From now on I only want hybrid roles."
    fields = {"skills": ["Python", "SQL"], "years_experience": 2}
    results = {
        variant: _drive(service, variant, _profile("mem", variant, **fields), [utterance])[0]
        for variant in _VARIANT_NAMES
    }

    for variant in ("full", "no_context"):
        state = results[variant].candidate_state
        assert state.version == 2, variant
        assert _long_term(state, "work_modes") == ["hybrid"], variant

    for variant in ("no_memory", "one_shot"):
        state = results[variant].candidate_state
        assert state.version == 1, variant
        assert _long_term(state, "work_modes") == [], variant
        # Not persisted, but not ignored either: the current search still uses it.
        assert results[variant].active_search_state.work_modes == ["hybrid"], variant

    # profile_only neither persists nor uses the turn.
    assert results["profile_only"].candidate_state.version == 1
    assert _long_term(results["profile_only"].candidate_state, "work_modes") == []
    assert results["profile_only"].active_search_state.work_modes == []


def test_multi_turn_continuation_gate_changes_behaviour(config):
    """The flag that separates ``one_shot`` from ``no_memory`` is behaviourally real.

    ``one_shot`` and ``no_memory`` resolve to different ``FeatureFlags`` on exactly
    ``use_multi_turn_continuation`` (see the R5.6 guard above). This drives the same
    two-turn dialogue through the same variant with only that gate flipped: with the gate
    on, the role stated in turn 1 still drives turn 2's search; with it off it does not.
    So the distinguishing flag is a live switch on the shared code path, not a label
    (R24.1).
    """
    turns = ["I want a data analyst role.", "In Kuala Lumpur, hybrid please."]
    roles: dict[bool, list[str]] = {}
    for continuation in (True, False):
        cfg = config.model_copy(deep=True)
        cfg.memory.use_multi_turn_continuation = continuation
        service = AppService(cfg, CATALOG_PATH)
        profile = {"candidate_id": f"cont-{continuation}", "skills": ["Python"],
                   "years_experience": 2}
        second = _drive(service, "full", profile, turns)[1]
        roles[continuation] = list(second.active_search_state.target_roles)

    assert roles[True] == ["data analyst"]
    assert roles[False] == []
