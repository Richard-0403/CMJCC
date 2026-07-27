"""Feature flags that switch behaviour between experiment variants.

A single code path supports the full system, baselines and ablations. Behaviour
is toggled through these flags (derived from the experiment variant and config),
never by forking the implementation.

See landing-plan section 8.4. The resolved matrix (``from_config`` below is
authoritative):

| variant       | profile | current turn | multi-turn | prior dialogue | persistent memory | explicit ctx |
|---------------|---------|--------------|------------|----------------|-------------------|--------------|
| full          | yes     | yes          | yes        | yes            | yes               | yes          |
| profile_only  | yes     | no           | yes        | no             | yes               | yes (basic)  |
| one_shot      | yes     | yes          | no         | no             | no                | yes          |
| no_memory     | yes     | yes          | yes        | no             | no                | yes          |
| no_context    | yes     | yes          | yes        | yes            | yes               | no           |

``one_shot`` and ``no_memory`` are distinct conditions: they differ on exactly
``use_multi_turn_continuation``, so ``one_shot`` is a genuine single-turn condition while
``no_memory`` keeps the multi-turn workflow without memory (R5.3/R5.6).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..config import AppConfig
from ..domain.enums import ExperimentVariant

#: Behaviour flags whose difference attributes an effect to the *memory* mechanism.
MEMORY_FLAGS = {
    "use_prior_dialogue",
    "use_persistent_memory",
    "persist_confirmed_updates",
    "use_multi_turn_continuation",
}
#: Behaviour flags whose difference attributes an effect to the *context* mechanism.
CONTEXT_FLAGS = {"explicit_constraint_orchestration"}


@dataclass(frozen=True)
class FeatureFlags:
    """Resolved behaviour switches for one run."""

    variant: ExperimentVariant
    use_profile: bool
    use_current_turn: bool
    use_multi_turn_continuation: bool
    use_prior_dialogue: bool
    use_persistent_memory: bool
    persist_confirmed_updates: bool
    explicit_constraint_orchestration: bool

    @classmethod
    def from_config(cls, config: AppConfig) -> FeatureFlags:
        variant = config.experiment.variant
        table = {
            ExperimentVariant.FULL: dict(
                use_profile=True, use_current_turn=True, use_multi_turn_continuation=True,
                use_prior_dialogue=True,
                use_persistent_memory=True, persist_confirmed_updates=True,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.PROFILE_ONLY: dict(
                use_profile=True, use_current_turn=False, use_multi_turn_continuation=True,
                use_prior_dialogue=False,
                use_persistent_memory=True, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.ONE_SHOT: dict(
                use_profile=True, use_current_turn=True, use_multi_turn_continuation=False,
                use_prior_dialogue=False,
                use_persistent_memory=False, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.NO_MEMORY: dict(
                use_profile=True, use_current_turn=True, use_multi_turn_continuation=True,
                use_prior_dialogue=False,
                use_persistent_memory=False, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.NO_CONTEXT: dict(
                use_profile=True, use_current_turn=True, use_multi_turn_continuation=True,
                use_prior_dialogue=True,
                use_persistent_memory=True, persist_confirmed_updates=True,
                explicit_constraint_orchestration=False,
            ),
        }
        flags = table[variant]
        # Config can further restrict (but not expand) memory behaviour.
        if not config.memory.enabled:
            flags = {**flags, "use_prior_dialogue": False, "use_persistent_memory": False,
                     "persist_confirmed_updates": False}
        if not config.memory.use_prior_dialogue:
            flags = {**flags, "use_prior_dialogue": False}
        if not config.memory.use_multi_turn_continuation:
            flags = {**flags, "use_multi_turn_continuation": False}
        if not config.context.explicit_constraint_orchestration:
            flags = {**flags, "explicit_constraint_orchestration": False}
        return cls(variant=variant, **flags)



def flag_diff(a: FeatureFlags, b: FeatureFlags) -> set[str]:
    """Return the set of behaviour-flag field names that differ between ``a`` and ``b``.

    The ``variant`` field is a label rather than a behaviour switch and is therefore
    excluded. Used for ablation attribution (R32): ``flag_diff(full, no_memory)`` must be a
    non-empty subset of :data:`MEMORY_FLAGS`, and ``flag_diff(full, no_context)`` a non-empty
    subset of :data:`CONTEXT_FLAGS`.
    """
    return {
        f.name
        for f in fields(FeatureFlags)
        if f.name != "variant" and getattr(a, f.name) != getattr(b, f.name)
    }
