"""Feature flags that switch behaviour between experiment variants.

A single code path supports the full system, baselines and ablations. Behaviour
is toggled through these flags (derived from the experiment variant and config),
never by forking the implementation.

See landing-plan section 8.4:

| variant       | profile | current turn | prior dialogue | persistent memory | explicit ctx |
|---------------|---------|--------------|----------------|-------------------|--------------|
| full          | yes     | yes          | yes            | yes               | yes          |
| profile_only  | yes     | no           | no             | yes               | yes (basic)  |
| one_shot      | base    | yes          | no             | no                | yes          |
| no_memory     | snapshot| yes          | no             | no                | yes          |
| no_context    | yes     | yes          | yes            | yes               | no           |
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..domain.enums import ExperimentVariant


@dataclass(frozen=True)
class FeatureFlags:
    """Resolved behaviour switches for one run."""

    variant: ExperimentVariant
    use_profile: bool
    use_current_turn: bool
    use_prior_dialogue: bool
    use_persistent_memory: bool
    persist_confirmed_updates: bool
    explicit_constraint_orchestration: bool

    @classmethod
    def from_config(cls, config: AppConfig) -> FeatureFlags:
        variant = config.experiment.variant
        table = {
            ExperimentVariant.FULL: dict(
                use_profile=True, use_current_turn=True, use_prior_dialogue=True,
                use_persistent_memory=True, persist_confirmed_updates=True,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.PROFILE_ONLY: dict(
                use_profile=True, use_current_turn=False, use_prior_dialogue=False,
                use_persistent_memory=True, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.ONE_SHOT: dict(
                use_profile=True, use_current_turn=True, use_prior_dialogue=False,
                use_persistent_memory=False, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.NO_MEMORY: dict(
                use_profile=True, use_current_turn=True, use_prior_dialogue=False,
                use_persistent_memory=False, persist_confirmed_updates=False,
                explicit_constraint_orchestration=True,
            ),
            ExperimentVariant.NO_CONTEXT: dict(
                use_profile=True, use_current_turn=True, use_prior_dialogue=True,
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
        if not config.context.explicit_constraint_orchestration:
            flags = {**flags, "explicit_constraint_orchestration": False}
        return cls(variant=variant, **flags)
