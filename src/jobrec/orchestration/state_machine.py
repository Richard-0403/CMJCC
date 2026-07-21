"""Conversation workflow state machine.

Encodes the allowed transitions from landing-plan section 9.1 and records the
ordered list of visited states for the RunRecord. Any state may transition to
FAILED; a failure may then recover to a previous safe state.
"""

from __future__ import annotations

from ..domain.enums import WorkflowState as S

_ALLOWED: dict[S, set[S]] = {
    S.RECEIVED: {S.UNDERSTANDING, S.FAILED},
    S.UNDERSTANDING: {S.VALIDATING, S.FAILED},
    S.VALIDATING: {S.CLARIFICATION_REQUIRED, S.MEMORY_UPDATED, S.FAILED},
    S.CLARIFICATION_REQUIRED: {S.EXPLAINED, S.COMPLETED, S.RECEIVED, S.FAILED},
    S.MEMORY_UPDATED: {S.CONTEXT_BUILT, S.FAILED},
    S.CONTEXT_BUILT: {S.RETRIEVED, S.FAILED},
    S.RETRIEVED: {S.FILTERED, S.FAILED},
    S.FILTERED: {S.RANKED, S.NO_MATCH, S.FAILED},
    S.RANKED: {S.EXPLAINED, S.FAILED},
    S.NO_MATCH: {S.EXPLAINED, S.FAILED},
    S.EXPLAINED: {S.COMPLETED, S.FAILED},
    S.COMPLETED: set(),
    S.FAILED: {S.RECEIVED},
}


class InvalidTransition(Exception):
    """Raised when an illegal workflow transition is attempted."""


class StateMachine:
    """Tracks and validates the workflow state sequence for one run."""

    def __init__(self) -> None:
        self.state: S = S.RECEIVED
        self.history: list[S] = [S.RECEIVED]

    def can(self, target: S) -> bool:
        return target in _ALLOWED.get(self.state, set())

    def to(self, target: S) -> S:
        if not self.can(target):
            raise InvalidTransition(f"{self.state} -> {target} is not allowed")
        self.state = target
        self.history.append(target)
        return target

    def fail(self) -> S:
        self.state = S.FAILED
        self.history.append(S.FAILED)
        return self.state

    def as_str_list(self) -> list[str]:
        return [s.value for s in self.history]
