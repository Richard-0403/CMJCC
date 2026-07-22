"""Load the tagged evaluation scenario set."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str
    difficulty: str
    memory_dependency: str
    context_dependency: str
    no_match_expected: bool
    clarification_expected: bool
    acceptable_slots: list[str] = field(default_factory=list)
    expected_response: str = "recommendation"
    turns: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) >= 2


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    """Load scenarios keyed by scenario_id."""
    out: dict[str, Scenario] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["scenario_id"]] = Scenario(
                scenario_id=d["scenario_id"],
                scenario_type=d.get("scenario_type", "unknown"),
                difficulty=d.get("difficulty", "medium"),
                memory_dependency=d.get("memory_dependency", "none"),
                context_dependency=d.get("context_dependency", "low"),
                no_match_expected=bool(d.get("no_match_expected", False)),
                clarification_expected=bool(d.get("clarification_expected", False)),
                acceptable_slots=d.get("acceptable_slots", []),
                expected_response=d.get("expects", {}).get("response_type", "recommendation"),
                turns=d.get("turns", []),
                notes=d.get("notes", ""),
            )
    return out
