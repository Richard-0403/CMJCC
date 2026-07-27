"""Deterministic-replay property test (R18.2).

A saved run bundle is only provably reproducible if re-executing it from its own
recorded inputs lands on exactly the same intermediate decisions. Property 21
asserts that over the whole input space of saved runs: the full golden scenario set
(single-turn and multi-turn; recommendation, no-match and clarification outcomes)
crossed with all five experiment variants.

The experiment is produced ONCE in a session-scoped fixture by the real
``ExperimentRunner`` against the real catalog; the property then samples from the
resulting bundles and replays them, because replaying a bundle is far cheaper than
producing one. No mocks: every hash is computed from real artifacts and a real
re-execution in ``RunMode.REPLAY``.

Example-based coverage of the same module (diff report, tampering, unreplayable
bundles) lives in ``tests/eval/test_replay_check.py`` and is not duplicated here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner, load_scenarios
from jobrec.evaluation.replay_check import (
    KEY_STATES,
    key_state_hashes,
    key_state_views,
    recorded_key_states,
    replay_run,
)

CATALOG_PATH = "data/processed/jobs.jsonl"
SCENARIOS_PATH = "data/scenarios/scenarios.jsonl"

#: Every experiment variant, so replay is exercised over all feature-flag paths.
_VARIANTS = ("full", "profile_only", "one_shot", "no_memory", "no_context")

#: Key-state hashes of an EMPTY artifact set, used as a non-vacuity guard: a
#: bundle whose hashes equal these carries no content to reproduce.
_EMPTY_HASHES = key_state_hashes(key_state_views(
    extracted_preferences=None,
    run_record=None,
    eligibility_results=None,
    decision=None,
    claims=None,
))


@dataclass(frozen=True)
class _Bundle:
    """One saved run bundle plus the input dimensions it represents."""

    path: Path
    label: str
    variant: str
    scenario_id: str
    response_type: str
    turn_count: int

    def __repr__(self) -> str:  # keeps Hypothesis output readable
        return f"<{self.label} {self.response_type} turns={self.turn_count}>"


@pytest.fixture(scope="session")
def replay_corpus(tmp_path_factory) -> tuple[Path, tuple[_Bundle, ...]]:
    """A real experiment spanning the property's whole input space, produced once.

    Runs the golden scenario set across every variant through the real
    ``ExperimentRunner`` and returns the experiment directory with one descriptor
    per saved run bundle. Session-scoped: producing the runs is the expensive part,
    replaying them is not, so the property samples from this one corpus.
    """
    scenarios = {s["scenario_id"]: s for s in load_scenarios(SCENARIOS_PATH)}
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.experiment.repeat_count = 1
    runner = ExperimentRunner(config, CATALOG_PATH, SCENARIOS_PATH,
                              out_dir=str(tmp_path_factory.mktemp("replay-prop") / "runs"))
    exp_dir = Path(runner.run(list(_VARIANTS))["experiment_dir"])

    with (exp_dir / "runs_index.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    bundles = []
    for row in rows:
        path = Path(row["run_dir"])
        bundles.append(_Bundle(
            path=path,
            label=path.relative_to(exp_dir).as_posix(),
            variant=row["experiment_variant"],
            scenario_id=row["scenario_id"],
            response_type=row["response_type"],
            turn_count=len(scenarios[row["scenario_id"]]["turns"]),
        ))

    # The corpus must genuinely span the input space, or the property is vacuous.
    assert len(bundles) == len(_VARIANTS) * len(scenarios)
    assert {b.variant for b in bundles} == set(_VARIANTS)
    assert {b.response_type for b in bundles} >= {
        "recommendation", "no_match", "clarification"
    }
    assert {b.turn_count for b in bundles} == {1, 2}
    return exp_dir, tuple(bundles)


# Feature: cmjcc-experiment-readiness, Property 21: Deterministic replay reproduces identical
# key-state hashes
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_property_deterministic_replay_reproduces_identical_key_state_hashes(
    replay_corpus, data
) -> None:
    """Any saved deterministic run replays to byte-identical key-state hashes.

    Draws a saved run bundle from a corpus spanning single-turn and multi-turn
    scenarios, all five experiment variants and the recommendation / no-match /
    clarification outcomes, then re-executes that turn in replay mode against its
    own recorded model calls. All five key states --
    extracted slots, state versions, filtered jobs, ranking output and explanation
    claims -- must recompute to the hashes recorded in the bundle: ``identical`` is
    True and the diff is empty. The catalog is alternately discovered from the
    experiment directory and passed explicitly, since neither route may change the
    recomputed decision.

    Two guards keep the property from passing trivially: the fixture asserts the
    corpus really spans those dimensions, and each drawn bundle must carry content
    (its hashes differ from the hashes of an empty artifact set, with the ranking
    and claim states additionally non-empty whenever the run recommended jobs).

    **Validates: Requirements 18.2**
    """
    exp_dir, bundles = replay_corpus
    bundle = data.draw(st.sampled_from(bundles), label="bundle")
    explicit_catalog = data.draw(st.booleans(), label="explicit_catalog")

    result = replay_run(
        bundle.path,
        catalog_path=(exp_dir / "catalog.jsonl") if explicit_catalog else None,
        label=bundle.label,
    )

    assert result.status == "ok", result.error
    assert result.differences == ()
    assert result.identical
    assert result.variant == bundle.variant
    assert result.scenario_id == bundle.scenario_id

    # Every key state is present, hashed, and reproduced exactly.
    assert set(result.original) == set(result.recomputed) == set(KEY_STATES)
    assert result.recomputed == result.original
    assert result.original == recorded_key_states(bundle.path)
    assert all(len(digest) == 64 for digest in result.recomputed.values())

    # Non-vacuity: there was real recorded content behind those hashes.
    assert result.recomputed["state_versions"] != _EMPTY_HASHES["state_versions"]
    assert result.recomputed["extracted_slots"] != _EMPTY_HASHES["extracted_slots"]
    if bundle.response_type == "recommendation":
        for key in ("filtered_jobs", "ranking_output", "explanation_claims"):
            assert result.recomputed[key] != _EMPTY_HASHES[key]
