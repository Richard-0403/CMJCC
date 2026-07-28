"""End-to-end property test for the clarification dialogue loop (R7.6).

Drives `ExperimentRunner._run_one` over the real deterministic pipeline (real catalog
records, real agents, no LLM) for clarification-dependent scenarios and checks the
max-turn guard: whatever the scenario, the simulated user's behaviour, or the configured
`max_dialogue_turns`, the loop halts and records why.

The simulated-user behaviours are real :class:`SimulatedUser` subclasses. Two of them are
adversarial (they answer every clarification with a vague utterance that never supplies
the missing constraint), which is the only way to force the guard rather than a natural
terminal outcome. Each subclass also carries a call tripwire: the loop may ask the user at
most `max_dialogue_turns` times, so an unbounded loop fails the test instead of hanging.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.config import load_config
from jobrec.domain.enums import ExperimentVariant
from jobrec.evaluation.experiment_runner import (
    TERMINATION_CONTINUATION_DISABLED,
    ExperimentRunner,
)
from jobrec.orchestration.feature_flags import FeatureFlags, flag_diff
from jobrec_eval import simulated_user as simulated_user_module
from jobrec_eval.simulated_user import SimulatedUser

CATALOG_PATH = "data/processed/jobs.jsonl"

#: Every termination reason a run can record (`_terminal_reason` outcomes, the three
#: in-loop guard exits, and the continuation gate that keeps a single-turn variant out of
#: the loop entirely). A run must always end on one of these.
_TERMINAL_REASONS = frozenset({
    "recommendation",
    "recommendation_empty",
    "no_match",
    "error",
    "no_result",
    "max_turns",
    "cannot_answer",
    "repeated_slot",
    TERMINATION_CONTINUATION_DISABLED,
})


@pytest.fixture(scope="session")
def tiny_catalog(tmp_path_factory) -> str:
    """A 6-job slice of the real catalog: enough to recommend, small enough to be fast."""
    jobs = [
        json.loads(line)
        for line in Path(CATALOG_PATH).read_text().splitlines()
        if line.strip()
    ]
    tiny = [job for job in jobs if job["city"] == "Kuala Lumpur"][:6]
    assert len(tiny) == 6
    path = tmp_path_factory.mktemp("catalog") / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(job) for job in tiny) + "\n")
    return str(path)


@pytest.fixture(scope="session")
def loop_env(tmp_path_factory, tiny_catalog) -> dict:
    """Session-scoped runner inputs: tiny catalog, one-scenario file, output dir."""
    root = tmp_path_factory.mktemp("clarification_loop")
    scenarios_path = root / "scenarios.jsonl"
    scenarios_path.write_text(json.dumps(_scenario(["target_roles"], _OPENINGS[0])) + "\n")
    return {
        "catalog_path": tiny_catalog,
        "scenarios_path": str(scenarios_path),
        "out_dir": root / "out",
        "config": load_config("configs/experiment_full.yaml", base_dir="configs"),
    }


#: Scripted opening turns: one under-specified (no role -> clarification), one wholly
#: vague, and one complete request (terminates before the loop needs a turn).
_OPENINGS = [
    ["I am looking for something in Kuala Lumpur, hybrid, around RM5000."],
    ["Any openings for me?"],
    ["I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."],
]

#: What the scenario reference declares answerable: nothing (a user who cannot answer),
#: just the role, or several slots.
_ACCEPTABLE_SLOTS = [
    [],
    ["target_roles"],
    ["target_roles", "preferred_locations", "work_modes"],
]

#: Simulated-user behaviours: the scenario-driven real answerer, plus two adversarial
#: always-vague users (one burning a fresh slot per turn, one re-answering one slot).
_BEHAVIOURS = ["scenario", "vague_new_slot", "vague_same_slot"]

_VAGUE = "Hmm, I am not really sure about that."


def _scenario(acceptable_slots: list[str], turns: list[str]) -> dict:
    """A clarification-dependent scenario in the runner's expected shape."""
    return {
        "scenario_id": "SC-P10",
        "scenario_type": "clarification",
        "profile": {
            "candidate_id": "SC-P10-cand",
            "skills": ["Python"],
            "years_experience": 2,
        },
        "turns": list(turns),
        "clarification_expected": True,
        "acceptable_slots": list(acceptable_slots),
        "expects": {"response_type": "clarification"},
    }


def _user_class(behaviour: str, budget: int) -> type[SimulatedUser]:
    """A real ``SimulatedUser`` subclass implementing ``behaviour`` with a tripwire.

    The loop only asks the user while ``extra_turns < max_dialogue_turns``, so it can
    call ``answer`` at most ``budget`` times. Exceeding that means the guard failed, so
    the tripwire raises instead of letting the test spin forever.
    """

    class _User(SimulatedUser):
        def __init__(self, scenario) -> None:
            super().__init__(scenario)
            self.calls = 0

        def answer(self, clarification, asked_slots=None):
            self.calls += 1
            assert self.calls <= budget, (
                f"clarification loop asked the user {self.calls} times "
                f"with max_dialogue_turns={budget}"
            )
            if behaviour == "scenario":
                return super().answer(clarification, asked_slots)
            if behaviour == "vague_new_slot":
                # Always answerable, never informative, and never a repeated slot: only
                # the max-turn guard can stop this user.
                return _VAGUE, f"slot_{self.calls}"
            # Always re-answers the same slot -> the repeated-slot guard should fire.
            return _VAGUE, "target_roles"

    return _User


# Feature: cmjcc-experiment-readiness, Property 10: The clarification loop always
# terminates within max_turns
@settings(max_examples=100, deadline=None)
@given(
    turns=st.sampled_from(_OPENINGS),
    acceptable_slots=st.sampled_from(_ACCEPTABLE_SLOTS),
    behaviour=st.sampled_from(_BEHAVIOURS),
    max_dialogue_turns=st.integers(min_value=1, max_value=3),
    variant=st.sampled_from(["full", "no_memory"]),
)
def test_property_clarification_loop_terminates_within_max_turns(
    loop_env, turns, acceptable_slots, behaviour, max_dialogue_turns, variant
) -> None:
    """The loop halts within ``max_dialogue_turns`` answer turns and records the reason.

    Holds for a cooperative user, a user who cannot answer, and an adversarial user who
    keeps giving unhelpful answers. The bound tracks the configured value: when the guard
    fires, the loop used exactly ``max_dialogue_turns`` turns, no more.

    **Validates: Requirements 7.6**
    """
    config = loop_env["config"].model_copy(deep=True)
    config.experiment.max_dialogue_turns = max_dialogue_turns
    runner = ExperimentRunner(
        config,
        loop_env["catalog_path"],
        loop_env["scenarios_path"],
        out_dir=str(loop_env["out_dir"]),
    )
    scenario = _scenario(acceptable_slots, turns)
    exp_dir = loop_env["out_dir"] / "exp"

    with mock.patch.object(
        simulated_user_module,
        "SimulatedUser",
        _user_class(behaviour, max_dialogue_turns),
    ):
        row, _failure = runner._run_one(variant, scenario, 0, exp_dir)

    # The run terminated with a recorded, known termination reason ...
    reason = row["termination_reason"]
    assert reason in _TERMINAL_REASONS
    # ... and produced a final system response.
    assert str(row["response_type"])

    # Simulated answer turns are what the guard bounds; scripted turns are not.
    loop_turns = row["response_turns"] - len(turns)
    assert 0 <= loop_turns <= max_dialogue_turns
    assert len(turns) <= row["response_turns"] <= len(turns) + max_dialogue_turns

    if reason == "max_turns":
        # The cap is the configured value, not a constant: the loop ran exactly
        # max_dialogue_turns answer turns and stopped while still clarifying.
        assert loop_turns == max_dialogue_turns
        assert str(row["response_type"]) == "clarification"
    elif reason in {"cannot_answer", "repeated_slot"}:
        # The other guards are only reachable before the budget is spent.
        assert loop_turns < max_dialogue_turns
        assert str(row["response_type"]) == "clarification"
    elif reason == "recommendation":
        assert str(row["response_type"]) == "recommendation"
        assert row["returned"] > 0
    elif reason == "no_match":
        assert str(row["response_type"]) == "no_match"

    # The trace records one row per turn, and only the final row carries the reason.
    trace = [
        json.loads(line)
        for line in (Path(row["run_dir"]) / "dialogue_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(trace) == row["response_turns"]
    assert trace[-1]["termination_reason"] == reason
    assert all(record["termination_reason"] is None for record in trace[:-1])

#: The slot the ``vague_same_slot`` behaviour always answers, so the runner sees the
#: same slot come back a second time and the guard has something to catch.
_REPEATED_SLOT = "target_roles"


# Feature: cmjcc-experiment-readiness, Property 11: Repeated-slot re-asking is guarded
# and recorded
@settings(max_examples=100, deadline=None)
@given(
    turns=st.sampled_from(_OPENINGS[:2]),
    acceptable_slots=st.sampled_from(_ACCEPTABLE_SLOTS),
    max_dialogue_turns=st.integers(min_value=2, max_value=4),
    variant=st.sampled_from(["full", "no_memory"]),
)
def test_property_repeated_slot_is_guarded_and_recorded(
    loop_env, turns, acceptable_slots, max_dialogue_turns, variant
) -> None:
    """A slot is never re-answered twice: the guard stops the loop and the trace says so.

    Drives the loop with the adversarial ``vague_same_slot`` user, which answers every
    clarification with the same uninformative slot, so the system keeps asking about that
    slot. Whatever the opening, the reference's acceptable slots, the dialogue cap or the
    variant: the user is asked about the slot at most once more after answering it, the
    run ends on ``repeated_slot`` (not on the max-turn cap), and the per-run
    ``dialogue_trace.jsonl`` records both the repeat and the termination reason.

    **Validates: Requirements 7.7**
    """
    config = loop_env["config"].model_copy(deep=True)
    config.experiment.max_dialogue_turns = max_dialogue_turns
    runner = ExperimentRunner(
        config,
        loop_env["catalog_path"],
        loop_env["scenarios_path"],
        out_dir=str(loop_env["out_dir"]),
    )
    scenario = _scenario(acceptable_slots, turns)
    exp_dir = loop_env["out_dir"] / "exp"

    # Observe every clarification put to the user: the slot the SYSTEM asked about and
    # the slots the runner already considers answered when it asks.
    asks: list[tuple[str | None, list[str]]] = []
    base_user = _user_class("vague_same_slot", max_dialogue_turns)

    class _RecordingUser(base_user):  # type: ignore[valid-type, misc]
        def answer(self, clarification, asked_slots=None):
            fields = list(getattr(clarification, "target_fields", None) or [])
            asks.append((fields[0] if fields else None, sorted(asked_slots or [])))
            return super().answer(clarification, asked_slots)

    with mock.patch.object(simulated_user_module, "SimulatedUser", _RecordingUser):
        row, _failure = runner._run_one(variant, scenario, 0, exp_dir)

    # The guard fired, and it fired instead of the max-turn cap: only one answer turn
    # was spent even though the cap allowed more.
    assert row["termination_reason"] == "repeated_slot"
    assert str(row["response_type"]) == "clarification"
    loop_turns = row["response_turns"] - len(turns)
    assert loop_turns == 1
    assert loop_turns < max_dialogue_turns

    # The system did re-ask the answered slot -- so this is a real repeat, not a case of
    # the loop simply running out of things to ask ...
    assert [slot for slot, _answered in asks].count(_REPEATED_SLOT) == 2
    # ... the first ask saw nothing answered, the second saw the slot already answered
    # (exactly the condition the guard checks) ...
    assert asks[0][1] == []
    assert _REPEATED_SLOT in asks[1][1]
    # ... and the loop stopped there rather than re-asking without bound.
    assert len(asks) == 2

    # The event is recorded: one row per processed turn (the guarded re-ask adds no
    # phantom row), the repeat is visible in clarification_slot, and only the final row
    # carries the termination reason.
    trace = [
        json.loads(line)
        for line in (Path(row["run_dir"]) / "dialogue_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(trace) == row["response_turns"]
    assert [record["clarification_slot"] for record in trace].count(_REPEATED_SLOT) >= 2
    assert trace[-1]["clarification_slot"] == _REPEATED_SLOT
    assert str(trace[-1]["system_action"]) == str(row["response_type"])
    assert trace[-1]["termination_reason"] == "repeated_slot"
    assert all(record["termination_reason"] is None for record in trace[:-1])


#: A clarification-dependent scenario whose role is stated ONCE, in the first turn, and
#: never restated. On the second turn only a variant that carries prior dialogue forward
#: still knows the role; the others have to ask for it again, which is what makes
#: ``response_turns`` differ across variants.
_E2E_TURNS = [
    "I want a data analyst role.",
    "In Kuala Lumpur, hybrid please.",
]

#: Slots the scenario reference declares answerable, so the simulated user can supply the
#: role when a variant re-asks for it.
_E2E_SLOTS = ["target_roles", "preferred_locations", "work_modes"]

#: Variants whose flags thread prior dialogue into the current turn (memory mechanism on).
_MEMORY_VARIANTS = ("full", "no_context")
#: Variants that do not carry prior dialogue (memory off, or the current turn ignored
#: entirely under profile_only) but MAY continue the dialogue, so the role is asked again
#: and answered in one extra turn.
_MEMORYLESS_VARIANTS = ("no_memory", "profile_only")
#: Variants that may not continue the dialogue at all (``use_multi_turn_continuation`` is
#: off): the system may still ask for the role, but no answer is fed back, so the run ends
#: on the scripted turns with ``termination_reason == "continuation_disabled"``.
_SINGLE_TURN_VARIANTS = ("one_shot",)

_E2E_VARIANTS = (*_MEMORY_VARIANTS, *_MEMORYLESS_VARIANTS, *_SINGLE_TURN_VARIANTS)

_E2E_MAX_TURNS = 3


def test_e2e_clarification_scenario_runs_across_variants(loop_env) -> None:
    """One clarification-dependent scenario runs end-to-end under every variant.

    Each variant drives the same scenario through ``_run_one`` with no manual
    intervention: the run finishes, records a termination reason, and writes a
    per-run ``dialogue_trace.jsonl`` with one row per turn. ``response_turns`` is
    not the same for all variants -- the variants that carry prior dialogue finish
    on the scripted turns alone, the memory-less variants have to spend an extra
    turn re-collecting the role stated in turn 1, and ``one_shot`` cannot spend that
    turn at all because its ``use_multi_turn_continuation`` is off.

    **Validates: Requirements 7.8, 7.9**
    """
    config = loop_env["config"].model_copy(deep=True)
    config.experiment.max_dialogue_turns = _E2E_MAX_TURNS
    runner = ExperimentRunner(
        config,
        loop_env["catalog_path"],
        loop_env["scenarios_path"],
        out_dir=str(loop_env["out_dir"]),
    )
    scenario = _scenario(_E2E_SLOTS, _E2E_TURNS)
    exp_dir = loop_env["out_dir"] / "exp_e2e"

    turns_by_variant: dict[str, int] = {}
    reason_by_variant: dict[str, str] = {}
    for variant in _E2E_VARIANTS:
        with mock.patch.object(
            simulated_user_module,
            "SimulatedUser",
            _user_class("scenario", _E2E_MAX_TURNS),
        ):
            row, failure = runner._run_one(variant, scenario, 0, exp_dir)

        # The scenario completed end-to-end: no failure, a final system response, and a
        # recorded termination reason (R7.9).
        assert failure is None, (variant, failure)
        assert row["success"], variant
        assert str(row["response_type"]), variant
        reason = row["termination_reason"]
        assert reason in _TERMINAL_REASONS, (variant, reason)
        if reason == "recommendation":
            assert row["returned"] > 0, variant

        # The per-run artifacts exist for this variant: one trace row per turn, with the
        # termination reason on the final row only.
        trace = [
            json.loads(line)
            for line in (Path(row["run_dir"]) / "dialogue_trace.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        assert len(trace) == row["response_turns"], variant
        assert trace[-1]["termination_reason"] == reason, variant
        assert all(record["termination_reason"] is None for record in trace[:-1]), variant

        # The scripted turns were processed before any simulated answer turn.
        assert row["response_turns"] >= len(_E2E_TURNS), variant
        if row["response_turns"] > len(_E2E_TURNS):
            # The extra turns are the simulated user answering the re-asked role slot.
            extra = trace[len(_E2E_TURNS):]
            assert all(record["clarification_slot"] == "target_roles" for record in extra), (
                variant,
                [record["clarification_slot"] for record in extra],
            )

        turns_by_variant[variant] = row["response_turns"]
        reason_by_variant[variant] = reason

    # R7.8: the same scenario does NOT yield one uniform turn count across variants.
    assert len(set(turns_by_variant.values())) > 1, turns_by_variant

    # And the difference tracks the mechanism rather than being incidental: carrying
    # prior dialogue forward means the role from turn 1 still counts, so the scripted
    # turns suffice; without it the role has to be re-collected in one more turn.
    for variant in _MEMORY_VARIANTS:
        assert turns_by_variant[variant] == len(_E2E_TURNS), (variant, turns_by_variant)
    for variant in _MEMORYLESS_VARIANTS:
        assert turns_by_variant[variant] == len(_E2E_TURNS) + 1, (variant, turns_by_variant)
    # one_shot is memory-less too, so it also loses the role -- but it may not continue
    # the dialogue, so it never spends the extra turn and says so in its reason.
    for variant in _SINGLE_TURN_VARIANTS:
        assert turns_by_variant[variant] == len(_E2E_TURNS), (variant, turns_by_variant)
        assert reason_by_variant[variant] == TERMINATION_CONTINUATION_DISABLED, (
            variant, reason_by_variant)


# ---------------------------------------------------------------------------
# R5.3/R5.6: one_shot and no_memory must differ in OBSERVED BEHAVIOUR, not only in
# their flag sets. The flag-level guarantee lives in
# tests/unit/test_variant_isolation.py; these two tests drive the real
# ``ExperimentRunner`` over the real deterministic pipeline, because a flag that no
# code path reads passes every flag-level assertion while producing byte-identical
# runs (which is exactly how ``use_multi_turn_continuation`` shipped dead).
# ---------------------------------------------------------------------------

#: A clarification-dependent scenario whose opening turn states no role, so the system
#: asks for one. Whether that ask is ANSWERED is the whole difference between a variant
#: that may continue the dialogue and one that may not.
_DIVERGENCE_TURNS = ["I am looking for something in Kuala Lumpur, hybrid, around RM5000."]

#: The run-row fields that describe the observable outcome of a run. ``run_id`` and
#: ``total_latency_ms`` are excluded: they differ between any two runs and would make the
#: divergence assertions pass vacuously.
_OBSERVABLE_FIELDS = ("response_turns", "termination_reason", "response_type",
                      "returned", "success")


def _observable(row: dict) -> dict:
    """The outcome of a run as the metrics pipeline sees it (no volatile fields)."""
    return {field: row[field] for field in _OBSERVABLE_FIELDS}


def _run_variant(loop_env, variant: str, exp_name: str, *, continuation: bool | None = None,
                 max_dialogue_turns: int = _E2E_MAX_TURNS) -> dict:
    """Drive one variant through the real runner and return its run row.

    ``continuation`` optionally overrides ``memory.use_multi_turn_continuation`` in the
    config, which is the only way to change that flag without changing the variant.
    Each call writes into its own experiment directory so nothing is overwritten.
    """
    config = loop_env["config"].model_copy(deep=True)
    config.experiment.max_dialogue_turns = max_dialogue_turns
    if continuation is not None:
        config.memory.use_multi_turn_continuation = continuation
    runner = ExperimentRunner(
        config,
        loop_env["catalog_path"],
        loop_env["scenarios_path"],
        out_dir=str(loop_env["out_dir"]),
    )
    scenario = _scenario(_E2E_SLOTS, _DIVERGENCE_TURNS)
    row, _failure = runner._run_one(
        variant, scenario, 0, loop_env["out_dir"] / exp_name)
    return row


def test_one_shot_and_no_memory_diverge_behaviourally(loop_env) -> None:
    """``one_shot`` and ``no_memory`` produce different OUTCOMES, not just different flags.

    Same scenario, same catalog, same config, same real ``SimulatedUser``: the two
    conditions differ on exactly ``use_multi_turn_continuation``, so ``no_memory`` answers
    the system's clarification and reaches a recommendation while ``one_shot`` ends on the
    asking turn with ``continuation_disabled``. A regression that makes the flag dead again
    collapses both rows onto the same values and fails here (R5.3/R5.6).

    **Validates: Requirements 5.3, 5.6**
    """
    one_shot = _run_variant(loop_env, "one_shot", "exp_divergence_one_shot")
    no_memory = _run_variant(loop_env, "no_memory", "exp_divergence_no_memory")

    assert _observable(one_shot) != _observable(no_memory), (
        "one_shot and no_memory produced identical observable outcomes: the only flag "
        "they differ on has no behavioural effect"
    )
    # Named, not incidental: the difference is turns / terminal outcome / task result.
    assert one_shot["response_turns"] < no_memory["response_turns"]
    assert one_shot["termination_reason"] != no_memory["termination_reason"]

    # one_shot is a genuine single-turn condition: the scripted turn only, the system's
    # clarification left unanswered, nothing recommended.
    assert one_shot["response_turns"] == len(_DIVERGENCE_TURNS)
    assert one_shot["termination_reason"] == TERMINATION_CONTINUATION_DISABLED
    assert str(one_shot["response_type"]) == "clarification"
    assert one_shot["returned"] == 0

    # no_memory keeps the multi-turn workflow: the ask is answered and the dialogue ends
    # on a real recommendation.
    assert no_memory["response_turns"] > len(_DIVERGENCE_TURNS)
    assert no_memory["termination_reason"] == "recommendation"
    assert no_memory["returned"] > 0

    # Neither run failed: this is a difference in dialogue behaviour, not an error.
    assert one_shot["success"] and no_memory["success"]


def test_multi_turn_continuation_flag_is_not_dead(loop_env) -> None:
    """``use_multi_turn_continuation`` has an observable effect on the shared code path.

    The guard against the flag going semantically dead again. Everything is held fixed --
    same variant (``no_memory``), same scenario, same catalog -- and only
    ``memory.use_multi_turn_continuation`` is flipped, which is the one input that changes
    that flag without changing anything else about the condition. The resolved flag sets
    must differ on exactly that field, AND the runs must differ observably; a flag no code
    path enforces satisfies the first assertion and fails the second.

    **Validates: Requirements 5.3, 5.6**
    """
    variant = "no_memory"
    flags = {}
    for continuation in (True, False):
        cfg = loop_env["config"].model_copy(deep=True)
        cfg.experiment.variant = ExperimentVariant(variant)
        cfg.memory.use_multi_turn_continuation = continuation
        flags[continuation] = FeatureFlags.from_config(cfg)

    assert flag_diff(flags[True], flags[False]) == {"use_multi_turn_continuation"}

    on = _run_variant(loop_env, variant, "exp_gate_on", continuation=True)
    off = _run_variant(loop_env, variant, "exp_gate_off", continuation=False)

    assert _observable(on) != _observable(off), (
        "flipping memory.use_multi_turn_continuation changed nothing observable: the "
        "flag is semantically dead again"
    )
    assert off["response_turns"] < on["response_turns"]
    assert off["termination_reason"] == TERMINATION_CONTINUATION_DISABLED
    assert on["termination_reason"] == "recommendation"


# ---------------------------------------------------------------------------
# R7.8: the continuation gate is a property of the CONDITION, not of the scenario's
# expectations. A single-turn variant that asks an UNEXPECTED clarification is just as
# truncated as one that asks an expected one, and used to record no terminal state at
# all -- which sent those runs into a memory-related error category and blamed stale
# memory for a dialogue that was never allowed to continue.
# ---------------------------------------------------------------------------

#: The divergence scenario with its clarification expectation REMOVED. The system still
#: asks (the opening turn states no role); the scenario simply does not declare it.
def _unexpected_clarification_scenario() -> dict:
    scenario = _scenario(_E2E_SLOTS, _DIVERGENCE_TURNS)
    scenario["scenario_id"] = "SC-P10-UNEXPECTED"
    scenario["clarification_expected"] = False
    scenario["expects"] = {"response_type": "recommendation"}
    return scenario


def _run_scenario(loop_env, variant: str, scenario: dict, exp_name: str, *,
                  continuation: bool | None = None) -> dict:
    cfg = loop_env["config"].model_copy(deep=True)
    cfg.experiment.max_dialogue_turns = _E2E_MAX_TURNS
    if continuation is not None:
        cfg.memory.use_multi_turn_continuation = continuation
    runner = ExperimentRunner(
        cfg, loop_env["catalog_path"], loop_env["scenarios_path"],
        out_dir=str(loop_env["out_dir"]))
    row, _failure = runner._run_one(
        variant, scenario, 0, loop_env["out_dir"] / exp_name)
    return row


def test_unexpected_clarification_under_a_single_turn_variant_records_truncation(
    loop_env,
) -> None:
    """A single-turn variant that asks an unexpected question still records WHY it stopped.

    The stamp used to require ``clarification_expected``, so this run ended with no
    terminal state. Nothing downstream could then tell that continuation was forbidden,
    and the error taxonomy fell through to a memory category.

    **Validates: Requirements 7.8, 7.9**
    """
    scenario = _unexpected_clarification_scenario()
    row = _run_scenario(loop_env, "one_shot", scenario, "exp_unexpected_one_shot")

    assert row["response_type"] == "clarification", row
    assert row["termination_reason"] == TERMINATION_CONTINUATION_DISABLED, row
    # The trace's final record carries the same reason, which is what the metrics read.
    trace = [
        json.loads(line)
        for line in (Path(row["run_dir"]) / "dialogue_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert trace[-1]["termination_reason"] == TERMINATION_CONTINUATION_DISABLED


def test_a_variant_that_may_continue_does_not_claim_truncation(loop_env) -> None:
    """The stamp is not applied to a condition that WAS allowed to continue.

    Otherwise every unanswered clarification would look like a forbidden continuation,
    and the truncation category would absorb genuine clarification-quality failures.

    **Validates: Requirements 7.8, 7.9**
    """
    scenario = _unexpected_clarification_scenario()
    row = _run_scenario(loop_env, "full", scenario, "exp_unexpected_full")

    assert row["termination_reason"] != TERMINATION_CONTINUATION_DISABLED, row


def test_every_single_turn_run_that_asks_records_a_terminal_state(loop_env) -> None:
    """No single-turn run ends on a clarification without a recorded reason.

    This is the invariant the error taxonomy depends on: its fallback path (classify by
    the scenario's expectation) is what mislabelled these runs, so the primary signal
    must never be absent for them.

    **Validates: Requirements 7.8, 7.9**
    """
    for index, scenario in enumerate((_unexpected_clarification_scenario(),
                                      _scenario(_E2E_SLOTS, _DIVERGENCE_TURNS))):
        row = _run_scenario(loop_env, "one_shot", scenario, f"exp_terminal_{index}")
        if row["response_type"] == "clarification":
            assert row["termination_reason"] == TERMINATION_CONTINUATION_DISABLED, row
        else:
            assert row["termination_reason"], row
