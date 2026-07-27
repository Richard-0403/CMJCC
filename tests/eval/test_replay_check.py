"""Tests for artifact replay and deterministic recomputation (R18).

These exercise the real thing end to end: a real (tiny) experiment is run through
`ExperimentRunner`, and its saved bundles are then replayed in `RunMode.REPLAY`
against the saved `model_calls.jsonl`. The tests assert

- a genuine replay of a genuine run recomputes IDENTICAL key-state hashes for the
  extracted slots, state versions, filtered jobs, ranking output and explanation
  claims -- an empty diff (R18.1, R18.2),
- `replay_diff.json` is written and is byte-stable for an unchanged tree (R18.3),
- when a recorded artifact drifts, the affected key state is recorded in the diff
  with the run and both hashes (R18.4),
- a bundle that cannot be replayed is reported rather than crashing the batch.

No mocks: every hash is computed from real artifacts and a real re-execution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner
from jobrec.evaluation.replay_check import (
    KEY_STATES,
    REPLAY_DIFF_FILENAME,
    recorded_key_states,
    replay_experiment,
    replay_run,
    write_replay_diff,
)

CATALOG_PATH = "data/processed/jobs.jsonl"

_SINGLE_TURN = {
    "scenario_id": "SC-R18-single",
    "scenario_type": "basic",
    "profile": {
        "candidate_id": "SC-R18-single-cand",
        "skills": ["Python", "SQL"],
        "years_experience": 3,
    },
    "turns": ["I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."],
    "expects": {"response_type": "recommendation"},
}

_MULTI_TURN = {
    "scenario_id": "SC-R18-multi",
    "scenario_type": "multi_turn",
    "profile": {
        "candidate_id": "SC-R18-multi-cand",
        "skills": ["Python", "SQL"],
        "years_experience": 3,
    },
    "turns": [
        "I am looking for a data analyst role.",
        "In Kuala Lumpur, hybrid, at least RM4000 please.",
    ],
    "expects": {"response_type": "recommendation"},
}


@pytest.fixture(scope="module")
def experiment_dir(tmp_path_factory) -> Path:
    """A real experiment directory produced by the runner (one variant, two scenarios)."""
    root = tmp_path_factory.mktemp("replay")

    jobs = [
        json.loads(line)
        for line in Path(CATALOG_PATH).read_text().splitlines()
        if line.strip()
    ]
    tiny = [job for job in jobs if job["city"] == "Kuala Lumpur"][:8]
    assert tiny, "expected Kuala Lumpur jobs in the catalog fixture"
    catalog_path = root / "jobs.jsonl"
    catalog_path.write_text("\n".join(json.dumps(job) for job in tiny) + "\n")

    scenarios_path = root / "scenarios.jsonl"
    scenarios_path.write_text(
        "\n".join(json.dumps(scenario) for scenario in (_SINGLE_TURN, _MULTI_TURN)) + "\n"
    )

    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.experiment.repeat_count = 1
    runner = ExperimentRunner(config, str(catalog_path), str(scenarios_path),
                              out_dir=str(root / "runs"))
    manifest = runner.run(["full"])
    return Path(manifest["experiment_dir"])


@pytest.fixture()
def workdir(experiment_dir: Path, tmp_path: Path) -> Path:
    """A throwaway copy of the experiment directory, safe to tamper with."""
    target = tmp_path / experiment_dir.name
    shutil.copytree(experiment_dir, target)
    return target


def test_replay_reproduces_identical_key_state_hashes(experiment_dir: Path):
    """A real replay of a real run recomputes every key-state hash (R18.1, R18.2)."""
    report = replay_experiment(experiment_dir)

    assert len(report.runs) == 2
    assert report.errors == ()
    assert report.differences == ()
    assert report.identical, report.summary()

    for run in report.runs:
        assert set(run.original) == set(KEY_STATES)
        assert run.recomputed == run.original
        assert all(len(digest) == 64 for digest in run.recomputed.values())

    # The comparison must be non-trivial: at least one run actually recommended
    # jobs, so the ranking and claim hashes cover real content.
    decisions = [
        json.loads((experiment_dir / run.run_dir / "recommendation_decision.json").read_text())
        for run in report.runs
    ]
    assert any(d and d.get("selected_job_ids") for d in decisions)


def test_replay_diff_report_is_written_and_deterministic(workdir: Path):
    """The diff report lands at the root, records no differences, and is stable (R18.3)."""
    target = write_replay_diff(workdir)
    assert target == workdir / REPLAY_DIFF_FILENAME

    payload = json.loads(target.read_text())
    assert payload["identical"] is True
    assert payload["differences"] == []
    assert payload["errors"] == []
    assert payload["key_states"] == list(KEY_STATES)
    assert payload["run_count"] == payload["replayed_count"] == 2

    before = target.read_bytes()
    write_replay_diff(workdir)
    assert target.read_bytes() == before


def test_drifted_artifacts_are_recorded_as_differences(workdir: Path):
    """A recorded key state that no longer matches the replay is reported (R18.4)."""
    run_dir = workdir / "full" / _SINGLE_TURN["scenario_id"] / "0"

    decision_path = run_dir / "recommendation_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["selected_job_ids"] = list(reversed(decision["selected_job_ids"]))
    decision_path.write_text(json.dumps(decision, indent=2))

    claims_path = run_dir / "response_claims.json"
    claims = json.loads(claims_path.read_text())
    assert claims, "expected grounded claims in the recommendation bundle"
    claims_path.write_text(json.dumps(claims[:-1], indent=2))

    result = replay_run(run_dir, label="tampered")
    assert result.status == "ok"
    assert not result.identical
    assert {d.key_state for d in result.differences} == {
        "ranking_output", "explanation_claims"
    }
    assert all(d.original != d.recomputed for d in result.differences)
    assert "ranking_output" in result.differences[0].describe()

    write_replay_diff(workdir)
    payload = json.loads((workdir / REPLAY_DIFF_FILENAME).read_text())
    assert payload["identical"] is False
    reported = {(d["run_dir"], d["key_state"]) for d in payload["differences"]}
    expected_dir = f"full/{_SINGLE_TURN['scenario_id']}/0"
    assert reported == {
        (expected_dir, "ranking_output"),
        (expected_dir, "explanation_claims"),
    }


def test_unreplayable_bundle_is_reported_not_raised(tmp_path: Path, experiment_dir: Path):
    """A bundle missing its replay inputs is recorded as an error (R18.3)."""
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    source = experiment_dir / "full" / _SINGLE_TURN["scenario_id"] / "0"
    for name in ("run_record.json", "extracted_preferences.json", "eligibility_results.json",
                 "recommendation_decision.json", "response_claims.json"):
        shutil.copy(source / name, run_dir / name)

    report = replay_experiment(run_dir)
    assert len(report.errors) == 1
    assert not report.identical
    assert "resolved_config.yaml" in report.errors[0].error
    assert report.runs[0].original == recorded_key_states(source)
