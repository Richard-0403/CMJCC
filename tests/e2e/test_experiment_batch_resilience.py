"""One crashed run must not destroy a batch, and must not be citable unnoticed (R16).

``ExperimentRunner.run`` called ``_run_one`` with no exception handling, so an unhandled
error aborted the loop before ``runs_index.csv``, ``failures.csv``, the experiment
manifest or ``checksums.json`` were written. In a multi-hour hybrid batch that means one
malformed model reply, hundreds of calls in, discards every run already completed and
forces a full re-run.

The fix must not trade that for a quieter failure: a crashed run produces no bundle, so it
is missing from every aggregate. These tests pin both halves -- the batch survives, AND
the shortfall is recorded in ``failures.csv``, counted in the manifest, and stated in the
report's summary in terms that forbid citing the figures as covering the full design.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner
from jobrec_eval.report import generate_markdown

CONFIG = "configs/experiment_full.yaml"
CATALOG = "data/processed/jobs.jsonl"
SCENARIOS = "evaluation/data/scenarios_subset.jsonl"
VARIANTS = ["full", "no_memory"]

#: The scenario whose runs are made to crash, in every variant.
_DOOMED = "SC-A-03"


class _Boom(RuntimeError):
    """An error class that no production handler knows about."""


@pytest.fixture(scope="module")
def crashed_batch(tmp_path_factory) -> dict:
    """A batch in which every run of one scenario raises an unknown exception."""
    root = tmp_path_factory.mktemp("batch-resilience")
    config = load_config(CONFIG, base_dir="configs")
    runner = ExperimentRunner(config, CATALOG, SCENARIOS, out_dir=str(root / "_runs"))
    original = runner._run_one

    def _run_one(variant, scenario, run_index, exp_dir):
        if scenario["scenario_id"] == _DOOMED:
            raise _Boom("simulated malformed upstream reply")
        return original(variant, scenario, run_index, exp_dir)

    runner._run_one = _run_one  # type: ignore[method-assign]
    manifest = runner.run(VARIANTS)
    return {"manifest": manifest, "dir": Path(manifest["experiment_dir"]),
            "scenario_count": len(runner.scenarios),
            # Read from the config rather than assumed: experiment_full.yaml ships a
            # repeat count above 1, and the accounting must hold for any of them.
            "repeats": config.experiment.repeat_count}


def test_the_batch_completes_and_writes_every_experiment_artifact(crashed_batch) -> None:
    """The surviving runs are archived: index, failures, manifest and checksums exist.

    Previously none of these files was written at all, so the batch was unusable and the
    completed runs were unrecoverable.

    **Validates: Requirements 16.1**
    """
    exp_dir = crashed_batch["dir"]
    for name in ("runs_index.csv", "failures.csv", "experiment_manifest.json",
                 "checksums.json", "resolved_config.yaml", "catalog.jsonl",
                 "scenarios.jsonl"):
        assert (exp_dir / name).exists(), name

    manifest = crashed_batch["manifest"]
    repeats = crashed_batch["repeats"]
    expected = len(VARIANTS) * crashed_batch["scenario_count"] * repeats
    crashed = len(VARIANTS) * repeats  # one doomed scenario, every variant, every repeat
    assert manifest["expected_run_count"] == expected
    assert manifest["crashed_run_count"] == crashed
    assert manifest["run_count"] == expected - crashed
    # The surviving runs really are on disk.
    rows = list(csv.DictReader((exp_dir / "runs_index.csv").open()))
    assert len(rows) == expected - crashed
    assert _DOOMED not in {r["scenario_id"] for r in rows}


def test_the_crash_is_recorded_with_its_exception_class(crashed_batch) -> None:
    """Each crash lands in ``failures.csv`` and in the manifest, naming only the class.

    The class, not the message: a transport error message can quote the request and
    therefore the credential.

    **Validates: Requirements 16.1, 26.1**
    """
    exp_dir = crashed_batch["dir"]
    failures = list(csv.DictReader((exp_dir / "failures.csv").open()))
    crash_rows = [r for r in failures if r["scenario_id"] == _DOOMED]
    assert len(crash_rows) == len(VARIANTS) * crashed_batch["repeats"]
    for row in crash_rows:
        assert row["failure_code"] == "runner_exception:_Boom"
        assert "simulated malformed upstream reply" not in json.dumps(row)

    crashed = crashed_batch["manifest"]["crashed_runs"]
    assert {c["scenario_id"] for c in crashed} == {_DOOMED}
    assert {c["variant"] for c in crashed} == set(VARIANTS)
    assert ({c["repeat_index"] for c in crashed}
            == set(range(crashed_batch["repeats"])))


def test_a_complete_batch_reports_no_crashes(tmp_path) -> None:
    """The accounting is not noise: an untouched batch records zero crashes.

    **Validates: Requirements 16.1**
    """
    config = load_config(CONFIG, base_dir="configs")
    runner = ExperimentRunner(config, CATALOG, SCENARIOS, out_dir=str(tmp_path / "_runs"))
    manifest = runner.run(["full"])
    assert manifest["crashed_run_count"] == 0
    assert manifest["crashed_runs"] == []
    assert manifest["run_count"] == manifest["expected_run_count"]


# ------------------------------------------------------------------ report disclosure
def _report_data(run_count: int, expected: int, crashed: int) -> dict:
    """The shared report fixture with the run-count accounting overridden.

    Reuses ``tests.eval.test_eval_report_framing`` rather than hand-rolling a second
    report payload, so this test cannot drift from the shape the renderer actually needs.
    """
    from tests.eval.test_eval_report_framing import _report_data as base

    data = base()
    data["experiment"] = {
        **data["experiment"],
        "run_count": run_count,
        "expected_run_count": expected,
        "crashed_run_count": crashed,
    }
    return data


def test_report_flags_an_incomplete_experiment_in_its_summary() -> None:
    """The shortfall is stated where the headline figures are, in citable-or-not terms.

    A crashed run is absent from every table, so without this the aggregate silently
    describes fewer runs than the design and nothing contradicts a reader who cites it as
    the whole experiment.

    **Validates: Requirements 16.1**
    """
    md = generate_markdown(_report_data(run_count=22, expected=24, crashed=2))
    squeezed = " ".join(md.split())
    assert "INCOMPLETE EXPERIMENT: 2 run(s) crashed" in squeezed
    assert "22 of 24 planned runs" in squeezed
    assert "Do not cite these figures as covering the full design" in squeezed


def test_report_derives_the_planned_count_from_an_older_manifest() -> None:
    """Re-analysing bundles whose manifest predates the accounting still states a count.

    ``--experiment-dir`` reuses the original manifest, which may not carry
    ``expected_run_count``; the design (variants x scenarios x repeats) is there, so the
    line reads as a real completeness statement instead of "378 of n/a".

    **Validates: Requirements 16.1**
    """
    data = _report_data(run_count=10, expected=10, crashed=0)
    del data["experiment"]["expected_run_count"]
    exp = data["experiment"]
    exp["run_count"] = len(exp["variants"]) * exp["scenario_count"] * exp["repeat_count"]

    squeezed = " ".join(generate_markdown(data).split())
    assert f"every planned run produced a bundle ({exp['run_count']} of " in squeezed
    assert "of n/a" not in squeezed
    assert "INCOMPLETE EXPERIMENT" not in squeezed


def test_report_refuses_to_claim_completeness_without_a_design() -> None:
    """With neither a planned count nor a derivable design, completeness is NOT claimed.

    Silence would read as "complete"; the line says explicitly that it is unestablished.

    **Validates: Requirements 16.1**
    """
    data = _report_data(run_count=10, expected=10, crashed=0)
    del data["experiment"]["expected_run_count"]
    data["experiment"]["repeat_count"] = None

    squeezed = " ".join(generate_markdown(data).split())
    assert "completeness is NOT established here" in squeezed
    assert "every planned run produced a bundle" not in squeezed


def test_report_confirms_completeness_when_nothing_crashed() -> None:
    """A complete run says so positively, so the absence of a warning is not ambiguous.

    **Validates: Requirements 16.1**
    """
    md = generate_markdown(_report_data(run_count=24, expected=24, crashed=0))
    squeezed = " ".join(md.split())
    assert "every planned run produced a bundle (24 of 24)" in squeezed
    assert "INCOMPLETE EXPERIMENT" not in squeezed
