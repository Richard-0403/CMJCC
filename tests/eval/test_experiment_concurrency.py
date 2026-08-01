"""Running the batch N-at-a-time produces the same artifacts as one-at-a-time.

Concurrency exists because a hybrid batch spends nearly all its wall clock waiting on a remote
endpoint: 126 runs took 75 minutes sequentially, and every code fix had to pay that again.
Threads are safe here only because runs are already isolated -- each builds its own deep-copied
config, its own :class:`~jobrec.app_service.AppService` with its own in-memory repository, and
the remote provider opens a fresh ``httpx.Client`` per request.

What has to hold, and is asserted below rather than assumed:

* the same runs exist, with the same ids -- session ids are content-addressed over
  (experiment, variant, scenario, repeat, ordinal), so they must not depend on scheduling;
* the artifacts are written in PLAN order, not completion order. ``runs_index.csv`` and the
  manifest's ``run_manifests`` are part of the archive, so a batch that reorders them under
  load would differ from its own sequential twin while being identical run by run;
* the per-run key states are identical, which is the actual reproducibility claim;
* ``concurrency`` is recorded, because every latency figure in the batch is wall-clock and
  contended at any value above 1.

Deterministic mode throughout: this is about the runner's scheduling, not about the model, and
a remote call would make the comparison depend on the endpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_identity import EXPERIMENT_MANIFEST_FILENAME
from jobrec.evaluation.experiment_runner import ExperimentRunner

CONFIG = "configs/experiment_full.yaml"
CATALOG = "data/processed/jobs.jsonl"
SCENARIOS = "evaluation/data/scenarios_subset.jsonl"

#: Two variants over the subset, so the plan spans more than one variant and the ordering
#: assertion has something to order.
VARIANTS = ["full", "no_memory"]


def _run(out_dir: Path, *, concurrency: int) -> dict:
    config = load_config(CONFIG, base_dir="configs")
    config.experiment.repeat_count = 1
    runner = ExperimentRunner(config, CATALOG, SCENARIOS, out_dir=str(out_dir))
    return runner.run(VARIANTS, concurrency=concurrency)


def _index_rows(exp_dir: Path) -> list[dict]:
    with (exp_dir / "runs_index.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _key_states(exp_dir: Path) -> dict[str, dict]:
    """Each run's identity and its decision/response ids, keyed by run directory.

    Latency is deliberately excluded: it is wall-clock, so it is EXPECTED to differ under
    contention, and asserting on it would be asserting that concurrency does nothing.
    """
    states: dict[str, dict] = {}
    for path in sorted(exp_dir.rglob("run_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = "/".join(path.parts[-4:-1])
        states[key] = {
            "run_id": record.get("run_id"),
            "session_id": record.get("session_id"),
            "scenario_id": record.get("scenario_id"),
            "success": record.get("success"),
            "failure_code": record.get("failure_code"),
        }
    return states


@pytest.fixture(scope="module")
def sequential(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("seq")
    manifest = _run(out, concurrency=1)
    return Path(manifest["experiment_dir"])


@pytest.fixture(scope="module")
def concurrent(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("par")
    manifest = _run(out, concurrency=6)
    return Path(manifest["experiment_dir"])


def test_the_same_experiment_id_and_run_count(sequential: Path, concurrent: Path):
    """Scheduling is not an input: it must not fork the artifact."""
    assert sequential.name == concurrent.name, (
        "concurrency changed the experiment id, so a re-run at a different speed would look "
        "like a different experiment")
    assert len(_index_rows(sequential)) == len(_index_rows(concurrent))
    assert len(_index_rows(concurrent)) == len(VARIANTS) * 12


def test_the_index_is_written_in_plan_order_not_completion_order(sequential: Path,
                                                                concurrent: Path):
    """The archive must not depend on which run happened to finish first."""
    columns = ("experiment_variant", "scenario_id", "run_index")
    order_of = [tuple(row[c] for c in columns) for row in _index_rows(concurrent)]
    assert order_of == [tuple(row[c] for c in columns) for row in _index_rows(sequential)]
    # And that order really is the plan: variant-major, then scenario, then repeat.
    assert order_of == sorted(order_of, key=lambda k: (VARIANTS.index(k[0]),))


def test_every_run_id_and_session_id_is_unchanged(sequential: Path, concurrent: Path):
    """Ids are content-addressed over the batch position, never over arrival order."""
    assert _key_states(concurrent) == _key_states(sequential)


def test_the_manifest_records_the_concurrency_and_still_lists_runs_in_order(
        sequential: Path, concurrent: Path):
    seq = json.loads((sequential / EXPERIMENT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    par = json.loads((concurrent / EXPERIMENT_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert seq["concurrency"] == 1
    assert par["concurrency"] == 6
    assert par["run_manifests"] == seq["run_manifests"]
    assert par["run_count"] == seq["run_count"]
    assert par["crashed_run_count"] == seq["crashed_run_count"] == 0
    # Everything the reproducibility claim rests on is untouched.
    for key in ("config_hash", "catalog_hash", "scenarios_hash", "prompt_hash",
                "execution_fingerprint", "source_fingerprint", "variants"):
        assert par[key] == seq[key], key


def test_the_bundles_verify_against_their_own_checksums(concurrent: Path):
    """A concurrent write must not tear a file, and the manifest must cover every artifact."""
    from jobrec.evaluation.checksums import verify_checksums

    assert verify_checksums(concurrent) == []
