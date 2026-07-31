"""The experiment id carries the CODE identity, and no complete artifact is clobbered.

The defect these tests pin down: ``experiment_id`` used to hash the experiment inputs only
(variants, scenario ids, config hash), so re-running the official experiment after a
behavioural bug fix produced the SAME id, the pipeline wrote into the existing directory,
and the pre-fix baseline was destroyed -- with no warning and no way to perform the
before/after comparison afterwards.

What is asserted here:

- the id is a function of the inputs AND the code content: identical inputs with a
  different source fingerprint yield different ids (the dangerous case),
- the id is idempotent for unchanged code and inputs, and does NOT move when only the
  commit hash / dirty flag change (committing does not change the code),
- the fingerprint itself really reads file bytes,
- the overwrite guard passes on a fresh/incomplete directory, refuses on a complete one
  (naming whether the code differs and how to proceed), and yields to an explicit opt-in,
- a real experiment written by ``ExperimentRunner`` records the full code identity in
  ``experiment_manifest.json``, refuses a second write into its own directory, and still
  supports an intentional idempotent re-run with ``allow_overwrite=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation import experiment_identity as ident
from jobrec.evaluation import experiment_runner as runner_module
from jobrec.evaluation.experiment_identity import (
    CODE_IDENTITY_FIELDS,
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentOverwriteError,
    code_identity,
    experiment_id,
    guard_output_dir,
    reset_code_identity_cache,
    source_fingerprint,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner

CATALOG_PATH = "data/processed/jobs.jsonl"

_INPUTS = {
    "variants": ["full", "no_memory"],
    "scenario_ids": ["SC-001", "SC-002"],
    "config_hash": "cfg-hash",
}

_SCENARIO = {
    "scenario_id": "SC-ID-GUARD",
    "scenario_type": "basic",
    "profile": {
        "candidate_id": "SC-ID-GUARD-cand",
        "skills": ["Python", "SQL"],
        "years_experience": 3,
    },
    "turns": ["I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."],
    "expects": {"response_type": "recommendation"},
}


def _identity(**overrides) -> dict:
    base = {
        "code_version": "0.1.0",
        "commit_hash": "a" * 40,
        "git_dirty": False,
        "source_fingerprint": "f" * 64,
        "execution_fingerprint": "e" * 64,
        "analysis_fingerprint": "d" * 64,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------- the id itself
def test_identical_inputs_with_different_code_yield_different_ids():
    """The dangerous case: same experiment inputs, different EXECUTION source -> new id.

    The id is keyed on the execution fingerprint, so this is what "different code" means
    for a run: code that could have changed the bundles. An analysis-only edit is the
    complementary case with the opposite requirement, asserted in
    ``tests/unit/test_experiment_identity_split.py``.
    """
    before = experiment_id(**_INPUTS, identity=_identity(execution_fingerprint="1" * 64))
    after = experiment_id(**_INPUTS, identity=_identity(execution_fingerprint="2" * 64))

    assert before != after
    assert before.startswith("exp-") and after.startswith("exp-")
    assert len(before) == len("exp-") + 12


def test_a_rerun_of_unchanged_code_is_idempotent():
    """Same inputs + same code identity -> same id, so an intentional re-run still works."""
    identity = _identity()
    assert experiment_id(**_INPUTS, identity=identity) == experiment_id(**_INPUTS,
                                                                       identity=identity)


def test_the_declared_code_version_also_moves_the_id():
    assert (experiment_id(**_INPUTS, identity=_identity(code_version="0.1.0"))
            != experiment_id(**_INPUTS, identity=_identity(code_version="0.2.0")))


def test_committing_the_same_sources_does_not_move_the_id():
    """Only code CONTENT counts: a new commit hash / dirty flag over identical sources
    must not invalidate an id, otherwise ``git commit`` alone would fork the identity."""
    committed = _identity(commit_hash="b" * 40, git_dirty=False)
    dirty = _identity(commit_hash="c" * 40, git_dirty=True)

    assert experiment_id(**_INPUTS, identity=committed) == experiment_id(**_INPUTS,
                                                                        identity=dirty)


def test_the_inputs_still_move_the_id():
    """Adding the code identity did not stop the inputs from being part of the id."""
    identity = _identity()
    baseline = experiment_id(**_INPUTS, identity=identity)
    assert experiment_id(variants=["full"], scenario_ids=_INPUTS["scenario_ids"],
                         config_hash="cfg-hash", identity=identity) != baseline
    assert experiment_id(variants=_INPUTS["variants"], scenario_ids=["SC-001"],
                         config_hash="cfg-hash", identity=identity) != baseline
    assert experiment_id(**{**_INPUTS, "config_hash": "other"}, identity=identity) != baseline


# ----------------------------------------------------------- the code identity
def test_code_identity_reports_every_field():
    identity = code_identity()
    assert set(identity) == set(CODE_IDENTITY_FIELDS)
    assert identity["code_version"]
    assert len(identity["source_fingerprint"]) == 64
    # commit_hash/git_dirty are None only when git is unavailable; in a checkout they are set.
    assert identity["commit_hash"] is None or len(identity["commit_hash"]) == 40
    assert identity["git_dirty"] in (True, False, None)


def test_source_fingerprint_follows_the_bytes_on_disk(tmp_path: Path, monkeypatch):
    """Real files, real digests: editing a source file changes the fingerprint."""
    package = tmp_path / "jobrec"
    package.mkdir()
    module = package / "thing.py"
    module.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(ident, "_PACKAGE_PARENT", tmp_path)

    reset_code_identity_cache()
    before = source_fingerprint()
    module.write_text("value = 2\n", encoding="utf-8")
    reset_code_identity_cache()
    after = source_fingerprint()
    module.write_text("value = 1\n", encoding="utf-8")
    reset_code_identity_cache()
    restored = source_fingerprint()

    assert before != after
    assert restored == before
    # Line-ending style alone is not a code change (same commit, Windows vs Linux checkout).
    module.write_bytes(b"value = 1\r\n")
    reset_code_identity_cache()
    assert source_fingerprint() == before

    monkeypatch.undo()
    reset_code_identity_cache()


# ------------------------------------------------------------------- the guard
def test_guard_allows_a_fresh_or_incomplete_directory(tmp_path: Path):
    """No manifest means no complete experiment: a crashed run may be re-run freely."""
    guard_output_dir(tmp_path / "missing")

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "runs_index.csv").write_text("x\n", encoding="utf-8")
    guard_output_dir(partial)


def _complete_experiment(target: Path, identity: dict,
                         manifest_name: str = EXPERIMENT_MANIFEST_FILENAME) -> Path:
    path = target / manifest_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiment_id": "exp-existing", **identity}), encoding="utf-8")
    return path


def test_guard_refuses_when_the_recorded_code_differs(tmp_path: Path):
    """The exact defect: an existing artifact from OTHER code is never replaced silently."""
    target = tmp_path / "exp-abc"
    _complete_experiment(target, _identity(source_fingerprint="1" * 64))

    with pytest.raises(ExperimentOverwriteError) as excinfo:
        guard_output_dir(target, identity=_identity(source_fingerprint="2" * 64))

    message = str(excinfo.value)
    assert "DIFFERS" in message
    assert str(target) in message
    # Actionable: it names both identities and the way forward.
    assert "--allow-overwrite" in message
    assert "1111" in message and "2222" in message


def test_guard_refuses_a_plain_rerun_but_says_the_code_matches(tmp_path: Path):
    target = tmp_path / "exp-abc"
    identity = _identity()
    _complete_experiment(target, identity)

    with pytest.raises(ExperimentOverwriteError, match="MATCHES"):
        guard_output_dir(target, identity=identity)


def test_guard_yields_to_the_explicit_opt_in(tmp_path: Path):
    target = tmp_path / "exp-abc"
    _complete_experiment(target, _identity())
    guard_output_dir(target, identity=_identity(source_fingerprint="9" * 64),
                     allow_overwrite=True)


def test_guard_reads_the_nested_analysis_manifest(tmp_path: Path):
    """The pipeline's analysis directory keeps its manifest under ``manifests/``."""
    target = tmp_path / "exp-abc"
    nested = f"manifests/{EXPERIMENT_MANIFEST_FILENAME}"
    _complete_experiment(target, _identity(), manifest_name=nested)

    # Only the nested manifest exists, so the default (root) lookup must not fire ...
    guard_output_dir(target, identity=_identity())
    # ... while the analysis lookup does.
    with pytest.raises(ExperimentOverwriteError):
        guard_output_dir(target, manifest_name=nested, identity=_identity())


def test_guard_message_names_the_flag_of_the_calling_entry_point(tmp_path: Path):
    target = tmp_path / "exp-abc"
    _complete_experiment(target, _identity())
    with pytest.raises(ExperimentOverwriteError, match="--force-rerun"):
        guard_output_dir(target, identity=_identity(), overwrite_flag="--force-rerun")


def test_an_unreadable_manifest_still_counts_as_a_complete_experiment(tmp_path: Path):
    target = tmp_path / "exp-abc"
    target.mkdir()
    (target / EXPERIMENT_MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ExperimentOverwriteError, match="unrecorded"):
        guard_output_dir(target, identity=_identity())


# ---------------------------------------------------- the runner, for real
@pytest.fixture(scope="module")
def tiny_inputs(tmp_path_factory) -> tuple[str, str]:
    """A 6-job catalog and a single scenario: a real experiment that runs in seconds."""
    root = tmp_path_factory.mktemp("identity-inputs")
    jobs = [json.loads(line) for line in Path(CATALOG_PATH).read_text().splitlines()
            if line.strip()]
    tiny = [job for job in jobs if job["city"] == "Kuala Lumpur"][:6]
    assert tiny, "expected Kuala Lumpur jobs in the catalog fixture"
    catalog_path = root / "jobs.jsonl"
    catalog_path.write_text("\n".join(json.dumps(job) for job in tiny) + "\n")
    scenarios_path = root / "scenarios.jsonl"
    scenarios_path.write_text(json.dumps(_SCENARIO) + "\n")
    return str(catalog_path), str(scenarios_path)


def _runner(tiny_inputs: tuple[str, str], out_dir: Path) -> ExperimentRunner:
    catalog_path, scenarios_path = tiny_inputs
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.experiment.repeat_count = 1
    return ExperimentRunner(config, catalog_path, scenarios_path, out_dir=str(out_dir))


def test_experiment_manifest_records_the_code_identity(tiny_inputs, tmp_path: Path):
    """Two artifacts can be told apart offline from the manifest alone."""
    manifest = _runner(tiny_inputs, tmp_path / "runs").run(["full"])
    exp_dir = Path(manifest["experiment_dir"])
    on_disk = json.loads((exp_dir / EXPERIMENT_MANIFEST_FILENAME).read_text())

    identity = code_identity()
    for field in CODE_IDENTITY_FIELDS:
        assert on_disk[field] == identity[field], field
    assert on_disk["commit_hash"] == identity["commit_hash"]
    assert on_disk["code_version"]
    # The id is derived from that identity, so the directory name matches the recorded code.
    assert exp_dir.name == manifest["experiment_id"]
    # The runtime block is recorded as the dict the id was derived from, so the id can be
    # re-derived from the manifest offline. Asserted here because the digest is only
    # trustworthy if what it consumed is written down: see
    # ``tests/eval/test_experiment_identity_runtime.py`` for the fields themselves.
    recorded = on_disk["runtime_identity"]
    assert list(recorded) == list(ident.RUNTIME_IDENTITY_FIELDS)
    assert recorded["catalog_hash"] == on_disk["catalog_hash"]
    assert recorded["prompt_hash"] == on_disk["prompt_hash"]
    assert ident.experiment_id(
        variants=on_disk["variants"],
        scenario_ids=[_SCENARIO["scenario_id"]],
        config_hash=on_disk["config_hash"],
        identity=identity,
        scenarios_fingerprint=on_disk["scenarios_hash"],
        runtime=recorded,
    ) == manifest["experiment_id"]
    # ... and each run bundle repeats the identity, so a single bundle is self-describing.
    run_manifest = json.loads((exp_dir / on_disk["run_manifests"][0]).read_text())
    assert run_manifest["commit_hash"] == identity["commit_hash"]
    assert run_manifest["code_version"] == identity["code_version"]
    assert run_manifest["git_dirty"] == identity["git_dirty"]
    assert run_manifest["source_fingerprint"] == identity["source_fingerprint"]


def test_rerunning_the_same_code_refuses_then_succeeds_with_the_opt_in(tiny_inputs,
                                                                      tmp_path: Path):
    """Unchanged code lands on the same id; replacing it has to be asked for."""
    runs = tmp_path / "runs"
    first = _runner(tiny_inputs, runs).run(["full"])

    with pytest.raises(ExperimentOverwriteError, match="MATCHES"):
        _runner(tiny_inputs, runs).run(["full"])

    again = _runner(tiny_inputs, runs).run(["full"], allow_overwrite=True)
    assert again["experiment_id"] == first["experiment_id"]
    assert again["run_count"] == first["run_count"]


def test_a_different_code_version_writes_a_new_directory_and_keeps_the_baseline(
        tiny_inputs, tmp_path: Path, monkeypatch):
    """The regression this fixes: the pre-fix artifact survives a post-fix re-run.

    "Different code" is simulated the way the code itself sees it -- a different code
    identity, i.e. what editing any ``src/**/*.py`` file produces.
    """
    runs = tmp_path / "runs"
    baseline = _runner(tiny_inputs, runs).run(["full"])
    baseline_dir = Path(baseline["experiment_dir"])
    baseline_manifest = (baseline_dir / EXPERIMENT_MANIFEST_FILENAME).read_text()

    patched = {**code_identity(), "execution_fingerprint": "9" * 64,
               "source_fingerprint": "9" * 64}
    monkeypatch.setattr(runner_module, "code_identity", lambda: patched)
    after_fix = _runner(tiny_inputs, runs).run(["full"])
    after_dir = Path(after_fix["experiment_dir"])

    assert after_fix["experiment_id"] != baseline["experiment_id"]
    assert after_dir != baseline_dir
    # The baseline is untouched, byte for byte, so before/after stays performable.
    assert (baseline_dir / EXPERIMENT_MANIFEST_FILENAME).read_text() == baseline_manifest
    assert json.loads((after_dir / EXPERIMENT_MANIFEST_FILENAME).read_text())[
        "source_fingerprint"] == "9" * 64
    assert {p.name for p in runs.iterdir()} == {baseline_dir.name, after_dir.name}
