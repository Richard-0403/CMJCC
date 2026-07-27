"""Tests for the unified artifact checksum manifest and `verify` command (R16).

These exercise the real thing end to end: a real (tiny) experiment is run through
`ExperimentRunner`, which now writes `checksums.json` over every artifact it
produced. The tests then assert

- coverage: the manifest names every input AND output artifact in the tree, and
  the superseded `checksums.sha256` is gone (R16.1),
- an intact directory verifies clean, and the `verify` CLI command exits 0
  (R16.2),
- each way a tree can drift (modified / deleted / added file) is reported by
  artifact name, and the CLI exits non-zero naming the offending artifact
  (R16.3).

No mocks or stubbed digests: every checksum is computed from bytes on disk.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from jobrec.config import load_config
from jobrec.evaluation.checksums import (
    CHECKSUMS_FILENAME,
    MissingChecksumsError,
    compute_checksums,
    read_checksums,
    verify_checksums,
    write_checksums,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner
from jobrec_eval import cli as eval_cli

CATALOG_PATH = "data/processed/jobs.jsonl"

#: Artifacts that must appear in the manifest: the experiment-level inputs and
#: outputs, plus a representative slice of a per-run bundle (json, jsonl, yaml).
_EXPECTED_ROOT_ARTIFACTS = (
    "resolved_config.yaml",
    "catalog.jsonl",
    "scenarios.jsonl",
    "experiment_manifest.json",
    "runs_index.csv",
    "failures.csv",
)
_EXPECTED_RUN_ARTIFACTS = (
    "run_record.json",
    "run_manifest.json",
    "recommendation_decision.json",
    "model_calls.jsonl",
    "dialogue_trace.jsonl",
    "evidence_log.jsonl",
    "resolved_config.yaml",
)

_SCENARIO = {
    "scenario_id": "SC-R16",
    "scenario_type": "basic",
    "profile": {
        "candidate_id": "SC-R16-cand",
        "skills": ["Python", "SQL"],
        "years_experience": 3,
    },
    "turns": ["I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."],
    "expects": {"response_type": "recommendation"},
}


@pytest.fixture(scope="module")
def experiment_dir(tmp_path_factory) -> Path:
    """A real experiment directory produced by the runner (one variant, one scenario)."""
    root = tmp_path_factory.mktemp("checksums")

    jobs = [
        json.loads(line)
        for line in Path(CATALOG_PATH).read_text().splitlines()
        if line.strip()
    ]
    tiny = [job for job in jobs if job["city"] == "Kuala Lumpur"][:6]
    assert tiny, "expected Kuala Lumpur jobs in the catalog fixture"
    catalog_path = root / "jobs.jsonl"
    catalog_path.write_text("\n".join(json.dumps(job) for job in tiny) + "\n")

    scenarios_path = root / "scenarios.jsonl"
    scenarios_path.write_text(json.dumps(_SCENARIO) + "\n")

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


def test_manifest_covers_every_input_and_output_artifact(experiment_dir: Path):
    """checksums.json names all artifacts and supersedes the old .sha256 file (R16.1)."""
    recorded = read_checksums(experiment_dir)

    on_disk = {
        path.relative_to(experiment_dir).as_posix()
        for path in experiment_dir.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_FILENAME
    }
    assert set(recorded) == on_disk
    assert all(len(digest) == 64 for digest in recorded.values())

    for name in _EXPECTED_ROOT_ARTIFACTS:
        assert name in recorded, f"missing experiment-level artifact {name}"
    run_dir = "full/SC-R16/0"
    for name in _EXPECTED_RUN_ARTIFACTS:
        assert f"{run_dir}/{name}" in recorded, f"missing run artifact {name}"

    # The partial, json-only manifest this replaces must not linger.
    assert not (experiment_dir / "checksums.sha256").exists()


def test_intact_directory_verifies_and_manifest_is_deterministic(workdir: Path):
    """Verification of an untouched tree finds nothing, and rewriting is stable (R16.2)."""
    assert verify_checksums(workdir) == []

    before = (workdir / CHECKSUMS_FILENAME).read_bytes()
    write_checksums(workdir)
    assert (workdir / CHECKSUMS_FILENAME).read_bytes() == before


def test_modified_artifact_is_named(workdir: Path):
    """A changed artifact is reported as modified, by name (R16.3)."""
    target = workdir / "full/SC-R16/0/run_record.json"
    expected = read_checksums(workdir)[target.relative_to(workdir).as_posix()]
    target.write_text(target.read_text() + "\n")

    findings = verify_checksums(workdir)
    assert [(f.artifact, f.kind) for f in findings] == [
        ("full/SC-R16/0/run_record.json", "modified")
    ]
    assert findings[0].expected == expected
    assert findings[0].actual != expected
    assert "run_record.json" in findings[0].describe()


def test_deleted_and_added_artifacts_are_reported(workdir: Path):
    """Deletions are `missing`; additions are `untracked` unless suppressed (R16.3)."""
    (workdir / "failures.csv").unlink()
    (workdir / "stray.json").write_text("{}")

    findings = verify_checksums(workdir)
    assert {(f.artifact, f.kind) for f in findings} == {
        ("failures.csv", "missing"),
        ("stray.json", "untracked"),
    }

    relaxed = verify_checksums(workdir, report_untracked=False)
    assert [(f.artifact, f.kind) for f in relaxed] == [("failures.csv", "missing")]


def test_verify_command_exits_zero_on_intact_dir(workdir: Path, capsys, monkeypatch):
    """`verify <dir>` succeeds on an intact directory (R16.2)."""
    monkeypatch.setattr("sys.argv", ["jobrec-eval", "verify", str(workdir)])
    eval_cli.main()
    assert "OK:" in capsys.readouterr().out


def test_verify_command_names_offender_and_exits_non_zero(workdir: Path, capsys, monkeypatch):
    """`verify <dir>` prints the tampered artifact and exits non-zero (R16.3)."""
    tampered = workdir / "catalog.jsonl"
    tampered.write_text(tampered.read_text().replace("Kuala Lumpur", "Penang", 1))

    monkeypatch.setattr("sys.argv", ["jobrec-eval", "verify", str(workdir)])
    with pytest.raises(SystemExit) as exit_info:
        eval_cli.main()

    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "catalog.jsonl" in out


def test_verify_command_reports_absent_manifest(tmp_path: Path, capsys, monkeypatch):
    """A directory with no manifest cannot be verified and exits non-zero (R16.3)."""
    with pytest.raises(MissingChecksumsError):
        read_checksums(tmp_path)

    monkeypatch.setattr("sys.argv", ["jobrec-eval", "verify", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        eval_cli.main()

    assert exit_info.value.code == 2
    assert CHECKSUMS_FILENAME in capsys.readouterr().out


def test_compute_checksums_uses_posix_relative_paths(workdir: Path):
    """Manifest keys are POSIX-relative so digests are platform-stable (R16.1)."""
    checksums = compute_checksums(workdir)
    assert checksums
    assert all("\\" not in key and not key.startswith("/") for key in checksums)


# ---------------------------------------------------------------------------
# Property 23
# ---------------------------------------------------------------------------

#: Directory prefixes mirroring the shape of a real experiment tree (root level,
#: per-run bundles, metrics, plots, nested manifests). No prefix contains a dot,
#: and every generated file name does, so a drawn file can never collide with a
#: drawn directory.
_DIR_PREFIXES = ("", "full/SC-1/0", "no_memory/SC-1/0", "metrics", "plots", "manifests/nested")
_EXTENSIONS = (".json", ".jsonl", ".csv", ".yaml", ".png", ".md", ".txt")

#: Stems the checksum module deliberately ignores (manifests, OS noise) plus the
#: Windows device names, none of which can exist as ordinary artifacts.
_RESERVED_STEMS = frozenset(
    {"checksums", "thumbs", "desktop", "con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)

_artifact_names = st.builds(
    lambda stem, extension: stem + extension,
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=10),
    st.sampled_from(_EXTENSIONS),
).filter(lambda name: Path(name).stem.lower() not in _RESERVED_STEMS)

_artifact_paths = st.builds(
    lambda prefix, name: f"{prefix}/{name}" if prefix else name,
    st.sampled_from(_DIR_PREFIXES),
    _artifact_names,
)

_artifact_contents = st.binary(min_size=0, max_size=24)


@st.composite
def _tree_and_mutation(draw) -> tuple[dict[str, bytes], tuple[str, str, bytes]]:
    """Draw an artifact tree plus exactly one mutation to apply to it.

    The mutation is `(kind, artifact, payload)` where `kind` is `modify` (append
    bytes to a recorded file), `delete` (remove a recorded file) or `add` (create
    a file the manifest does not know about).
    """
    tree = draw(st.dictionaries(_artifact_paths, _artifact_contents, min_size=1, max_size=6))
    kind = draw(st.sampled_from(["modify", "delete", "add"]))
    if kind == "add":
        artifact = draw(_artifact_paths)
        assume(artifact not in tree)
        payload = draw(st.binary(min_size=0, max_size=16))
    else:
        artifact = draw(st.sampled_from(sorted(tree)))
        payload = draw(st.binary(min_size=1, max_size=8))
    return tree, (kind, artifact, payload)


def _materialize(root: Path, tree: dict[str, bytes]) -> None:
    """Write a drawn artifact tree to disk under `root`."""
    for relative, content in tree.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


# Feature: cmjcc-experiment-readiness, Property 23: Checksums round-trip and detect tampering
@settings(max_examples=100, deadline=None)
@given(_tree_and_mutation())
def test_property_checksums_round_trip_and_detect_tampering(
    case: tuple[dict[str, bytes], tuple[str, str, bytes]],
):
    """Writing then verifying any artifact tree is clean; any single mutation is caught.

    The manifest covers every artifact in the tree (R16.1) and an untouched tree
    verifies with no findings (R16.2). Changing a file's bytes, deleting a
    recorded file, or adding an unrecorded one each produces exactly one finding
    naming that artifact with the matching kind (R16.3).

    **Validates: Requirements 16.1, 16.2, 16.3**
    """
    tree, (mutation, artifact, payload) = case
    expected_kind = {"modify": "modified", "delete": "missing", "add": "untracked"}[mutation]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _materialize(root, tree)

        # Round-trip: the manifest covers exactly the tree, and verifies clean.
        assert write_checksums(root) == root / CHECKSUMS_FILENAME
        recorded = read_checksums(root)
        assert set(recorded) == set(tree)
        assert recorded == compute_checksums(root)
        assert verify_checksums(root, report_untracked=True) == []

        target = root / artifact
        if mutation == "modify":
            target.write_bytes(tree[artifact] + payload)
        elif mutation == "delete":
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        findings = verify_checksums(root, report_untracked=True)
        assert [(f.artifact, f.kind) for f in findings] == [(artifact, expected_kind)]

        finding = findings[0]
        assert artifact in finding.describe()
        if mutation == "modify":
            assert finding.expected == recorded[artifact]
            assert finding.actual not in (None, recorded[artifact])
        elif mutation == "delete":
            assert (finding.expected, finding.actual) == (recorded[artifact], None)
        else:
            assert finding.expected is None
            assert finding.actual is not None

        # Rewriting the manifest re-accepts the mutated tree as the new baseline.
        write_checksums(root)
        assert verify_checksums(root, report_untracked=True) == []
