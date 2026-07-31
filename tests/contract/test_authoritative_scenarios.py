"""The authoritative evaluation scenario set must not be reachable by a builder run.

``evaluation/data/scenarios.jsonl`` carries the 42 hand-reviewed ``reference``
declarations that canonical oracle v3.0.0 is a pure function of.
``scripts/build_eval_scenarios.py`` emits scenario TEXT and no ``reference`` block at
all, and its ``--output`` used to default to that very path -- so running it with no
arguments replaced declared ground truth with a file that had none, silently.

These tests pin both halves of the fix: the builder refuses the authoritative path even
when it is asked for explicitly, and the authoritative file itself is byte-for-byte what
the sealed experiments were graded against.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts" / "build_eval_scenarios.py"
AUTHORITATIVE = REPO_ROOT / "evaluation" / "data" / "scenarios.jsonl"

#: Tripwire over the CONTENT, with line endings normalised to LF first.
#:
#: Hashing the raw bytes would make this test platform-specific and it would have failed
#: on CI: nothing pins this file's line endings, so a Windows checkout with
#: core.autocrlf=true holds CRLF (38309 bytes, sha256 775f5975...) while a Linux checkout
#: holds LF (38267 bytes, sha256 571bcec9...). Normalising first pins what actually
#: matters -- the scenario text and its declared references -- on every platform.
AUTHORITATIVE_SHA256_LF = "571bcec9f2c4c2f80dddbad649661b349842298574a0e13c98e97c23c4ba9dbb"
AUTHORITATIVE_BYTES_LF = 38267

#: The identity the sealed experiments actually recorded, from ``stable_hash`` over the
#: PARSED scenarios. It matches neither raw byte form, which is the point: experiment
#: identity is already line-ending independent, and pinning it here ties this test to the
#: value the official pair was graded under rather than to a checkout detail.
SEALED_SCENARIOS_HASH = "5f08208ff840aa48d2d2f0954c2c0c318a598b966f17c666a3cc6c1568f4d428"

EXPECTED_SCENARIO_COUNT = 42


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_eval_scenarios", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenarios() -> list[dict]:
    return [json.loads(line) for line in
            AUTHORITATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------- the builder's guard
def test_default_output_is_a_draft() -> None:
    module = _load_builder()
    assert module.DEFAULT_OUTPUT == "evaluation/data/scenarios_draft.jsonl"
    assert Path(module.DEFAULT_OUTPUT).name != AUTHORITATIVE.name


def test_no_argument_run_writes_only_the_draft(tmp_path: Path) -> None:
    """A bare run must produce a draft and leave the authoritative file untouched."""
    before = AUTHORITATIVE.read_bytes()
    draft = REPO_ROOT / "evaluation" / "data" / "scenarios_draft.jsonl"
    draft_existed = draft.exists()
    draft_before = draft.read_bytes() if draft_existed else None
    try:
        result = subprocess.run([sys.executable, str(BUILDER)], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "DRAFT" in result.stdout
        assert draft.exists(), "the bare run produced no draft"
        assert AUTHORITATIVE.read_bytes() == before, "the authoritative file was modified"
    finally:
        if draft_before is not None:
            draft.write_bytes(draft_before)
        elif draft.exists():
            draft.unlink()


@pytest.mark.parametrize("requested", [
    "evaluation/data/scenarios.jsonl",
    "./evaluation/data/scenarios.jsonl",
    "evaluation/data/../data/scenarios.jsonl",
    "data/scenarios/scenarios.jsonl",
    # Case variants. On macOS the filesystem is case-insensitive but case-preserving, so
    # these resolve with the requested case and open the SAME file -- a case-sensitive
    # name check would let them through and the authoritative set would be overwritten.
    "evaluation/data/SCENARIOS.JSONL",
    "evaluation/data/Scenarios.JsonL",
    "DATA/SCENARIOS/scenarios.JSONL",
])
def test_authoritative_paths_are_refused(requested: str) -> None:
    """Asking for the authoritative path explicitly must fail, not just be non-default."""
    module = _load_builder()
    with pytest.raises(module.ProtectedOutputError):
        module.resolve_output(requested)


@pytest.mark.parametrize("name", ["scenarios.jsonl", "SCENARIOS.JSONL", "Scenarios.Jsonl"])
def test_a_protected_basename_is_refused_from_any_directory(tmp_path: Path,
                                                            name: str) -> None:
    """The guard is on the name, so a different cwd or a copy of the tree is covered too."""
    module = _load_builder()
    with pytest.raises(module.ProtectedOutputError):
        module.resolve_output(tmp_path / name)


def test_refusing_the_authoritative_path_exits_non_zero() -> None:
    before = AUTHORITATIVE.read_bytes()
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", "evaluation/data/scenarios.jsonl"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode != 0, "the builder accepted the authoritative path"
    assert "refusing to write" in result.stderr
    assert AUTHORITATIVE.read_bytes() == before


def test_a_draft_path_is_accepted(tmp_path: Path) -> None:
    module = _load_builder()
    target = tmp_path / "scenarios_draft.jsonl"
    assert module.resolve_output(target) == target


def test_the_builder_still_emits_no_reference_block() -> None:
    """The premise of the guard. If the builder ever learns to emit references this test
    fails, and the guard should be revisited rather than silently outgrown."""
    assert '"reference"' not in BUILDER.read_text(encoding="utf-8")


# ------------------------------------------------------ the authoritative file itself
def test_authoritative_content_is_unchanged() -> None:
    """Platform-independent: normalise line endings, then hash."""
    normalised = AUTHORITATIVE.read_bytes().replace(b"\r\n", b"\n")
    assert len(normalised) == AUTHORITATIVE_BYTES_LF
    assert hashlib.sha256(normalised).hexdigest() == AUTHORITATIVE_SHA256_LF, (
        "evaluation/data/scenarios.jsonl changed. The sealed official pair was graded "
        "against the previous content, so this must be a deliberate edit that also "
        "updates AUTHORITATIVE_SHA256_LF and re-derives the canonical oracle."
    )


def test_scenario_identity_matches_the_sealed_experiments() -> None:
    """The hash the runner records, computed the way the runner computes it.

    ``stable_hash`` runs over the parsed scenarios, so this is immune to line endings and
    to key ordering in the file -- it changes only when the scenario content changes.
    """
    from jobrec.utils.hashing import stable_hash

    assert stable_hash(_scenarios()) == SEALED_SCENARIOS_HASH, (
        "the scenario set no longer hashes to the value recorded in the sealed "
        "experiment manifests, so the official pair's grading no longer applies to it."
    )


def test_authoritative_file_has_42_unique_scenario_ids() -> None:
    scenarios = _scenarios()
    ids = [s["scenario_id"] for s in scenarios]
    assert len(scenarios) == EXPECTED_SCENARIO_COUNT
    assert len(set(ids)) == EXPECTED_SCENARIO_COUNT, "duplicate scenario_id"


def test_every_scenario_declares_a_reference() -> None:
    missing = [s["scenario_id"] for s in _scenarios() if not s.get("reference")]
    assert not missing, f"scenarios without a declared reference: {missing}"


def test_clarification_answers_are_complete() -> None:
    """Delegates to the runner's own assertion, so the gate cannot drift from the check
    the official experiment enforces at start-up."""
    from jobrec.evaluation.experiment_runner import assert_clarification_answers_declared

    assert_clarification_answers_declared(_scenarios())
