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

#: Tripwire. The sealed official pair (exp-e748800507ef / exp-6db1e87daed5) was graded
#: against exactly these bytes, and the canonical oracle's inputs_fingerprint is derived
#: from them. A change here invalidates that grading, so it must be a deliberate edit
#: that updates this constant, never a side effect of running a tool.
AUTHORITATIVE_SHA256 = "775f5975d8ecac8d0879871c0d0046955a633a42f0c4f851a1c772a3797d1af4"
AUTHORITATIVE_BYTES = 38309

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
])
def test_authoritative_paths_are_refused(requested: str) -> None:
    """Asking for the authoritative path explicitly must fail, not just be non-default."""
    module = _load_builder()
    with pytest.raises(module.ProtectedOutputError):
        module.resolve_output(requested)


def test_a_protected_basename_is_refused_from_any_directory(tmp_path: Path) -> None:
    """The guard is on the name, so a different cwd or a copy of the tree is covered too."""
    module = _load_builder()
    with pytest.raises(module.ProtectedOutputError):
        module.resolve_output(tmp_path / "scenarios.jsonl")


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
def test_authoritative_file_is_unchanged() -> None:
    raw = AUTHORITATIVE.read_bytes()
    assert len(raw) == AUTHORITATIVE_BYTES
    assert hashlib.sha256(raw).hexdigest() == AUTHORITATIVE_SHA256, (
        "evaluation/data/scenarios.jsonl changed. The sealed official pair was graded "
        "against the previous bytes, so this must be a deliberate edit that also updates "
        "AUTHORITATIVE_SHA256 and re-derives the canonical oracle."
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
