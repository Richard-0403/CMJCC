"""Unit tests for the pre-comparison configuration-consistency gate (R15, R32.7).

The gate reads real ``run_manifest.json`` payloads, so these tests build manifests
with :func:`jobrec.evaluation.manifest.build_run_manifest` rather than hand-rolled
dicts, then perturb one field at a time:

- matching runs pass and every verified field is flagged consistent (R15.1),
- a catalog/prompt/commit mismatch stops the gate (R15.2),
- an ablation pair passes only against its own mechanism's flag group (R32.7),
- the consistency flags are written into each affected manifest and persist to
  disk (R15.3).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields
from functools import cache
from itertools import combinations
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.config import AppConfig
from jobrec.domain.enums import ExperimentVariant
from jobrec.domain.run_record import RunRecord
from jobrec.evaluation.manifest import build_run_manifest
from jobrec.orchestration.feature_flags import CONTEXT_FLAGS, MEMORY_FLAGS, FeatureFlags
from jobrec_eval.consistency import (
    ConsistencyError,
    check_consistency,
    load_run_manifests,
    require_consistent,
    save_run_manifests,
)

_VERSIONS = {"db_version": "PostgreSQL 16.2", "migration_version": 3}


def _config(variant: str = "full") -> AppConfig:
    config = AppConfig()
    config.experiment.variant = ExperimentVariant(variant)
    return config


def _manifest(variant: str = "full", **record_overrides: Any) -> dict[str, Any]:
    config = _config(variant)
    flags = asdict(FeatureFlags.from_config(config))
    flags["variant"] = config.experiment.variant.value
    record_fields: dict[str, Any] = {
        "run_id": f"r-{variant}",
        "session_id": "s1",
        "candidate_id": "c1",
        "experiment_variant": variant,
        "started_at": "2026-01-01T00:00:00Z",
        "config_hash": config.config_hash(),
        "catalog_hash": "cat-hash",
        "prompt_hash": "prompt-hash",
        "code_version": "0.1.0",
        "feature_flags": flags,
        "model_manifest": {"provider": "mock", "model": "stub", "mode": "deterministic"},
    }
    record_fields.update(record_overrides)
    return build_run_manifest(config, RunRecord(**record_fields), _VERSIONS)


def test_matching_runs_pass_and_record_verified_fields():
    manifests = [_manifest(), _manifest()]

    report = check_consistency(manifests)

    assert report.consistent
    assert report.mismatched_fields == ()
    for verified in ("catalog_hash", "prompt_hash", "commit_hash", "model_settings",
                     "config_hash", "feature_flags"):
        assert report.flags[verified] is True
    # Fields R15.1 names that the run manifest does not (yet) record are reported
    # as unverified rather than silently passing as equal.
    assert set(report.unavailable_fields) == {"scenario_hash", "top_k", "pool_size", "seed"}
    # R15.3: the result is written into every inspected manifest.
    for manifest in manifests:
        assert manifest["consistency"]["consistent"] is True
        assert manifest["consistency"]["flags"]["catalog_hash"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("catalog_hash", "other-catalog"), ("prompt_hash", "other-prompt")],
)
def test_hash_mismatch_stops_the_gate(field: str, value: str):
    manifests = [_manifest(), _manifest(**{field: value})]

    with pytest.raises(ConsistencyError) as excinfo:
        require_consistent(manifests)

    assert field in str(excinfo.value)
    assert excinfo.value.report.flags[field] is False
    for manifest in manifests:
        assert manifest["consistency"]["consistent"] is False
        assert field in manifest["consistency"]["mismatched_fields"]


def test_commit_mismatch_stops_the_gate():
    manifests = [_manifest(), _manifest()]
    manifests[1]["commit_hash"] = "deadbeef"

    with pytest.raises(ConsistencyError):
        require_consistent(manifests)


def test_partially_recorded_field_is_a_mismatch():
    manifests = [_manifest(), _manifest()]
    manifests[0]["hashes"]["scenario_hash"] = "scen-hash"

    report = check_consistency(manifests)

    assert not report.consistent
    assert "scenario_hash" in report.mismatched_fields

    manifests[1]["hashes"]["scenario_hash"] = "scen-hash"
    assert check_consistency(manifests).consistent


def test_ablation_pair_passes_only_against_its_own_mechanism():
    pair = [_manifest("full"), _manifest("no_memory")]

    assert require_consistent(copy.deepcopy(pair), MEMORY_FLAGS).consistent

    with pytest.raises(ConsistencyError) as excinfo:
        require_consistent(copy.deepcopy(pair), CONTEXT_FLAGS)
    assert "outside the target mechanism" in str(excinfo.value)


def test_context_ablation_pair_isolates_context_flags():
    pair = [_manifest("full"), _manifest("no_context")]

    assert require_consistent(copy.deepcopy(pair), CONTEXT_FLAGS).consistent

    with pytest.raises(ConsistencyError):
        require_consistent(copy.deepcopy(pair), MEMORY_FLAGS)


def test_repeats_of_one_variant_must_resolve_identical_flags():
    manifests = [_manifest(), _manifest()]
    manifests[1]["feature_flags"]["use_persistent_memory"] = False

    report = check_consistency(manifests)

    assert not report.consistent
    assert report.flags["feature_flags"] is False


def test_flags_persist_to_the_manifest_files(tmp_path):
    for index, variant in enumerate(("full", "no_memory")):
        run_dir = tmp_path / variant / "s1" / str(index)
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(json.dumps(_manifest(variant)))

    manifests = load_run_manifests(tmp_path)
    assert len(manifests) == 2

    report = check_consistency(manifests, MEMORY_FLAGS)
    written = save_run_manifests(manifests)

    assert report.consistent
    assert len(written) == 2
    for path in written:
        payload = json.loads(path.read_text())
        assert payload["consistency"]["consistent"] is True
        assert not any(key.startswith("_") for key in payload)


# --------------------------------------------------------------- Property 22
_PROPERTY_VARIANTS = ("full", "profile_only", "one_shot", "no_memory", "no_context")

#: Behaviour flags (everything on ``FeatureFlags`` except the ``variant`` label).
_BEHAVIOUR_FLAGS = tuple(sorted(f.name for f in fields(FeatureFlags) if f.name != "variant"))

#: How one generated run may deviate from its siblings. ``value`` is the field the
#: gate should name; ``None`` means the perturbation must not produce a mismatch.
_PERTURBATIONS = (
    "none",
    "catalog_hash",
    "prompt_hash",
    "commit_hash",
    "model_settings",
    "config_hash",
    "scenario_hash_partial",
    "scenario_hash_agreeing",
    "flag_flip",
)

_TARGETS = {"none": None, "memory": MEMORY_FLAGS, "context": CONTEXT_FLAGS}


@cache
def _base_manifest(variant: str) -> dict[str, Any]:
    """Cached pristine manifest per variant (building one probes git + platform)."""
    return _manifest(variant)


def _perturb(manifests: list[dict[str, Any]], kind: str, index: int, flag_name: str) -> None:
    """Apply one deviation to ``manifests[index]`` (or to all runs, for agreement)."""
    target = manifests[index]
    if kind == "catalog_hash":
        target["hashes"]["catalog_hash"] = "other-catalog"
    elif kind == "prompt_hash":
        target["hashes"]["prompt_hash"] = "other-prompt"
    elif kind == "commit_hash":
        target["commit_hash"] = "deadbeefdeadbeef"
    elif kind == "model_settings":
        target["api_summary"]["model"] = "other-model"
    elif kind == "config_hash":
        target["hashes"]["config_hash"] = "other-config"
    elif kind == "scenario_hash_partial":
        target["hashes"]["scenario_hash"] = "scen-hash"
    elif kind == "scenario_hash_agreeing":
        for manifest in manifests:
            manifest["hashes"]["scenario_hash"] = "scen-hash"
    elif kind == "flag_flip":
        flags = target["feature_flags"]
        flags[flag_name] = not flags[flag_name]


def _expected_field_mismatch(kind: str, index: int, variants: list[str]) -> str | None:
    """The field the gate must flag, derived from the generated intent alone."""
    run_count = len(variants)
    if kind in {"catalog_hash", "prompt_hash", "commit_hash", "model_settings"}:
        # Compared across every run: a lone deviation only shows up with a peer.
        return kind if run_count >= 2 else None
    if kind == "scenario_hash_partial":
        # Recorded by one run only: partial provenance cannot be verified.
        return "scenario_hash" if run_count >= 2 else None
    if kind == "config_hash":
        # Variant-scoped: only compared against repeats of the same variant.
        return "config_hash" if variants.count(variants[index]) >= 2 else None
    return None


def _expected_flags_ok(
    manifests: list[dict[str, Any]],
    target_flag_set: set[str] | None,
) -> bool:
    """R32.7 stated directly over the flag payloads the test itself constructed."""
    for left, right in combinations(manifests, 2):
        left_flags, right_flags = left["feature_flags"], right["feature_flags"]
        diff = {name for name in _BEHAVIOUR_FLAGS if left_flags[name] != right_flags[name]}
        if left_flags["variant"] == right_flags["variant"]:
            ok = not diff
        elif target_flag_set is None:
            ok = True
        else:
            ok = bool(diff) and diff <= target_flag_set
        if not ok:
            return False
    return True


# Feature: cmjcc-experiment-readiness, Property 22: The consistency gate proceeds iff all
# compared runs match
@settings(max_examples=100, deadline=None)
@given(
    variants=st.lists(st.sampled_from(_PROPERTY_VARIANTS), min_size=1, max_size=4),
    kind=st.sampled_from(_PERTURBATIONS),
    raw_index=st.integers(min_value=0, max_value=3),
    flag_name=st.sampled_from(_BEHAVIOUR_FLAGS),
    target_key=st.sampled_from(sorted(_TARGETS)),
)
def test_property_consistency_gate_proceeds_iff_all_compared_runs_match(
    variants: list[str],
    kind: str,
    raw_index: int,
    flag_name: str,
    target_key: str,
) -> None:
    """``require_consistent`` raises exactly when a compared field or the flag rule breaks.

    **Validates: Requirements 15.1, 15.2, 32.7**
    """
    index = raw_index % len(variants)
    target_flag_set = _TARGETS[target_key]
    manifests = [copy.deepcopy(_base_manifest(variant)) for variant in variants]
    _perturb(manifests, kind, index, flag_name)

    mismatched_field = _expected_field_mismatch(kind, index, variants)
    flags_ok = _expected_flags_ok(manifests, target_flag_set)
    expected_stop = mismatched_field is not None or not flags_ok

    if expected_stop:
        with pytest.raises(ConsistencyError) as excinfo:
            require_consistent(manifests, target_flag_set)
        report = excinfo.value.report
    else:
        report = require_consistent(manifests, target_flag_set)
        assert report.consistent

    # Both directions of the biconditional, plus the reason the gate gives.
    assert report.consistent is not expected_stop
    if mismatched_field is not None:
        assert mismatched_field in report.mismatched_fields
    else:
        assert report.mismatched_fields == ()
    assert report.flags["feature_flags"] is flags_ok

    # A field no manifest records is unavailable, never a mismatch.
    assert {"top_k", "pool_size", "seed"} <= set(report.unavailable_fields)
    assert not set(report.mismatched_fields) & set(report.unavailable_fields)
    # R15.3: the verdict is stamped on every compared run either way.
    for manifest in manifests:
        assert manifest["consistency"]["consistent"] is not expected_stop
