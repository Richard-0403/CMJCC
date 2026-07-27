"""Contract tests: the run manifest carries every reproducibility key (R11.2/R11.3).

`run_manifest.json` is the artifact a reviewer uses to re-create a run, so its shape is a
contract: the three content hashes, the resolved feature flags, the environment summary and
the db/migration versions must always be present (and JSON-serializable), never merely
"some non-empty dict".
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from jobrec.config import AppConfig
from jobrec.domain.run_record import RunRecord
from jobrec.evaluation.manifest import build_run_manifest
from jobrec.orchestration.feature_flags import FeatureFlags

#: Live probe returned by ``SqlRepository.versions()``.
_VERSIONS = {"db_version": "PostgreSQL 16.2", "migration_version": 3}

#: Every top-level section the manifest must expose.
_REQUIRED_TOP_LEVEL = {
    "generated_at",
    "commit_hash",
    "code_version",
    "python",
    "host",
    "dependencies",
    "hashes",
    "feature_flags",
    "api_summary",
    "versions",
}


def _resolved_flags(config: AppConfig) -> dict[str, Any]:
    """Resolve feature flags the way the orchestrator records them on a RunRecord."""
    flags = FeatureFlags.from_config(config)
    resolved = asdict(flags)
    resolved["variant"] = flags.variant.value
    return resolved


def _run_record(**overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {
        "run_id": "r1",
        "session_id": "s1",
        "candidate_id": "c1",
        "experiment_variant": "full",
        "started_at": "2026-01-01T00:00:00Z",
        "config_hash": "cfg-hash",
        "catalog_hash": "cat-hash",
        "prompt_hash": "prompt-hash",
        "code_version": "0.1.0",
        "model_manifest": {
            "provider": "mock",
            "model": "deterministic-stub",
            "mode": "deterministic",
            "base_url": "https://api.example.com/v1/chat?api_key=SUPERSECRET",
        },
        "db_version": "record-db-version",
        "migration_version": 1,
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_manifest_contains_all_reproducibility_sections():
    config = AppConfig()
    flags = _resolved_flags(config)
    record = _run_record(feature_flags=flags)

    manifest = build_run_manifest(config, record, _VERSIONS)

    assert _REQUIRED_TOP_LEVEL <= set(manifest)

    # The three content hashes, carried through verbatim from the run record.
    assert manifest["hashes"] == {
        "config_hash": "cfg-hash",
        "catalog_hash": "cat-hash",
        "prompt_hash": "prompt-hash",
    }

    # The resolved feature flags, not the raw config section.
    assert manifest["feature_flags"] == flags
    assert manifest["feature_flags"]["variant"] == config.experiment.variant.value
    assert "use_persistent_memory" in manifest["feature_flags"]

    # DB / migration versions from the live probe.
    assert manifest["versions"] == {
        "db_version": "PostgreSQL 16.2",
        "migration_version": 3,
    }

    # Environment summary: interpreter, host and dependency versions.
    assert manifest["code_version"] == "0.1.0"
    assert {"version", "implementation", "executable"} <= set(manifest["python"])
    assert manifest["python"]["version"]
    assert {"system", "machine", "cpu_count", "total_memory_bytes"} <= set(manifest["host"])
    assert manifest["dependencies"]["pydantic"], "pydantic version must be recorded"
    assert {"provider", "model", "mode", "base_url_host"} <= set(manifest["api_summary"])
    assert manifest["api_summary"]["model"] == "deterministic-stub"


def test_manifest_versions_fall_back_to_run_record_when_probe_missing():
    manifest = build_run_manifest(AppConfig(), _run_record(), None)

    assert manifest["versions"] == {
        "db_version": "record-db-version",
        "migration_version": 1,
    }


def test_manifest_versions_keys_present_even_when_unknown():
    record = _run_record(db_version=None, migration_version=None)

    manifest = build_run_manifest(AppConfig(), record, {})

    assert manifest["versions"] == {"db_version": None, "migration_version": None}


def test_manifest_is_json_serializable_and_redacts_api_credentials():
    record = _run_record(feature_flags=_resolved_flags(AppConfig()))

    manifest = build_run_manifest(AppConfig(), record, _VERSIONS)
    serialized = json.dumps(manifest)
    round_tripped = json.loads(serialized)

    assert round_tripped["hashes"] == manifest["hashes"]
    assert round_tripped["feature_flags"] == manifest["feature_flags"]
    assert round_tripped["versions"] == manifest["versions"]

    # Only the host survives from any base URL; no key material anywhere.
    assert manifest["api_summary"]["base_url_host"] == "api.example.com"
    assert "SUPERSECRET" not in serialized
    assert "api_key" not in serialized
