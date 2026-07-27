"""The report gate stops output when compared runs disagree (R15.2, R15.3).

`write_report` is the point at which report output is produced, so the gate sits
there. These tests build a two-run experiment directory (real
``run_manifest.json`` + ``run_record.json`` bundles) and check that a mismatch
blocks every written artifact while still recording the verification outcome on
the affected runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from jobrec.config import AppConfig
from jobrec.domain.enums import ExperimentVariant
from jobrec.domain.run_record import RunRecord
from jobrec.evaluation.manifest import build_run_manifest
from jobrec.orchestration.feature_flags import FeatureFlags
from jobrec_eval.consistency import ConsistencyError
from jobrec_eval.report import write_report
from tests.eval.test_eval_report_framing import _report_data


def _bundle(exp_dir: Path, variant: str, **record_overrides: Any) -> Path:
    config = AppConfig()
    config.experiment.variant = ExperimentVariant(variant)
    flags = asdict(FeatureFlags.from_config(config))
    flags["variant"] = variant
    fields: dict[str, Any] = {
        "run_id": f"r-{variant}", "session_id": "s1", "candidate_id": "c1",
        "experiment_variant": variant, "started_at": "2026-01-01T00:00:00Z",
        "config_hash": config.config_hash(), "catalog_hash": "cat-hash",
        "prompt_hash": "prompt-hash", "code_version": "0.1.0", "feature_flags": flags,
        "model_manifest": {"provider": "mock", "model": "stub", "mode": "deterministic"},
    }
    fields.update(record_overrides)
    record = RunRecord(**fields)
    run_dir = exp_dir / variant / "s1" / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(record.model_dump_json())
    (run_dir / "run_manifest.json").write_text(json.dumps(
        build_run_manifest(config, record, {"db_version": "PostgreSQL 16.2",
                                            "migration_version": 3})))
    return run_dir


def _run_records(exp_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(exp_dir.rglob("run_record.json"))]


def test_matching_runs_let_the_report_through_and_record_the_flags(tmp_path):
    exp_dir = tmp_path / "exp"
    _bundle(exp_dir, "full")
    _bundle(exp_dir, "no_memory")

    report_path = write_report(_report_data(), tmp_path / "out", experiment_dir=exp_dir)

    assert report_path.exists()
    records = _run_records(exp_dir)
    assert len(records) == 2
    for record in records:
        assert record["consistency_flags"]["consistent"] is True
        # The stored flags round-trip into RunRecord (the field is part of the schema).
        assert RunRecord.model_validate(record).consistency_flags == record["consistency_flags"]


def test_catalog_mismatch_halts_report_generation(tmp_path):
    exp_dir = tmp_path / "exp"
    _bundle(exp_dir, "full")
    _bundle(exp_dir, "no_memory", catalog_hash="other-catalog")
    out_dir = tmp_path / "out"

    with pytest.raises(ConsistencyError) as excinfo:
        write_report(_report_data(), out_dir, experiment_dir=exp_dir)

    assert "catalog_hash" in str(excinfo.value)
    # R15.2: no report artifact was produced.
    assert not (out_dir / "report").exists()
    # R15.3: the failure is recorded on every affected run.
    for record in _run_records(exp_dir):
        assert record["consistency_flags"]["consistent"] is False
        assert record["consistency_flags"]["catalog_hash"] is False


def test_report_output_requires_manifests_to_verify(tmp_path):
    with pytest.raises(ValueError, match="consistency"):
        write_report(_report_data(), tmp_path / "out")

    with pytest.raises(ValueError, match="no run_manifest.json"):
        write_report(_report_data(), tmp_path / "out", experiment_dir=tmp_path / "empty")

    assert not (tmp_path / "out").exists()
