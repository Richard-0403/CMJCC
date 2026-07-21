"""Batch experiment runner.

Runs a fixed scenario set across the five experiment variants, using the same
catalog snapshot, prompts, model settings and top-k, and writes a full artifact
bundle per run plus a batch manifest, index, failures list and checksums.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..app_service import AppService
from ..config import AppConfig
from ..utils.hashing import stable_hash
from ..utils.time import to_iso, utcnow
from .exporters import write_run_bundle


def load_scenarios(path: str | Path) -> list[dict]:
    path = Path(path)
    scenarios: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


class ExperimentRunner:
    """Runs scenarios x variants x repeats and exports artifacts."""

    def __init__(
        self,
        config: AppConfig,
        catalog_path: str,
        scenarios_path: str,
        out_dir: str = "artifacts/runs",
    ) -> None:
        self.config = config
        self.catalog_path = catalog_path
        self.scenarios = load_scenarios(scenarios_path)
        self.out_dir = Path(out_dir)

    def run(self, variants: list[str]) -> dict[str, Any]:
        experiment_id = "exp-" + stable_hash({
            "variants": variants, "scenarios": [s["scenario_id"] for s in self.scenarios],
            "config": self.config.config_hash(),
        })[:12]
        exp_dir = self.out_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        index_rows: list[dict] = []
        failures: list[dict] = []
        repeat = self.config.experiment.repeat_count

        for variant in variants:
            for scenario in self.scenarios:
                for run_index in range(repeat):
                    row, failure = self._run_one(variant, scenario, run_index, exp_dir)
                    index_rows.append(row)
                    if failure:
                        failures.append(failure)

        self._write_index(exp_dir, index_rows)
        self._write_failures(exp_dir, failures)
        manifest = {
            "experiment_id": experiment_id,
            "experiment_dir": str(exp_dir),
            "variants": variants,
            "scenario_count": len(self.scenarios),
            "repeat_count": repeat,
            "run_count": len(index_rows),
            "config_hash": self.config.config_hash(),
            "created_at": to_iso(utcnow()),
        }
        (exp_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
        self._write_checksums(exp_dir)
        return manifest

    def _run_one(self, variant, scenario, run_index, exp_dir):
        cfg = self.config.model_copy(deep=True)
        from ..domain.enums import ExperimentVariant

        cfg.experiment.variant = ExperimentVariant(variant)
        # Deterministic in-memory run (no external DB dependency for experiments).
        svc = AppService(cfg, self.catalog_path)
        profile = dict(scenario["profile"])
        profile.setdefault("candidate_id", scenario["scenario_id"] + "-cand")
        cand = svc.create_candidate(profile)
        session_id = svc.create_session(cand.candidate_id, variant)

        last_result = None
        for text in scenario.get("turns", []):
            last_result = svc.process_turn(session_id, text, scenario_id=scenario["scenario_id"])

        run_dir = exp_dir / variant / scenario["scenario_id"] / str(run_index)
        write_run_bundle(last_result, run_dir, cfg)

        rr = last_result.run_record
        decision = last_result.decision
        row = {
            "experiment_variant": variant,
            "scenario_id": scenario["scenario_id"],
            "run_index": run_index,
            "run_id": rr.run_id,
            "success": rr.success,
            "response_type": last_result.response.response_type,
            "no_match": bool(decision.no_match) if decision else "",
            "returned": len(decision.selected_job_ids) if decision else 0,
            "eligible": sum(1 for e in decision.eligibility_results if e.eligible) if decision else 0,
            "claims": len(last_result.response.claims),
            "dropped_claims": len(last_result.dropped_claims),
            "total_latency_ms": rr.total_latency_ms,
            "run_dir": str(run_dir),
        }
        failure = None if rr.success else {"run_id": rr.run_id, "variant": variant,
                                           "scenario_id": scenario["scenario_id"],
                                           "failure_code": rr.failure_code}
        return row, failure

    def _write_index(self, exp_dir: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with (exp_dir / "runs_index.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_failures(self, exp_dir: Path, failures: list[dict]) -> None:
        fields = ["run_id", "variant", "scenario_id", "failure_code"]
        with (exp_dir / "failures.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures)

    def _write_checksums(self, exp_dir: Path) -> None:
        lines = []
        for path in sorted(exp_dir.rglob("*.json")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(exp_dir)}")
        (exp_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")
