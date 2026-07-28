"""Shared fixtures for the annotation-tool tests: one real experiment, built once.

The annotation data layer is only meaningful against REAL run bundles (deduplicating claims,
resolving evidence ids, enumerating the pairs actually returned), so these tests drive the
deterministic :class:`~jobrec.evaluation.experiment_runner.ExperimentRunner` over the small CI
scenario subset instead of hand-building bundles. The runner alone is used rather than the
whole pipeline: item building reads ``_runs/<experiment_id>/...`` and nothing else, so metrics,
bootstrap and plots would only add runtime.

Every rater id these tests use is prefixed ``SYNTHETIC-`` and every label is invented by the
test. No fixture here may be mistaken for a collected human label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner
from jobrec_eval.loaders import RunBundle, load_bundles

CONFIG = "configs/experiment_full.yaml"
SCENARIOS_SUBSET = "evaluation/data/scenarios_subset.jsonl"
CATALOG = "data/processed/jobs.jsonl"

#: Two variants are enough to prove claim deduplication collapses identical claims ACROSS
#: variants, which is the property the export's occurrence expansion depends on.
ANNOTATION_VARIANTS = ["full", "no_memory"]


@dataclass(frozen=True)
class Experiment:
    """A real experiment directory plus its loaded bundles and the inputs behind it."""

    experiment_id: str
    experiment_dir: Path
    scenarios_path: str
    catalog_path: str
    bundles: list[RunBundle]


@pytest.fixture(scope="session")
def annotation_experiment(tmp_path_factory) -> Experiment:
    """Run the deterministic experiment once and load its bundles."""
    out_root = tmp_path_factory.mktemp("annotation-ui-runs")
    cfg = load_config(CONFIG, base_dir="configs")
    cfg.experiment.repeat_count = 1
    runner = ExperimentRunner(cfg, CATALOG, SCENARIOS_SUBSET, out_dir=str(out_root))
    manifest = runner.run(ANNOTATION_VARIANTS)
    experiment_dir = Path(manifest["experiment_dir"])
    return Experiment(
        experiment_id=manifest["experiment_id"],
        experiment_dir=experiment_dir,
        scenarios_path=SCENARIOS_SUBSET,
        catalog_path=CATALOG,
        bundles=load_bundles(experiment_dir),
    )
