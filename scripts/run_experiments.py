"""Run the scenario suite across experiment variants and export artifacts.

Usage:
    python scripts/run_experiments.py --config configs/experiment_full.yaml \
        --scenarios data/scenarios/scenarios.jsonl \
        --variants full,profile_only,one_shot,no_memory,no_context
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_full.yaml")
    parser.add_argument("--scenarios", default="data/scenarios/scenarios.jsonl")
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    parser.add_argument("--out-dir", default="artifacts/runs")
    parser.add_argument("--variants", default="full,profile_only,one_shot,no_memory,no_context")
    parser.add_argument("--repeats", type=int, default=None,
                        help=("override the config's repeat_count. Only for PILOTS: the "
                              "reported experiment must use the repeat count its config "
                              "declares, because the repeat is what the per-scenario variance "
                              "is estimated from. The override is recorded in the run's "
                              "resolved_config.yaml either way."))
    parser.add_argument("--allow-overwrite", action="store_true",
                        help=("reuse/overwrite an experiment directory that already holds a "
                              "complete experiment (without it, the run refuses instead of "
                              "replacing the existing artifact)"))
    args = parser.parse_args()

    cfg = load_config(args.config, base_dir=str(Path(args.config).parent))
    if args.repeats is not None:
        if args.repeats < 1:
            parser.error("--repeats must be at least 1")
        cfg.experiment.repeat_count = args.repeats
    runner = ExperimentRunner(cfg, args.catalog, args.scenarios, out_dir=args.out_dir)
    manifest = runner.run(args.variants.split(","), allow_overwrite=args.allow_overwrite)
    print(f"experiment_id={manifest['experiment_id']} runs={manifest['run_count']} "
          f"repeats={cfg.experiment.repeat_count}")
    print(f"artifacts -> {manifest['experiment_dir']}")


if __name__ == "__main__":
    main()
