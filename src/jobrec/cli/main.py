"""CLI entry point. The CLI and API call the same AppService."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..app_service import AppService, build_default_service
from ..config import load_config

app = typer.Typer(add_completion=False, help="CMJCC conversational job recommendation prototype")


def _load(config_path: str):
    base = str(Path(config_path).parent) or "configs"
    return load_config(config_path, base_dir=base)


@app.command("prepare-catalog")
def prepare_catalog(
    input: str = typer.Option("data/raw/jobs.csv", help="Raw catalog CSV"),
    out_dir: str = typer.Option("data/processed", help="Output directory"),
    snapshot_id: str = typer.Option("catalog-2026-01-v1"),
    reference_date: str = typer.Option("2026-01-01"),
) -> None:
    """Normalise a raw catalog CSV into jobs.jsonl + manifest."""
    import csv

    from ..catalog import build_manifest, normalize_job, write_catalog, write_json

    with open(input) as fh:
        rows = list(csv.DictReader(fh))
    jobs = [normalize_job(r, snapshot_id) for r in rows]
    write_catalog(jobs, Path(out_dir) / "jobs.jsonl")
    manifest = build_manifest(jobs, snapshot_id, [Path(input).name], reference_date)
    write_json(manifest, Path(out_dir) / "catalog_manifest.json")
    typer.echo(f"Prepared {len(jobs)} jobs -> {out_dir}/jobs.jsonl (hash {manifest['catalog_hash'][:12]})")


@app.command("validate-catalog")
def validate_catalog(catalog: str = typer.Option("data/processed/jobs.jsonl")) -> None:
    """Validate that every catalog record parses against the schema."""
    from ..catalog import load_catalog

    jobs = load_catalog(catalog)
    typer.echo(f"OK: {len(jobs)} valid job postings")


@app.command("build-index")
def build_index_cmd(
    config: str = typer.Option("configs/base.yaml"),
    catalog: str = typer.Option("data/processed/jobs.jsonl"),
    out_dir: str = typer.Option("artifacts/indexes"),
) -> None:
    """Build a retrieval index manifest for the catalog."""
    from ..retrieval.index_builder import build_index

    manifest = build_index(catalog, out_dir)
    typer.echo(f"Index built: {manifest['record_count']} records, hash {manifest['catalog_hash'][:12]}")


@app.command("recommend")
def recommend(
    profile: str = typer.Option(..., help="Path to a candidate profile JSON"),
    query: str = typer.Option(..., help="Candidate utterance"),
    variant: str = typer.Option("full"),
    config: str = typer.Option("configs/experiment_full.yaml"),
    catalog: str = typer.Option("data/processed/jobs.jsonl"),
    no_db: bool = typer.Option(True, help="Use in-memory storage (no PostgreSQL)"),
) -> None:
    """One-shot recommendation from a profile JSON and a single utterance."""
    cfg = _load(config)
    svc = build_default_service(cfg, catalog_path=catalog, use_database=(not no_db))
    prof = json.loads(Path(profile).read_text())
    cand = svc.create_candidate(prof)
    session_id = svc.create_session(cand.candidate_id, variant)
    result = svc.process_turn(session_id, query)
    typer.echo(f"[{result.response.response_type}] run={result.run_record.run_id}")
    typer.echo(result.response.message)
    typer.echo(f"\n(claims: {len(result.response.claims)}, dropped: {len(result.dropped_claims)})")


@app.command("chat")
def chat(
    candidate: str = typer.Option(..., help="Path to candidate profile JSON"),
    variant: str = typer.Option("full"),
    query: str = typer.Option(None, help="A single utterance (non-interactive)"),
    config: str = typer.Option("configs/experiment_full.yaml"),
    catalog: str = typer.Option("data/processed/jobs.jsonl"),
) -> None:
    """Chat with the recommender. Provide --query for a single non-interactive turn."""
    cfg = _load(config)
    svc = build_default_service(cfg, catalog_path=catalog, use_database=False)
    prof = json.loads(Path(candidate).read_text())
    cand = svc.create_candidate(prof)
    session_id = svc.create_session(cand.candidate_id, variant)
    if query:
        result = svc.process_turn(session_id, query)
        typer.echo(result.response.message)
        return
    typer.echo("Enter messages (Ctrl-D to exit).")
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        result = svc.process_turn(session_id, line)
        typer.echo(f"\n[{result.response.response_type}]")
        typer.echo(result.response.message)


@app.command("run-scenarios")
def run_scenarios(
    config: str = typer.Option("configs/experiment_full.yaml"),
    scenarios: str = typer.Option("data/scenarios/scenarios.jsonl"),
    catalog: str = typer.Option("data/processed/jobs.jsonl"),
    out_dir: str = typer.Option("artifacts/runs"),
    variants: str = typer.Option("full,profile_only,one_shot,no_memory,no_context"),
) -> None:
    """Run all scenarios across the given experiment variants and export artifacts."""
    from ..evaluation.experiment_runner import ExperimentRunner

    cfg = _load(config)
    runner = ExperimentRunner(cfg, catalog_path=catalog, scenarios_path=scenarios, out_dir=out_dir)
    manifest = runner.run(variants.split(","))
    typer.echo(f"Ran {manifest['run_count']} runs across {len(manifest['variants'])} variants")
    typer.echo(f"Artifacts: {manifest['experiment_dir']}")


@app.command("export-run")
def export_run(
    run_id: str = typer.Option(...),
    output: str = typer.Option(...),
    config: str = typer.Option("configs/experiment_full.yaml"),
    catalog: str = typer.Option("data/processed/jobs.jsonl"),
) -> None:
    """Export a stored run bundle to a directory."""
    cfg = _load(config)
    svc = build_default_service(cfg, catalog_path=catalog)
    run = svc.get_run(run_id, include_states=True, include_evidence=True, include_handoffs=True)
    if run is None:
        typer.echo("run not found", err=True)
        raise typer.Exit(code=1)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_bundle.json").write_text(json.dumps(run, indent=2, default=str))
    typer.echo(f"Exported run {run_id} -> {out}/run_bundle.json")


if __name__ == "__main__":
    app()
