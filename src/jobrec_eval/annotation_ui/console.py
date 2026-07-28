"""Command line front end for the headless annotation data layer.

    python -m jobrec_eval.annotation_ui build --experiment-dir <out_root>/_runs/<experiment_id> \
        --scenarios evaluation/data/scenarios.jsonl --catalog data/processed/jobs.jsonl \
        --annotation-dir evaluation/annotation --raters r1,r2 --seed 2026

    python -m jobrec_eval.annotation_ui status --annotation-dir evaluation/annotation
    python -m jobrec_eval.annotation_ui disagreements --annotation-dir evaluation/annotation
    python -m jobrec_eval.annotation_ui adjudicate --annotation-dir evaluation/annotation \
        --item-key rel::SC-A-01::job-0088 --adjudicator lead --label 2 --reason "..."
    python -m jobrec_eval.annotation_ui export --annotation-dir evaluation/annotation \
        --out-dir evaluation/data --release-dir final_release/human_annotations
    python -m jobrec_eval.annotation_ui serve --annotation-dir evaluation/annotation

Everything a rater does goes through
:class:`~jobrec_eval.annotation_ui.store.AnnotationStore`, so the web UI
(:mod:`~jobrec_eval.annotation_ui.app`) stays a presentation layer and every step above stays
reproducible from a terminal without it.

``serve`` PRINTS the URL and the uvicorn command; it deliberately does not start a server. A
long-running dev server belongs in the operator's own terminal where they can see its log and
stop it, not inside a tool invocation. Note the security warning it prints: the web UI has NO
authentication and must stay bound to a loopback address.

Named ``console`` rather than ``cli`` so nothing here can shadow :mod:`jobrec_eval.cli`, which
is the evaluation pipeline entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .app import DEFAULT_HOST, DEFAULT_PORT, NO_AUTH_NOTICE, is_loopback_host
from .assignment import assign_two_raters
from .export import export_annotations
from .loader import build_items
from .store import (
    DB_FILENAME,
    KIND_CLAIM,
    KIND_RELEVANCE,
    META_CATALOG_PATH,
    META_EXPERIMENT_DIR,
    META_EXPERIMENT_ID,
    META_SCENARIOS_PATH,
    open_store,
)

#: Default assignment seed. Matches the pipeline's ``--bootstrap-seed`` default so every
#: seeded artifact of this thesis carries the same number unless one is passed explicitly.
DEFAULT_SEED = 2026


def build_command(args: argparse.Namespace) -> dict:
    """Build items from real bundles, register the raters and store the assignment plan."""
    result = build_items(args.experiment_dir, args.scenarios, args.catalog,
                         oracle_labels=args.oracle_labels)
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    with open_store(args.annotation_dir) as store:
        store.register_raters(raters)
        store.add_items(result.all_items)
        plan = assign_two_raters(store.item_keys(), raters, args.seed)
        store.save_assignment_plan(plan)
        store.set_meta({
            META_EXPERIMENT_ID: Path(args.experiment_dir).name,
            META_EXPERIMENT_DIR: str(args.experiment_dir),
            META_SCENARIOS_PATH: str(args.scenarios),
            META_CATALOG_PATH: str(args.catalog),
        })
        return {
            "annotation_store": str(store.path),
            "stats": result.stats.to_dict() if result.stats else {},
            "raters": list(store.raters()),
            "assignment_seed": args.seed,
            "assignment_counts": store.assignment_counts(),
            "max_load_imbalance": plan.max_load_imbalance,
        }


def status_command(args: argparse.Namespace) -> dict:
    """Per-rater progress plus how many items are ready for export or adjudication."""
    with open_store(args.annotation_dir, create=False) as store:
        return {
            "items": {KIND_RELEVANCE: store.item_count(KIND_RELEVANCE),
                      KIND_CLAIM: store.item_count(KIND_CLAIM)},
            "raters": {
                rater: {
                    "assigned": progress.assigned, "completed": progress.completed,
                    "remaining": progress.remaining,
                    "fraction_complete": round(progress.fraction_complete, 4),
                    "median_duration_ms": progress.median_duration_ms,
                }
                for rater, progress in ((r, store.progress(r)) for r in store.raters())},
            "both_slots_complete": len(store.completed_item_keys()),
            "disagreements": len(store.disagreements()),
            "unadjudicated_disagreements": len(store.disagreements(unadjudicated_only=True)),
            "annotation_effort": store.annotation_effort(),
        }


def disagreements_command(args: argparse.Namespace) -> dict:
    """The adjudication worklist."""
    with open_store(args.annotation_dir, create=False) as store:
        rows = store.disagreements(kind=args.kind, unadjudicated_only=args.unadjudicated_only)
        return {"count": len(rows), "disagreements": [
            {"item_key": d.item_key, "kind": d.kind, "scenario_id": d.scenario_id,
             "job_id": d.job_id, "claim_id": d.claim_id,
             "rater_1": {"rater_id": d.slot_1_rater, "label": d.slot_1_label},
             "rater_2": {"rater_id": d.slot_2_rater, "label": d.slot_2_label},
             "adjudicated_label": d.adjudicated_label, "adjudicator": d.adjudicator}
            for d in rows]}


def adjudicate_command(args: argparse.Namespace) -> dict:
    """Record one adjudicated verdict."""
    with open_store(args.annotation_dir, create=False) as store:
        verdict = store.record_adjudication(args.item_key, args.adjudicator, args.label,
                                            args.reason)
        return {"item_key": verdict.item_key, "final_label": verdict.final_label,
                "adjudicator": verdict.adjudicator, "created_at": verdict.created_at}


def export_command(args: argparse.Namespace) -> dict:
    """Write the two CSVs, the archive dump and the manifest."""
    with open_store(args.annotation_dir, create=False) as store:
        result = export_annotations(store, args.out_dir, release_dir=args.release_dir)
        return {
            "relevance_csv": str(result.relevance_path),
            "claims_csv": str(result.claims_path),
            "dump": str(result.dump_path),
            "manifest": str(result.manifest_path),
            "row_counts": result.row_counts,
            "sha256": result.hashes,
            "skipped_incomplete": {
                KIND_RELEVANCE: result.incomplete_count(KIND_RELEVANCE),
                KIND_CLAIM: result.incomplete_count(KIND_CLAIM),
            },
            "counts": result.manifest["counts"],
        }


def serve_command(args: argparse.Namespace) -> dict:
    """Print the URL and the uvicorn command for the web UI. Does NOT start a server.

    Starting a blocking dev server from inside a tool invocation hides its log and leaves a
    process nobody owns; the operator runs the printed command in their own terminal instead.

    Refuses a non-loopback ``--host`` unless ``--allow-remote-host`` is passed, and shouts about
    it either way: the UI has no authentication, so a routable bind hands anybody on the network
    the ability to write labels as any rater.
    """
    annotation_dir = Path(args.annotation_dir)
    store_path = annotation_dir / DB_FILENAME
    if not store_path.is_file():
        raise FileNotFoundError(
            f"no annotation store at {store_path}; run the build subcommand first")
    host, port = args.host, int(args.port)
    loopback = is_loopback_host(host)
    if not loopback and not args.allow_remote_host:
        raise SystemExit(
            f"refusing to print a command binding to {host!r}: the annotation UI has NO "
            f"authentication, so anybody able to reach that address could label as any rater. "
            f"Bind {DEFAULT_HOST} instead, or pass --allow-remote-host if you have an "
            f"authenticating proxy in front of it and accept the risk.")

    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{display_host}:{port}/"
    env = {"CMJCC_ANNOTATION_DIR": str(annotation_dir)}
    if args.export_dir:
        env["CMJCC_ANNOTATION_EXPORT_DIR"] = str(args.export_dir)
    if args.release_dir:
        env["CMJCC_ANNOTATION_RELEASE_DIR"] = str(args.release_dir)
    uvicorn_command = (
        f"python -m uvicorn jobrec_eval.annotation_ui.app:app_factory --factory "
        f"--host {host} --port {port}")
    powershell = ["# in the repo root, with the venv active"]
    powershell += [f'$env:{key} = "{value}"' for key, value in env.items()]
    powershell.append(f".venv\\Scripts\\{uvicorn_command}")

    warnings = [NO_AUTH_NOTICE]
    if not loopback:
        warnings.insert(0, (
            f"!!! {host} IS NOT A LOOPBACK ADDRESS. The annotation UI has NO authentication and "
            f"no authorisation. Anybody who can reach {url} can select any rater and write, "
            f"revise or adjudicate labels as them. Only do this behind an authenticating "
            f"reverse proxy on a network you control."))
    for line in warnings:
        print(line, file=sys.stderr)
    for line in powershell:
        print(line, file=sys.stderr)

    return {
        "url": url,
        "host": host,
        "port": port,
        "loopback": loopback,
        "authentication": "none",
        "annotation_store": str(store_path),
        "environment": env,
        "uvicorn_command": uvicorn_command,
        "powershell": powershell,
        "warnings": warnings,
        "started": False,
        "note": "this command prints the server command; run it yourself so you own the process",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jobrec_eval.annotation_ui",
        description="Headless data layer for the human annotation pass (checklist items 10/11)")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build items from run bundles and assign raters")
    build.add_argument("--experiment-dir", required=True,
                       help="<out_root>/_runs/<experiment_id>")
    build.add_argument("--scenarios", default="evaluation/data/scenarios.jsonl")
    build.add_argument("--catalog", default="data/processed/jobs.jsonl")
    build.add_argument("--annotation-dir", required=True,
                       help="directory holding the annotation SQLite file")
    build.add_argument("--raters", required=True,
                       help="comma-separated rater ids (at least two)")
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--oracle-labels", default=None,
                       help="normalized/relevance_labels.csv, for the ANALYSIS side only "
                            "(never shown to a rater)")
    build.set_defaults(handler=build_command)

    status = sub.add_parser("status", help="per-rater progress and adjudication backlog")
    status.add_argument("--annotation-dir", required=True)
    status.set_defaults(handler=status_command)

    disagree = sub.add_parser("disagreements", help="list items whose raters disagree")
    disagree.add_argument("--annotation-dir", required=True)
    disagree.add_argument("--kind", choices=[KIND_RELEVANCE, KIND_CLAIM], default=None)
    disagree.add_argument("--unadjudicated-only", action="store_true")
    disagree.set_defaults(handler=disagreements_command)

    adjudicate = sub.add_parser("adjudicate", help="record an adjudicated verdict")
    adjudicate.add_argument("--annotation-dir", required=True)
    adjudicate.add_argument("--item-key", required=True)
    adjudicate.add_argument("--adjudicator", required=True)
    adjudicate.add_argument("--label", type=int, required=True)
    adjudicate.add_argument("--reason", default="")
    adjudicate.set_defaults(handler=adjudicate_command)

    export = sub.add_parser("export", help="write the human label CSVs, dump and manifest")
    export.add_argument("--annotation-dir", required=True)
    export.add_argument("--out-dir", required=True,
                        help="where the two CSVs go (beside the scenario file the pipeline "
                             "reads with --relevance-source human)")
    export.add_argument("--release-dir", default=None,
                        help="where the JSONL dump and manifest go "
                             "(e.g. final_release/human_annotations)")
    export.set_defaults(handler=export_command)

    serve = sub.add_parser(
        "serve",
        help=("print the URL and uvicorn command for the annotation web UI (does not start a "
              "server). WARNING: the UI has NO AUTHENTICATION -- the rater cookie is identity "
              "for attribution only. Keep it on 127.0.0.1 and never bind it to a routable "
              "interface without an authenticating proxy in front of it."),
        description=("Print how to start the annotation web UI. NO AUTHENTICATION: anybody who "
                     "can reach the port can label as any rater, so the default bind is "
                     f"{DEFAULT_HOST} and a non-loopback --host requires --allow-remote-host."))
    serve.add_argument("--annotation-dir", required=True)
    serve.add_argument("--host", default=DEFAULT_HOST,
                       help=f"interface to bind (default {DEFAULT_HOST}; anything that is not a "
                            f"loopback address requires --allow-remote-host)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--export-dir", default=None,
                       help="where the /export screen writes the CSVs "
                            "(default <annotation-dir>/export)")
    serve.add_argument("--release-dir", default=None,
                       help="where the archive dump and manifest go "
                            "(default <export-dir>/human_annotations)")
    serve.add_argument("--allow-remote-host", action="store_true",
                       help="explicit opt-in required to print a command binding to a "
                            "non-loopback address; prints a loud warning as well")
    serve.set_defaults(handler=serve_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand and print its result as JSON. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    print(json.dumps(args.handler(args), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
