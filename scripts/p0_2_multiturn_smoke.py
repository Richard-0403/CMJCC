"""P0-2 acceptance: run the multi-turn scenarios in hybrid and audit their provenance.

Why a dedicated list
--------------------
``evaluation/data/scenarios_subset.jsonl`` holds 12 scenarios but only 3 multi-turn ones
(SC-D-01, SC-D-06, SC-D-12), so it cannot show whether prior-turn extractions are being
carried forward. This script derives the list from the authoritative set instead: every
scenario with more than one declared turn, which is 12 of the 42. The authoritative file is
only READ; the derived list is written to the output directory.

What it checks
--------------
1. ``extraction_method`` per turn. Before the fix, a hybrid run's turn-2 state was built
   from the RULE extractor's reading of turn 1, so prior fields were attributed to ``rule``
   no matter what the model returned.
2. ``legacy_rule_reparse`` absent. Its presence means the run recovered history by
   re-parsing text, which is what the fix removed.
3. Evidence integrity: no dialogue EvidenceItem duplicated across turns, and no evidence
   cited by a turn that belongs to a different turn (turn_id drift).
4. before/after diff. The "before" arm re-drives each recorded run with the stored
   extraction snapshots stripped, which is exactly the old code path, using that run's OWN
   recorded model responses so the model cannot be the source of any difference.

Usage
-----
    python scripts/p0_2_multiturn_smoke.py --run      # execute the 12 hybrid runs
    python scripts/p0_2_multiturn_smoke.py --audit    # audit bundles already on disk

Nothing here writes to ``evaluation/`` or to any sealed artifact: the default output root
is ``artifacts/p0_2_smoke/``, which .gitignore already covers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

AUTHORITATIVE = Path("evaluation/data/scenarios.jsonl")
CATALOG = "data/processed/jobs.jsonl"
#: The REAL hybrid config: ``mode: hybrid`` with ``provider: remote``. Not
#: ``experiment_full.yaml`` (``provider: mock``) and not ``hybrid.yaml`` (also mock) -- both
#: run the hybrid code path against the deterministic mock, which cannot show whether a
#: MODEL's extraction is carried forward. A first attempt at this smoke used
#: ``experiment_full.yaml`` with only ``mode`` overridden and was answered entirely by the
#: mock; the preferences were still tagged ``llm``, so the result looked like a pass.
CONFIG = "configs/hybrid_vectorengine.yaml"
DEFAULT_OUT = Path("artifacts/p0_2_smoke")


def multi_turn_scenarios() -> list[dict]:
    """Every authoritative scenario with more than one declared turn."""
    rows = [json.loads(line) for line in
            AUTHORITATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [s for s in rows if len(s.get("turns") or []) > 1]


def write_pilot_list(out_root: Path) -> Path:
    scenarios = multi_turn_scenarios()
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "scenarios_multiturn.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for scenario in scenarios:
            fh.write(json.dumps(scenario, default=str))
            fh.write("\n")
    print(f"pilot list: {path} ({len(scenarios)} multi-turn scenarios)")
    print("  " + ", ".join(f"{s['scenario_id']}({len(s['turns'])}t)" for s in scenarios))
    return path


# --------------------------------------------------------------------------- run
def run_smoke(out_root: Path) -> Path:
    from jobrec.config import load_config
    from jobrec.evaluation.experiment_runner import ExperimentRunner
    from jobrec.orchestration.orchestrator import uses_remote_backend

    scenarios_path = write_pilot_list(out_root)
    config = load_config(CONFIG, base_dir="configs")
    config.experiment.repeat_count = 1
    if not uses_remote_backend(config):
        raise SystemExit(
            f"{CONFIG} resolves to mode={config.llm.mode} provider={config.llm.provider}, "
            "which is answered by the mock. A P0-2 hybrid smoke has to reach a real model "
            "or it cannot show that the MODEL's extraction is carried forward."
        )
    runs_root = out_root / "runs"
    runner = ExperimentRunner(config, CATALOG, str(scenarios_path), out_dir=str(runs_root))
    manifest = runner.run(["full"])
    print(f"\nexperiment: {manifest['experiment_id']}")
    print(f"  runs {manifest['run_count']}/{manifest['expected_run_count']}, "
          f"crashed {manifest['crashed_run_count']}")
    print(f"  runtime identity: {json.dumps(manifest['runtime_identity'])}")
    return Path(manifest["experiment_dir"])


# ------------------------------------------------------------------------- audit
def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _run_dirs(exp_dir: Path) -> list[Path]:
    return sorted(p.parent for p in exp_dir.rglob("dialogue_state.json"))


def audit(exp_dir: Path, out_root: Path | None = None) -> int:
    """Report provenance for every run bundle. Returns a process exit code.

    ``out_root`` is where the JSON report goes, and it defaults to beside the experiment rather
    than inside it. The report used to be written INTO ``exp_dir``, after the runner had already
    written ``checksums.json`` -- so auditing an experiment added an unrecorded file to it and
    ``jobrec_eval.cli verify`` then failed on the batch with "present on disk but not recorded".
    An audit must not modify the artifact it audits.
    """
    problems: list[str] = []
    method_by_position: dict[int, Counter] = {}
    legacy_runs: list[str] = []
    rows: list[dict] = []
    # Every place the pipeline degraded rather than failing. "0 fallback" is a release
    # criterion, and a fallback is silent by design -- the run succeeds and the artifact looks
    # normal -- so it has to be looked for rather than waited for.
    fallbacks: list[str] = []

    for run_dir in _run_dirs(exp_dir):
        dialogue = _read(run_dir / "dialogue_state.json") or {}
        extraction = _read(run_dir / "extracted_preferences.json") or {}
        active = _read(run_dir / "active_search_state.json") or {}
        decision = _read(run_dir / "recommendation_decision.json") or {}
        label = "/".join(run_dir.parts[-3:])

        warnings = list(extraction.get("extraction_warnings") or [])
        if "legacy_rule_reparse" in warnings:
            legacy_runs.append(label)

        candidate_turns = [t for t in dialogue.get("turns", [])
                           if t.get("speaker") == "candidate"]

        # (0) fallbacks. Four independent kinds, each of which substitutes something weaker
        # for what was asked for: a field's value coming from the rule extractor after the
        # model's was rejected, a whole model call failing over to rules, a retried request,
        # and lexical recall coming back empty so the whole catalogue was substituted.
        for position, turn in enumerate(candidate_turns):
            for pref in (turn.get("extraction_snapshot") or {}).get("preferences", []):
                source = (pref.get("metadata") or {}).get("extraction_source") or ""
                if "fallback" in source:
                    fallbacks.append(f"{label} turn {position}: {pref.get('field_name')} "
                                     f"extraction_source={source}")
        for warning in warnings:
            if "fallback" in warning or "unrecoverable" in warning:
                fallbacks.append(f"{label}: {warning}")
        retrieval = _read(run_dir / "retrieval_results.json") or {}
        if retrieval.get("expanded") or retrieval.get("expansion_reason"):
            fallbacks.append(f"{label}: retrieval expanded "
                             f"({retrieval.get('expansion_reason')})")
        for line in (run_dir / "model_calls.jsonl").read_text(
                encoding="utf-8").splitlines() if (run_dir / "model_calls.jsonl").exists() \
                else []:
            if not line.strip():
                continue
            call = json.loads(line)
            meta = call.get("response_metadata") or {}
            if meta.get("fell_back") or call.get("parsed_ok") is False:
                fallbacks.append(f"{label}: model call {call.get('purpose')} fell back")
            if meta.get("retry_reason"):
                fallbacks.append(f"{label}: model call {call.get('purpose')} retried "
                                 f"({meta.get('retry_reason')})")

        # (1) extraction_method per turn position, read from each turn's own snapshot.
        for position, turn in enumerate(candidate_turns):
            snapshot = turn.get("extraction_snapshot")
            if snapshot is None:
                problems.append(f"{label}: turn {position} has no extraction_snapshot")
                continue
            counter = method_by_position.setdefault(position, Counter())
            for pref in snapshot.get("preferences", []):
                counter[str((pref.get("metadata") or {}).get("extraction_method"))] += 1

        # (3) evidence integrity: ids unique across turns, and each turn cites only ids
        # its own snapshot names.
        seen: dict[str, int] = {}
        for position, turn in enumerate(candidate_turns):
            snapshot = turn.get("extraction_snapshot") or {}
            own = {p.get("evidence_id") for p in snapshot.get("preferences", [])
                   if p.get("evidence_id")}
            for eid in turn.get("evidence_ids") or []:
                if eid in seen:
                    problems.append(
                        f"{label}: evidence {eid} appears on turns {seen[eid]} and {position}")
                seen[eid] = position
                if own and eid not in own:
                    problems.append(
                        f"{label}: turn {position} cites {eid}, absent from its snapshot")
            # turn_id drift: a snapshot preference must name its own turn.
            for pref in snapshot.get("preferences", []):
                origin = pref.get("origin_turn_id")
                if origin and origin != turn.get("turn_id"):
                    problems.append(
                        f"{label}: turn {position} snapshot preference "
                        f"{pref.get('field_name')} claims origin {origin}")

        if not (run_dir / "recommendation_decision.json").exists():
            # A clarification-terminated run legitimately has no decision. Anything else
            # missing means the bundle is incomplete and must not be read as "0 jobs".
            if not (run_dir / "clarification.json").exists():
                problems.append(f"{label}: no recommendation_decision.json and no "
                                "clarification.json -- incomplete bundle")

        rows.append({
            "run": label,
            "turns": len(candidate_turns),
            "hard": sorted(active.get("hard_constraint_fields") or []),
            "returned": list(decision.get("selected_job_ids") or []),
            "eligible": sum(1 for e in decision.get("eligibility_results") or []
                            if e.get("eligible")),
            "no_match": bool(decision.get("no_match")),
            "state": _state_view(active),
        })

    print(f"\n{'=' * 72}\nP0-2 multi-turn provenance audit: {exp_dir}\n{'=' * 72}")
    print(f"run bundles: {len(rows)}")

    print("\nextraction_method by turn position (from each turn's own snapshot):")
    for position in sorted(method_by_position):
        counts = method_by_position[position]
        total = sum(counts.values())
        detail = ", ".join(f"{m}={n}" for m, n in sorted(counts.items()))
        print(f"  turn {position}: {total} preferences -- {detail}")

    print(f"\nlegacy_rule_reparse: {len(legacy_runs)} run(s)"
          + (f" -- {legacy_runs}" if legacy_runs else " (none, as required)"))

    print(f"\nfallbacks: {len(fallbacks)}"
          + ("" if fallbacks else " (none)"))
    for entry in fallbacks:
        print(f"  - {entry}")

    print("\nper-run final state:")
    for row in rows:
        print(f"  {row['run']}: turns={row['turns']} hard={row['hard']} "
              f"eligible={row['eligible']} returned={len(row['returned'])} "
              f"no_match={row['no_match']}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nevidence integrity: no duplicated ids, no turn_id drift.")

    destination = (out_root if out_root is not None else exp_dir.parent)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"p0_2_audit.{exp_dir.name}.json"
    report_path.write_text(
        json.dumps({"experiment_dir": str(exp_dir),
                    "runs": rows, "legacy_runs": legacy_runs, "fallbacks": fallbacks,
                    "method_by_position": {str(k): dict(v)
                                           for k, v in method_by_position.items()},
                    "problems": problems}, indent=2, sort_keys=True),
        encoding="utf-8")
    print(f"\naudit report: {report_path}")
    return 1 if (problems or legacy_runs or fallbacks) else 0


_STATE_FIELDS = ("target_roles", "preferred_locations", "work_modes", "salary_min",
                 "salary_currency", "seniority_level", "industries", "skills_have",
                 "employment_types", "hard_constraint_fields", "soft_preference_fields")


def _state_view(active: dict) -> dict:
    """The comparable part of an ActiveSearchState: values and strengths, no ids/times."""
    out: dict[str, Any] = {}
    for field in _STATE_FIELDS:
        value = active.get(field)
        out[field] = sorted(value) if isinstance(value, list) else value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the hybrid runs")
    parser.add_argument("--audit", action="store_true", help="audit bundles on disk")
    parser.add_argument("--diff", action="store_true",
                        help="compare final state and jobs with vs without snapshots")
    parser.add_argument("--exp-dir", default=None, help="experiment dir for --audit")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--list-only", action="store_true",
                        help="write the pilot scenario list and stop")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if args.list_only:
        write_pilot_list(out_root)
        return 0

    exp_dir: Path | None = Path(args.exp_dir) if args.exp_dir else None
    if args.run:
        exp_dir = run_smoke(out_root)
    if (args.audit or args.diff) and exp_dir is None:
        candidates = sorted((out_root / "runs").glob("exp-*"))
        if not candidates:
            print(f"no experiment found under {out_root / 'runs'}")
            return 1
        exp_dir = candidates[-1]

    status = 0
    if args.audit:
        status |= audit(exp_dir, out_root)
    if args.diff:
        status |= diff_arms(exp_dir, out_root)
    if not (args.run or args.audit or args.diff):
        parser.error("choose --run, --audit, --diff or --list-only")
    return status


# -------------------------------------------------------------- before/after diff
class CachingProvider:
    """Wraps the real remote provider and memoises answers by (purpose, prompt).

    The before/after arms have to see the SAME model answers, otherwise the diff measures
    sampling noise rather than the change under test. Replaying the run bundle's
    ``model_calls.jsonl`` cannot do it: the exported records carry no ``prompt`` (it is
    redacted, since prompts contain candidate text), so ``ReplayProvider`` can only index by
    ``call_id`` and a fresh run misses every lookup -- which silently sends BOTH arms down
    the rule-extractor fallback and makes the comparison look like "no difference".

    So the arms share one cache instead. The first arm's real calls populate it; the second
    arm reuses them wherever the prompt is identical, which is every extraction call, since
    an extraction prompt is a function of the turn text alone. ``fresh_calls`` counts what
    the second arm had to ask anew, i.e. how far the dialogue diverged.
    """

    def __init__(self) -> None:
        from jobrec.config import load_config
        from jobrec.orchestration.orchestrator import make_provider

        self._inner = make_provider(load_config(CONFIG, base_dir="configs"))
        self.name = getattr(self._inner, "name", "caching")
        self.model = getattr(self._inner, "model", "caching")
        self._json: dict[tuple[str, str], Any] = {}
        self._text: dict[tuple[str, str], Any] = {}
        self.fresh_calls = 0
        self.serving_from_cache = False

    def complete_json(self, prompt: str, *, purpose: str):
        key = (purpose, prompt)
        if key not in self._json:
            if self.serving_from_cache:
                self.fresh_calls += 1
            self._json[key] = self._inner.complete_json(prompt, purpose=purpose)
        return self._json[key]

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = ""):
        key = (purpose, prompt)
        if key not in self._text:
            if self.serving_from_cache:
                self.fresh_calls += 1
            self._text[key] = self._inner.complete_text(
                prompt, purpose=purpose, fallback=fallback)
        return self._text[key]

    def manifest(self) -> dict:
        return self._inner.manifest()


def _drive_arm(scenario: dict, provider: CachingProvider, *, legacy: bool):
    """Drive a scenario's declared turns; ``legacy=True`` reproduces the OLD code path.

    The old behaviour IS the legacy fallback: strip the stored snapshots and prior turns get
    re-parsed with the rule extractor, exactly as before the fix. So the "before" arm is the
    current code walking the old path rather than a separate checkout, and the only thing
    that differs between arms is prior-turn handling.
    """
    from jobrec import app_service as app_service_module
    from jobrec.app_service import AppService
    from jobrec.config import load_config
    from jobrec.orchestration.orchestrator import ConversationOrchestrator

    original_make = app_service_module.make_provider
    original_prior = ConversationOrchestrator._prior_dialogue_preferences

    def stripped(self, dialogue_state):
        bare = dialogue_state.model_copy(update={
            "turns": [t.model_copy(update={"extraction_snapshot": None})
                      for t in dialogue_state.turns],
        })
        return original_prior(self, bare)

    app_service_module.make_provider = lambda cfg, replay_path=None: provider
    if legacy:
        ConversationOrchestrator._prior_dialogue_preferences = stripped
    try:
        service = AppService(load_config(CONFIG, base_dir="configs"), CATALOG)
        candidate = service.create_candidate(scenario["profile"])
        session = service.create_session(candidate.candidate_id, "full")
        last = None
        for text in scenario["turns"]:
            last = service.process_turn(session, text, scenario_id=scenario["scenario_id"])
        return last
    finally:
        app_service_module.make_provider = original_make
        ConversationOrchestrator._prior_dialogue_preferences = original_prior


def diff_arms(exp_dir: Path, out_root: Path) -> int:
    """Compare final state and returned jobs with vs without carried-forward snapshots."""
    scenarios = {s["scenario_id"]: s for s in multi_turn_scenarios()}
    report: list[dict] = []

    print(f"\n{'=' * 72}\nbefore/after diff (model held constant by replaying each run's "
          f"own calls)\n{'=' * 72}")
    for scenario_id in sorted(scenarios):
        scenario = scenarios[scenario_id]
        provider = CachingProvider()
        try:
            # "after" first so its real calls populate the cache; "before" then reuses
            # them, so any difference is attributable to prior-turn handling alone.
            after = _drive_arm(scenario, provider, legacy=False)
            provider.serving_from_cache = True
            before = _drive_arm(scenario, provider, legacy=True)
        except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
            print(f"  {scenario_id}: UNRELIABLE ({type(exc).__name__}: {exc})")
            report.append({"scenario_id": scenario_id, "status": "unreliable",
                           "error": f"{type(exc).__name__}: {exc}"})
            continue

        after_state = _state_view(after.active_search_state.model_dump(mode="json"))
        before_state = _state_view(before.active_search_state.model_dump(mode="json"))
        after_jobs = list(after.decision.selected_job_ids) if after.decision else []
        before_jobs = list(before.decision.selected_job_ids) if before.decision else []

        changed = {k: {"before": before_state[k], "after": after_state[k]}
                   for k in after_state if before_state[k] != after_state[k]}
        entry = {
            "scenario_id": scenario_id,
            # The "before" arm asking the model something new means the dialogue diverged
            # far enough to change a prompt. Reported, not treated as a defect.
            "status": "ok" if not provider.fresh_calls else "before_arm_diverged",
            "state_fields_changed": sorted(changed),
            "state_diff": changed,
            "jobs_before": before_jobs,
            "jobs_after": after_jobs,
            "jobs_changed": before_jobs != after_jobs,
            "before_arm_fresh_calls": provider.fresh_calls,
        }
        report.append(entry)
        flag = "" if entry["status"] == "ok" else f"  [{entry['status']}]"
        print(f"  {scenario_id}: state_changed={sorted(changed) or 'none'} "
              f"jobs_changed={entry['jobs_changed']}{flag}")
        for field, values in changed.items():
            print(f"      {field}: {values['before']!r} -> {values['after']!r}")
        if entry["jobs_changed"]:
            print(f"      jobs: {before_jobs} -> {after_jobs}")

    changed_state = [e for e in report if e.get("state_fields_changed")]
    changed_jobs = [e for e in report if e.get("jobs_changed")]
    unreliable = [e for e in report if e.get("status") != "ok"]
    print(f"\nsummary: {len(report)} scenarios, {len(changed_state)} with a state change, "
          f"{len(changed_jobs)} with a different job list, {len(unreliable)} unreliable")

    out = out_root / "p0_2_before_after_diff.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
