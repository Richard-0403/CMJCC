"""Batch experiment runner.

Runs a fixed scenario set across the five experiment variants, using the same
catalog snapshot, prompts, model settings and top-k, and writes a full artifact
bundle per run plus a batch manifest, index, failures list and checksums.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..app_service import AppService
from ..catalog import catalog_hash, load_catalog
from ..config import AppConfig
from ..domain.enums import ResponseType
from ..orchestration.feature_flags import FeatureFlags
from ..prompts import prompt_hash
from ..utils.hashing import stable_hash
from ..utils.time import to_iso, utcnow
from .checksums import write_checksums
from .experiment_identity import (
    CODE_IDENTITY_FIELDS,
    EXPERIMENT_MANIFEST_FILENAME,
    code_identity,
    experiment_id,
    guard_output_dir,
)
from .exporters import (
    _extracted_value_view,
    _system_clarification_slot,
    trace_record,
    write_run_bundle,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..orchestration.orchestrator import TurnResult

#: Termination reason recorded when the resolved variant is not allowed to continue the
#: dialogue: ``FeatureFlags.use_multi_turn_continuation`` is off (``one_shot``), so a
#: pending clarification is never answered and the run ends on the turn that asked it.
#:
#: Deliberately distinct from the loop's own exits, which describe entirely different
#: causes: ``cannot_answer`` (the simulated user had no answer for the asked slot),
#: ``max_turns`` (the dialogue budget was spent) and ``repeated_slot`` (the system would
#: have re-asked an answered slot). Those three can only be reached by a variant that IS
#: allowed to continue; this one records that continuation was never permitted, which is
#: a property of the experiment condition rather than of the user or of the dialogue.
TERMINATION_CONTINUATION_DISABLED = "continuation_disabled"


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

    def run(self, variants: list[str], allow_overwrite: bool = False) -> dict[str, Any]:
        """Run every variant x scenario x repeat and write the experiment bundle.

        The experiment id is content-addressed over the experiment inputs AND the code
        identity (see :mod:`jobrec.evaluation.experiment_identity`), so a run of different
        source code can never land in an older run's directory. ``allow_overwrite`` is the
        explicit opt-in for reusing a directory that already holds a complete experiment,
        which is how an intentional idempotent re-run is expressed; without it such a
        write raises
        :class:`~jobrec.evaluation.experiment_identity.ExperimentOverwriteError`.
        """
        identity = code_identity()
        exp_id = experiment_id(
            variants=variants,
            scenario_ids=[s["scenario_id"] for s in self.scenarios],
            config_hash=self.config.config_hash(),
            identity=identity,
        )
        exp_dir = self.out_dir / exp_id
        # Never silently replace a complete experiment (R16/R17 reproducibility freeze).
        guard_output_dir(exp_dir, identity=identity, allow_overwrite=allow_overwrite)
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
        # Experiment-level reproducibility snapshot: a single copy of the
        # resolved config, the catalog used, and the scenario set shared by every
        # variant. This makes the full/no_memory/no_context comparison (and the
        # five-variant path) reconstructable from the experiment directory alone,
        # without depending on the original input files (R1.2, R1.3, R32.3).
        snapshot = self._write_experiment_snapshot(exp_dir)
        # Reference each per-run manifest (written by write_run_bundle) so the
        # experiment-level manifest ties the batch to its reproducibility data.
        run_manifests = [
            str((Path(row["run_dir"]) / "run_manifest.json").relative_to(exp_dir))
            for row in index_rows
        ]
        manifest = {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "variants": variants,
            "scenario_count": len(self.scenarios),
            "repeat_count": repeat,
            "run_count": len(index_rows),
            "config_hash": self.config.config_hash(),
            "catalog_hash": snapshot["catalog_hash"],
            "scenarios_hash": snapshot["scenarios_hash"],
            "prompt_hash": prompt_hash(),
            "created_at": to_iso(utcnow()),
            # Code identity of the run (commit_hash / code_version / git_dirty /
            # source_fingerprint): what makes two experiment artifacts distinguishable
            # offline, and what the experiment id is partly derived from.
            **{key: identity[key] for key in CODE_IDENTITY_FIELDS},
            "artifacts": snapshot["artifacts"],
            "run_manifests": run_manifests,
        }
        (exp_dir / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
        self._write_checksums(exp_dir)
        return manifest

    def _write_experiment_snapshot(self, exp_dir: Path) -> dict[str, Any]:
        """Write the shared, experiment-level reproducibility artifacts.

        Emits exactly one copy each of the resolved config, the catalog, and the
        scenario set at the experiment root so every variant's comparison is
        reproducible from the experiment directory alone. Returns the content
        hashes and the relative artifact paths for inclusion in the manifest.
        """
        # (a) resolved config -- the single config shared by all variants; the
        # per-variant override is only the variant field.
        config_path = exp_dir / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False)
        )

        # (b) catalog snapshot -- copy the exact catalog file used for the batch
        # and compute its content hash via the canonical catalog hasher.
        jobs = load_catalog(self.catalog_path)
        chash = catalog_hash(jobs)
        catalog_snapshot = exp_dir / "catalog.jsonl"
        catalog_snapshot.write_text(Path(self.catalog_path).read_text())

        # (c) scenarios snapshot -- the exact scenario set run across variants.
        scenarios_snapshot = exp_dir / "scenarios.jsonl"
        with scenarios_snapshot.open("w") as fh:
            for scenario in self.scenarios:
                fh.write(json.dumps(scenario, default=str))
                fh.write("\n")
        shash = stable_hash(self.scenarios)

        return {
            "catalog_hash": chash,
            "scenarios_hash": shash,
            "artifacts": {
                "resolved_config": config_path.name,
                "catalog": catalog_snapshot.name,
                "scenarios": scenarios_snapshot.name,
            },
        }

    def _run_one(self, variant, scenario, run_index, exp_dir):
        cfg = self.config.model_copy(deep=True)
        from ..domain.enums import ExperimentVariant

        cfg.experiment.variant = ExperimentVariant(variant)
        # The variant's behaviour switches, resolved through the SAME
        # ``FeatureFlags.from_config`` the orchestrator uses on this very config, so the
        # runner and the orchestrator can never disagree about what a variant is (never a
        # string comparison on the variant name).
        flags = FeatureFlags.from_config(cfg)
        # Deterministic in-memory run (no external DB dependency for experiments).
        svc = AppService(cfg, self.catalog_path)
        profile = dict(scenario["profile"])
        profile.setdefault("candidate_id", scenario["scenario_id"] + "-cand")
        cand = svc.create_candidate(profile)
        session_id = svc.create_session(cand.candidate_id, variant)

        # Process the scripted scenario turns on a SINGLE session so that memory
        # and dialogue-state thread correctly across turns (single code path).
        # Collect one dialogue-trace record per turn (R7.3): scripted turns record
        # what the candidate said and what the system extracted/asked; the loop
        # below appends the simulated answer turns.
        last_result = None
        response_turns = 0
        trace: list[dict] = []
        # Structured log records of every turn, exported as log_trace.jsonl (R27.3).
        log_trace: list[dict] = []
        # EVERY turn's result, in order. The bundle used to export only the final
        # turn's model calls / run record, so earlier turns of a multi-turn run were
        # absent from the archive entirely; threading the results lets the exporter
        # write whole-run call accounting and per-turn records (R7.3/R11.1).
        turn_results: list[TurnResult] = []
        for text in scenario.get("turns", []):
            last_result = svc.process_turn(session_id, text, scenario_id=scenario["scenario_id"])
            response_turns += 1
            turn_results.append(last_result)
            log_trace.extend(last_result.log_trace)
            trace.append(trace_record(
                last_result,
                user_utterance=text,
                clarification_slot=_system_clarification_slot(last_result),
                extracted_value=_extracted_value_view(last_result),
            ))

        # For clarification-dependent scenarios, keep driving the dialogue: answer
        # each system clarification with the SimulatedUser and feed the answer back
        # as the next turn on the same session until a terminal outcome, the
        # max-turn guard, the repeated-slot guard, or an unanswerable clarification.
        # Non-clarification scenarios keep the existing single-pass behaviour.
        #
        # A variant whose resolved flags disable multi-turn continuation never enters the
        # loop: the system may still ASK, but nothing is fed back, so the run ends on the
        # asking turn. This is a gate on the shared path, not a second pipeline.
        termination_reason = self._terminal_reason(last_result)
        if last_result is not None and self._is_clarification_dependent(scenario):
            if self._continues_dialogue(flags):
                last_result, extra_turns, termination_reason, loop_trace = (
                    self._run_clarification_loop(
                        svc, session_id, scenario, last_result,
                        log_sink=log_trace, result_sink=turn_results)
                )
                response_turns += extra_turns
                trace.extend(loop_trace)
            elif termination_reason is None:
                # Non-terminal (the system asked a clarification) and the condition
                # cannot continue: record WHY the dialogue stopped here.
                termination_reason = TERMINATION_CONTINUATION_DISABLED

        # Stamp the loop's termination reason onto the final record so run-level
        # metrics can read the terminal outcome from the last trace row (R7.8).
        if trace:
            trace[-1]["termination_reason"] = termination_reason

        run_dir = exp_dir / variant / scenario["scenario_id"] / str(run_index)
        write_run_bundle(last_result, run_dir, cfg, dialogue_trace=trace,
                         log_trace=log_trace, turn_results=turn_results)

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
            "response_turns": response_turns,
            "termination_reason": termination_reason,
            "total_latency_ms": rr.total_latency_ms,
            "run_dir": str(run_dir),
        }
        failure = None if rr.success else {"run_id": rr.run_id, "variant": variant,
                                           "scenario_id": scenario["scenario_id"],
                                           "failure_code": rr.failure_code}
        return row, failure

    # ------------------------------------------------------- clarification loop
    @staticmethod
    def _is_clarification_dependent(scenario: dict) -> bool:
        """A scenario NEEDS clarification when its reference expects one.

        Determined from the scenario's ``clarification_expected`` flag (the
        ``Scenario`` reference carries the same field); every other scenario keeps
        the existing single-pass behaviour.

        This is the scenario half of the loop condition only. Whether the dialogue may
        actually continue is the variant's half -- see :meth:`_continues_dialogue`.
        """
        return bool(scenario.get("clarification_expected", False))

    @staticmethod
    def _continues_dialogue(flags: FeatureFlags) -> bool:
        """Whether the resolved variant may continue a dialogue past a clarification.

        ``use_multi_turn_continuation`` is exactly this capability, so a condition that
        has it switched off (``one_shot``, or any variant under
        ``memory.use_multi_turn_continuation: false``) is a genuine single-turn condition:
        the runner never feeds a simulated answer back and the run terminates on the turn
        that asked, with :data:`TERMINATION_CONTINUATION_DISABLED`.
        """
        return bool(flags.use_multi_turn_continuation)

    @staticmethod
    def _has_results(result) -> bool:
        """True when a recommendation actually returned at least one selected job."""
        decision = result.decision
        return bool(decision and decision.selected_job_ids)

    def _terminal_reason(self, result) -> str | None:
        """Classify a turn result's terminal state for the dialogue loop.

        Returns a termination reason string for terminal outcomes, or ``None`` when
        the system is asking a clarification and the dialogue can continue.
        """
        if result is None:
            return "no_result"
        rtype = str(result.response.response_type)
        if rtype == ResponseType.RECOMMENDATION:
            return "recommendation" if self._has_results(result) else "recommendation_empty"
        if rtype == ResponseType.NO_MATCH:
            return "no_match"
        if rtype == ResponseType.CLARIFICATION:
            return None
        return "error"

    def _run_clarification_loop(
        self, svc, session_id, scenario, last_result, log_sink: list[dict] | None = None,
        result_sink: list | None = None,
    ):
        """Answer system clarifications until the dialogue reaches a terminal state.

        Repeatedly feeds a :class:`SimulatedUser` answer back as the next turn on the
        SAME session (so memory/dialogue-state thread correctly). Terminates on:

        * recommendation success (recommendation response with results), OR
        * a correct no-match (no_match response), OR
        * the ``config.experiment.max_dialogue_turns`` hard cap, OR
        * failure / the SimulatedUser cannot answer (returns ``None``), OR
        * the repeated-slot guard (a slot would be re-asked with no progress).

        When ``log_sink`` is given, each answer turn's structured log records are
        appended to it so the run bundle's ``log_trace.jsonl`` covers the whole
        dialogue rather than only the final turn (R27.3). ``result_sink`` does the
        same for the turn results themselves, so the exporter can attribute model
        calls and latency to the answer turns instead of dropping them.

        Returns ``(last_result, extra_turns, termination_reason, trace)`` where
        ``extra_turns`` counts the simulated answer turns fed into the session and
        ``trace`` is one dialogue-trace record per simulated answer turn (R7.3).
        For an answer turn, ``clarification_slot`` is the slot being answered and
        ``extracted_value`` is that answered value; intermediate records carry a
        ``None`` termination reason (the caller stamps the final reason).
        """
        from jobrec_eval.simulated_user import SimulatedUser

        max_turns = self.config.experiment.max_dialogue_turns
        sim_user = SimulatedUser(scenario)
        scenario_id = scenario["scenario_id"]
        asked: set[str] = set()
        extra_turns = 0
        trace: list[dict] = []

        while True:
            reason = self._terminal_reason(last_result)
            if reason is not None:
                # Terminal outcome (recommendation success, no-match, or error).
                return last_result, extra_turns, reason, trace

            # Hard max-turn guard: never exceed the configured dialogue cap.
            if extra_turns >= max_turns:
                return last_result, extra_turns, "max_turns", trace

            answer = sim_user.answer(last_result.clarification, asked)
            if answer is None:
                # The user cannot/won't answer -> terminate the dialogue.
                return last_result, extra_turns, "cannot_answer", trace

            utterance, slot = answer
            # Repeated-slot guard: do not re-ask/re-answer the same slot endlessly.
            if slot in asked:
                return last_result, extra_turns, "repeated_slot", trace
            asked.add(slot)

            last_result = svc.process_turn(session_id, utterance, scenario_id=scenario_id)
            extra_turns += 1
            if result_sink is not None:
                result_sink.append(last_result)
            if log_sink is not None:
                # Accumulate each answer turn's structured records (R27.3).
                log_sink.extend(last_result.log_trace)
            # extracted_value reflects what the system extracted from the simulated
            # answer (keyed by field); the answered slot is recorded separately.
            trace.append(trace_record(
                last_result,
                user_utterance=utterance,
                clarification_slot=slot,
                extracted_value=_extracted_value_view(last_result),
            ))

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
        """Write the unified ``checksums.json`` over every artifact (R16.1).

        Delegates to :mod:`jobrec.evaluation.checksums`, which supersedes the
        earlier ``checksums.sha256`` that covered only ``*.json`` files.
        """
        write_checksums(exp_dir)
