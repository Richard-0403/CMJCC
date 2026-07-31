"""Batch experiment runner.

Runs a fixed scenario set across the five experiment variants, using the same
catalog snapshot, prompts, model settings and top-k, and writes a full artifact
bundle per run plus a batch manifest, index, failures list and checksums.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..app_service import AppService
from ..catalog import catalog_hash, load_catalog
from ..config import AppConfig
from ..domain.enums import ResponseType
from ..llm.remote_provider import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_ENV,
)
from ..orchestration.feature_flags import FeatureFlags
from ..orchestration.orchestrator import LEGACY_REPARSE_WARNING, uses_remote_backend
from ..prompts import prompt_hash
from ..utils.hashing import content_id, stable_hash
from ..utils.time import to_iso, utcnow
from .checksums import write_checksums
from .experiment_identity import (
    CODE_IDENTITY_FIELDS,
    EXPERIMENT_MANIFEST_FILENAME,
    code_identity,
    experiment_id,
    guard_output_dir,
    runtime_identity,
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

logger = logging.getLogger(__name__)


class UndeclaredClarificationAnswerError(RuntimeError):
    """A clarification-dependent scenario does not declare what the candidate answers.

    Raised BEFORE any run starts. Without a declaration the simulated user answers from a
    global default table, so a scenario's answer -- and therefore what the relevance oracle
    grades it against -- came from a constant in the evaluation harness rather than from the
    scenario. Two scenarios asking for different things were answered identically, and
    nothing forced the harness's answer and the oracle's reference to agree.

    Failing here rather than mid-batch is deliberate: this is a property of the inputs, and
    a multi-hour run must not spend anything before it is checked.
    """


class LegacyReparseError(RuntimeError):
    """Raised when a batch's runs fell back to re-parsing earlier utterances.

    Separate from a per-run failure on purpose: re-parsing is not a property of one
    scenario going wrong, it is the run pipeline failing to carry per-turn extractions, so
    every multi-turn run in the batch is affected the same way. See
    :data:`jobrec.orchestration.orchestrator.LEGACY_REPARSE_WARNING`.
    """


def assert_clarification_answers_declared(scenarios: list[dict]) -> None:
    """Every ``clarification_expected`` scenario must declare an answer for each of its
    ``acceptable_slots``.

    Scenarios that do NOT expect a clarification are exempt: a variant may still ask one
    unexpectedly (a memory-ablated condition re-asking a forgotten slot), and that ask is
    the finding rather than a gap in the scenario, so the default table stays available for
    it.
    """
    missing: list[str] = []
    for scenario in scenarios:
        if not scenario.get("clarification_expected"):
            continue
        slots = list(scenario.get("acceptable_slots") or [])
        if not slots:
            continue
        declared = ((scenario.get("reference") or {}).get("clarification_answer") or {})
        absent = [slot for slot in slots if declared.get(slot) in (None, "", [], {})]
        if absent:
            missing.append(
                f"{scenario.get('scenario_id')}: no reference.clarification_answer for "
                f"{absent}")
    if missing:
        raise UndeclaredClarificationAnswerError(
            "clarification-dependent scenarios must declare the answer the candidate "
            "gives, so the simulated user and the relevance oracle read the SAME value "
            "instead of a harness default:\n  - " + "\n  - ".join(missing))


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
        # Input gate, before a single run is spent (R33.1).
        assert_clarification_answers_declared(self.scenarios)
        identity = code_identity()
        # The catalog is hashed here rather than in ``_write_experiment_snapshot`` because
        # the id has to know it BEFORE the output directory is chosen; the snapshot writer
        # is handed the same value so the manifest and the id cannot disagree.
        chash = catalog_hash(load_catalog(self.catalog_path))
        runtime = self._runtime_identity(chash)
        exp_id = experiment_id(
            variants=variants,
            scenario_ids=[s["scenario_id"] for s in self.scenarios],
            config_hash=self.config.config_hash(),
            identity=identity,
            # The catalog, the prompts and the LLM backend: run inputs that no source
            # fingerprint and no config hash covers (see ``runtime_identity``).
            runtime=runtime,
            # The scenarios' CONTENT, so editing a scenario -- notably the authoritative
            # reference the oracle grades against, or a declared clarification answer the
            # simulated user feeds back -- yields a new experiment instead of colliding
            # with the old one on the same id.
            scenarios_fingerprint=stable_hash(self.scenarios),
        )
        exp_dir = self.out_dir / exp_id
        # Never silently replace a complete experiment (R16/R17 reproducibility freeze).
        guard_output_dir(exp_dir, identity=identity, allow_overwrite=allow_overwrite)
        exp_dir.mkdir(parents=True, exist_ok=True)

        index_rows: list[dict] = []
        failures: list[dict] = []
        repeat = self.config.experiment.repeat_count

        crashed: list[dict] = []
        for variant in variants:
            for scenario in self.scenarios:
                for run_index in range(repeat):
                    try:
                        row, failure = self._run_one(variant, scenario, run_index, exp_dir)
                    except Exception as exc:  # noqa: BLE001 - see the note below
                        # One run must not destroy the batch. An unhandled exception used
                        # to abort this loop, so neither runs_index.csv, nor failures.csv,
                        # nor the manifest, nor checksums were ever written: a single
                        # malformed model reply hundreds of calls into a multi-hour hybrid
                        # experiment discarded every completed run with it.
                        #
                        # The crash is recorded, never swallowed: it lands in
                        # failures.csv, it is counted in the manifest as
                        # ``crashed_run_count``, and the run contributes NO bundle -- so
                        # the analysis's run count is visibly short of
                        # variants x scenarios x repeats. Only the exception class is
                        # recorded; a message can quote the request.
                        logger.error(
                            "run crashed and was skipped: variant=%s scenario=%s repeat=%s "
                            "error=%s", variant, scenario["scenario_id"], run_index,
                            type(exc).__name__,
                        )
                        crash = {
                            "run_id": "", "variant": variant,
                            "scenario_id": scenario["scenario_id"],
                            "failure_code": f"runner_exception:{type(exc).__name__}",
                        }
                        failures.append(crash)
                        crashed.append({**crash, "repeat_index": run_index})
                        continue
                    index_rows.append(row)
                    if failure:
                        failures.append(failure)

        # R33.2 -- provenance gate. A run that recovered a prior turn's preferences by
        # re-parsing its text substituted the RULE extractor's reading for whatever
        # actually produced it, which in hybrid mode silently replaces the model's
        # extraction. Only a dialogue persisted before extraction snapshots existed can
        # reach that path, so in a fresh official batch its presence means the run
        # pipeline is not carrying per-turn extractions at all.
        #
        # Raised after the loop rather than inside it, and deliberately NOT caught by the
        # per-run handler above: this is not a bad run among good ones, it is the whole
        # batch being unfit to cite. No manifest is written, so the directory stays
        # incomplete and may be re-run freely once the cause is fixed.
        tainted = [row for row in index_rows if row.get("legacy_rule_reparse_turns")]
        if tainted:
            raise LegacyReparseError(
                f"{len(tainted)} of {len(index_rows)} runs recovered prior-turn "
                f"preferences by re-parsing utterance text "
                f"({LEGACY_REPARSE_WARNING}), which replaces the recorded extraction with "
                f"the rule extractor's. An official experiment cannot be built from those "
                f"runs. Affected: "
                + ", ".join(f"{r['experiment_variant']}/{r['scenario_id']}#{r['run_index']}"
                            for r in tainted[:5])
                + (" ..." if len(tainted) > 5 else "")
            )

        self._write_index(exp_dir, index_rows)
        self._write_failures(exp_dir, failures)
        # Experiment-level reproducibility snapshot: a single copy of the
        # resolved config, the catalog used, and the scenario set shared by every
        # variant. This makes the full/no_memory/no_context comparison (and the
        # five-variant path) reconstructable from the experiment directory alone,
        # without depending on the original input files (R1.2, R1.3, R32.3).
        snapshot = self._write_experiment_snapshot(exp_dir, catalog_digest=chash)
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
            # Runs that raised and produced no bundle. Recorded so an experiment can
            # never be cited as complete while it is short of
            # ``len(variants) * scenario_count * repeat_count`` runs.
            "expected_run_count": len(variants) * len(self.scenarios) * repeat,
            "crashed_run_count": len(crashed),
            "crashed_runs": crashed,
            "config_hash": self.config.config_hash(),
            "catalog_hash": snapshot["catalog_hash"],
            "scenarios_hash": snapshot["scenarios_hash"],
            "prompt_hash": prompt_hash(),
            # The run inputs that are neither source nor resolved config, recorded as the
            # exact dict the experiment id was derived from, so the id can be re-derived
            # from this manifest offline instead of being taken on trust. Carries the LLM
            # endpoint's HOST only and never the API key.
            "runtime_identity": runtime,
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

    @staticmethod
    def _session_id(exp_dir: Path, variant: str, scenario_id: str, run_index: int,
                    ordinal: int) -> str:
        """A session id determined by WHICH session of the experiment this is.

        Content-addressed over the experiment id (the directory name, itself derived from
        the frozen inputs and the code), the variant, the scenario, the repeat index and the
        session's ordinal within the run. Every one of those five is load-bearing: dropping
        any of them would make two distinct sessions of one batch share an id.

        Why this matters beyond tidiness: ``run_id`` is ``content_id("run", session_id,
        version, text)``, so a random session id gave every run a different id on a second
        execution of the SAME frozen inputs. The two batches could not be diffed run by run,
        and the experiment's own idempotence claim -- same inputs, same code, same artifact --
        held for the directory name while being false for everything inside it.

        The session id deliberately does NOT feed the experiment id. That derivation runs
        before any session exists, and it must stay a function of the inputs, the code and the
        runtime backend; adding a per-session value would make it circular.
        """
        return content_id("sess", exp_dir.name, variant, scenario_id, str(run_index),
                          str(ordinal))

    def _runtime_identity(self, chash: str) -> dict[str, Any]:
        """The non-source, non-config run inputs for the experiment id and the manifest.

        The LLM backend is only named when the run actually contacts one, which is decided
        by :func:`~jobrec.orchestration.orchestrator.uses_remote_backend` -- the same
        predicate ``make_provider`` uses to choose the provider, so the recorded backend
        cannot disagree with the one that answers.

        Testing the MODE alone was wrong and a mock-backed hybrid smoke proved it: ``mode:
        hybrid`` with ``provider: mock`` exercises the hybrid code path against the
        deterministic mock and contacts nothing, yet the identity was stamped with
        ``JOBREC_LLM_MODEL`` and ``JOBREC_LLM_BASE_URL`` from the environment -- provenance
        naming an endpoint no call had used, and an experiment id that moved when an
        unrelated variable was exported.

        Where no remote backend is used the fields are ``None`` rather than the environment's
        values, so the id cannot depend on the operator's shell for a run the shell cannot
        influence.
        """
        calls_llm = uses_remote_backend(self.config)
        return runtime_identity(
            catalog_hash=chash,
            prompt_hash=prompt_hash(),
            llm_mode=str(self.config.llm.mode),
            llm_provider=self.config.llm.provider,
            llm_model=(os.environ.get(MODEL_ENV, DEFAULT_MODEL) if calls_llm else None),
            # Reduced to its host by ``runtime_identity``; a credential embedded in the
            # base URL never reaches the id or the manifest.
            llm_endpoint=(os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
                          if calls_llm else None),
        )

    def _write_experiment_snapshot(
        self, exp_dir: Path, *, catalog_digest: str | None = None
    ) -> dict[str, Any]:
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
        # and record its content hash from the canonical catalog hasher. ``run`` already
        # needed the hash to derive the experiment id, so it is reused rather than
        # recomputed: one hash, one value in both the id and the manifest.
        chash = (catalog_digest if catalog_digest is not None
                 else catalog_hash(load_catalog(self.catalog_path)))
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
        session_id = svc.create_session(
            cand.candidate_id, variant,
            session_id=self._session_id(exp_dir, variant, scenario["scenario_id"],
                                        run_index, 0),
        )

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
        # Turn indices at which a NEW session starts for the SAME candidate on the SAME
        # service. Empty for every scenario that does not declare it, so the 42-scenario
        # set is byte-identical -- see :meth:`_session_breaks`.
        session_breaks = self._session_breaks(scenario)
        session_ids = [session_id]
        for index, text in enumerate(scenario.get("turns", [])):
            if index in session_breaks:
                # A new session: fresh dialogue state, same candidate. Anything this turn
                # knows about earlier turns can now ONLY have come through long-term
                # candidate memory, which is what makes cross-session inheritance
                # observable in an archived run instead of only in a unit test.
                session_id = svc.create_session(
                    cand.candidate_id, variant,
                    # The ORDINAL, not the turn index: a second session is the second
                    # session whichever turn opened it, and without it every session of one
                    # run would collide on a single id.
                    session_id=self._session_id(exp_dir, variant, scenario["scenario_id"],
                                                run_index, len(session_ids)),
                )
                session_ids.append(session_id)
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
        if (last_result is not None and self._is_clarification_dependent(scenario)
                and self._continues_dialogue(flags)):
            last_result, extra_turns, termination_reason, loop_trace = (
                self._run_clarification_loop(
                    svc, session_id, scenario, last_result,
                    log_sink=log_trace, result_sink=turn_results)
            )
            response_turns += extra_turns
            trace.extend(loop_trace)
        elif (last_result is not None and termination_reason is None
                and not self._continues_dialogue(flags)):
            # The run ended on a clarification (``_terminal_reason`` is None only for a
            # clarification response) under a condition that may not continue: record
            # WHY the dialogue stopped here.
            #
            # This deliberately does NOT require the scenario to expect a clarification.
            # It used to, and the consequence was a mis-attribution rather than a missing
            # label: a single-turn variant that asked an UNEXPECTED question got no
            # terminal state at all, so the error taxonomy fell through to a
            # memory-related category and blamed stale memory for a truncated dialogue.
            # Whether continuation was permitted is a property of the experiment
            # condition, not of the scenario's expectations.
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
            # How many sessions the run spanned (1 unless the scenario declares
            # ``session_breaks``) and the candidate-state version it ended on. A version
            # above 1 is exactly the signature of a long-term write-back having fired,
            # which no run-level artifact used to report.
            "session_count": len(session_ids),
            # How many turns had to recover an earlier turn's preferences by re-parsing
            # its text. Must be 0 in an official run; ``run`` refuses to write a manifest
            # otherwise. Recorded as a count rather than a flag so the index says how far
            # the taint spread.
            "legacy_rule_reparse_turns": sum(
                1 for tr in turn_results
                if tr.extracted_preferences is not None
                and LEGACY_REPARSE_WARNING in tr.extracted_preferences.extraction_warnings
            ),
            "candidate_state_version": last_result.candidate_state.version,
            "termination_reason": termination_reason,
            "total_latency_ms": rr.total_latency_ms,
            "run_dir": str(run_dir),
        }
        failure = None if rr.success else {"run_id": rr.run_id, "variant": variant,
                                           "scenario_id": scenario["scenario_id"],
                                           "failure_code": rr.failure_code}
        return row, failure

    # --------------------------------------------------------- session boundaries
    @staticmethod
    def _session_breaks(scenario: dict) -> frozenset[int]:
        """Turn indices at which the scenario asks for a NEW session (R19.1).

        Declared as ``"session_breaks": [i, ...]`` on the scenario, meaning "turn ``i``
        starts a fresh session for the same candidate". Index 0 is ignored: the run
        already begins on a new session, so breaking there would only create an unused
        one.

        Why the runner needs this at all: every run builds its own ``AppService``, so a
        scenario could never span two sessions and the experiment could not observe
        long-term memory INHERITANCE -- only within-session threading. The 210-run
        deterministic experiment consequently recorded zero long-term-eligible
        preferences, and the write-back mechanism had no archived evidence at all. A
        scenario that declares no breaks behaves exactly as before, so this cannot move
        any existing result.
        """
        raw = scenario.get("session_breaks") or []
        return frozenset(int(i) for i in raw if int(i) > 0)

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
