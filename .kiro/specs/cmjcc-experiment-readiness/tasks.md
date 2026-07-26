# Implementation Plan: CMJCC Experiment Readiness

## Overview

This plan turns the approved design into incremental, test-driven coding tasks against the
**existing** CMJCC codebase (`src/jobrec`, `src/jobrec_eval`, `tests/`). It is a
completion/hardening effort: every task extends an existing module or adds a focused new
module on the single shared code path — no forked pipelines. Tasks are ordered by the
design's phasing:

- **Phase 1 — Minimum Defensible Version (MDV):** R4, R5+R32 (flags/attribution), R6, R8,
  R9, plus the run-manifest/artifact saving needed for the `full`/`no_memory`/`no_context`
  comparison.
- **Phase 2 — Remaining P0:** R7 (clarification loop), R10 (failure-path tests + metrics),
  R32 reporting framing.
- **Phase 3 — P1:** R11–R18 plus the R19–R24 dedicated test suites.
- **Phase 4 — P2:** R25–R30.

Conventions used throughout:
- Test-related sub-tasks are marked with `*` and MAY be skipped for a faster MVP; core
  implementation sub-tasks are never marked optional.
- `_Requirements: X.Y_` traces each task to acceptance criteria.
- `_Properties: N_` references a Correctness Property from design.md; each such property gets
  a dedicated Hypothesis property-based test (`@settings(max_examples=100)`, tagged
  `# Feature: cmjcc-experiment-readiness, Property N: <text>`).
- **Human relevance/grounding annotation is OUT OF SCOPE** (performed manually by the
  author); no annotation-production tasks appear below.

## Tasks

- [x] 1. Phase 1 groundwork: config + FeatureFlags field for variant differentiation (R5/R32)
  - [x] 1.1 Add `use_multi_turn_continuation` to `FeatureFlags` and the variant matrix
    - Add `use_multi_turn_continuation: bool` to the `FeatureFlags` dataclass in
      `src/jobrec/orchestration/feature_flags.py`
    - Set the authoritative resolved matrix: `one_shot=False`, all other variants `True`;
      keep every other flag per the design's matrix table so `one_shot` and `no_memory`
      differ on exactly this field
    - _Requirements: 5.1, 5.2, 5.3, 5.8_
    - _Properties: 6_
  - [x] 1.2 Extend `FeatureFlags.from_config` and config models
    - Add `MemoryConfig.use_multi_turn_continuation: bool = True` and
      `ExperimentConfig.max_dialogue_turns: int = 6` in `src/jobrec/config.py`
    - Read the new flag in `from_config` using the existing "config may only restrict, not
      expand" pattern
    - _Requirements: 5.3, 5.4_
  - [x] 1.3 Add the `flag_diff` helper and flag-group sets for ablation attribution
    - In `src/jobrec/orchestration/feature_flags.py` add `flag_diff(a, b) -> set[str]`
      (variant field excluded) and the `MEMORY_FLAGS` / `CONTEXT_FLAGS` constants
    - _Requirements: 32.1, 32.2, 32.7_
    - _Properties: 7_
  - [x] 1.4 Gate the multi-turn continuation branch in the orchestrator (single path)
    - In `src/jobrec/orchestration/orchestrator.py`, guard the prior-dialogue fold with
      `use_prior_dialogue and use_multi_turn_continuation` so `one_shot` behaves as a genuine
      single-turn condition without a separate pipeline
    - _Requirements: 5.1, 5.5, 5.7, 5.8, 31.5_
  - [ ]* 1.5 Property test: distinct variants resolve to distinct FeatureFlags
    - Add to `tests/unit/test_variant_isolation.py`
    - **Property 6: All distinct variants resolve to distinct FeatureFlags (one_shot != no_memory)**
    - **Validates: Requirements 5.3, 5.6**
  - [ ]* 1.6 Property test: ablation pairs differ only in target-mechanism flags
    - Add to `tests/unit/test_variant_isolation.py`
    - **Property 7: Ablation pairs differ only in their target-mechanism flags**
    - **Validates: Requirements 32.1, 32.2, 32.7**
  - [ ]* 1.7 Unit test guarding one_shot != no_memory configuration
    - Add an explicit test to `tests/unit/test_variant_isolation.py` that fails if resolved
      `one_shot` and `no_memory` flag sets ever become identical
    - _Requirements: 5.6_

- [x] 2. Record resolved FeatureFlags in the run record (R5.4)
  - [x] 2.1 Add `feature_flags` to `RunRecord` and populate on finish
    - Add `feature_flags: dict[str, Any] = Field(default_factory=dict)` to
      `src/jobrec/domain/run_record.py`
    - Populate it in the orchestrator's `_finish` from `dataclasses.asdict(self.flags)`
    - _Requirements: 5.4_
  - [ ]* 2.2 Contract test for the new RunRecord field
    - Extend `tests/contract/test_schemas.py` to assert the field serializes with a default
    - _Requirements: 5.4_

- [x] 3. R4 — CandidateState long-term memory write-back
  - [x] 3.1 Add the pure scope-resolution helper
    - In `src/jobrec/agents/memory_agent.py` add `resolve_scope(p) -> PersistenceScope`
      combining `persistence_scope` and `temporal_scope` per the design rule
      ("from now on"->long_term; "this time only"->active_search; else fall back)
    - _Requirements: 4.5, 4.7_
    - _Properties: 4_
  - [x] 3.2 Add temporal-scope phrase cues to the rule extractor
    - In `src/jobrec/agents/candidate_understanding.py` add cue tables mapping durable
      phrases (from now on, always, going forward, in general, permanently) -> `long_term`
      and one-off phrases (this time, just this search, for now, only this) ->
      `current_search`
    - Amend the extraction prompt in `src/jobrec/prompts.py` to instruct the model to emit
      `temporal_scope` (parser in `structured_output` already reads it)
    - _Requirements: 4.1, 4.6, 4.7_
  - [x] 3.3 Implement `MemoryAgent.apply_confirmed_updates`
    - Add the method to `src/jobrec/agents/memory_agent.py` returning a NEW versioned
      `CandidateState` (version+1, `updated_at=now`) via `model_copy`, never mutating input;
      return the same instance when nothing resolves to a durable long-term write
    - Apply the resolution rule: confirmed status, `long_term` resolved scope, confidence
      threshold, and not blocked by a non-`override` conflict
    - Scalar fields: supersede old `PreferenceValue` (`is_active=False`, `effective_to=now`)
      and store the deactivated record under `metadata["superseded"][field]`; add new active
      value. List fields: deactivate matching prior entry and append the new active value
    - Register a long-term `EvidenceItem` per written value and attach its id
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.10_
    - _Properties: 1, 2, 3, 4_
  - [x] 3.4 Implement the R4.11 non-override conflict guard
    - Before writing, intersect writable fields with conflicts whose `resolution != override`;
      skip those long-term writes, keep the existing value, and ensure the conflict is
      recorded on `DialogueState.conflicts`
    - _Requirements: 4.11_
    - _Properties: 5_
  - [x] 3.5 Wire write-back into the CMJCC path (single code path)
    - In `src/jobrec/orchestration/cmjcc.py`, after conflict detection and before returning,
      call `apply_confirmed_updates` when
      `flags.persist_confirmed_updates and flags.use_persistent_memory`; thread the new
      `candidate_state` onto `CMJCCOutput` and emit a `memory_updated` log line
    - Confirm the orchestrator threads `cmjcc_out.candidate_state` forward and that
      `AppService.process_turn` -> `repo.save_turn` -> existing versioned
      `upsert_candidate_state` persists the new version (no new persistence method)
    - _Requirements: 4.2, 4.6, 4.8, 4.9_
  - [ ]* 3.6 Property test: write-back increments version and never mutates input
    - Add to `tests/unit/test_memory_writeback.py`
    - **Property 1: Long-term write-back increments version monotonically and never mutates input**
    - **Validates: Requirements 4.2**
  - [ ]* 3.7 Property test: superseded values deactivated with effective_to
    - Add to `tests/unit/test_memory_writeback.py`
    - **Property 2: Superseded values are deactivated with an effective_to timestamp**
    - **Validates: Requirements 4.3**
  - [ ]* 3.8 Property test: every long-term write is evidence-bound
    - Add to `tests/unit/test_memory_writeback.py`
    - **Property 3: Every long-term write is evidence-bound**
    - **Validates: Requirements 4.4, 4.10, 31.4**
  - [ ]* 3.9 Property test: only long-term, confirmed preferences write to long-term memory
    - Add to `tests/unit/test_memory_writeback.py`
    - **Property 4: Only long-term-scoped, confirmed preferences write to long-term memory**
    - **Validates: Requirements 4.5, 4.7, 4.9**
  - [ ]* 3.10 Property test: non-override conflicts never overwrite long-term memory
    - Add to `tests/unit/test_memory_writeback.py`
    - **Property 5: Non-override conflicts never overwrite long-term memory**
    - **Validates: Requirements 4.11**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass (including the 68 existing tests and new memory/flag tests) and
    `ruff` is clean; ask the user if questions arise.

- [x] 5. R8 — LLM field-level validation and normalization
  - [x] 5.1 Create the field-validation module and normalizers
    - New `src/jobrec/llm/field_validation.py` with `FieldResult` and total normalizers:
      `normalize_salary`, `normalize_work_mode`, `normalize_experience_level`,
      `normalize_skills`, `normalize_location`, `normalize_deadline`, plus `validate_field`
      and `validate_extraction`
    - Salary accepts int/float/string/object and returns
      `{min_salary, max_salary, currency, period}`; enums constrained via `taxonomy.py`;
      each normalizer returns value-or-None + structured warnings without raising
    - Add `taxonomy.canonical_location` if missing
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.11_
    - _Properties: 13, 14, 15_
  - [x] 5.2 Simplify `cmjcc._as_float` to reuse the salary parser (single parser)
    - Refactor `src/jobrec/orchestration/cmjcc.py` `_as_float` to delegate to
      `normalize_salary` so there is exactly one salary parser
    - _Requirements: 8.2, 8.10_
  - [x] 5.3 Hook validation into extraction with repair -> retry -> rule fallback
    - In `src/jobrec/orchestration/orchestrator.py` `_extract`, call `validate_extraction`
      right after `parse_extraction_lenient`; on failure attempt schema repair, then a single
      bounded model retry (hybrid only), then rule fallback, logging a warning/error at each
      step; never drop a stated constraint (preserve as `UNCONFIRMED` with a warning)
    - Record per-field `extraction_method` (`rule`|`llm`) on `ExtractedPreference.metadata`
    - _Requirements: 8.8, 8.9, 31.9_
  - [ ]* 5.4 Property test: salary normalization preserves the stated amount across shapes
    - Add to `tests/unit/test_field_validation.py`
    - **Property 13: Salary normalization preserves the stated amount across input shapes**
    - **Validates: Requirements 8.2, 8.10**
  - [ ]* 5.5 Property test: field validation is total and never silently drops a constraint
    - Add to `tests/unit/test_field_validation.py`
    - **Property 14: Field validation is total and never silently drops a stated constraint**
    - **Validates: Requirements 8.7, 8.9, 8.11, 31.9**
  - [ ]* 5.6 Property test: enum/skills/location/deadline normalizers produce well-typed output
    - Add to `tests/unit/test_field_validation.py`
    - **Property 15: Enum, skills, location, and deadline normalizers produce well-typed output**
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6**
  - [ ]* 5.7 Unit tests for repair/retry/fallback ordering and warning emission
    - Add concrete examples to `tests/unit/test_field_validation.py` (repair->retry->fallback
      order, warning for unrecoverable values)
    - _Requirements: 8.8, 8.11_

- [x] 6. R6 — Scenario-level statistical unit for task success
  - [x] 6.1 Add scenario-level aggregation helper
    - In `src/jobrec_eval/statistics.py` add
      `aggregate_scenario_success(run_metrics, variant, subset=None)` collapsing repeats to
      one binary per scenario (majority vote; even-repeat ties resolve to 0)
    - _Requirements: 6.1, 6.2, 6.5, 6.6_
  - [x] 6.2 Move task-success McNemar pairing to the scenario level in `compare`
    - Update `src/jobrec_eval/statistics.py` `compare` so that for `metric == "task_success"`
      it pairs on scenario ids (`n_pairs == number of scenarios`), and add reporting fields:
      `scenario_count`, `total_run_count`, `repeats_per_scenario`, `valid_pairs`,
      `discordant_pairs`
    - _Requirements: 6.3, 6.4, 6.7, 6.8_
  - [ ]* 6.3 Update existing eval-stats tests for scenario-level pairing
    - Update `tests/eval/test_eval_stats.py` fixtures/expectations to scenario-level
      `n_pairs`, keeping them green
    - _Requirements: 6.3, 6.4, 31.10_
  - [ ]* 6.4 Property test: task-success McNemar pairs at the scenario level
    - Add to `tests/eval/test_eval_stats.py`
    - **Property 8: Task-success McNemar pairs at the scenario level**
    - **Validates: Requirements 6.1, 6.3**
  - [ ]* 6.5 Property test: deterministic repeat duplication changes neither pairs nor p-values
    - Add to `tests/eval/test_eval_stats.py`
    - **Property 9: Deterministic repeat duplication does not change pairs or p-values**
    - **Validates: Requirements 6.2, 6.7, 6.8**

- [x] 7. R9 — PostgreSQL persistence, fail-fast, and migration versioning
  - [x] 7.1 Add fail-fast experiment-mode behaviour to service construction
    - Update `src/jobrec/app_service.py` `build_default_service` to accept `require_db` and
      raise `RuntimeError` (new `ErrorCode.DB_UNAVAILABLE`) when the DB is required but
      unreachable; `require_db` true when `JOBREC_REQUIRE_DB=1` or
      `config.project.environment in {experiment, production}`; deterministic unit tests pass
      `require_db=False`
    - Add `ErrorCode.DB_UNAVAILABLE` to `src/jobrec/domain/enums.py`
    - _Requirements: 9.6, 31.9_
  - [x] 7.2 Add the schema_version table and migrations module
    - Add `SchemaVersion` model to `src/jobrec/storage/models.py`; create
      `src/jobrec/storage/migrations.py` with an ordered list of idempotent migration
      callables and `ensure_schema_version(engine)`; call it after
      `Base.metadata.create_all` in `src/jobrec/storage/db.py`
    - _Requirements: 9.7_
  - [x] 7.3 Record db/migration versions on the run record
    - Add `db_version: str | None` and `migration_version: int | None` to
      `src/jobrec/domain/run_record.py`; add `SqlRepository.versions() -> dict` returning
      server `version()` and current `schema_version`; populate both in orchestrator `_finish`
    - _Requirements: 9.8_
  - [x] 7.4 Verify restart-recovery loaders reconstruct state
    - Confirm/adjust `src/jobrec/storage/repositories.py` loaders
      (`get_candidate_state`, `get_latest_dialogue_state`, `get_run` with
      include_states/evidence/handoffs) so a rebuilt `AppService` resumes the same session
      with valid evidence links and preserved version history
    - _Requirements: 9.2, 9.3, 9.4, 9.5_
  - [x] 7.5 Add the single-command PG test target
    - Extend the `test-postgres` target in `Makefile` (or add `make test-pg`) to run
      `scripts/pg_local.sh` up -> `pytest -m postgres` -> down
    - _Requirements: 9.1_
  - [ ]* 7.6 PostgreSQL integration tests for persistence and restart
    - Add `tests/integration/test_pg_persistence.py` marked `@pytest.mark.postgres` covering
      save/restore of all state records, session continuation after restart, valid evidence
      links, preserved candidate version history; skip cleanly without a DB
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.9_
  - [ ]* 7.7 Unit test for fail-fast behaviour
    - Add to `tests/unit` a test asserting experiment mode with no DB raises
      `DB_UNAVAILABLE` and never falls back to in-memory
    - _Requirements: 9.6_

- [x] 8. R11 (manifest) + run-bundle/artifact saving for the final comparison
  - [x] 8.1 Build the run manifest
    - Create `src/jobrec/evaluation/manifest.py` `build_run_manifest(config, run_record,
      versions)` capturing commit hash, python/dependency versions, OS/CPU/memory/API
      summary, `config_hash`/`catalog_hash`/`prompt_hash`, resolved `feature_flags`, and
      db/migration versions
    - _Requirements: 11.2, 11.3_
  - [x] 8.2 Emit run_manifest.json and enriched model_calls into each run bundle
    - Extend `src/jobrec/evaluation/exporters.py` `write_run_bundle` to write
      `run_manifest.json` per run and add per-call `request_params` + `response_metadata` to
      the model-call output; aggregate manifests into the experiment manifest
    - _Requirements: 11.1, 1.3_
  - [x] 8.3 Confirm full/no_memory/no_context comparison path saves complete artifacts
    - Ensure `src/jobrec/evaluation/experiment_runner.py` writes resolved config, catalog,
      scenarios, and manifests for each of the three MDV variants so the comparison is
      reproducible
    - _Requirements: 1.2, 1.3, 32.3_
  - [ ]* 8.4 Unit/contract test for manifest contents
    - Add a test asserting manifest keys (hashes, feature_flags, db/migration versions) are
      present
    - _Requirements: 11.2, 11.3_

- [x] 9. Checkpoint - MDV complete
  - Ensure all tests pass, the 68 existing tests remain green, `ruff` is clean, and the
    `full`/`no_memory`/`no_context` comparison runs end-to-end in deterministic mode; ask the
    user if questions arise.

- [x] 10. R7 — Evaluable clarification dialogue loop
  - [x] 10.1 Implement the SimulatedUser
    - Create `src/jobrec_eval/simulated_user.py` with `SimulatedUser` that maps a
      clarification's `target_fields`/`reason_code` to an answer utterance using the
      `Scenario` reference (acceptable slots / profile / expected outcome), returning None
      when it cannot answer
    - _Requirements: 7.1_
  - [x] 10.2 Add the clarification loop to the experiment runner
    - Extend `src/jobrec/evaluation/experiment_runner.py` `_run_one` for
      `clarification_dependent` scenarios: loop until recommendation success, correct
      no-match, `max_dialogue_turns`, or failure/cannot-answer; enforce the max-turn guard and
      a repeated-slot guard
    - _Requirements: 7.2, 7.6, 7.7, 7.9_
  - [x] 10.3 Persist the per-turn dialogue trace
    - Write `dialogue_trace.jsonl` per run with `{user_utterance, system_action,
      clarification_slot, extracted_value, state_version, termination_reason}` and record
      `response_turns` per run
    - _Requirements: 7.3, 7.8_
  - [x] 10.4 Add necessary/unnecessary clarification scoring
    - In `src/jobrec_eval/metrics_extra.py` add `clarification_efficiency(run_metrics)`:
      classify slots as Necessary (in `scenario.acceptable_slots`) vs Unnecessary and apply
      the penalty so skipping a necessary clarification is never scored more efficient
    - _Requirements: 7.4, 7.5_
  - [ ]* 10.5 Property test: the clarification loop always terminates within max_turns
    - Add to `tests/e2e/test_clarification_loop.py`
    - **Property 10: The clarification loop always terminates within max_turns**
    - **Validates: Requirements 7.6**
  - [ ]* 10.6 Property test: repeated-slot re-asking is guarded and recorded
    - Add to `tests/e2e/test_clarification_loop.py`
    - **Property 11: Repeated-slot re-asking is guarded and recorded**
    - **Validates: Requirements 7.7**
  - [ ]* 10.7 Property test: skipping a necessary clarification is never scored more efficient
    - Add to `tests/eval/test_eval_metrics.py`
    - **Property 12: Skipping a necessary clarification is never scored more efficient**
    - **Validates: Requirements 7.4, 7.5**
  - [ ]* 10.8 E2E test: clarification-dependent scenario runs end-to-end across variants
    - Add to `tests/e2e/test_clarification_loop.py`, asserting differing `response_turns`
      across variants
    - _Requirements: 7.8, 7.9_

- [ ] 11. R10 — Failure-path tests and non-trivial grounding/handoff metrics
  - [ ] 11.1 Add fault-injection support helpers
    - Create `tests/support/fault_injection.py`: a provider raising `LLMTimeout` N times then
      succeeding, a claim factory with dangling evidence ids, and a handoff factory omitting
      required fields
    - _Requirements: 10.4_
  - [ ] 11.2 Add failure/recovery/grounding/handoff-rate metrics
    - In `src/jobrec_eval/metrics_extra.py` add `failure_detection_rate`,
      `recovery_success_rate`, `grounding_rate`, `handoff_success_rate`
    - _Requirements: 10.8_
  - [ ]* 11.3 Failure-path unit tests
    - Add `tests/unit/test_failure_paths.py`: invalid/missing/wrong-field evidence;
      unsupported salary/location/skill claims; schema-invalid and missing-field handoffs;
      agent exception, timeout-with-retry, partial failure with recovery; assert logging of
      event + final status
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  - [ ]* 11.4 Failure-metric integration test
    - Add `tests/integration/test_failure_metrics.py` aggregating a failure-containing
      scenario set
    - _Requirements: 10.8_
  - [ ]* 11.5 Property test: every supported claim resolves to a registered evidence id
    - Add to `tests/unit/test_failure_paths.py`
    - **Property 18: Every supported response claim resolves to a registered evidence id**
    - **Validates: Requirements 10.6, 31.4**
  - [ ]* 11.6 Property test: invalid handoffs prevent a run from being scored as success
    - Add to `tests/unit/test_failure_paths.py`
    - **Property 19: Invalid handoffs prevent a run from being scored as success**
    - **Validates: Requirements 10.7**
  - [ ]* 11.7 Property test: grounding and handoff rates are below 1.0 over failure sets
    - Add to `tests/integration/test_failure_metrics.py`
    - **Property 20: Grounding and handoff rates are below 1.0 over failure-containing sets**
    - **Validates: Requirements 10.9**

- [ ] 12. R32 — Ablation reporting framing (Δmemory / Δcontext)
  - [ ] 12.1 Compute and frame the ablation deltas in the report
    - In `src/jobrec_eval/report.py` compute `Δmemory = M_full − M_no_memory` and
      `Δcontext = M_full − M_no_context`; frame each difference as a "framework mechanism
      contribution under the controlled prototype instantiation" and avoid any claim of
      comprehensive superiority over external frameworks
    - _Requirements: 32.4, 32.5, 32.6_
  - [ ]* 12.2 Unit test for delta computation and framing strings
    - Add a test asserting deltas are computed and the report omits superiority language
    - _Requirements: 32.4, 32.5, 32.6_

- [ ] 13. Checkpoint - P0 complete
  - Ensure all tests pass, the 68 existing tests remain green, and `ruff` is clean; ask the
    user if questions arise.

- [ ] 14. R13 — Extraction-source and fallback statistics
  - [ ] 14.1 Persist extraction method and aggregate source metrics
    - Ensure `extracted_preferences.json` carries per-field `extraction_method` and
      `FieldResult.source`; add `extraction_source_metrics` to
      `src/jobrec_eval/metrics_extra.py` aggregating rule-vs-LLM and fallback counts per
      variant and per scenario type; report `schema_failure_rate`/`fallback_rate` for hybrid
    - _Requirements: 8.12, 13.1, 13.2_

- [ ] 15. R14 — Retrieval-layer evaluation
  - [ ] 15.1 Persist retrieval outcome and compute retrieval metrics
    - Persist `result.retrieval_outcome` to `retrieval_results.json` (initial pool, retrieval
      score, full-catalog-fallback count, pool size, retrieval latency); add
      `retrieval_metrics(bundles)` to `src/jobrec_eval/metrics_extra.py` computing Recall@pool
      and relevant-job coverage against `src/jobrec_eval/relevance.py`, reporting retrieval
      errors separately from ranking errors
    - _Requirements: 14.1, 14.2_

- [ ] 16. R12 — Run-detail API parameters in the SQL repository
  - [ ] 16.1 Implement include_states and include_raw_model_outputs with redaction
    - In `src/jobrec/storage/repositories.py` `get_run`, load state versions when
      `include_states` and model-call payloads when `include_raw_model_outputs`; add a
      `redact(text)` helper honouring `config.logging.redact_candidate_text`
    - _Requirements: 12.1, 12.2, 12.3_
  - [ ]* 16.2 API tests for both parameters and redaction
    - Add `tests/integration/test_run_detail_api.py` exercising both params and redaction
    - _Requirements: 12.4_

- [ ] 17. R15 + R32.7 — Pre-comparison configuration-consistency gate
  - [ ] 17.1 Implement the consistency checker
    - Create `src/jobrec_eval/consistency.py` with `check_consistency(manifests)` and
      `require_consistent(manifests, target_flag_set=None)` verifying equality of catalog/
      scenario/prompt hashes, model settings, top-k, pool size, seed, commit; for ablation
      pairs assert via `flag_diff` that only target-mechanism flags differ; write consistency
      flags into each run manifest
    - _Requirements: 15.1, 15.3, 32.7_
  - [ ] 17.2 Stop report generation on mismatch
    - In `src/jobrec_eval/report.py` call `require_consistent` before generating output and
      halt on mismatch; store `consistency_flags` on `RunRecord`
    - _Requirements: 15.2_
  - [ ]* 17.3 Property test: consistency gate proceeds iff all compared runs match
    - Add to `tests/eval/test_consistency.py`
    - **Property 22: The consistency gate proceeds iff all compared runs match**
    - **Validates: Requirements 15.1, 15.2, 32.7**

- [ ] 18. R16 — Unified checksums for all artifacts
  - [ ] 18.1 Write unified checksums.json and add a verify command
    - Create `src/jobrec/evaluation/checksums.py` `write_checksums(exp_dir)` covering all
      input+output artifacts; add a `verify` subcommand to `src/jobrec_eval/cli.py` that
      recomputes, prints the offending artifact on mismatch, and exits non-zero
    - _Requirements: 16.1, 16.2, 16.3_
  - [ ]* 18.2 Property test: checksums round-trip and detect tampering
    - Add to `tests/eval/test_checksums.py`
    - **Property 23: Checksums round-trip and detect tampering**
    - **Validates: Requirements 16.1, 16.2, 16.3**

- [ ] 19. R17 — Data-quality validation
  - [ ] 19.1 Implement the dataset validator and report
    - Create `src/jobrec_eval/data_quality.py` `validate_dataset(catalog, scenarios)`
      checking duplicate ids, salary min>max, unknown currencies, invalid enums, expired
      deadlines, empty required fields, per-scenario relevance/hard-constraint references, and
      true no-match scenarios (via `JobContextAgent`); emit `data_quality_report.json` with
      offending identifier + violation type
    - _Requirements: 17.1, 17.2, 17.3, 17.4_
  - [ ]* 19.2 Property test: data-quality validation flags every injected defect
    - Add to `tests/eval/test_data_quality.py`
    - **Property 24: Data-quality validation flags every injected defect**
    - **Validates: Requirements 17.1, 17.2, 17.4**

- [ ] 20. R18 — Artifact replay and deterministic recomputation
  - [ ] 20.1 Implement replay recomputation and diff
    - Create `src/jobrec/evaluation/replay_check.py` using `RunMode.REPLAY` + saved
      `model_calls.jsonl` (existing `ReplayProvider`) to recompute key-state hashes (extracted
      slots, state versions, filtered jobs, ranking output, explanation claims) and write
      `replay_diff.json` recording any differences
    - _Requirements: 18.1, 18.2, 18.3, 18.4_
  - [ ]* 20.2 Property test: deterministic replay reproduces identical key-state hashes
    - Add to `tests/golden/test_replay.py`
    - **Property 21: Deterministic replay reproduces identical key-state hashes**
    - **Validates: Requirements 18.2**

- [ ] 21. R19–R24 — Dedicated test suites
  - [ ]* 21.1 Candidate-memory test suite (R19)
    - Complete `tests/unit/test_memory_writeback.py` covering long-term write-back,
      Persistence_Scope handling, versioning, and cross-session inheritance
    - _Requirements: 19.1, 19.2_
  - [ ]* 21.2 Constraint-orchestration test suite (R20)
    - Add `tests/unit/test_constraint_orchestration.py` covering hard-constraint filtering,
      unknown-constraint policy, and no-match diagnosis
    - _Requirements: 20.1, 20.2_
    - _Properties: 16_
  - [ ]* 21.3 Property test: no hard-violating job is selected under the full variant
    - Add to `tests/unit/test_constraint_orchestration.py`
    - **Property 16: No hard-violating job is ever selected under the full variant**
    - **Validates: Requirements 31.2**
  - [ ]* 21.4 Dialogue-conflict test suite (R21)
    - Add `tests/unit/test_dialogue_conflicts.py` covering value mismatch, temporal override,
      and scope mismatch resolutions
    - _Requirements: 21.1, 21.2_
  - [ ]* 21.5 Explanation-grounding test suite (R22)
    - Add `tests/unit/test_explanation_grounding.py` covering supported, unsupported, and
      dropped claims
    - _Requirements: 22.1, 22.2_
  - [ ]* 21.6 Agent-handoff test suite (R23)
    - Add `tests/unit/test_agent_handoff.py` covering valid, schema-invalid, and
      missing-field handoffs
    - _Requirements: 23.1, 23.2_
  - [ ]* 21.7 Variant-isolation test suite (R24)
    - Complete `tests/unit/test_variant_isolation.py` verifying each variant resolves distinct
      FeatureFlags and behaviour plus `flag_diff` attribution
    - _Requirements: 24.1, 24.2_

- [ ] 22. Checkpoint - P1 complete
  - Ensure all tests pass, the 68 existing tests remain green, and `ruff` is clean; ask the
    user if questions arise.

- [ ] 23. R25 — Ranking score-breakdown persistence and top-k table
  - [ ] 23.1 Persist score breakdowns and build the top-k contribution table
    - Confirm `RankedJob.features` breakdown is persisted in `recommendation_decision.json`;
      add `topk_contribution_table(bundles)` to `src/jobrec_eval/metrics_extra.py`
    - _Requirements: 25.1, 25.2_
    - _Properties: 17_
  - [ ]* 23.2 Property test: ranking total_score equals sum of feature contributions
    - Add to `tests/unit/test_memory_and_ranking.py`
    - **Property 17: Ranking total_score equals the sum of feature contributions**
    - **Validates: Requirements 25.1, 31.1**

- [ ] 24. R26 — Secret and configuration management
  - [ ] 24.1 Enforce env-only keys, templates, and startup validation
    - Ensure `src/jobrec/llm/remote_provider.py` reads keys only from env and add a log filter
      that never logs keys; add `config/deterministic.yaml` + `config/hybrid.yaml` templates
      and keep `.env.example`; add `validate_startup(config)` in `src/jobrec/app_service.py`
      that fails fast with an explicit error when required config/API env is missing
    - _Requirements: 26.1, 26.2, 26.3, 26.4_
  - [ ]* 24.2 Unit tests for redaction and startup validation
    - Add tests asserting keys are never logged and missing config fails fast
    - _Requirements: 26.1, 26.3, 26.4_

- [ ] 25. R27 — Structured JSON logging
  - [ ] 25.1 Add the structured JSON logger and per-run trace export
    - Create `src/jobrec/utils/observability.py` emitting
      `{run_id, session_id, scenario_id, variant, component, event, severity}` with
      warning/validation_error/system_failure severities; export `log_trace.jsonl` per run in
      the bundle
    - _Requirements: 27.1, 27.2, 27.3_

- [ ] 26. R28 — Performance tests
  - [ ]* 26.1 Latency tests across catalog sizes
    - Add `tests/perf/test_latency.py` measuring end-to-end and per-component latency (median,
      IQR, P95) for catalog sizes 100/200/300 and reporting LLM latency separately from rule
      latency using `component_latency_ms`
    - _Requirements: 28.1, 28.2, 28.3_

- [ ] 27. R29 — Continuous integration gate
  - [ ] 27.1 Extend CI to gate release tagging
    - Extend `.github/workflows/ci.yml` with jobs for unit tests, ruff lint, type-check,
      coverage, a deterministic smoke eval (`full,no_memory,no_context` repeats=1 on a tiny
      fixture), and data-quality/catalog validation, blocking release tagging on any failure
    - _Requirements: 29.1, 29.2_

- [ ] 28. R30 — Code and version freeze
  - [ ] 28.1 Add the freeze script
    - Create `scripts/freeze.sh` that creates an annotated git tag and records commit hash,
      dependency lock, run instructions, DB schema dump, and a final manifest referencing the
      frozen commit + lock
    - _Requirements: 30.1, 30.2_

- [ ] 29. Regression guard - keep the 68 existing tests green and ruff clean
  - [ ] 29.1 Reconcile the two behavioural changes with the existing suite
    - After the `FeatureFlags` matrix change (task 1) and the `statistics.compare`
      scenario-level change (task 6), update any existing test that assumed
      `one_shot == no_memory` or run-level `n_pairs` so the full existing suite passes
    - _Requirements: 5.6, 6.3, 31.10_
  - [ ] 29.2 Run the full default suite and ruff
    - Run the 68-test default suite plus all new non-optional tests and `ruff` (clean),
      confirming no regressions across all phases
    - _Requirements: 31.10_

- [ ] 30. Final checkpoint - all phases complete
  - Ensure all tests pass, the 68 existing tests remain green, `ruff` is clean, and MDV +
    P0/P1/P2 artifacts are produced; ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (tests) and can be skipped for a faster MVP; core
  implementation tasks are never optional.
- Each task references specific requirements for traceability; `_Properties: N_` lines link
  implementation tasks to the design's Correctness Properties, each realised as a single
  Hypothesis property-based test (>=100 examples) tagged
  `# Feature: cmjcc-experiment-readiness, Property N: <text>`.
- Every phase preserves the single-code-path, hard-filter-before-rank, evidence-bound,
  deterministic-primary, PostgreSQL architecture and keeps the 68 existing tests green.
- Human relevance/grounding annotation is out of scope and intentionally absent from this
  plan.
