# Requirements Document

## Introduction

CMJCC is an existing, working research prototype: an agent-oriented conversational job
recommendation system built to support a Master's thesis. The codebase already provides
typed domain models, CMJCC orchestration, deterministic and hybrid (real `gpt-5.5`) run
modes, retrieval/ranking/explanation components, PostgreSQL persistence
(SQLAlchemy 2.x + psycopg3 via `DATABASE_URL`), a FastAPI + CLI surface, a `jobrec_eval`
evaluation pipeline, and 68 passing tests.

This spec does **not** rebuild the prototype. It captures only the **outstanding work**
required to make the code *final-experiment-ready* and *defensible at the viva*, as
enumerated in the author's "Code Completion Checklist". Each checklist item's "完成标准"
(completion criteria) is expressed here as EARS-style acceptance criteria.

Requirements are grouped and prioritised so the author can complete the highest-value work
first:

- **Group A — Minimum Defensible Version (MDV)**: the smallest coherent subset that must
  ship for the thesis to be defensible. This group is a *pointer/grouping* over specific
  P0 requirements plus final-comparison and artifact-saving requirements.
- **Group B — P0 (must complete before final experiment)**.
- **Group C — P1 (strongly recommended before code freeze)**, including dedicated test
  suites.
- **Group D — P2 (engineering/paper quality)**.

**Explicit exclusion.** Human relevance and grounding annotation will be performed
personally by the author. This spec does **not** include requirements to *produce* human
labels. The import/agreement machinery (two-rater relevance/claim files, weighted Cohen's
kappa, oracle-vs-human agreement) already exists in `src/jobrec_eval/annotation.py` and is
treated as **done**; at most this spec includes light requirements for *validating and
importing* human label files when they are supplied.

**Non-negotiable architecture principles** (encoded as cross-cutting non-functional
requirements in Group E) must be preserved throughout: structured typed state;
hard-filter-before-rank; the LLM never makes final factual decisions; every claim is bound
to an evidence id; a single code path with feature flags for every variant; reproducibility
via config/catalog/prompt hashes; deterministic mode as the primary mode; PostgreSQL as the
database via `DATABASE_URL`; never silently degrade; and the 68 existing tests must not
break.

## Glossary

- **CMJCC**: The Candidate-Memory and Job-Context Connector; the orchestration core that
  merges preferences, detects conflicts, and produces the constraint bundle. The name maps
  to its two core mechanisms: candidate memory and job context. It never ranks jobs itself
  and never mutates `CandidateState` implicitly; long-term write-back is performed
  explicitly by the MemoryAgent.
- **System**: The CMJCC application as a whole (orchestrator plus authorised components),
  unless a more specific component is named.
- **Orchestrator**: `ConversationOrchestrator`; drives the workflow state machine and
  produces a unified `RunRecord`.
- **MemoryAgent**: The component that owns `CandidateState`/`DialogueState` creation,
  evidence generation, conflict detection, and version control.
- **CandidateState**: The immutable, versioned record of stable/confirmed candidate
  information. Each change produces a new version.
- **DialogueState**: The versioned, turn-level record of a conversation session.
- **ActiveSearchState**: The per-search working state (constraints/preferences applied to
  the current search).
- **RecommendationDecision**: The typed record of retrieval, eligibility, ranking, and
  selection outcomes for one turn.
- **EvidenceLog / EvidenceItem**: Content-addressed evidence records; every preference
  value and claim traces to one or more evidence ids.
- **Handoff (AgentHandoff)**: A validated transfer of a typed contract between two
  components, recorded with status and validation outcome.
- **Persistence_Scope**: The lifetime of a preference value; one of `long_term`, `session`,
  `active_search`, `turn_only` (see `PersistenceScope`).
- **Temporal_Scope**: The intended temporal applicability of a stated preference (e.g.
  "from now on" vs "this time only"), used together with Persistence_Scope to decide
  long-term write-back.
- **Confirmation_Status**: One of `confirmed`, `unconfirmed`, `inferred`, `rejected`
  (see `ConfirmationStatus`).
- **Experiment_Variant**: One of `full`, `profile_only`, `one_shot`, `no_memory`,
  `no_context` (see `ExperimentVariant`).
- **Feature_Flags**: The resolved behaviour switches for one run (`FeatureFlags`), derived
  from the variant and config; the single mechanism used to differentiate variants.
- **Run_Mode**: One of `deterministic`, `hybrid`, `replay` (see `RunMode`).
- **Deterministic_Mode**: The primary, reproducible mode using the mock provider and rule
  extractor.
- **Hybrid_Mode**: The mode that calls the real `gpt-5.5` provider with rule fallback.
- **Scenario**: A single evaluation case identified by `scenario_id`, with a reference
  answer used to judge success.
- **Repeat_Index**: The index of a repeated run of the same scenario/variant, used only for
  stability/variance analysis.
- **Task_Success**: The binary per-scenario outcome (recommendation success or correct
  no-match) used for paired comparison.
- **Necessary_Clarification**: A clarification question whose answer is required to reach a
  correct recommendation or a correct no-match for a scenario.
- **Unnecessary_Clarification**: A clarification question not required to reach a correct
  outcome for a scenario.
- **RunRecord**: The unified per-turn record produced by the Orchestrator (states,
  handoffs, evidence ids, latencies, hashes, model manifest).
- **Run_Manifest**: The per-run reproducibility descriptor (commit hash, environment,
  config/catalog/prompt hashes, and consistency flags).
- **Config_Hash / Catalog_Hash / Prompt_Hash**: Stable content hashes used to guarantee
  reproducibility and comparability across runs.
- **jobrec_eval**: The evaluation pipeline package that aggregates run artifacts into
  statistics and reports.
- **Evaluation_Pipeline**: The `jobrec_eval` machinery that computes metrics, statistics,
  and reports from run artifacts.
- **Replay**: Regeneration of statistics/reports from saved artifacts, with deterministic
  recomputation and key-state-hash comparison.

## Requirements

---

## Group A — Minimum Defensible Version (MDV) [Highest priority]

This group defines the smallest coherent scope required for a defensible thesis. It is a
**grouping/pointer** over other requirements: completing it means completing P0
Requirements 4, 5, 6, 8, and 9, **plus** the `full` / `no_memory` / `no_context` final
comparison and complete artifact/config saving. Human relevance and grounding annotation
are the author's manual step and are referenced here but not built by this spec.

### Requirement 1: Minimum Defensible Version scope gate

**User Story:** As the thesis author, I want a clearly defined and verifiable Minimum
Defensible Version, so that I can prioritise the work that makes the code defensible at the
viva before spending effort on lower-priority items.

#### Acceptance Criteria

1. THE System SHALL define the Minimum Defensible Version as the union of Requirement 4
   (memory write-back), Requirement 5 (one_shot vs no_memory differentiation),
   Requirement 6 (task-success statistical unit), Requirement 8 (LLM field validation and
   normalization), and Requirement 9 (PostgreSQL persistence and restart recovery).
2. THE System SHALL include, within the Minimum Defensible Version, the ability to produce
   a final comparison across the `full`, `no_memory`, and `no_context` variants.
3. THE System SHALL include, within the Minimum Defensible Version, complete saving of run
   artifacts and configuration required to reproduce the final comparison.
4. WHERE human relevance or grounding labels are required by the final comparison, THE
   System SHALL treat their production as an external manual step and SHALL reference,
   rather than generate, those labels.
5. WHEN every requirement listed in acceptance criterion 1 is satisfied and the final
   `full`/`no_memory`/`no_context` comparison plus artifact/config saving are available,
   THE System SHALL be considered to meet the Minimum Defensible Version.

---

## Group B — P0 (must complete before final experiment)

### Requirement 4: CandidateState long-term memory write-back

**User Story:** As a candidate, I want my durable preferences (e.g. "from now on prefer
hybrid") to be remembered across sessions while one-off preferences (e.g. "this time only
remote") stay confined to the current search, so that long-term memory reflects my real
standing intent and does not get polluted by transient statements.

#### Acceptance Criteria

1. THE MemoryAgent SHALL provide an `apply_confirmed_updates` operation that decides
   long-term writes from Persistence_Scope, Temporal_Scope, confidence, Confirmation_Status,
   and conflict status.
2. WHEN `apply_confirmed_updates` writes a confirmed long-term update, THE MemoryAgent SHALL
   create a new immutable `CandidateState` version with an incremented version number.
3. WHEN a new `CandidateState` version supersedes prior values, THE MemoryAgent SHALL set
   the superseded `PreferenceValue` records to `is_active = false` and set their
   `effective_to` to the update time.
4. WHEN a new long-term value is written, THE MemoryAgent SHALL retain the evidence ids that
   support that value.
5. THE MemoryAgent SHALL distinguish `long_term`, `session`, `active_search`, and
   `turn_only` scopes and SHALL apply write-back only to values whose resolved scope is
   `long_term`.
6. WHEN a candidate states a durable preference (for example "from now on prefer hybrid"),
   THE MemoryAgent SHALL increment the `CandidateState` version such that a subsequently
   started session reads the updated long-term value.
7. WHEN a candidate states a search-scoped preference (for example "this time only remote"),
   THE MemoryAgent SHALL confine the value to the active search and SHALL NOT write it to
   long-term memory.
8. WHILE persisted long-term preferences exist, THE System SHALL make those preferences
   available after an application restart.
9. WHEN a new session is created, THE System SHALL inherit prior values only according to
   each value's Persistence_Scope.
10. FOR every long-term write, THE MemoryAgent SHALL bind the written value to at least one
    evidence id that traces to an utterance or clarification.
11. IF a candidate statement conflicts with an existing long-term value and the conflict
    resolution is not `override`, THEN THE MemoryAgent SHALL NOT overwrite the long-term
    value and SHALL record the conflict.

### Requirement 5: Genuinely distinct `one_shot` and `no_memory` variants

**User Story:** As the thesis author, I want `one_shot` and `no_memory` to be genuinely
different experimental conditions, so that the paper can explain what research question each
variant answers and reviewers cannot dismiss them as identical configurations.

#### Acceptance Criteria

1. WHERE the variant is `one_shot`, THE System SHALL execute through the SAME orchestrator
   code path as all other variants, WHILE Feature_Flags disable prior-dialogue access,
   persistent-memory access, and multi-turn continuation stages.
2. WHERE the variant is `no_memory`, THE System SHALL retain the full agent workflow
   including CMJCC, handoffs, and job-context orchestration, AND SHALL forbid use of prior
   dialogue and persistent `CandidateState` history.
3. THE System SHALL define explicit Feature_Flags per variant in configuration so that the
   `one_shot` and `no_memory` behaviours are distinguishable by configuration alone.
4. WHEN a run executes, THE System SHALL record the actual resolved Feature_Flags for that
   run in the run logs.
5. WHEN a multi-turn, memory-dependent scenario is executed under `one_shot` versus
   `no_memory`, THE System SHALL produce non-identical intermediate states.
6. THE System SHALL include a unit test that fails if the resolved `one_shot` and
   `no_memory` configurations become identical.
7. WHEN the `one_shot` and `no_memory` variants are compared, THE System SHALL produce
   differing configurations, state flow, and logs.
8. THE System SHALL NOT implement `one_shot` as a separate or duplicated pipeline; its
   simplified behaviour SHALL be produced by Feature_Flags on the single shared code path.

### Requirement 6: Correct statistical unit for task success

**User Story:** As the thesis author, I want paired statistical tests computed at the
scenario level, so that repeating stochastic runs does not artificially inflate sample size
or shrink p-values.

#### Acceptance Criteria

1. WHEN a paired statistical test is computed, THE Evaluation_Pipeline SHALL aggregate
   observations to a unique `scenario_id` before pairing.
2. THE Evaluation_Pipeline SHALL use Repeat_Index only for stability and variance analysis
   and SHALL NOT use it to create additional independent paired samples.
3. WHEN the McNemar test is computed for Task_Success, THE Evaluation_Pipeline SHALL pair on
   scenario-level success such that `n_pairs` equals the number of scenarios and not the
   total number of runs.
4. THE Evaluation_Pipeline SHALL report the scenario count, the total run count, the repeats
   per scenario, the number of valid paired scenarios, and the number of discordant pairs.
5. WHERE a configuration is deterministic, THE System SHALL default to one repeat.
6. WHERE a configuration is stochastic or LLM-backed, THE System SHALL permit more than one
   repeat.
7. WHEN the repeat count is changed from 1 to 3, THE Evaluation_Pipeline SHALL NOT triple
   the number of independent paired samples.
8. WHEN deterministic results are duplicated across repeats, THE Evaluation_Pipeline SHALL
   NOT reduce the reported p-values as a result of that duplication.

### Requirement 7: Evaluable clarification dialogue loop

**User Story:** As the thesis author, I want an automated, scenario-level clarification
dialogue loop, so that clarification-dependent scenarios run end-to-end and clarification
behaviour can be measured fairly across variants.

#### Acceptance Criteria

1. THE Evaluation_Pipeline SHALL provide a scenario-level simulated user that answers
   clarification questions using the scenario reference.
2. WHEN a scenario is executed, THE System SHALL continue the clarification loop until one
   of the following occurs: recommendation success, a correct no-match, the maximum turn
   count is reached, or a failure occurs.
3. WHEN each turn completes, THE System SHALL persist the user utterance, the system action,
   the clarification slot, the extracted value, the state version, and the termination
   reason.
4. THE System SHALL define which clarifications are Necessary_Clarification and which are
   Unnecessary_Clarification for a scenario.
5. WHEN a scenario skips a Necessary_Clarification, THE Evaluation_Pipeline SHALL NOT score
   that run as more efficient than a run that asked the necessary question.
6. THE System SHALL enforce a maximum-turn guard so that a scenario cannot loop
   indefinitely.
7. IF the System asks a clarification question that repeats an already-answered slot, THEN
   THE System SHALL apply a repeated-question guard and SHALL record the event.
8. WHEN the same scenario is run under different variants, THE Evaluation_Pipeline SHALL be
   able to produce differing `response_turns` values across variants.
9. WHEN a clarification-dependent scenario is executed, THE System SHALL complete the
   scenario end-to-end without manual intervention.

### Requirement 8: LLM field-level validation and normalization

**User Story:** As the thesis author, I want every LLM-extracted field validated and
normalized against a schema, so that malformed model output never silently drops a
constraint and hybrid runs can report schema-failure and fallback rates.

#### Acceptance Criteria

1. THE System SHALL define a per-field schema and validator for LLM-extracted fields.
2. THE System SHALL normalize salary into a structure containing `min_salary`,
   `max_salary`, `currency`, and `period`.
3. THE System SHALL constrain `work_mode` and `experience_level` to fixed enumerations.
4. THE System SHALL normalize `skills` into an array of strings.
5. THE System SHALL normalize `location` into a canonical form.
6. THE System SHALL normalize `deadline` into a unified date form.
7. WHEN an extracted field is provided as a number, a string, a nested object, missing, of
   the wrong type, or as an invalid enum value, THE System SHALL handle the input without
   raising an unhandled error.
8. IF field validation fails, THEN THE System SHALL attempt schema repair, then retry, then
   apply a rule-based fallback, and SHALL emit a warning or error log entry for the failure.
9. THE System SHALL NOT silently drop a stated constraint during validation or
   normalization.
10. WHEN salary is provided as a string, as a number, or as an object, THE System SHALL
    normalize each form to the salary structure defined in acceptance criterion 2.
11. IF a value cannot be normalized, THEN THE System SHALL emit a structured warning
    describing the field and the reason.
12. WHILE running in Hybrid_Mode, THE Evaluation_Pipeline SHALL report the schema-failure
    rate and the fallback rate.

### Requirement 9: PostgreSQL persistence and restart recovery

**User Story:** As the thesis author, I want durable PostgreSQL persistence with verified
restart recovery, so that experiment runs are reproducible, sessions survive restarts, and
the production experiment never silently falls back to volatile storage.

#### Acceptance Criteria

1. THE System SHALL run repository integration tests against a real PostgreSQL instance
   (for example a Docker or test container).
2. THE System SHALL save and restore `CandidateState`, `DialogueState`,
   `ActiveSearchState`, `RecommendationDecision`, `EvidenceLog`, and `Handoff` records.
3. WHEN the application is restarted, THE System SHALL allow the same session to continue.
4. WHEN state is restored after a restart, THE System SHALL keep evidence links valid.
5. WHEN state is restored after a restart, THE System SHALL preserve the long-term
   `CandidateState` version history.
6. WHILE running in production (experiment) mode, IF the database is unavailable, THEN THE
   System SHALL fail fast and SHALL NOT switch silently to in-memory storage.
7. THE System SHALL provide database migration version management.
8. WHEN a run is recorded, THE System SHALL include the database version and the migration
   version in the run record or Run_Manifest.
9. WHEN the PostgreSQL integration tests are executed, THE System SHALL pass those tests.

### Requirement 10: Failure-path tests for evidence grounding and handoffs

**User Story:** As the thesis author, I want failure-path coverage for evidence grounding
and handoffs, so that unsupported claims and invalid handoffs are detected rather than
scored as success, and grounding metrics are not trivially perfect on the happy path only.

#### Acceptance Criteria

1. THE System SHALL include tests covering an invalid evidence id, a missing source, and a
   claim that references the wrong field.
2. THE System SHALL include tests covering an unsupported salary claim, an unsupported
   location claim, and an unsupported skill claim.
3. THE System SHALL include tests covering a schema-invalid handoff and a handoff missing
   required fields.
4. THE System SHALL include tests covering an agent exception, a timeout with retry, and a
   partial failure with recovery.
5. WHEN a validation failure, a rejected claim, a failed handoff, a retry, or a recovery
   occurs, THE System SHALL log the event and the final status.
6. IF a response claim is unsupported by evidence, THEN THE System SHALL reject or flag that
   claim.
7. IF a handoff is invalid, THEN THE System SHALL NOT count the affected run as a success.
8. THE Evaluation_Pipeline SHALL report the failure-detection rate and the recovery-success
   rate.
9. WHEN grounding and handoff metrics are computed over scenarios that include failure
   paths, THE Evaluation_Pipeline SHALL produce values that are not fixed at 1.000.

---

## Group C — P1 (strongly recommended before code freeze)

### Requirement 11: Enriched model-call logs and full run manifest

**User Story:** As the thesis author, I want richer model-call logs and a complete run
manifest, so that every run is fully reproducible and auditable.

#### Acceptance Criteria

1. THE System SHALL write a `model_calls.jsonl` record that includes, per call, the
   prompt purpose, request parameters, and response metadata.
2. WHEN a run completes, THE System SHALL produce a Run_Manifest that includes the commit
   hash, the Python and dependency versions, and an operating-system, CPU, memory, and API
   summary.
3. THE Run_Manifest SHALL include the Config_Hash, the Catalog_Hash, and the Prompt_Hash for
   the run.

### Requirement 12: Run detail API parameters implemented in the SQL repository

**User Story:** As an evaluator, I want the run-detail endpoint to truly return the
requested detail levels from the SQL repository, so that state and raw-output inspection is
accurate and safe.

#### Acceptance Criteria

1. THE System SHALL implement the `include_states` parameter of `/v1/runs/{run_id}` in the
   SQL repository so that state objects are returned when requested.
2. THE System SHALL implement the `include_raw_model_outputs` parameter of
   `/v1/runs/{run_id}` in the SQL repository so that raw model outputs are returned when
   requested.
3. WHERE raw model outputs are returned, THE System SHALL apply redaction to sensitive
   content.
4. THE System SHALL include API tests that exercise both parameters and the redaction
   behaviour.

### Requirement 13: Extraction-source and fallback statistics

**User Story:** As the thesis author, I want per-variant and per-scenario-type statistics on
rule-based versus LLM extraction and fallback usage, so that the paper can quantify the
contribution of the LLM.

#### Acceptance Criteria

1. WHEN extraction is performed, THE System SHALL record whether each field was extracted by
   rule or by LLM and whether a fallback was used.
2. THE Evaluation_Pipeline SHALL aggregate rule-versus-LLM extraction counts and fallback
   counts per variant and per scenario type.

### Requirement 14: Retrieval-layer evaluation

**User Story:** As the thesis author, I want retrieval-layer metrics separated from ranking
metrics, so that retrieval quality and ranking quality can be assessed independently.

#### Acceptance Criteria

1. WHEN retrieval executes, THE System SHALL record the initial pool, the retrieval score,
   the count of fallbacks to the full catalog, Recall@pool, the relevant-job coverage, the
   pool size, and the retrieval latency.
2. THE Evaluation_Pipeline SHALL report retrieval errors separately from ranking errors.

### Requirement 15: Pre-comparison configuration-consistency verification

**User Story:** As the thesis author, I want a configuration-consistency check before any
comparison report is generated, so that runs compared against each other share the same
catalog, scenarios, prompts, and settings.

#### Acceptance Criteria

1. WHEN a comparison report is requested, THE Evaluation_Pipeline SHALL verify consistency
   of the catalog hash, the scenario hash, the prompt hash, the model settings, the top-k,
   the pool size, the seed, and the commit hash across the runs being compared.
2. IF a consistency mismatch is detected, THEN THE Evaluation_Pipeline SHALL stop report
   generation.
3. WHEN a consistency check is performed, THE System SHALL write the consistency result
   flags into each affected run's Run_Manifest.

### Requirement 16: Unified checksums for all artifacts

**User Story:** As the thesis author, I want a single checksum manifest covering all inputs
and outputs, so that artifact integrity can be verified with one command.

#### Acceptance Criteria

1. THE System SHALL compute checksums for all input and output artifacts and SHALL write
   them into a unified `checksums.json` file.
2. THE System SHALL provide a verify command that validates artifacts against
   `checksums.json`.
3. IF a checksum does not match, THEN THE verify command SHALL report the mismatched
   artifact and SHALL exit with a non-success status.

### Requirement 17: Data-quality validation

**User Story:** As the thesis author, I want machine-readable data-quality validation for
the job catalog and scenarios, so that malformed inputs are caught before an experiment
runs.

#### Acceptance Criteria

1. THE System SHALL validate the input data for duplicate job ids and duplicate scenario
   ids, salary ranges where the minimum exceeds the maximum, unknown currencies, invalid
   `work_mode` values, invalid `experience_level` values, expired deadlines, empty titles,
   empty skills, and empty locations.
2. THE System SHALL validate that each scenario has a relevance label and a hard-constraint
   reference where required, and that each no-match scenario truly has no eligible job.
3. WHEN data-quality validation completes, THE System SHALL emit a machine-readable report
   of the findings.
4. IF a data-quality violation is found, THEN THE System SHALL record the violation in the
   report with the offending identifier and the violation type.

### Requirement 18: Artifact replay and deterministic recomputation

**User Story:** As the thesis author, I want to replay saved artifacts to regenerate
statistics and reports and to verify deterministic recomputation, so that results are
provably reproducible.

#### Acceptance Criteria

1. THE System SHALL replay saved artifacts to regenerate statistics and reports.
2. WHEN Replay recomputes results deterministically, THE System SHALL compare key state
   hashes for the extracted slots, the state versions, the filtered jobs, the ranking
   output, and the explanation claims against the original run.
3. WHEN a Replay comparison completes, THE System SHALL produce a replay diff report.
4. IF a recomputed key state hash differs from the original, THEN THE replay diff report
   SHALL record the difference.

### Requirement 19: Candidate memory test suite

**User Story:** As the thesis author, I want a dedicated candidate-memory test suite, so
that long-term write-back and scope handling remain correct.

#### Acceptance Criteria

1. THE System SHALL include a candidate-memory test suite covering long-term write-back,
   Persistence_Scope handling, versioning, and cross-session inheritance.
2. WHEN the candidate-memory test suite is executed, THE System SHALL pass that suite.

### Requirement 20: Constraint orchestration test suite

**User Story:** As the thesis author, I want a constraint-orchestration test suite, so that
hard-filter-before-rank behaviour is verified.

#### Acceptance Criteria

1. THE System SHALL include a constraint-orchestration test suite covering hard-constraint
   filtering, unknown-constraint policy, and no-match diagnosis.
2. WHEN the constraint-orchestration test suite is executed, THE System SHALL pass that
   suite.

### Requirement 21: Dialogue conflict test suite

**User Story:** As the thesis author, I want a dialogue-conflict test suite, so that
conflict detection and resolution remain correct.

#### Acceptance Criteria

1. THE System SHALL include a dialogue-conflict test suite covering value mismatch, temporal
   override, and scope mismatch resolutions.
2. WHEN the dialogue-conflict test suite is executed, THE System SHALL pass that suite.

### Requirement 22: Explanation grounding test suite

**User Story:** As the thesis author, I want an explanation-grounding test suite, so that
every claim is bound to valid evidence.

#### Acceptance Criteria

1. THE System SHALL include an explanation-grounding test suite covering supported claims,
   unsupported claims, and dropped claims.
2. WHEN the explanation-grounding test suite is executed, THE System SHALL pass that suite.

### Requirement 23: Agent handoff test suite

**User Story:** As the thesis author, I want an agent-handoff test suite, so that contract
validation between components is verified.

#### Acceptance Criteria

1. THE System SHALL include an agent-handoff test suite covering valid handoffs,
   schema-invalid handoffs, and handoffs missing required fields.
2. WHEN the agent-handoff test suite is executed, THE System SHALL pass that suite.

### Requirement 24: Variant isolation test suite

**User Story:** As the thesis author, I want a variant-isolation test suite, so that each
experiment variant remains behaviourally distinct.

#### Acceptance Criteria

1. THE System SHALL include a variant-isolation test suite that verifies each
   Experiment_Variant resolves distinct Feature_Flags and distinct behaviour.
2. WHEN the variant-isolation test suite is executed, THE System SHALL pass that suite.

---

## Group D — P2 (engineering and paper quality)

### Requirement 25: Ranking score breakdown persistence

**User Story:** As the thesis author, I want ranking score breakdowns persisted, so that the
paper can present a top-k contribution table.

#### Acceptance Criteria

1. WHEN ranking produces a result, THE System SHALL persist the per-feature score breakdown
   for each ranked job.
2. THE Evaluation_Pipeline SHALL produce a top-k contribution table from the persisted score
   breakdowns.

### Requirement 26: Secret and configuration management

**User Story:** As the thesis author, I want disciplined secret and configuration
management, so that API keys are never logged and deterministic and hybrid runs use vetted
templates.

#### Acceptance Criteria

1. THE System SHALL read API keys from environment variables and SHALL NOT write API keys to
   any log.
2. THE System SHALL provide a `.env.example` file and configuration templates for
   deterministic and hybrid runs.
3. WHEN the application starts, THE System SHALL validate that required configuration is
   present.
4. IF required configuration is missing at startup, THEN THE System SHALL fail fast with an
   explicit error.

### Requirement 27: Structured JSON logging

**User Story:** As the thesis author, I want structured JSON logging with consistent fields,
so that logs can be filtered by run, session, scenario, variant, component, and event and
exported per run.

#### Acceptance Criteria

1. WHEN the System logs an event, THE System SHALL emit a structured JSON record including
   the run id, session id, scenario id, variant, component, and event.
2. THE System SHALL distinguish warning, validation-error, and system-failure severities in
   the log records.
3. THE System SHALL export a per-run trace of log records.

### Requirement 28: Performance tests

**User Story:** As the thesis author, I want performance tests, so that the paper can report
latency characteristics across catalog sizes and separate LLM latency from rule latency.

#### Acceptance Criteria

1. THE System SHALL measure end-to-end latency and per-component latency and SHALL report
   the median, the interquartile range, and the P95.
2. THE System SHALL measure latency for catalog sizes of 100, 200, and 300 jobs.
3. THE System SHALL report LLM latency separately from rule latency.

### Requirement 29: Continuous integration gate

**User Story:** As the thesis author, I want a CI gate, so that unit tests, linting, type
checks, coverage, a deterministic smoke evaluation, and scenario/catalog validation all pass
before a release tag is created.

#### Acceptance Criteria

1. THE System SHALL run unit tests, lint, type-check, coverage, a deterministic smoke
   evaluation, and scenario and catalog validation in continuous integration.
2. IF any CI check fails, THEN THE System SHALL block tagging a release.

### Requirement 30: Code and version freeze

**User Story:** As the thesis author, I want a reproducible code and version freeze, so that
the exact experiment code, dependencies, and schema are recorded for the viva.

#### Acceptance Criteria

1. WHEN the code is frozen, THE System SHALL create a git tag and SHALL record the commit
   hash, the dependency lock, run instructions, the database schema, and a final manifest.
2. THE final manifest SHALL reference the frozen commit hash and the dependency lock.

---

## Group E — Cross-cutting non-functional requirements

### Requirement 31: Preserve architecture principles and existing tests

**User Story:** As the thesis author, I want all changes to preserve the established
architecture principles and the existing test suite, so that the system remains defensible
and no regressions are introduced.

#### Acceptance Criteria

1. THE System SHALL maintain structured, typed state for all domain objects.
2. THE System SHALL apply hard-constraint filtering before ranking.
3. THE System SHALL NOT allow the LLM to make final factual decisions.
4. THE System SHALL bind every response claim to an evidence id.
5. THE System SHALL implement all Experiment_Variants through a single code path controlled
   by Feature_Flags rather than by forked implementations.
6. THE System SHALL preserve reproducibility through the Config_Hash, the Catalog_Hash, and
   the Prompt_Hash.
7. THE System SHALL keep Deterministic_Mode as the primary run mode.
8. THE System SHALL use PostgreSQL as the database via `DATABASE_URL`.
9. IF a dependency or resource required for the configured behaviour is unavailable, THEN
   THE System SHALL fail explicitly and SHALL NOT silently degrade.
10. WHEN the changes required by this spec are implemented, THE System SHALL keep the 68
    existing tests passing.

---

## Group F — Evaluation attribution (P0)

### Requirement 32: Architecture-level ablation attribution

**User Story:** As the thesis author, I want each ablation comparison to isolate exactly one
framework mechanism, so that the study can attribute observed differences to specific
framework mechanisms rather than to prototype performance in general, and so that the
research positioning (evaluating framework mechanisms, not just a prototype) is defensible.

#### Acceptance Criteria

1. WHEN `full` is compared with `no_memory`, THE System SHALL ensure their configurations
   are identical except for memory-related Feature_Flags.
2. WHEN `full` is compared with `no_context`, THE System SHALL ensure their configurations
   are identical except for context-orchestration Feature_Flags.
3. WHEN any ablation comparison is produced, THE System SHALL use the identical catalog,
   scenarios, top-k, retrieval pool size, ranking weights, prompts, model settings, and
   random seeds across the compared variants.
4. THE Evaluation_Pipeline SHALL compute Δmemory = M_full − M_no_memory and
   Δcontext = M_full − M_no_context for the reported metrics.
5. THE report SHALL frame each observed difference as a "framework mechanism contribution
   under the controlled prototype instantiation".
6. THE report SHALL NOT state or imply comprehensive superiority over all existing external
   frameworks.
7. WHERE Requirement 15 verifies that compared runs SHARE the same catalog, scenario,
   prompt, model settings, top-k, pool size, seed, and commit, THE System SHALL additionally
   require, between `full` and each ablation, that ONLY the target mechanism's Feature_Flags
   differ while all other Feature_Flags remain identical, so that any measured Δ is
   attributable to that single mechanism.
