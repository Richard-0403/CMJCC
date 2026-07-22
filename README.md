# CMJCC — Agent-Oriented Conversational Job Recommendation

A research prototype for **constraint-aware conversational job recommendation**
built around the **Candidate-Memory and Job-Context Connector (CMJCC)**.

The design goal is not the most elaborate multi-agent system, but one that is
**inspectable, typed, traceable, constraint-verifiable and reproducible**:

- Candidate preferences, dialogue evidence, job context, constraint decisions,
  ranking features and explanation claims are **structured, typed objects** — not
  text hidden inside an LLM context.
- **Hard constraints filter before ranking; soft preferences only score.**
- The **LLM never makes final factual judgements** — extraction and phrasing may
  use an LLM, but eligibility, scoring, evidence binding and logging are
  deterministic code.
- **Every factual claim in a response is bound to an evidence id** and validated.
- One codebase supports the **full system, baselines and ablations** via feature
  flags.

## Architecture (six layers)

```
1. Interface            REST API (FastAPI) / CLI (Typer)
2. Orchestration        ConversationOrchestrator + WorkflowState machine
3. Candidate & Memory   CandidateUnderstandingAgent, MemoryAgent
4. Job Context          JobContextAgent, CMJCC (state coordination)
5. Retrieval/Rank/Expl. HybridRetriever, RankingAgent, ExplanationAgent
6. Data & Logging       PostgreSQL, EvidenceStore, RunRecord/Handoff/EvidenceLog
```

The **CMJCC** validates state, merges candidate profile + dialogue evidence into
an **ActiveSearchState**, resolves conflicts, decides clarifications and produces
a typed **constraint bundle** (`JobContextState`). It does not rank, retrieve or
generate language.

## Quickstart

```bash
make install                 # create .venv and install (editable) with dev extras
make prepare-data            # generate + normalise a 200-job catalog
make demo                    # one-shot recommendation from a profile + utterance
make test                    # deterministic tests + coverage
```

One-shot recommendation via CLI:

```bash
python -m jobrec.cli.main recommend \
  --profile data/fixtures/candidate.json \
  --query "I want a junior data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000." \
  --config configs/experiment_full.yaml
```

## Run modes

- `deterministic` — no remote LLM; rule-based extraction + fixed mock. Used by
  tests and CI.
- `hybrid` — LLM does slot extraction and phrasing; rules do merge, filtering,
  ranking and evidence checks. Falls back to rules on any LLM failure.
- `replay` — replays recorded model responses for exact reproduction.

## Experiment variants (one code path)

| variant        | profile | current turn | prior dialogue | persistent memory | explicit hard/soft |
|----------------|:------:|:------------:|:--------------:|:-----------------:|:------------------:|
| `full`         | ✓ | ✓ | ✓ | ✓ | ✓ |
| `profile_only` | ✓ | ✗ | ✗ | ✓ | ✓ |
| `one_shot`     | ✓ | ✓ | ✗ | ✗ | ✓ |
| `no_memory`    | ✓ | ✓ | ✗ | ✗ | ✓ |
| `no_context`   | ✓ | ✓ | ✓ | ✓ | ✗ (no hard filter) |

Run the whole scenario suite across all variants and export artifacts:

```bash
python scripts/run_experiments.py \
  --scenarios data/scenarios/scenarios.jsonl \
  --variants full,profile_only,one_shot,no_memory,no_context
```

Each run writes a full bundle (run record, states, extracted preferences, active
search, constraints, retrieval, eligibility, decision, response, claims,
handoffs, evidence log, latencies, model calls, resolved config) plus a batch
manifest, index, failures list and checksums under `artifacts/runs/`.

## API

```bash
make serve   # uvicorn on :8000
```

- `POST /v1/candidates` — create a candidate profile (returns state + evidence ids)
- `POST /v1/sessions` — start a session with an experiment variant
- `POST /v1/sessions/{id}/turns` — send an utterance, get a grounded response
- `GET  /v1/runs/{id}?include_handoffs=&include_evidence=` — inspect a run
- `POST /v1/runs/{id}/replay` — reproduce a run without calling a model
- `GET  /health/live`, `GET /health/ready`

## Database (PostgreSQL)

PostgreSQL is the database (SQLAlchemy 2.x + psycopg3), configured via
`DATABASE_URL`. See **ADR-011** — this overrides the SQLite choice in the
original plan. Docker Compose brings up Postgres + the API:

```bash
docker compose -f docker/docker-compose.yml up --build
```

For a local (non-Docker) database during development:

```bash
source scripts/pg_local.sh && pg_up      # start + create role/db, exports DATABASE_URL
pytest -m postgres                         # run the PostgreSQL-backed test
pg_down                                    # stop
```

The recommendation pipeline is storage-agnostic and runs fully in-memory for
tests and offline experiments, so results never depend on the database.

## Testing

```bash
pytest -m "not postgres"     # deterministic unit/contract/integration/e2e/golden
pytest -m postgres           # requires a reachable DATABASE_URL
```

Golden scenarios and property tests assert the key invariants: no hard-violating
job is ever selected by `full`; `total_score == Σ(feature contributions)`; every
claim resolves to real evidence; top-k is respected; expired jobs are never
recommended; and the ablations differ as designed.

## Reproducibility

Every `RunRecord` carries `config_hash`, `catalog_hash`, `prompt_hash`, a model
manifest and the code version. Ids are content-addressed where determinism
matters. A fixed `reference_date` (2026-01-01) drives deadline/expiry logic.

## Evaluation (`jobrec_eval`)

A standalone evaluation pipeline reads the exported run bundles and produces
RQ4 metrics, ablation analysis (memory / job-context), paired statistics, plots
and an analysis report:

```bash
pip install -e ".[eval]"
python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
python -m jobrec_eval.cli pipeline --repeats 3 --bootstrap-iters 5000
```

Relevance is scored by a transparent **automatic oracle** (not human raters) and
grounding by the claim validator; both limitations are documented in the
generated report. See `evaluation/README.md` and
`evaluation/outputs/{experiment_id}/report/analysis_report.md`.

## Safety boundary

This is a **job-seeker-facing discovery aid**, not a hiring-decision tool. It
does not screen or rank candidates for employers, and protected attributes are
excluded from the recommendation logic. "You are not qualified" is never
asserted; the system reports concrete gaps against stated job requirements.

## Known limitations

- Rule-based extractor is English-first and heuristic (spans, strengths).
- Curated synthetic catalog (~200 postings); not a live job market.
- No embedding/semantic retriever wired by default (hybrid = lexical + structured);
  the semantic weight is redistributed when no provider is configured.
- Remote LLM path is provided but not exercised in CI.

See `docs/adr/` for architecture decisions.
