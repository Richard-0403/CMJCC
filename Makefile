.PHONY: install gate-local gate-list lint typecheck test test-unit test-e2e test-perf \
        test-postgres test-pg prepare-data build-index serve demo run-experiments \
        export-artifacts clean

VENV ?= .venv
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# The gate. scripts/preflight.py owns the definition of every check, and CI calls the
# same script with --only per job, so a green local run and a green CI run mean the same
# thing. The individual targets below delegate to it rather than re-spelling the
# commands, which is how they drifted apart before: this file used to lint a narrower
# path set, swallow mypy failures with `|| true`, and report coverage with no threshold.
gate-local:
	$(PY) scripts/preflight.py

gate-list:
	$(PY) scripts/preflight.py --list

lint:
	$(PY) scripts/preflight.py --only lint

typecheck:
	$(PY) scripts/preflight.py --only typecheck

test:
	$(PY) scripts/preflight.py --only tests

test-unit:
	$(VENV)/bin/pytest tests/unit tests/contract -q

test-e2e:
	$(VENV)/bin/pytest tests/e2e tests/golden -q

# Latency measurements across catalog sizes 100/200/300 (R28). Excluded from `make test`
# (marker `perf`); the numbers are printed and written to artifacts/reports/perf_latency.json.
test-perf:
	$(VENV)/bin/pytest tests/perf -q

test-postgres:
	$(VENV)/bin/pytest -m postgres -q

# Single-command PostgreSQL integration run: brings up a local PG instance,
# runs the postgres-marked tests against the DATABASE_URL that pg_up exports,
# then always tears the instance down (even if the tests fail).
# scripts/pg_local.sh is sourced (it exposes pg_up/pg_down shell functions and
# exports DATABASE_URL from pg_up), so the whole flow runs in one shell.
test-pg:
	@bash -c 'set -u; \
		source scripts/pg_local.sh; \
		pg_up; \
		$(VENV)/bin/pytest -m postgres -q; \
		status=$$?; \
		pg_down; \
		exit $$status'

prepare-data:
	$(PY) scripts/generate_raw_catalog.py --output data/raw/jobs.csv --count 200
	PYTHONPATH=src $(PY) scripts/prepare_catalog.py --input data/raw/jobs.csv --out-dir data/processed

build-index:
	$(PY) scripts/build_index.py --catalog data/processed/jobs.jsonl --out-dir artifacts/indexes

serve:
	$(VENV)/bin/uvicorn jobrec.api.app:app --host 0.0.0.0 --port 8000

demo:
	$(PY) -m jobrec.cli.main recommend --profile data/fixtures/candidate.json \
		--query "I want a junior data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000." \
		--config configs/experiment_full.yaml

run-experiments:
	$(PY) scripts/run_experiments.py --config configs/experiment_full.yaml \
		--scenarios data/scenarios/scenarios.jsonl \
		--variants full,profile_only,one_shot,no_memory,no_context

export-artifacts:
	@echo "Artifacts are written under artifacts/runs/ by run-experiments"

clean:
	rm -rf artifacts/runs/* artifacts/indexes/* .pytest_cache .coverage
