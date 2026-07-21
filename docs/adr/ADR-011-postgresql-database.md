# ADR-011: Use PostgreSQL as the database (overrides SQLite in the landing plan)

## Status
Accepted.

## Context
The engineering landing plan (sections 2.1 and 18) specifies SQLite as the local
database for the research prototype. The project owner requested PostgreSQL
instead.

## Decision
Use **PostgreSQL** as the database, accessed through SQLAlchemy 2.x with the
`psycopg` (v3) driver. The connection is configured via the `DATABASE_URL`
environment variable (default
`postgresql+psycopg://jobrec:jobrec@localhost:5432/jobrec`). Docker Compose runs
a `postgres:15` service that the app connects to.

## Rationale
- Closer parity with a realistic deployment target than SQLite.
- JSON/JSONB columns store large structured objects while metric columns remain
  queryable, matching the storage design in landing-plan section 18.
- Concurrent access and richer types are available if the prototype grows.

## Alternatives considered
- **SQLite (original plan):** zero-setup and great for CI, but the owner
  explicitly requested PostgreSQL.
- **DuckDB:** analytical focus, weaker as an application store.

## Impact on evaluation
- None on the recommendation logic: the core pipeline is storage-agnostic and
  runs fully in-memory (`InMemoryRepository`) for deterministic tests and
  offline experiment runs, so results are unaffected by the database choice.
- Reproducibility is preserved: run bundles and `config_hash` / `catalog_hash` /
  `prompt_hash` do not depend on the storage backend.
- CI runs the deterministic suite without a database; the PostgreSQL-backed test
  is marked `postgres` and skipped when no `DATABASE_URL` is reachable. A local
  instance can be started with `scripts/pg_local.sh` or via Docker Compose.

## Notes
The ORM avoids SQLite-only or PostgreSQL-only SQL so the models remain portable,
but PostgreSQL is the supported and default target.
