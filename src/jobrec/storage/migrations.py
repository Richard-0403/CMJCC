"""Lightweight schema/migration versioning (design R9.7).

This is a deliberately minimal, prototype-grade migration scheme — *not* Alembic.
The schema itself is created via ``Base.metadata.create_all``; this module only
records a *reproducible* schema version and provides a hook for applying any
future idempotent migrations.

``ensure_schema_version(engine)`` is DB-agnostic (works against PostgreSQL as
well as the SQLite/in-memory engines used in tests) and idempotent: a fresh
database with no prior version is treated as version 0 and brought up to
``CURRENT_SCHEMA_VERSION``; repeated calls are safe no-ops.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import SchemaVersion

# A migration is an idempotent callable that upgrades the DB by one version.
Migration = Callable[[Session], None]


def _baseline(session: Session) -> None:
    """Baseline schema (version 1).

    The baseline tables are created by ``Base.metadata.create_all``, so this
    migration only marks that the current schema is in place. It is intentionally
    a no-op and safe to run repeatedly.
    """


# Ordered list of idempotent migration callables. The 1-based index of each
# callable is the schema version it brings the database up to. Append new
# migrations here (never reorder or remove existing entries).
MIGRATIONS: list[Migration] = [
    _baseline,
]

# The current target schema version = number of migrations in the ordered list.
# Exposed as a module-level constant so other modules (e.g. run-record version
# recording) can read the current target without re-deriving it.
CURRENT_SCHEMA_VERSION: int = len(MIGRATIONS)


def _current_version(session: Session) -> int:
    """Return the recorded schema version, treating missing/empty as 0."""
    row = session.get(SchemaVersion, 1)
    if row is None:
        return 0
    return row.version


def ensure_schema_version(engine: Engine | None = None) -> int:
    """Apply any not-yet-applied migrations and record the resulting version.

    Idempotent and DB-agnostic. Safe to call repeatedly and against a fresh DB
    with no prior version (treated as version 0). Returns the resulting schema
    version.
    """
    from .db import make_engine

    engine = engine or make_engine()
    now = datetime.now(UTC)

    with Session(engine) as session:
        row = session.get(SchemaVersion, 1)
        current = row.version if row is not None else 0

        if current < CURRENT_SCHEMA_VERSION:
            # Apply each pending migration in order.
            for target in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
                MIGRATIONS[target - 1](session)

            if row is None:
                row = SchemaVersion(
                    id=1,
                    version=CURRENT_SCHEMA_VERSION,
                    applied_at=now,
                    description=f"schema v{CURRENT_SCHEMA_VERSION}",
                )
                session.add(row)
            else:
                row.version = CURRENT_SCHEMA_VERSION
                row.applied_at = now
                row.description = f"schema v{CURRENT_SCHEMA_VERSION}"
            session.commit()

    return CURRENT_SCHEMA_VERSION
