"""Database engine / session management.

PostgreSQL is the database for this prototype (SQLAlchemy 2.x + psycopg3),
configured through the ``DATABASE_URL`` environment variable. Docker Compose
provides a ``postgres`` service on the default URL.

The core recommendation pipeline does not require a live database; persistence
is optional and injected. ``is_database_available`` lets callers degrade to an
in-memory repository explicitly (never silently) when no DB is reachable.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://jobrec:jobrec@localhost:5432/jobrec"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def make_engine(url: str | None = None, echo: bool = False):
    return create_engine(url or database_url(), echo=echo, future=True, pool_pre_ping=True)


def make_session_factory(engine=None):
    engine = engine or make_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def create_all(engine=None) -> None:
    """Create all tables. Imports models to register them on the metadata."""
    from . import models  # noqa: F401  (registers tables)

    engine = engine or make_engine()
    Base.metadata.create_all(engine)


def is_database_available(url: str | None = None) -> bool:
    """Return True if a database connection can be established."""
    try:
        engine = make_engine(url)
        with engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
