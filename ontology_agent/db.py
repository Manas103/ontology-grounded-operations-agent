"""Engine and session setup, with an honest PostgreSQL-or-SQLite fallback.

PostgreSQL is not installed as a service on the build machine (the same
situation `model-validation-alerting` documents for Kafka/PostgreSQL). This
module tries a real PostgreSQL connection when `OPERATIONS_AGENT_DATABASE_URL`
points at one, and falls back to a real, on-disk SQLite database otherwise.
Callers (and tests) that specifically need PostgreSQL should call
`postgres_is_reachable()` first and skip cleanly if it returns False; that is
exactly what `tests/test_postgres_backend.py` does.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ontology_agent.models import Base

DEFAULT_SQLITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "operations.db"
)
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

ENV_VAR = "OPERATIONS_AGENT_DATABASE_URL"


def resolve_database_url() -> str:
    """The connection string actually used. Env var wins; SQLite is the default."""
    return os.environ.get(ENV_VAR, DEFAULT_SQLITE_URL)


def postgres_is_reachable(url: str | None = None) -> bool:
    """True only if a real PostgreSQL connection succeeds. Never raises."""
    url = url or resolve_database_url()
    if not url.startswith("postgresql"):
        return False
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 2})
        with engine.connect():
            pass
        engine.dispose()
        return True
    except Exception:
        return False


def make_engine(url: str | None = None) -> Engine:
    url = url or resolve_database_url()
    if url.startswith("sqlite"):
        os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH), exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(url)


def init_db(engine: Engine, drop_first: bool = False) -> None:
    if drop_first:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
