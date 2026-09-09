"""PostgreSQL-backed tests. Skip cleanly when PostgreSQL is unreachable
(this build machine does not run it as a service; see README and
`ontology_agent/db.py`), following the same precedent as
`model-validation-alerting` and `payment-exception-triage`.

Stand up a real PostgreSQL locally to exercise this file:

    createdb opsagent
    set OPERATIONS_AGENT_DATABASE_URL=postgresql+psycopg2://postgres@localhost/opsagent
    python -m pytest tests/test_postgres_backend.py -q
"""
from __future__ import annotations

import pytest

from ontology_agent.db import (
    init_db,
    make_engine,
    make_session_factory,
    postgres_is_reachable,
    resolve_database_url,
)
from ontology_agent.seed import DEFAULT_SEED, seed_database
from ontology_agent.tools import get_tool_status

pytestmark = pytest.mark.skipif(
    not postgres_is_reachable(),
    reason="PostgreSQL is not reachable at OPERATIONS_AGENT_DATABASE_URL "
    "(not installed as a service on the build machine); this module "
    "documents the property against a real PostgreSQL instance when one is available",
)


@pytest.fixture()
def postgres_session():
    engine = make_engine(resolve_database_url())
    init_db(engine, drop_first=True)
    Session = make_session_factory(engine)
    with Session() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)
        yield session, graph
    init_db(engine, drop_first=True)


def test_typed_model_round_trips_through_real_postgresql(postgres_session):
    session, graph = postgres_session
    tool_id = graph.tool_ids[0]
    result = get_tool_status(session, tool_id=tool_id)
    assert tool_id in result.source_object_ids
