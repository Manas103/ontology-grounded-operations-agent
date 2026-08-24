from __future__ import annotations

import pytest

from ontology_agent.db import init_db, make_engine, make_session_factory
from ontology_agent.questions import create_and_delete_holdout_objects
from ontology_agent.seed import DEFAULT_SEED, seed_database


@pytest.fixture()
def engine():
    return make_engine("sqlite:///:memory:")


@pytest.fixture()
def session_factory(engine):
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture()
def seeded_session(session_factory):
    with session_factory() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)
        yield session, graph


@pytest.fixture()
def seeded_session_with_holdout(session_factory):
    with session_factory() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)
        removed_ids = create_and_delete_holdout_objects(session, graph, seed=DEFAULT_SEED)
        yield session, graph, removed_ids
