import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ontology_agent.models import Base
from ontology_agent import clinical_models  # noqa: F401
from ontology_agent.clinical_loader import load_directory
from ontology_agent.clinical_langgraph_agent import (
    build_clinical_agent_graph,
    deterministic_tool_chooser,
    run_clinical_agent,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    load_directory(s, "tests/fixtures/synthea_sample")
    yield s
    s.close()


def test_graph_compiles():
    session_placeholder = None
    app = build_clinical_agent_graph(session_placeholder, deterministic_tool_chooser)
    assert app is not None


def test_graph_answers_a_real_question(session):
    from ontology_agent.clinical_models import ClinicalPatient

    pid = session.query(ClinicalPatient.id).first()[0]
    app = build_clinical_agent_graph(session, deterministic_tool_chooser)
    result = run_clinical_agent(app, f"What are the demographics on file for patient {pid}?")
    assert result["refused"] is False
    assert result["source_object_ids"] == [pid]


def test_graph_refuses_a_missing_object(session):
    app = build_clinical_agent_graph(session, deterministic_tool_chooser)
    result = run_clinical_agent(
        app,
        "What are the demographics on file for patient 00000000-0000-0000-0000-000000000000?",
    )
    assert result["refused"] is True
    assert result["refusal_reason"]


def test_graph_refuses_unroutable_text(session):
    app = build_clinical_agent_graph(session, deterministic_tool_chooser)
    result = run_clinical_agent(app, "What is the weather today?")
    assert result["refused"] is True
