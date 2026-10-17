import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ontology_agent.models import Base
from ontology_agent import clinical_models  # noqa: F401
from ontology_agent.clinical_loader import load_directory
from ontology_agent.clinical_qa_router import (
    ClinicalRoutingError,
    answer_clinical_question,
    route_clinical_question,
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


def test_route_matches_demographics_marker():
    route = route_clinical_question(
        "What are the demographics on file for patient e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01?"
    )
    assert route.tool == "get_patient_demographics"
    assert route.args == {"patient_id": "e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01"}


def test_route_raises_on_unmatched_text():
    with pytest.raises(ClinicalRoutingError):
        route_clinical_question("What is the weather today?")


def test_route_raises_when_marker_matches_but_no_id_present():
    with pytest.raises(ClinicalRoutingError):
        route_clinical_question("What are the demographics on file for patient?")


def test_answer_clinical_question_end_to_end(session):
    from ontology_agent.clinical_models import ClinicalPatient

    pid = session.query(ClinicalPatient.id).first()[0]
    answered = answer_clinical_question(
        session, f"What are the demographics on file for patient {pid}?"
    )
    assert not answered.refused
    assert answered.source_object_ids == [pid]


def test_answer_clinical_question_refuses_on_missing_object(session):
    answered = answer_clinical_question(
        session,
        "What are the demographics on file for patient 00000000-0000-0000-0000-000000000000?",
    )
    assert answered.refused
    assert answered.refusal_reason
