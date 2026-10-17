import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ontology_agent.models import Base
from ontology_agent import clinical_models  # noqa: F401
from ontology_agent.clinical_loader import load_directory
from ontology_agent.clinical_qa_router import answer_clinical_question
from ontology_agent.clinical_questions import (
    TOTAL_QUESTIONS,
    build_question_set,
    create_and_delete_holdout_objects,
    load_id_pools,
)


@pytest.fixture()
def loaded_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    load_directory(s, "tests/fixtures/synthea_sample")
    yield s
    s.close()


def test_question_set_totals_300_with_242_answerable_58_refusal(loaded_session):
    removed = create_and_delete_holdout_objects(loaded_session, seed=20261017)
    pools = load_id_pools(loaded_session)
    questions = build_question_set(loaded_session, pools, removed, seed=20261017)
    assert len(questions) == TOTAL_QUESTIONS == 300
    answerable = [q for q in questions if not q.is_refusal]
    refusal = [q for q in questions if q.is_refusal]
    assert len(answerable) == 242
    assert len(refusal) == 58


def test_every_answerable_question_cites_and_every_refusal_question_refuses(loaded_session):
    removed = create_and_delete_holdout_objects(loaded_session, seed=5)
    pools = load_id_pools(loaded_session)
    questions = build_question_set(loaded_session, pools, removed, seed=5)
    for q in questions:
        answered = answer_clinical_question(loaded_session, q.text)
        if q.is_refusal:
            assert answered.refused, f"expected refusal for {q.text!r}"
        else:
            assert not answered.refused, f"expected an answer for {q.text!r}"
            assert set(answered.source_object_ids) == set(q.expected_source_object_ids)
