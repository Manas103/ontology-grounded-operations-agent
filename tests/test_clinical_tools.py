import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ontology_agent.models import Base
from ontology_agent import clinical_models  # noqa: F401
from ontology_agent.clinical_loader import load_directory
from ontology_agent.clinical_tools import (
    CLINICAL_TOOL_CATALOG,
    CLINICAL_TOOL_REQUIRED_ARGS,
    ClinicalObjectNotFoundError,
    get_condition_detail,
    get_patient_demographics,
    list_conditions_for_patient,
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


def _first_patient_id(session):
    from ontology_agent.clinical_models import ClinicalPatient

    return session.query(ClinicalPatient.id).first()[0]


def test_eleven_tools_in_catalog():
    assert len(CLINICAL_TOOL_CATALOG) == 11
    assert set(CLINICAL_TOOL_CATALOG) == set(CLINICAL_TOOL_REQUIRED_ARGS)


def test_get_patient_demographics(session):
    pid = _first_patient_id(session)
    result = get_patient_demographics(session, pid)
    assert result.source_object_ids == [pid]
    assert result.value["gender"] in ("male", "female")


def test_list_conditions_for_patient_cites_patient_and_children(session):
    pid = _first_patient_id(session)
    result = list_conditions_for_patient(session, pid)
    assert result.source_object_ids[0] == pid
    assert set(result.source_object_ids[1:]) == set(result.value)


def test_get_condition_detail(session):
    pid = _first_patient_id(session)
    conditions = list_conditions_for_patient(session, pid)
    cond_id = conditions.value[0]
    result = get_condition_detail(session, cond_id)
    assert result.source_object_ids == [cond_id]
    assert result.value["code_display"]


@pytest.mark.parametrize(
    "tool_name,bad_id",
    [
        ("get_patient_demographics", "00000000-0000-0000-0000-000000000000"),
        ("get_condition_detail", "00000000-0000-0000-0000-000000000000"),
        ("get_medication_request_status", "00000000-0000-0000-0000-000000000000"),
        ("get_observation_value", "00000000-0000-0000-0000-000000000000"),
        ("get_encounter_detail", "00000000-0000-0000-0000-000000000000"),
        ("get_procedure_detail", "00000000-0000-0000-0000-000000000000"),
    ],
)
def test_missing_object_raises(session, tool_name, bad_id):
    tool_fn = CLINICAL_TOOL_CATALOG[tool_name]
    arg_name = CLINICAL_TOOL_REQUIRED_ARGS[tool_name][0]
    with pytest.raises(ClinicalObjectNotFoundError):
        tool_fn(session, **{arg_name: bad_id})
