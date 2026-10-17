"""The clinical domain's entire read surface: 11 named, typed functions.

Same discipline as `tools.py` for the equipment domain: every function
takes typed parameters, issues a parameterized SQLAlchemy ORM query
against a specific, predefined table, and returns a `ClinicalToolResult`
whose `source_object_ids` names the real FHIR resource id(s) the answer
came from. There is no free-text or SQL-string path in; a question
answering path may only reach the clinical store by calling one of these
functions by name. A missing object raises `ClinicalObjectNotFoundError`
rather than returning a null-shaped answer, exactly as in the equipment
domain, so "cited or refused" is a property every caller is forced to
handle, not a convention a caller might skip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from ontology_agent.clinical_models import (
    ClinicalCondition,
    ClinicalEncounter,
    ClinicalMedicationRequest,
    ClinicalObservation,
    ClinicalPatient,
    ClinicalProcedure,
)


class ClinicalObjectNotFoundError(Exception):
    """Raised when a tool's referenced clinical resource does not exist."""

    def __init__(self, object_type: str, object_id: str):
        self.object_type = object_type
        self.object_id = object_id
        super().__init__(f"no {object_type} with id {object_id!r} in the clinical store")


@dataclass
class ClinicalToolResult:
    summary: str
    value: object
    source_object_ids: list[str] = field(default_factory=list)


def _require_patient(session: Session, patient_id: str) -> ClinicalPatient:
    obj = session.get(ClinicalPatient, patient_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("patient", patient_id)
    return obj


def _require_encounter(session: Session, encounter_id: str) -> ClinicalEncounter:
    obj = session.get(ClinicalEncounter, encounter_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("encounter", encounter_id)
    return obj


def _require_condition(session: Session, condition_id: str) -> ClinicalCondition:
    obj = session.get(ClinicalCondition, condition_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("condition", condition_id)
    return obj


def _require_observation(session: Session, observation_id: str) -> ClinicalObservation:
    obj = session.get(ClinicalObservation, observation_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("observation", observation_id)
    return obj


def _require_medication_request(
    session: Session, medication_request_id: str
) -> ClinicalMedicationRequest:
    obj = session.get(ClinicalMedicationRequest, medication_request_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("medication_request", medication_request_id)
    return obj


def _require_procedure(session: Session, procedure_id: str) -> ClinicalProcedure:
    obj = session.get(ClinicalProcedure, procedure_id)
    if obj is None:
        raise ClinicalObjectNotFoundError("procedure", procedure_id)
    return obj


def get_patient_demographics(session: Session, patient_id: str) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    return ClinicalToolResult(
        summary=(
            f"Patient {patient.id} is {patient.given_name} {patient.family_name}, "
            f"gender {patient.gender}, born {patient.birth_date}."
        ),
        value={
            "given_name": patient.given_name,
            "family_name": patient.family_name,
            "gender": patient.gender,
            "birth_date": str(patient.birth_date) if patient.birth_date else None,
        },
        source_object_ids=[patient.id],
    )


def list_conditions_for_patient(session: Session, patient_id: str) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    ids = [c.id for c in patient.conditions]
    return ClinicalToolResult(
        summary=f"Patient {patient.id} has {len(ids)} recorded condition(s): "
        f"{', '.join(ids) if ids else 'none'}.",
        value=ids,
        source_object_ids=[patient.id, *ids],
    )


def get_condition_detail(session: Session, condition_id: str) -> ClinicalToolResult:
    condition = _require_condition(session, condition_id)
    return ClinicalToolResult(
        summary=f"Condition {condition.id} is '{condition.code_display}', "
        f"clinical status '{condition.clinical_status}'.",
        value={
            "code_display": condition.code_display,
            "clinical_status": condition.clinical_status,
        },
        source_object_ids=[condition.id],
    )


def list_medication_requests_for_patient(
    session: Session, patient_id: str
) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    ids = [m.id for m in patient.medication_requests]
    return ClinicalToolResult(
        summary=f"Patient {patient.id} has {len(ids)} medication request(s): "
        f"{', '.join(ids) if ids else 'none'}.",
        value=ids,
        source_object_ids=[patient.id, *ids],
    )


def get_medication_request_status(
    session: Session, medication_request_id: str
) -> ClinicalToolResult:
    med = _require_medication_request(session, medication_request_id)
    return ClinicalToolResult(
        summary=f"Medication request {med.id} ('{med.medication_display}') "
        f"has status '{med.status}'.",
        value={"medication_display": med.medication_display, "status": med.status},
        source_object_ids=[med.id],
    )


def list_observations_for_patient(session: Session, patient_id: str) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    ids = [o.id for o in patient.observations]
    return ClinicalToolResult(
        summary=f"Patient {patient.id} has {len(ids)} observation(s): "
        f"{', '.join(ids) if ids else 'none'}.",
        value=ids,
        source_object_ids=[patient.id, *ids],
    )


def get_observation_value(session: Session, observation_id: str) -> ClinicalToolResult:
    obs = _require_observation(session, observation_id)
    return ClinicalToolResult(
        summary=f"Observation {obs.id} ('{obs.code_display}') value is "
        f"{obs.value_text} {obs.unit or ''}".strip() + ".",
        value={"value_text": obs.value_text, "unit": obs.unit},
        source_object_ids=[obs.id],
    )


def list_encounters_for_patient(session: Session, patient_id: str) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    ids = [e.id for e in patient.encounters]
    return ClinicalToolResult(
        summary=f"Patient {patient.id} has {len(ids)} encounter(s): "
        f"{', '.join(ids) if ids else 'none'}.",
        value=ids,
        source_object_ids=[patient.id, *ids],
    )


def get_encounter_detail(session: Session, encounter_id: str) -> ClinicalToolResult:
    enc = _require_encounter(session, encounter_id)
    return ClinicalToolResult(
        summary=f"Encounter {enc.id} is class '{enc.class_code}', status "
        f"'{enc.status}', reason '{enc.reason_display or 'unspecified'}'.",
        value={
            "class_code": enc.class_code,
            "status": enc.status,
            "reason_display": enc.reason_display,
        },
        source_object_ids=[enc.id],
    )


def list_procedures_for_patient(session: Session, patient_id: str) -> ClinicalToolResult:
    patient = _require_patient(session, patient_id)
    ids = [p.id for p in patient.procedures]
    return ClinicalToolResult(
        summary=f"Patient {patient.id} has {len(ids)} procedure(s): "
        f"{', '.join(ids) if ids else 'none'}.",
        value=ids,
        source_object_ids=[patient.id, *ids],
    )


def get_procedure_detail(session: Session, procedure_id: str) -> ClinicalToolResult:
    proc = _require_procedure(session, procedure_id)
    return ClinicalToolResult(
        summary=f"Procedure {proc.id} ('{proc.code_display}') performed "
        f"{proc.performed_date}, status '{proc.status}'.",
        value={"code_display": proc.code_display, "performed_date": str(proc.performed_date)},
        source_object_ids=[proc.id],
    )


# The clinical domain's entire read surface: 11 named tools. An LLM or a
# rule router may only reach the clinical store by calling one of these by
# name; there is no other read path in.
CLINICAL_TOOL_CATALOG: dict[str, Callable[..., ClinicalToolResult]] = {
    "get_patient_demographics": get_patient_demographics,
    "list_conditions_for_patient": list_conditions_for_patient,
    "get_condition_detail": get_condition_detail,
    "list_medication_requests_for_patient": list_medication_requests_for_patient,
    "get_medication_request_status": get_medication_request_status,
    "list_observations_for_patient": list_observations_for_patient,
    "get_observation_value": get_observation_value,
    "list_encounters_for_patient": list_encounters_for_patient,
    "get_encounter_detail": get_encounter_detail,
    "list_procedures_for_patient": list_procedures_for_patient,
    "get_procedure_detail": get_procedure_detail,
}

CLINICAL_TOOL_REQUIRED_ARGS: dict[str, list[str]] = {
    "get_patient_demographics": ["patient_id"],
    "list_conditions_for_patient": ["patient_id"],
    "get_condition_detail": ["condition_id"],
    "list_medication_requests_for_patient": ["patient_id"],
    "get_medication_request_status": ["medication_request_id"],
    "list_observations_for_patient": ["patient_id"],
    "get_observation_value": ["observation_id"],
    "list_encounters_for_patient": ["patient_id"],
    "get_encounter_detail": ["encounter_id"],
    "list_procedures_for_patient": ["patient_id"],
    "get_procedure_detail": ["procedure_id"],
}
