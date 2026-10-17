"""The held-out clinician question set: 300 questions over the loaded
Synthea FHIR R4 store, 11 templates (one per `clinical_tools` function),
most referencing a resource genuinely present (answerable, with a known
correct tool call and expected citation), some referencing a resource
that was created and then deliberately deleted (refusal-designed), so
"300 of 300 answers cited their sources or refused" has a real population
of both kinds to measure against, exactly as `questions.py` does for the
equipment domain.

Unlike the equipment domain (whose object graph is itself synthetic and
seeded), the answerable half of this question set draws its ids from
real, already-loaded Synthea patient data; only the holdout objects below
are purpose-built and then deleted.
"""
from __future__ import annotations

import dataclasses
import random
import uuid

from sqlalchemy.orm import Session

from ontology_agent.clinical_models import (
    ClinicalCondition,
    ClinicalEncounter,
    ClinicalMedicationRequest,
    ClinicalObservation,
    ClinicalPatient,
    ClinicalProcedure,
)

TOTAL_QUESTIONS = 300
ANSWERABLE_PER_TEMPLATE = 22
# Sums to 58 across 11 templates (3 templates at 6, 8 templates at 5).
_REFUSAL_COUNTS = {
    "get_patient_demographics": 6,
    "list_conditions_for_patient": 6,
    "get_condition_detail": 6,
    "list_medication_requests_for_patient": 5,
    "get_medication_request_status": 5,
    "list_observations_for_patient": 5,
    "get_observation_value": 5,
    "list_encounters_for_patient": 5,
    "get_encounter_detail": 5,
    "list_procedures_for_patient": 5,
    "get_procedure_detail": 5,
}
assert 11 * ANSWERABLE_PER_TEMPLATE + sum(_REFUSAL_COUNTS.values()) == TOTAL_QUESTIONS


@dataclasses.dataclass
class ClinicalQuestion:
    qid: str
    text: str
    tool: str
    args: dict[str, str]
    is_refusal: bool
    expected_source_object_ids: list[str]


_TEMPLATES = [
    (
        "get_patient_demographics",
        lambda a: f"What are the demographics on file for patient {a['patient_id']}?",
        ["patient_id"],
    ),
    (
        "list_conditions_for_patient",
        lambda a: f"What conditions are recorded for patient {a['patient_id']}?",
        ["patient_id"],
    ),
    (
        "get_condition_detail",
        lambda a: f"What is the clinical status of condition {a['condition_id']}?",
        ["condition_id"],
    ),
    (
        "list_medication_requests_for_patient",
        lambda a: f"What medication requests exist for patient {a['patient_id']}?",
        ["patient_id"],
    ),
    (
        "get_medication_request_status",
        lambda a: f"What is the status of medication request {a['medication_request_id']}?",
        ["medication_request_id"],
    ),
    (
        "list_observations_for_patient",
        lambda a: f"What observations are recorded for patient {a['patient_id']}?",
        ["patient_id"],
    ),
    (
        "get_observation_value",
        lambda a: f"What value was recorded for observation {a['observation_id']}?",
        ["observation_id"],
    ),
    (
        "list_encounters_for_patient",
        lambda a: f"What encounters does patient {a['patient_id']} have on record?",
        ["patient_id"],
    ),
    (
        "get_encounter_detail",
        lambda a: f"What was the reason for encounter {a['encounter_id']}?",
        ["encounter_id"],
    ),
    (
        "list_procedures_for_patient",
        lambda a: f"What procedures are recorded for patient {a['patient_id']}?",
        ["patient_id"],
    ),
    (
        "get_procedure_detail",
        lambda a: f"When was procedure {a['procedure_id']} performed?",
        ["procedure_id"],
    ),
]


@dataclasses.dataclass
class LoadedIdPools:
    patient_ids: list[str]
    condition_ids: list[str]
    medication_request_ids: list[str]
    observation_ids: list[str]
    encounter_ids: list[str]
    procedure_ids: list[str]


def load_id_pools(session: Session) -> LoadedIdPools:
    return LoadedIdPools(
        patient_ids=[p.id for p in session.query(ClinicalPatient.id)],
        condition_ids=[c.id for c in session.query(ClinicalCondition.id)],
        medication_request_ids=[m.id for m in session.query(ClinicalMedicationRequest.id)],
        observation_ids=[o.id for o in session.query(ClinicalObservation.id)],
        encounter_ids=[e.id for e in session.query(ClinicalEncounter.id)],
        procedure_ids=[pr.id for pr in session.query(ClinicalProcedure.id)],
    )


def create_and_delete_holdout_objects(session: Session, seed: int) -> dict[str, list[str]]:
    """Creates one standalone patient plus one dependent object of each
    other type, then deletes all of them, returning their (now genuinely
    nonexistent) ids by pool key. Mirrors `questions.create_and_delete_holdout_objects`
    for the equipment domain."""
    rng = random.Random(seed + 101)
    removed_patients = [
        ClinicalPatient(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            given_name="Holdout",
            family_name=f"Patient{i}",
            gender="unknown",
            birth_date=None,
        )
        for i in range(6)
    ]
    session.add_all(removed_patients)
    session.flush()
    anchor_patient_id = removed_patients[0].id

    removed_conditions = [
        ClinicalCondition(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            patient_id=anchor_patient_id,
            encounter_id=None,
            code=None,
            code_display="Holdout condition",
            clinical_status="inactive",
            onset_date=None,
        )
        for _ in range(6)
    ]
    removed_medication_requests = [
        ClinicalMedicationRequest(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            patient_id=anchor_patient_id,
            encounter_id=None,
            medication_code=None,
            medication_display="Holdout medication",
            status="stopped",
            authored_on=None,
        )
        for _ in range(5)
    ]
    removed_observations = [
        ClinicalObservation(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            patient_id=anchor_patient_id,
            encounter_id=None,
            code=None,
            code_display="Holdout observation",
            value_text=None,
            unit=None,
            effective_date=None,
        )
        for _ in range(5)
    ]
    removed_encounters = [
        ClinicalEncounter(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            patient_id=anchor_patient_id,
            status="cancelled",
            class_code="AMB",
            type_display=None,
            period_start=None,
            period_end=None,
            reason_display=None,
        )
        for _ in range(5)
    ]
    removed_procedures = [
        ClinicalProcedure(
            id=f"holdout-{uuid.UUID(int=rng.getrandbits(128))}",
            patient_id=anchor_patient_id,
            encounter_id=None,
            code=None,
            code_display="Holdout procedure",
            status="not-done",
            performed_date=None,
        )
        for _ in range(5)
    ]
    session.add_all(
        removed_conditions
        + removed_medication_requests
        + removed_observations
        + removed_encounters
        + removed_procedures
    )
    session.commit()

    removed_ids = {
        "patient_id": [p.id for p in removed_patients],
        "condition_id": [c.id for c in removed_conditions],
        "medication_request_id": [m.id for m in removed_medication_requests],
        "observation_id": [o.id for o in removed_observations],
        "encounter_id": [e.id for e in removed_encounters],
        "procedure_id": [p.id for p in removed_procedures],
    }

    for coll in (
        removed_conditions,
        removed_medication_requests,
        removed_observations,
        removed_encounters,
        removed_procedures,
    ):
        for obj in coll:
            session.delete(obj)
    for p in removed_patients:
        session.delete(p)
    session.commit()

    return removed_ids


def _expected_sources(tool: str, args: dict[str, str], session: Session) -> list[str]:
    if tool == "get_patient_demographics":
        return [args["patient_id"]]
    if tool == "list_conditions_for_patient":
        patient = session.get(ClinicalPatient, args["patient_id"])
        return [patient.id, *[c.id for c in patient.conditions]]
    if tool == "get_condition_detail":
        return [args["condition_id"]]
    if tool == "list_medication_requests_for_patient":
        patient = session.get(ClinicalPatient, args["patient_id"])
        return [patient.id, *[m.id for m in patient.medication_requests]]
    if tool == "get_medication_request_status":
        return [args["medication_request_id"]]
    if tool == "list_observations_for_patient":
        patient = session.get(ClinicalPatient, args["patient_id"])
        return [patient.id, *[o.id for o in patient.observations]]
    if tool == "get_observation_value":
        return [args["observation_id"]]
    if tool == "list_encounters_for_patient":
        patient = session.get(ClinicalPatient, args["patient_id"])
        return [patient.id, *[e.id for e in patient.encounters]]
    if tool == "get_encounter_detail":
        return [args["encounter_id"]]
    if tool == "list_procedures_for_patient":
        patient = session.get(ClinicalPatient, args["patient_id"])
        return [patient.id, *[p.id for p in patient.procedures]]
    if tool == "get_procedure_detail":
        return [args["procedure_id"]]
    raise ValueError(f"unknown tool {tool}")


_POOL_KEY_BY_ARG = {
    "patient_id": "patient_ids",
    "condition_id": "condition_ids",
    "medication_request_id": "medication_request_ids",
    "observation_id": "observation_ids",
    "encounter_id": "encounter_ids",
    "procedure_id": "procedure_ids",
}


def build_question_set(
    session: Session,
    pools: LoadedIdPools,
    removed_ids: dict[str, list[str]],
    seed: int,
) -> list[ClinicalQuestion]:
    rng = random.Random(seed + 102)
    questions: list[ClinicalQuestion] = []
    qnum = 0

    pool_by_key = {
        "patient_ids": pools.patient_ids,
        "condition_ids": pools.condition_ids,
        "medication_request_ids": pools.medication_request_ids,
        "observation_ids": pools.observation_ids,
        "encounter_ids": pools.encounter_ids,
        "procedure_ids": pools.procedure_ids,
    }

    for tool, phrase_fn, arg_keys in _TEMPLATES:
        for _ in range(ANSWERABLE_PER_TEMPLATE):
            args = {}
            for key in arg_keys:
                args[key] = rng.choice(pool_by_key[_POOL_KEY_BY_ARG[key]])
            qnum += 1
            questions.append(
                ClinicalQuestion(
                    qid=f"CQ-{qnum:04d}",
                    text=phrase_fn(args),
                    tool=tool,
                    args=args,
                    is_refusal=False,
                    expected_source_object_ids=_expected_sources(tool, args, session),
                )
            )

        for _ in range(_REFUSAL_COUNTS[tool]):
            args = {}
            missing_key = arg_keys[0]
            for key in arg_keys:
                if key == missing_key:
                    args[key] = rng.choice(removed_ids[key])
                else:
                    args[key] = rng.choice(pool_by_key[_POOL_KEY_BY_ARG[key]])
            qnum += 1
            questions.append(
                ClinicalQuestion(
                    qid=f"CQ-{qnum:04d}",
                    text=phrase_fn(args),
                    tool=tool,
                    args=args,
                    is_refusal=True,
                    expected_source_object_ids=[],
                )
            )

    rng.shuffle(questions)
    return questions
