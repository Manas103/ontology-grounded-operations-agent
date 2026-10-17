"""Deterministic marker-phrase router for the clinical domain, same
discipline as `qa_router.py`: match question text against a small set of
marker phrases to pick one of the 11 tools, pull the single FHIR resource
id out of the question by regex, and call the tool. Never builds or
executes a free-text query; the only thing it can do is call a function
out of `clinical_tools.CLINICAL_TOOL_CATALOG` by name.

Every clinical question template in `clinical_questions.py` names exactly
one resource id, unlike the equipment domain's one two-id template
(`get_part_usage_in_maintenance_event`), so this router does not need the
equipment router's type-prefix disambiguation: whichever marker matches
decides the tool, and the single id found in the text is that tool's one
required argument.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ontology_agent.clinical_tools import (
    CLINICAL_TOOL_CATALOG,
    CLINICAL_TOOL_REQUIRED_ARGS,
    ClinicalObjectNotFoundError,
    ClinicalToolResult,
)

# Matches a real Synthea FHIR resource id (a UUID, optionally a
# "holdout-" prefixed one from the refusal-designed question set) found
# anywhere in the question text.
ID_PATTERN = re.compile(r"\b((?:holdout-)?[0-9a-fA-F]{8}-[0-9a-fA-F-]{18,})\b")

# Ordered most-specific-marker-first so a longer phrase never loses to a
# shorter substring of itself.
_MARKERS = [
    ("demographics on file for patient", "get_patient_demographics"),
    ("conditions are recorded for patient", "list_conditions_for_patient"),
    ("clinical status of condition", "get_condition_detail"),
    ("medication requests exist for patient", "list_medication_requests_for_patient"),
    ("status of medication request", "get_medication_request_status"),
    ("observations are recorded for patient", "list_observations_for_patient"),
    ("value was recorded for observation", "get_observation_value"),
    ("encounters does patient", "list_encounters_for_patient"),
    ("reason for encounter", "get_encounter_detail"),
    ("procedures are recorded for patient", "list_procedures_for_patient"),
    ("when was procedure", "get_procedure_detail"),
]


class ClinicalRoutingError(Exception):
    """The question could not be matched to a known clinical tool at all."""


@dataclass
class ClinicalRouteResult:
    tool: str
    args: dict[str, str]


def route_clinical_question(text: str) -> ClinicalRouteResult:
    lowered = text.lower()
    tool = None
    for marker, candidate_tool in _MARKERS:
        if marker in lowered:
            tool = candidate_tool
            break
    if tool is None:
        raise ClinicalRoutingError(f"no known intent matched question: {text!r}")

    found_ids = ID_PATTERN.findall(text)
    if not found_ids:
        raise ClinicalRoutingError(
            f"tool {tool!r} needs an id but none was found in question: {text!r}"
        )

    required_arg_names = CLINICAL_TOOL_REQUIRED_ARGS[tool]
    args = {required_arg_names[0]: found_ids[0]}
    return ClinicalRouteResult(tool=tool, args=args)


@dataclass
class ClinicalAnsweredQuestion:
    text: str
    refused: bool
    answer_summary: str | None
    source_object_ids: list[str]
    tool_used: str | None
    refusal_reason: str | None = None


def answer_clinical_question(session: Session, text: str) -> ClinicalAnsweredQuestion:
    try:
        route = route_clinical_question(text)
    except ClinicalRoutingError as exc:
        return ClinicalAnsweredQuestion(
            text=text, refused=True, answer_summary=None, source_object_ids=[],
            tool_used=None, refusal_reason=str(exc),
        )

    tool_fn = CLINICAL_TOOL_CATALOG[route.tool]
    try:
        result: ClinicalToolResult = tool_fn(session, **route.args)
    except ClinicalObjectNotFoundError as exc:
        return ClinicalAnsweredQuestion(
            text=text, refused=True, answer_summary=None, source_object_ids=[],
            tool_used=route.tool, refusal_reason=str(exc),
        )

    return ClinicalAnsweredQuestion(
        text=text, refused=False, answer_summary=result.summary,
        source_object_ids=result.source_object_ids, tool_used=route.tool,
    )
