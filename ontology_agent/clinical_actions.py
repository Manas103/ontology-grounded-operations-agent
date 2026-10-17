"""JSON-Schema-defined clinical write proposals and the validating layer.

Two intents this agent can propose, matching the resume's own language:
an order-placing intent (`place_order`, standing in for a medication,
laboratory, or imaging order) and a note-editing intent (`edit_note`,
appending a clinical note to an encounter). Neither is ever applied to
`clinical_models.py` directly. `ClinicalActionValidator.validate` is the
single schema gate between any model or tool output and the write
-confirmation queue in `clinical_approval.py`; nothing reaches that queue
without passing this gate, and `additionalProperties: false` plus
explicit `enum`s and id patterns mean "schema-invalid" has one
unambiguous meaning: `jsonschema.validate` raised.
"""
from __future__ import annotations

import dataclasses

import jsonschema

PATIENT_ID_PATTERN = r"^[0-9a-fA-F-]{8,64}$"
ENCOUNTER_ID_PATTERN = r"^[0-9a-fA-F-]{8,64}$"

_ORDER_TYPES = ["medication", "laboratory", "imaging"]

PLACE_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "place_order"},
        "patient_id": {"type": "string", "pattern": PATIENT_ID_PATTERN},
        "order_type": {"type": "string", "enum": _ORDER_TYPES},
        "description": {"type": "string", "minLength": 1, "maxLength": 500},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "patient_id", "order_type", "description", "reason"],
    "additionalProperties": False,
}

EDIT_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "edit_note"},
        "encounter_id": {"type": "string", "pattern": ENCOUNTER_ID_PATTERN},
        "note_text": {"type": "string", "minLength": 1, "maxLength": 4000},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "encounter_id", "note_text", "reason"],
    "additionalProperties": False,
}

CLINICAL_ACTION_SCHEMAS: dict[str, dict] = {
    "place_order": PLACE_ORDER_SCHEMA,
    "edit_note": EDIT_NOTE_SCHEMA,
}


class ClinicalActionValidationError(Exception):
    def __init__(self, reason: str, raw_action: object):
        self.reason = reason
        self.raw_action = raw_action
        super().__init__(reason)


@dataclasses.dataclass
class ValidatedClinicalAction:
    action_type: str
    payload: dict


class ClinicalActionValidator:
    """The schema gate. Nothing downstream sees a write intent that has not
    passed `validate()`."""

    def validate(self, raw_action: object) -> ValidatedClinicalAction:
        if not isinstance(raw_action, dict):
            raise ClinicalActionValidationError(
                f"action proposal is not a JSON object (got {type(raw_action).__name__})",
                raw_action,
            )
        action_type = raw_action.get("action_type")
        if action_type not in CLINICAL_ACTION_SCHEMAS:
            raise ClinicalActionValidationError(
                f"unknown or missing action_type: {action_type!r}", raw_action
            )
        schema = CLINICAL_ACTION_SCHEMAS[action_type]
        try:
            jsonschema.validate(instance=raw_action, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ClinicalActionValidationError(
                f"schema violation: {exc.message}", raw_action
            ) from exc
        return ValidatedClinicalAction(action_type=action_type, payload=dict(raw_action))
