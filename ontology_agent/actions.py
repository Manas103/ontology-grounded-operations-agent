"""JSON-Schema-defined governed actions and the validating layer.

Every state change the system can propose is an instance of one of the
schemas below. `ActionValidator.validate` is the single gate between any
model or tool output and anything a human ever sees: nothing is applied
automatically, and nothing reaches the approval queue in `approval.py`
without passing this gate first. The schemas use `additionalProperties:
false`, explicit `required`, explicit `enum`s, and id patterns, so
"schema-invalid" has one unambiguous meaning: `jsonschema.validate`
raised.

Five equipment-domain action types: scheduling a maintenance event on a
tool or a chamber, retiring a part, updating a recipe's process
parameters, flagging a chamber for service, and recording part usage on a
maintenance event.
"""
from __future__ import annotations

import dataclasses

import jsonschema

TOOL_ID_PATTERN = r"^TL-\d{4}$"
CHAMBER_ID_PATTERN = r"^CH-\d{5}$"
RECIPE_ID_PATTERN = r"^RC-\d{5}$"
PART_ID_PATTERN = r"^PRT-\d{4}$"
MAINTENANCE_EVENT_ID_PATTERN = r"^ME-\d{6}$"
# A schedule_maintenance target is either a tool or a chamber id.
TARGET_ID_PATTERN = r"^(TL-\d{4}|CH-\d{5})$"

_TARGET_TYPES = ["tool", "chamber"]
_MAINTENANCE_TYPES = ["preventive", "corrective", "calibration", "inspection"]
_SEVERITIES = ["low", "medium", "high", "critical"]
_RECIPE_PARAMETERS = ["temperature_c", "pressure_mtorr", "gas_flow_sccm", "rf_power_w"]

SCHEDULE_MAINTENANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "schedule_maintenance"},
        "target_type": {"type": "string", "enum": _TARGET_TYPES},
        "target_id": {"type": "string", "pattern": TARGET_ID_PATTERN},
        "maintenance_type": {"type": "string", "enum": _MAINTENANCE_TYPES},
        "notes": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": ["action_type", "target_type", "target_id", "maintenance_type", "notes"],
    "additionalProperties": False,
}

RETIRE_PART_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "retire_part"},
        "part_id": {"type": "string", "pattern": PART_ID_PATTERN},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "part_id", "reason"],
    "additionalProperties": False,
}

UPDATE_RECIPE_PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "update_recipe_parameters"},
        "recipe_id": {"type": "string", "pattern": RECIPE_ID_PATTERN},
        "parameter_name": {"type": "string", "enum": _RECIPE_PARAMETERS},
        "new_value": {"type": "number"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "recipe_id", "parameter_name", "new_value", "reason"],
    "additionalProperties": False,
}

FLAG_CHAMBER_FOR_SERVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "flag_chamber_for_service"},
        "chamber_id": {"type": "string", "pattern": CHAMBER_ID_PATTERN},
        "severity": {"type": "string", "enum": _SEVERITIES},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "chamber_id", "severity", "reason"],
    "additionalProperties": False,
}

RECORD_PART_USAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "record_part_usage_in_maintenance_event"},
        "maintenance_event_id": {"type": "string", "pattern": MAINTENANCE_EVENT_ID_PATTERN},
        "part_id": {"type": "string", "pattern": PART_ID_PATTERN},
        "qty_used": {"type": "integer", "minimum": 1, "maximum": 1000},
    },
    "required": ["action_type", "maintenance_event_id", "part_id", "qty_used"],
    "additionalProperties": False,
}

ACTION_SCHEMAS: dict[str, dict] = {
    "schedule_maintenance": SCHEDULE_MAINTENANCE_SCHEMA,
    "retire_part": RETIRE_PART_SCHEMA,
    "update_recipe_parameters": UPDATE_RECIPE_PARAMETERS_SCHEMA,
    "flag_chamber_for_service": FLAG_CHAMBER_FOR_SERVICE_SCHEMA,
    "record_part_usage_in_maintenance_event": RECORD_PART_USAGE_SCHEMA,
}


class ActionValidationError(Exception):
    def __init__(self, reason: str, raw_action: object):
        self.reason = reason
        self.raw_action = raw_action
        super().__init__(reason)


@dataclasses.dataclass
class ValidatedAction:
    action_type: str
    payload: dict


class ActionValidator:
    """The validating layer. Nothing downstream sees an action that has not
    passed `validate()`; that is the entire enforcement claim this class
    exists to make checkable rather than asserted."""

    def validate(self, raw_action: object) -> ValidatedAction:
        if not isinstance(raw_action, dict):
            raise ActionValidationError(
                f"action proposal is not a JSON object (got {type(raw_action).__name__})",
                raw_action,
            )
        action_type = raw_action.get("action_type")
        if action_type not in ACTION_SCHEMAS:
            raise ActionValidationError(
                f"unknown or missing action_type: {action_type!r}", raw_action
            )
        schema = ACTION_SCHEMAS[action_type]
        try:
            jsonschema.validate(instance=raw_action, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ActionValidationError(f"schema violation: {exc.message}", raw_action) from exc
        return ValidatedAction(action_type=action_type, payload=dict(raw_action))
