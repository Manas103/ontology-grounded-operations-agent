"""JSON-Schema-defined governed actions and the validating layer.

Every state change the system can propose is an instance of one of the
schemas below. `ActionValidator.validate` is the single gate between any
model or tool output and anything a human ever sees: nothing is applied
automatically, and nothing reaches the approval queue in `approval.py`
without passing this gate first. The schemas use `additionalProperties:
false`, explicit `required`, explicit `enum`s, and id patterns, so
"schema-invalid" has one unambiguous meaning: `jsonschema.validate`
raised.
"""
from __future__ import annotations

import dataclasses

import jsonschema

WORK_ORDER_ID_PATTERN = r"^WO-\d{6}$"
ASSET_ID_PATTERN = r"^AST-\d{5}$"
TECHNICIAN_ID_PATTERN = r"^TCH-\d{4}$"
PART_ID_PATTERN = r"^PRT-\d{4}$"

_ASSET_STATUSES = ["operational", "degraded", "down", "decommissioned"]

CLOSE_WORK_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "close_work_order"},
        "work_order_id": {"type": "string", "pattern": WORK_ORDER_ID_PATTERN},
        "closed_by_technician_id": {"type": "string", "pattern": TECHNICIAN_ID_PATTERN},
        "resolution_notes": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": ["action_type", "work_order_id", "closed_by_technician_id", "resolution_notes"],
    "additionalProperties": False,
}

REASSIGN_TECHNICIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "reassign_technician"},
        "work_order_id": {"type": "string", "pattern": WORK_ORDER_ID_PATTERN},
        "new_technician_id": {"type": "string", "pattern": TECHNICIAN_ID_PATTERN},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "work_order_id", "new_technician_id", "reason"],
    "additionalProperties": False,
}

REOPEN_WORK_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "reopen_work_order"},
        "work_order_id": {"type": "string", "pattern": WORK_ORDER_ID_PATTERN},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "work_order_id", "reason"],
    "additionalProperties": False,
}

UPDATE_ASSET_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "update_asset_status"},
        "asset_id": {"type": "string", "pattern": ASSET_ID_PATTERN},
        "new_status": {"type": "string", "enum": _ASSET_STATUSES},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["action_type", "asset_id", "new_status", "reason"],
    "additionalProperties": False,
}

RECORD_PART_USAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"const": "record_part_usage"},
        "work_order_id": {"type": "string", "pattern": WORK_ORDER_ID_PATTERN},
        "part_id": {"type": "string", "pattern": PART_ID_PATTERN},
        "qty_used": {"type": "integer", "minimum": 1, "maximum": 1000},
    },
    "required": ["action_type", "work_order_id", "part_id", "qty_used"],
    "additionalProperties": False,
}

ACTION_SCHEMAS: dict[str, dict] = {
    "close_work_order": CLOSE_WORK_ORDER_SCHEMA,
    "reassign_technician": REASSIGN_TECHNICIAN_SCHEMA,
    "reopen_work_order": REOPEN_WORK_ORDER_SCHEMA,
    "update_asset_status": UPDATE_ASSET_STATUS_SCHEMA,
    "record_part_usage": RECORD_PART_USAGE_SCHEMA,
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
