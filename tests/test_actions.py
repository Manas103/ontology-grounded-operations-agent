from __future__ import annotations

import pytest

from ontology_agent.actions import ActionValidationError, ActionValidator, ValidatedAction


@pytest.fixture()
def validator():
    return ActionValidator()


def test_valid_close_work_order_passes(validator):
    action = {
        "action_type": "close_work_order",
        "work_order_id": "WO-000001",
        "closed_by_technician_id": "TCH-0001",
        "resolution_notes": "Replaced failed contactor and verified operation.",
    }
    validated = validator.validate(action)
    assert isinstance(validated, ValidatedAction)
    assert validated.action_type == "close_work_order"


def test_valid_record_part_usage_passes(validator):
    action = {
        "action_type": "record_part_usage",
        "work_order_id": "WO-000001",
        "part_id": "PRT-0001",
        "qty_used": 2,
    }
    assert validator.validate(action).action_type == "record_part_usage"


def test_missing_required_field_rejected(validator):
    action = {"action_type": "close_work_order", "work_order_id": "WO-000001"}
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_unknown_action_type_rejected(validator):
    with pytest.raises(ActionValidationError):
        validator.validate({"action_type": "delete_everything", "target": "*"})


def test_extra_field_rejected(validator):
    action = {
        "action_type": "reopen_work_order",
        "work_order_id": "WO-000001",
        "reason": "fault recurred",
        "auto_approve": True,
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_invalid_enum_rejected(validator):
    action = {
        "action_type": "update_asset_status",
        "asset_id": "AST-00001",
        "new_status": "on_fire",
        "reason": "smoke observed",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_bad_id_pattern_rejected(validator):
    action = {
        "action_type": "close_work_order",
        "work_order_id": "not-an-id",
        "closed_by_technician_id": "TCH-0001",
        "resolution_notes": "x",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_negative_quantity_rejected(validator):
    action = {
        "action_type": "record_part_usage",
        "work_order_id": "WO-000001",
        "part_id": "PRT-0001",
        "qty_used": -1,
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_non_dict_input_rejected(validator):
    for bad in ["close it", ["close_work_order"], 42, None]:
        with pytest.raises(ActionValidationError):
            validator.validate(bad)


def test_wrong_type_for_quantity_rejected(validator):
    action = {
        "action_type": "record_part_usage",
        "work_order_id": "WO-000001",
        "part_id": "PRT-0001",
        "qty_used": "two",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)
