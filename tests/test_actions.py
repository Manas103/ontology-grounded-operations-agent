from __future__ import annotations

import pytest

from ontology_agent.actions import ActionValidationError, ActionValidator, ValidatedAction


@pytest.fixture()
def validator():
    return ActionValidator()


def test_valid_schedule_maintenance_passes(validator):
    action = {
        "action_type": "schedule_maintenance",
        "target_type": "chamber",
        "target_id": "CH-00001",
        "maintenance_type": "preventive",
        "notes": "Quarterly PM due next week.",
    }
    validated = validator.validate(action)
    assert isinstance(validated, ValidatedAction)
    assert validated.action_type == "schedule_maintenance"


def test_valid_record_part_usage_passes(validator):
    action = {
        "action_type": "record_part_usage_in_maintenance_event",
        "maintenance_event_id": "ME-000001",
        "part_id": "PRT-0001",
        "qty_used": 2,
    }
    assert validator.validate(action).action_type == "record_part_usage_in_maintenance_event"


def test_missing_required_field_rejected(validator):
    action = {"action_type": "retire_part", "part_id": "PRT-0001"}
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_unknown_action_type_rejected(validator):
    with pytest.raises(ActionValidationError):
        validator.validate({"action_type": "delete_everything", "target": "*"})


def test_extra_field_rejected(validator):
    action = {
        "action_type": "retire_part",
        "part_id": "PRT-0001",
        "reason": "obsolete",
        "auto_approve": True,
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_invalid_enum_rejected(validator):
    action = {
        "action_type": "flag_chamber_for_service",
        "chamber_id": "CH-00001",
        "severity": "catastrophic",
        "reason": "smoke observed",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_bad_id_pattern_rejected(validator):
    action = {
        "action_type": "retire_part",
        "part_id": "not-an-id",
        "reason": "x",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_negative_quantity_rejected(validator):
    action = {
        "action_type": "record_part_usage_in_maintenance_event",
        "maintenance_event_id": "ME-000001",
        "part_id": "PRT-0001",
        "qty_used": -1,
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_non_dict_input_rejected(validator):
    for bad in ["retire it", ["retire_part"], 42, None]:
        with pytest.raises(ActionValidationError):
            validator.validate(bad)


def test_wrong_type_for_quantity_rejected(validator):
    action = {
        "action_type": "record_part_usage_in_maintenance_event",
        "maintenance_event_id": "ME-000001",
        "part_id": "PRT-0001",
        "qty_used": "two",
    }
    with pytest.raises(ActionValidationError):
        validator.validate(action)


def test_schedule_maintenance_accepts_a_tool_target(validator):
    action = {
        "action_type": "schedule_maintenance",
        "target_type": "tool",
        "target_id": "TL-0001",
        "maintenance_type": "calibration",
        "notes": "Annual calibration.",
    }
    assert validator.validate(action).action_type == "schedule_maintenance"


def test_update_recipe_parameters_valid(validator):
    action = {
        "action_type": "update_recipe_parameters",
        "recipe_id": "RC-00001",
        "parameter_name": "temperature_c",
        "new_value": 210.5,
        "reason": "Process engineering approved shift.",
    }
    assert validator.validate(action).action_type == "update_recipe_parameters"
