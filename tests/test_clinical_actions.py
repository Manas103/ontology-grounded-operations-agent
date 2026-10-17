import pytest

from ontology_agent.clinical_actions import (
    ClinicalActionValidationError,
    ClinicalActionValidator,
    ValidatedClinicalAction,
)

VALID_PLACE_ORDER = {
    "action_type": "place_order",
    "patient_id": "e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01",
    "order_type": "laboratory",
    "description": "Hemoglobin A1c",
    "reason": "Routine monitoring.",
}

VALID_EDIT_NOTE = {
    "action_type": "edit_note",
    "encounter_id": "e7cd7d56-ef89-5c07-0761-c7460f079588",
    "note_text": "Patient reports improvement.",
    "reason": "Documenting visit.",
}


def test_valid_place_order_passes():
    validated = ClinicalActionValidator().validate(VALID_PLACE_ORDER)
    assert isinstance(validated, ValidatedClinicalAction)
    assert validated.action_type == "place_order"


def test_valid_edit_note_passes():
    validated = ClinicalActionValidator().validate(VALID_EDIT_NOTE)
    assert validated.action_type == "edit_note"


@pytest.mark.parametrize(
    "mutated",
    [
        {k: v for k, v in VALID_PLACE_ORDER.items() if k != "reason"},
        {**VALID_PLACE_ORDER, "order_type": "surgical"},
        {**VALID_PLACE_ORDER, "patient_id": "not-a-real-id"},
        {**VALID_PLACE_ORDER, "skip_confirmation": True},
        {**VALID_PLACE_ORDER, "action_type": "delete_patient_record"},
        "place_order for patient x",
        None,
        42,
    ],
)
def test_malformed_place_order_rejected(mutated):
    with pytest.raises(ClinicalActionValidationError):
        ClinicalActionValidator().validate(mutated)


def test_empty_required_string_rejected():
    with pytest.raises(ClinicalActionValidationError):
        ClinicalActionValidator().validate({**VALID_EDIT_NOTE, "note_text": ""})
