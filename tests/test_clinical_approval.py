import pytest

from ontology_agent.clinical_actions import ClinicalActionValidator
from ontology_agent.clinical_approval import NotConfirmedError, WriteConfirmationQueue

VALID_PLACE_ORDER = {
    "action_type": "place_order",
    "patient_id": "e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01",
    "order_type": "laboratory",
    "description": "Hemoglobin A1c",
    "reason": "Routine monitoring.",
}


def test_submit_requires_validated_action():
    queue = WriteConfirmationQueue()
    with pytest.raises(TypeError):
        queue.submit(VALID_PLACE_ORDER)  # a raw dict, not a ValidatedClinicalAction


def test_execute_without_confirmation_raises():
    queue = WriteConfirmationQueue()
    validated = ClinicalActionValidator().validate(VALID_PLACE_ORDER)
    pending = queue.submit(validated)
    with pytest.raises(NotConfirmedError):
        queue.execute(pending.confirmation_id)


def test_confirm_then_execute_succeeds():
    queue = WriteConfirmationQueue()
    validated = ClinicalActionValidator().validate(VALID_PLACE_ORDER)
    pending = queue.submit(validated)
    queue.confirm(pending.confirmation_id, confirmed_by="Dr. Test")
    executed = queue.execute(pending.confirmation_id)
    assert executed.executed is True


def test_rejected_proposal_cannot_be_executed():
    queue = WriteConfirmationQueue()
    validated = ClinicalActionValidator().validate(VALID_PLACE_ORDER)
    pending = queue.submit(validated)
    queue.reject(pending.confirmation_id, confirmed_by="Dr. Test")
    with pytest.raises(NotConfirmedError):
        queue.execute(pending.confirmation_id)
