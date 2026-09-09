from __future__ import annotations

import pytest

from ontology_agent.actions import ActionValidator
from ontology_agent.approval import ApprovalQueue


def test_submit_requires_a_validated_action():
    queue = ApprovalQueue()
    with pytest.raises(TypeError):
        queue.submit({"action_type": "retire_part"})  # a raw dict, not ValidatedAction


def test_validated_action_can_be_submitted_and_decided():
    validator = ActionValidator()
    queue = ApprovalQueue()
    validated = validator.validate(
        {
            "action_type": "flag_chamber_for_service",
            "chamber_id": "CH-00001",
            "severity": "medium",
            "reason": "particle count trending up",
        }
    )
    item = queue.submit(validated)
    assert item.status == "pending"
    assert item in queue.pending()

    decided = queue.decide(item.approval_id, approved=True, decided_by="manas")
    assert decided.status == "approved"
    assert decided not in queue.pending()


def test_nothing_is_applied_automatically_by_the_queue():
    """The queue only ever records a decision; it has no method that
    mutates the object model, which is what makes 'a human approves before
    anything is applied' checkable rather than a comment."""
    import inspect

    from ontology_agent import approval

    source = inspect.getsource(approval)
    assert "ontology_agent.models" not in source
    assert "session.add" not in source
    assert "session.commit" not in source
