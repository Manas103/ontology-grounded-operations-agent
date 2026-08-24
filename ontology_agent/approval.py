"""The human approval queue.

`ApprovalQueue.submit` only accepts a `ValidatedAction`, which can only be
constructed by `ActionValidator.validate` succeeding. There is no
constructor path that lets an unvalidated dict reach this queue; that is
what makes "0 schema-invalid actions produced" (meaning: reached the human)
a property of the type system here, not just a property of how the
benchmark script happens to call things. Nothing in this queue is ever
applied to the object model automatically; `decide()` only records a human
decision, it never mutates `models.py` tables itself.
"""
from __future__ import annotations

import dataclasses
import itertools

from ontology_agent.actions import ValidatedAction

_counter = itertools.count(1)


@dataclasses.dataclass
class PendingApproval:
    approval_id: str
    action: ValidatedAction
    status: str = "pending"  # pending | approved | rejected
    decided_by: str | None = None


class ApprovalQueue:
    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}

    def submit(self, action: ValidatedAction) -> PendingApproval:
        if not isinstance(action, ValidatedAction):
            # Defence in depth: even if a caller bypasses type hints, this
            # queue refuses anything that did not come out of the validator.
            raise TypeError("ApprovalQueue.submit requires a ValidatedAction")
        approval_id = f"APR-{next(_counter):06d}"
        item = PendingApproval(approval_id=approval_id, action=action)
        self._items[approval_id] = item
        return item

    def decide(self, approval_id: str, approved: bool, decided_by: str) -> PendingApproval:
        item = self._items[approval_id]
        item.status = "approved" if approved else "rejected"
        item.decided_by = decided_by
        return item

    def pending(self) -> list[PendingApproval]:
        return [i for i in self._items.values() if i.status == "pending"]

    def all_items(self) -> list[PendingApproval]:
        return list(self._items.values())
