"""The write-confirmation boundary.

The resume's own phrase for this layer is "a write-confirmation boundary
instead of a generic approval queue": every order-placing or note
-editing intent is held as a `PendingWriteProposal` until a human calls
`confirm()`. The property this module exists to make checkable is
"0 of N write proposals executed without confirmation": `execute()`
raises `NotConfirmedError` on anything that has not been confirmed, and
confirming is the only path that flips a proposal's state to executable.
Exactly like `approval.ApprovalQueue` in the equipment domain, nothing
here ever touches `clinical_models.py`; `execute()` only records that the
proposal would now be allowed to run, it does not perform the write
itself, because this repository's claim is about the gate, not about a
real EHR write path.
"""
from __future__ import annotations

import dataclasses
import itertools

from ontology_agent.clinical_actions import ValidatedClinicalAction

_counter = itertools.count(1)


class NotConfirmedError(Exception):
    """Raised when `execute()` is called on a proposal nobody has confirmed."""


@dataclasses.dataclass
class PendingWriteProposal:
    confirmation_id: str
    action: ValidatedClinicalAction
    status: str = "pending"  # pending | confirmed | rejected | executed
    confirmed_by: str | None = None
    executed: bool = False


class WriteConfirmationQueue:
    """Holds validated write proposals until a human confirms or rejects
    them. `submit()` only accepts a `ValidatedClinicalAction`, constructible
    only by `ClinicalActionValidator.validate` succeeding."""

    def __init__(self) -> None:
        self._items: dict[str, PendingWriteProposal] = {}

    def submit(self, action: ValidatedClinicalAction) -> PendingWriteProposal:
        if not isinstance(action, ValidatedClinicalAction):
            raise TypeError("WriteConfirmationQueue.submit requires a ValidatedClinicalAction")
        confirmation_id = f"CONF-{next(_counter):06d}"
        item = PendingWriteProposal(confirmation_id=confirmation_id, action=action)
        self._items[confirmation_id] = item
        return item

    def confirm(self, confirmation_id: str, confirmed_by: str) -> PendingWriteProposal:
        item = self._items[confirmation_id]
        item.status = "confirmed"
        item.confirmed_by = confirmed_by
        return item

    def reject(self, confirmation_id: str, confirmed_by: str) -> PendingWriteProposal:
        item = self._items[confirmation_id]
        item.status = "rejected"
        item.confirmed_by = confirmed_by
        return item

    def execute(self, confirmation_id: str) -> PendingWriteProposal:
        """Marks a proposal executed. Raises unless it was confirmed first;
        this is the single enforcement point for "0 executed without
        confirmation"."""
        item = self._items[confirmation_id]
        if item.status != "confirmed":
            raise NotConfirmedError(
                f"{confirmation_id} is '{item.status}', not confirmed; refusing to execute"
            )
        item.executed = True
        return item

    def pending(self) -> list[PendingWriteProposal]:
        return [i for i in self._items.values() if i.status == "pending"]

    def all_items(self) -> list[PendingWriteProposal]:
        return list(self._items.values())
