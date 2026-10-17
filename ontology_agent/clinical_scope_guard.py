"""Refuses any tool or action request outside the declared catalog, before
anything is ever called.

This is a different property from the Q&A benchmark's "refuses a missing
object id" (`ClinicalObjectNotFoundError`, raised only after a real,
catalog-listed tool was called) and from the write-confirmation
benchmark's "rejects a schema-invalid proposal" (raised only after a
real, catalog-listed action type was matched). `ScopeGuard.check` runs
first, before either of those, and answers a narrower question: is the
requested tool or action name one this agent has at all. An LLM
hallucinating a tool name, a hostile prompt asking for a capability this
agent was never given (deleting a record, running raw SQL, emailing a
result, overriding the confirmation step itself), or a request for an
action type that is a close spelling of a real one, all fail here, named
by the exact unknown identifier, never passed through to `tools.py` or
`clinical_actions.py` at all.
"""
from __future__ import annotations

import dataclasses

from ontology_agent.clinical_actions import CLINICAL_ACTION_SCHEMAS
from ontology_agent.clinical_tools import CLINICAL_TOOL_CATALOG

KNOWN_READ_TOOLS = frozenset(CLINICAL_TOOL_CATALOG.keys())
KNOWN_WRITE_ACTIONS = frozenset(CLINICAL_ACTION_SCHEMAS.keys())
KNOWN_IDENTIFIERS = KNOWN_READ_TOOLS | KNOWN_WRITE_ACTIONS


@dataclasses.dataclass
class ScopeCheckResult:
    requested_name: str
    in_scope: bool
    kind: str | None  # "read_tool" | "write_action" | None when out of scope
    refusal_reason: str | None = None


class ScopeGuard:
    """Stateless membership check against the declared 11 read tools and
    2 write action types. Deliberately has no fallback path: an unknown
    name is refused, never forwarded to a "closest match" guess."""

    def check(self, requested_name: str) -> ScopeCheckResult:
        if requested_name in KNOWN_READ_TOOLS:
            return ScopeCheckResult(requested_name, True, "read_tool")
        if requested_name in KNOWN_WRITE_ACTIONS:
            return ScopeCheckResult(requested_name, True, "write_action")
        return ScopeCheckResult(
            requested_name,
            False,
            None,
            refusal_reason=(
                f"{requested_name!r} is not one of the 11 read tools or 2 write "
                "action types this agent is permitted to call; refusing before "
                "any execution"
            ),
        )
