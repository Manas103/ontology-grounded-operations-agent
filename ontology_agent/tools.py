"""The constrained tool catalog: the only way anything gets to read the model.

Every function here takes typed, named parameters and issues a parameterized
SQLAlchemy ORM query against a specific, predefined table and column set.
None of them build a SQL string from caller input, none of them expose a
generic "run this query" escape hatch, and none of them return anything
without also returning the exact object id(s) the answer came from. A
question-answering path (deterministic router or LLM) may only reach the
typed object model by calling one of these functions by name; there is no
other path in, which is what "answers operational questions only from a
typed object model" means as a checkable property rather than a slogan.

If the object referenced does not exist, every function raises
`ObjectNotFoundError` rather than returning a null-shaped answer. Callers
(the router, the LLM client, the benchmark harness) are required to turn
that into a refusal, never a guess.

Twelve named tools cover the equipment domain's five object types (tools,
chambers, recipes, parts, maintenance events) and the relationships between
them (a tool has chambers, a chamber runs recipes, a maintenance event
targets a tool or a chamber and consumes parts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from ontology_agent.models import Chamber, MaintenanceEvent, MaintenanceEventPart, Part, Recipe, Tool


class ObjectNotFoundError(Exception):
    """Raised when a tool's referenced object does not exist in the store."""

    def __init__(self, object_type: str, object_id: str):
        self.object_type = object_type
        self.object_id = object_id
        super().__init__(f"no {object_type} with id {object_id!r} in the store")


@dataclass
class ToolResult:
    """Every tool call returns one of these. `source_object_ids` is mandatory."""

    summary: str
    value: object
    source_object_ids: list[str] = field(default_factory=list)


def _require_tool(session: Session, tool_id: str) -> Tool:
    obj = session.get(Tool, tool_id)
    if obj is None:
        raise ObjectNotFoundError("tool", tool_id)
    return obj


def _require_chamber(session: Session, chamber_id: str) -> Chamber:
    obj = session.get(Chamber, chamber_id)
    if obj is None:
        raise ObjectNotFoundError("chamber", chamber_id)
    return obj


def _require_recipe(session: Session, recipe_id: str) -> Recipe:
    obj = session.get(Recipe, recipe_id)
    if obj is None:
        raise ObjectNotFoundError("recipe", recipe_id)
    return obj


def _require_part(session: Session, part_id: str) -> Part:
    obj = session.get(Part, part_id)
    if obj is None:
        raise ObjectNotFoundError("part", part_id)
    return obj


def _require_maintenance_event(session: Session, maintenance_event_id: str) -> MaintenanceEvent:
    obj = session.get(MaintenanceEvent, maintenance_event_id)
    if obj is None:
        raise ObjectNotFoundError("maintenance_event", maintenance_event_id)
    return obj


def get_tool_status(session: Session, tool_id: str) -> ToolResult:
    tool = _require_tool(session, tool_id)
    return ToolResult(
        summary=f"Tool {tool.id} status is '{tool.status}'.",
        value=tool.status,
        source_object_ids=[tool.id],
    )


def get_tool_type(session: Session, tool_id: str) -> ToolResult:
    tool = _require_tool(session, tool_id)
    return ToolResult(
        summary=f"Tool {tool.id} is a '{tool.tool_type}' tool.",
        value=tool.tool_type,
        source_object_ids=[tool.id],
    )


def get_chamber_status(session: Session, chamber_id: str) -> ToolResult:
    chamber = _require_chamber(session, chamber_id)
    return ToolResult(
        summary=f"Chamber {chamber.id} status is '{chamber.status}'.",
        value=chamber.status,
        source_object_ids=[chamber.id],
    )


def get_chamber_tool(session: Session, chamber_id: str) -> ToolResult:
    chamber = _require_chamber(session, chamber_id)
    return ToolResult(
        summary=f"Chamber {chamber.id} belongs to tool {chamber.tool_id}.",
        value=chamber.tool_id,
        source_object_ids=[chamber.id, chamber.tool_id],
    )


def get_recipe_chamber(session: Session, recipe_id: str) -> ToolResult:
    recipe = _require_recipe(session, recipe_id)
    return ToolResult(
        summary=f"Recipe {recipe.id} is assigned to chamber {recipe.chamber_id}.",
        value=recipe.chamber_id,
        source_object_ids=[recipe.id, recipe.chamber_id],
    )


def get_recipe_active(session: Session, recipe_id: str) -> ToolResult:
    recipe = _require_recipe(session, recipe_id)
    return ToolResult(
        summary=f"Recipe {recipe.id} active status is {recipe.is_active}.",
        value=recipe.is_active,
        source_object_ids=[recipe.id],
    )


def get_part_stock(session: Session, part_id: str) -> ToolResult:
    part = _require_part(session, part_id)
    return ToolResult(
        summary=f"Part {part.id} has {part.stock_qty} units in stock.",
        value=part.stock_qty,
        source_object_ids=[part.id],
    )


def get_maintenance_event_status(session: Session, maintenance_event_id: str) -> ToolResult:
    event = _require_maintenance_event(session, maintenance_event_id)
    return ToolResult(
        summary=f"Maintenance event {event.id} is currently '{event.status}'.",
        value=event.status,
        source_object_ids=[event.id],
    )


def get_maintenance_event_target(session: Session, maintenance_event_id: str) -> ToolResult:
    event = _require_maintenance_event(session, maintenance_event_id)
    target_id = event.tool_id if event.tool_id is not None else event.chamber_id
    target_kind = "tool" if event.tool_id is not None else "chamber"
    return ToolResult(
        summary=f"Maintenance event {event.id} was performed on {target_kind} {target_id}.",
        value=target_id,
        source_object_ids=[event.id, target_id],
    )


def list_maintenance_events_for_chamber(session: Session, chamber_id: str) -> ToolResult:
    chamber = _require_chamber(session, chamber_id)
    open_events = [
        e.id for e in chamber.maintenance_events if e.status in ("scheduled", "in_progress")
    ]
    return ToolResult(
        summary=f"Chamber {chamber.id} has {len(open_events)} open maintenance event(s): "
        f"{', '.join(open_events) if open_events else 'none'}.",
        value=open_events,
        source_object_ids=[chamber.id, *open_events],
    )


def get_part_usage_in_maintenance_event(
    session: Session, maintenance_event_id: str, part_id: str
) -> ToolResult:
    event = _require_maintenance_event(session, maintenance_event_id)
    part = _require_part(session, part_id)
    link = session.get(MaintenanceEventPart, (event.id, part.id))
    qty = link.qty_used if link is not None else 0
    return ToolResult(
        summary=f"Maintenance event {event.id} used {qty} unit(s) of part {part.id}.",
        value=qty,
        source_object_ids=[event.id, part.id],
    )


def list_chambers_for_tool(session: Session, tool_id: str) -> ToolResult:
    tool = _require_tool(session, tool_id)
    chamber_ids = [c.id for c in tool.chambers]
    return ToolResult(
        summary=f"Tool {tool.id} has {len(chamber_ids)} chamber(s): "
        f"{', '.join(chamber_ids) if chamber_ids else 'none'}.",
        value=chamber_ids,
        source_object_ids=[tool.id, *chamber_ids],
    )


# The catalog an LLM (or a rule router) is allowed to choose from by name.
# Deliberately not exposed as "run arbitrary SQL"; this dict is the entire
# read surface of the system.
TOOL_CATALOG: dict[str, Callable[..., ToolResult]] = {
    "get_tool_status": get_tool_status,
    "get_tool_type": get_tool_type,
    "get_chamber_status": get_chamber_status,
    "get_chamber_tool": get_chamber_tool,
    "get_recipe_chamber": get_recipe_chamber,
    "get_recipe_active": get_recipe_active,
    "get_part_stock": get_part_stock,
    "get_maintenance_event_status": get_maintenance_event_status,
    "get_maintenance_event_target": get_maintenance_event_target,
    "list_maintenance_events_for_chamber": list_maintenance_events_for_chamber,
    "get_part_usage_in_maintenance_event": get_part_usage_in_maintenance_event,
    "list_chambers_for_tool": list_chambers_for_tool,
}
