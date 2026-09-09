"""The manual-baseline simulator: the value harness's other half.

Everything in `tools.py` answers a question with one indexed lookup (a
SQLAlchemy `Session.get(Model, pk)`, backed by the identity map and, for
anything not yet loaded, a primary-key index) because the tool catalog is
built on a typed, indexed object model. A person doing the same lookup by
hand against an unindexed source (a spreadsheet export, a printed report,
a screen with no search box) has no index to use: finding a record by id
costs one comparison per row until a match turns up, and confirming a
record does *not* exist, which is exactly what the refusal-designed
questions require, costs a comparison against every row in the table,
because there is no way to know a row is missing without checking all of
them.

This module reimplements each of the 12 tools in `tools.py` one for one,
same inputs, same outputs, same citation contract, with every
`Session.get` replaced by a linear scan over a plain Python list and every
ORM relationship access (`tool.chambers`, `chamber.maintenance_events`)
replaced by a full-table filter. It does not touch the ORM at all after
the initial one-time load; that load stands in for a person already
having the report in front of them; what is timed is the lookup, not the
export.

One deliberate, disclosed simplification: routing (deciding which of the
12 lookups a question requires, and pulling the typed ids out of the
question text) is shared with the assistant path (`qa_router.route_question`)
rather than reimplemented here. See README, Measured results, for why
this is a disclosed proxy for manual effort and not a live user study.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from sqlalchemy.orm import Session

from ontology_agent.models import Chamber, MaintenanceEvent, MaintenanceEventPart, Part, Recipe, Tool


class ManualNotFoundError(Exception):
    """Raised when a manual scan of every row in a table finds no match.

    The manual-baseline equivalent of `tools.ObjectNotFoundError`: it is
    raised only after every row has genuinely been compared, never as a
    shortcut.
    """

    def __init__(self, object_type: str, object_id: str):
        self.object_type = object_type
        self.object_id = object_id
        super().__init__(f"no {object_type} with id {object_id!r} in the manual records")


@dataclasses.dataclass
class ManualResult:
    """The manual-baseline equivalent of `tools.ToolResult`."""

    value: object
    source_object_ids: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ScanStats:
    """Counts the raw row comparisons a manual lookup actually performed.

    This is the operation-count side of the value harness: the number a
    person would rack up flipping through an unindexed report, as opposed
    to the wall-clock side, which times the same work in this process.
    """

    comparisons: int = 0
    tables_scanned: int = 0

    def scan(self, n: int) -> None:
        self.comparisons += n
        self.tables_scanned += 1


@dataclasses.dataclass(frozen=True)
class ManualRow:
    """A minimal, flat stand-in for one exported record. Only the fields a
    manual lookup or join actually needs are kept; this is what a printed
    report column set would realistically carry, not the full ORM row."""

    id: str
    tool_id: str | None = None
    chamber_id: str | None = None
    part_id: str | None = None
    status: str | None = None
    kind: str | None = None
    is_active: bool | None = None
    stock_qty: int | None = None
    qty_used: int | None = None


@dataclasses.dataclass(frozen=True)
class ManualTables:
    """The one-time export from the typed store into flat, unindexed lists.

    Loading this is not part of what the harness times: it stands in for a
    person already holding the report, exactly as `Session.get` calls in
    the real tool catalog assume the database is already up and reachable.
    """

    tools: list[ManualRow]
    chambers: list[ManualRow]
    recipes: list[ManualRow]
    parts: list[ManualRow]
    maintenance_events: list[ManualRow]
    maintenance_event_parts: list[ManualRow]


def load_manual_tables(session: Session) -> ManualTables:
    """One-time, untimed export of every row into flat, unindexed lists."""
    tools = [ManualRow(id=t.id, status=t.status, kind=t.tool_type) for t in session.query(Tool).all()]
    chambers = [
        ManualRow(id=c.id, tool_id=c.tool_id, status=c.status)
        for c in session.query(Chamber).all()
    ]
    recipes = [
        ManualRow(id=r.id, chamber_id=r.chamber_id, is_active=r.is_active)
        for r in session.query(Recipe).all()
    ]
    parts = [ManualRow(id=p.id, stock_qty=p.stock_qty) for p in session.query(Part).all()]
    maintenance_events = [
        ManualRow(id=m.id, tool_id=m.tool_id, chamber_id=m.chamber_id, status=m.status)
        for m in session.query(MaintenanceEvent).all()
    ]
    maintenance_event_parts = [
        ManualRow(id=f"{mep.maintenance_event_id}:{mep.part_id}", tool_id=mep.maintenance_event_id,
                   part_id=mep.part_id, qty_used=mep.qty_used)
        for mep in session.query(MaintenanceEventPart).all()
    ]
    return ManualTables(
        tools=tools, chambers=chambers, recipes=recipes, parts=parts,
        maintenance_events=maintenance_events, maintenance_event_parts=maintenance_event_parts,
    )


def _scan_find(rows: list[ManualRow], target_id: str, stats: ScanStats) -> ManualRow | None:
    """Linear scan: compare against every row in order until a match, or
    until the whole table has been checked and there is none."""
    found = None
    for row in rows:
        stats.comparisons += 1
        if row.id == target_id:
            found = row
            break
    stats.tables_scanned += 1
    return found


def _require(rows: list[ManualRow], target_id: str, object_type: str, stats: ScanStats) -> ManualRow:
    row = _scan_find(rows, target_id, stats)
    if row is None:
        raise ManualNotFoundError(object_type, target_id)
    return row


def manual_get_tool_status(tables: ManualTables, stats: ScanStats, tool_id: str) -> ManualResult:
    tool = _require(tables.tools, tool_id, "tool", stats)
    return ManualResult(value=tool.status, source_object_ids=[tool.id])


def manual_get_tool_type(tables: ManualTables, stats: ScanStats, tool_id: str) -> ManualResult:
    tool = _require(tables.tools, tool_id, "tool", stats)
    return ManualResult(value=tool.kind, source_object_ids=[tool.id])


def manual_get_chamber_status(tables: ManualTables, stats: ScanStats, chamber_id: str) -> ManualResult:
    chamber = _require(tables.chambers, chamber_id, "chamber", stats)
    return ManualResult(value=chamber.status, source_object_ids=[chamber.id])


def manual_get_chamber_tool(tables: ManualTables, stats: ScanStats, chamber_id: str) -> ManualResult:
    chamber = _require(tables.chambers, chamber_id, "chamber", stats)
    return ManualResult(value=chamber.tool_id, source_object_ids=[chamber.id, chamber.tool_id])


def manual_get_recipe_chamber(tables: ManualTables, stats: ScanStats, recipe_id: str) -> ManualResult:
    recipe = _require(tables.recipes, recipe_id, "recipe", stats)
    return ManualResult(value=recipe.chamber_id, source_object_ids=[recipe.id, recipe.chamber_id])


def manual_get_recipe_active(tables: ManualTables, stats: ScanStats, recipe_id: str) -> ManualResult:
    recipe = _require(tables.recipes, recipe_id, "recipe", stats)
    return ManualResult(value=recipe.is_active, source_object_ids=[recipe.id])


def manual_get_part_stock(tables: ManualTables, stats: ScanStats, part_id: str) -> ManualResult:
    part = _require(tables.parts, part_id, "part", stats)
    return ManualResult(value=part.stock_qty, source_object_ids=[part.id])


def manual_get_maintenance_event_status(
    tables: ManualTables, stats: ScanStats, maintenance_event_id: str
) -> ManualResult:
    event = _require(tables.maintenance_events, maintenance_event_id, "maintenance_event", stats)
    return ManualResult(value=event.status, source_object_ids=[event.id])


def manual_get_maintenance_event_target(
    tables: ManualTables, stats: ScanStats, maintenance_event_id: str
) -> ManualResult:
    event = _require(tables.maintenance_events, maintenance_event_id, "maintenance_event", stats)
    target_id = event.tool_id if event.tool_id is not None else event.chamber_id
    return ManualResult(value=target_id, source_object_ids=[event.id, target_id])


def manual_list_maintenance_events_for_chamber(
    tables: ManualTables, stats: ScanStats, chamber_id: str
) -> ManualResult:
    chamber = _require(tables.chambers, chamber_id, "chamber", stats)
    open_ids = []
    for event in tables.maintenance_events:
        stats.comparisons += 1
        if event.chamber_id == chamber.id and event.status in ("scheduled", "in_progress"):
            open_ids.append(event.id)
    stats.tables_scanned += 1
    return ManualResult(value=open_ids, source_object_ids=[chamber.id, *open_ids])


def manual_get_part_usage_in_maintenance_event(
    tables: ManualTables, stats: ScanStats, maintenance_event_id: str, part_id: str
) -> ManualResult:
    event = _require(tables.maintenance_events, maintenance_event_id, "maintenance_event", stats)
    part = _require(tables.parts, part_id, "part", stats)
    qty = 0
    for link in tables.maintenance_event_parts:
        stats.comparisons += 1
        if link.tool_id == event.id and link.part_id == part.id:
            qty = link.qty_used
            break
    stats.tables_scanned += 1
    return ManualResult(value=qty, source_object_ids=[event.id, part.id])


def manual_list_chambers_for_tool(tables: ManualTables, stats: ScanStats, tool_id: str) -> ManualResult:
    tool = _require(tables.tools, tool_id, "tool", stats)
    chamber_ids = []
    for chamber in tables.chambers:
        stats.comparisons += 1
        if chamber.tool_id == tool.id:
            chamber_ids.append(chamber.id)
    stats.tables_scanned += 1
    return ManualResult(value=chamber_ids, source_object_ids=[tool.id, *chamber_ids])


MANUAL_TOOL_CATALOG: dict[str, Callable[..., ManualResult]] = {
    "get_tool_status": manual_get_tool_status,
    "get_tool_type": manual_get_tool_type,
    "get_chamber_status": manual_get_chamber_status,
    "get_chamber_tool": manual_get_chamber_tool,
    "get_recipe_chamber": manual_get_recipe_chamber,
    "get_recipe_active": manual_get_recipe_active,
    "get_part_stock": manual_get_part_stock,
    "get_maintenance_event_status": manual_get_maintenance_event_status,
    "get_maintenance_event_target": manual_get_maintenance_event_target,
    "list_maintenance_events_for_chamber": manual_list_maintenance_events_for_chamber,
    "get_part_usage_in_maintenance_event": manual_get_part_usage_in_maintenance_event,
    "list_chambers_for_tool": manual_list_chambers_for_tool,
}


@dataclasses.dataclass
class ManualAnsweredQuestion:
    text: str
    refused: bool
    source_object_ids: list[str]
    tool_used: str | None
    refusal_reason: str | None = None


def manual_answer_question(tables: ManualTables, text: str, stats: ScanStats) -> ManualAnsweredQuestion:
    """Route (shared with the assistant path, see module docstring), then
    execute the lookup through linear scans and manual joins only."""
    from ontology_agent.qa_router import RoutingError, route_question

    try:
        route = route_question(text)
    except RoutingError as exc:
        return ManualAnsweredQuestion(
            text=text, refused=True, source_object_ids=[], tool_used=None,
            refusal_reason=str(exc),
        )

    tool_fn = MANUAL_TOOL_CATALOG[route.tool]
    try:
        result = tool_fn(tables, stats, **route.args)
    except ManualNotFoundError as exc:
        return ManualAnsweredQuestion(
            text=text, refused=True, source_object_ids=[], tool_used=route.tool,
            refusal_reason=str(exc),
        )

    return ManualAnsweredQuestion(
        text=text, refused=False, source_object_ids=result.source_object_ids,
        tool_used=route.tool,
    )
