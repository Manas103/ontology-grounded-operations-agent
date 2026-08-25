"""The manual-baseline simulator: the value harness's other half.

Everything in `tools.py` answers a question with one indexed lookup (a
SQLAlchemy `Session.get(Model, pk)`, backed by the identity map and, for
anything not yet loaded, a primary-key index) because the tool catalog is
built on a typed, indexed object model. A person doing the same lookup by
hand against an unindexed source (a spreadsheet export, a printed report,
a screen with no search box) has no index to use: finding a record by id
costs one comparison per row until a match turns up, and confirming a
record does *not* exist, which is exactly what the 74 refusal-designed
questions require, costs a comparison against every row in the table,
because there is no way to know a row is missing without checking all of
them.

This module reimplements each of the 12 tools in `tools.py` one for one,
same inputs, same outputs, same citation contract, with every
`Session.get` replaced by a linear scan over a plain Python list and every
ORM relationship access (`tech.work_orders`, `site.technicians`) replaced
by a full-table filter. It does not touch the ORM at all after the initial
one-time load; that load stands in for a person already having the report
in front of them; what is timed is the lookup, not the export.

One deliberate, disclosed simplification: routing (deciding which of the
12 lookups a question requires, and pulling the typed ids out of the
question text) is shared with the assistant path (`qa_router.route_question`)
rather than reimplemented here. A human reading "What priority level is
assigned to work order WO-000123?" and a typed-tool router both parse that
sentence the same way; the thing the assistant is actually claimed to be
faster at is looking the record up once it knows what to look up, so the
harness isolates exactly that and does not also charge the manual path for
re-deriving something neither path actually struggles with. See README,
Measured results, for why this is a disclosed proxy for manual effort and
not a live user study.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from sqlalchemy.orm import Session

from ontology_agent.models import Asset, Part, Site, Technician, WorkOrder, WorkOrderPart


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
    site_id: str | None = None
    technician_id: str | None = None
    asset_id: str | None = None
    part_id: str | None = None
    status: str | None = None
    priority: str | None = None
    active: bool | None = None
    stock_qty: int | None = None
    qty_used: int | None = None


@dataclasses.dataclass(frozen=True)
class ManualTables:
    """The one-time export from the typed store into flat, unindexed lists.

    Loading this is not part of what the harness times: it stands in for a
    person already holding the report, exactly as `Session.get` calls in
    the real tool catalog assume the database is already up and reachable.
    """

    sites: list[ManualRow]
    technicians: list[ManualRow]
    assets: list[ManualRow]
    parts: list[ManualRow]
    work_orders: list[ManualRow]
    work_order_parts: list[ManualRow]


def load_manual_tables(session: Session) -> ManualTables:
    """One-time, untimed export of every row into flat, unindexed lists."""
    sites = [ManualRow(id=s.id) for s in session.query(Site).all()]
    technicians = [
        ManualRow(id=t.id, site_id=t.site_id, active=t.active)
        for t in session.query(Technician).all()
    ]
    assets = [ManualRow(id=a.id, site_id=a.site_id, status=a.status) for a in session.query(Asset).all()]
    parts = [ManualRow(id=p.id, stock_qty=p.stock_qty) for p in session.query(Part).all()]
    work_orders = [
        ManualRow(id=w.id, asset_id=w.asset_id, technician_id=w.technician_id,
                   status=w.status, priority=w.priority)
        for w in session.query(WorkOrder).all()
    ]
    work_order_parts = [
        ManualRow(id=f"{wop.work_order_id}:{wop.part_id}", asset_id=wop.work_order_id,
                   part_id=wop.part_id, qty_used=wop.qty_used)
        for wop in session.query(WorkOrderPart).all()
    ]
    return ManualTables(
        sites=sites, technicians=technicians, assets=assets, parts=parts,
        work_orders=work_orders, work_order_parts=work_order_parts,
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


def manual_get_work_order_status(tables: ManualTables, stats: ScanStats, work_order_id: str) -> ManualResult:
    wo = _require(tables.work_orders, work_order_id, "work_order", stats)
    return ManualResult(value=wo.status, source_object_ids=[wo.id])


def manual_get_work_order_priority(tables: ManualTables, stats: ScanStats, work_order_id: str) -> ManualResult:
    wo = _require(tables.work_orders, work_order_id, "work_order", stats)
    return ManualResult(value=wo.priority, source_object_ids=[wo.id])


def manual_get_work_order_technician(tables: ManualTables, stats: ScanStats, work_order_id: str) -> ManualResult:
    wo = _require(tables.work_orders, work_order_id, "work_order", stats)
    if wo.technician_id is None:
        return ManualResult(value=None, source_object_ids=[wo.id])
    return ManualResult(value=wo.technician_id, source_object_ids=[wo.id, wo.technician_id])


def manual_get_work_order_asset(tables: ManualTables, stats: ScanStats, work_order_id: str) -> ManualResult:
    wo = _require(tables.work_orders, work_order_id, "work_order", stats)
    return ManualResult(value=wo.asset_id, source_object_ids=[wo.id, wo.asset_id])


def manual_get_asset_status(tables: ManualTables, stats: ScanStats, asset_id: str) -> ManualResult:
    asset = _require(tables.assets, asset_id, "asset", stats)
    return ManualResult(value=asset.status, source_object_ids=[asset.id])


def manual_get_asset_site(tables: ManualTables, stats: ScanStats, asset_id: str) -> ManualResult:
    asset = _require(tables.assets, asset_id, "asset", stats)
    return ManualResult(value=asset.site_id, source_object_ids=[asset.id, asset.site_id])


def manual_get_technician_site(tables: ManualTables, stats: ScanStats, technician_id: str) -> ManualResult:
    tech = _require(tables.technicians, technician_id, "technician", stats)
    return ManualResult(value=tech.site_id, source_object_ids=[tech.id, tech.site_id])


def manual_get_technician_active(tables: ManualTables, stats: ScanStats, technician_id: str) -> ManualResult:
    tech = _require(tables.technicians, technician_id, "technician", stats)
    return ManualResult(value=tech.active, source_object_ids=[tech.id])


def manual_get_part_stock(tables: ManualTables, stats: ScanStats, part_id: str) -> ManualResult:
    part = _require(tables.parts, part_id, "part", stats)
    return ManualResult(value=part.stock_qty, source_object_ids=[part.id])


def manual_list_open_work_orders_for_technician(
    tables: ManualTables, stats: ScanStats, technician_id: str
) -> ManualResult:
    tech = _require(tables.technicians, technician_id, "technician", stats)
    open_ids = []
    for wo in tables.work_orders:
        stats.comparisons += 1
        if wo.technician_id == tech.id and wo.status in ("open", "in_progress", "on_hold"):
            open_ids.append(wo.id)
    stats.tables_scanned += 1
    return ManualResult(value=open_ids, source_object_ids=[tech.id, *open_ids])


def manual_get_part_usage_in_work_order(
    tables: ManualTables, stats: ScanStats, work_order_id: str, part_id: str
) -> ManualResult:
    wo = _require(tables.work_orders, work_order_id, "work_order", stats)
    part = _require(tables.parts, part_id, "part", stats)
    qty = 0
    for link in tables.work_order_parts:
        stats.comparisons += 1
        if link.asset_id == wo.id and link.part_id == part.id:
            qty = link.qty_used
            break
    stats.tables_scanned += 1
    return ManualResult(value=qty, source_object_ids=[wo.id, part.id])


def manual_get_site_technician_count(tables: ManualTables, stats: ScanStats, site_id: str) -> ManualResult:
    site = _require(tables.sites, site_id, "site", stats)
    count = 0
    for tech in tables.technicians:
        stats.comparisons += 1
        if tech.site_id == site.id:
            count += 1
    stats.tables_scanned += 1
    return ManualResult(value=count, source_object_ids=[site.id])


MANUAL_TOOL_CATALOG: dict[str, Callable[..., ManualResult]] = {
    "get_work_order_status": manual_get_work_order_status,
    "get_work_order_priority": manual_get_work_order_priority,
    "get_work_order_technician": manual_get_work_order_technician,
    "get_work_order_asset": manual_get_work_order_asset,
    "get_asset_status": manual_get_asset_status,
    "get_asset_site": manual_get_asset_site,
    "get_technician_site": manual_get_technician_site,
    "get_technician_active": manual_get_technician_active,
    "get_part_stock": manual_get_part_stock,
    "list_open_work_orders_for_technician": manual_list_open_work_orders_for_technician,
    "get_part_usage_in_work_order": manual_get_part_usage_in_work_order,
    "get_site_technician_count": manual_get_site_technician_count,
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
