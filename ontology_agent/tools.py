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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from ontology_agent.models import Asset, Part, Site, Technician, WorkOrder, WorkOrderPart


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


def _require_asset(session: Session, asset_id: str) -> Asset:
    obj = session.get(Asset, asset_id)
    if obj is None:
        raise ObjectNotFoundError("asset", asset_id)
    return obj


def _require_technician(session: Session, technician_id: str) -> Technician:
    obj = session.get(Technician, technician_id)
    if obj is None:
        raise ObjectNotFoundError("technician", technician_id)
    return obj


def _require_site(session: Session, site_id: str) -> Site:
    obj = session.get(Site, site_id)
    if obj is None:
        raise ObjectNotFoundError("site", site_id)
    return obj


def _require_part(session: Session, part_id: str) -> Part:
    obj = session.get(Part, part_id)
    if obj is None:
        raise ObjectNotFoundError("part", part_id)
    return obj


def _require_work_order(session: Session, work_order_id: str) -> WorkOrder:
    obj = session.get(WorkOrder, work_order_id)
    if obj is None:
        raise ObjectNotFoundError("work_order", work_order_id)
    return obj


def get_work_order_status(session: Session, work_order_id: str) -> ToolResult:
    wo = _require_work_order(session, work_order_id)
    return ToolResult(
        summary=f"Work order {wo.id} is currently '{wo.status}'.",
        value=wo.status,
        source_object_ids=[wo.id],
    )


def get_work_order_priority(session: Session, work_order_id: str) -> ToolResult:
    wo = _require_work_order(session, work_order_id)
    return ToolResult(
        summary=f"Work order {wo.id} has priority '{wo.priority}'.",
        value=wo.priority,
        source_object_ids=[wo.id],
    )


def get_work_order_technician(session: Session, work_order_id: str) -> ToolResult:
    wo = _require_work_order(session, work_order_id)
    if wo.technician_id is None:
        return ToolResult(
            summary=f"Work order {wo.id} has no technician assigned.",
            value=None,
            source_object_ids=[wo.id],
        )
    return ToolResult(
        summary=f"Work order {wo.id} is assigned to technician {wo.technician_id}.",
        value=wo.technician_id,
        source_object_ids=[wo.id, wo.technician_id],
    )


def get_work_order_asset(session: Session, work_order_id: str) -> ToolResult:
    wo = _require_work_order(session, work_order_id)
    return ToolResult(
        summary=f"Work order {wo.id} is for asset {wo.asset_id}.",
        value=wo.asset_id,
        source_object_ids=[wo.id, wo.asset_id],
    )


def get_asset_status(session: Session, asset_id: str) -> ToolResult:
    asset = _require_asset(session, asset_id)
    return ToolResult(
        summary=f"Asset {asset.id} status is '{asset.status}'.",
        value=asset.status,
        source_object_ids=[asset.id],
    )


def get_asset_site(session: Session, asset_id: str) -> ToolResult:
    asset = _require_asset(session, asset_id)
    return ToolResult(
        summary=f"Asset {asset.id} is located at site {asset.site_id}.",
        value=asset.site_id,
        source_object_ids=[asset.id, asset.site_id],
    )


def get_technician_site(session: Session, technician_id: str) -> ToolResult:
    tech = _require_technician(session, technician_id)
    return ToolResult(
        summary=f"Technician {tech.id} is based at site {tech.site_id}.",
        value=tech.site_id,
        source_object_ids=[tech.id, tech.site_id],
    )


def get_technician_active(session: Session, technician_id: str) -> ToolResult:
    tech = _require_technician(session, technician_id)
    return ToolResult(
        summary=f"Technician {tech.id} active status is {tech.active}.",
        value=tech.active,
        source_object_ids=[tech.id],
    )


def get_part_stock(session: Session, part_id: str) -> ToolResult:
    part = _require_part(session, part_id)
    return ToolResult(
        summary=f"Part {part.id} has {part.stock_qty} units in stock.",
        value=part.stock_qty,
        source_object_ids=[part.id],
    )


def list_open_work_orders_for_technician(
    session: Session, technician_id: str
) -> ToolResult:
    tech = _require_technician(session, technician_id)
    open_wos = [
        wo.id
        for wo in tech.work_orders
        if wo.status in ("open", "in_progress", "on_hold")
    ]
    return ToolResult(
        summary=f"Technician {tech.id} has {len(open_wos)} open work order(s): "
        f"{', '.join(open_wos) if open_wos else 'none'}.",
        value=open_wos,
        source_object_ids=[tech.id, *open_wos],
    )


def get_part_usage_in_work_order(
    session: Session, work_order_id: str, part_id: str
) -> ToolResult:
    wo = _require_work_order(session, work_order_id)
    part = _require_part(session, part_id)
    link = session.get(WorkOrderPart, (wo.id, part.id))
    qty = link.qty_used if link is not None else 0
    return ToolResult(
        summary=f"Work order {wo.id} used {qty} unit(s) of part {part.id}.",
        value=qty,
        source_object_ids=[wo.id, part.id],
    )


def get_site_technician_count(session: Session, site_id: str) -> ToolResult:
    site = _require_site(session, site_id)
    count = len(site.technicians)
    return ToolResult(
        summary=f"Site {site.id} has {count} technician(s) assigned.",
        value=count,
        source_object_ids=[site.id],
    )


# The catalog an LLM (or a rule router) is allowed to choose from by name.
# Deliberately not exposed as "run arbitrary SQL"; this dict is the entire
# read surface of the system.
TOOL_CATALOG: dict[str, Callable[..., ToolResult]] = {
    "get_work_order_status": get_work_order_status,
    "get_work_order_priority": get_work_order_priority,
    "get_work_order_technician": get_work_order_technician,
    "get_work_order_asset": get_work_order_asset,
    "get_asset_status": get_asset_status,
    "get_asset_site": get_asset_site,
    "get_technician_site": get_technician_site,
    "get_technician_active": get_technician_active,
    "get_part_stock": get_part_stock,
    "list_open_work_orders_for_technician": list_open_work_orders_for_technician,
    "get_part_usage_in_work_order": get_part_usage_in_work_order,
    "get_site_technician_count": get_site_technician_count,
}
