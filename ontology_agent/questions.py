"""The held-out synthetic question set and the holdout-removal step.

`build_question_set` produces a reproducible set of natural-language
operational questions from the seeded object graph: most reference an
object that is genuinely present in the store (answerable, with a known
correct tool call and expected citation), and exactly 74 are built to
reference an object id that has been created and then deliberately
deleted from the store, so the only correct system behaviour is refusal.

Each template's question text carries a unique marker phrase; that is a
deliberate, documented simplification (see README, Limitations) standing
in for a full NLU/intent-classification layer, which is out of scope for
what this repository is measuring: citation correctness and refusal
behaviour of the constrained tool layer, not intent parsing quality.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import random

from sqlalchemy.orm import Session

from ontology_agent.models import Asset, Part, Site, Technician, WorkOrder
from ontology_agent.seed import SeededObjectGraph

REFUSAL_TOTAL = 74


@dataclasses.dataclass
class Question:
    qid: str
    text: str
    tool: str
    args: dict[str, str]
    is_refusal: bool
    # For answerable questions, the citation ids a correct answer must include.
    expected_source_object_ids: list[str]


# Each entry: (tool_name, marker phrase used by the router, phrase builder,
# id-type key(s) consumed from the args dict passed to phrase builder,
# which arg key is the one that gets swapped for a missing id on refusal
# questions -- the "primary" missing type).
_TEMPLATES = [
    (
        "get_work_order_status",
        lambda a: f"What is the current status of work order {a['work_order_id']}?",
        ["work_order_id"],
    ),
    (
        "get_work_order_priority",
        lambda a: f"What priority level is assigned to work order {a['work_order_id']}?",
        ["work_order_id"],
    ),
    (
        "get_work_order_technician",
        lambda a: f"Which technician is assigned to work order {a['work_order_id']}?",
        ["work_order_id"],
    ),
    (
        "get_work_order_asset",
        lambda a: f"Which asset is associated with work order {a['work_order_id']}?",
        ["work_order_id"],
    ),
    (
        "get_asset_status",
        lambda a: f"What is the operating status of asset {a['asset_id']}?",
        ["asset_id"],
    ),
    (
        "get_asset_site",
        lambda a: f"Which site is asset {a['asset_id']} located at?",
        ["asset_id"],
    ),
    (
        "get_technician_site",
        lambda a: f"Which site is technician {a['technician_id']} based at?",
        ["technician_id"],
    ),
    (
        "get_technician_active",
        lambda a: f"Is technician {a['technician_id']} currently active?",
        ["technician_id"],
    ),
    (
        "list_open_work_orders_for_technician",
        lambda a: f"How many open work orders does technician {a['technician_id']} currently have?",
        ["technician_id"],
    ),
    (
        "get_part_stock",
        lambda a: f"How many units of part {a['part_id']} are currently in stock?",
        ["part_id"],
    ),
    (
        "get_part_usage_in_work_order",
        lambda a: f"How many units of part {a['part_id']} were used on work order {a['work_order_id']}?",
        ["part_id", "work_order_id"],
    ),
    (
        "get_site_technician_count",
        lambda a: f"How many technicians are assigned to site {a['site_id']}?",
        ["site_id"],
    ),
]

# Number of refusal questions built per template; sums to REFUSAL_TOTAL (74).
_REFUSAL_COUNTS = {
    "get_work_order_status": 7,
    "get_work_order_priority": 6,
    "get_work_order_technician": 7,
    "get_work_order_asset": 6,
    "get_asset_status": 6,
    "get_asset_site": 6,
    "get_technician_site": 6,
    "get_technician_active": 6,
    "list_open_work_orders_for_technician": 6,
    "get_part_stock": 6,
    "get_part_usage_in_work_order": 6,
    "get_site_technician_count": 6,
}
assert sum(_REFUSAL_COUNTS.values()) == REFUSAL_TOTAL

# Number of answerable questions built per template.
_ANSWERABLE_PER_TEMPLATE = 26


def create_and_delete_holdout_objects(
    session: Session, graph: SeededObjectGraph, seed: int
) -> dict[str, list[str]]:
    """Create standalone objects unreferenced by anything else, then delete
    them, and return their (now nonexistent) ids by type. This is the
    literal "deliberately removed from the store" step: the rows are real
    rows that genuinely existed and were genuinely deleted, not ids that
    were simply never used.
    """
    rng = random.Random(seed + 1)
    kept_site = graph.site_ids[0]

    removed_sites = [Site(id=f"STE-{900+i:04d}", name=f"Decommissioned Site {i}",
                          region="Retired", timezone="UTC") for i in range(4)]
    removed_technicians = [
        Technician(id=f"TCH-{9000+i:04d}", name=f"Former Technician {i}",
                   site_id=kept_site, certification_level="journeyman", active=False)
        for i in range(8)
    ]
    removed_assets = [
        Asset(id=f"AST-{90000+i:05d}", site_id=kept_site, asset_type="Retired Unit",
              model="N/A", serial_number=f"RETIRED-{seed}-{i}",
              install_date=dt.date(2015, 1, 1), status="decommissioned")
        for i in range(8)
    ]
    removed_parts = [
        Part(id=f"PRT-{900+i:04d}", name=f"Obsolete Part {i}", sku=f"OBS-{i}",
             unit_cost=1.0, stock_qty=0)
        for i in range(8)
    ]
    session.add_all(removed_sites + removed_technicians + removed_assets + removed_parts)
    session.flush()

    removed_work_orders = [
        WorkOrder(
            id=f"WO-{900000+i:06d}", asset_id=graph.asset_ids[0], technician_id=None,
            status="closed", priority="low", opened_at=dt.datetime(2025, 1, 1),
            closed_at=dt.datetime(2025, 1, 2), description="Cancelled test work order.",
        )
        for i in range(10)
    ]
    session.add_all(removed_work_orders)
    session.commit()

    removed_ids = {
        "site_id": [s.id for s in removed_sites],
        "technician_id": [t.id for t in removed_technicians],
        "asset_id": [a.id for a in removed_assets],
        "part_id": [p.id for p in removed_parts],
        "work_order_id": [w.id for w in removed_work_orders],
    }

    for wo in removed_work_orders:
        session.delete(wo)
    for a in removed_assets:
        session.delete(a)
    for t in removed_technicians:
        session.delete(t)
    for p in removed_parts:
        session.delete(p)
    for s in removed_sites:
        session.delete(s)
    session.commit()

    return removed_ids


def _expected_sources(tool: str, args: dict[str, str], session: Session) -> list[str]:
    """Compute the correct citation set for an answerable question directly
    from the store, independent of any tool implementation, so the
    benchmark has a reference oracle to check the tool layer against."""
    if tool == "get_work_order_status" or tool == "get_work_order_priority":
        return [args["work_order_id"]]
    if tool == "get_work_order_technician":
        wo = session.get(WorkOrder, args["work_order_id"])
        return [wo.id] if wo.technician_id is None else [wo.id, wo.technician_id]
    if tool == "get_work_order_asset":
        wo = session.get(WorkOrder, args["work_order_id"])
        return [wo.id, wo.asset_id]
    if tool == "get_asset_status":
        return [args["asset_id"]]
    if tool == "get_asset_site":
        asset = session.get(Asset, args["asset_id"])
        return [asset.id, asset.site_id]
    if tool == "get_technician_site":
        tech = session.get(Technician, args["technician_id"])
        return [tech.id, tech.site_id]
    if tool == "get_technician_active":
        return [args["technician_id"]]
    if tool == "list_open_work_orders_for_technician":
        tech = session.get(Technician, args["technician_id"])
        open_ids = [
            wo.id for wo in tech.work_orders if wo.status in ("open", "in_progress", "on_hold")
        ]
        return [tech.id, *open_ids]
    if tool == "get_part_stock":
        return [args["part_id"]]
    if tool == "get_part_usage_in_work_order":
        return [args["work_order_id"], args["part_id"]]
    if tool == "get_site_technician_count":
        return [args["site_id"]]
    raise ValueError(f"unknown tool {tool}")


def build_question_set(
    session: Session,
    graph: SeededObjectGraph,
    removed_ids: dict[str, list[str]],
    seed: int,
) -> list[Question]:
    rng = random.Random(seed + 2)
    questions: list[Question] = []
    qnum = 0

    for tool, phrase_fn, arg_keys in _TEMPLATES:
        for _ in range(_ANSWERABLE_PER_TEMPLATE):
            args = {}
            for key in arg_keys:
                pool = {
                    "work_order_id": graph.work_order_ids,
                    "asset_id": graph.asset_ids,
                    "technician_id": graph.technician_ids,
                    "part_id": graph.part_ids,
                    "site_id": graph.site_ids,
                }[key]
                args[key] = rng.choice(pool)
            qnum += 1
            questions.append(
                Question(
                    qid=f"Q-{qnum:04d}",
                    text=phrase_fn(args),
                    tool=tool,
                    args=args,
                    is_refusal=False,
                    expected_source_object_ids=_expected_sources(tool, args, session),
                )
            )

        for _ in range(_REFUSAL_COUNTS[tool]):
            args = {}
            missing_key = arg_keys[0]
            for key in arg_keys:
                if key == missing_key:
                    args[key] = rng.choice(removed_ids[key])
                else:
                    pool = {
                        "work_order_id": graph.work_order_ids,
                        "asset_id": graph.asset_ids,
                        "technician_id": graph.technician_ids,
                        "part_id": graph.part_ids,
                        "site_id": graph.site_ids,
                    }[key]
                    args[key] = rng.choice(pool)
            qnum += 1
            questions.append(
                Question(
                    qid=f"Q-{qnum:04d}",
                    text=phrase_fn(args),
                    tool=tool,
                    args=args,
                    is_refusal=True,
                    expected_source_object_ids=[],
                )
            )

    rng.shuffle(questions)
    return questions
