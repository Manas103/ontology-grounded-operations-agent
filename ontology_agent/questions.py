"""The held-out synthetic question set and the holdout-removal step.

`build_question_set` produces a reproducible set of natural-language
equipment questions from the seeded object graph: most reference an
object that is genuinely present in the store (answerable, with a known
correct tool call and expected citation), and a fixed number are built to
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

from ontology_agent.models import Chamber, MaintenanceEvent, Part, Recipe, Tool
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


# Each entry: (tool_name, phrase builder, id-type key(s) consumed from the
# args dict passed to phrase builder). The first key in each list is the
# one swapped for a missing id on refusal questions.
_TEMPLATES = [
    (
        "get_tool_status",
        lambda a: f"What is the current status of tool {a['tool_id']}?",
        ["tool_id"],
    ),
    (
        "get_tool_type",
        lambda a: f"What type of tool is {a['tool_id']}?",
        ["tool_id"],
    ),
    (
        "get_chamber_status",
        lambda a: f"What is the current status of chamber {a['chamber_id']}?",
        ["chamber_id"],
    ),
    (
        "get_chamber_tool",
        lambda a: f"Which tool does chamber {a['chamber_id']} belong to?",
        ["chamber_id"],
    ),
    (
        "get_recipe_chamber",
        lambda a: f"Which chamber is recipe {a['recipe_id']} assigned to?",
        ["recipe_id"],
    ),
    (
        "get_recipe_active",
        lambda a: f"Is recipe {a['recipe_id']} currently active?",
        ["recipe_id"],
    ),
    (
        "get_part_stock",
        lambda a: f"How many units of part {a['part_id']} are currently in stock?",
        ["part_id"],
    ),
    (
        "get_maintenance_event_status",
        lambda a: f"What is the current status of maintenance event {a['maintenance_event_id']}?",
        ["maintenance_event_id"],
    ),
    (
        "get_maintenance_event_target",
        lambda a: f"Which tool or chamber was maintenance event {a['maintenance_event_id']} performed on?",
        ["maintenance_event_id"],
    ),
    (
        "list_maintenance_events_for_chamber",
        lambda a: f"How many open maintenance events does chamber {a['chamber_id']} currently have?",
        ["chamber_id"],
    ),
    (
        "get_part_usage_in_maintenance_event",
        lambda a: f"How many units of part {a['part_id']} were used during maintenance event {a['maintenance_event_id']}?",
        ["maintenance_event_id", "part_id"],
    ),
    (
        "list_chambers_for_tool",
        lambda a: f"How many chambers are installed on tool {a['tool_id']}?",
        ["tool_id"],
    ),
]

# Number of refusal questions built per template; sums to REFUSAL_TOTAL (74).
_REFUSAL_COUNTS = {
    "get_tool_status": 7,
    "get_tool_type": 6,
    "get_chamber_status": 7,
    "get_chamber_tool": 6,
    "get_recipe_chamber": 6,
    "get_recipe_active": 6,
    "get_part_stock": 6,
    "get_maintenance_event_status": 6,
    "get_maintenance_event_target": 6,
    "list_maintenance_events_for_chamber": 6,
    "get_part_usage_in_maintenance_event": 6,
    "list_chambers_for_tool": 6,
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
    kept_tool = graph.tool_ids[0]
    kept_chamber = graph.chamber_ids[0]

    removed_tools = [
        Tool(id=f"TL-{9000+i:04d}", name=f"Decommissioned Tool {i}",
             tool_type="Retired Platform", fab_bay="Retired",
             install_date=dt.date(2015, 1, 1), status="decommissioned")
        for i in range(4)
    ]
    removed_chambers = [
        Chamber(id=f"CH-{90000+i:05d}", tool_id=kept_tool, chamber_number=900 + i,
                chamber_type="process", status="decommissioned")
        for i in range(8)
    ]
    removed_recipes = [
        Recipe(id=f"RC-{90000+i:05d}", chamber_id=kept_chamber, name=f"Retired Recipe {i}",
               process_step="retired step", revision=0, is_active=False,
               primary_part_id=None)
        for i in range(8)
    ]
    removed_parts = [
        Part(id=f"PRT-{900+i:04d}", name=f"Obsolete Part {i}", sku=f"OBS-{i}",
             unit_cost=1.0, stock_qty=0, is_consumable=False)
        for i in range(8)
    ]
    session.add_all(removed_tools + removed_chambers + removed_recipes + removed_parts)
    session.flush()

    removed_maintenance_events = [
        MaintenanceEvent(
            id=f"ME-{900000+i:06d}", tool_id=kept_tool, chamber_id=None,
            maintenance_type="corrective", status="cancelled",
            opened_at=dt.datetime(2025, 1, 1), closed_at=dt.datetime(2025, 1, 2),
            description="Cancelled test maintenance event.",
        )
        for i in range(10)
    ]
    session.add_all(removed_maintenance_events)
    session.commit()

    removed_ids = {
        "tool_id": [t.id for t in removed_tools],
        "chamber_id": [c.id for c in removed_chambers],
        "recipe_id": [r.id for r in removed_recipes],
        "part_id": [p.id for p in removed_parts],
        "maintenance_event_id": [m.id for m in removed_maintenance_events],
    }

    for m in removed_maintenance_events:
        session.delete(m)
    for r in removed_recipes:
        session.delete(r)
    for c in removed_chambers:
        session.delete(c)
    for p in removed_parts:
        session.delete(p)
    for t in removed_tools:
        session.delete(t)
    session.commit()

    return removed_ids


def _expected_sources(tool: str, args: dict[str, str], session: Session) -> list[str]:
    """Compute the correct citation set for an answerable question directly
    from the store, independent of any tool implementation, so the
    benchmark has a reference oracle to check the tool layer against."""
    if tool == "get_tool_status" or tool == "get_tool_type":
        return [args["tool_id"]]
    if tool == "get_chamber_status":
        return [args["chamber_id"]]
    if tool == "get_chamber_tool":
        chamber = session.get(Chamber, args["chamber_id"])
        return [chamber.id, chamber.tool_id]
    if tool == "get_recipe_chamber":
        recipe = session.get(Recipe, args["recipe_id"])
        return [recipe.id, recipe.chamber_id]
    if tool == "get_recipe_active":
        return [args["recipe_id"]]
    if tool == "get_part_stock":
        return [args["part_id"]]
    if tool == "get_maintenance_event_status":
        return [args["maintenance_event_id"]]
    if tool == "get_maintenance_event_target":
        event = session.get(MaintenanceEvent, args["maintenance_event_id"])
        target_id = event.tool_id if event.tool_id is not None else event.chamber_id
        return [event.id, target_id]
    if tool == "list_maintenance_events_for_chamber":
        chamber = session.get(Chamber, args["chamber_id"])
        open_ids = [
            e.id for e in chamber.maintenance_events if e.status in ("scheduled", "in_progress")
        ]
        return [chamber.id, *open_ids]
    if tool == "get_part_usage_in_maintenance_event":
        return [args["maintenance_event_id"], args["part_id"]]
    if tool == "list_chambers_for_tool":
        tool_obj = session.get(Tool, args["tool_id"])
        return [tool_obj.id, *[c.id for c in tool_obj.chambers]]
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

    pools = {
        "tool_id": graph.tool_ids,
        "chamber_id": graph.chamber_ids,
        "recipe_id": graph.recipe_ids,
        "part_id": graph.part_ids,
        "maintenance_event_id": graph.maintenance_event_ids,
    }

    for tool, phrase_fn, arg_keys in _TEMPLATES:
        for _ in range(_ANSWERABLE_PER_TEMPLATE):
            args = {}
            for key in arg_keys:
                args[key] = rng.choice(pools[key])
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
                    args[key] = rng.choice(pools[key])
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
