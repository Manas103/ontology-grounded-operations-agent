"""Deterministic rule-based tool router.

This is the primary measured path for the 300+ question benchmark: fast
(no network call) and perfectly reproducible run to run, which is what a
300+ question, 74-refusal held-out evaluation needs to be trustworthy
rather than anecdotal. It is intentionally simple: match a question
against a small set of marker phrases to pick a tool, pull out typed
object ids by their prefix, and call the tool. It never sees, builds, or
executes SQL text; the only thing it can do is call a function out of
`tools.TOOL_CATALOG` with keyword arguments extracted from the question.

A second, independently exercised path (`llm_client.ClaudeCLIClient`) does
the same job by asking a real LLM to choose the tool and arguments; see
`llm_client.py` and the README for what was actually run with which
backend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ontology_agent.tools import TOOL_CATALOG, ObjectNotFoundError, ToolResult

ID_PATTERN = re.compile(r"\b(WO|AST|TCH|PRT|STE)-(\d+)\b")

PREFIX_TO_ARG = {
    "WO": "work_order_id",
    "AST": "asset_id",
    "TCH": "technician_id",
    "PRT": "part_id",
    "STE": "site_id",
}

# Ordered most-specific-marker-first so a longer, more specific phrase never
# loses to a shorter substring of itself.
_MARKERS = [
    ("were used on work order", "get_part_usage_in_work_order"),
    ("current status of work order", "get_work_order_status"),
    ("priority level is assigned to work order", "get_work_order_priority"),
    ("technician is assigned to work order", "get_work_order_technician"),
    ("asset is associated with work order", "get_work_order_asset"),
    ("operating status of asset", "get_asset_status"),
    ("site is asset", "get_asset_site"),
    ("site is technician", "get_technician_site"),
    ("currently active", "get_technician_active"),
    ("open work orders does technician", "list_open_work_orders_for_technician"),
    ("currently in stock", "get_part_stock"),
    ("technicians are assigned to site", "get_site_technician_count"),
]


class RoutingError(Exception):
    """The question could not be matched to a known tool at all."""


@dataclass
class RouteResult:
    tool: str
    args: dict[str, str]


def route_question(text: str) -> RouteResult:
    """Pick a tool and extract typed arguments from question text.

    Deliberately does not fall back to any generic query path: if no
    marker matches, this raises rather than guessing a tool.
    """
    lowered = text.lower()
    tool = None
    for marker, candidate_tool in _MARKERS:
        if marker in lowered:
            tool = candidate_tool
            break
    if tool is None:
        raise RoutingError(f"no known intent matched question: {text!r}")

    found_ids = ID_PATTERN.findall(text)
    args: dict[str, str] = {}
    for prefix, digits in found_ids:
        arg_name = PREFIX_TO_ARG[prefix]
        args[arg_name] = f"{prefix}-{digits}"

    required_arg_names = required_args_for_tool(tool)
    missing = [a for a in required_arg_names if a not in args]
    if missing:
        raise RoutingError(
            f"tool {tool!r} needs {missing} but question did not name them: {text!r}"
        )
    return RouteResult(tool=tool, args={k: args[k] for k in required_arg_names})


def required_args_for_tool(tool: str) -> list[str]:
    return {
        "get_work_order_status": ["work_order_id"],
        "get_work_order_priority": ["work_order_id"],
        "get_work_order_technician": ["work_order_id"],
        "get_work_order_asset": ["work_order_id"],
        "get_asset_status": ["asset_id"],
        "get_asset_site": ["asset_id"],
        "get_technician_site": ["technician_id"],
        "get_technician_active": ["technician_id"],
        "list_open_work_orders_for_technician": ["technician_id"],
        "get_part_stock": ["part_id"],
        "get_part_usage_in_work_order": ["work_order_id", "part_id"],
        "get_site_technician_count": ["site_id"],
    }[tool]


@dataclass
class AnsweredQuestion:
    text: str
    refused: bool
    answer_summary: str | None
    source_object_ids: list[str]
    tool_used: str | None
    refusal_reason: str | None = None


def answer_question(session: Session, text: str) -> AnsweredQuestion:
    """Route, call the tool, and turn `ObjectNotFoundError` into a refusal.

    This is the single function both the benchmark harness and (indirectly,
    through the same tool catalog) the LLM path exercise, so "refuses
    rather than guesses" is one code path, not a policy repeated in two
    places that could drift apart.
    """
    try:
        route = route_question(text)
    except RoutingError as exc:
        return AnsweredQuestion(
            text=text, refused=True, answer_summary=None, source_object_ids=[],
            tool_used=None, refusal_reason=str(exc),
        )

    tool_fn = TOOL_CATALOG[route.tool]
    try:
        result: ToolResult = tool_fn(session, **route.args)
    except ObjectNotFoundError as exc:
        return AnsweredQuestion(
            text=text, refused=True, answer_summary=None, source_object_ids=[],
            tool_used=route.tool, refusal_reason=str(exc),
        )

    return AnsweredQuestion(
        text=text, refused=False, answer_summary=result.summary,
        source_object_ids=result.source_object_ids, tool_used=route.tool,
    )
