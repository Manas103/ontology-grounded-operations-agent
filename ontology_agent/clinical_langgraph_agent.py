"""A real LangGraph orchestration of the clinical read path, satisfying
the resume's "LangGraph" stack entry with a genuinely compiled and
invoked graph rather than a decorative import.

Two nodes: `choose_tool` picks one of the 11 catalog tools (and its
typed argument) for a question, `execute_tool` runs the scope check
(`clinical_scope_guard.ScopeGuard`) and then the tool call itself,
turning `ClinicalObjectNotFoundError` into a refusal. `choose_tool` is
pluggable, same split as `llm_client.py` in the equipment domain: the
free, deterministic `deterministic_tool_chooser` (wraps
`clinical_qa_router.route_clinical_question`) is what the full 300
-question benchmark actually runs on, because that benchmark needs to be
exactly reproducible; `claude_cli_tool_chooser` shells out to the real
`claude` CLI and is exercised on a smaller, explicitly-counted sample,
exactly like `llm_client.ClaudeCLIClient`. The graph itself, and the
state it threads through its two nodes, is identical either way; only
the chooser function passed to `build_clinical_agent_graph` differs.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from ontology_agent.clinical_qa_router import ClinicalRoutingError, route_clinical_question
from ontology_agent.clinical_scope_guard import ScopeGuard
from ontology_agent.clinical_tools import CLINICAL_TOOL_CATALOG, ClinicalObjectNotFoundError

CLAUDE_CLI_PATH = "claude"

ToolChoice = tuple[str | None, dict[str, str]]
ToolChooser = Callable[[str], ToolChoice]


class ClinicalAgentState(TypedDict):
    question: str
    tool: str | None
    args: dict[str, str]
    refused: bool
    answer_summary: str | None
    source_object_ids: list[str]
    refusal_reason: str | None


def deterministic_tool_chooser(question: str) -> ToolChoice:
    try:
        route = route_clinical_question(question)
    except ClinicalRoutingError:
        return None, {}
    return route.tool, route.args


def _tool_catalog_prompt_block() -> str:
    from ontology_agent.clinical_tools import CLINICAL_TOOL_REQUIRED_ARGS

    lines = []
    for name, args in CLINICAL_TOOL_REQUIRED_ARGS.items():
        lines.append(f"- {name}({', '.join(args)})")
    return "\n".join(lines)


TOOL_SELECTION_PROMPT_TEMPLATE = """You are a tool-routing component for a clinical chart \
assistant. You may not answer questions from general knowledge. Your only job is to \
choose exactly one tool from the catalog below and the arguments to call it with, based \
on the clinician's question. FHIR resource ids are UUIDs (for example \
e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01), always quoted exactly as they appear in the question.

Tool catalog:
{catalog}

Respond with exactly one line of JSON and nothing else, in this shape:
{{"tool": "<tool_name>", "args": {{"<arg_name>": "<value from the question>"}}}}

If the question does not name a resource id that lets you call one of these tools, \
respond with:
{{"tool": null, "args": {{}}}}

Question: {question}
"""


def claude_cli_tool_chooser(
    question: str, claude_path: str = CLAUDE_CLI_PATH, timeout_seconds: int = 60
) -> ToolChoice:
    prompt = TOOL_SELECTION_PROMPT_TEMPLATE.format(
        catalog=_tool_catalog_prompt_block(), question=question
    )
    result = subprocess.run(
        [claude_path, "-p", prompt],
        capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    raw_output = result.stdout.strip()
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return None, {}
    try:
        choice = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, {}
    return choice.get("tool"), (choice.get("args") or {})


def build_clinical_agent_graph(session: Session, tool_chooser: ToolChooser):
    guard = ScopeGuard()

    def choose_tool(state: ClinicalAgentState) -> ClinicalAgentState:
        tool, args = tool_chooser(state["question"])
        return {**state, "tool": tool, "args": args}

    def execute_tool(state: ClinicalAgentState) -> ClinicalAgentState:
        tool_name = state["tool"]
        if not tool_name:
            return {
                **state, "refused": True, "answer_summary": None, "source_object_ids": [],
                "refusal_reason": "no tool could be chosen for this question",
            }
        scope_result = guard.check(tool_name)
        if not scope_result.in_scope:
            return {
                **state, "refused": True, "answer_summary": None, "source_object_ids": [],
                "refusal_reason": scope_result.refusal_reason,
            }
        tool_fn = CLINICAL_TOOL_CATALOG[tool_name]
        try:
            result = tool_fn(session, **state["args"])
        except ClinicalObjectNotFoundError as exc:
            return {
                **state, "refused": True, "answer_summary": None, "source_object_ids": [],
                "refusal_reason": str(exc),
            }
        except TypeError as exc:
            return {
                **state, "refused": True, "answer_summary": None, "source_object_ids": [],
                "refusal_reason": f"tool call could not be executed: {exc}",
            }
        return {
            **state, "refused": False, "answer_summary": result.summary,
            "source_object_ids": result.source_object_ids, "refusal_reason": None,
        }

    graph = StateGraph(ClinicalAgentState)
    graph.add_node("choose_tool", choose_tool)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("choose_tool")
    graph.add_edge("choose_tool", "execute_tool")
    graph.add_edge("execute_tool", END)
    return graph.compile()


def run_clinical_agent(app, question: str) -> ClinicalAgentState:
    initial: ClinicalAgentState = {
        "question": question, "tool": None, "args": {}, "refused": False,
        "answer_summary": None, "source_object_ids": [], "refusal_reason": None,
    }
    return app.invoke(initial)
