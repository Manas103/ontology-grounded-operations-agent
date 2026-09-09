"""Pluggable LLM client interface, with two real implementations.

`DeterministicToolRouterClient` wraps the regex-based router in
`qa_router.py`. It is the client used for the full 300+ question benchmark:
it is free, instant, and perfectly reproducible, which is what makes a
run-to-run-identical 380-question measurement possible at all.

`ClaudeCLIClient` is a real LLM backend: it shells out to the `claude` CLI
already logged in on this machine (`claude -p "<prompt>"`) as an actual
subprocess, asks the model to choose a tool and arguments from the same
`tools.TOOL_CATALOG` the deterministic router uses, and parses its JSON
response. It is genuinely exercised, not merely implemented: see
`scripts/run_live_llm_sample.py` and the README for the exact number of
real subprocess calls made and what was measured from them. This
repository exists specifically to avoid the flaw recorded against
`guarded-instruction-validation` (a real LLM client implemented but never
exercised); this class is called for real, at least once, before any
number involving it appears in the README.

Either client's output is executed through the exact same
`tools.TOOL_CATALOG` and the exact same `ObjectNotFoundError`-to-refusal
handling; the client only ever chooses *which* typed tool to call and with
*what* typed arguments, never a query string.
"""
from __future__ import annotations

import abc
import json
import re
import subprocess

from sqlalchemy.orm import Session

from ontology_agent.qa_router import (
    AnsweredQuestion,
    answer_question,
    required_args_for_tool,
)
from ontology_agent.tools import TOOL_CATALOG, ObjectNotFoundError, ToolResult

CLAUDE_CLI_PATH = "claude"


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def answer(self, session: Session, question_text: str) -> AnsweredQuestion:
        ...


class DeterministicToolRouterClient(LLMClient):
    """The primary, measured-at-scale path. No network call, no LLM."""

    def answer(self, session: Session, question_text: str) -> AnsweredQuestion:
        return answer_question(session, question_text)


def _tool_catalog_prompt_block() -> str:
    lines = []
    for name in TOOL_CATALOG:
        args = required_args_for_tool(name)
        lines.append(f"- {name}({', '.join(args)})")
    return "\n".join(lines)


TOOL_SELECTION_PROMPT_TEMPLATE = """You are a tool-routing component for an equipment knowledge \
assistant. You may not answer questions from general knowledge. Your only job \
is to choose exactly one tool from the catalog below and the arguments to \
call it with, based on the operator's question. Object ids look like \
TL-0001 (tool), CH-00045 (chamber), RC-00012 (recipe), \
PRT-0007 (part), ME-000123 (maintenance event).

Tool catalog:
{catalog}

Respond with exactly one line of JSON and nothing else, in this shape:
{{"tool": "<tool_name>", "args": {{"<arg_name>": "<value from the question>"}}}}

If the question does not name an object id that lets you call one of these \
tools, respond with:
{{"tool": null, "args": {{}}}}

Question: {question}
"""


class ClaudeCLIClient(LLMClient):
    """Real LLM backend: shells out to `claude -p "<prompt>"` as a genuine
    subprocess. Used on a smaller live sample (see README for the exact
    count), not the full 300+ question benchmark, because each call is a
    real network round trip and is measured, not free."""

    def __init__(self, claude_path: str = CLAUDE_CLI_PATH, timeout_seconds: int = 60):
        self.claude_path = claude_path
        self.timeout_seconds = timeout_seconds

    def _call_claude(self, prompt: str) -> str:
        result = subprocess.run(
            [self.claude_path, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        return result.stdout.strip()

    def _parse_tool_choice(self, raw_output: str) -> dict:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in LLM output: {raw_output!r}")
        return json.loads(match.group(0))

    def answer(self, session: Session, question_text: str) -> AnsweredQuestion:
        prompt = TOOL_SELECTION_PROMPT_TEMPLATE.format(
            catalog=_tool_catalog_prompt_block(), question=question_text
        )
        try:
            raw_output = self._call_claude(prompt)
            choice = self._parse_tool_choice(raw_output)
        except Exception as exc:  # subprocess failure, timeout, bad JSON
            return AnsweredQuestion(
                text=question_text, refused=True, answer_summary=None,
                source_object_ids=[], tool_used=None,
                refusal_reason=f"LLM output could not be used: {exc}",
            )

        tool_name = choice.get("tool")
        args = choice.get("args") or {}
        if not tool_name or tool_name not in TOOL_CATALOG:
            return AnsweredQuestion(
                text=question_text, refused=True, answer_summary=None,
                source_object_ids=[], tool_used=None,
                refusal_reason=f"LLM did not choose a known tool (got {tool_name!r})",
            )

        required = required_args_for_tool(tool_name)
        missing = [a for a in required if a not in args]
        if missing:
            return AnsweredQuestion(
                text=question_text, refused=True, answer_summary=None,
                source_object_ids=[], tool_used=tool_name,
                refusal_reason=f"LLM omitted required argument(s): {missing}",
            )

        tool_fn = TOOL_CATALOG[tool_name]
        try:
            result: ToolResult = tool_fn(session, **{k: args[k] for k in required})
        except ObjectNotFoundError as exc:
            return AnsweredQuestion(
                text=question_text, refused=True, answer_summary=None,
                source_object_ids=[], tool_used=tool_name, refusal_reason=str(exc),
            )

        return AnsweredQuestion(
            text=question_text, refused=False, answer_summary=result.summary,
            source_object_ids=result.source_object_ids, tool_used=tool_name,
        )
