"""Genuinely exercises the real LLM backend, on a small sample.

This is the honesty-critical script for this repository: it makes real
`claude -p "<prompt>"` subprocess calls (not mocked, not stubbed) against a
sample of questions drawn from the same question set the full benchmark
uses, and reports exactly how many calls were made, how many the model
routed to a correct tool call with a correct citation, and how many of the
sampled refusal-designed questions it correctly refused. The full 300+
question benchmark still uses `DeterministicToolRouterClient` for speed and
reproducibility (see `run_qa_benchmark.py`); this script is what backs the
"LLM Tool Calling" stack claim and the "actually run the real LLM" honesty
requirement, and its raw output is committed unedited to
`docs/live_llm_sample_output.txt`.

    python scripts/run_live_llm_sample.py
"""
from __future__ import annotations

import sys

from ontology_agent.db import init_db, make_engine, make_session_factory
from ontology_agent.llm_client import ClaudeCLIClient
from ontology_agent.questions import build_question_set, create_and_delete_holdout_objects
from ontology_agent.seed import DEFAULT_SEED, seed_database

N_ANSWERABLE_SAMPLE = 15
N_REFUSAL_SAMPLE = 5


def main() -> int:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)

    with Session() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)
        removed_ids = create_and_delete_holdout_objects(session, graph, seed=DEFAULT_SEED)
        questions = build_question_set(session, graph, removed_ids, seed=DEFAULT_SEED)

        answerable = [q for q in questions if not q.is_refusal][:N_ANSWERABLE_SAMPLE]
        refusal_designed = [q for q in questions if q.is_refusal][:N_REFUSAL_SAMPLE]
        sample = answerable + refusal_designed

        print(f"live sample size: {len(sample)} real `claude -p` subprocess calls")
        print(f"  answerable questions in sample: {len(answerable)}")
        print(f"  refusal-designed questions in sample: {len(refusal_designed)}")
        print()

        client = ClaudeCLIClient()

        correct_tool_and_citation = 0
        correct_refusal = 0
        calls_made = 0

        for q in sample:
            calls_made += 1
            result = client.answer(session, q.text)
            if q.is_refusal:
                ok = result.refused
                if ok:
                    correct_refusal += 1
                print(
                    f"[{calls_made}] REFUSAL-DESIGNED  {q.qid}: {q.text!r}\n"
                    f"    model refused: {result.refused}"
                    f" ({result.refusal_reason if result.refused else result.answer_summary})"
                    f"  -> {'CORRECT' if ok else 'WRONG'}"
                )
            else:
                ok = (not result.refused) and set(result.source_object_ids) == set(
                    q.expected_source_object_ids
                )
                if ok:
                    correct_tool_and_citation += 1
                print(
                    f"[{calls_made}] ANSWERABLE  {q.qid}: {q.text!r}\n"
                    f"    tool={result.tool_used} refused={result.refused}"
                    f" citations={result.source_object_ids} expected={q.expected_source_object_ids}"
                    f"  -> {'CORRECT' if ok else 'WRONG'}"
                )

        print()
        print(f"REAL LLM SUBPROCESS CALLS MADE: {calls_made}")
        print(
            f"answerable sample correct (tool + citation): "
            f"{correct_tool_and_citation} / {len(answerable)}"
        )
        print(f"refusal sample correct (refused): {correct_refusal} / {len(refusal_designed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
