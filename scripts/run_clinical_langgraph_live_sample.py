"""The real LangGraph + live LLM sample: 20 genuine `claude -p` subprocess
calls routed through a compiled LangGraph graph.

    PYTHONPATH=. python scripts/run_clinical_langgraph_live_sample.py

Requires the clinical tables to already be loaded and the `claude` CLI on
PATH and logged in. Draws 15 answerable and 5 refusal-designed questions
from the same 300-question set `run_clinical_qa_benchmark.py` uses, and
answers each one by invoking `build_clinical_agent_graph(session,
claude_cli_tool_chooser)`, i.e. a real compiled LangGraph app whose
`choose_tool` node is backed by a genuine `claude -p` subprocess call, not
the deterministic router. Exactly like `run_live_llm_sample.py` in the
equipment domain, this is a disclosed sample, not the full benchmark.
"""
from __future__ import annotations

from ontology_agent.db import make_engine, make_session_factory
from ontology_agent.clinical_langgraph_agent import (
    build_clinical_agent_graph,
    claude_cli_tool_chooser,
    run_clinical_agent,
)
from ontology_agent.clinical_questions import (
    build_question_set,
    create_and_delete_holdout_objects,
    load_id_pools,
)

SEED = 20261017
N_ANSWERABLE = 15
N_REFUSAL = 5


def main() -> None:
    engine = make_engine()
    Session = make_session_factory(engine)
    session = Session()

    removed_ids = create_and_delete_holdout_objects(session, seed=SEED)
    pools = load_id_pools(session)
    questions = build_question_set(session, pools, removed_ids, seed=SEED)
    answerable_sample = [q for q in questions if not q.is_refusal][:N_ANSWERABLE]
    refusal_sample = [q for q in questions if q.is_refusal][:N_REFUSAL]
    sample = answerable_sample + refusal_sample

    app = build_clinical_agent_graph(session, claude_cli_tool_chooser)

    calls_made = 0
    answerable_correct = 0
    refusal_correct = 0
    for i, q in enumerate(sample, start=1):
        result = run_clinical_agent(app, q.text)
        calls_made += 1
        kind = "REFUSAL-DESIGNED" if q.is_refusal else "ANSWERABLE"
        print(f"[{i}] {kind}  {q.qid}: {q.text!r}")
        if q.is_refusal:
            correct = result["refused"]
            if correct:
                refusal_correct += 1
            print(f"    model refused: {result['refused']}  -> {'CORRECT' if correct else 'WRONG'}")
        else:
            correct = (not result["refused"]) and (
                set(result["source_object_ids"]) == set(q.expected_source_object_ids)
            )
            if correct:
                answerable_correct += 1
            print(
                f"    tool={result['tool']} refused={result['refused']} "
                f"citations={result['source_object_ids']} -> {'CORRECT' if correct else 'WRONG'}"
            )

    print(f"\nREAL LLM SUBPROCESS CALLS MADE: {calls_made}")
    print(f"answerable sample correct (tool + citation): {answerable_correct} / {len(answerable_sample)}")
    print(f"refusal sample correct (refused): {refusal_correct} / {len(refusal_sample)}")

    session.close()


if __name__ == "__main__":
    main()
