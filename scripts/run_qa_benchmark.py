"""The held-out Q&A benchmark: 300+ questions, 74 built to refuse.

Builds a fresh in-memory SQLite database, seeds it deterministically,
creates-and-deletes the holdout objects, generates the question set, and
runs every question through `DeterministicToolRouterClient` (the primary,
reproducible measured path; see `scripts/run_live_llm_sample.py` for the
real-LLM sample). Reports citation correctness on answerable questions and
the refusal rate on the 74 designed to fail.

    python scripts/run_qa_benchmark.py
"""
from __future__ import annotations

import sys

from ontology_agent.db import init_db, make_engine, make_session_factory
from ontology_agent.llm_client import DeterministicToolRouterClient
from ontology_agent.questions import build_question_set, create_and_delete_holdout_objects
from ontology_agent.seed import DEFAULT_SEED, seed_database


def main() -> int:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)

    with Session() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)
        removed_ids = create_and_delete_holdout_objects(session, graph, seed=DEFAULT_SEED)
        questions = build_question_set(session, graph, removed_ids, seed=DEFAULT_SEED)

        print(f"total questions: {len(questions)}")
        answerable = [q for q in questions if not q.is_refusal]
        refusal_designed = [q for q in questions if q.is_refusal]
        print(f"answerable (object present): {len(answerable)}")
        print(f"refusal-designed (object removed): {len(refusal_designed)}")

        client = DeterministicToolRouterClient()

        citation_correct = 0
        citation_total = 0
        wrongly_refused_answerable = []
        wrong_citation_examples = []

        for q in answerable:
            result = client.answer(session, q.text)
            citation_total += 1
            if result.refused:
                wrongly_refused_answerable.append((q.qid, q.text, result.refusal_reason))
                continue
            if set(result.source_object_ids) == set(q.expected_source_object_ids):
                citation_correct += 1
            else:
                wrong_citation_examples.append(
                    (q.qid, q.text, result.source_object_ids, q.expected_source_object_ids)
                )

        refused_count = 0
        wrongly_answered_refusal = []
        for q in refusal_designed:
            result = client.answer(session, q.text)
            if result.refused:
                refused_count += 1
            else:
                wrongly_answered_refusal.append((q.qid, q.text, result.answer_summary))

        print()
        print("--- citation correctness on answerable questions ---")
        print(f"answerable questions answered without refusing: {citation_total - len(wrongly_refused_answerable)}")
        print(f"answerable questions incorrectly refused: {len(wrongly_refused_answerable)}")
        for qid, text, reason in wrongly_refused_answerable[:10]:
            print(f"  {qid}: {text!r} -> refused ({reason})")
        print(f"citations correct: {citation_correct} / {citation_total}")
        citation_rate = citation_correct / citation_total if citation_total else 0.0
        print(f"CITATION CORRECTNESS RATE: {citation_rate:.4%}")
        for qid, text, got, expected in wrong_citation_examples[:10]:
            print(f"  {qid}: {text!r} got={got} expected={expected}")

        print()
        print("--- refusal rate on the 74 holdout (missing-object) questions ---")
        print(f"refused: {refused_count} / {len(refusal_designed)}")
        refusal_rate = refused_count / len(refusal_designed) if refusal_designed else 0.0
        print(f"REFUSAL RATE: {refusal_rate:.4%}")
        for qid, text, summary in wrongly_answered_refusal:
            print(f"  {qid}: {text!r} -> ANSWERED INSTEAD OF REFUSING: {summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
