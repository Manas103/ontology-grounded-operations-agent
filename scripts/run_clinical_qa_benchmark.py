"""The 300-question clinical held-out benchmark (deterministic client).

    PYTHONPATH=. python scripts/run_clinical_qa_benchmark.py

Requires the clinical tables to already be loaded (`init_clinical_db.py`).
Builds the 300-question set against the real loaded patients, creates and
deletes the holdout objects for the refusal-designed questions, answers
every question through the deterministic router, and reports citation
correctness on the answerable half and the refusal rate on the holdout
half, exactly as `run_qa_benchmark.py` does for the equipment domain.
"""
from __future__ import annotations

from ontology_agent.db import make_engine, make_session_factory
from ontology_agent.clinical_qa_router import answer_clinical_question
from ontology_agent.clinical_questions import (
    TOTAL_QUESTIONS,
    build_question_set,
    create_and_delete_holdout_objects,
    load_id_pools,
)

SEED = 20261017


def main() -> None:
    engine = make_engine()
    Session = make_session_factory(engine)
    session = Session()

    removed_ids = create_and_delete_holdout_objects(session, seed=SEED)
    pools = load_id_pools(session)
    questions = build_question_set(session, pools, removed_ids, seed=SEED)
    assert len(questions) == TOTAL_QUESTIONS

    answerable = [q for q in questions if not q.is_refusal]
    refusal = [q for q in questions if q.is_refusal]

    print(f"total questions: {len(questions)}")
    print(f"answerable (object present): {len(answerable)}")
    print(f"refusal-designed (object removed): {len(refusal)}")

    answered_correctly = 0
    wrongly_refused = 0
    for q in answerable:
        result = answer_clinical_question(session, q.text)
        if result.refused:
            wrongly_refused += 1
            continue
        if set(result.source_object_ids) == set(q.expected_source_object_ids):
            answered_correctly += 1

    print("\n--- citation correctness on answerable questions ---")
    print(f"answerable questions answered without refusing: {len(answerable) - wrongly_refused}")
    print(f"answerable questions incorrectly refused: {wrongly_refused}")
    print(f"citations correct: {answered_correctly} / {len(answerable)}")
    print(f"CITATION CORRECTNESS RATE: {100.0 * answered_correctly / len(answerable):.4f}%")

    refused_correctly = 0
    for q in refusal:
        result = answer_clinical_question(session, q.text)
        if result.refused:
            refused_correctly += 1

    print(f"\n--- refusal rate on the {len(refusal)} holdout (missing-object) questions ---")
    print(f"refused: {refused_correctly} / {len(refusal)}")
    print(f"REFUSAL RATE: {100.0 * refused_correctly / len(refusal):.4f}%")

    print(
        f"\n300-QUESTION SUMMARY: {answered_correctly + refused_correctly} of {len(questions)} "
        "answers cited their sources or refused"
    )

    session.close()


if __name__ == "__main__":
    main()
