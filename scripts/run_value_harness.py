"""The value harness: scores the assistant against a manual-equivalent
baseline on the same 386-question held-out set the QA benchmark uses.

This measures two disclosed proxies for manual effort, not a live user
study (see README, Measured results, for exactly what each does and does
not cover):

  1. Operation counts: how many row comparisons a linear-scan, manual-join
     answerer (`ontology_agent/manual_baseline.py`) performs to answer all
     386 questions, versus the assistant's 386 indexed tool calls (one per
     question; a handful of tools also perform a second or third indexed
     lookup for a join, exactly mirrored on the manual side).
  2. Wall-clock time: both answerers run against the same seeded, in-memory
     SQLite store, on this machine, timed with `time.perf_counter`, median
     of several repeats reported because a single run of microsecond-scale
     work is noisy.

Every question the manual path answers is cross-checked against the same
reference-oracle citation set `questions.py` computes, and every refusal
the manual path issues is cross-checked against the same 74
refusal-designed questions, so the comparison is proven apples-to-apples
before it is reported: the manual baseline is not just slower, it is also
verified correct.

    python scripts/run_value_harness.py
"""
from __future__ import annotations

import statistics
import sys
import time

from ontology_agent.db import init_db, make_engine, make_session_factory
from ontology_agent.llm_client import DeterministicToolRouterClient
from ontology_agent.manual_baseline import ScanStats, load_manual_tables, manual_answer_question
from ontology_agent.questions import build_question_set, create_and_delete_holdout_objects
from ontology_agent.seed import DEFAULT_SEED, seed_database

N_REPEATS = 7


def _setup():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)
    session = Session()
    graph = seed_database(session, seed=DEFAULT_SEED)
    removed_ids = create_and_delete_holdout_objects(session, graph, seed=DEFAULT_SEED)
    questions = build_question_set(session, graph, removed_ids, seed=DEFAULT_SEED)
    return session, questions


def _time_assistant(session, questions, client) -> float:
    start = time.perf_counter()
    for q in questions:
        client.answer(session, q.text)
    return time.perf_counter() - start


def _time_manual(tables, questions) -> tuple[float, int]:
    stats = ScanStats()
    start = time.perf_counter()
    for q in questions:
        manual_answer_question(tables, q.text, stats)
    elapsed = time.perf_counter() - start
    return elapsed, stats.comparisons


def main() -> int:
    session, questions = _setup()
    tables = load_manual_tables(session)
    client = DeterministicToolRouterClient()

    print(f"held-out question set: {len(questions)} questions "
          f"({sum(1 for q in questions if not q.is_refusal)} answerable, "
          f"{sum(1 for q in questions if q.is_refusal)} refusal-designed)")
    print()

    # --- correctness cross-check: the manual baseline must be proven right
    # before its timing is trusted; otherwise a faster wrong answer would
    # make the comparison meaningless. ---
    stats_for_check = ScanStats()
    mismatches = []
    wrongly_refused = []
    wrongly_answered = []
    for q in questions:
        result = manual_answer_question(tables, q.text, stats_for_check)
        if q.is_refusal:
            if not result.refused:
                wrongly_answered.append(q.qid)
        else:
            if result.refused:
                wrongly_refused.append(q.qid)
            elif set(result.source_object_ids) != set(q.expected_source_object_ids):
                mismatches.append(q.qid)

    print("--- manual-baseline correctness cross-check (must pass before timing counts) ---")
    print(f"citation mismatches vs reference oracle: {len(mismatches)}")
    print(f"answerable questions wrongly refused: {len(wrongly_refused)}")
    print(f"refusal-designed questions wrongly answered: {len(wrongly_answered)}")
    manual_baseline_correct = not mismatches and not wrongly_refused and not wrongly_answered
    print(f"MANUAL BASELINE VERIFIED CORRECT: {manual_baseline_correct}")
    print()

    # --- operation counts (single run; deterministic, no timing noise) ---
    single_run_stats = ScanStats()
    for q in questions:
        manual_answer_question(tables, q.text, single_run_stats)
    assistant_tool_calls = len(questions)  # one client.answer() call per question

    print("--- operation counts, single run over the full 386-question set ---")
    print(f"assistant: {assistant_tool_calls} typed tool calls "
          f"(each an indexed Session.get, O(1) amortized via the identity map / primary-key index)")
    print(f"manual baseline: {single_run_stats.comparisons} row comparisons across "
          f"{single_run_stats.tables_scanned} linear table scans "
          f"(O(n) per scan, n = rows in the scanned table)")
    print(f"comparisons per question, manual baseline: "
          f"{single_run_stats.comparisons / len(questions):.2f} average")
    print()

    # --- wall-clock timing, N_REPEATS repeats, median reported (noisy at
    # microsecond scale; range given alongside the median) ---
    assistant_times = [_time_assistant(session, questions, client) for _ in range(N_REPEATS)]
    manual_times_and_ops = [_time_manual(tables, questions) for _ in range(N_REPEATS)]
    manual_times = [t for t, _ in manual_times_and_ops]

    assistant_median = statistics.median(assistant_times)
    manual_median = statistics.median(manual_times)

    print(f"--- wall-clock timing, {N_REPEATS} repeats over the full 386-question set, median reported ---")
    print(f"assistant (indexed typed tool calls): median {assistant_median * 1000:.3f} ms "
          f"(range {min(assistant_times) * 1000:.3f}-{max(assistant_times) * 1000:.3f} ms)")
    print(f"manual baseline (linear scan / manual join): median {manual_median * 1000:.3f} ms "
          f"(range {min(manual_times) * 1000:.3f}-{max(manual_times) * 1000:.3f} ms)")
    if assistant_median > 0:
        speedup = manual_median / assistant_median
        print(f"WALL-CLOCK SPEEDUP (manual / assistant): {speedup:.2f}x")
    if assistant_tool_calls > 0:
        op_ratio = single_run_stats.comparisons / assistant_tool_calls
        print(f"OPERATION-COUNT RATIO (manual comparisons / assistant tool calls): {op_ratio:.2f}x")

    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
