from __future__ import annotations

import pytest

from ontology_agent.manual_baseline import (
    MANUAL_TOOL_CATALOG,
    ManualNotFoundError,
    ScanStats,
    load_manual_tables,
    manual_answer_question,
    manual_get_asset_status,
    manual_get_site_technician_count,
)
from ontology_agent.questions import build_question_set
from ontology_agent.tools import TOOL_CATALOG


def test_manual_tool_catalog_mirrors_the_real_one():
    """The value harness only means something if it is answering the same
    12 question shapes the assistant answers, no more, no fewer."""
    assert set(MANUAL_TOOL_CATALOG.keys()) == set(TOOL_CATALOG.keys())
    assert len(MANUAL_TOOL_CATALOG) == 12


def test_load_manual_tables_row_counts_match_the_seed(seeded_session):
    session, graph = seeded_session
    tables = load_manual_tables(session)
    assert len(tables.sites) == len(graph.site_ids)
    assert len(tables.technicians) == len(graph.technician_ids)
    assert len(tables.assets) == len(graph.asset_ids)
    assert len(tables.parts) == len(graph.part_ids)
    assert len(tables.work_orders) == len(graph.work_order_ids)


def test_manual_lookup_scans_every_row_to_confirm_a_missing_id(seeded_session):
    """The discipline the whole harness stands on: a manual lookup may not
    shortcut a missing-object check. Confirming absence must cost exactly
    one comparison per row in the table, never fewer."""
    session, graph = seeded_session
    tables = load_manual_tables(session)
    stats = ScanStats()
    with pytest.raises(ManualNotFoundError):
        manual_get_asset_status(tables, stats, asset_id="AST-99999")
    assert stats.comparisons == len(tables.assets)


def test_manual_lookup_scans_every_technician_row_for_a_site_count(seeded_session):
    session, graph = seeded_session
    tables = load_manual_tables(session)
    stats = ScanStats()
    manual_get_site_technician_count(tables, stats, site_id=graph.site_ids[0])
    # one scan to find the site (found on first comparison in the worst
    # case among 6 sites) plus one full pass over every technician
    assert stats.comparisons >= len(tables.technicians)


def test_manual_baseline_matches_reference_oracle_on_every_answerable_question(
    seeded_session_with_holdout,
):
    session, graph, removed_ids = seeded_session_with_holdout
    questions = build_question_set(session, graph, removed_ids, seed=20260724)
    tables = load_manual_tables(session)
    stats = ScanStats()

    answerable = [q for q in questions if not q.is_refusal]
    assert len(answerable) == 312

    mismatches = []
    for q in answerable:
        result = manual_answer_question(tables, q.text, stats)
        if result.refused:
            mismatches.append((q.qid, "wrongly refused"))
        elif set(result.source_object_ids) != set(q.expected_source_object_ids):
            mismatches.append((q.qid, "citation mismatch"))
    assert mismatches == []


def test_manual_baseline_refuses_on_every_holdout_question(seeded_session_with_holdout):
    session, graph, removed_ids = seeded_session_with_holdout
    questions = build_question_set(session, graph, removed_ids, seed=20260724)
    tables = load_manual_tables(session)
    stats = ScanStats()

    refusal_designed = [q for q in questions if q.is_refusal]
    assert len(refusal_designed) == 74

    wrongly_answered = [
        q.qid for q in refusal_designed
        if not manual_answer_question(tables, q.text, stats).refused
    ]
    assert wrongly_answered == []


def test_manual_and_assistant_agree_on_a_multi_hop_join_citation(seeded_session):
    """get_part_usage_in_work_order is the one tool that needs a genuine
    join (work order id and part id both resolved, then a third table
    consulted for the linking row); manual and assistant must cite
    identically for it."""
    from ontology_agent.tools import get_part_usage_in_work_order

    session, graph = seeded_session
    tables = load_manual_tables(session)
    stats = ScanStats()

    wo_id = graph.work_order_ids[0]
    part_id = graph.part_ids[0]

    real_result = get_part_usage_in_work_order(session, work_order_id=wo_id, part_id=part_id)
    manual_result = MANUAL_TOOL_CATALOG["get_part_usage_in_work_order"](
        tables, stats, work_order_id=wo_id, part_id=part_id
    )
    assert manual_result.value == real_result.value
    assert set(manual_result.source_object_ids) == set(real_result.source_object_ids)
