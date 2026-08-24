from __future__ import annotations

from ontology_agent.qa_router import answer_question, route_question


def test_router_maps_each_marker_phrase_to_its_tool():
    cases = [
        ("What is the current status of work order WO-000123?", "get_work_order_status"),
        ("What priority level is assigned to work order WO-000123?", "get_work_order_priority"),
        ("Which technician is assigned to work order WO-000123?", "get_work_order_technician"),
        ("Which asset is associated with work order WO-000123?", "get_work_order_asset"),
        ("What is the operating status of asset AST-00045?", "get_asset_status"),
        ("Which site is asset AST-00045 located at?", "get_asset_site"),
        ("Which site is technician TCH-0012 based at?", "get_technician_site"),
        ("Is technician TCH-0012 currently active?", "get_technician_active"),
        (
            "How many open work orders does technician TCH-0012 currently have?",
            "list_open_work_orders_for_technician",
        ),
        ("How many units of part PRT-0007 are currently in stock?", "get_part_stock"),
        (
            "How many units of part PRT-0007 were used on work order WO-000123?",
            "get_part_usage_in_work_order",
        ),
        ("How many technicians are assigned to site STE-0001?", "get_site_technician_count"),
    ]
    for text, expected_tool in cases:
        route = route_question(text)
        assert route.tool == expected_tool, text


def test_router_extracts_correct_typed_arguments():
    route = route_question(
        "How many units of part PRT-0007 were used on work order WO-000123?"
    )
    assert route.args == {"work_order_id": "WO-000123", "part_id": "PRT-0007"}


def test_answer_question_end_to_end_answerable(seeded_session):
    session, graph = seeded_session
    wo_id = graph.work_order_ids[0]
    result = answer_question(session, f"What is the current status of work order {wo_id}?")
    assert result.refused is False
    assert wo_id in result.source_object_ids


def test_answer_question_end_to_end_refuses_on_missing_object(seeded_session):
    session, _graph = seeded_session
    result = answer_question(session, "What is the current status of work order WO-999999?")
    assert result.refused is True
    assert result.source_object_ids == []
