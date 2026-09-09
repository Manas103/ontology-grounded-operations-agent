from __future__ import annotations

from ontology_agent.qa_router import answer_question, route_question


def test_router_maps_each_marker_phrase_to_its_tool():
    cases = [
        ("What is the current status of tool TL-0001?", "get_tool_status"),
        ("What type of tool is TL-0001?", "get_tool_type"),
        ("What is the current status of chamber CH-00001?", "get_chamber_status"),
        ("Which tool does chamber CH-00001 belong to?", "get_chamber_tool"),
        ("Which chamber is recipe RC-00001 assigned to?", "get_recipe_chamber"),
        ("Is recipe RC-00001 currently active?", "get_recipe_active"),
        ("How many units of part PRT-0007 are currently in stock?", "get_part_stock"),
        (
            "What is the current status of maintenance event ME-000123?",
            "get_maintenance_event_status",
        ),
        (
            "Which tool or chamber was maintenance event ME-000123 performed on?",
            "get_maintenance_event_target",
        ),
        (
            "How many open maintenance events does chamber CH-00001 currently have?",
            "list_maintenance_events_for_chamber",
        ),
        (
            "How many units of part PRT-0007 were used during maintenance event ME-000123?",
            "get_part_usage_in_maintenance_event",
        ),
        ("How many chambers are installed on tool TL-0001?", "list_chambers_for_tool"),
    ]
    for text, expected_tool in cases:
        route = route_question(text)
        assert route.tool == expected_tool, text


def test_router_extracts_correct_typed_arguments():
    route = route_question(
        "How many units of part PRT-0007 were used during maintenance event ME-000123?"
    )
    assert route.args == {"maintenance_event_id": "ME-000123", "part_id": "PRT-0007"}


def test_answer_question_end_to_end_answerable(seeded_session):
    session, graph = seeded_session
    tool_id = graph.tool_ids[0]
    result = answer_question(session, f"What is the current status of tool {tool_id}?")
    assert result.refused is False
    assert tool_id in result.source_object_ids


def test_answer_question_end_to_end_refuses_on_missing_object(seeded_session):
    session, _graph = seeded_session
    result = answer_question(session, "What is the current status of tool TL-9999?")
    assert result.refused is True
    assert result.source_object_ids == []
