from __future__ import annotations

import pytest

from ontology_agent.tools import (
    ObjectNotFoundError,
    TOOL_CATALOG,
    get_chamber_status,
    get_chamber_tool,
    get_maintenance_event_target,
    get_part_stock,
    get_part_usage_in_maintenance_event,
    get_recipe_active,
    get_recipe_chamber,
    get_tool_status,
    get_tool_type,
    list_chambers_for_tool,
    list_maintenance_events_for_chamber,
)


def test_every_tool_cites_its_source_object(seeded_session):
    session, graph = seeded_session
    tool_id = graph.tool_ids[0]
    chamber_id = graph.chamber_ids[0]
    recipe_id = graph.recipe_ids[0]
    part_id = graph.part_ids[0]
    event_id = graph.maintenance_event_ids[0]

    checks = [
        (get_tool_status(session, tool_id=tool_id), tool_id),
        (get_tool_type(session, tool_id=tool_id), tool_id),
        (get_chamber_status(session, chamber_id=chamber_id), chamber_id),
        (get_recipe_active(session, recipe_id=recipe_id), recipe_id),
        (get_part_stock(session, part_id=part_id), part_id),
        (get_maintenance_event_target(session, maintenance_event_id=event_id), event_id),
    ]
    for result, expected_id in checks:
        assert expected_id in result.source_object_ids
        assert len(result.source_object_ids) >= 1


def test_get_chamber_tool_cites_chamber_and_tool(seeded_session):
    session, graph = seeded_session
    result = get_chamber_tool(session, chamber_id=graph.chamber_ids[0])
    assert len(result.source_object_ids) == 2
    assert graph.chamber_ids[0] in result.source_object_ids


def test_get_recipe_chamber_cites_recipe_and_chamber(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import Recipe

    recipe = session.get(Recipe, graph.recipe_ids[0])
    result = get_recipe_chamber(session, recipe_id=recipe.id)
    assert result.source_object_ids == [recipe.id, recipe.chamber_id]


def test_get_maintenance_event_target_cites_event_and_actual_target(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import MaintenanceEvent

    event = session.get(MaintenanceEvent, graph.maintenance_event_ids[0])
    expected_target = event.tool_id if event.tool_id is not None else event.chamber_id
    result = get_maintenance_event_target(session, maintenance_event_id=event.id)
    assert result.source_object_ids == [event.id, expected_target]


@pytest.mark.parametrize(
    "tool_name, kwargs",
    [
        ("get_tool_status", {"tool_id": "TL-9999"}),
        ("get_tool_type", {"tool_id": "TL-9999"}),
        ("get_chamber_status", {"chamber_id": "CH-99999"}),
        ("get_chamber_tool", {"chamber_id": "CH-99999"}),
        ("get_recipe_chamber", {"recipe_id": "RC-99999"}),
        ("get_recipe_active", {"recipe_id": "RC-99999"}),
        ("get_part_stock", {"part_id": "PRT-9999"}),
        ("get_maintenance_event_status", {"maintenance_event_id": "ME-999999"}),
        ("get_maintenance_event_target", {"maintenance_event_id": "ME-999999"}),
        ("list_maintenance_events_for_chamber", {"chamber_id": "CH-99999"}),
        ("list_chambers_for_tool", {"tool_id": "TL-9999"}),
    ],
)
def test_every_tool_refuses_on_missing_object_rather_than_guessing(
    seeded_session, tool_name, kwargs
):
    session, _graph = seeded_session
    tool_fn = TOOL_CATALOG[tool_name]
    with pytest.raises(ObjectNotFoundError):
        tool_fn(session, **kwargs)


def test_part_usage_in_maintenance_event_refuses_on_missing_part(seeded_session):
    session, graph = seeded_session
    with pytest.raises(ObjectNotFoundError):
        get_part_usage_in_maintenance_event(
            session, maintenance_event_id=graph.maintenance_event_ids[0], part_id="PRT-9999"
        )


def test_part_usage_returns_zero_not_a_guess_when_never_used(seeded_session):
    """Distinguishes a real, present-but-zero fact from a missing object:
    zero usage is a legitimate answer with a citation, not a refusal."""
    session, graph = seeded_session
    from ontology_agent.models import MaintenanceEventPart

    used_pairs = {
        (mep.maintenance_event_id, mep.part_id)
        for mep in session.query(MaintenanceEventPart).all()
    }
    never_used_pair = next(
        (event_id, part_id)
        for event_id in graph.maintenance_event_ids
        for part_id in graph.part_ids
        if (event_id, part_id) not in used_pairs
    )
    result = get_part_usage_in_maintenance_event(
        session, maintenance_event_id=never_used_pair[0], part_id=never_used_pair[1]
    )
    assert result.value == 0
    assert set(result.source_object_ids) == set(never_used_pair)


def test_list_chambers_for_tool_cites_tool_and_every_chamber(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import Tool

    for tool_id in graph.tool_ids[:5]:
        tool = session.get(Tool, tool_id)
        result = list_chambers_for_tool(session, tool_id=tool_id)
        expected = [c.id for c in tool.chambers]
        assert result.value == expected
        assert set(result.source_object_ids) == {tool_id, *expected}


def test_list_maintenance_events_for_chamber_cites_only_open_events(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import Chamber

    for chamber_id in graph.chamber_ids[:5]:
        chamber = session.get(Chamber, chamber_id)
        result = list_maintenance_events_for_chamber(session, chamber_id=chamber_id)
        expected_open = [
            e.id for e in chamber.maintenance_events if e.status in ("scheduled", "in_progress")
        ]
        assert result.value == expected_open
        assert set(result.source_object_ids) == {chamber_id, *expected_open}
