from __future__ import annotations

from ontology_agent.models import Chamber, MaintenanceEvent, Recipe, Tool


def test_seed_creates_expected_object_counts(seeded_session):
    session, graph = seeded_session
    assert len(graph.tool_ids) == 40
    assert len(graph.chamber_ids) > 0
    assert len(graph.recipe_ids) > 0
    assert len(graph.part_ids) == 28
    assert len(graph.maintenance_event_ids) == 420


def test_foreign_keys_resolve_to_real_rows(seeded_session):
    session, graph = seeded_session
    event = session.get(MaintenanceEvent, graph.maintenance_event_ids[0])
    assert (event.tool_id is not None) != (event.chamber_id is not None)
    if event.tool_id is not None:
        assert session.get(Tool, event.tool_id) is not None
    else:
        chamber = session.get(Chamber, event.chamber_id)
        assert chamber is not None
        assert session.get(Tool, chamber.tool_id) is not None

    recipe = session.get(Recipe, graph.recipe_ids[0])
    chamber = session.get(Chamber, recipe.chamber_id)
    assert chamber is not None
    assert session.get(Tool, chamber.tool_id) is not None


def test_tool_status_is_within_declared_domain(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import TOOL_STATUSES

    for tool_id in graph.tool_ids:
        tool = session.get(Tool, tool_id)
        assert tool.status in TOOL_STATUSES


def test_maintenance_event_targets_exactly_one_of_tool_or_chamber(seeded_session):
    session, graph = seeded_session
    for event_id in graph.maintenance_event_ids:
        event = session.get(MaintenanceEvent, event_id)
        assert (event.tool_id is None) != (event.chamber_id is None)


def test_foreign_key_enforcement_rejects_dangling_reference(session_factory):
    from sqlalchemy.exc import IntegrityError

    with session_factory() as session:
        bogus = Chamber(
            id="CH-999999",
            tool_id="TL-DOES-NOT-EXIST",
            chamber_number=1,
            chamber_type="process",
            status="operational",
        )
        session.add(bogus)
        try:
            session.commit()
            raised = False
        except IntegrityError:
            session.rollback()
            raised = True
        assert raised, "SQLite foreign_keys pragma should reject a dangling tool_id"
