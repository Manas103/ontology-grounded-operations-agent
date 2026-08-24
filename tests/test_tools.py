from __future__ import annotations

import pytest

from ontology_agent.tools import (
    ObjectNotFoundError,
    TOOL_CATALOG,
    get_asset_site,
    get_asset_status,
    get_part_stock,
    get_part_usage_in_work_order,
    get_site_technician_count,
    get_technician_active,
    get_technician_site,
    get_work_order_asset,
    get_work_order_priority,
    get_work_order_status,
    get_work_order_technician,
    list_open_work_orders_for_technician,
)


def test_every_tool_cites_its_source_object(seeded_session):
    session, graph = seeded_session
    wo_id = graph.work_order_ids[0]
    asset_id = graph.asset_ids[0]
    tech_id = graph.technician_ids[0]
    part_id = graph.part_ids[0]
    site_id = graph.site_ids[0]

    checks = [
        (get_work_order_status(session, work_order_id=wo_id), wo_id),
        (get_work_order_priority(session, work_order_id=wo_id), wo_id),
        (get_work_order_asset(session, work_order_id=wo_id), wo_id),
        (get_asset_status(session, asset_id=asset_id), asset_id),
        (get_technician_active(session, technician_id=tech_id), tech_id),
        (get_part_stock(session, part_id=part_id), part_id),
        (get_site_technician_count(session, site_id=site_id), site_id),
    ]
    for result, expected_id in checks:
        assert expected_id in result.source_object_ids
        assert len(result.source_object_ids) >= 1


def test_get_work_order_technician_cites_both_when_assigned(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import WorkOrder

    assigned = next(
        wo_id
        for wo_id in graph.work_order_ids
        if session.get(WorkOrder, wo_id).technician_id is not None
    )
    result = get_work_order_technician(session, work_order_id=assigned)
    wo = session.get(WorkOrder, assigned)
    assert result.source_object_ids == [wo.id, wo.technician_id]


def test_get_asset_site_cites_asset_and_site(seeded_session):
    session, graph = seeded_session
    result = get_asset_site(session, asset_id=graph.asset_ids[0])
    assert len(result.source_object_ids) == 2
    assert graph.asset_ids[0] in result.source_object_ids


@pytest.mark.parametrize(
    "tool_name, kwargs",
    [
        ("get_work_order_status", {"work_order_id": "WO-999999"}),
        ("get_work_order_priority", {"work_order_id": "WO-999999"}),
        ("get_work_order_technician", {"work_order_id": "WO-999999"}),
        ("get_work_order_asset", {"work_order_id": "WO-999999"}),
        ("get_asset_status", {"asset_id": "AST-99999"}),
        ("get_asset_site", {"asset_id": "AST-99999"}),
        ("get_technician_site", {"technician_id": "TCH-9999"}),
        ("get_technician_active", {"technician_id": "TCH-9999"}),
        ("list_open_work_orders_for_technician", {"technician_id": "TCH-9999"}),
        ("get_part_stock", {"part_id": "PRT-9999"}),
        ("get_site_technician_count", {"site_id": "STE-9999"}),
    ],
)
def test_every_tool_refuses_on_missing_object_rather_than_guessing(
    seeded_session, tool_name, kwargs
):
    session, _graph = seeded_session
    tool_fn = TOOL_CATALOG[tool_name]
    with pytest.raises(ObjectNotFoundError):
        tool_fn(session, **kwargs)


def test_part_usage_in_work_order_refuses_on_missing_part(seeded_session):
    session, graph = seeded_session
    with pytest.raises(ObjectNotFoundError):
        get_part_usage_in_work_order(
            session, work_order_id=graph.work_order_ids[0], part_id="PRT-9999"
        )


def test_part_usage_returns_zero_not_a_guess_when_never_used(seeded_session):
    """Distinguishes a real, present-but-zero fact from a missing object:
    zero usage is a legitimate answer with a citation, not a refusal."""
    session, graph = seeded_session
    from ontology_agent.models import WorkOrder, WorkOrderPart

    used_pairs = {(wop.work_order_id, wop.part_id) for wop in session.query(WorkOrderPart).all()}
    never_used_pair = next(
        (wo_id, part_id)
        for wo_id in graph.work_order_ids
        for part_id in graph.part_ids
        if (wo_id, part_id) not in used_pairs
    )
    result = get_part_usage_in_work_order(
        session, work_order_id=never_used_pair[0], part_id=never_used_pair[1]
    )
    assert result.value == 0
    assert set(result.source_object_ids) == set(never_used_pair)


def test_list_open_work_orders_cites_technician_and_every_open_work_order(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import Technician

    for tech_id in graph.technician_ids:
        tech = session.get(Technician, tech_id)
        result = list_open_work_orders_for_technician(session, technician_id=tech_id)
        expected_open = [
            wo.id for wo in tech.work_orders if wo.status in ("open", "in_progress", "on_hold")
        ]
        assert result.value == expected_open
        assert set(result.source_object_ids) == {tech_id, *expected_open}
