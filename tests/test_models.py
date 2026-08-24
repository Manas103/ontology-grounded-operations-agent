from __future__ import annotations

from ontology_agent.models import Asset, Site, Technician, WorkOrder


def test_seed_creates_expected_object_counts(seeded_session):
    session, graph = seeded_session
    assert len(graph.site_ids) == 6
    assert len(graph.technician_ids) == 36
    assert len(graph.asset_ids) == 160
    assert len(graph.part_ids) == 28
    assert len(graph.work_order_ids) == 420


def test_foreign_keys_resolve_to_real_rows(seeded_session):
    session, graph = seeded_session
    wo = session.get(WorkOrder, graph.work_order_ids[0])
    assert session.get(Asset, wo.asset_id) is not None
    asset = session.get(Asset, wo.asset_id)
    assert session.get(Site, asset.site_id) is not None
    if wo.technician_id is not None:
        tech = session.get(Technician, wo.technician_id)
        assert tech is not None
        assert session.get(Site, tech.site_id) is not None


def test_asset_status_is_within_declared_domain(seeded_session):
    session, graph = seeded_session
    from ontology_agent.models import ASSET_STATUSES

    for asset_id in graph.asset_ids:
        asset = session.get(Asset, asset_id)
        assert asset.status in ASSET_STATUSES


def test_foreign_key_enforcement_rejects_dangling_reference(session_factory):
    from sqlalchemy.exc import IntegrityError

    with session_factory() as session:
        bogus = WorkOrder(
            id="WO-999999",
            asset_id="AST-DOES-NOT-EXIST",
            technician_id=None,
            status="open",
            priority="low",
            opened_at=__import__("datetime").datetime(2026, 1, 1),
            description="should fail",
        )
        session.add(bogus)
        try:
            session.commit()
            raised = False
        except IntegrityError:
            session.rollback()
            raised = True
        assert raised, "SQLite foreign_keys pragma should reject a dangling asset_id"
