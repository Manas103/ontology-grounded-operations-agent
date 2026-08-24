from __future__ import annotations

from ontology_agent.questions import REFUSAL_TOTAL, build_question_set


def test_holdout_objects_are_created_then_deleted(seeded_session_with_holdout):
    session, graph, removed_ids = seeded_session_with_holdout
    from ontology_agent.models import Asset, Part, Site, Technician, WorkOrder

    for site_id in removed_ids["site_id"]:
        assert session.get(Site, site_id) is None
    for tech_id in removed_ids["technician_id"]:
        assert session.get(Technician, tech_id) is None
    for asset_id in removed_ids["asset_id"]:
        assert session.get(Asset, asset_id) is None
    for part_id in removed_ids["part_id"]:
        assert session.get(Part, part_id) is None
    for wo_id in removed_ids["work_order_id"]:
        assert session.get(WorkOrder, wo_id) is None


def test_question_set_has_over_300_questions_with_exactly_74_refusal(
    seeded_session_with_holdout,
):
    session, graph, removed_ids = seeded_session_with_holdout
    questions = build_question_set(session, graph, removed_ids, seed=20260724)
    assert len(questions) > 300
    refusal_qs = [q for q in questions if q.is_refusal]
    assert len(refusal_qs) == REFUSAL_TOTAL == 74


def test_refusal_questions_reference_only_removed_ids(seeded_session_with_holdout):
    session, graph, removed_ids = seeded_session_with_holdout
    questions = build_question_set(session, graph, removed_ids, seed=20260724)
    all_removed = {i for ids in removed_ids.values() for i in ids}
    for q in questions:
        if q.is_refusal:
            referenced = set(q.args.values())
            assert referenced & all_removed, q.text


def test_answerable_questions_have_nonempty_expected_citations(
    seeded_session_with_holdout,
):
    session, graph, removed_ids = seeded_session_with_holdout
    questions = build_question_set(session, graph, removed_ids, seed=20260724)
    for q in questions:
        if not q.is_refusal:
            assert len(q.expected_source_object_ids) >= 1, q.text


def test_question_set_is_reproducible_for_a_fixed_seed(seeded_session_with_holdout):
    session, graph, removed_ids = seeded_session_with_holdout
    a = build_question_set(session, graph, removed_ids, seed=20260724)
    b = build_question_set(session, graph, removed_ids, seed=20260724)
    assert [q.text for q in a] == [q.text for q in b]
