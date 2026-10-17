from ontology_agent.clinical_scope_guard import KNOWN_IDENTIFIERS, ScopeGuard
from ontology_agent.clinical_scope_requests import (
    OUT_OF_SCOPE_REQUEST_COUNT,
    build_out_of_scope_requests,
)


def test_known_read_tool_is_in_scope():
    result = ScopeGuard().check("get_patient_demographics")
    assert result.in_scope
    assert result.kind == "read_tool"


def test_known_write_action_is_in_scope():
    result = ScopeGuard().check("place_order")
    assert result.in_scope
    assert result.kind == "write_action"


def test_unknown_name_is_refused_with_a_named_reason():
    result = ScopeGuard().check("delete_patient_record")
    assert not result.in_scope
    assert "delete_patient_record" in result.refusal_reason


def test_out_of_scope_request_set_is_exactly_120_and_disjoint_from_the_real_catalog():
    requests = build_out_of_scope_requests()
    assert len(requests) == OUT_OF_SCOPE_REQUEST_COUNT == 120
    assert not (set(requests) & KNOWN_IDENTIFIERS)


def test_all_120_out_of_scope_requests_are_refused():
    guard = ScopeGuard()
    requests = build_out_of_scope_requests()
    results = [guard.check(name) for name in requests]
    assert all(not r.in_scope for r in results)
    assert sum(1 for r in results if not r.in_scope) == 120
