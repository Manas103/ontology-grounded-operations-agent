from __future__ import annotations

from ontology_agent.action_generator import generate_action_proposals
from ontology_agent.actions import ActionValidationError, ActionValidator
from ontology_agent.seed import DEFAULT_SEED


def test_ground_truth_valid_proposals_actually_pass_the_real_validator(seeded_session):
    _session, graph = seeded_session
    proposals = generate_action_proposals(graph, seed=DEFAULT_SEED, n_valid=60, n_malformed=0)
    validator = ActionValidator()
    for p in proposals:
        assert p.ground_truth_valid
        validator.validate(p.raw_action)  # must not raise


def test_ground_truth_malformed_proposals_are_all_genuinely_invalid(seeded_session):
    """Guards against a generator bug that accidentally produces a
    'malformed' proposal that is actually schema-valid, which would make
    the 0-schema-invalid-actions benchmark meaningless."""
    _session, graph = seeded_session
    proposals = generate_action_proposals(graph, seed=DEFAULT_SEED, n_valid=0, n_malformed=100)
    validator = ActionValidator()
    for p in proposals:
        assert not p.ground_truth_valid
        raised = False
        try:
            validator.validate(p.raw_action)
        except ActionValidationError:
            raised = True
        assert raised, f"mutation {p.mutation!r} produced an accidentally-valid action: {p.raw_action}"


def test_all_ten_mutation_functions_are_exercised(seeded_session):
    """10 mutation functions exist; some produce more than one distinct
    labelled outcome depending on the base payload they mutate, so the
    label set can be larger than 10 (asserted as >=) while every function
    is confirmed reachable (asserted by name prefix)."""
    _session, graph = seeded_session
    proposals = generate_action_proposals(graph, seed=DEFAULT_SEED, n_valid=0, n_malformed=200)
    seen = {p.mutation for p in proposals}
    assert len(seen) >= 10
    expected_prefixes = {
        "missing_required_field",
        "wrong_type",
        "unknown_action_type",
        "extra_disallowed_field",
        "invalid_enum",
        "bad_id_pattern",
        "negative_quantity",
        "null_action_type",
        "not_a_dict",
        "empty_required_string",
    }
    seen_prefixes = {next(p for p in expected_prefixes if label.startswith(p)) for label in seen}
    assert seen_prefixes == expected_prefixes
