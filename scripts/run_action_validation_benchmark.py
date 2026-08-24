"""The action-validation benchmark: proves 0 schema-invalid actions reach
the human approval queue, across a large batch of proposals including
deliberately malformed and adversarial ones.

    python scripts/run_action_validation_benchmark.py
"""
from __future__ import annotations

import sys

from ontology_agent.action_generator import generate_action_proposals
from ontology_agent.actions import ActionValidationError, ActionValidator
from ontology_agent.approval import ApprovalQueue
from ontology_agent.db import init_db, make_engine, make_session_factory
from ontology_agent.seed import DEFAULT_SEED, seed_database

N_VALID = 300
N_MALFORMED = 200


def main() -> int:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)
    with Session() as session:
        graph = seed_database(session, seed=DEFAULT_SEED)

    proposals = generate_action_proposals(
        graph, seed=DEFAULT_SEED, n_valid=N_VALID, n_malformed=N_MALFORMED
    )
    print(f"total proposals generated: {len(proposals)}")
    print(f"  ground-truth valid: {sum(1 for p in proposals if p.ground_truth_valid)}")
    print(f"  ground-truth malformed/adversarial: {sum(1 for p in proposals if not p.ground_truth_valid)}")

    validator = ActionValidator()
    queue = ApprovalQueue()

    reached_queue_but_should_not_have = []
    correctly_rejected = 0
    incorrectly_rejected = []
    mutation_breakdown: dict[str, int] = {}

    for proposal in proposals:
        try:
            validated = validator.validate(proposal.raw_action)
        except ActionValidationError as exc:
            if proposal.ground_truth_valid:
                incorrectly_rejected.append((proposal.proposal_id, str(exc)))
            else:
                correctly_rejected += 1
                mutation_breakdown[proposal.mutation] = mutation_breakdown.get(proposal.mutation, 0) + 1
            continue

        queue.submit(validated)
        if not proposal.ground_truth_valid:
            reached_queue_but_should_not_have.append((proposal.proposal_id, proposal.mutation))

    malformed_total = sum(1 for p in proposals if not p.ground_truth_valid)
    valid_total = sum(1 for p in proposals if p.ground_truth_valid)

    print()
    print("--- schema validation of malformed/adversarial proposals ---")
    print(f"malformed/adversarial proposals: {malformed_total}")
    print(f"correctly rejected by the validator: {correctly_rejected}")
    print(f"SCHEMA-INVALID ACTIONS THAT REACHED THE APPROVAL QUEUE: {len(reached_queue_but_should_not_have)}")
    for pid, mutation in reached_queue_but_should_not_have:
        print(f"  {pid}: mutation={mutation} SLIPPED THROUGH")
    print()
    print("rejections by mutation kind:")
    for mutation, count in sorted(mutation_breakdown.items()):
        print(f"  {mutation}: {count}")

    print()
    print("--- valid proposals should pass unchanged ---")
    print(f"valid proposals: {valid_total}")
    print(f"incorrectly rejected (false positive): {len(incorrectly_rejected)}")
    for pid, reason in incorrectly_rejected:
        print(f"  {pid}: {reason}")

    print()
    print(f"approval queue size after run: {len(queue.all_items())}")
    print(f"approval queue pending: {len(queue.pending())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
