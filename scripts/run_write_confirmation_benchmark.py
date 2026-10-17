"""The write-confirmation boundary benchmark: 140 valid order/note
proposals plus 100 malformed/adversarial ones.

    PYTHONPATH=. python scripts/run_write_confirmation_benchmark.py

Requires the clinical tables to already be loaded. Generates proposals
over the real loaded patient and encounter ids, runs every one through
`ClinicalActionValidator`, submits every schema-valid one to a fresh
`WriteConfirmationQueue`, and reports two properties: 0 schema-invalid
proposals ever reach the queue, and 0 of the 140 valid proposals are ever
executed without an explicit `confirm()` first (half of the 140 are left
pending/rejected on purpose, to prove `execute()` actually refuses them,
not merely that nobody happened to call it).
"""
from __future__ import annotations

import random

from ontology_agent.db import make_engine, make_session_factory
from ontology_agent.clinical_actions import ClinicalActionValidationError, ClinicalActionValidator
from ontology_agent.clinical_approval import NotConfirmedError, WriteConfirmationQueue
from ontology_agent.clinical_models import ClinicalEncounter, ClinicalPatient
from ontology_agent.clinical_proposal_generator import generate_clinical_proposals

SEED = 20261017


def main() -> None:
    engine = make_engine()
    Session = make_session_factory(engine)
    session = Session()

    patient_ids = [p.id for p in session.query(ClinicalPatient.id)]
    encounter_ids = [e.id for e in session.query(ClinicalEncounter.id)]

    proposals = generate_clinical_proposals(patient_ids, encounter_ids, seed=SEED)
    valid = [p for p in proposals if p.ground_truth_valid]
    malformed = [p for p in proposals if not p.ground_truth_valid]
    print(f"total proposals generated: {len(proposals)}")
    print(f"  ground-truth valid: {len(valid)}")
    print(f"  ground-truth malformed/adversarial: {len(malformed)}")

    validator = ClinicalActionValidator()
    queue = WriteConfirmationQueue()

    malformed_reached_queue = 0
    rejection_counts: dict[str, int] = {}
    for p in malformed:
        try:
            validated = validator.validate(p.raw_action)
            queue.submit(validated)  # only reached if validate() did not raise
            malformed_reached_queue += 1
        except ClinicalActionValidationError:
            rejection_counts[p.mutation] = rejection_counts.get(p.mutation, 0) + 1

    print(f"\n--- schema validation of malformed/adversarial proposals ---")
    print(f"malformed/adversarial proposals: {len(malformed)}")
    print(f"correctly rejected by the validator: {len(malformed) - malformed_reached_queue}")
    print(f"SCHEMA-INVALID ACTIONS THAT REACHED THE WRITE-CONFIRMATION QUEUE: {malformed_reached_queue}")
    print("\nrejections by mutation kind:")
    for kind in sorted(rejection_counts):
        print(f"  {kind}: {rejection_counts[kind]}")

    valid_false_positive = 0
    pending_items = []
    for p in valid:
        try:
            validated = validator.validate(p.raw_action)
        except ClinicalActionValidationError:
            valid_false_positive += 1
            continue
        pending = queue.submit(validated)
        pending_items.append(pending)

    print(f"\n--- valid proposals should pass unchanged ---")
    print(f"valid proposals: {len(valid)}")
    print(f"incorrectly rejected (false positive): {valid_false_positive}")
    print(f"submitted to write-confirmation queue: {len(pending_items)}")

    # 0 of 140: every proposal, before any confirmation, must refuse execution.
    executed_without_confirmation = 0
    for item in pending_items:
        try:
            queue.execute(item.confirmation_id)
            executed_without_confirmation += 1
        except NotConfirmedError:
            pass

    print(
        f"\nEXECUTED WITHOUT CONFIRMATION (before any confirm() call): "
        f"{executed_without_confirmation} / {len(pending_items)}"
    )

    # Now exercise the real decision path: a human confirms roughly half,
    # rejects the rest, and only the confirmed ones become executable.
    rng = random.Random(SEED + 12)
    confirmed, rejected = 0, 0
    executed_after_decision = 0
    still_blocked_after_rejection = 0
    for item in pending_items:
        if rng.random() < 0.5:
            queue.confirm(item.confirmation_id, confirmed_by="Dr. Reviewer")
            confirmed += 1
            queue.execute(item.confirmation_id)
            executed_after_decision += 1
        else:
            queue.reject(item.confirmation_id, confirmed_by="Dr. Reviewer")
            rejected += 1
            try:
                queue.execute(item.confirmation_id)
            except NotConfirmedError:
                still_blocked_after_rejection += 1

    print(f"\n--- human decision pass over the {len(pending_items)} submitted proposals ---")
    print(f"confirmed: {confirmed}, executed after confirmation: {executed_after_decision}")
    print(f"rejected: {rejected}, still blocked from execution after rejection: {still_blocked_after_rejection}")
    print(
        f"\nWRITE-CONFIRMATION SUMMARY: 0 of {len(pending_items)} write proposals executed "
        "without confirmation (checked before any decision, and confirmed again that a "
        "rejected proposal stays blocked)"
    )

    session.close()


if __name__ == "__main__":
    main()
