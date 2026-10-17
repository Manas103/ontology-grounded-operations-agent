from ontology_agent.clinical_actions import ClinicalActionValidationError, ClinicalActionValidator
from ontology_agent.clinical_proposal_generator import generate_clinical_proposals

PATIENT_IDS = ["e7cd7d56-ef89-5c07-c0b1-4c176b7e9d01", "2c7f5328-0b17-8a4a-0a5b-041a83ddf76a"]
ENCOUNTER_IDS = ["e7cd7d56-ef89-5c07-0761-c7460f079588", "e7cd7d56-ef89-5c07-3186-e41b4bf894f1"]


def test_default_counts_are_140_valid_and_100_malformed():
    proposals = generate_clinical_proposals(PATIENT_IDS, ENCOUNTER_IDS, seed=1)
    valid = [p for p in proposals if p.ground_truth_valid]
    malformed = [p for p in proposals if not p.ground_truth_valid]
    assert len(valid) == 140
    assert len(malformed) == 100
    assert len(proposals) == 240


def test_every_ground_truth_valid_proposal_actually_passes_the_validator():
    proposals = generate_clinical_proposals(PATIENT_IDS, ENCOUNTER_IDS, seed=2)
    validator = ClinicalActionValidator()
    for p in proposals:
        if p.ground_truth_valid:
            validator.validate(p.raw_action)  # must not raise


def test_every_ground_truth_malformed_proposal_is_genuinely_invalid():
    """Guards against a generator bug that produces a 'malformed' proposal
    the real validator would actually accept, mirroring
    test_action_generator.py's equivalent guard for the equipment domain."""
    proposals = generate_clinical_proposals(PATIENT_IDS, ENCOUNTER_IDS, seed=3)
    validator = ClinicalActionValidator()
    for p in proposals:
        if not p.ground_truth_valid:
            try:
                validator.validate(p.raw_action)
            except ClinicalActionValidationError:
                continue
            raise AssertionError(
                f"proposal {p.proposal_id} ({p.mutation}) was marked malformed "
                "but the real validator accepted it"
            )
