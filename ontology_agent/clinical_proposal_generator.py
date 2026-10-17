"""Generates write proposals to measure the write-confirmation boundary
against: a fixed batch of schema-valid order/note intents (the 140 the
resume's "0 of 140 write proposals executed without confirmation" claim
counts), plus a separately-reported batch of deliberately malformed or
adversarial ones standing in for an unreliable tool-calling model's
output, used only to show the schema gate in `clinical_actions.py` also
holds on this domain's two action types.

Mirrors `action_generator.py`'s discipline: this module only produces raw
dicts (and, for adversarial cases, non-dicts); it never touches the
clinical store or the confirmation queue itself.
"""
from __future__ import annotations

import dataclasses
import random

VALID_PROPOSAL_COUNT = 140

_ORDER_TYPES = ["medication", "laboratory", "imaging"]

_ORDER_DESCRIPTIONS = {
    "medication": [
        "Lisinopril 10mg oral tablet, once daily",
        "Atorvastatin 20mg oral tablet, once daily at bedtime",
        "Metformin 500mg oral tablet, twice daily with meals",
    ],
    "laboratory": [
        "Comprehensive metabolic panel",
        "Hemoglobin A1c",
        "Lipid panel, fasting",
    ],
    "imaging": [
        "Chest X-ray, 2 views",
        "Abdominal ultrasound",
        "MRI, lumbar spine without contrast",
    ],
}

_ORDER_REASONS = [
    "Follow-up on an abnormal result from the prior encounter.",
    "New symptom reported at today's visit warrants evaluation.",
    "Routine monitoring per the patient's chronic condition management plan.",
]

_NOTE_TEXTS = [
    "Patient reports symptoms improving since last visit; continuing current plan.",
    "Discussed medication adherence; patient agrees to follow-up in 4 weeks.",
    "Reviewed recent labs with patient; no changes to treatment plan at this time.",
]

_NOTE_REASONS = [
    "Documenting today's clinical discussion for the permanent record.",
    "Addendum requested after reviewing a delayed lab result.",
]


@dataclasses.dataclass
class GeneratedClinicalProposal:
    proposal_id: str
    raw_action: object
    ground_truth_valid: bool
    mutation: str  # "none" for valid proposals, else a short label


def _valid_place_order(rng, patient_ids):
    order_type = rng.choice(_ORDER_TYPES)
    return {
        "action_type": "place_order",
        "patient_id": rng.choice(patient_ids),
        "order_type": order_type,
        "description": rng.choice(_ORDER_DESCRIPTIONS[order_type]),
        "reason": rng.choice(_ORDER_REASONS),
    }


def _valid_edit_note(rng, encounter_ids):
    return {
        "action_type": "edit_note",
        "encounter_id": rng.choice(encounter_ids),
        "note_text": rng.choice(_NOTE_TEXTS),
        "reason": rng.choice(_NOTE_REASONS),
    }


def _mutate_missing_required_field(rng, payload):
    payload = dict(payload)
    keys = [k for k in payload if k != "action_type"]
    del payload[rng.choice(keys)]
    return payload, "missing_required_field"


def _mutate_wrong_type(rng, payload):
    payload = dict(payload)
    for k in payload:
        if k != "action_type" and isinstance(payload[k], str):
            payload[k] = ["not", "a", "string"]
            return payload, "wrong_type_field_as_list"
    payload["reason"] = 99999
    return payload, "wrong_type_reason_as_int"


def _mutate_unknown_action_type(rng, payload):
    payload = dict(payload)
    payload["action_type"] = "delete_patient_record"
    return payload, "unknown_action_type"


def _mutate_extra_field(rng, payload):
    payload = dict(payload)
    payload["skip_confirmation"] = True
    return payload, "extra_disallowed_field"


def _mutate_invalid_enum(rng, payload):
    payload = dict(payload)
    if "order_type" in payload:
        payload["order_type"] = "surgical"
        return payload, "invalid_enum_order_type"
    payload["action_type"] = "place_order"
    payload.setdefault("patient_id", "00000000-0000-0000-0000-000000000000")
    payload.setdefault("description", "x")
    payload.setdefault("reason", "x")
    payload["order_type"] = "surgical"
    return payload, "invalid_enum_order_type"


def _mutate_bad_id_pattern(rng, payload):
    payload = dict(payload)
    for k in list(payload):
        if k.endswith("_id") and k != "action_type":
            payload[k] = "not a real resource id"
            return payload, "bad_id_pattern"
    payload["patient_id"] = "not a real resource id"
    return payload, "bad_id_pattern"


def _mutate_null_action_type(rng, payload):
    payload = dict(payload)
    payload["action_type"] = None
    return payload, "null_action_type"


def _mutate_not_a_dict(rng, payload):
    choice = rng.choice(["string", "list", "int", "none"])
    if choice == "string":
        return "place_order for patient x", "not_a_dict_string"
    if choice == "list":
        return [payload], "not_a_dict_list"
    if choice == "int":
        return 7, "not_a_dict_int"
    return None, "not_a_dict_none"


def _mutate_empty_required_string(rng, payload):
    payload = dict(payload)
    for k in ("reason", "note_text", "description"):
        if k in payload:
            payload[k] = ""
            return payload, "empty_required_string"
    payload["reason"] = ""
    return payload, "empty_required_string"


_MUTATIONS = [
    _mutate_missing_required_field,
    _mutate_wrong_type,
    _mutate_unknown_action_type,
    _mutate_extra_field,
    _mutate_invalid_enum,
    _mutate_bad_id_pattern,
    _mutate_null_action_type,
    _mutate_not_a_dict,
    _mutate_empty_required_string,
]


def generate_clinical_proposals(
    patient_ids: list[str],
    encounter_ids: list[str],
    seed: int,
    n_valid: int = VALID_PROPOSAL_COUNT,
    n_malformed: int = 100,
) -> list[GeneratedClinicalProposal]:
    rng = random.Random(seed + 11)
    proposals: list[GeneratedClinicalProposal] = []
    counter = 0

    _valid_builders = [
        lambda r: _valid_place_order(r, patient_ids),
        lambda r: _valid_edit_note(r, encounter_ids),
    ]

    for _ in range(n_valid):
        builder = rng.choice(_valid_builders)
        payload = builder(rng)
        counter += 1
        proposals.append(
            GeneratedClinicalProposal(
                proposal_id=f"CPROP-{counter:05d}",
                raw_action=payload,
                ground_truth_valid=True,
                mutation="none",
            )
        )

    for i in range(n_malformed):
        builder = rng.choice(_valid_builders)
        base_payload = builder(rng)
        mutation_fn = _MUTATIONS[i % len(_MUTATIONS)]
        mutated, label = mutation_fn(rng, base_payload)
        counter += 1
        proposals.append(
            GeneratedClinicalProposal(
                proposal_id=f"CPROP-{counter:05d}",
                raw_action=mutated,
                ground_truth_valid=False,
                mutation=label,
            )
        )

    rng.shuffle(proposals)
    return proposals
