"""Generates a large, varied batch of action proposals to measure the
validating layer against: some schema-valid, many deliberately malformed
or adversarial, standing in for what an unreliable tool-calling model
output (or a hostile input) can actually produce.

This module never talks to the object model or the approval queue; it only
produces raw dicts (and, for adversarial cases, non-dicts) the same shape
an LLM's tool-call output would arrive in. `scripts/run_action_validation_benchmark.py`
is what proves the validator rejects all of the invalid ones before anything
would reach `approval.ApprovalQueue`.
"""
from __future__ import annotations

import dataclasses
import random

from ontology_agent.seed import SeededObjectGraph

_SEVERITIES = ["low", "medium", "high", "critical"]
_MAINTENANCE_TYPES = ["preventive", "corrective", "calibration", "inspection"]
_RECIPE_PARAMETERS = ["temperature_c", "pressure_mtorr", "gas_flow_sccm", "rf_power_w"]


@dataclasses.dataclass
class GeneratedProposal:
    proposal_id: str
    raw_action: object
    ground_truth_valid: bool
    mutation: str  # "none" for valid proposals, else a short label


def _valid_schedule_maintenance(rng, graph):
    target_is_chamber = rng.random() > 0.5
    return {
        "action_type": "schedule_maintenance",
        "target_type": "chamber" if target_is_chamber else "tool",
        "target_id": rng.choice(graph.chamber_ids) if target_is_chamber else rng.choice(graph.tool_ids),
        "maintenance_type": rng.choice(_MAINTENANCE_TYPES),
        "notes": "Scheduled after trend review flagged early wear.",
    }


def _valid_retire_part(rng, graph):
    return {
        "action_type": "retire_part",
        "part_id": rng.choice(graph.part_ids),
        "reason": "Superseded by a newer revision from the supplier.",
    }


def _valid_update_recipe_parameters(rng, graph):
    return {
        "action_type": "update_recipe_parameters",
        "recipe_id": rng.choice(graph.recipe_ids),
        "parameter_name": rng.choice(_RECIPE_PARAMETERS),
        "new_value": round(rng.uniform(10.0, 900.0), 1),
        "reason": "Process engineering approved a parameter shift after a test run.",
    }


def _valid_flag_chamber_for_service(rng, graph):
    return {
        "action_type": "flag_chamber_for_service",
        "chamber_id": rng.choice(graph.chamber_ids),
        "severity": rng.choice(_SEVERITIES),
        "reason": "Particle count trending above baseline for three consecutive lots.",
    }


def _valid_record_part_usage_in_maintenance_event(rng, graph):
    return {
        "action_type": "record_part_usage_in_maintenance_event",
        "maintenance_event_id": rng.choice(graph.maintenance_event_ids),
        "part_id": rng.choice(graph.part_ids),
        "qty_used": rng.randint(1, 5),
    }


_VALID_BUILDERS = [
    _valid_schedule_maintenance,
    _valid_retire_part,
    _valid_update_recipe_parameters,
    _valid_flag_chamber_for_service,
    _valid_record_part_usage_in_maintenance_event,
]


def _mutate_missing_required_field(rng, payload):
    payload = dict(payload)
    keys = [k for k in payload if k != "action_type"]
    del payload[rng.choice(keys)]
    return payload, "missing_required_field"


def _mutate_wrong_type(rng, payload):
    payload = dict(payload)
    if "qty_used" in payload:
        payload["qty_used"] = "five"
        return payload, "wrong_type_qty_used_as_string"
    for k in payload:
        if k != "action_type" and isinstance(payload[k], str):
            payload[k] = ["not", "a", "string"]
            return payload, "wrong_type_field_as_list"
    payload["reason"] = 12345
    return payload, "wrong_type_reason_as_int"


def _mutate_unknown_action_type(rng, payload):
    payload = dict(payload)
    payload["action_type"] = "delete_all_maintenance_events"
    return payload, "unknown_action_type"


def _mutate_extra_field(rng, payload):
    payload = dict(payload)
    payload["override_approval"] = True
    return payload, "extra_disallowed_field"


def _mutate_invalid_enum(rng, payload):
    payload = dict(payload)
    if "severity" in payload:
        payload["severity"] = "catastrophic"
        return payload, "invalid_enum_severity"
    if "maintenance_type" in payload:
        payload["maintenance_type"] = "sabotage"
        return payload, "invalid_enum_maintenance_type"
    if "parameter_name" in payload:
        payload["parameter_name"] = "warp_factor"
        return payload, "invalid_enum_parameter_name"
    payload["action_type"] = "flag_chamber_for_service"
    payload["severity"] = "catastrophic"
    return payload, "invalid_enum_severity"


def _mutate_bad_id_pattern(rng, payload):
    payload = dict(payload)
    for k in list(payload):
        if k.endswith("_id") and k != "action_type":
            payload[k] = "not-a-real-id-42"
            return payload, "bad_id_pattern"
    payload["part_id"] = "PRT-1"
    return payload, "bad_id_pattern"


def _mutate_negative_qty(rng, payload):
    payload = dict(payload)
    payload["action_type"] = "record_part_usage_in_maintenance_event"
    payload.setdefault("maintenance_event_id", "ME-000001")
    payload.setdefault("part_id", "PRT-0001")
    payload["qty_used"] = -3
    return payload, "negative_quantity"


def _mutate_null_action_type(rng, payload):
    payload = dict(payload)
    payload["action_type"] = None
    return payload, "null_action_type"


def _mutate_not_a_dict(rng, payload):
    choice = rng.choice(["string", "list", "int", "none"])
    if choice == "string":
        return "retire_part PRT-0001", "not_a_dict_string"
    if choice == "list":
        return [payload], "not_a_dict_list"
    if choice == "int":
        return 42, "not_a_dict_int"
    return None, "not_a_dict_none"


def _mutate_empty_required_string(rng, payload):
    payload = dict(payload)
    for k in ("reason", "notes"):
        if k in payload:
            payload[k] = ""
            return payload, "empty_required_string"
    payload["action_type"] = "retire_part"
    payload.setdefault("part_id", "PRT-0001")
    payload["reason"] = ""
    return payload, "empty_required_string"


_MUTATIONS = [
    _mutate_missing_required_field,
    _mutate_wrong_type,
    _mutate_unknown_action_type,
    _mutate_extra_field,
    _mutate_invalid_enum,
    _mutate_bad_id_pattern,
    _mutate_negative_qty,
    _mutate_null_action_type,
    _mutate_not_a_dict,
    _mutate_empty_required_string,
]


def generate_action_proposals(
    graph: SeededObjectGraph, seed: int, n_valid: int = 300, n_malformed: int = 200
) -> list[GeneratedProposal]:
    rng = random.Random(seed + 3)
    proposals: list[GeneratedProposal] = []
    counter = 0

    for _ in range(n_valid):
        builder = rng.choice(_VALID_BUILDERS)
        payload = builder(rng, graph)
        counter += 1
        proposals.append(
            GeneratedProposal(
                proposal_id=f"PROP-{counter:05d}",
                raw_action=payload,
                ground_truth_valid=True,
                mutation="none",
            )
        )

    for i in range(n_malformed):
        builder = rng.choice(_VALID_BUILDERS)
        base_payload = builder(rng, graph)
        mutation_fn = _MUTATIONS[i % len(_MUTATIONS)]
        mutated, label = mutation_fn(rng, base_payload)
        counter += 1
        proposals.append(
            GeneratedProposal(
                proposal_id=f"PROP-{counter:05d}",
                raw_action=mutated,
                ground_truth_valid=False,
                mutation=label,
            )
        )

    rng.shuffle(proposals)
    return proposals
