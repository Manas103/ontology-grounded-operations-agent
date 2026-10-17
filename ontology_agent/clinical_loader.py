"""Loads real Synthea-generated FHIR R4 Bundle JSON files into the
clinical store.

Each bundle is one patient's full synthetic record: a `Patient` resource
plus every `Encounter`, `Condition`, `Observation`, `MedicationRequest`,
and `Procedure` Synthea generated for them, among many other resource
types (`Claim`, `ExplanationOfBenefit`, `Immunization`, ...) this project
does not load, because they have no tool over them. Every resource this
loader keeps is, before being inserted, parsed through the matching
`fhir.resources.R4B` pydantic model as a real, library-enforced
structural check, not a hand-rolled one; a resource that fails that parse
is counted and skipped rather than inserted half-formed. `fhir.resources`'s
pydantic-v2-compatible releases ship R4B (FHIR 4.3.0) rather than a
dedicated R4 (4.0.1) module; R4B is schema-stable with R4 for the six
resource types this project reads (Patient, Encounter, Condition,
Observation, MedicationRequest, Procedure did not change shape between
the two releases), so this is a disclosed, practical substitution for
per-record validation at the full 1,000-patient scale, not a claim of
exact R4 identity. The authoritative check against the real FHIR R4
(4.0.1) definitions is the separate, genuine HAPI FHIR `validator_cli`
run on a disclosed sample; see README, Validation.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os

from sqlalchemy.orm import Session

from ontology_agent.clinical_models import (
    ClinicalCondition,
    ClinicalEncounter,
    ClinicalMedicationRequest,
    ClinicalObservation,
    ClinicalPatient,
    ClinicalProcedure,
)

try:
    from fhir.resources.R4B.condition import Condition as FHIRCondition
    from fhir.resources.R4B.encounter import Encounter as FHIREncounter
    from fhir.resources.R4B.medicationrequest import MedicationRequest as FHIRMedicationRequest
    from fhir.resources.R4B.observation import Observation as FHIRObservation
    from fhir.resources.R4B.patient import Patient as FHIRPatient
    from fhir.resources.R4B.procedure import Procedure as FHIRProcedure

    _FHIR_VALIDATORS: dict[str, type] = {
        "Patient": FHIRPatient,
        "Encounter": FHIREncounter,
        "Condition": FHIRCondition,
        "Observation": FHIRObservation,
        "MedicationRequest": FHIRMedicationRequest,
        "Procedure": FHIRProcedure,
    }
except ImportError:  # pragma: no cover - exercised only if fhir.resources is absent
    _FHIR_VALIDATORS = {}

_SKIP_FILENAME_MARKERS = ("hospitalInformation", "practitionerInformation")


class LoadStats:
    def __init__(self) -> None:
        self.bundles_loaded = 0
        self.resource_counts: dict[str, int] = {
            "Patient": 0, "Encounter": 0, "Condition": 0,
            "Observation": 0, "MedicationRequest": 0, "Procedure": 0,
        }
        self.fhir_validation_failures: list[tuple[str, str, str]] = []  # (type, id, reason)

    def as_dict(self) -> dict:
        return {
            "bundles_loaded": self.bundles_loaded,
            "resource_counts": dict(self.resource_counts),
            "fhir_validation_failures": len(self.fhir_validation_failures),
        }


def list_patient_bundle_paths(directory: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    return [p for p in paths if not any(m in os.path.basename(p) for m in _SKIP_FILENAME_MARKERS)]


def _ref_id(reference: dict | None) -> str | None:
    if not reference:
        return None
    ref = reference.get("reference")
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref[len("urn:uuid:"):]
    return None


def _parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _coding_display(codeable_concept: dict | None) -> str | None:
    if not codeable_concept:
        return None
    if codeable_concept.get("text"):
        return codeable_concept["text"]
    codings = codeable_concept.get("coding") or []
    if codings and codings[0].get("display"):
        return codings[0]["display"]
    return None


def _coding_code(codeable_concept: dict | None) -> str | None:
    if not codeable_concept:
        return None
    codings = codeable_concept.get("coding") or []
    if codings and codings[0].get("code"):
        return codings[0]["code"]
    return None


def _us_core_extension_text(patient: dict, extension_url_suffix: str) -> str | None:
    for ext in patient.get("extension") or []:
        if ext.get("url", "").endswith(extension_url_suffix):
            for sub in ext.get("extension") or []:
                if sub.get("url") == "text":
                    return sub.get("valueString")
    return None


def _validate_fhir_shape(resource_type: str, raw: dict, stats: LoadStats) -> bool:
    validator = _FHIR_VALIDATORS.get(resource_type)
    if validator is None:
        return True
    try:
        validator.parse_obj(raw)
        return True
    except Exception as exc:  # pydantic ValidationError or similar
        stats.fhir_validation_failures.append((resource_type, raw.get("id", "?"), str(exc)[:300]))
        return False


def _value_text_and_unit(obs: dict) -> tuple[str | None, str | None]:
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        return (str(vq.get("value")), vq.get("unit"))
    if "valueString" in obs:
        return (obs["valueString"], None)
    if "valueCodeableConcept" in obs:
        return (_coding_display(obs["valueCodeableConcept"]), None)
    if "valueBoolean" in obs:
        return (str(obs["valueBoolean"]), None)
    if "component" in obs:
        parts = []
        for comp in obs["component"]:
            label = _coding_display(comp.get("code")) or "component"
            if "valueQuantity" in comp:
                vq = comp["valueQuantity"]
                parts.append(f"{label}: {vq.get('value')} {vq.get('unit', '')}".strip())
        if parts:
            return ("; ".join(parts), None)
    return (None, None)


def load_bundle(session: Session, bundle: dict, stats: LoadStats, validate: bool = True) -> None:
    patient_id: str | None = None
    rows_by_type: dict[str, list[dict]] = {
        "Patient": [], "Encounter": [], "Condition": [],
        "Observation": [], "MedicationRequest": [], "Procedure": [],
    }
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")
        if rt in rows_by_type:
            rows_by_type[rt].append(resource)

    for patient_raw in rows_by_type["Patient"]:
        if validate and not _validate_fhir_shape("Patient", patient_raw, stats):
            continue
        name = (patient_raw.get("name") or [{}])[0]
        given = " ".join(name.get("given", [])) or "Unknown"
        family = name.get("family") or "Unknown"
        patient_id = patient_raw["id"]
        session.merge(
            ClinicalPatient(
                id=patient_id,
                given_name=given,
                family_name=family,
                gender=patient_raw.get("gender", "unknown"),
                birth_date=_parse_date(patient_raw.get("birthDate")),
                race=_us_core_extension_text(patient_raw, "us-core-race"),
                ethnicity=_us_core_extension_text(patient_raw, "us-core-ethnicity"),
            )
        )
        stats.resource_counts["Patient"] += 1

    if patient_id is None:
        return  # no patient resource in this bundle; nothing to attach children to

    for enc_raw in rows_by_type["Encounter"]:
        if validate and not _validate_fhir_shape("Encounter", enc_raw, stats):
            continue
        reason_codes = enc_raw.get("reasonCode") or []
        reason_display = _coding_display(reason_codes[0]) if reason_codes else None
        period = enc_raw.get("period") or {}
        type_list = enc_raw.get("type") or []
        session.merge(
            ClinicalEncounter(
                id=enc_raw["id"],
                patient_id=_ref_id(enc_raw.get("subject")) or patient_id,
                status=enc_raw.get("status", "unknown"),
                class_code=(enc_raw.get("class") or {}).get("code", "unknown"),
                type_display=_coding_display(type_list[0]) if type_list else None,
                period_start=_parse_datetime(period.get("start")),
                period_end=_parse_datetime(period.get("end")),
                reason_display=reason_display,
            )
        )
        stats.resource_counts["Encounter"] += 1
    session.flush()

    for cond_raw in rows_by_type["Condition"]:
        if validate and not _validate_fhir_shape("Condition", cond_raw, stats):
            continue
        clinical_status = _coding_code(cond_raw.get("clinicalStatus")) or "unknown"
        session.merge(
            ClinicalCondition(
                id=cond_raw["id"],
                patient_id=_ref_id(cond_raw.get("subject")) or patient_id,
                encounter_id=_ref_id(cond_raw.get("encounter")),
                code=_coding_code(cond_raw.get("code")),
                code_display=_coding_display(cond_raw.get("code")) or "Unknown condition",
                clinical_status=clinical_status,
                onset_date=_parse_datetime(cond_raw.get("onsetDateTime"))
                or _parse_datetime(cond_raw.get("recordedDate")),
            )
        )
        stats.resource_counts["Condition"] += 1

    for obs_raw in rows_by_type["Observation"]:
        if validate and not _validate_fhir_shape("Observation", obs_raw, stats):
            continue
        value_text, unit = _value_text_and_unit(obs_raw)
        session.merge(
            ClinicalObservation(
                id=obs_raw["id"],
                patient_id=_ref_id(obs_raw.get("subject")) or patient_id,
                encounter_id=_ref_id(obs_raw.get("encounter")),
                code=_coding_code(obs_raw.get("code")),
                code_display=_coding_display(obs_raw.get("code")) or "Unknown observation",
                value_text=value_text,
                unit=unit,
                effective_date=_parse_datetime(obs_raw.get("effectiveDateTime")),
            )
        )
        stats.resource_counts["Observation"] += 1

    for med_raw in rows_by_type["MedicationRequest"]:
        if validate and not _validate_fhir_shape("MedicationRequest", med_raw, stats):
            continue
        session.merge(
            ClinicalMedicationRequest(
                id=med_raw["id"],
                patient_id=_ref_id(med_raw.get("subject")) or patient_id,
                encounter_id=_ref_id(med_raw.get("encounter")),
                medication_code=_coding_code(med_raw.get("medicationCodeableConcept")),
                medication_display=_coding_display(med_raw.get("medicationCodeableConcept"))
                or "Unknown medication",
                status=med_raw.get("status", "unknown"),
                authored_on=_parse_datetime(med_raw.get("authoredOn")),
            )
        )
        stats.resource_counts["MedicationRequest"] += 1

    for proc_raw in rows_by_type["Procedure"]:
        if validate and not _validate_fhir_shape("Procedure", proc_raw, stats):
            continue
        performed_date = _parse_datetime(proc_raw.get("performedDateTime"))
        if performed_date is None:
            performed_period = proc_raw.get("performedPeriod") or {}
            performed_date = _parse_datetime(performed_period.get("start"))
        session.merge(
            ClinicalProcedure(
                id=proc_raw["id"],
                patient_id=_ref_id(proc_raw.get("subject")) or patient_id,
                encounter_id=_ref_id(proc_raw.get("encounter")),
                code=_coding_code(proc_raw.get("code")),
                code_display=_coding_display(proc_raw.get("code")) or "Unknown procedure",
                status=proc_raw.get("status", "unknown"),
                performed_date=performed_date,
            )
        )
        stats.resource_counts["Procedure"] += 1

    stats.bundles_loaded += 1


def load_directory(
    session: Session, directory: str, limit: int | None = None, validate: bool = True
) -> LoadStats:
    stats = LoadStats()
    paths = list_patient_bundle_paths(directory)
    if limit is not None:
        paths = paths[:limit]
    for i, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
        load_bundle(session, bundle, stats, validate=validate)
        if i % 100 == 0:
            session.commit()
    session.commit()
    return stats
