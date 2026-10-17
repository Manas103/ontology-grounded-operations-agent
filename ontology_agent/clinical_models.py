"""The clinical FHIR R4 object model.

Second domain layer added to this repository (the first was the fab
equipment model in `models.py`, built for the Tokyo Electron role). This
layer does not replace the equipment tables; it sits beside them, sharing
the same `Base`, the same `db.py` engine/session setup, and the same
architectural pattern (typed read tools over a real schema, JSON-Schema
-validated write proposals held for a human), applied to a clinical
domain instead of a fab. See README, "Two domains, one pattern," for why
this repository now hosts both instead of the clinical layer replacing
the equipment one the way the Tokyo Electron extension replaced the
original flat asset registry.

Six tables, one per FHIR R4 resource type this project actually loads
from real Synthea-generated bundles (Patient, Encounter, Condition,
Observation, MedicationRequest, Procedure). Every primary key is the real
FHIR resource id Synthea assigned, not a relabeled surrogate, so a
citation printed by a tool call is the actual resource id a clinician (or
a real FHIR server) would use to look the same resource up.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ontology_agent.models import Base


class ClinicalPatient(Base):
    __tablename__ = "clinical_patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    given_name: Mapped[str] = mapped_column(String(120), nullable=False)
    family_name: Mapped[str] = mapped_column(String(120), nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    birth_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    race: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ethnicity: Mapped[str | None] = mapped_column(String(80), nullable=True)

    encounters: Mapped[list["ClinicalEncounter"]] = relationship(back_populates="patient")
    conditions: Mapped[list["ClinicalCondition"]] = relationship(back_populates="patient")
    observations: Mapped[list["ClinicalObservation"]] = relationship(back_populates="patient")
    medication_requests: Mapped[list["ClinicalMedicationRequest"]] = relationship(
        back_populates="patient"
    )
    procedures: Mapped[list["ClinicalProcedure"]] = relationship(back_populates="patient")


class ClinicalEncounter(Base):
    __tablename__ = "clinical_encounters"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("clinical_patients.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    class_code: Mapped[str] = mapped_column(String(16), nullable=False)
    type_display: Mapped[str | None] = mapped_column(String(200), nullable=True)
    period_start: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    period_end: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    reason_display: Mapped[str | None] = mapped_column(String(200), nullable=True)

    patient: Mapped["ClinicalPatient"] = relationship(back_populates="encounters")
    conditions: Mapped[list["ClinicalCondition"]] = relationship(back_populates="encounter")
    observations: Mapped[list["ClinicalObservation"]] = relationship(back_populates="encounter")
    medication_requests: Mapped[list["ClinicalMedicationRequest"]] = relationship(
        back_populates="encounter"
    )
    procedures: Mapped[list["ClinicalProcedure"]] = relationship(back_populates="encounter")


class ClinicalCondition(Base):
    __tablename__ = "clinical_conditions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("clinical_patients.id"), nullable=False
    )
    encounter_id: Mapped[str | None] = mapped_column(
        ForeignKey("clinical_encounters.id"), nullable=True
    )
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    code_display: Mapped[str] = mapped_column(String(200), nullable=False)
    clinical_status: Mapped[str] = mapped_column(String(32), nullable=False)
    onset_date: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped["ClinicalPatient"] = relationship(back_populates="conditions")
    encounter: Mapped["ClinicalEncounter | None"] = relationship(back_populates="conditions")


class ClinicalObservation(Base):
    __tablename__ = "clinical_observations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("clinical_patients.id"), nullable=False
    )
    encounter_id: Mapped[str | None] = mapped_column(
        ForeignKey("clinical_encounters.id"), nullable=True
    )
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    code_display: Mapped[str] = mapped_column(String(200), nullable=False)
    value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    effective_date: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped["ClinicalPatient"] = relationship(back_populates="observations")
    encounter: Mapped["ClinicalEncounter | None"] = relationship(back_populates="observations")


class ClinicalMedicationRequest(Base):
    __tablename__ = "clinical_medication_requests"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("clinical_patients.id"), nullable=False
    )
    encounter_id: Mapped[str | None] = mapped_column(
        ForeignKey("clinical_encounters.id"), nullable=True
    )
    medication_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    medication_display: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    authored_on: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped["ClinicalPatient"] = relationship(back_populates="medication_requests")
    encounter: Mapped["ClinicalEncounter | None"] = relationship(
        back_populates="medication_requests"
    )


class ClinicalProcedure(Base):
    __tablename__ = "clinical_procedures"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("clinical_patients.id"), nullable=False
    )
    encounter_id: Mapped[str | None] = mapped_column(
        ForeignKey("clinical_encounters.id"), nullable=True
    )
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    code_display: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    performed_date: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped["ClinicalPatient"] = relationship(back_populates="procedures")
    encounter: Mapped["ClinicalEncounter | None"] = relationship(back_populates="procedures")
