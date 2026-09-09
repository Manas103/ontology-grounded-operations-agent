"""The typed object model.

Five real tables with real columns and foreign keys, not a generic
key-value blob. Everything the Q&A path and the action-proposal path
touch is defined here, once. `id` columns are human-legible, prefixed,
zero-padded strings (TL-0001, CH-00001, ...) so a citation printed in an
answer or an action proposal is unambiguous on its own.

This is the equipment domain model: a semiconductor-fab-style hierarchy
of tools (the physical machines), chambers (a tool's process modules),
recipes (the process programs a chamber runs), parts (consumables and
spares), and maintenance events (work performed on a tool or a chamber
that consumes parts). It replaces the sites/technicians/assets/parts/work
orders model this repository originally shipped with; see README,
"Extending the original operations agent," for what changed and why.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

TOOL_STATUSES = ("operational", "degraded", "down", "decommissioned")
CHAMBER_STATUSES = ("operational", "degraded", "down", "decommissioned")
CHAMBER_TYPES = ("process", "load_lock", "transfer")
MAINTENANCE_TYPES = ("preventive", "corrective", "calibration", "inspection")
MAINTENANCE_STATUSES = ("scheduled", "in_progress", "completed", "cancelled")


class Base(DeclarativeBase):
    pass


class Tool(Base):
    """A physical piece of fab equipment (an etch tool, a deposition tool,
    a CMP tool, ...). The top of the hierarchy: a tool has many chambers."""

    __tablename__ = "tools"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    tool_type: Mapped[str] = mapped_column(String(64), nullable=False)
    fab_bay: Mapped[str] = mapped_column(String(64), nullable=False)
    install_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        CheckConstraint(f"status in {TOOL_STATUSES!r}", name="ck_tool_status"),
    )

    chambers: Mapped[list["Chamber"]] = relationship(back_populates="tool")
    maintenance_events: Mapped[list["MaintenanceEvent"]] = relationship(
        back_populates="tool"
    )


class Chamber(Base):
    """A process module inside a tool. A chamber runs recipes and can
    itself be the target of a maintenance event (chamber-level service,
    as opposed to whole-tool service)."""

    __tablename__ = "chambers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tool_id: Mapped[str] = mapped_column(ForeignKey("tools.id"), nullable=False)
    chamber_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chamber_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        CheckConstraint(f"status in {CHAMBER_STATUSES!r}", name="ck_chamber_status"),
        CheckConstraint(f"chamber_type in {CHAMBER_TYPES!r}", name="ck_chamber_type"),
    )

    tool: Mapped["Tool"] = relationship(back_populates="chambers")
    recipes: Mapped[list["Recipe"]] = relationship(back_populates="chamber")
    maintenance_events: Mapped[list["MaintenanceEvent"]] = relationship(
        back_populates="chamber"
    )


class Recipe(Base):
    """A process program a chamber runs. References a primary consumable
    part when the process has one (an etch recipe's electrode, a
    deposition recipe's target material, and so on); nullable, because not
    every recipe has a distinguished consumable."""

    __tablename__ = "recipes"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    chamber_id: Mapped[str] = mapped_column(ForeignKey("chambers.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    process_step: Mapped[str] = mapped_column(String(64), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    primary_part_id: Mapped[str | None] = mapped_column(
        ForeignKey("parts.id"), nullable=True
    )

    chamber: Mapped["Chamber"] = relationship(back_populates="recipes")
    primary_part: Mapped["Part | None"] = relationship(back_populates="recipes_using")


class Part(Base):
    """A consumable or spare part: seals, liners, filters, sensors, and so
    on. Referenced by recipes (as a process consumable) and by maintenance
    events (as something a repair or PM used up)."""

    __tablename__ = "parts"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    sku: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    unit_cost: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    stock_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    is_consumable: Mapped[bool] = mapped_column(nullable=False, default=True)

    recipes_using: Mapped[list["Recipe"]] = relationship(back_populates="primary_part")
    usages: Mapped[list["MaintenanceEventPart"]] = relationship(back_populates="part")


class MaintenanceEvent(Base):
    """Work performed on a tool or on one of its chambers: a preventive
    maintenance visit, a corrective repair, a calibration, an inspection.
    Exactly one of `tool_id` (whole-tool service) or `chamber_id`
    (chamber-level service) is set, enforced by a CHECK constraint, not
    just by convention."""

    __tablename__ = "maintenance_events"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tool_id: Mapped[str | None] = mapped_column(ForeignKey("tools.id"), nullable=True)
    chamber_id: Mapped[str | None] = mapped_column(
        ForeignKey("chambers.id"), nullable=True
    )
    maintenance_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(
            f"maintenance_type in {MAINTENANCE_TYPES!r}", name="ck_me_type"
        ),
        CheckConstraint(f"status in {MAINTENANCE_STATUSES!r}", name="ck_me_status"),
        CheckConstraint(
            "(tool_id is not null and chamber_id is null) "
            "or (tool_id is null and chamber_id is not null)",
            name="ck_me_single_target",
        ),
    )

    tool: Mapped["Tool | None"] = relationship(back_populates="maintenance_events")
    chamber: Mapped["Chamber | None"] = relationship(
        back_populates="maintenance_events"
    )
    parts_used: Mapped[list["MaintenanceEventPart"]] = relationship(
        back_populates="maintenance_event"
    )


class MaintenanceEventPart(Base):
    """Join table: which parts, and how many, a maintenance event consumed."""

    __tablename__ = "maintenance_event_parts"

    maintenance_event_id: Mapped[str] = mapped_column(
        ForeignKey("maintenance_events.id"), primary_key=True
    )
    part_id: Mapped[str] = mapped_column(ForeignKey("parts.id"), primary_key=True)
    qty_used: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (CheckConstraint("qty_used > 0", name="ck_mep_qty_positive"),)

    maintenance_event: Mapped["MaintenanceEvent"] = relationship(
        back_populates="parts_used"
    )
    part: Mapped["Part"] = relationship(back_populates="usages")
