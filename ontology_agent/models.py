"""The typed object model.

Five real tables with real columns and foreign keys, not a generic
key-value blob. Everything the Q&A path and the action-proposal path
touch is defined here, once. `id` columns are human-legible, prefixed,
zero-padded strings (WO-000123, AST-000045, ...) so a citation printed
in an answer or an action proposal is unambiguous on its own.
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

ASSET_STATUSES = ("operational", "degraded", "down", "decommissioned")
WORK_ORDER_STATUSES = ("open", "in_progress", "on_hold", "closed")
WORK_ORDER_PRIORITIES = ("low", "medium", "high", "critical")


class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    timezone: Mapped[str] = mapped_column(String(32), nullable=False)

    technicians: Mapped[list["Technician"]] = relationship(back_populates="site")
    assets: Mapped[list["Asset"]] = relationship(back_populates="site")


class Technician(Base):
    __tablename__ = "technicians"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), nullable=False)
    certification_level: Mapped[str] = mapped_column(String(32), nullable=False)
    active: Mapped[bool] = mapped_column(nullable=False, default=True)

    site: Mapped["Site"] = relationship(back_populates="technicians")
    work_orders: Mapped[list["WorkOrder"]] = relationship(back_populates="technician")


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    serial_number: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    install_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        CheckConstraint(f"status in {ASSET_STATUSES!r}", name="ck_asset_status"),
    )

    site: Mapped["Site"] = relationship(back_populates="assets")
    work_orders: Mapped[list["WorkOrder"]] = relationship(back_populates="asset")


class Part(Base):
    __tablename__ = "parts"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    sku: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    unit_cost: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    stock_qty: Mapped[int] = mapped_column(Integer, nullable=False)

    usages: Mapped[list["WorkOrderPart"]] = relationship(back_populates="part")


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), nullable=False)
    technician_id: Mapped[str | None] = mapped_column(
        ForeignKey("technicians.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(f"status in {WORK_ORDER_STATUSES!r}", name="ck_wo_status"),
        CheckConstraint(f"priority in {WORK_ORDER_PRIORITIES!r}", name="ck_wo_priority"),
    )

    asset: Mapped["Asset"] = relationship(back_populates="work_orders")
    technician: Mapped["Technician | None"] = relationship(back_populates="work_orders")
    parts_used: Mapped[list["WorkOrderPart"]] = relationship(back_populates="work_order")


class WorkOrderPart(Base):
    """Join table: which parts, and how many, were used on which work order."""

    __tablename__ = "work_order_parts"

    work_order_id: Mapped[str] = mapped_column(
        ForeignKey("work_orders.id"), primary_key=True
    )
    part_id: Mapped[str] = mapped_column(ForeignKey("parts.id"), primary_key=True)
    qty_used: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (CheckConstraint("qty_used > 0", name="ck_wop_qty_positive"),)

    work_order: Mapped["WorkOrder"] = relationship(back_populates="parts_used")
    part: Mapped["Part"] = relationship(back_populates="usages")
