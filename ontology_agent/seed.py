"""Deterministic synthetic data for a field-equipment-maintenance operation.

Everything here is synthetic: no real sites, technicians, or equipment.
The generator is seeded so the same seed always produces the same object
graph, which is what makes the evaluation harness in `questions.py`
reproducible run to run.
"""
from __future__ import annotations

import datetime as dt
import random

from sqlalchemy.orm import Session

from ontology_agent.models import (
    Asset,
    ASSET_STATUSES,
    Part,
    Site,
    Technician,
    WorkOrder,
    WORK_ORDER_PRIORITIES,
    WorkOrderPart,
    WORK_ORDER_STATUSES,
)

DEFAULT_SEED = 20260724

SITE_NAMES = [
    ("Riverbend Distribution Center", "Midwest", "America/Chicago"),
    ("Harborview Processing Plant", "Northeast", "America/New_York"),
    ("Desert Ridge Fulfillment Hub", "Southwest", "America/Phoenix"),
    ("Cascade Falls Cold Storage", "Northwest", "America/Los_Angeles"),
    ("Piedmont Assembly Facility", "Southeast", "America/New_York"),
    ("Prairie Junction Rail Yard", "Central", "America/Chicago"),
]

ASSET_TYPES = [
    ("HVAC Rooftop Unit", ["Carrier 48TC", "Trane RTU-950", "York YZ-800"]),
    ("Standby Generator", ["Cummins C900D6", "Generac SD500", "Kohler 800REOZK"]),
    ("Centrifugal Pump", ["Grundfos CR90", "Goulds 3196", "Flowserve 3410"]),
    ("Air Compressor", ["Atlas Copco GA90", "Ingersoll Rand R90", "Kaeser CSD125"]),
    ("Conveyor Drive", ["Dorner 2200", "Hytrol EZLogic", "Interroll RollerDrive"]),
    ("Industrial Boiler", ["Cleaver-Brooks CB700", "Fulton FB-A", "Miura EX-300"]),
    ("Loading Dock Leveler", ["Rite-Hite RHL", "Blue Giant AS", "Kelley KL"]),
    ("Cold Storage Chiller", ["York YK", "Trane CenTraVac", "Carrier AquaEdge"]),
]

CERT_LEVELS = ["apprentice", "journeyman", "senior", "master"]

PART_CATALOG = [
    ("Compressor Contactor", "CC"),
    ("HVAC Belt Kit", "HB"),
    ("Bearing Assembly", "BA"),
    ("Refrigerant R-410A (25lb)", "RF"),
    ("Control Board", "CB"),
    ("Motor Starter Relay", "MS"),
    ("Fuel Injector", "FI"),
    ("Hydraulic Seal Kit", "HS"),
    ("Air Filter Cartridge", "AF"),
    ("Coolant Pump", "CP"),
    ("Pressure Transducer", "PT"),
    ("Drive Belt", "DB"),
    ("Igniter Assembly", "IA"),
    ("Thermostat Module", "TM"),
    ("Gasket Set", "GS"),
    ("Voltage Regulator", "VR"),
    ("Oil Filter", "OF"),
    ("Solenoid Valve", "SV"),
    ("Wiring Harness", "WH"),
    ("Circuit Breaker", "CB2"),
    ("Fan Blade Assembly", "FB"),
    ("Level Sensor", "LS"),
    ("Expansion Valve", "EV"),
    ("Roller Chain", "RC"),
    ("Terminal Block", "TB"),
    ("Coupling Assembly", "CA"),
    ("Actuator Motor", "AM"),
    ("Safety Relay", "SR"),
]

WORK_ORDER_TEMPLATES = [
    "Scheduled preventive maintenance inspection.",
    "Unplanned failure reported by site operations.",
    "Vibration analysis flagged an anomaly during routine check.",
    "Follow-up repair after parts backorder resolved.",
    "Seasonal changeover service.",
    "Emergency callout after alarm triggered.",
    "Routine calibration and filter replacement.",
    "Corrosion and leak inspection.",
]


class SeededObjectGraph:
    """The full set of object ids created by `seed_database`, by type."""

    def __init__(self) -> None:
        self.site_ids: list[str] = []
        self.technician_ids: list[str] = []
        self.asset_ids: list[str] = []
        self.part_ids: list[str] = []
        self.work_order_ids: list[str] = []
        self.work_order_part_pairs: list[tuple[str, str]] = []


def seed_database(
    session: Session,
    seed: int = DEFAULT_SEED,
    n_technicians: int = 36,
    n_assets: int = 160,
    n_work_orders: int = 420,
) -> SeededObjectGraph:
    rng = random.Random(seed)
    graph = SeededObjectGraph()

    sites = []
    for i, (name, region, tz) in enumerate(SITE_NAMES, start=1):
        site = Site(id=f"STE-{i:04d}", name=name, region=region, timezone=tz)
        sites.append(site)
        graph.site_ids.append(site.id)
    session.add_all(sites)

    technicians = []
    for i in range(1, n_technicians + 1):
        site = rng.choice(sites)
        tech = Technician(
            id=f"TCH-{i:04d}",
            name=f"Technician {i:04d}",
            site_id=site.id,
            certification_level=rng.choice(CERT_LEVELS),
            active=rng.random() > 0.08,
        )
        technicians.append(tech)
        graph.technician_ids.append(tech.id)
    session.add_all(technicians)

    assets = []
    base_install = dt.date(2018, 1, 1)
    for i in range(1, n_assets + 1):
        site = rng.choice(sites)
        asset_type, models = rng.choice(ASSET_TYPES)
        install_offset = rng.randint(0, 365 * 7)
        asset = Asset(
            id=f"AST-{i:05d}",
            site_id=site.id,
            asset_type=asset_type,
            model=rng.choice(models),
            serial_number=f"SN{seed}{i:06d}",
            install_date=base_install + dt.timedelta(days=install_offset),
            status=rng.choices(
                ASSET_STATUSES, weights=[0.72, 0.16, 0.08, 0.04], k=1
            )[0],
        )
        assets.append(asset)
        graph.asset_ids.append(asset.id)
    session.add_all(assets)

    parts = []
    for i, (name, sku_prefix) in enumerate(PART_CATALOG, start=1):
        part = Part(
            id=f"PRT-{i:04d}",
            name=name,
            sku=f"{sku_prefix}-{1000 + i}",
            unit_cost=round(rng.uniform(8.5, 640.0), 2),
            stock_qty=rng.randint(0, 250),
        )
        parts.append(part)
        graph.part_ids.append(part.id)
    session.add_all(parts)

    session.flush()

    work_orders = []
    wo_parts = []
    base_open = dt.datetime(2026, 1, 5, 8, 0, 0)
    for i in range(1, n_work_orders + 1):
        asset = rng.choice(assets)
        status = rng.choices(
            WORK_ORDER_STATUSES, weights=[0.18, 0.14, 0.08, 0.60], k=1
        )[0]
        assign_tech = rng.random() > 0.10
        tech_id = rng.choice(technicians).id if assign_tech else None
        opened_at = base_open + dt.timedelta(
            hours=rng.randint(0, 24 * 200), minutes=rng.randint(0, 59)
        )
        closed_at = None
        if status == "closed":
            closed_at = opened_at + dt.timedelta(hours=rng.randint(1, 96))
        wo = WorkOrder(
            id=f"WO-{i:06d}",
            asset_id=asset.id,
            technician_id=tech_id,
            status=status,
            priority=rng.choices(
                WORK_ORDER_PRIORITIES, weights=[0.30, 0.40, 0.22, 0.08], k=1
            )[0],
            opened_at=opened_at,
            closed_at=closed_at,
            description=rng.choice(WORK_ORDER_TEMPLATES),
        )
        work_orders.append(wo)
        graph.work_order_ids.append(wo.id)

        if rng.random() > 0.45:
            n_parts_used = rng.randint(1, 3)
            chosen_parts = rng.sample(parts, k=min(n_parts_used, len(parts)))
            for part in chosen_parts:
                pair = (wo.id, part.id)
                if pair in graph.work_order_part_pairs:
                    continue
                wo_parts.append(
                    WorkOrderPart(
                        work_order_id=wo.id, part_id=part.id, qty_used=rng.randint(1, 6)
                    )
                )
                graph.work_order_part_pairs.append(pair)
    session.add_all(work_orders)
    session.add_all(wo_parts)
    session.commit()

    return graph
