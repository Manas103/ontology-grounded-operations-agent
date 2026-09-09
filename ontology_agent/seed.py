"""Deterministic synthetic data for a fab equipment maintenance operation.

Everything here is synthetic: no real tools, chambers, or recipes. The
generator is seeded so the same seed always produces the same object
graph, which is what makes the evaluation harness in `questions.py`
reproducible run to run.
"""
from __future__ import annotations

import datetime as dt
import random

from sqlalchemy.orm import Session

from ontology_agent.models import (
    Chamber,
    CHAMBER_TYPES,
    MaintenanceEvent,
    MAINTENANCE_STATUSES,
    MAINTENANCE_TYPES,
    MaintenanceEventPart,
    Part,
    Recipe,
    Tool,
    TOOL_STATUSES,
)

DEFAULT_SEED = 20260724

TOOL_TYPES = [
    ("Plasma Etch", ["Lam Kiyo45", "AMAT Centura AdvantEdge", "Tokyo Electron Tactras"]),
    ("PECVD Deposition", ["AMAT Producer XP", "Novellus Vector Express", "TEL Triase"]),
    ("Chemical-Mechanical Polish", ["AMAT Reflexion LK", "Ebara F-REX300"]),
    ("Ion Implant", ["Applied Varian VIISta", "Axcelis Purion"]),
    ("Rapid Thermal Anneal", ["Mattson Helios", "AMAT Radiance"]),
    ("Wet Clean", ["TEL Cellesta", "SEMES Sepion"]),
    ("Physical Vapor Deposition", ["AMAT Endura", "TEL Eclipse"]),
    ("Diffusion Furnace", ["TEL Alpha-8SE", "Kokusai Vertical Furnace"]),
]

FAB_BAYS = ["Bay 1", "Bay 2", "Bay 3", "Bay 4", "Bay 5", "Bay 6"]

PROCESS_STEPS = [
    "gate etch", "spacer etch", "contact etch", "barrier deposition",
    "seed layer deposition", "bulk fill", "planarization", "implant anneal",
    "pre-clean", "post-etch clean", "diffusion drive-in", "metal liner deposition",
]

PART_CATALOG = [
    ("O-ring Seal Kit", "OR"),
    ("RF Match Network Module", "RF"),
    ("Shower Head Assembly", "SH"),
    ("Susceptor", "SU"),
    ("Throttle Valve Actuator", "TV"),
    ("Turbo Pump Cartridge", "TP"),
    ("Chamber Liner", "CL"),
    ("Edge Ring", "ER"),
    ("Gas Injector Nozzle", "GI"),
    ("Heater Coil", "HC"),
    ("Pressure Gauge", "PG"),
    ("Leveling Sensor", "LS"),
    ("Slit Valve Door", "SV"),
    ("Wafer Lift Pin Set", "LP"),
    ("Vacuum Seal O-Ring", "VS"),
    ("Plasma Source Coil", "PS"),
    ("Mass Flow Controller", "MF"),
    ("Ion Gauge", "IG"),
    ("Cryopump Cold Head", "CC"),
    ("Robot End Effector", "EE"),
    ("Load Lock Door Seal", "LL"),
    ("Chiller Pump", "CP"),
    ("RF Generator Module", "RG"),
    ("Match Box Capacitor", "MC"),
    ("Endpoint Detector Window", "EW"),
    ("Particle Filter", "PF"),
    ("Exhaust Valve", "EV"),
    ("Temperature Controller Board", "TC"),
]

MAINTENANCE_TEMPLATES = [
    "Scheduled preventive maintenance inspection.",
    "Unplanned failure reported by fab operations.",
    "Particle excursion flagged an anomaly during routine check.",
    "Follow-up repair after parts backorder resolved.",
    "Quarterly requalification service.",
    "Emergency callout after chamber pressure alarm triggered.",
    "Routine calibration and consumable replacement.",
    "Leak check and vacuum integrity inspection.",
]


class SeededObjectGraph:
    """The full set of object ids created by `seed_database`, by type."""

    def __init__(self) -> None:
        self.tool_ids: list[str] = []
        self.chamber_ids: list[str] = []
        self.recipe_ids: list[str] = []
        self.part_ids: list[str] = []
        self.maintenance_event_ids: list[str] = []
        self.maintenance_event_part_pairs: list[tuple[str, str]] = []


def seed_database(
    session: Session,
    seed: int = DEFAULT_SEED,
    n_tools: int = 40,
    n_maintenance_events: int = 420,
) -> SeededObjectGraph:
    rng = random.Random(seed)
    graph = SeededObjectGraph()

    base_install = dt.date(2016, 1, 1)
    tools = []
    for i in range(1, n_tools + 1):
        tool_type, models = rng.choice(TOOL_TYPES)
        install_offset = rng.randint(0, 365 * 9)
        tool = Tool(
            id=f"TL-{i:04d}",
            name=f"{tool_type} Tool {i:04d}",
            tool_type=tool_type,
            fab_bay=rng.choice(FAB_BAYS),
            install_date=base_install + dt.timedelta(days=install_offset),
            status=rng.choices(TOOL_STATUSES, weights=[0.72, 0.16, 0.08, 0.04], k=1)[0],
        )
        tools.append(tool)
        graph.tool_ids.append(tool.id)
    session.add_all(tools)
    session.flush()

    chambers = []
    chamber_counter = 0
    for tool in tools:
        n_chambers = rng.randint(2, 5)
        for _ in range(n_chambers):
            chamber_counter += 1
            chamber = Chamber(
                id=f"CH-{chamber_counter:05d}",
                tool_id=tool.id,
                chamber_number=len([c for c in chambers if c.tool_id == tool.id]) + 1,
                chamber_type=rng.choices(
                    CHAMBER_TYPES, weights=[0.6, 0.25, 0.15], k=1
                )[0],
                status=rng.choices(
                    ("operational", "degraded", "down", "decommissioned"),
                    weights=[0.75, 0.15, 0.07, 0.03], k=1,
                )[0],
            )
            chambers.append(chamber)
            graph.chamber_ids.append(chamber.id)
    session.add_all(chambers)
    session.flush()

    parts = []
    for i, (name, sku_prefix) in enumerate(PART_CATALOG, start=1):
        part = Part(
            id=f"PRT-{i:04d}",
            name=name,
            sku=f"{sku_prefix}-{1000 + i}",
            unit_cost=round(rng.uniform(12.0, 980.0), 2),
            stock_qty=rng.randint(0, 250),
            is_consumable=rng.random() > 0.15,
        )
        parts.append(part)
        graph.part_ids.append(part.id)
    session.add_all(parts)
    session.flush()

    recipes = []
    recipe_counter = 0
    for chamber in chambers:
        n_recipes = rng.randint(1, 3)
        for _ in range(n_recipes):
            recipe_counter += 1
            recipe = Recipe(
                id=f"RC-{recipe_counter:05d}",
                chamber_id=chamber.id,
                name=f"Recipe {recipe_counter:05d}",
                process_step=rng.choice(PROCESS_STEPS),
                revision=rng.randint(1, 12),
                is_active=rng.random() > 0.12,
                primary_part_id=rng.choice(parts).id if rng.random() > 0.35 else None,
            )
            recipes.append(recipe)
            graph.recipe_ids.append(recipe.id)
    session.add_all(recipes)
    session.flush()

    maintenance_events = []
    me_parts = []
    base_open = dt.datetime(2026, 1, 5, 8, 0, 0)
    for i in range(1, n_maintenance_events + 1):
        target_is_chamber = rng.random() > 0.30
        tool_id = None
        chamber_id = None
        if target_is_chamber:
            chamber_id = rng.choice(chambers).id
        else:
            tool_id = rng.choice(tools).id
        status = rng.choices(
            MAINTENANCE_STATUSES, weights=[0.18, 0.14, 0.60, 0.08], k=1
        )[0]
        opened_at = base_open + dt.timedelta(
            hours=rng.randint(0, 24 * 200), minutes=rng.randint(0, 59)
        )
        closed_at = None
        if status == "completed":
            closed_at = opened_at + dt.timedelta(hours=rng.randint(1, 96))
        event = MaintenanceEvent(
            id=f"ME-{i:06d}",
            tool_id=tool_id,
            chamber_id=chamber_id,
            maintenance_type=rng.choices(
                MAINTENANCE_TYPES, weights=[0.40, 0.30, 0.20, 0.10], k=1
            )[0],
            status=status,
            opened_at=opened_at,
            closed_at=closed_at,
            description=rng.choice(MAINTENANCE_TEMPLATES),
        )
        maintenance_events.append(event)
        graph.maintenance_event_ids.append(event.id)

        if rng.random() > 0.45:
            n_parts_used = rng.randint(1, 3)
            chosen_parts = rng.sample(parts, k=min(n_parts_used, len(parts)))
            for part in chosen_parts:
                pair = (event.id, part.id)
                if pair in graph.maintenance_event_part_pairs:
                    continue
                me_parts.append(
                    MaintenanceEventPart(
                        maintenance_event_id=event.id, part_id=part.id,
                        qty_used=rng.randint(1, 6),
                    )
                )
                graph.maintenance_event_part_pairs.append(pair)
    session.add_all(maintenance_events)
    session.add_all(me_parts)
    session.commit()

    return graph
