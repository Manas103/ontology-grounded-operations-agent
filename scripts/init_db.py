"""Stand up the schema and load deterministic synthetic data.

Usage:
    python scripts/init_db.py            # sqlite, data/operations.db (default)
    OPERATIONS_AGENT_DATABASE_URL=postgresql://user:pass@localhost/opsdb \
        python scripts/init_db.py        # real PostgreSQL, if reachable
"""
from __future__ import annotations

import sys

from ontology_agent.db import init_db, make_engine, make_session_factory, resolve_database_url
from ontology_agent.seed import seed_database


def main() -> None:
    url = resolve_database_url()
    print(f"database url: {url}")
    engine = make_engine(url)
    init_db(engine, drop_first=True)
    Session = make_session_factory(engine)
    with Session() as session:
        graph = seed_database(session)
        print(f"seeded {len(graph.site_ids)} sites")
        print(f"seeded {len(graph.technician_ids)} technicians")
        print(f"seeded {len(graph.asset_ids)} assets")
        print(f"seeded {len(graph.part_ids)} parts")
        print(f"seeded {len(graph.work_order_ids)} work orders")
        print(f"seeded {len(graph.work_order_part_pairs)} work-order/part usage rows")


if __name__ == "__main__":
    sys.exit(main())
