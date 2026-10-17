"""Stands up the clinical tables and loads real Synthea FHIR R4 bundles.

    PYTHONPATH=. python scripts/init_clinical_db.py <path-to-synthea-fhir-output-dir>

The directory is Synthea's `output/fhir/` from a real run (see README,
"Building and running," for the exact `java -jar synthea-with-dependencies.jar`
command used to generate it). This script does not regenerate Synthea's
output itself; a 1,000-patient run takes a few minutes and several
gigabytes of JSON, both too large to shell out to from a short-lived
script invocation without a human watching, so the generation command is
documented, not wrapped.
"""
from __future__ import annotations

import sys

from ontology_agent.db import init_db, make_engine, make_session_factory, resolve_database_url
from ontology_agent.clinical_loader import load_directory


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-synthea-fhir-output-dir>")
        raise SystemExit(2)
    directory = sys.argv[1]

    engine = make_engine()
    init_db(engine)
    Session = make_session_factory(engine)
    session = Session()

    print(f"database: {resolve_database_url()}")
    print(f"loading Synthea FHIR bundles from: {directory}")
    stats = load_directory(session, directory)
    print(stats.as_dict())
    session.close()


if __name__ == "__main__":
    main()
