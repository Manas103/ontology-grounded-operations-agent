from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ontology_agent.models import Base
from ontology_agent import clinical_models  # noqa: F401
from ontology_agent.clinical_loader import list_patient_bundle_paths, load_directory


def test_list_patient_bundle_paths_excludes_metadata_files():
    paths = list_patient_bundle_paths("tests/fixtures/synthea_sample")
    assert len(paths) == 3
    assert all("hospitalInformation" not in p and "practitionerInformation" not in p for p in paths)


def test_load_directory_populates_all_six_resource_types_with_no_validation_failures():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    stats = load_directory(session, "tests/fixtures/synthea_sample")
    assert stats.bundles_loaded == 3
    for resource_type, count in stats.resource_counts.items():
        assert count > 0, f"{resource_type} had 0 rows loaded"
    assert stats.fhir_validation_failures == []


def test_load_directory_is_idempotent_on_patient_merge():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    load_directory(session, "tests/fixtures/synthea_sample")
    stats_second = load_directory(session, "tests/fixtures/synthea_sample")
    assert stats_second.bundles_loaded == 3
    from ontology_agent.clinical_models import ClinicalPatient

    assert session.query(ClinicalPatient).count() == 3
