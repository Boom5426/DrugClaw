from __future__ import annotations

from urllib.error import URLError

from skills.drug_labeling.dailymed.dailymed_skill import DailyMedSkill
from skills.drug_labeling.medlineplus.medlineplus_skill import MedlinePlusSkill
from skills.drug_labeling.openfda.openfda_skill import OpenFDASkill


def _raise_url_error(*args, **kwargs):
    raise URLError("[Errno 111] Connection refused")


def test_dailymed_returns_offline_fixture_for_metformin_when_network_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)

    skill = DailyMedSkill()
    results = skill.retrieve({"drug": ["metformin"]}, max_results=3)

    assert results
    assert any("metformin" in result.source_entity.lower() for result in results)
    assert any("label" in (result.evidence_text or "").lower() for result in results)


def test_openfda_returns_offline_fixture_for_metformin_when_network_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)

    skill = OpenFDASkill()
    results = skill.retrieve({"drug": ["metformin"]}, max_results=5)

    assert results
    relationships = {result.relationship for result in results}
    assert "indicated_for" in relationships or "has_warning" in relationships


def test_medlineplus_returns_offline_fixture_for_metformin_when_network_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)

    skill = MedlinePlusSkill()
    results = skill.retrieve({"drug": ["metformin"]}, max_results=3)

    assert results
    assert any("metformin" in result.source_entity.lower() for result in results)
    assert any(result.relationship == "has_patient_drug_info" for result in results)


def test_medlineplus_offline_evidence_items_keep_table_metadata(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)

    skill = MedlinePlusSkill()
    records = [result.to_dict() for result in skill.retrieve({"drug": ["metformin"]}, max_results=1)]
    evidence_items = skill.build_evidence_items(records, query="What prescribing and safety information is available for metformin?")

    assert evidence_items
    assert evidence_items[0].metadata["source_entity"].lower() == "metformin"
    assert evidence_items[0].metadata["relationship"] == "has_patient_drug_info"
    assert evidence_items[0].metadata["target_entity"] == "Metformin"
