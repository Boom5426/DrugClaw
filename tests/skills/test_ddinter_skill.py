from __future__ import annotations

from skills.ddi.ddinter.ddinter_skill import DDInterSkill


def _raise_offline(*args, **kwargs):
    raise OSError("offline")


def test_ddinter_offline_fixture_exposes_severity_mechanism_and_management(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", _raise_offline)

    skill = DDInterSkill()
    results = skill.retrieve({"drug": ["warfarin"]}, max_results=5)

    assert results

    amiodarone = next(
        result
        for result in results
        if result.source_entity == "warfarin" and result.target_entity == "amiodarone"
    )
    assert amiodarone.relationship == "drug_drug_interaction_major"
    assert "cyp" in amiodarone.metadata["mechanism"].lower()
    assert "monitor inr" in amiodarone.metadata["management"].lower()
