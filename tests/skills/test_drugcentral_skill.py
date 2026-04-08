from __future__ import annotations

from pathlib import Path

from skills.drug_knowledgebase.drugcentral.drugcentral_skill import DrugCentralSkill
from skills.drug_knowledgebase.drugcentral import drugcentral_skill as drugcentral_skill_module


def _make_local_contract_paths(tmp_path: Path) -> dict[str, str]:
    structures_path = tmp_path / "structures.smiles.tsv"
    targets_path = tmp_path / "drug.target.interaction.tsv"
    approved_path = tmp_path / "FDA+EMA+PMDA_Approved.csv"

    structures_path.write_text("placeholder\n", encoding="utf-8")
    targets_path.write_text("placeholder\n", encoding="utf-8")
    approved_path.write_text("placeholder\n", encoding="utf-8")

    return {
        "structures_path": str(structures_path),
        "targets_path": str(targets_path),
        "approved_path": str(approved_path),
    }


def test_drugcentral_keeps_roster_rows_as_has_approved_entry_when_no_indication_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        drugcentral_skill_module.drugcentral_example,
        "search",
        lambda name: {
            "structures": [{"INN": "Metformin"}],
            "targets": [],
            "approved": [{"id": "123", "name": "Metformin Hydrochloride"}],
        },
    )

    skill = DrugCentralSkill(config=_make_local_contract_paths(tmp_path))
    results = skill.retrieve({"drug": ["metformin"]}, max_results=5)

    assert len(results) == 1
    assert results[0].relationship == "has_approved_entry"
    assert results[0].target_entity == "Metformin Hydrochloride"


def test_drugcentral_promotes_disease_level_indication_rows_when_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        drugcentral_skill_module.drugcentral_example,
        "search",
        lambda name: {
            "structures": [{"INN": "Metformin"}],
            "targets": [],
            "approved": [
                {
                    "id": "123",
                    "name": "Metformin Hydrochloride",
                    "indication": "type 2 diabetes mellitus",
                }
            ],
        },
    )

    skill = DrugCentralSkill(config=_make_local_contract_paths(tmp_path))
    results = skill.retrieve(
        {"drug": ["metformin"]},
        query="What are the approved indications and repurposing evidence of metformin?",
        max_results=5,
    )

    assert len(results) == 1
    assert results[0].relationship == "indicated_for"
    assert results[0].target_entity == "type 2 diabetes mellitus"
