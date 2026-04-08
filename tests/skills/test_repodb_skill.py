from __future__ import annotations

from pathlib import Path

from skills.drug_repurposing.repodb.repodb_skill import RepoDBSkill


def test_repodb_returns_repurposing_record_for_non_approved_outcome_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "repodb.csv"
    csv_path.write_text(
        "\n".join(
            [
                "drug_name,drugbank_id,ind_name,ind_id,NCT,status,phase,DetailedStatus",
                "metformin,DB00331,type 2 diabetes mellitus,C0011860,NCT00000001,Terminated,Phase 2,Lack of efficacy",
            ]
        ),
        encoding="utf-8",
    )

    skill = RepoDBSkill(
        config={
            "csv_path": str(csv_path),
            "include_failed": False,
        }
    )

    results = skill.retrieve(
        {"drug": ["metformin"]},
        query="What are the approved indications and repurposing evidence of metformin?",
        max_results=5,
    )

    assert len(results) == 1
    assert results[0].relationship == "repurposing_evidence"
    assert results[0].target_entity == "type 2 diabetes mellitus"
    assert results[0].metadata["status"] == "Terminated"
    assert results[0].metadata["phase"] == "Phase 2"
