from __future__ import annotations

from skills.dti.open_targets.open_targets_skill import OpenTargetsSkill


def test_open_targets_uses_current_mechanism_of_action_schema(
    monkeypatch,
) -> None:
    skill = OpenTargetsSkill()

    def _fake_graphql(query: str, path: list[str]):
        if "search(" in query:
            return [{"id": "CHEMBL941", "name": "IMATINIB", "entity": "drug"}]

        assert "linkedTargets" not in query

        return {
            "id": "CHEMBL941",
            "name": "IMATINIB",
            "mechanismsOfAction": {
                "rows": [
                    {
                        "mechanismOfAction": "Bcr/Abl fusion protein inhibitor",
                        "actionType": "INHIBITOR",
                        "targets": [
                            {
                                "id": "ENSG00000097007",
                                "approvedSymbol": "ABL1",
                                "approvedName": "ABL proto-oncogene 1, non-receptor tyrosine kinase",
                            },
                            {
                                "id": "ENSG00000186716",
                                "approvedSymbol": "BCR",
                                "approvedName": "BCR activator of RhoGEF and GTPase",
                            },
                        ],
                    },
                    {
                        "mechanismOfAction": "Stem cell growth factor receptor inhibitor",
                        "actionType": "INHIBITOR",
                        "targets": [
                            {
                                "id": "ENSG00000157404",
                                "approvedSymbol": "KIT",
                                "approvedName": "KIT proto-oncogene, receptor tyrosine kinase",
                            }
                        ],
                    },
                ]
            },
        }

    monkeypatch.setattr(skill, "_graphql", _fake_graphql)

    results = skill.retrieve({"drug": ["imatinib"]}, max_results=5)

    assert len(results) == 3
    assert {result.metadata["gene_symbol"] for result in results} == {"ABL1", "BCR", "KIT"}
    assert {result.relationship for result in results} == {"inhibitor"}
    assert any("Bcr/Abl fusion protein inhibitor" in (result.evidence_text or "") for result in results)


def test_open_targets_returns_sorted_indications_for_repurposing_query(
    monkeypatch,
) -> None:
    skill = OpenTargetsSkill()

    def _fake_graphql(query: str, path: list[str]):
        if "search(" in query:
            return [{"id": "CHEMBL1431", "name": "METFORMIN", "entity": "drug"}]

        assert "indications" in query

        return {
            "id": "CHEMBL1431",
            "name": "METFORMIN",
            "indications": {
                "rows": [
                    {
                        "disease": {"id": "EFO_0001073", "name": "obesity"},
                        "maxClinicalStage": "PHASE_3",
                    },
                    {
                        "disease": {"id": "MONDO_0005148", "name": "type 2 diabetes mellitus"},
                        "maxClinicalStage": "APPROVAL",
                    },
                    {
                        "disease": {"id": "MONDO_0100096", "name": "COVID-19"},
                        "maxClinicalStage": "UNKNOWN",
                    },
                ]
            },
        }

    monkeypatch.setattr(skill, "_graphql", _fake_graphql)

    results = skill.retrieve(
        {"drug": ["metformin"]},
        query="What are the approved indications and repurposing evidence of metformin?",
        max_results=3,
    )

    assert [result.target_entity for result in results] == [
        "type 2 diabetes mellitus",
        "obesity",
        "COVID-19",
    ]
    assert [result.relationship for result in results] == [
        "indicated_for",
        "investigated_for",
        "investigated_for",
    ]
    assert results[0].metadata["max_clinical_stage"] == "APPROVAL"
