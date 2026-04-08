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


def test_open_targets_build_evidence_items_preserves_phase_2a_classification_metadata() -> None:
    skill = OpenTargetsSkill()

    evidence_items = skill.build_evidence_items(
        [
            {
                "source_entity": "IMATINIB",
                "source_type": "drug",
                "target_entity": "ABL1",
                "target_type": "protein",
                "relationship": "inhibitor",
                "source": "Open Targets Platform",
                "evidence_text": "IMATINIB inhibitor ABL1 via Bcr/Abl fusion protein inhibitor (Open Targets MoA)",
                "metadata": {
                    "chembl_id": "CHEMBL941",
                    "target_id": "ENSG00000097007",
                    "gene_symbol": "ABL1",
                    "mechanism_of_action": "Bcr/Abl fusion protein inhibitor",
                },
            },
            {
                "source_entity": "METFORMIN",
                "source_type": "drug",
                "target_entity": "type 2 diabetes mellitus",
                "target_type": "disease",
                "relationship": "indicated_for",
                "source": "Open Targets Platform",
                "evidence_text": "METFORMIN indicated_for type 2 diabetes mellitus (Open Targets max stage APPROVAL)",
                "metadata": {
                    "chembl_id": "CHEMBL1431",
                    "disease_id": "MONDO_0005148",
                    "max_clinical_stage": "APPROVAL",
                },
            },
        ],
        query="What are the approved indications and repurposing evidence of metformin?",
    )

    assert evidence_items[0].structured_payload["mechanism_of_action"] == "Bcr/Abl fusion protein inhibitor"
    assert evidence_items[1].structured_payload["max_clinical_stage"] == "APPROVAL"
