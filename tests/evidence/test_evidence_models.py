from __future__ import annotations

from drugclaw.evidence import (
    ClaimSummary,
    EvidenceItem,
    build_evidence_items_for_skill,
    score_answer_confidence,
    score_claim_confidence,
    score_evidence_item,
)


def test_rule_based_scoring_prefers_curated_database_over_prediction() -> None:
    curated = EvidenceItem(
        evidence_id="E1",
        source_skill="BindingDB",
        source_type="database",
        source_title="BindingDB binding record",
        source_locator="PMID:12345678",
        snippet="Imatinib Ki=21 nM against ABL1.",
        structured_payload={"affinity_type": "Ki", "affinity_value": "21"},
        claim="Imatinib targets ABL1.",
        evidence_kind="database_record",
        support_direction="supports",
        confidence=0.0,
        retrieval_score=0.92,
        timestamp="2026-03-18T00:00:00Z",
        metadata={},
    )
    predicted = EvidenceItem(
        evidence_id="E2",
        source_skill="Open Targets Platform",
        source_type="prediction",
        source_title="Open Targets linked target",
        source_locator="CHEMBL941:ENSG00000121410",
        snippet="Imatinib linked to target ABL1.",
        structured_payload={"relationship": "linked_target"},
        claim="Imatinib targets ABL1.",
        evidence_kind="model_prediction",
        support_direction="supports",
        confidence=0.0,
        retrieval_score=0.55,
        timestamp="2026-03-18T00:00:00Z",
        metadata={},
    )

    assert score_evidence_item(curated) > score_evidence_item(predicted)


def test_claim_and_answer_confidence_drop_when_evidence_conflicts() -> None:
    supporting = EvidenceItem(
        evidence_id="E1",
        source_skill="BindingDB",
        source_type="database",
        source_title="BindingDB binding record",
        source_locator="PMID:12345678",
        snippet="Imatinib Ki=21 nM against ABL1.",
        structured_payload={},
        claim="Imatinib targets ABL1.",
        evidence_kind="database_record",
        support_direction="supports",
        confidence=0.0,
        retrieval_score=0.92,
        timestamp="2026-03-18T00:00:00Z",
        metadata={},
    )
    conflicting = EvidenceItem(
        evidence_id="E2",
        source_skill="SpeculativeModel",
        source_type="prediction",
        source_title="Predictive ranking output",
        source_locator="model://abl1",
        snippet="The model ranks ABL1 as unlikely for imatinib.",
        structured_payload={},
        claim="Imatinib targets ABL1.",
        evidence_kind="model_prediction",
        support_direction="contradicts",
        confidence=0.0,
        retrieval_score=0.41,
        timestamp="2026-03-18T00:00:00Z",
        metadata={},
    )

    claim_score = score_claim_confidence([supporting, conflicting])
    answer_score = score_answer_confidence(
        [ClaimSummary(claim="Imatinib targets ABL1.", confidence=claim_score, evidence_ids=["E1", "E2"])]
    )

    assert claim_score < 0.8
    assert answer_score == claim_score


def test_build_evidence_items_semanticizes_kegg_ddi_enzyme_without_raw_partner_ids() -> None:
    items = build_evidence_items_for_skill(
        skill_name="KEGG Drug",
        query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        records=[
            {
                "source": "KEGG Drug",
                "source_entity": "warfarin",
                "relationship": "drug_drug_interaction",
                "target_entity": "dr:D00015",
                "evidence_text": "KEGG Drug DDI: dr:D00564 interacts with dr:D00015 (CI; Enzyme: CYP2C9)",
                "ddi_description": "Enzyme: CYP2C9",
                "target_type": "drug_or_compound",
            }
        ],
    )

    assert items[0].claim == "warfarin interaction mechanism involves CYP2C9"


def test_build_evidence_items_marks_unclassified_kegg_ddi_as_unresolved() -> None:
    items = build_evidence_items_for_skill(
        skill_name="KEGG Drug",
        query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        records=[
            {
                "source": "KEGG Drug",
                "source_entity": "warfarin",
                "relationship": "drug_drug_interaction",
                "target_entity": "cpd:C00304",
                "evidence_text": "KEGG Drug DDI: dr:D00564 interacts with cpd:C00304 (P; unclassified)",
                "ddi_description": "unclassified",
                "target_type": "drug_or_compound",
            }
        ],
    )

    assert items[0].claim == "warfarin has unresolved KEGG interaction entries"
