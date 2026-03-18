from __future__ import annotations

from drugclaw.evidence import (
    ClaimSummary,
    EvidenceItem,
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
