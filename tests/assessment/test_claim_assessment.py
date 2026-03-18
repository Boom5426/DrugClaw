from __future__ import annotations

from drugclaw.claim_assessment import assess_claims
from drugclaw.evidence import EvidenceItem


def test_claim_assessment_marks_conflicting_claim_as_uncertain() -> None:
    evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB record",
            source_locator="PMID:1",
            snippet="Imatinib binds ABL1.",
            structured_payload={"affinity_value": "21"},
            claim="Imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.92,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
        EvidenceItem(
            evidence_id="E2",
            source_skill="Open Targets Platform",
            source_type="prediction",
            source_title="Open Targets prediction",
            source_locator="CHEMBL941:ABL1",
            snippet="Predicted linked target signal for ABL1.",
            structured_payload={"relationship": "linked_target"},
            claim="Imatinib targets ABL1.",
            evidence_kind="model_prediction",
            support_direction="contradicts",
            confidence=0.0,
            retrieval_score=0.41,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
    ]

    assessments = assess_claims(evidence_items)

    assert assessments[0].verdict == "uncertain"
    assert assessments[0].supporting_evidence_ids == ["E1"]
    assert assessments[0].contradicting_evidence_ids == ["E2"]


def test_claim_assessment_marks_supported_claim_with_single_source_limit() -> None:
    evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB record",
            source_locator="PMID:1",
            snippet="Imatinib binds ABL1 with nanomolar potency.",
            structured_payload={"affinity_value": "21"},
            claim="Imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.92,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        )
    ]

    assessments = assess_claims(evidence_items)

    assert assessments[0].verdict == "supported"
    assert "single supporting evidence" in assessments[0].limitations[0].lower()


def test_claim_assessment_marks_contradicted_claim() -> None:
    evidence_items = [
        EvidenceItem(
            evidence_id="E9",
            source_skill="ModelOnly",
            source_type="prediction",
            source_title="Prediction",
            source_locator="pred:1",
            snippet="Predicted not to target ABL1.",
            structured_payload={"relationship": "not_linked"},
            claim="Imatinib targets ABL1.",
            evidence_kind="model_prediction",
            support_direction="contradicts",
            confidence=0.0,
            retrieval_score=0.11,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        )
    ]

    assessments = assess_claims(evidence_items)

    assert assessments[0].verdict == "contradicted"
    assert assessments[0].supporting_evidence_ids == []
    assert assessments[0].contradicting_evidence_ids == ["E9"]


def test_claim_assessment_marks_neutral_evidence_as_insufficient() -> None:
    evidence_items = [
        EvidenceItem(
            evidence_id="E5",
            source_skill="DrugBank",
            source_type="database",
            source_title="DrugBank note",
            source_locator="DB:1",
            snippet="Imatinib entry mentions oncology use.",
            structured_payload={},
            claim="Imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="neutral",
            confidence=0.0,
            retrieval_score=0.20,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        )
    ]

    assessments = assess_claims(evidence_items)

    assert assessments[0].verdict == "insufficient"
