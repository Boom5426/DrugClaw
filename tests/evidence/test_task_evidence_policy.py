from __future__ import annotations

from drugclaw.evidence import EvidenceItem
from drugclaw.task_evidence_policy import classify_evidence_item


def _make_item(
    *,
    source_skill: str,
    relationship: str,
    target_entity: str,
    target_type: str,
    evidence_kind: str = "database_record",
    snippet: str = "",
    structured_payload: dict | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"{source_skill}:{relationship}:{target_entity}",
        source_skill=source_skill,
        source_type="database",
        source_title=f"{source_skill} evidence",
        source_locator=source_skill,
        snippet=snippet or f"{source_skill} {relationship} {target_entity}",
        structured_payload=structured_payload or {},
        claim=f"metformin {relationship} {target_entity}",
        evidence_kind=evidence_kind,
        support_direction="supports",
        confidence=0.0,
        retrieval_score=0.9,
        timestamp="2026-04-03T00:00:00Z",
        metadata={
            "source_entity": "metformin",
            "relationship": relationship,
            "target_entity": target_entity,
            "source_type": "drug",
            "target_type": target_type,
        },
    )


def test_policy_classifies_repodb_repurposing_rows_as_strong_structured() -> None:
    item = _make_item(
        source_skill="RepoDB",
        relationship="repurposing_evidence",
        target_entity="type 2 diabetes mellitus",
        target_type="disease",
        structured_payload={
            "status": "Terminated",
            "phase": "Phase 2",
        },
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.tier == "strong_structured"
    assert classification.slot == "repurposing_evidence"


def test_policy_classifies_drugrepobank_rows_as_strong_structured_repurposing_evidence() -> None:
    item = _make_item(
        source_skill="DrugRepoBank",
        relationship="repurposed_for",
        target_entity="polycystic ovary syndrome",
        target_type="disease",
        structured_payload={
            "status": "Clinical trial",
            "pmid": "23456789",
        },
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.tier == "strong_structured"
    assert classification.slot == "repurposing_evidence"


def test_policy_keeps_exploratory_repurposing_sources_in_repurposing_section_without_upgrading_them() -> None:
    oregano_item = _make_item(
        source_skill="OREGANO",
        relationship="clinical_signal",
        target_entity="ovarian cancer",
        target_type="disease",
        structured_payload={
            "type": "clinical_signal",
            "evidence": "observational evidence",
        },
    )
    repurpose_item = _make_item(
        source_skill="RepurposeDrugs",
        relationship="repurposed_for",
        target_entity="ovarian cancer",
        target_type="disease",
        structured_payload={
            "status": "Investigational",
            "score": "0.81",
        },
    )

    oregano_classification = classify_evidence_item("drug_repurposing", oregano_item)
    repurpose_classification = classify_evidence_item("drug_repurposing", repurpose_item)

    assert oregano_classification.tier == "generic_weak_support"
    assert oregano_classification.slot == "repurposing_evidence"
    assert repurpose_classification.tier == "generic_weak_support"
    assert repurpose_classification.slot == "repurposing_evidence"


def test_policy_classifies_drugbank_indications_as_strong_structured() -> None:
    item = _make_item(
        source_skill="DrugBank",
        relationship="indicated_for",
        target_entity="type 2 diabetes mellitus",
        target_type="disease",
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.tier == "strong_structured"
    assert classification.slot == "approved_indications"


def test_policy_classifies_official_label_indication_support_as_secondary_official() -> None:
    item = _make_item(
        source_skill="openFDA Human Drug",
        relationship="indicated_for",
        target_entity="indications and usage",
        target_type="label_section",
        evidence_kind="label_text",
        snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
        structured_payload={"field": "indications_and_usage"},
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.tier == "secondary_official_support"
    assert classification.slot == "approved_indications"


def test_policy_never_treats_openfda_label_sections_as_repurposing_evidence() -> None:
    item = _make_item(
        source_skill="openFDA Human Drug",
        relationship="has_warning",
        target_entity="warnings",
        target_type="label_section",
        evidence_kind="label_text",
        snippet="Postmarketing cases of metformin-associated lactic acidosis have been reported.",
        structured_payload={"field": "warnings"},
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.slot != "repurposing_evidence"


def test_policy_downgrades_open_targets_indications_to_generic_weak_support_for_phase_2a() -> None:
    item = _make_item(
        source_skill="Open Targets Platform",
        relationship="indicated_for",
        target_entity="type 2 diabetes mellitus",
        target_type="disease",
        structured_payload={"max_clinical_stage": "APPROVAL"},
    )

    classification = classify_evidence_item("drug_repurposing", item)

    assert classification.tier == "generic_weak_support"
    assert classification.slot == "approved_indications"
