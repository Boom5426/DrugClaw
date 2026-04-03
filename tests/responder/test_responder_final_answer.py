from __future__ import annotations

from drugclaw.claim_assessment import ClaimAssessment
from drugclaw.agent_responder import ResponderAgent
from drugclaw.evidence import EvidenceItem
from drugclaw.models import AgentState


class _LLMStub:
    def generate(self, messages, temperature=0.5):
        raise AssertionError("Responder should use structured evidence path in this test")


def _make_evidence_item(
    *,
    evidence_id: str,
    source_skill: str,
    claim: str,
    snippet: str,
    relationship: str,
    source_entity: str,
    target_entity: str,
    evidence_kind: str = "database_record",
    retrieval_score: float = 0.9,
    structured_payload: dict | None = None,
    metadata: dict | None = None,
) -> EvidenceItem:
    merged_metadata = {
        "relationship": relationship,
        "source_entity": source_entity,
        "target_entity": target_entity,
        "source_type": "drug",
        "target_type": "protein",
    }
    if metadata:
        merged_metadata.update(metadata)
    return EvidenceItem(
        evidence_id=evidence_id,
        source_skill=source_skill,
        source_type="database",
        source_title=f"{source_skill} evidence",
        source_locator=source_skill,
        snippet=snippet,
        structured_payload=structured_payload or {},
        claim=claim,
        evidence_kind=evidence_kind,
        support_direction="supports",
        confidence=0.0,
        retrieval_score=retrieval_score,
        timestamp="2026-03-18T00:00:00Z",
        metadata=merged_metadata,
    )


def test_responder_builds_structured_final_answer_with_conflict_warning() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What does imatinib target?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB binding record",
            source_locator="PMID:12345678",
            snippet="Imatinib Ki=21 nM against ABL1.",
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
            source_title="Open Targets linked target",
            source_locator="CHEMBL941:ENSG00000121410",
            snippet="Imatinib linked to ABL1.",
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

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.key_claims
    assert updated.final_answer_structured.summary_confidence < 0.8
    assert any("conflict" in warning.lower() for warning in updated.final_answer_structured.warnings)
    assert "Imatinib targets ABL1." in updated.current_answer


def test_responder_uses_claim_assessments_when_present() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What does imatinib target?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB binding record",
            source_locator="PMID:12345678",
            snippet="Imatinib Ki=21 nM against ABL1.",
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
    state.claim_assessments = [
        ClaimAssessment(
            claim="Imatinib targets ABL1.",
            verdict="supported",
            supporting_evidence_ids=["E1"],
            contradicting_evidence_ids=[],
            confidence=0.88,
            rationale="Supported by direct binding evidence.",
            limitations=["Claim relies on a single supporting evidence item."],
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.key_claims[0].claim == "Imatinib targets ABL1."
    assert updated.final_answer_structured.key_claims[0].confidence == 0.88


def test_responder_reports_insufficient_evidence_instead_of_hallucinating() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.retrieved_text = (
        "Query: What are the known drug targets of imatinib?\n"
        "Key Entities: {}\n"
        "Skills Used: ['TTD', 'ChEMBL']\n\n"
        "=== Results from TTD ===\n(no results retrieved; error: file not found)\n"
        "=== Results from ChEMBL ===\n(no results retrieved)\n"
    )
    state.retrieved_content = [
        {
            "source": "TTD",
            "source_entity": "",
            "target_entity": "",
            "relationship": "",
            "evidence_text": "(no results retrieved; error: file not found)",
        },
        {
            "source": "ChEMBL",
            "source_entity": "",
            "target_entity": "",
            "relationship": "",
            "evidence_text": "(no results retrieved)",
        },
    ]
    state.retrieval_diagnostics = [
        {"skill": "TTD", "strategy": "fallback_retrieve", "error": "file not found", "records": 0},
        {"skill": "ChEMBL", "strategy": "fallback_retrieve", "error": "", "records": 0},
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.summary_confidence == 0.0
    assert "No structured evidence was retrieved" in updated.current_answer
    assert "TTD" in updated.current_answer


def test_responder_summarizes_repetitive_single_source_limitations() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What prescribing and safety information is available for metformin?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="DailyMed",
            source_type="database",
            source_title="DailyMed label",
            source_locator="https://example.test/label-1",
            snippet="Metformin official label for extended-release tablet with prescribing details.",
            structured_payload={},
            claim="metformin has_official_label Label A",
            evidence_kind="label_text",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.91,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
        EvidenceItem(
            evidence_id="E2",
            source_skill="DailyMed",
            source_type="database",
            source_title="DailyMed label",
            source_locator="https://example.test/label-2",
            snippet="Metformin official label for film-coated tablet with prescribing details.",
            structured_payload={},
            claim="metformin has_official_label Label B",
            evidence_kind="label_text",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.90,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
        EvidenceItem(
            evidence_id="E3",
            source_skill="DailyMed",
            source_type="database",
            source_title="DailyMed guidance",
            source_locator="https://example.test/label-3",
            snippet="Metformin guidance includes indications, warnings, and patient counseling details.",
            structured_payload={},
            claim="metformin has_patient_guidance patient guidance",
            evidence_kind="label_text",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.89,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
        EvidenceItem(
            evidence_id="E4",
            source_skill="DailyMed",
            source_type="database",
            source_title="DailyMed interaction section",
            source_locator="https://example.test/label-4",
            snippet="Metformin label documents interaction precautions and monitoring recommendations.",
            structured_payload={},
            claim="metformin interacts_with drug interactions",
            evidence_kind="label_text",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.88,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    limitations = updated.final_answer_structured.limitations
    assert any(
        limitation.startswith("Multiple claims rely on a single source (4 claims).")
        for limitation in limitations
    )
    assert not any(
        limitation.startswith("Claim relies on a single source:")
        for limitation in limitations
    )
    assert "Multiple claims rely on a single source (4 claims)." in updated.current_answer


def test_responder_filters_target_lookup_noise_and_renders_target_summary() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="IMATINIB IC50=12 nM against ABL1",
            structured_payload={},
            claim="IMATINIB has_ic50_activity Tyrosine-protein kinase ABL1",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.92,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Tyrosine-protein kinase ABL1",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="E2",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="IMATINIB IC50=20 nM against KIT",
            structured_payload={},
            claim="IMATINIB has_ic50_activity Mast/stem cell growth factor receptor Kit",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.89,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Mast/stem cell growth factor receptor Kit",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="E3",
            source_skill="Molecular Targets",
            source_type="database",
            source_title="search result",
            source_locator="Molecular Targets",
            snippet="CCDI: leukemia",
            structured_payload={},
            claim="imatinib search_hit leukemia",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.40,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "search_hit",
                "source_entity": "imatinib",
                "target_entity": "leukemia",
                "source_type": "query",
                "target_type": "disease",
            },
        ),
        EvidenceItem(
            evidence_id="E4",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="IMATINIB ratio=Unchecked",
            structured_payload={},
            claim="IMATINIB has_ratio_activity Unchecked",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.20,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ratio_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Unchecked",
                "source_type": "drug",
                "target_type": "unknown",
            },
        ),
        EvidenceItem(
            evidence_id="E5",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="IMATINIB IC50=200 nM against K562",
            structured_payload={},
            claim="IMATINIB has_ic50_activity K562",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.30,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "K562",
                "source_type": "drug",
                "target_type": "cell_line",
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim for claim in updated.final_answer_structured.key_claims]
    assert any("ABL1" in claim for claim in claims)
    assert any("Kit" in claim or "KIT" in claim for claim in claims)
    assert all("search_hit" not in claim for claim in claims)
    assert all("Unchecked" not in claim for claim in claims)
    assert all("K562" not in claim for claim in claims)
    assert "Known Targets" in updated.current_answer
    assert "leukemia" not in updated.current_answer


def test_responder_deduplicates_repeated_limitations() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="E1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB record",
            source_locator="PMID:1",
            snippet="Imatinib binds ABL1.",
            structured_payload={},
            claim="Imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.91,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
        EvidenceItem(
            evidence_id="E2",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL record",
            source_locator="CHEMBL941",
            snippet="Imatinib binds KIT.",
            structured_payload={},
            claim="Imatinib targets KIT.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.88,
            timestamp="2026-03-18T00:00:00Z",
            metadata={},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    single_support_limitations = [
        limitation
        for limitation in updated.final_answer_structured.limitations
        if "single supporting evidence item" in limitation.lower()
    ]
    assert len(single_support_limitations) <= 1


def test_responder_prioritizes_core_targets_in_target_summary() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.evidence_items = [
        EvidenceItem(
            evidence_id="B1",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB record",
            source_locator="BindingDB",
            snippet="Imatinib binds ABL1.",
            structured_payload={},
            claim="imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.96,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "targets",
                "source_entity": "imatinib",
                "target_entity": "ABL1",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="D1",
            source_skill="DGIdb",
            source_type="database",
            source_title="DGIdb interaction",
            source_locator="DGIdb",
            snippet="Imatinib interacts with ABL1.",
            structured_payload={},
            claim="imatinib targets ABL1.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.88,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "targets",
                "source_entity": "imatinib",
                "target_entity": "ABL1",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="B2",
            source_skill="BindingDB",
            source_type="database",
            source_title="BindingDB record",
            source_locator="BindingDB",
            snippet="Imatinib binds KIT.",
            structured_payload={},
            claim="imatinib targets KIT.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.95,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "targets",
                "source_entity": "imatinib",
                "target_entity": "KIT",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="C1",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="Imatinib IC50 against PDGFRB.",
            structured_payload={},
            claim="IMATINIB has_ic50_activity Platelet-derived growth factor receptor beta",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.90,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Platelet-derived growth factor receptor beta",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="D2",
            source_skill="DGIdb",
            source_type="database",
            source_title="DGIdb interaction",
            source_locator="DGIdb",
            snippet="Imatinib interacts with PDGFRB.",
            structured_payload={},
            claim="imatinib targets PDGFRB.",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.86,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "targets",
                "source_entity": "imatinib",
                "target_entity": "PDGFRB",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="C2",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="Imatinib IC50 against FLT3.",
            structured_payload={},
            claim="IMATINIB has_ic50_activity Receptor-type tyrosine-protein kinase FLT3",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.70,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Receptor-type tyrosine-protein kinase FLT3",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
        EvidenceItem(
            evidence_id="C3",
            source_skill="ChEMBL",
            source_type="database",
            source_title="ChEMBL activity",
            source_locator="CHEMBL941",
            snippet="Imatinib IC50 against SRC.",
            structured_payload={},
            claim="IMATINIB has_ic50_activity Proto-oncogene tyrosine-protein kinase Src",
            evidence_kind="database_record",
            support_direction="supports",
            confidence=0.0,
            retrieval_score=0.68,
            timestamp="2026-03-18T00:00:00Z",
            metadata={
                "relationship": "has_ic50_activity",
                "source_entity": "IMATINIB",
                "target_entity": "Proto-oncogene tyrosine-protein kinase Src",
                "source_type": "drug",
                "target_type": "protein",
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    top_claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("ABL1" in claim for claim in top_claims)
    assert any("KIT" in claim for claim in top_claims)
    assert any("PDGFR" in claim for claim in top_claims)
    assert "Proto-oncogene tyrosine-protein kinase Src" not in "\n".join(top_claims)


def test_responder_keeps_target_like_inhibitor_evidence_in_target_summary() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets and mechanism of action of imatinib?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            snippet="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="ABL proto-oncogene 1, non-receptor tyrosine kinase",
            retrieval_score=0.78,
        ),
        _make_evidence_item(
            evidence_id="OT2",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor platelet derived growth factor receptor beta",
            snippet="IMATINIB inhibitor platelet derived growth factor receptor beta via Platelet-derived growth factor receptor beta inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="platelet derived growth factor receptor beta",
            retrieval_score=0.78,
        ),
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.92,
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    top_claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("ABL1" in claim for claim in top_claims)
    assert any("PDGFRB" in claim or "platelet derived growth factor receptor beta" in claim for claim in top_claims)
    assert any(
        evidence_id == "OT1" or evidence_id == "OT2"
        for claim in updated.final_answer_structured.key_claims
        for evidence_id in claim.evidence_ids
    )


def test_responder_semanticizes_ddi_claims_instead_of_exposing_raw_kegg_ids() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction cpd:C00304",
            snippet="KEGG Drug DDI: dr:D00682 interacts with cpd:C00304 (CI; contraindication with aspirin)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="cpd:C00304",
            retrieval_score=0.86,
            structured_payload={
                "ddi_label": "CI",
                "ddi_description": "contraindication with aspirin",
                "target_id": "cpd:C00304",
            },
            metadata={"target_type": "drug_or_compound"},
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "aspirin" in updated.final_answer_structured.key_claims[0].claim.lower()
    assert "cpd:c00304" not in updated.current_answer.lower()


def test_responder_prioritizes_informative_ddi_mechanism_claims_over_unresolved_kegg_noise() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction dr:D00015",
            snippet="KEGG Drug DDI: dr:D00564 interacts with dr:D00015 (CI; Enzyme: CYP2C9)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="dr:D00015",
            retrieval_score=0.86,
            structured_payload={
                "ddi_description": "Enzyme: CYP2C9",
                "target_id": "dr:D00015",
            },
            metadata={"target_type": "drug_or_compound"},
        ),
        _make_evidence_item(
            evidence_id="K2",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction cpd:C00304",
            snippet="KEGG Drug DDI: dr:D00564 interacts with cpd:C00304 (P; unclassified)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="cpd:C00304",
            retrieval_score=0.86,
            structured_payload={
                "ddi_description": "unclassified",
                "target_id": "cpd:C00304",
            },
            metadata={"target_type": "drug_or_compound"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("cyp2c9" in claim for claim in claims)
    assert all("unresolved kegg" not in claim for claim in claims)
    assert " interacts with dr " not in updated.current_answer.lower()
    assert " interacts with cpd " not in updated.current_answer.lower()


def test_responder_updates_state_claim_assessments_after_query_specific_filtering() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction dr:D00015",
            snippet="KEGG Drug DDI: dr:D00564 interacts with dr:D00015 (CI; Enzyme: CYP2C9)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="dr:D00015",
            retrieval_score=0.86,
            structured_payload={"ddi_description": "Enzyme: CYP2C9"},
            metadata={"target_type": "drug_or_compound"},
        ),
        _make_evidence_item(
            evidence_id="K2",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction cpd:C00304",
            snippet="KEGG Drug DDI: dr:D00564 interacts with cpd:C00304 (P; unclassified)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="cpd:C00304",
            retrieval_score=0.86,
            structured_payload={"ddi_description": "unclassified"},
            metadata={"target_type": "drug_or_compound"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assessment_claims = [assessment.claim.lower() for assessment in updated.claim_assessments]
    assert assessment_claims == ["warfarin interaction mechanism involves cyp2c9"]


def test_responder_uses_label_text_not_section_headings_for_labeling_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning warnings",
            snippet="Postmarketing cases of metformin-associated lactic acidosis have been reported; assess renal function and risk factors before use.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "warnings"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("type 2 diabetes mellitus" in claim for claim in claims)
    assert all("indications and usage" not in claim for claim in claims)
    assert all(" warnings" not in claim for claim in claims)


def test_responder_prefers_richer_label_sections_over_generic_patient_guidance_titles() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="MedlinePlus Drug Info",
            claim="metformin has_patient_drug_info Metformin",
            snippet="MedlinePlus: Metformin",
            relationship="has_patient_drug_info",
            source_entity="metformin",
            target_entity="Metformin",
            evidence_kind="label_text",
            retrieval_score=0.96,
            metadata={"target_type": "patient_info"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning warnings",
            snippet="Postmarketing cases of metformin-associated lactic acidosis have been reported; assess renal function and risk factors before use.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "warnings"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("type 2 diabetes mellitus" in claim for claim in claims)
    assert any("lactic acidosis" in claim for claim in claims)
    assert all("patient guidance" not in claim for claim in claims)
    assert "medlineplus" not in updated.current_answer.lower()


def test_responder_prioritizes_pgx_claims_and_filters_non_pgx_noise_for_pgx_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="P1",
            source_skill="CPIC",
            claim="clopidogrel has_pgx_guideline CYP2C19",
            snippet="CPIC: clopidogrel has a pharmacogenomic association with CYP2C19 (CPIC=A, ClinPGx=1A)",
            relationship="has_pgx_guideline",
            source_entity="clopidogrel",
            target_entity="CYP2C19",
            retrieval_score=0.86,
            structured_payload={
                "cpiclevel": "A",
                "clinpgxlevel": "1A",
                "pgxtesting": "Actionable PGx",
                "usedforrecommendation": True,
            },
            metadata={"target_type": "gene"},
        ),
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="clopidogrel drug_drug_interaction dr:D00069",
            snippet="KEGG Drug DDI: dr:D00109 interacts with dr:D00069 (P; precaution with omeprazole)",
            relationship="drug_drug_interaction",
            source_entity="clopidogrel",
            target_entity="dr:D00069",
            retrieval_score=0.86,
            structured_payload={"ddi_description": "precaution with omeprazole"},
            metadata={"target_type": "drug_or_compound"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("cyp2c19" in claim for claim in claims)
    assert all("drug_drug_interaction" not in claim for claim in claims)


def test_responder_filters_pgx_claims_to_the_query_drug() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="P1",
            source_skill="CPIC",
            claim="clopidogrel has_pgx_guideline CYP2C19",
            snippet="CPIC: clopidogrel has a pharmacogenomic association with CYP2C19 (CPIC=A, ClinPGx=1A)",
            relationship="has_pgx_guideline",
            source_entity="clopidogrel",
            target_entity="CYP2C19",
            retrieval_score=0.86,
            structured_payload={
                "cpiclevel": "A",
                "clinpgxlevel": "1A",
                "pgxtesting": "Actionable PGx",
                "usedforrecommendation": True,
            },
            metadata={"target_type": "gene"},
        ),
        _make_evidence_item(
            evidence_id="P2",
            source_skill="CPIC",
            claim="citalopram has_pgx_guideline CYP2C19",
            snippet="CPIC: citalopram has a pharmacogenomic association with CYP2C19 (CPIC=A, ClinPGx=1A)",
            relationship="has_pgx_guideline",
            source_entity="citalopram",
            target_entity="CYP2C19",
            retrieval_score=0.86,
            structured_payload={
                "cpiclevel": "A",
                "clinpgxlevel": "1A",
                "pgxtesting": "Actionable PGx",
                "usedforrecommendation": True,
            },
            metadata={"target_type": "gene"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("clopidogrel" in claim for claim in claims)
    assert all("citalopram" not in claim for claim in claims)
    assert "citalopram" not in updated.current_answer.lower()


def test_responder_filters_obvious_non_safety_noise_from_adr_summaries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the major safety risks and serious adverse reactions of clozapine?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="Clozaril causes_adverse_event DEATH",
            snippet="FAERS: Clozaril associated with DEATH (6,416 spontaneous reports)",
            relationship="causes_adverse_event",
            source_entity="Clozaril",
            target_entity="DEATH",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F2",
            source_skill="FAERS",
            claim="Clozaril causes_adverse_event SCHIZOPHRENIA",
            snippet="FAERS: Clozaril associated with SCHIZOPHRENIA (3,109 spontaneous reports)",
            relationship="causes_adverse_event",
            source_entity="Clozaril",
            target_entity="SCHIZOPHRENIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F3",
            source_skill="FAERS",
            claim="Clozaril causes_adverse_event NEUTROPENIA",
            snippet="FAERS: Clozaril associated with NEUTROPENIA (2,401 spontaneous reports)",
            relationship="causes_adverse_event",
            source_entity="Clozaril",
            target_entity="NEUTROPENIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("NEUTROPENIA" in claim for claim in claims)
    assert all("SCHIZOPHRENIA" not in claim for claim in claims)


def test_responder_semanticizes_known_adverse_drug_reactions_queries_as_adr() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known adverse drug reactions of aspirin?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="aspirin causes_adverse_event NAUSEA",
            snippet="FAERS: aspirin associated with NAUSEA",
            relationship="causes_adverse_event",
            source_entity="aspirin",
            target_entity="NAUSEA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.key_claims[0].claim == "aspirin serious safety signal: NAUSEA"
    assert "causes_adverse_event" not in updated.current_answer


def test_responder_filters_generic_faers_outcome_noise_from_adr_summaries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the major safety risks and serious adverse reactions of clozapine?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event HOSPITALISATION",
            snippet="FAERS: clozapine associated with HOSPITALISATION",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="HOSPITALISATION",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F2",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event DRUG INEFFECTIVE",
            snippet="FAERS: clozapine associated with DRUG INEFFECTIVE",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="DRUG INEFFECTIVE",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F3",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event NEUTROPENIA",
            snippet="FAERS: clozapine associated with NEUTROPENIA",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="NEUTROPENIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("NEUTROPENIA" in claim for claim in claims)
    assert all("HOSPITALISATION" not in claim for claim in claims)
    assert all("DRUG INEFFECTIVE" not in claim for claim in claims)


def test_responder_normalizes_open_targets_mechanism_labels_to_canonical_symbols() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets and mechanism of action of imatinib?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            snippet="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="ABL proto-oncogene 1, non-receptor tyrosine kinase",
            retrieval_score=0.86,
        ),
        _make_evidence_item(
            evidence_id="OT2",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.86,
        ),
        _make_evidence_item(
            evidence_id="OT3",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor KIT proto-oncogene, receptor tyrosine kinase",
            snippet="IMATINIB inhibitor KIT proto-oncogene, receptor tyrosine kinase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="KIT proto-oncogene, receptor tyrosine kinase",
            retrieval_score=0.86,
        ),
        _make_evidence_item(
            evidence_id="OT4",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor platelet derived growth factor receptor beta",
            snippet="IMATINIB inhibitor platelet derived growth factor receptor beta",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="platelet derived growth factor receptor beta",
            retrieval_score=0.86,
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    rendered = "\n".join(claim.claim for claim in updated.final_answer_structured.key_claims[:5])
    assert "ABL1" in rendered
    assert "BCR" in rendered
    assert "KIT" in rendered
    assert "PDGFRB" in rendered
    assert "GTPASE" not in rendered
    assert "ABL proto-oncogene 1, non-receptor tyrosine kinase" not in rendered
