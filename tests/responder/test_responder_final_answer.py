from __future__ import annotations

from drugclaw.claim_assessment import ClaimAssessment
from drugclaw.agent_responder import ResponderAgent
from drugclaw.evidence import EvidenceItem
from drugclaw.models import AgentState


class _LLMStub:
    def generate(self, messages, temperature=0.5):
        raise AssertionError("Responder should use structured evidence path in this test")


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
