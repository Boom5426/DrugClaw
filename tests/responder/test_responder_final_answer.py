from __future__ import annotations

from drugclaw.claim_assessment import ClaimAssessment
from drugclaw.agent_responder import ResponderAgent
from drugclaw.evidence import EvidenceItem
from drugclaw.models import AgentState
from drugclaw.query_plan import QueryPlan


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


def test_responder_reports_deterministic_failure_modes_more_specifically() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.retrieval_diagnostics = [
        {
            "skill": "TTD",
            "strategy": "deterministic_only",
            "final_status": "deterministic_failed",
            "structured_status": "error",
            "structured_error": "retrieve() error: file not found",
            "script_status": "not_available",
            "records": 0,
        },
        {
            "skill": "LegacyTargetCLI",
            "strategy": "deterministic_only",
            "final_status": "success_text_only",
            "structured_status": "empty",
            "script_status": "success",
            "records": 0,
        },
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "structured retrieval failed" in updated.current_answer
    assert "text-only fallback output was available" in updated.current_answer


def test_responder_shapes_pgx_web_evidence_into_guideline_support_section() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="E1",
            source_skill="PharmGKB",
            claim="clopidogrel has a pharmacogenomic association with CYP2C19",
            snippet="Clopidogrel exposure and efficacy vary with CYP2C19 metabolizer status.",
            relationship="has_pgx_association",
            source_entity="clopidogrel",
            target_entity="CYP2C19",
            metadata={"target_type": "gene"},
        )
    ]
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "CPIC-guided clopidogrel therapy",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "snippet": "Clinical guidance supports altered antiplatelet strategy in CYP2C19 poor metabolizers.",
        }
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "Structured PGx Findings:" in updated.current_answer
    assert "Authority-First PGx Guidance:" in updated.current_answer
    assert "Guideline support" in updated.current_answer
    assert "CPIC-guided clopidogrel therapy" in updated.current_answer
    assert "Authority-first web evidence:" not in updated.current_answer
    assert "[web:PubMed] https://pubmed.ncbi.nlm.nih.gov/12345678/" in updated.final_answer_structured.citations


def test_responder_shapes_ddi_web_evidence_without_overriding_structured_interaction_claims() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DDInter",
            claim="warfarin drug_drug_interaction amiodarone",
            snippet="Warfarin interacts with amiodarone and requires INR monitoring.",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="amiodarone",
            retrieval_score=0.91,
            structured_payload={"ddi_description": "requires INR monitoring"},
            metadata={"target_type": "drug"},
        )
    ]
    state.web_search_results = [
        {
            "source": "DailyMed",
            "title": "Warfarin label interaction precautions",
            "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=example",
            "snippet": "Monitor INR closely when combined with CYP2C9 inhibitors because exposure may increase.",
        }
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.key_claims[0].claim == "warfarin interacts with amiodarone (requires INR monitoring)"
    assert "Clinically Important Interactions:" in updated.current_answer
    assert "warfarin interacts with amiodarone (requires INR monitoring)" in updated.current_answer
    assert "Authority-First Clinical Interaction Support:" in updated.current_answer
    assert "Mechanistic support" in updated.current_answer
    assert "Authority-first web evidence:" not in updated.current_answer


def test_responder_filters_off_topic_pubmed_support_for_warfarin_ddi_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        normalized_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        resolved_entities={"drug": ["warfarin"]},
        query_plan=QueryPlan(
            question_type="ddi_mechanism",
            entities={"drug": ["warfarin"]},
            plan_type="composite_query",
            primary_task={"task_type": "ddi_mechanism"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DDInter",
            claim="warfarin drug_drug_interaction amiodarone",
            snippet="DDInter: warfarin and amiodarone [Major] — CYP2C9 inhibition may increase warfarin exposure; monitor INR closely and consider dose reduction.",
            relationship="drug_drug_interaction_major",
            source_entity="warfarin",
            target_entity="amiodarone",
            retrieval_score=0.92,
            structured_payload={
                "severity": "Major",
                "mechanism": "CYP2C9 inhibition may increase warfarin exposure",
                "management": "Monitor INR closely and consider dose reduction",
            },
            metadata={"target_type": "drug"},
        )
    ]
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "Vericiguat, a novel sGC stimulator: Mechanism of action, clinical, and translational science",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39999991/",
            "snippet": "Clinical and translational science review of vericiguat.",
        },
        {
            "source": "PubMed",
            "title": "Warfarin and amiodarone interaction management",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39999992/",
            "snippet": "Monitor INR closely when warfarin is combined with amiodarone because exposure may increase.",
        },
    ]

    updated = responder.execute_simple(state)

    rendered = updated.current_answer.lower()
    assert "authority-first clinical interaction support:" in rendered
    assert "warfarin and amiodarone interaction management" in rendered
    assert "vericiguat" not in rendered


def test_responder_excludes_partner_mismatched_pubmed_support_for_warfarin_ddi_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        normalized_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        resolved_entities={"drug": ["warfarin"]},
        query_plan=QueryPlan(
            question_type="ddi_mechanism",
            entities={"drug": ["warfarin"]},
            plan_type="composite_query",
            primary_task={"task_type": "ddi_mechanism"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DDInter",
            claim="warfarin drug_drug_interaction amiodarone",
            snippet="DDInter: warfarin and amiodarone [Major] — CYP2C9 inhibition may increase warfarin exposure; monitor INR closely and consider dose reduction.",
            relationship="drug_drug_interaction_major",
            source_entity="warfarin",
            target_entity="amiodarone",
            retrieval_score=0.92,
            structured_payload={
                "severity": "Major",
                "mechanism": "CYP2C9 inhibition may increase warfarin exposure",
                "management": "Monitor INR closely and consider dose reduction",
            },
            metadata={"target_type": "drug"},
        )
    ]
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "A physiologically based pharmacokinetic/pharmacodynamic modeling approach for drug-drug interaction evaluation of warfarin enantiomers with sorafenib.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/35555555/",
            "snippet": "Warfarin drug-drug interaction evaluation with sorafenib using PBPK/PD modeling.",
        },
    ]

    updated = responder.execute_simple(state)

    rendered = updated.current_answer.lower()
    assert "sorafenib" not in rendered
    assert "authority-first clinical interaction support:" not in rendered


def test_responder_filters_off_topic_pubmed_support_for_clozapine_safety_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        resolved_entities={"drug": ["clozapine"]},
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "Clozapine treatment: Ensuring ongoing monitoring during the COVID-19 pandemic",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39999993/",
            "snippet": "Monitoring guidance for ANC surveillance and safe clozapine treatment continuation.",
        },
        {
            "source": "PubMed",
            "title": "Clozapine efficacy for treatment-resistant schizophrenia",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39999994/",
            "snippet": "Review of clozapine efficacy in treatment-resistant schizophrenia.",
        },
    ]

    updated = responder.execute_simple(state)

    rendered = updated.current_answer.lower()
    assert "authority-first safety support:" in rendered
    assert "ensuring ongoing monitoring" in rendered
    assert "treatment-resistant schizophrenia" not in rendered


def test_responder_filters_bulk_unclassified_ddi_noise_when_informative_interactions_exist() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction kanamycin",
            snippet="KEGG Drug DDI: dr:D00564 interacts with Kanamycin (P; unclassified)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="Kanamycin",
            retrieval_score=0.86,
            structured_payload={"ddi_label": "P", "ddi_description": "unclassified"},
            metadata={"target_type": "drug_or_compound"},
        ),
        _make_evidence_item(
            evidence_id="K2",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction oleandomycin",
            snippet="KEGG Drug DDI: dr:D00564 interacts with Oleandomycin (P; unclassified)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="Oleandomycin",
            retrieval_score=0.86,
            structured_payload={"ddi_label": "P", "ddi_description": "unclassified"},
            metadata={"target_type": "drug_or_compound"},
        ),
        _make_evidence_item(
            evidence_id="K3",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction menatetrenone",
            snippet="KEGG Drug DDI: dr:D00564 interacts with Menatetrenone (CI,P; unclassified)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="Menatetrenone",
            retrieval_score=0.86,
            structured_payload={"ddi_label": "CI,P", "ddi_description": "unclassified"},
            metadata={"target_type": "drug_or_compound"},
        ),
        _make_evidence_item(
            evidence_id="K4",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction ibuprofen",
            snippet="KEGG Drug DDI: dr:D00564 interacts with Ibuprofen (P; Enzyme: CYP2C9)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="Ibuprofen",
            retrieval_score=0.86,
            structured_payload={"ddi_label": "P", "ddi_description": "Enzyme: CYP2C9"},
            metadata={"target_type": "drug_or_compound"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("ibuprofen" in claim for claim in claims)
    assert all("kanamycin" not in claim for claim in claims)
    assert all("oleandomycin" not in claim for claim in claims)
    assert "kanamycin" not in updated.current_answer.lower()
    assert "oleandomycin" not in updated.current_answer.lower()


def test_claim_to_target_fragment_keeps_ddi_claims_intact() -> None:
    claim = "warfarin interacts with Ibuprofen (Enzyme: CYP2C9)"
    assert ResponderAgent._claim_to_target_fragment(claim) == claim


def test_responder_shapes_labeling_web_evidence_into_regulatory_support_section() -> None:
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
    state.web_search_results = [
        {
            "source": "DailyMed",
            "title": "Metformin prescribing information",
            "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=metformin",
            "snippet": "Official labeling reinforces renal function assessment and lactic acidosis precautions before and during therapy.",
        }
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "Structured Labeling Findings:" in updated.current_answer
    assert "type 2 diabetes mellitus" in updated.current_answer
    assert "Authority-First Labeling Support:" in updated.current_answer
    assert "Regulatory warning support" in updated.current_answer
    assert "Authority-first web evidence:" not in updated.current_answer


def test_responder_backfills_labeling_authority_support_from_official_evidence_when_web_is_pubmed_only() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    dailymed_item = _make_evidence_item(
        evidence_id="DM1",
        source_skill="DailyMed",
        claim="metformin has_official_label metformin label",
        snippet="DailyMed label: METFORMIN HYDROCHLORIDE TABLET [REMEDYREPACK INC.]",
        relationship="has_official_label",
        source_entity="metformin",
        target_entity="METFORMIN HYDROCHLORIDE TABLET [REMEDYREPACK INC.]",
        evidence_kind="label_text",
        retrieval_score=0.88,
        structured_payload={"sources": ["https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=metformin"]},
        metadata={"target_type": "drug_label"},
    )
    dailymed_item.source_locator = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=metformin"
    state.evidence_items = [
        dailymed_item,
        _make_evidence_item(
            evidence_id="L1",
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
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "Metformin observational safety review",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "snippet": "Observational literature discusses safety events reported with metformin.",
        }
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "Authority-First Labeling Support:" in updated.current_answer
    assert "DailyMed / dailymed.nlm.nih.gov" in updated.current_answer
    if "PubMed / pubmed.ncbi.nlm.nih.gov" in updated.current_answer:
        assert updated.current_answer.index("DailyMed / dailymed.nlm.nih.gov") < updated.current_answer.index(
            "PubMed / pubmed.ncbi.nlm.nih.gov"
        )


def test_responder_reports_partial_with_weak_support_for_label_only_repurposing_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="R1",
            source_skill="openFDA Human Drug",
            claim="Metformin indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="DailyMed",
            claim="metformin has_official_label Metformin hydrochloride tablet, film coated",
            snippet="DailyMed labeling for metformin includes indications for improving glycemic control in type 2 diabetes mellitus.",
            relationship="has_official_label",
            source_entity="metformin",
            target_entity="Metformin hydrochloride tablet, film coated",
            evidence_kind="label_text",
            retrieval_score=0.82,
            metadata={"target_type": "drug_label"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.task_type == "drug_repurposing"
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"
    assert updated.final_answer_structured.diagnostics["strong_approved_indication_count"] == 1
    assert updated.final_answer_structured.diagnostics["secondary_official_support_count"] == 1
    assert updated.final_answer_structured.diagnostics["online_weak_fallback_used"] is True
    assert "Approved indications:" in updated.current_answer
    assert "Repurposing evidence:" in updated.current_answer
    assert "Supporting signals:" in updated.current_answer
    assert "Structured Repurposing Evidence:" not in updated.current_answer


def test_responder_surfaces_local_repurposing_evidence_separately_from_approved_indications() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.86,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R3",
            source_skill="OREGANO",
            claim="metformin clinical_signal ovarian cancer",
            snippet="OREGANO: metformin -> ovarian cancer [clinical_signal] (evidence: observational evidence)",
            relationship="clinical_signal",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.75,
            structured_payload={"type": "clinical_signal", "evidence": "observational evidence"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    approved_block, repurposing_and_beyond = updated.current_answer.split("Repurposing evidence:", 1)
    repurposing_block = repurposing_and_beyond.split("Supporting signals:", 1)[0].lower()
    assert "metformin is approved for type 2 diabetes mellitus" in approved_block.lower()
    assert "label support:" not in approved_block.lower()
    assert "polycystic ovary syndrome" in repurposing_block
    assert "ovarian cancer" in repurposing_block


def test_responder_renders_single_task_repurposing_approved_indications_as_concise_disease_claims() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.86,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    approved_block = updated.current_answer.split("Repurposing evidence:", 1)[0].lower()
    assert "metformin is approved for type 2 diabetes mellitus" in approved_block
    assert "label support:" not in approved_block


def test_responder_composite_repurposing_query_keeps_label_support_out_of_primary_answer() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "repurposing_evidence"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "ZITUVIMET",
                "generic_names": ["sitagliptin", "metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.85,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    short_answer_block = updated.current_answer.split("Repurposing Evidence:", 1)[0]
    repurposing_block, labeling_block = updated.current_answer.split("Labeling Summary:", 1)
    repurposing_block = repurposing_block.split("Repurposing Evidence:", 1)[1].lower()
    labeling_block = labeling_block.lower()

    assert "polycystic ovary syndrome" in short_answer_block.lower()
    assert "zituvimet" not in short_answer_block.lower()
    assert "polycystic ovary syndrome" in repurposing_block
    assert "ovarian cancer" in repurposing_block
    assert "zituvimet" not in repurposing_block
    assert "metformin hydrochloride label support" in labeling_block
    assert "zituvimet" not in labeling_block


def test_responder_composite_repurposing_short_answer_includes_approved_indication_context() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "repurposing_evidence"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.85,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    short_answer_block = updated.current_answer.split("Repurposing Evidence:", 1)[0].lower()
    assert "type 2 diabetes mellitus" in short_answer_block
    assert "polycystic ovary syndrome" in short_answer_block
    assert "ovarian cancer" in short_answer_block
    assert short_answer_block.index("type 2 diabetes mellitus") < short_answer_block.index("ovarian cancer")


def test_responder_excludes_combination_label_noise_from_repurposing_approved_indications() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is a combination of sitagliptin and metformin hydrochloride indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "ZITUVIMET",
                "generic_names": ["sitagliptin and metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    approved_block = updated.current_answer.split("Repurposing evidence:", 1)[0].lower()
    assert "metformin is approved for type 2 diabetes mellitus" in approved_block
    assert "label support:" not in approved_block
    assert "zituvimet" not in approved_block


def test_responder_renders_repurposedrugs_rows_as_exploratory_repurposing_support() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert "exploratory repurposing signal for ovarian cancer" in updated.current_answer.lower()
    assert "has repurposing evidence for ovarian cancer" not in updated.current_answer.lower()


def test_responder_does_not_promote_weak_repurposing_with_strong_indication_to_strong_answer() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="I1",
            source_skill="DrugCentral",
            claim="metformin indicated_for type 2 diabetes mellitus",
            snippet="DrugCentral: metformin indicated for type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="metformin",
            target_entity="type 2 diabetes mellitus",
            retrieval_score=0.88,
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"
    assert updated.final_answer_structured.diagnostics["strong_record_count"] == 0
    assert updated.final_answer_structured.diagnostics["strong_approved_indication_count"] == 1


def test_responder_filters_combination_product_noise_out_of_repurposing_evidence_items() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "repurposing_evidence"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="Z1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "ZITUVIMET",
                "generic_names": ["sitagliptin and metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    filtered_claims = [item.claim.lower() for item in updated.final_answer_structured.evidence_items]
    assert any("metformin hydrochloride label support" in claim for claim in filtered_claims)
    assert any("polycystic ovary syndrome" in claim for claim in filtered_claims)
    assert all("zituvimet" not in claim for claim in filtered_claims)
    assert all("zituvimet" not in item.claim.lower() for item in updated.evidence_items)


def test_responder_downgrades_single_strong_plus_exploratory_repurposing_to_partial() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "repurposing_evidence"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R3",
            source_skill="OREGANO",
            claim="metformin clinical_signal ovarian cancer",
            snippet="OREGANO: metformin -> ovarian cancer [clinical_signal] (evidence: observational evidence)",
            relationship="clinical_signal",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.75,
            structured_payload={"type": "clinical_signal", "evidence": "observational evidence"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.diagnostics["strong_record_count"] == 1
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"


def test_responder_deduplicates_repeated_approved_indication_rows_in_repurposing_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="A2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride tablets, USP indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets, USP are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride tablets, USP",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="A3",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride tablets indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride tablets",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    approved_block = updated.current_answer.split("Approved indications:", 1)[1].split("Repurposing evidence:", 1)[0]
    approved_lines = [line for line in approved_block.splitlines() if line.startswith("- ")]
    assert len(approved_lines) == 1
    assert "metformin is approved for type 2 diabetes mellitus" in approved_lines[0].lower()


def test_responder_filters_off_topic_pubmed_support_for_repurposing_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]
    state.web_search_results = [
        {
            "source": "PubMed",
            "title": "Therapeutic effects of metformin on cocaine conditioned place preference and locomotion",
            "url": "https://pubmed.ncbi.nlm.nih.gov/40000001/",
            "snippet": "Behavioral neuroscience study of metformin effects on cocaine conditioned place preference and locomotion.",
        },
        {
            "source": "PubMed",
            "title": "Metformin repurposing for ovarian cancer",
            "url": "https://pubmed.ncbi.nlm.nih.gov/40000002/",
            "snippet": "Review of metformin repurposing evidence in ovarian cancer and related clinical development signals.",
        },
    ]

    updated = responder.execute_simple(state)

    rendered = updated.current_answer.lower()
    assert "authority-first repurposing support:" in rendered
    assert "metformin repurposing for ovarian cancer" in rendered
    assert "cocaine conditioned place preference" not in rendered


def test_responder_deduplicates_verbose_openfda_indication_variants_in_repurposing_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="single_task",
            primary_task={"task_type": "repurposing_evidence"},
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet=(
                "Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control "
                "in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus. "
                "Metformin hydrochloride tablets are biguanide indicated as an adjunct to diet and exercise to improve "
                "glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus."
            ),
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="A2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride tablets, USP indicated_for indications and usage",
            snippet=(
                "Metformin hydrochloride tablets, USP are indicated as an adjunct to diet and exercise to improve glycemic control "
                "in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus. "
                "Metformin hydrochloride tablets, USP are a biguanide indicated as an adjunct to diet and exercise to improve "
                "glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus."
            ),
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride tablets, USP",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="A3",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride tablets indicated_for indications and usage",
            snippet=(
                "Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control "
                "in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus. "
                "Metformin hydrochloride tablets is a biguanide indicated as an adjunct to diet and exercise to improve "
                "glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus."
            ),
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride tablets",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    approved_block = updated.current_answer.split("Approved indications:", 1)[1].split("Repurposing evidence:", 1)[0]
    approved_lines = [line for line in approved_block.splitlines() if line.startswith("- ")]
    assert len(approved_lines) == 1


def test_responder_filters_combination_product_repurposing_noise_from_final_evidence_and_downgrades_mixed_support() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the approved indications and repurposing evidence of metformin?",
        normalized_query="What are the approved indications and repurposing evidence of metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="drug_repurposing",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "repurposing_evidence"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="Z1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "ZITUVIMET",
                "generic_names": ["sitagliptin and metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin hydrochloride tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients 10 years of age and older with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="R1",
            source_skill="DrugRepoBank",
            claim="metformin repurposed_for polycystic ovary syndrome",
            snippet="DrugRepoBank: metformin repurposed for polycystic ovary syndrome [Clinical trial]",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="polycystic ovary syndrome",
            retrieval_score=0.84,
            structured_payload={"status": "Clinical trial", "pmid": "23456789"},
            metadata={"target_type": "disease"},
        ),
        _make_evidence_item(
            evidence_id="R2",
            source_skill="RepurposeDrugs",
            claim="metformin repurposed_for ovarian cancer",
            snippet="RepurposeDrugs: metformin -> ovarian cancer [Investigational] (score=0.81)",
            relationship="repurposed_for",
            source_entity="metformin",
            target_entity="ovarian cancer",
            retrieval_score=0.81,
            structured_payload={"status": "Investigational", "score": "0.81"},
            metadata={"target_type": "disease"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"
    assert "zituvimet" not in updated.current_answer.lower()
    assert all(
        "zituvimet" not in str(item.claim).lower()
        and "zituvimet" not in str(item.metadata.get("source_entity", "")).lower()
        for item in updated.final_answer_structured.evidence_items
    )


def test_responder_marks_target_without_mechanism_as_partial_for_mechanism_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.92,
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.task_type == "mechanism"
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"
    assert "Targets supported:" in updated.current_answer
    assert "Mechanism coverage:" in updated.current_answer


def test_responder_marks_strong_pgx_guidance_as_strong_answer() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What pharmacogenomic factors affect clopidogrel efficacy and safety?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="P-STRONG",
            source_skill="CPIC",
            claim="clopidogrel has_pgx_guideline CYP2C19",
            snippet="CPIC: clopidogrel has a pharmacogenomic association with CYP2C19 (CPIC=A, ClinPGx=1A)",
            relationship="has_pgx_guideline",
            source_entity="clopidogrel",
            target_entity="CYP2C19",
            retrieval_score=0.90,
            structured_payload={
                "cpiclevel": "A",
                "clinpgxlevel": "1A",
                "pgxtesting": "Actionable PGx",
                "usedforrecommendation": True,
            },
            metadata={"target_type": "gene"},
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.task_type == "pharmacogenomics"
    assert updated.final_answer_structured.final_outcome == "strong_answer"


def test_responder_marks_insufficient_safety_answers_as_honest_gap() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?"
    )

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.task_type == "adr"
    assert updated.final_answer_structured.final_outcome == "honest_gap"


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


def test_responder_separates_established_direct_targets_from_association_only_signals() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.query_plan = QueryPlan(
        question_type="target_lookup",
        entities={"drug": ["imatinib"]},
        subquestions=["What are the established direct targets of imatinib?"],
        preferred_skills=["BindingDB", "ChEMBL", "DGIdb", "Open Targets Platform"],
        preferred_evidence_types=["database_record"],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level="medium",
        notes=["Direct target query."],
        plan_type="single_task",
        primary_task={
            "task_type": "direct_targets",
            "question": "What are the established direct targets of imatinib?",
            "entities": {"drug": ["imatinib"]},
            "preferred_skills": ["BindingDB", "ChEMBL", "DGIdb", "Open Targets Platform"],
            "preferred_evidence_types": ["database_record"],
            "requires_graph_reasoning": False,
            "requires_prediction_sources": False,
            "requires_web_fallback": False,
            "answer_risk_level": "medium",
            "notes": ["Separate direct targets from weaker associations."],
        },
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.96,
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="imatinib targets KIT",
            snippet="imatinib IC50=100 nM against KIT",
            relationship="binds_ic50",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.94,
        ),
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DGIdb",
            claim="imatinib targets PDGFRB",
            snippet="DGIdb: imatinib inhibitor PDGFRB",
            relationship="inhibitor",
            source_entity="imatinib",
            target_entity="PDGFRB",
            retrieval_score=0.84,
        ),
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.80,
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Established Direct Targets:" in updated.current_answer
    assert "Association-Only Signals:" in updated.current_answer
    established_block = updated.current_answer.split("Association-Only Signals:")[0]
    association_block = updated.current_answer.split("Association-Only Signals:")[1]
    assert "ABL1" in established_block
    assert "KIT" in established_block
    assert "PDGFRB" not in established_block
    assert "BCR" not in established_block
    assert "PDGFRB" in association_block
    assert "BCR" not in association_block


def test_responder_renders_composite_query_with_summary_and_ordered_sections() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?"
    )
    state.query_plan = QueryPlan(
        question_type="mechanism",
        entities={"drug": ["imatinib"]},
        subquestions=[
            "What are the established direct targets of imatinib?",
            "What is the mechanism of action of imatinib?",
        ],
        preferred_skills=["BindingDB", "ChEMBL", "Open Targets Platform", "DRUGMECHDB"],
        preferred_evidence_types=["database_record"],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=True,
        answer_risk_level="medium",
        notes=["Composite targets plus mechanism query."],
        plan_type="composite_query",
        primary_task={
            "task_type": "direct_targets",
            "question": "What are the established direct targets of imatinib?",
            "entities": {"drug": ["imatinib"]},
            "preferred_skills": ["BindingDB", "ChEMBL"],
            "preferred_evidence_types": ["database_record"],
            "requires_graph_reasoning": False,
            "requires_prediction_sources": False,
            "requires_web_fallback": False,
            "answer_risk_level": "medium",
            "notes": ["Lead with direct targets."],
        },
        supporting_tasks=[
            {
                "task_type": "mechanism_of_action",
                "question": "What is the mechanism of action of imatinib?",
                "entities": {"drug": ["imatinib"]},
                "preferred_skills": ["Open Targets Platform", "DRUGMECHDB"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": True,
                "answer_risk_level": "medium",
                "notes": ["Mechanism section."],
            }
        ],
        answer_contract={
            "summary_style": "direct_answer_first",
            "section_order": [
                "summary",
                "direct_targets",
                "mechanism_of_action",
                "limitations",
            ],
        },
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.96,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="imatinib IC50=100 nM against KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.94,
        ),
        _make_evidence_item(
            evidence_id="M1",
            source_skill="Open Targets Platform",
            claim="imatinib mechanism ABL signaling inhibition",
            snippet="Imatinib inhibits BCR-ABL tyrosine kinase signaling.",
            relationship="mechanism",
            source_entity="imatinib",
            target_entity="ABL signaling inhibition",
            retrieval_score=0.83,
            structured_payload={"mechanism_of_action": "BCR-ABL tyrosine kinase inhibition"},
            metadata={"mechanism_of_action": "BCR-ABL tyrosine kinase inhibition"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Short Answer:" in updated.current_answer
    assert "Established Direct Targets:" in updated.current_answer
    assert "Mechanism Coverage:" in updated.current_answer
    assert "Targets Supported:" not in updated.current_answer
    assert updated.current_answer.index("Established Direct Targets:") < updated.current_answer.index(
        "Mechanism Coverage:"
    )


def test_responder_renders_task_aware_knowhow_guidance_from_query_plan() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?"
    )
    state.query_plan = QueryPlan(
        question_type="mechanism",
        entities={"drug": ["imatinib"]},
        subquestions=[
            "What are the established direct targets of imatinib?",
            "What is the mechanism of action of imatinib?",
        ],
        preferred_skills=["BindingDB", "Open Targets Platform"],
        preferred_evidence_types=["database_record"],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=True,
        answer_risk_level="medium",
        notes=["Composite targets plus mechanism query."],
        knowhow_doc_ids=["direct_targets_grounding", "mechanism_explanation"],
        knowhow_hints=[
            {
                "doc_id": "direct_targets_grounding",
                "title": "Direct target grounding",
                "task_id": "primary",
                "task_type": "direct_targets",
                "snippet": "Prioritize established direct binding evidence and separate association-only target claims.",
                "risk_level": "medium",
                "evidence_types": ["database_record"],
                "declared_by_skills": ["BindingDB", "DrugBank", "Open Targets Platform"],
            },
            {
                "doc_id": "mechanism_explanation",
                "title": "Mechanism explanation",
                "task_id": "support_1",
                "task_type": "mechanism_of_action",
                "snippet": "Explain mechanism after the direct target section and call out evidence limits explicitly.",
                "risk_level": "medium",
                "evidence_types": ["database_record"],
                "declared_by_skills": ["Open Targets Platform"],
            },
        ],
        plan_type="composite_query",
        primary_task={
            "task_type": "direct_targets",
            "task_id": "primary",
            "question": "What are the established direct targets of imatinib?",
            "entities": {"drug": ["imatinib"]},
            "preferred_skills": ["BindingDB"],
            "preferred_evidence_types": ["database_record"],
            "answer_risk_level": "medium",
            "knowhow_doc_ids": ["direct_targets_grounding"],
            "knowhow_hints": [
                {
                    "doc_id": "direct_targets_grounding",
                    "title": "Direct target grounding",
                    "task_id": "primary",
                    "task_type": "direct_targets",
                    "snippet": "Prioritize established direct binding evidence and separate association-only target claims.",
                    "risk_level": "medium",
                    "evidence_types": ["database_record"],
                    "declared_by_skills": ["BindingDB", "DrugBank", "Open Targets Platform"],
                }
            ],
        },
        supporting_tasks=[
            {
                "task_type": "mechanism_of_action",
                "task_id": "support_1",
                "question": "What is the mechanism of action of imatinib?",
                "entities": {"drug": ["imatinib"]},
                "preferred_skills": ["Open Targets Platform"],
                "preferred_evidence_types": ["database_record"],
                "answer_risk_level": "medium",
                "knowhow_doc_ids": ["mechanism_explanation"],
                "knowhow_hints": [
                    {
                        "doc_id": "mechanism_explanation",
                        "title": "Mechanism explanation",
                        "task_id": "support_1",
                        "task_type": "mechanism_of_action",
                        "snippet": "Explain mechanism after the direct target section and call out evidence limits explicitly.",
                        "risk_level": "medium",
                        "evidence_types": ["database_record"],
                        "declared_by_skills": ["Open Targets Platform"],
                    }
                ],
            }
        ],
        answer_contract={
            "summary_style": "direct_answer_first",
            "section_order": [
                "summary",
                "direct_targets",
                "mechanism_of_action",
                "limitations",
            ],
        },
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.96,
        ),
        _make_evidence_item(
            evidence_id="M1",
            source_skill="Open Targets Platform",
            claim="imatinib mechanism ABL signaling inhibition",
            snippet="Imatinib inhibits BCR-ABL tyrosine kinase signaling.",
            relationship="mechanism",
            source_entity="imatinib",
            target_entity="ABL signaling inhibition",
            retrieval_score=0.83,
            structured_payload={"mechanism_of_action": "BCR-ABL tyrosine kinase inhibition"},
            metadata={"mechanism_of_action": "BCR-ABL tyrosine kinase inhibition"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "Evidence interpretation guidance:" in updated.current_answer
    assert "Direct target grounding [BindingDB, DrugBank, Open Targets Platform]:" in updated.current_answer
    assert "Mechanism explanation [Open Targets Platform]:" in updated.current_answer
    assert updated.final_answer_structured.diagnostics["knowhow_doc_ids"] == [
        "direct_targets_grounding",
        "mechanism_explanation",
    ]


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


def test_responder_collapses_duplicate_direct_target_sections_from_malformed_composite_plan() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What does imatinib target?")
    state.query_plan = QueryPlan(
        question_type="target_lookup",
        entities={"drug": ["imatinib"]},
        subquestions=["What are the established direct targets of imatinib?"],
        preferred_skills=["BindingDB", "ChEMBL", "DGIdb"],
        preferred_evidence_types=["database_record"],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level="medium",
        notes=["Malformed composite target-only plan."],
        plan_type="composite_query",
        primary_task={
            "task_type": "direct_targets",
            "question": "What are the established direct targets of imatinib?",
            "entities": {"drug": ["imatinib"]},
            "preferred_skills": ["BindingDB", "ChEMBL"],
        },
        supporting_tasks=[
            {"task_type": "direct_targets", "question": "Repeat direct targets."},
            {"task_type": "target_profile", "question": "Broader target profile."},
        ],
        answer_contract={
            "summary_style": "direct_answer_first",
            "section_order": [
                "summary",
                "direct_targets",
                "direct_targets",
                "target_profile",
                "limitations",
            ],
        },
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib targets ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib targets KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
        ),
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DGIdb",
            claim="imatinib targets PDGFRB",
            snippet="DGIdb: imatinib inhibitor PDGFRB",
            relationship="targets",
            source_entity="imatinib",
            target_entity="PDGFRB",
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.current_answer.count("Established Direct Targets:") == 1
    assert updated.current_answer.count("Association-Only Signals:") == 1
    assert "Short Answer:" not in updated.current_answer


def test_responder_separates_core_direct_targets_from_additional_direct_activity_hits() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the known drug targets of imatinib?")
    state.query_plan = QueryPlan(
        question_type="target_lookup",
        entities={"drug": ["imatinib"]},
        subquestions=["What are the established direct targets of imatinib?"],
        preferred_skills=["BindingDB", "ChEMBL", "Open Targets Platform"],
        preferred_evidence_types=["database_record"],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level="medium",
        notes=["Direct target query."],
        plan_type="single_task",
        primary_task={
            "task_type": "direct_targets",
            "question": "What are the established direct targets of imatinib?",
            "entities": {"drug": ["imatinib"]},
            "preferred_skills": ["BindingDB", "ChEMBL", "Open Targets Platform"],
        },
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib binds KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="IMATINIB has_ic50_activity Platelet-derived growth factor receptor alpha",
            snippet="Imatinib IC50 = 85 nM against PDGFRA",
            relationship="has_ic50_activity",
            source_entity="IMATINIB",
            target_entity="Platelet-derived growth factor receptor alpha",
        ),
        _make_evidence_item(
            evidence_id="C2",
            source_skill="ChEMBL",
            claim="IMATINIB has_ic50_activity Receptor-type tyrosine-protein kinase FLT3",
            snippet="Imatinib IC50 = 120 nM against FLT3",
            relationship="has_ic50_activity",
            source_entity="IMATINIB",
            target_entity="Receptor-type tyrosine-protein kinase FLT3",
        ),
        _make_evidence_item(
            evidence_id="C3",
            source_skill="ChEMBL",
            claim="IMATINIB has_ic50_activity Proto-oncogene tyrosine-protein kinase Src",
            snippet="Imatinib IC50 = 150 nM against SRC",
            relationship="has_ic50_activity",
            source_entity="IMATINIB",
            target_entity="Proto-oncogene tyrosine-protein kinase Src",
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Established Direct Targets:" in updated.current_answer
    assert "Additional Direct Activity Hits:" in updated.current_answer
    established_block = updated.current_answer.split("Established Direct Targets:", 1)[1].split(
        "Additional Direct Activity Hits:",
        1,
    )[0]
    additional_block = updated.current_answer.split("Additional Direct Activity Hits:", 1)[1].split(
        "Association-Only Signals:",
        1,
    )[0]
    assert "ABL1" in established_block
    assert "KIT" in established_block
    assert "PDGFRA" in established_block
    assert "FLT3" not in established_block
    assert "SRC" not in established_block
    assert "FLT3" in additional_block
    assert "SRC" in additional_block


def test_responder_normalizes_target_labels_without_emitting_fragment_tokens() -> None:
    assert ResponderAgent._normalize_target_label(
        "Cytochrome P450 family 17 subfamily A member 1"
    ) == "Cytochrome P450 family 17 subfamily A member 1"
    assert ResponderAgent._normalize_target_label(
        "Protein kinase C alpha type"
    ) == "Protein kinase C alpha type"
    assert ResponderAgent._normalize_target_label(
        "Proto-oncogene tyrosine-protein kinase Src"
    ) == "SRC"


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


def test_responder_prefers_partner_named_ddi_claims_when_kegg_name_resolution_is_available() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="K1",
            source_skill="KEGG Drug",
            claim="warfarin drug_drug_interaction aspirin",
            snippet="KEGG Drug DDI: dr:D00564 interacts with Aspirin (CI; Enzyme: CYP2C9)",
            relationship="drug_drug_interaction",
            source_entity="warfarin",
            target_entity="Aspirin",
            retrieval_score=0.86,
            structured_payload={
                "ddi_description": "Enzyme: CYP2C9",
                "target_id": "dr:D00109",
            },
            metadata={
                "target_type": "drug_or_compound",
                "target_name": "Aspirin",
            },
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("warfarin interacts with aspirin" in claim for claim in claims)
    assert any("cyp2c9" in claim for claim in claims)


def test_responder_adds_ddi_mechanism_coverage_note_when_mechanistic_support_is_sparse() -> None:
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
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert "Mechanism coverage:" in updated.current_answer
    assert "does not establish complete interaction-mechanism coverage" in updated.current_answer


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


def test_responder_prefers_structured_label_sections_over_generic_official_label_titles() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DailyMed",
            claim="metformin has_official_label DailyMed official label",
            snippet="DailyMed label: METFORMIN HYDROCHLORIDE TABLET.",
            relationship="has_official_label",
            source_entity="metformin",
            target_entity="DailyMed official label",
            evidence_kind="label_text",
            retrieval_score=0.92,
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in patients with severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "contraindications"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride use_in_special_population use in specific populations",
            snippet="Assess renal function more frequently in older adults and other at-risk populations.",
            relationship="use_in_special_population",
            source_entity="Metformin Hydrochloride",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("contraindications" in claim for claim in claims)
    assert any("special-population use" in claim for claim in claims)
    assert all("official label summary" not in claim for claim in claims)


def test_responder_uses_glucophage_canonical_alias_context_for_labeling_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What prescribing and safety information is available for Glucophage?",
        normalized_query="What prescribing and safety information is available for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Use official labeling sources for canonical metformin evidence."],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="Z1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M1",
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
        _make_evidence_item(
            evidence_id="M2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.85,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("metformin" in claim for claim in claims)
    assert all("zituvimet" not in claim for claim in claims)
    assert "zituvimet" not in updated.current_answer.lower()


def test_responder_prefers_single_agent_alias_label_over_combination_noise() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What prescribing and safety information is available for Glucophage?",
        normalized_query="What prescribing and safety information is available for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Use official labeling sources for canonical metformin evidence."],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="G1",
            source_skill="openFDA Human Drug",
            claim="Glucophage indicated_for indications and usage",
            snippet="Glucophage is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Glucophage",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.83,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Glucophage",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="openFDA Human Drug",
            claim="Metformin and Saxagliptin indicated_for indications and usage",
            snippet="Metformin and saxagliptin tablets are indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin and Saxagliptin",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin and Saxagliptin",
                "generic_names": ["metformin hydrochloride and saxagliptin"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("glucophage" in claim for claim in claims)
    assert all("metformin and saxagliptin" not in claim for claim in claims)
    assert "metformin and saxagliptin" not in updated.current_answer.lower()
    assert updated.final_answer_structured.final_outcome != "honest_gap"


def test_responder_excludes_combination_product_noise_from_single_drug_labeling_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        normalized_query="What prescribing and safety information is available for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Use official labeling sources for canonical metformin evidence."],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning warnings",
            snippet="Postmarketing cases of metformin-associated lactic acidosis have been reported; assess renal function and risk factors before use.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "warnings"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="C2",
            source_skill="openFDA Human Drug",
            claim="Glipizide and Metformin Hydrochloride has_warning warnings",
            snippet="Glipizide and metformin hydrochloride tablets carry hypoglycemia and lactic acidosis warnings.",
            relationship="has_warning",
            source_entity="Glipizide and Metformin Hydrochloride",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.95,
            structured_payload={"field": "warnings"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("metformin" in claim for claim in claims)
    assert all("zituvimet" not in claim for claim in claims)
    assert all("glipizide and metformin hydrochloride" not in claim for claim in claims)
    assert "zituvimet" not in updated.current_answer.lower()
    assert "glipizide and metformin hydrochloride" not in updated.current_answer.lower()


def test_responder_downgrades_wrong_drug_only_labeling_answers_to_honest_gap() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What prescribing and safety information is available for Glucophage?",
        normalized_query="What prescribing and safety information is available for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Use official labeling sources for canonical metformin evidence."],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="Z1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET indicated_for indications and usage",
            snippet="ZITUVIMET is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="ZITUVIMET",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.96,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.final_outcome == "honest_gap"
    assert updated.final_answer_structured.summary_confidence < 0.4
    assert "wrong_drug_labeling_answer" in updated.final_answer_structured.diagnostics.get(
        "output_validation_issue_codes",
        [],
    )
    assert "zituvimet" not in updated.current_answer.lower()


def test_responder_caps_partial_labeling_answer_confidence_below_high_threshold() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        normalized_query="What prescribing and safety information is available for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Use official labeling sources for canonical metformin evidence."],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
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
        _make_evidence_item(
            evidence_id="M2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Used as an adjunct to diet and exercise to improve glycemic control in adults and pediatric patients with type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.85,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    assert updated.final_answer_structured.final_outcome == "partial_with_weak_support"
    assert updated.final_answer_structured.summary_confidence < 0.7


def test_responder_omits_empty_support_sections_from_composite_labeling_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[
                {"task_type": "major_adrs"},
                {"task_type": "pgx_guidance"},
            ],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in patients with severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "contraindications"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="M2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride use_in_special_population use in specific populations",
            snippet="Assess renal function more frequently in older adults and other at-risk populations.",
            relationship="use_in_special_population",
            source_entity="Metformin Hydrochloride",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Labeling Summary:" in updated.current_answer
    assert "Major Adrs:" not in updated.current_answer
    assert "Pgx Guidance:" not in updated.current_answer


def test_responder_omits_bogus_ddi_support_sections_when_only_generic_official_labels_are_available() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DailyMed",
            claim="metformin has_official_label DailyMed official label",
            snippet="DailyMed label: METFORMIN HYDROCHLORIDE TABLET.",
            relationship="has_official_label",
            source_entity="metformin",
            target_entity="DailyMed official label",
            evidence_kind="label_text",
            retrieval_score=0.92,
            metadata={
                "target_type": "label_section",
                "queried_drug": "metformin",
                "generic_names": ["metformin hydrochloride"],
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in patients with severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "contraindications"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Labeling Summary:" in updated.current_answer
    assert "Clinically Relevant Ddi:" not in updated.current_answer
    assert "interacts with DailyMed official label" not in updated.current_answer


def test_responder_filters_combination_product_noise_from_labeling_ddi_support() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="I1",
            source_skill="openFDA Human Drug",
            claim="ZITUVIMET interacts_with drug interactions",
            snippet="Concomitant use of cationic drugs may reduce metformin elimination and increase exposure.",
            relationship="interacts_with",
            source_entity="ZITUVIMET",
            target_entity="drug interactions",
            evidence_kind="label_text",
            retrieval_score=0.89,
            structured_payload={"field": "drug_interactions"},
            metadata={
                "target_type": "label_section",
                "brand_name": "ZITUVIMET",
                "generic_names": ["sitagliptin and metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": True,
            },
        ),
        _make_evidence_item(
            evidence_id="I2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride interacts_with drug interactions",
            snippet="Carbonic Anhydrase Inhibitors Clinical Impact: concomitant use may increase the risk of lactic acidosis.",
            relationship="interacts_with",
            source_entity="Metformin Hydrochloride",
            target_entity="drug interactions",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "drug_interactions"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Clinically Relevant Ddi:" in updated.current_answer
    assert "ZITUVIMET" not in updated.current_answer
    assert "Metformin Hydrochloride" in updated.current_answer


def test_responder_extracts_label_ddi_partner_from_openfda_interaction_text() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="I1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride interacts_with drug interactions",
            snippet="Table 3: Clinically Significant Drug Interactions with Metformin Hydrochloride Tablets Carbonic Anhydrase Inhibitors Clinical Impact: concomitant use may increase the risk of lactic acidosis.",
            relationship="interacts_with",
            source_entity="Metformin Hydrochloride",
            target_entity="drug interactions",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "drug_interactions"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claim = updated.final_answer_structured.key_claims[0].claim
    assert "Carbonic Anhydrase Inhibitors" in claim
    assert "drug interactions" not in claim.lower()


def test_responder_deduplicates_repeated_labeling_slots_to_surface_special_population_guidance() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="W1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning boxed warning",
            snippet="BOXED WARNING: Lactic acidosis may occur in patients with severe renal impairment.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="boxed warning",
            evidence_kind="label_text",
            retrieval_score=0.89,
            structured_payload={"field": "boxed_warning"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="W2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning boxed warning",
            snippet="BOXED WARNING: Stop metformin if lactic acidosis is suspected.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="boxed warning",
            evidence_kind="label_text",
            retrieval_score=0.88,
            structured_payload={"field": "boxed_warning"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="S1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride use_in_special_population use in specific populations",
            snippet="Assess renal function more frequently in older adults and other at-risk populations.",
            relationship="use_in_special_population",
            source_entity="Metformin Hydrochloride",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    top_claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:3]]
    assert sum("warning:" in claim.lower() for claim in top_claims) == 1
    assert any("special-population use" in claim.lower() for claim in top_claims)


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


def test_responder_retains_curated_adr_relationships_for_serious_safety_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(original_query="What are the major safety risks and serious adverse reactions of clozapine?")
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="A1",
            source_skill="ADReCS",
            claim="clozapine classified_adr AGRANULOCYTOSIS",
            snippet="ADReCS: clozapine is linked to agranulocytosis as a classified adverse drug reaction.",
            relationship="classified_adr",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.88,
            metadata={"target_type": "adverse_event"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:5]]
    assert any("AGRANULOCYTOSIS" in claim for claim in claims)


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


def test_responder_uses_official_label_risk_and_monitoring_in_composite_adr_labeling_answers() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="Clozaril causes_adverse_event NEUTROPENIA",
            snippet="FAERS: Clozaril associated with NEUTROPENIA",
            relationship="causes_adverse_event",
            source_entity="Clozaril",
            target_entity="NEUTROPENIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Labeling Summary:" in updated.current_answer
    labeling_block = updated.current_answer.split("Labeling Summary:", 1)[1].lower()
    assert "anc" in labeling_block
    assert "monitor" in labeling_block
    assert "serious safety signal" not in labeling_block


def test_responder_prioritizes_clinically_meaningful_clozapine_risks_over_generic_faers_outcomes() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event DEATH",
            snippet="FAERS: clozapine associated with DEATH",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="DEATH",
            retrieval_score=0.95,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F2",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event NEUTROPHILIA",
            snippet="FAERS: clozapine associated with NEUTROPHILIA",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="NEUTROPHILIA",
            retrieval_score=0.94,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F3",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.87,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    top_claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("AGRANULOCYTOSIS" in claim for claim in top_claims)
    assert all("DEATH" not in claim for claim in top_claims)
    assert all("NEUTROPHILIA" not in claim for claim in top_claims)


def test_responder_includes_ddinter_severity_and_management_when_available() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DDInter",
            claim="warfarin drug_drug_interaction amiodarone",
            snippet="DDInter: warfarin and amiodarone [Major] — CYP2C9 inhibition may increase warfarin exposure; monitor INR closely and consider dose reduction.",
            relationship="drug_drug_interaction_major",
            source_entity="warfarin",
            target_entity="amiodarone",
            retrieval_score=0.92,
            structured_payload={
                "severity": "Major",
                "mechanism": "CYP2C9 inhibition may increase warfarin exposure",
                "management": "Monitor INR closely and consider dose reduction",
            },
            metadata={"target_type": "drug"},
        )
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    claim = updated.final_answer_structured.key_claims[0].claim.lower()
    assert "amiodarone" in claim
    assert "major" in claim
    assert "monitor inr" in claim


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


def test_responder_uses_action_relationships_for_imatinib_mechanism_coverage_without_promoting_bcr_to_primary_summary() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.96,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="imatinib IC50=100 nM against KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.94,
        ),
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
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    top_claims = [claim.claim for claim in updated.final_answer_structured.key_claims[:3]]
    assert any("ABL1" in claim for claim in top_claims)
    assert any("KIT" in claim for claim in top_claims)
    assert all("BCR" not in claim for claim in top_claims)
    assert "No direct mechanism-of-action evidence was retrieved." not in updated.current_answer


def test_responder_filters_unbacked_mechanism_targets_and_deduplicates_short_answer_for_imatinib() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="imatinib Ki=21 nM against ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.96,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="imatinib IC50=100 nM against KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.94,
        ),
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
    ]

    updated = responder.execute_simple(state)

    short_answer_block = updated.current_answer.split("Short Answer:", 1)[1].split("Mechanism Coverage:", 1)[0]
    short_answer_lines = [line for line in short_answer_block.splitlines() if line.startswith("- ")]
    assert len(short_answer_lines) == 1
    assert short_answer_lines[0].count("ABL1") == 1
    assert short_answer_lines[0].count("KIT") == 1
    assert "BCR" not in short_answer_lines[0]

    mechanism_block = updated.current_answer.split("Mechanism Coverage:", 1)[1].split("Established Direct Targets:", 1)[0]
    assert "ABL1" in mechanism_block
    assert "KIT" in mechanism_block
    assert "BCR" not in mechanism_block


def test_responder_drops_hidden_bcr_signal_from_mechanism_evidence_and_claim_assessments() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.91,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib binds KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.88,
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="imatinib targets PDGFRB",
            snippet="ChEMBL: imatinib binds PDGFRB",
            relationship="targets",
            source_entity="imatinib",
            target_entity="PDGFRB",
            retrieval_score=0.87,
        ),
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            snippet="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="ABL proto-oncogene 1, non-receptor tyrosine kinase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
        ),
        _make_evidence_item(
            evidence_id="OT2",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert updated.final_answer_structured is not None
    evidence_claims = [item.claim for item in updated.final_answer_structured.evidence_items]
    assert all("BCR" not in claim for claim in evidence_claims)
    assessment_claims = [assessment.claim for assessment in updated.claim_assessments]
    assert all("BCR" not in claim for claim in assessment_claims)


def test_responder_omits_bcr_from_association_only_signals_for_imatinib_targets() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        normalized_query="What are the known drug targets of imatinib?",
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.91,
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="imatinib has_ic50_activity KIT",
            snippet="ChEMBL: imatinib IC50 against KIT",
            relationship="has_ic50_activity",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.88,
        ),
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DGIdb",
            claim="imatinib targets PDGFRB",
            snippet="DGIdb: imatinib inhibitor PDGFRB",
            relationship="inhibitor",
            source_entity="imatinib",
            target_entity="PDGFRB",
            retrieval_score=0.84,
        ),
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.80,
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Association-Only Signals:" in updated.current_answer
    association_block = updated.current_answer.split("Association-Only Signals:")[1]
    assert "PDGFRB" in association_block
    assert "BCR" not in association_block


def test_responder_omits_fusion_component_only_signals_from_composite_imatinib_answer() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.91,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib binds KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.88,
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="imatinib targets PDGFRB",
            snippet="ChEMBL: imatinib binds PDGFRB",
            relationship="targets",
            source_entity="imatinib",
            target_entity="PDGFRB",
            retrieval_score=0.87,
        ),
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            snippet="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="ABL proto-oncogene 1, non-receptor tyrosine kinase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
        ),
        _make_evidence_item(
            evidence_id="OT2",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
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

    assert "Association-Only Signals:" in updated.current_answer
    association_block = updated.current_answer.split("Association-Only Signals:", 1)[1]
    for marker in ("Authority-First", "Evidence interpretation guidance:", "Limitations:"):
        association_block = association_block.split(marker, 1)[0]
    assert "ABL1" not in association_block
    assert "KIT" not in association_block
    assert "PDGFRB" not in association_block
    assert "BCR" not in association_block
    assert "No weaker association-only target signals were retrieved." in association_block


def test_responder_drops_hidden_bcr_items_from_mechanism_evidence_and_assessments() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.91,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib binds KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.88,
        ),
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase",
            snippet="IMATINIB inhibitor ABL proto-oncogene 1, non-receptor tyrosine kinase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="ABL proto-oncogene 1, non-receptor tyrosine kinase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
        ),
        _make_evidence_item(
            evidence_id="OT2",
            source_skill="Open Targets Platform",
            claim="IMATINIB inhibitor BCR activator of RhoGEF and GTPase",
            snippet="IMATINIB inhibitor BCR activator of RhoGEF and GTPase via Bcr/Abl fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="BCR activator of RhoGEF and GTPase",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
            metadata={"mechanism_of_action": "Bcr/Abl fusion protein inhibitor"},
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
    ]

    updated = responder.execute_simple(state)

    evidence_claims = [item.claim.lower() for item in updated.final_answer_structured.evidence_items]
    assessment_claims = [assessment.claim.lower() for assessment in updated.claim_assessments]

    assert all("bcr" not in claim for claim in evidence_claims)
    assert all("bcr" not in claim for claim in assessment_claims)


def test_responder_preserves_fusion_specific_mechanism_text_without_independent_target_support() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What is the mechanism of action of testdrug?",
        normalized_query="What is the mechanism of action of testdrug?",
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="OT1",
            source_skill="Open Targets Platform",
            claim="TESTDRUG inhibitor ALK",
            snippet="TESTDRUG inhibitor ALK via EML4-ALK fusion protein inhibitor",
            relationship="inhibitor",
            source_entity="TESTDRUG",
            target_entity="ALK",
            retrieval_score=0.86,
            structured_payload={"mechanism_of_action": "EML4-ALK fusion protein inhibitor"},
            metadata={"mechanism_of_action": "EML4-ALK fusion protein inhibitor"},
        )
    ]

    updated = responder.execute_simple(state)

    claims = [claim.claim.lower() for claim in updated.final_answer_structured.key_claims]
    assert any("fusion protein inhibitor" in claim for claim in claims)
    assert all("mechanism: inhibits alk" not in claim for claim in claims)


def test_responder_prioritizes_meaningful_clozapine_risks_over_faers_outcome_noise() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event DEATH",
            snippet="FAERS: clozapine associated with DEATH",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="DEATH",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F2",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event NEUTROPHILIA",
            snippet="FAERS: clozapine associated with NEUTROPHILIA",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="NEUTROPHILIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F3",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="F4",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event NEUTROPENIA",
            snippet="FAERS: clozapine associated with NEUTROPENIA",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="NEUTROPENIA",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Major Adrs:" in updated.current_answer
    adr_lines = [
        line
        for line in updated.current_answer.split("Major Adrs:", 1)[1].split("Labeling Summary:", 1)[0].splitlines()
        if line.startswith("- ")
    ]
    first_three = "\n".join(adr_lines[:3])
    assert "AGRANULOCYTOSIS" in first_three
    assert "NEUTROPENIA" in first_three
    assert "DEATH" not in first_three
    assert "NEUTROPHILIA" not in first_three


def test_responder_includes_official_label_support_in_adr_authority_block() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        resolved_entities={"drug": ["clozapine"]},
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DailyMed",
            claim="clozapine has_official_label DailyMed official label",
            snippet="DailyMed label: Clozapine can cause severe neutropenia and requires baseline and ongoing ANC monitoring.",
            relationship="has_official_label",
            source_entity="clozapine",
            target_entity="DailyMed official label",
            evidence_kind="label_text",
            retrieval_score=0.92,
            metadata={
                "target_type": "label_section",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]
    state.web_search_results = []

    updated = responder.execute_simple(state)

    assert "Authority-First Safety Support:" in updated.current_answer
    authority_block = updated.current_answer.split("Authority-First Safety Support:", 1)[1].split(
        "Evidence interpretation guidance:",
        1,
    )[0]
    assert "DailyMed" in authority_block or "openFDA" in authority_block
    assert "ANC" in authority_block or "neutropenia" in authority_block.lower()
    assert "indicated for" not in authority_block.lower()


def test_responder_prefers_warning_over_indication_when_synthesizing_adr_authority_support() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        resolved_entities={"drug": ["clozapine"]},
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine indicated_for indications and usage",
            snippet="Clozapine is indicated for treatment-resistant schizophrenia.",
            relationship="indicated_for",
            source_entity="Clozapine",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "indications_and_usage"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_warning warnings",
            snippet="Severe neutropenia can occur with clozapine; obtain a baseline ANC before treatment and monitor ANC regularly during therapy.",
            relationship="has_warning",
            source_entity="Clozapine",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "warnings"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]
    state.web_search_results = []

    updated = responder.execute_simple(state)

    authority_block = updated.current_answer.split("Authority-First Safety Support:", 1)[1].split(
        "Evidence interpretation guidance:",
        1,
    )[0]
    assert "has warning" in authority_block.lower() or "severe neutropenia" in authority_block.lower()
    assert "indicated for" not in authority_block.lower()


def test_responder_normalizes_obvious_openfda_label_typos_in_rendered_clozapine_answer() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the major safety risks and serious adverse reactions of clozapine?",
        normalized_query="What are the major safety risks and serious adverse reactions of clozapine?",
        resolved_entities={"drug": ["clozapine"]},
        query_plan=QueryPlan(
            question_type="adr",
            entities={"drug": ["clozapine"]},
            plan_type="composite_query",
            primary_task={"task_type": "major_adrs"},
            supporting_tasks=[{"task_type": "labeling_summary"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="F1",
            source_skill="FAERS",
            claim="clozapine causes_adverse_event AGRANULOCYTOSIS",
            snippet="FAERS: clozapine associated with AGRANULOCYTOSIS",
            relationship="causes_adverse_event",
            source_entity="clozapine",
            target_entity="AGRANULOCYTOSIS",
            retrieval_score=0.86,
            metadata={"target_type": "adverse_event"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Clozapine use_in_special_population use in specific populations",
            snippet=(
                "Lactaton: Infants exposed to Clozapine ODT through breast milk should be monitored "
                "for excess sedation and neutropenia."
            ),
            relationship="use_in_special_population",
            source_entity="Clozapine",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
        _make_evidence_item(
            evidence_id="L2",
            source_skill="openFDA Human Drug",
            claim="Clozapine has_dosing_guidance dosage and administration",
            snippet=(
                "Recommended starting oral dosage is 12.5 mg once daily or twice daily. "
                "Subsequently may increase the doage in increments up to 100 mg, once or twice weekly."
            ),
            relationship="has_dosing_guidance",
            source_entity="Clozapine",
            target_entity="dosage and administration",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "dosage_and_administration"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Clozapine",
                "generic_names": ["clozapine"],
                "queried_drug": "clozapine",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Lactaton" not in updated.current_answer
    assert "doage" not in updated.current_answer
    assert "Lactation:" in updated.current_answer
    assert "dosage in increments" in updated.current_answer


def test_responder_omits_ddi_support_sections_without_interaction_evidence() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="D1",
            source_skill="DailyMed",
            claim="metformin has_official_label DailyMed official label",
            snippet="DailyMed label: METFORMIN HYDROCHLORIDE TABLET.",
            relationship="has_official_label",
            source_entity="metformin",
            target_entity="DailyMed official label",
            evidence_kind="label_text",
            retrieval_score=0.92,
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="L1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in patients with severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "contraindications"},
            metadata={
                "target_type": "label_section",
                "brand_name": "Metformin Hydrochloride",
                "generic_names": ["metformin hydrochloride"],
                "queried_drug": "metformin",
                "is_combination_product": False,
            },
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Labeling Summary:" in updated.current_answer
    assert "Clinically Relevant Ddi:" not in updated.current_answer


def test_responder_promotes_mechanism_backed_pdgfrb_into_established_targets_for_imatinib() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="B1",
            source_skill="BindingDB",
            claim="imatinib targets ABL1",
            snippet="BindingDB: imatinib binds ABL1",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL1",
            retrieval_score=0.91,
        ),
        _make_evidence_item(
            evidence_id="B2",
            source_skill="BindingDB",
            claim="imatinib targets KIT",
            snippet="BindingDB: imatinib binds KIT",
            relationship="targets",
            source_entity="imatinib",
            target_entity="KIT",
            retrieval_score=0.88,
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="ChEMBL",
            claim="imatinib targets ABL2",
            snippet="ChEMBL: imatinib binds ABL2",
            relationship="targets",
            source_entity="imatinib",
            target_entity="ABL2",
            retrieval_score=0.87,
        ),
        _make_evidence_item(
            evidence_id="C2",
            source_skill="ChEMBL",
            claim="imatinib targets PDGFRB",
            snippet="ChEMBL: imatinib binds PDGFRB",
            relationship="targets",
            source_entity="imatinib",
            target_entity="PDGFRB",
            retrieval_score=0.86,
        ),
        _make_evidence_item(
            evidence_id="OT1",
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

    assert "Established Direct Targets:" in updated.current_answer
    established_block = updated.current_answer.split("Established Direct Targets:", 1)[1].split(
        "Additional Direct Activity Hits:",
        1,
    )[0]
    additional_block = updated.current_answer.split("Additional Direct Activity Hits:", 1)[1].split(
        "Association-Only Signals:",
        1,
    )[0]
    assert "PDGFRB" in established_block
    assert "ABL2" not in established_block
    assert "ABL2" in additional_block


def test_responder_prioritizes_special_population_and_monitoring_label_sections() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[{"task_type": "clinically_relevant_ddi"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="M1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_adverse_reaction adverse reactions",
            snippet="The most common adverse reactions are diarrhea and nausea.",
            relationship="has_adverse_reaction",
            source_entity="Metformin Hydrochloride",
            target_entity="adverse reactions",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "adverse_reactions"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin is indicated for type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M3",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning boxed warning",
            snippet="Lactic acidosis is a boxed warning.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="boxed warning",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "boxed_warning"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M4",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "contraindications"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M5",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride interacts_with drug interactions",
            snippet="Carbonic anhydrase inhibitors may increase the risk of lactic acidosis.",
            relationship="interacts_with",
            source_entity="Metformin Hydrochloride",
            target_entity="drug interactions",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "drug_interactions"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M6",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride use_in_special_population use in specific populations",
            snippet="Assess renal function more frequently in older adults and other at-risk populations.",
            relationship="use_in_special_population",
            source_entity="Metformin Hydrochloride",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="M7",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_dosing_guidance dosage and administration",
            snippet="Assess renal function before starting metformin and as clinically indicated thereafter.",
            relationship="has_dosing_guidance",
            source_entity="Metformin Hydrochloride",
            target_entity="dosage and administration",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "dosage_and_administration"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Labeling Summary:" in updated.current_answer
    labeling_block = updated.current_answer.split("Labeling Summary:", 1)[1].split("Authority-First", 1)[0]
    assert "contraindications" in labeling_block.lower()
    assert "special-population use" in labeling_block.lower()
    assert "dosing guidance" in labeling_block.lower()


def test_responder_deduplicates_repeated_label_section_types_to_preserve_breadth() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What key prescribing and clinical use information should be considered for metformin?",
        normalized_query="What key prescribing and clinical use information should be considered for metformin?",
        resolved_entities={"drug": ["metformin"]},
        query_plan=QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            plan_type="composite_query",
            primary_task={"task_type": "labeling_summary"},
            supporting_tasks=[{"task_type": "pgx_guidance"}],
        ),
    )
    state.evidence_items = [
        _make_evidence_item(
            evidence_id="W1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning boxed warning",
            snippet="BOXED WARNING: Lactic acidosis can occur with metformin.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="boxed warning",
            evidence_kind="label_text",
            retrieval_score=0.87,
            structured_payload={"field": "boxed_warning"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="W2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_warning warnings",
            snippet="WARNING: Lactic acidosis risk increases with renal impairment.",
            relationship="has_warning",
            source_entity="Metformin Hydrochloride",
            target_entity="warnings",
            evidence_kind="label_text",
            retrieval_score=0.86,
            structured_payload={"field": "warnings"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="C1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in severe renal impairment.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.85,
            structured_payload={"field": "contraindications"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="C2",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_contraindication contraindications",
            snippet="Metformin is contraindicated in metabolic acidosis.",
            relationship="has_contraindication",
            source_entity="Metformin Hydrochloride",
            target_entity="contraindications",
            evidence_kind="label_text",
            retrieval_score=0.84,
            structured_payload={"field": "contraindications"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="S1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride use_in_special_population use in specific populations",
            snippet="Assess renal function more frequently in older adults and other at-risk populations.",
            relationship="use_in_special_population",
            source_entity="Metformin Hydrochloride",
            target_entity="use in specific populations",
            evidence_kind="label_text",
            retrieval_score=0.83,
            structured_payload={"field": "use_in_specific_populations"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="D1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride has_dosing_guidance dosage and administration",
            snippet="Assess renal function before starting metformin and as clinically indicated thereafter.",
            relationship="has_dosing_guidance",
            source_entity="Metformin Hydrochloride",
            target_entity="dosage and administration",
            evidence_kind="label_text",
            retrieval_score=0.82,
            structured_payload={"field": "dosage_and_administration"},
            metadata={"target_type": "label_section"},
        ),
        _make_evidence_item(
            evidence_id="I1",
            source_skill="openFDA Human Drug",
            claim="Metformin Hydrochloride indicated_for indications and usage",
            snippet="Metformin is indicated for type 2 diabetes mellitus.",
            relationship="indicated_for",
            source_entity="Metformin Hydrochloride",
            target_entity="indications and usage",
            evidence_kind="label_text",
            retrieval_score=0.81,
            structured_payload={"field": "indications_and_usage"},
            metadata={"target_type": "label_section"},
        ),
    ]

    updated = responder.execute_simple(state)

    labeling_block = updated.current_answer.split("Labeling Summary:", 1)[1]
    labeling_block = labeling_block.split("Authority-First", 1)[0]
    labeling_block = labeling_block.split("Limitations:", 1)[0]
    labeling_lines = [line for line in labeling_block.splitlines() if line.startswith("- ")]
    assert sum(" warning:" in line.lower() for line in labeling_lines) == 1
    assert sum(" contraindications:" in line.lower() for line in labeling_lines) == 1
    joined = "\n".join(labeling_lines).lower()
    assert "special-population use" in joined
    assert "dosing guidance" in joined


def test_responder_treats_action_relationships_as_mechanism_support_for_moa_queries() -> None:
    responder = ResponderAgent(_LLMStub())
    state = AgentState(
        original_query="What are the known drug targets and mechanism of action of imatinib?",
        normalized_query="What are the known drug targets and mechanism of action of imatinib?",
        query_plan=QueryPlan(
            question_type="mechanism",
            entities={"drug": ["imatinib"]},
            plan_type="composite_query",
            primary_task={"task_type": "mechanism_of_action"},
            supporting_tasks=[{"task_type": "direct_targets"}],
        ),
    )
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
            claim="IMATINIB inhibitor KIT proto-oncogene, receptor tyrosine kinase",
            snippet="IMATINIB inhibitor KIT proto-oncogene, receptor tyrosine kinase",
            relationship="inhibitor",
            source_entity="IMATINIB",
            target_entity="KIT proto-oncogene, receptor tyrosine kinase",
            retrieval_score=0.86,
        ),
    ]

    updated = responder.execute_simple(state)

    assert "Mechanism Coverage:" in updated.current_answer
    mechanism_block = updated.current_answer.split("Mechanism Coverage:", 1)[1].split("Established Direct Targets:", 1)[0].lower()
    assert "no direct mechanism-of-action evidence was retrieved" not in mechanism_block
    assert "inhibit" in mechanism_block
    assert "abl1" in mechanism_block
