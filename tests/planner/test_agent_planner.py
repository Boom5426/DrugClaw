from __future__ import annotations

from drugclaw.agent_planner import PlannerAgent
from drugclaw.query_plan import build_fallback_query_plan, is_direct_target_lookup
from drugclaw.resource_registry import ResourceEntry, ResourceRegistry


class _LLMStub:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload or {
            "question_type": "target_lookup",
            "entities": {"drug": ["imatinib"]},
            "subquestions": ["What are the known targets of imatinib?"],
            "preferred_skills": ["BindingDB", "ChEMBL"],
            "preferred_evidence_types": ["database_record"],
            "requires_graph_reasoning": False,
            "requires_prediction_sources": False,
            "requires_web_fallback": False,
            "answer_risk_level": "medium",
            "notes": ["Direct target lookup."],
        }
        self.error = error

    def generate_json(self, messages, temperature=0.2):
        if self.error is not None:
            raise self.error
        return self.payload


def test_fallback_query_plan_is_conservative() -> None:
    plan = build_fallback_query_plan("Tell me about imatinib")

    assert plan.question_type == "unknown"
    assert plan.requires_graph_reasoning is False
    assert plan.preferred_skills == []


def test_fallback_query_plan_infers_label_query_type_and_entity() -> None:
    plan = build_fallback_query_plan(
        "What prescribing and safety information is available for metformin?"
    )

    assert plan.question_type == "labeling"
    assert plan.entities == {"drug": ["metformin"]}
    assert plan.preferred_skills == [
        "DailyMed",
        "openFDA Human Drug",
        "MedlinePlus Drug Info",
    ]


def test_fallback_query_plan_infers_adr_query_type_for_adverse_drug_reactions_phrase() -> None:
    plan = build_fallback_query_plan(
        "What are the known adverse drug reactions of aspirin?"
    )

    assert plan.question_type == "adr"
    assert plan.entities == {"drug": ["aspirin"]}
    assert plan.preferred_skills == ["ADReCS", "FAERS", "nSIDES", "SIDER"]


def test_fallback_query_plan_infers_pgx_query_type_and_entity() -> None:
    plan = build_fallback_query_plan(
        "What pharmacogenomic factors affect clopidogrel efficacy and safety?"
    )

    assert plan.question_type == "pharmacogenomics"
    assert plan.entities == {"drug": ["clopidogrel"]}
    assert plan.preferred_skills == ["PharmGKB", "CPIC"]


def test_fallback_query_plan_prefers_mechanism_for_target_plus_moa_query() -> None:
    plan = build_fallback_query_plan(
        "What are the known drug targets and mechanism of action of imatinib?"
    )

    assert plan.question_type == "mechanism"
    assert plan.entities == {"drug": ["imatinib"]}
    assert plan.preferred_skills == [
        "Open Targets Platform",
        "DRUGMECHDB",
        "BindingDB",
        "ChEMBL",
    ]


def test_planner_classifies_direct_target_lookup_without_graph() -> None:
    plan = PlannerAgent(_LLMStub()).plan("What does imatinib target?")

    assert plan.question_type == "target_lookup"
    assert plan.requires_graph_reasoning is False


def test_direct_target_lookup_recognizes_does_target_question() -> None:
    assert is_direct_target_lookup(query="What does imatinib target?") is True


def test_planner_marks_label_query_as_non_graph_lookup() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "labeling",
                "entities": {"drug": ["metformin"]},
                "subquestions": ["What prescribing and safety information is available for metformin?"],
                "preferred_skills": ["DailyMed", "MedlinePlus Drug Info"],
                "preferred_evidence_types": ["label_text"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Prefer label and patient-facing sources."],
            }
        )
    ).plan("What prescribing and safety information is available for metformin?")

    assert plan.question_type == "labeling"
    assert plan.preferred_evidence_types == ["label_text"]
    assert plan.requires_graph_reasoning is False


def test_planner_allows_graph_for_ddi_mechanism_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "ddi_mechanism",
                "entities": {"drug": ["warfarin", "amiodarone"]},
                "subquestions": ["What interaction mechanisms link warfarin and amiodarone?"],
                "preferred_skills": ["DDInter", "DrugBank"],
                "preferred_evidence_types": ["database_record", "literature_statement"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": True,
                "answer_risk_level": "high",
                "notes": ["Interaction mechanism may require multi-hop composition."],
            }
        )
    ).plan("How does amiodarone interact with warfarin?")

    assert plan.question_type == "ddi_mechanism"
    assert plan.requires_graph_reasoning is True
    assert "DDInter" in plan.preferred_skills


def test_planner_uses_fallback_plan_when_llm_output_is_invalid() -> None:
    plan = PlannerAgent(_LLMStub(error=ValueError("bad json"))).plan("Tell me about imatinib")

    assert plan.question_type == "unknown"
    assert plan.subquestions == ["Tell me about imatinib"]


def test_planner_infers_drug_entity_when_model_returns_no_entities() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "Drug-Target Interaction",
                "entities": {},
                "subquestions": ["What are the primary protein targets of imatinib?"],
                "preferred_skills": ["ChEMBL", "BindingDB"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "low",
                "notes": ["Focus on known targets."],
            }
        )
    ).plan("What are the known drug targets of imatinib?")

    assert plan.entities == {"drug": ["imatinib"]}


def test_planner_normalizes_noncanonical_repurposing_question_type_to_fallback() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_indications_and_repurposing_evidence",
                "entities": {"drug": ["metformin"]},
                "subquestions": ["What are the approved indications and repurposing evidence of metformin?"],
                "preferred_skills": [
                    "DailyMed",
                    "DrugCentral",
                    "FDA Orange Book",
                    "openFDA Human Drug",
                ],
                "preferred_evidence_types": ["label_text", "database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Mixed indication and repurposing query."],
            }
        )
    ).plan("What are the approved indications and repurposing evidence of metformin?")

    assert plan.question_type == "drug_repurposing"
    assert plan.preferred_skills == [
        "RepoDB",
        "DrugCentral",
        "DrugBank",
        "Open Targets Platform",
        "DRUGMECHDB",
        "openFDA Human Drug",
    ]
    assert plan.requires_graph_reasoning is False


def test_planner_normalizes_real_world_noncanonical_repurposing_query_type_to_fallback() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_indication_and_repurposing_evidence_query",
                "entities": {"drug": ["metformin"]},
                "subquestions": ["What are the approved indications and repurposing evidence of metformin?"],
                "preferred_skills": [
                    "DailyMed",
                    "FDA Orange Book",
                    "openFDA Human Drug",
                ],
                "preferred_evidence_types": ["label_text", "database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Mixed indication and repurposing query."],
            }
        )
    ).plan("What are the approved indications and repurposing evidence of metformin?")

    assert plan.question_type == "drug_repurposing"
    assert plan.preferred_skills == [
        "RepoDB",
        "DrugCentral",
        "DrugBank",
        "Open Targets Platform",
        "DRUGMECHDB",
        "openFDA Human Drug",
    ]
    assert plan.requires_graph_reasoning is False


def test_planner_normalizes_real_world_noncanonical_question_types_to_supported_fallbacks() -> None:
    cases = [
        (
            "What are the major safety risks and serious adverse reactions of clozapine?",
            "drug_safety_adverse_reactions",
            "clozapine",
            "adr",
            ["ADReCS", "FAERS", "nSIDES", "SIDER"],
        ),
        (
            "What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
            "drug_drug_interaction_mechanism_query_with_clinical_relevance_prioritization",
            "warfarin",
            "ddi_mechanism",
            ["DDInter", "KEGG Drug", "MecDDI"],
        ),
        (
            "What key prescribing and clinical use information should be considered for metformin?",
            "drug_prescribing_and_clinical_use_summary",
            "metformin",
            "labeling",
            ["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
        ),
        (
            "What pharmacogenomic factors affect clopidogrel efficacy and safety?",
            "pharmacogenomics_mechanism_and_clinical_impact_query",
            "clopidogrel",
            "pharmacogenomics",
            ["PharmGKB", "CPIC"],
        ),
    ]

    for query, raw_question_type, drug_name, expected_question_type, expected_skills in cases:
        plan = PlannerAgent(
            _LLMStub(
                {
                    "question_type": raw_question_type,
                    "entities": {"drug": [drug_name]},
                    "subquestions": [query],
                    "preferred_skills": ["DailyMed", "KEGG Drug", "FAERS"],
                    "preferred_evidence_types": ["database_record"],
                    "requires_graph_reasoning": True,
                    "requires_prediction_sources": False,
                    "requires_web_fallback": False,
                    "answer_risk_level": "high",
                    "notes": ["Real-world noncanonical type from CLI logs."],
                }
            )
        ).plan(query)

        assert plan.question_type == expected_question_type
        assert plan.preferred_skills == expected_skills
        assert plan.requires_graph_reasoning is False


def test_planner_normalizes_noncanonical_target_plus_moa_question_type_to_mechanism() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_target_and_mechanism_of_action_lookup",
                "entities": {"drug": ["imatinib"]},
                "subquestions": ["What are the known drug targets and mechanism of action of imatinib?"],
                "preferred_skills": ["BindingDB", "ChEMBL", "DGIdb", "Open Targets Platform"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Mixed target and MoA query."],
            }
        )
    ).plan("What are the known drug targets and mechanism of action of imatinib?")

    assert plan.question_type == "mechanism"
    assert plan.preferred_skills == [
        "Open Targets Platform",
        "DRUGMECHDB",
        "BindingDB",
        "ChEMBL",
    ]
    assert plan.requires_graph_reasoning is False


class _PlannerRegistryStub:
    def get_skills_for_query(self, query):
        return ["TTD", "ChEMBL", "TarKG", "Open Targets Platform", "DGIdb"]


def test_planner_prompt_prefers_ready_non_local_skills() -> None:
    planner = PlannerAgent(
        _LLMStub(),
        skill_registry=_PlannerRegistryStub(),
        resource_registry=ResourceRegistry(
            [
                ResourceEntry(
                    id="ttd",
                    name="TTD",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=True,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="missing_metadata",
                    status_reason="missing local metadata",
                    access_mode="LOCAL_FILE",
                ),
                ResourceEntry(
                    id="chembl",
                    name="ChEMBL",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="CLI",
                ),
                ResourceEntry(
                    id="tarkg",
                    name="TarKG",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=True,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="missing_metadata",
                    status_reason="missing local metadata",
                    access_mode="LOCAL_FILE",
                ),
                ResourceEntry(
                    id="open_targets_platform",
                    name="Open Targets Platform",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
                ResourceEntry(
                    id="dgidb",
                    name="DGIdb",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
            ]
        ),
    )

    prompt = planner.get_planning_prompt("What are the known drug targets of imatinib?")

    assert "- ChEMBL" in prompt
    assert "- Open Targets Platform" in prompt
    assert "- DGIdb" in prompt
    assert "- TTD" not in prompt
    assert "- TarKG" not in prompt


def test_planner_prompt_prioritizes_primary_dti_sources_for_target_lookup() -> None:
    class _TargetLookupPlannerRegistryStub:
        def get_skills_for_query(self, query):
            return [
                "ChEMBL",
                "Molecular Targets",
                "DRUGMECHDB",
                "BindingDB",
                "DGIdb",
                "Open Targets Platform",
            ]

    planner = PlannerAgent(
        _LLMStub(),
        skill_registry=_TargetLookupPlannerRegistryStub(),
        resource_registry=ResourceRegistry(
            [
                ResourceEntry(
                    id="chembl",
                    name="ChEMBL",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="CLI",
                ),
                ResourceEntry(
                    id="bindingdb",
                    name="BindingDB",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
                ResourceEntry(
                    id="open_targets_platform",
                    name="Open Targets Platform",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
                ResourceEntry(
                    id="dgidb",
                    name="DGIdb",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
                ResourceEntry(
                    id="molecular_targets",
                    name="Molecular Targets",
                    category="dti",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
                ResourceEntry(
                    id="drugmechdb",
                    name="DRUGMECHDB",
                    category="drug_mechanism",
                    description="",
                    entrypoint="",
                    enabled=True,
                    requires_metadata=False,
                    required_metadata_paths=[],
                    required_dependencies=[],
                    supports_code_generation=True,
                    fallback_retrieve_supported=True,
                    status="ready",
                    status_reason="available",
                    access_mode="REST_API",
                ),
            ]
        ),
    )

    prompt = planner.get_planning_prompt("What are the known drug targets of imatinib?")

    assert "- BindingDB" in prompt
    assert "- ChEMBL" in prompt
    assert "- DGIdb" in prompt
    assert "- Open Targets Platform" in prompt
    assert "- Molecular Targets" not in prompt
    assert "- DRUGMECHDB" not in prompt
