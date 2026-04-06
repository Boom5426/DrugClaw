from __future__ import annotations

from drugclaw.agent_planner import PlannerAgent
from drugclaw.knowhow_models import KnowHowDocument
from drugclaw.knowhow_registry import KnowHowRegistry
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


def _make_knowhow_registry() -> KnowHowRegistry:
    return KnowHowRegistry.from_documents(
        [
            KnowHowDocument(
                doc_id="direct_targets_grounding",
                title="Direct target grounding",
                task_types=["direct_targets"],
                evidence_types=["database_record"],
                declared_by_skills=["BindingDB", "DrugBank", "Open Targets Platform"],
                risk_level="medium",
                conflict_policy="Prefer direct binding evidence before association-only evidence.",
                answer_template="direct_targets",
                max_prompt_snippets=1,
                body_path="",
                body_text="Prioritize established direct binding evidence and separate association-only target claims.",
            ),
            KnowHowDocument(
                doc_id="mechanism_explanation",
                title="Mechanism explanation",
                task_types=["mechanism_of_action"],
                evidence_types=["database_record"],
                declared_by_skills=["Open Targets Platform"],
                risk_level="medium",
                conflict_policy="Mark mechanism claims as hypothesis when direct support is thin.",
                answer_template="mechanism_of_action",
                max_prompt_snippets=1,
                body_path="",
                body_text="Explain mechanism after the direct target section and call out evidence limits explicitly.",
            ),
        ]
    )


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


def test_fallback_query_plan_builds_composite_plan_for_targets_plus_moa_query() -> None:
    plan = build_fallback_query_plan(
        "What are the known drug targets and mechanism of action of imatinib?"
    )

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert [task.task_type for task in plan.supporting_tasks] == [
        "mechanism_of_action"
    ]
    assert plan.answer_contract is not None
    assert plan.answer_contract.section_order[:3] == [
        "summary",
        "direct_targets",
        "mechanism_of_action",
    ]


def test_fallback_query_plan_builds_single_task_direct_targets_plan() -> None:
    plan = build_fallback_query_plan("What are the known drug targets of imatinib?")

    assert plan.plan_type == "single_task"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert plan.question_type == "target_lookup"
    assert plan.primary_task.preferred_skills == [
        "BindingDB",
        "ChEMBL",
        "DGIdb",
        "Open Targets Platform",
    ]


def test_planner_classifies_direct_target_lookup_without_graph() -> None:
    plan = PlannerAgent(_LLMStub()).plan("What does imatinib target?")

    assert plan.question_type == "target_lookup"
    assert plan.requires_graph_reasoning is False


def test_planner_normalizes_v2_payload_into_primary_and_supporting_tasks() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "composite_query",
                "primary_task": {
                    "task_type": "direct_targets",
                    "question": "What are the established direct targets of imatinib?",
                    "entities": {"drug": ["imatinib"]},
                    "preferred_skills": ["BindingDB", "ChEMBL"],
                    "preferred_evidence_types": ["database_record"],
                    "requires_graph_reasoning": False,
                    "requires_prediction_sources": False,
                    "requires_web_fallback": False,
                    "answer_risk_level": "medium",
                    "notes": ["Use direct binding resources first."],
                },
                "supporting_tasks": [
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
                        "notes": ["Add mechanism coverage as supporting context."],
                    }
                ],
                "execution_tasks": [
                    {"task_id": "primary", "priority": 100},
                    {"task_id": "support_1", "priority": 80},
                ],
                "answer_contract": {
                    "summary_style": "direct_answer_first",
                    "section_order": [
                        "summary",
                        "direct_targets",
                        "mechanism_of_action",
                        "limitations",
                    ],
                },
                "entities": {"drug": ["imatinib"]},
                "subquestions": [
                    "What are the established direct targets of imatinib?",
                    "What is the mechanism of action of imatinib?",
                ],
                "preferred_skills": [
                    "BindingDB",
                    "ChEMBL",
                    "Open Targets Platform",
                    "DRUGMECHDB",
                ],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": True,
                "answer_risk_level": "medium",
                "notes": ["Composite plan."],
            }
        )
    ).plan("What are the known drug targets and mechanism of action of imatinib?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert [task.task_type for task in plan.supporting_tasks] == [
        "mechanism_of_action"
    ]
    assert plan.answer_contract is not None
    assert plan.answer_contract.section_order[1:3] == [
        "direct_targets",
        "mechanism_of_action",
    ]


def test_planner_accepts_string_task_types_in_v2_payload() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "composite_query",
                "primary_task": "direct_targets",
                "supporting_tasks": ["mechanism_of_action"],
                "entities": {"drug": ["imatinib"]},
                "subquestions": [
                    "What are the established direct targets of imatinib?",
                    "What is the mechanism of action of imatinib?",
                ],
                "preferred_skills": [
                    "BindingDB",
                    "Open Targets Platform",
                ],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": True,
                "answer_risk_level": "medium",
                "notes": ["Composite plan."],
            }
        )
    ).plan("What are the known drug targets and mechanism of action of imatinib?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert [task.task_type for task in plan.supporting_tasks] == [
        "mechanism_of_action"
    ]


def test_planner_collapses_malformed_composite_target_lookup_into_single_task() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "composite_query",
                "question_type": "target_lookup",
                "primary_task": "direct_targets",
                "supporting_tasks": [
                    "direct_targets",
                    "target_profile",
                    "known target associations",
                ],
                "entities": {"drug": ["imatinib"]},
                "subquestions": [
                    "What are the established direct targets of imatinib?",
                ],
                "preferred_skills": [
                    "BindingDB",
                    "ChEMBL",
                    "DGIdb",
                ],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Malformed target-only plan."],
            }
        )
    ).plan("What does imatinib target?")

    assert plan.plan_type == "single_task"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert plan.supporting_tasks == []


def test_planner_accepts_execution_task_alias_fields_in_v2_payload() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "single_task",
                "question_type": "target_lookup",
                "primary_task": "direct_targets",
                "execution_tasks": [
                    {
                        "id": "primary",
                        "task": "direct_targets",
                        "priority": 100,
                        "skills": ["BindingDB", "ChEMBL"],
                    }
                ],
                "entities": {"drug": ["imatinib"]},
                "subquestions": ["What are the established direct targets of imatinib?"],
                "preferred_skills": ["BindingDB", "ChEMBL"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Single direct-target plan."],
            }
        )
    ).plan("What does imatinib target?")

    assert len(plan.execution_tasks) == 1
    assert plan.execution_tasks[0].task_id == "primary"
    assert plan.execution_tasks[0].task_type == "direct_targets"


def test_planner_ignores_extra_answer_contract_fields_in_v2_payload() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "single_task",
                "question_type": "target_lookup",
                "primary_task": "direct_targets",
                "answer_contract": {
                    "summary_style": "direct_answer_first",
                    "section_order": ["summary", "direct_targets", "limitations"],
                    "expected_output": "plain_english_answer",
                },
                "entities": {"drug": ["imatinib"]},
                "subquestions": ["What are the established direct targets of imatinib?"],
                "preferred_skills": ["BindingDB", "ChEMBL"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Single direct-target plan."],
            }
        )
    ).plan("What does imatinib target?")

    assert plan.answer_contract is not None
    assert plan.answer_contract.section_order == [
        "summary",
        "direct_targets",
        "limitations",
    ]


def test_planner_enriches_v2_tasks_with_task_aware_knowhow_hints() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "plan_type": "composite_query",
                "primary_task": {
                    "task_type": "direct_targets",
                    "question": "What are the established direct targets of imatinib?",
                    "entities": {"drug": ["imatinib"]},
                    "preferred_skills": ["BindingDB", "ChEMBL"],
                    "preferred_evidence_types": ["database_record"],
                    "requires_graph_reasoning": False,
                    "requires_prediction_sources": False,
                    "requires_web_fallback": False,
                    "answer_risk_level": "medium",
                    "notes": ["Use direct binding resources first."],
                },
                "supporting_tasks": [
                    {
                        "task_type": "mechanism_of_action",
                        "question": "What is the mechanism of action of imatinib?",
                        "entities": {"drug": ["imatinib"]},
                        "preferred_skills": ["Open Targets Platform", "DRUGMECHDB"],
                        "preferred_evidence_types": ["database_record"],
                        "requires_graph_reasoning": False,
                        "requires_prediction_sources": False,
                        "requires_web_fallback": False,
                        "answer_risk_level": "medium",
                        "notes": ["Add mechanism context after direct targets."],
                    }
                ],
                "answer_contract": {
                    "summary_style": "direct_answer_first",
                    "section_order": [
                        "summary",
                        "direct_targets",
                        "mechanism_of_action",
                        "limitations",
                    ],
                },
                "entities": {"drug": ["imatinib"]},
                "subquestions": [
                    "What are the established direct targets of imatinib?",
                    "What is the mechanism of action of imatinib?",
                ],
                "preferred_skills": [
                    "BindingDB",
                    "ChEMBL",
                    "Open Targets Platform",
                    "DRUGMECHDB",
                ],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Composite plan."],
            }
        ),
        knowhow_registry=_make_knowhow_registry(),
    ).plan("What are the known drug targets and mechanism of action of imatinib?")

    assert plan.primary_task is not None
    assert plan.primary_task.knowhow_doc_ids == ["direct_targets_grounding"]
    assert plan.primary_task.knowhow_hints[0]["task_id"] == "primary"
    assert [task.task_type for task in plan.supporting_tasks] == ["mechanism_of_action"]
    assert plan.supporting_tasks[0].knowhow_doc_ids == ["mechanism_explanation"]
    assert plan.supporting_tasks[0].knowhow_hints[0]["task_id"] == "support_1"
    assert plan.knowhow_doc_ids == [
        "direct_targets_grounding",
        "mechanism_explanation",
    ]
    assert plan.question_type == "mechanism"


def test_direct_target_lookup_recognizes_does_target_question() -> None:
    assert is_direct_target_lookup(query="What does imatinib target?") is True


def test_direct_target_lookup_recognizes_parenthetical_mixed_same_drug_question() -> None:
    assert is_direct_target_lookup(
        query="What does Gleevec (imatinib, CHEMBL941) target?"
    ) is True


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
        "DailyMed",
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
        "DailyMed",
        "openFDA Human Drug",
    ]
    assert plan.requires_graph_reasoning is False


def test_planner_enforces_phase_2a_repurposing_bundle_for_canonical_question_type() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_repurposing",
                "entities": {"drug": ["metformin"]},
                "subquestions": ["What are the approved indications and repurposing evidence of metformin?"],
                "preferred_skills": [
                    "Open Targets Platform",
                    "DRUGMECHDB",
                    "DailyMed",
                    "RepoDB",
                    "DrugCentral",
                ],
                "preferred_evidence_types": ["database_record"],
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
        "DailyMed",
        "openFDA Human Drug",
    ]
    assert plan.requires_graph_reasoning is False


def test_planner_enforces_phase_2a_pgx_bundle_for_canonical_question_type() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "pharmacogenomics",
                "entities": {"drug": ["clopidogrel"]},
                "subquestions": ["What pharmacogenomic factors affect clopidogrel efficacy and safety?"],
                "preferred_skills": ["FAERS", "PharmGKB", "CPIC"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Prefer PGx resources."],
            }
        )
    ).plan("What pharmacogenomic factors affect clopidogrel efficacy and safety?")

    assert plan.question_type == "pharmacogenomics"
    assert plan.preferred_skills == ["PharmGKB", "CPIC"]
    assert plan.requires_graph_reasoning is False


def test_planner_builds_composite_adr_and_labeling_plan_for_major_safety_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_safety_adverse_reactions",
                "entities": {"drug": ["clozapine"]},
                "subquestions": [
                    "What are the major safety risks and serious adverse reactions of clozapine?"
                ],
                "preferred_skills": ["ADReCS", "FAERS", "nSIDES", "SIDER"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": True,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Prefer ADR resources."],
            }
        )
    ).plan("What are the major safety risks and serious adverse reactions of clozapine?")

    assert plan.plan_type == "composite_query"
    assert plan.question_type == "adr"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "major_adrs"
    assert [task.task_type for task in plan.supporting_tasks] == ["labeling_summary"]
    assert "openFDA Human Drug" in plan.supporting_tasks[0].preferred_skills


def test_planner_prunes_irrelevant_ddi_support_from_pgx_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "pharmacogenomics",
                "entities": {"drug": ["clopidogrel"]},
                "subquestions": ["What pharmacogenomic factors affect clopidogrel efficacy and safety?"],
                "preferred_skills": ["PharmGKB", "CPIC", "KEGG Drug"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner attached an unnecessary DDI support task."],
                "plan_type": "composite_query",
                "primary_task": {"task_type": "pgx_guidance"},
                "supporting_tasks": [{"task_type": "clinically_relevant_ddi"}],
            }
        )
    ).plan("What pharmacogenomic factors affect clopidogrel efficacy and safety?")

    assert plan.plan_type == "single_task"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "pgx_guidance"
    assert plan.supporting_tasks == []


def test_planner_prunes_irrelevant_mechanism_support_from_ddi_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "ddi_mechanism",
                "entities": {"drug": ["warfarin"]},
                "subquestions": ["What are the clinically important drug-drug interactions of warfarin and their mechanisms?"],
                "preferred_skills": ["DDInter", "KEGG Drug", "Open Targets Platform"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner attached a target-MOA support task that does not answer the DDI question."],
                "plan_type": "composite_query",
                "primary_task": {"task_type": "ddi_mechanism"},
                "supporting_tasks": [
                    {"task_type": "clinically_relevant_ddi"},
                    {"task_type": "mechanism_of_action"},
                ],
            }
        )
    ).plan("What are the clinically important drug-drug interactions of warfarin and their mechanisms?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "ddi_mechanism"
    assert [task.task_type for task in plan.supporting_tasks] == ["clinically_relevant_ddi"]


def test_planner_backfills_fallback_labeling_support_for_serious_adr_v2_plans() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "adr",
                "entities": {"drug": ["clozapine"]},
                "subquestions": [
                    "What are the major safety risks and serious adverse reactions of clozapine?"
                ],
                "preferred_skills": ["openFDA Human Drug", "FAERS"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner emitted single-task v2 ADR plan."],
                "plan_type": "single_task",
                "primary_task": {
                    "task_type": "major_adrs",
                    "preferred_skills": ["openFDA Human Drug", "FAERS"],
                },
                "supporting_tasks": [],
            }
        )
    ).plan("What are the major safety risks and serious adverse reactions of clozapine?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "major_adrs"
    assert [task.task_type for task in plan.supporting_tasks] == ["labeling_summary"]


def test_planner_drops_irrelevant_ddi_support_for_pgx_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "pharmacogenomics",
                "entities": {"drug": ["clopidogrel"]},
                "subquestions": ["What pharmacogenomic factors affect clopidogrel efficacy and safety?"],
                "preferred_skills": ["PharmGKB", "CPIC", "KEGG Drug"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner over-expanded the query."],
                "plan_type": "composite_query",
                "primary_task": {"task_type": "pgx_guidance"},
                "supporting_tasks": [{"task_type": "clinically_relevant_ddi"}],
            }
        )
    ).plan("What pharmacogenomic factors affect clopidogrel efficacy and safety?")

    assert plan.plan_type == "single_task"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "pgx_guidance"
    assert plan.supporting_tasks == []


def test_planner_drops_unrelated_mechanism_support_for_ddi_queries() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "ddi_mechanism",
                "entities": {"drug": ["warfarin"]},
                "subquestions": [
                    "What are the clinically important drug-drug interactions of warfarin and their mechanisms?"
                ],
                "preferred_skills": ["DDInter", "KEGG Drug", "Open Targets Platform"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner added a generic mechanism task."],
                "plan_type": "composite_query",
                "primary_task": {"task_type": "ddi_mechanism"},
                "supporting_tasks": [
                    {"task_type": "clinically_relevant_ddi"},
                    {"task_type": "mechanism_of_action"},
                ],
            }
        )
    ).plan("What are the clinically important drug-drug interactions of warfarin and their mechanisms?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "ddi_mechanism"
    assert [task.task_type for task in plan.supporting_tasks] == ["clinically_relevant_ddi"]


def test_planner_preserves_labeling_support_for_repurposing_queries_with_approved_indications() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "drug_repurposing",
                "entities": {"drug": ["metformin"]},
                "subquestions": ["What are the approved indications and repurposing evidence of metformin?"],
                "preferred_skills": ["DrugRepoBank", "RepurposeDrugs", "openFDA Human Drug"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "high",
                "notes": ["Planner kept useful label support for approved indications."],
                "plan_type": "composite_query",
                "primary_task": {"task_type": "repurposing_evidence"},
                "supporting_tasks": [{"task_type": "labeling_summary"}],
            }
        )
    ).plan("What are the approved indications and repurposing evidence of metformin?")

    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "repurposing_evidence"
    assert [task.task_type for task in plan.supporting_tasks] == ["labeling_summary"]


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
    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert [task.task_type for task in plan.supporting_tasks] == [
        "mechanism_of_action"
    ]
    assert "BindingDB" in plan.preferred_skills
    assert "DRUGMECHDB" in plan.preferred_skills
    assert plan.requires_graph_reasoning is False


def test_planner_promotes_legacy_mechanism_payload_to_composite_plan() -> None:
    plan = PlannerAgent(
        _LLMStub(
            {
                "question_type": "mechanism",
                "entities": {"drug": ["imatinib"]},
                "subquestions": [
                    "What are the known drug targets and mechanism of action of imatinib?"
                ],
                "preferred_skills": ["Open Targets Platform", "DRUGMECHDB"],
                "preferred_evidence_types": ["database_record"],
                "requires_graph_reasoning": False,
                "requires_prediction_sources": False,
                "requires_web_fallback": False,
                "answer_risk_level": "medium",
                "notes": ["Legacy mechanism-only planner output."],
            }
        )
    ).plan("What are the known drug targets and mechanism of action of imatinib?")

    assert plan.question_type == "mechanism"
    assert plan.plan_type == "composite_query"
    assert plan.primary_task is not None
    assert plan.primary_task.task_type == "direct_targets"
    assert [task.task_type for task in plan.supporting_tasks] == [
        "mechanism_of_action"
    ]
    assert plan.subquestions == [
        "What are the established direct targets of imatinib?",
        "What is the mechanism of action of imatinib?",
    ]
    assert "BindingDB" in plan.preferred_skills
    assert "DRUGMECHDB" in plan.preferred_skills
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
    assert "- Open Targets Platform" in prompt
    assert "- Molecular Targets" not in prompt
    assert "- DRUGMECHDB" not in prompt
