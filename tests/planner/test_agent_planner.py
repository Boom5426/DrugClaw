from __future__ import annotations

from drugclaw.agent_planner import PlannerAgent
from drugclaw.query_plan import build_fallback_query_plan


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
    plan = build_fallback_query_plan("What does imatinib target?")

    assert plan.question_type == "unknown"
    assert plan.requires_graph_reasoning is False
    assert plan.preferred_skills == []


def test_planner_classifies_direct_target_lookup_without_graph() -> None:
    plan = PlannerAgent(_LLMStub()).plan("What does imatinib target?")

    assert plan.question_type == "target_lookup"
    assert plan.requires_graph_reasoning is False


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
