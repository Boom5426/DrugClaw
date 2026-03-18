from __future__ import annotations

import drugclaw.main_system as main_system_module
from drugclaw.agent_retriever import RetrieverAgent
from drugclaw.models import AgentState
from drugclaw.query_plan import QueryPlan


class _PlannerBypassLLMStub:
    def generate_json(self, messages, temperature=0.3):
        raise AssertionError("Retriever should consume state.query_plan instead of re-planning")


class _CoderStub:
    def generate_and_execute(self, skill_names, entities, query, max_results_per_skill=30):
        per_skill = {}
        for skill_name in skill_names:
            per_skill[skill_name] = {
                "output": f"Results from {skill_name}",
                "records": [],
                "code": "",
            }
        return {
            "text": "\n".join(f"Results from {name}" for name in skill_names),
            "per_skill": per_skill,
        }


class _RegistryStub:
    def get_skills_for_query(self, query):
        return ["ChEMBL"]

    def get_skill(self, skill_name):
        return None

    @property
    def skill_tree_prompt(self):
        return "stub tree"


def test_retriever_consumes_query_plan_when_present() -> None:
    state = AgentState(
        original_query="What does imatinib target?",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["BindingDB"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Use direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _RegistryStub(),
        coder_agent=_CoderStub(),
    )

    updated = retriever.execute(state)

    assert "BindingDB" in updated.retrieved_text
    assert updated.current_query_entities == {"drug": ["imatinib"]}


def test_retriever_prefers_resource_filter_over_query_plan_hints() -> None:
    state = AgentState(
        original_query="What does imatinib target?",
        resource_filter=["DrugBank"],
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["BindingDB"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Use direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _RegistryStub(),
        coder_agent=_CoderStub(),
    )

    updated = retriever.execute(state)

    assert "DrugBank" in updated.retrieved_text
    assert "BindingDB" not in updated.retrieved_text


class _NoOpAgent:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, state):
        return state


class _RetrieverNodeStub(_NoOpAgent):
    def execute(self, state):
        state.retrieved_text = "retrieved"
        return state


class _ResponderNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_answer = "answer"
        return state

    def execute_simple(self, state):
        state.current_answer = "answer"
        return state


class _WebSkillStub:
    pass


class _RuntimeRegistryStub:
    def get_skill(self, name):
        if name == "WebSearch":
            return _WebSkillStub()
        return None


def test_simple_mode_uses_explicit_stage_trace(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query("What does imatinib target?", thinking_mode="simple")

    assert result["success"] is True
    assert result["execution_trace"] == [
        "PLAN",
        "RETRIEVE",
        "NORMALIZE_EVIDENCE",
        "ASSESS_CLAIMS",
        "ANSWER",
    ]
