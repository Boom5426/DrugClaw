from __future__ import annotations

import re

import drugclaw.main_system as main_system_module
from drugclaw.models import AgentState
from drugclaw.query_plan import QueryPlan


def test_agent_state_supports_input_resolution_fields() -> None:
    state = AgentState(
        original_query="What does Gleevec target?",
        normalized_query="What does imatinib target?",
        resolved_entities={"drug": ["imatinib"]},
        input_resolution={
            "status": "resolved",
            "canonical_drug_names": ["imatinib"],
        },
    )

    assert state.normalized_query == "What does imatinib target?"
    assert state.resolved_entities == {"drug": ["imatinib"]}
    assert state.input_resolution["status"] == "resolved"


class _NoOpAgent:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, state):
        return state


class _PlannerCaptureStub:
    def __init__(self):
        self.calls: list[str] = []

    def plan(self, query, omics_constraints=None):
        self.calls.append(query)
        match = re.search(r"does\s+([a-z0-9\-]+)\s+target", query.lower())
        drug_name = match.group(1) if match else "unknown"
        return QueryPlan(
            question_type="target_lookup",
            entities={"drug": [drug_name]},
            subquestions=[query],
            preferred_skills=["BindingDB"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Direct target lookup."],
        )


class _PlannerMustNotRunStub:
    def plan(self, query, omics_constraints=None):
        raise AssertionError("resource_filter queries should bypass PlannerAgent")


class _RetrieverNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_query_entities = dict(state.query_plan.entities)
        state.retrieved_text = f"retrieved for {state.current_query_entities}"
        return state


class _ResponderNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_answer = "answer"
        return state

    def execute_simple(self, state):
        state.current_answer = "answer"
        return state


class _RuntimeRegistryStub:
    def get_skill(self, name):
        return None


class _StructuredResolverStub:
    @classmethod
    def default(cls, config):
        return cls()

    def resolve_query(self, query):
        if "CHEMBL941" in query:
            return {
                "original_query": query,
                "normalized_query": query.replace("CHEMBL941", "imatinib"),
                "status": "resolved",
                "detected_identifiers": [
                    {
                        "type": "chembl_id",
                        "raw_text": "CHEMBL941",
                        "normalized_value": "CHEMBL941",
                    }
                ],
                "resolved_records": [
                    {
                        "identifier_type": "chembl_id",
                        "identifier_value": "CHEMBL941",
                        "canonical_name": "imatinib",
                        "source": "stub",
                        "status": "resolved",
                    }
                ],
                "errors": [],
            }

        return {
            "original_query": query,
            "normalized_query": query,
            "status": "unresolved",
            "detected_identifiers": [],
            "resolved_records": [],
            "errors": [],
        }


class _StructuredResolverUnknownCanonicalStub:
    @classmethod
    def default(cls, config):
        return cls()

    def resolve_query(self, query):
        return {
            "original_query": query,
            "normalized_query": query.replace("CHEMBL1234", "dasatinib"),
            "status": "resolved",
            "detected_identifiers": [
                {
                    "type": "chembl_id",
                    "raw_text": "CHEMBL1234",
                    "normalized_value": "CHEMBL1234",
                }
            ],
            "resolved_records": [
                {
                    "identifier_type": "chembl_id",
                    "identifier_value": "CHEMBL1234",
                    "canonical_drug_name": "dasatinib",
                    "source": "stub",
                    "status": "resolved",
                }
            ],
            "errors": [],
        }


class _StructuredResolverMixedSameDrugStub:
    @classmethod
    def default(cls, config):
        return cls()

    def resolve_query(self, query):
        if "CHEMBL941" not in query:
            return {
                "original_query": query,
                "normalized_query": query,
                "status": "unresolved",
                "detected_identifiers": [],
                "resolved_records": [],
                "errors": [],
                "drug_mentions": [],
                "rewrite_applied": False,
            }

        return {
            "original_query": query,
            "normalized_query": query.replace("CHEMBL941", "imatinib"),
            "status": "resolved",
            "detected_identifiers": [
                {
                    "type": "chembl_id",
                    "raw_text": "CHEMBL941",
                    "normalized_value": "CHEMBL941",
                }
            ],
            "resolved_records": [
                {
                    "identifier_type": "chembl_id",
                    "identifier_value": "CHEMBL941",
                    "canonical_name": "imatinib",
                    "source": "stub",
                    "status": "resolved",
                }
            ],
            "errors": [],
            "drug_mentions": [
                {
                    "raw_text": "CHEMBL941",
                    "mention_type": "chembl_id",
                    "normalized_value": "CHEMBL941",
                    "canonical_drug_name": "imatinib",
                    "resolution_stage": "identifier",
                    "source": "stub",
                }
            ],
            "rewrite_applied": True,
        }


def test_query_normalizes_alias_before_planning_and_exposes_resolution(monkeypatch) -> None:
    planner_stub = _PlannerCaptureStub()

    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "StructuredInputResolver", _StructuredResolverStub, raising=False)
    monkeypatch.setattr(main_system_module, "PlannerAgent", lambda *args, **kwargs: planner_stub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query("What does Gleevec target?", thinking_mode="simple")

    assert planner_stub.calls == ["What does imatinib target?"]
    assert result["query"] == "What does Gleevec target?"
    assert result["normalized_query"] == "What does imatinib target?"
    assert result["resolved_entities"] == {"drug": ["imatinib"]}
    assert result["input_resolution"]["status"] == "resolved"
    assert result["input_resolution"]["canonical_drug_names"] == ["imatinib"]
    assert result["query_plan"]["entities"] == {"drug": ["imatinib"]}


def test_query_with_resource_filter_bypasses_planner_and_builds_deterministic_plan(monkeypatch) -> None:
    class _DeterministicRetrieverNodeStub(_NoOpAgent):
        def execute(self, state):
            state.query_plan = QueryPlan(
                question_type="target_lookup",
                entities={"drug": ["imatinib"]},
                subquestions=[state.normalized_query or state.original_query],
                preferred_skills=["ChEMBL", "DGIdb", "Open Targets Platform"],
                preferred_evidence_types=[],
                requires_graph_reasoning=False,
                requires_prediction_sources=False,
                requires_web_fallback=False,
                answer_risk_level="medium",
                notes=["resource_filter active: using deterministic plan"],
            )
            state.current_query_entities = {"drug": ["imatinib"]}
            state.retrieved_text = "retrieved for canonical imatinib"
            return state

    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "StructuredInputResolver", _StructuredResolverStub, raising=False)
    monkeypatch.setattr(main_system_module, "PlannerAgent", lambda *args, **kwargs: _PlannerMustNotRunStub())
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _DeterministicRetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query(
        "What are the known drug targets of Gleevec?",
        thinking_mode="simple",
        resource_filter=["ChEMBL", "DGIdb", "Open Targets Platform"],
    )

    assert result["success"] is True
    assert result["normalized_query"] == "What are the known drug targets of imatinib?"
    assert result["query_plan"]["question_type"] == "target_lookup"
    assert result["query_plan"]["entities"] == {"drug": ["imatinib"]}
    assert result["query_plan"]["preferred_skills"] == [
        "ChEMBL",
        "DGIdb",
        "Open Targets Platform",
    ]


def test_query_resolves_chembl_identifier_before_planning(monkeypatch) -> None:
    planner_stub = _PlannerCaptureStub()

    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "StructuredInputResolver", _StructuredResolverStub, raising=False)
    monkeypatch.setattr(main_system_module, "PlannerAgent", lambda *args, **kwargs: planner_stub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query(
        "What are the known drug targets of CHEMBL941?",
        thinking_mode="simple",
    )

    assert planner_stub.calls == ["What are the known drug targets of imatinib?"]
    assert result["normalized_query"] == "What are the known drug targets of imatinib?"
    assert result["resolved_entities"] == {"drug": ["imatinib"]}
    assert result["input_resolution"]["status"] == "resolved"
    assert result["input_resolution"]["identifier_resolution"]["status"] == "resolved"
    assert result["input_resolution"]["identifier_resolution"]["resolved_records"][0]["identifier_value"] == "CHEMBL941"


def test_query_keeps_identifier_resolved_canonical_when_alias_seed_does_not_know_it(monkeypatch) -> None:
    planner_stub = _PlannerCaptureStub()

    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(
        main_system_module,
        "StructuredInputResolver",
        _StructuredResolverUnknownCanonicalStub,
        raising=False,
    )
    monkeypatch.setattr(main_system_module, "PlannerAgent", lambda *args, **kwargs: planner_stub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query(
        "What are the known drug targets of CHEMBL1234?",
        thinking_mode="simple",
    )

    assert planner_stub.calls == ["What are the known drug targets of dasatinib?"]
    assert result["normalized_query"] == "What are the known drug targets of dasatinib?"
    assert result["resolved_entities"] == {"drug": ["dasatinib"]}
    assert result["input_resolution"]["canonical_drug_names"] == ["dasatinib"]


def test_query_preserves_original_text_for_mixed_same_drug_mentions(monkeypatch) -> None:
    planner_stub = _PlannerCaptureStub()

    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(
        main_system_module,
        "StructuredInputResolver",
        _StructuredResolverMixedSameDrugStub,
        raising=False,
    )
    monkeypatch.setattr(main_system_module, "PlannerAgent", lambda *args, **kwargs: planner_stub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    query = "What does Gleevec (imatinib, CHEMBL941) target?"
    result = system.query(query, thinking_mode="simple")

    assert result["query"] == query
    assert result["normalized_query"] == query
    assert result["resolved_entities"] == {"drug": ["imatinib"]}
    assert result["input_resolution"]["canonical_drug_names"] == ["imatinib"]
    assert result["input_resolution"]["rewrite_applied"] is False
    assert result["input_resolution"]["drug_mentions"] == [
        {
            "raw_text": "CHEMBL941",
            "mention_type": "chembl_id",
            "normalized_value": "CHEMBL941",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "identifier",
            "source": "stub",
        },
        {
            "raw_text": "Gleevec",
            "mention_type": "alias",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "name",
            "source": "alias_seed",
        },
        {
            "raw_text": "imatinib",
            "mention_type": "canonical_name",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "name",
            "source": "alias_seed",
        },
    ]
    assert result["query_plan"]["entities"] == {"drug": ["imatinib"]}
