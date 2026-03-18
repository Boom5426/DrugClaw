from __future__ import annotations

import drugclaw.main_system as main_system_module
from drugclaw.agent_retriever import RetrieverAgent
from drugclaw.models import AgentState, ThinkingMode
from drugclaw.query_plan import QueryPlan


class _PlannerBypassLLMStub:
    def generate_json(self, messages, temperature=0.3):
        raise AssertionError("Retriever should consume state.query_plan instead of re-planning")


class _CoderStub:
    def __init__(self):
        self.last_entities = None
        self.last_skill_names = None
        self.last_execution_strategy = None

    def generate_and_execute(
        self,
        skill_names,
        entities,
        query,
        max_results_per_skill=30,
        execution_strategy="auto",
    ):
        self.last_entities = entities
        self.last_skill_names = skill_names
        self.last_execution_strategy = execution_strategy
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


class _SelectiveRegistryStub(_RegistryStub):
    def __init__(self, valid_skills):
        self.valid_skills = list(valid_skills)
        self._valid_skill_set = set(valid_skills)

    def get_skills_for_query(self, query):
        return list(self.valid_skills)

    def get_skill(self, skill_name):
        if skill_name in self._valid_skill_set:
            return object()
        return None


class _AvailabilitySkillStub:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self):
        return self._available


class _AvailabilityRegistryStub(_RegistryStub):
    def __init__(self, availability):
        self.availability = dict(availability)

    def get_skills_for_query(self, query):
        return list(self.availability)

    def get_skill(self, skill_name):
        if skill_name not in self.availability:
            return None
        return _AvailabilitySkillStub(self.availability[skill_name])


class _ResourceRegistryStub:
    def __init__(self, entries):
        self.entries = {entry.name: entry for entry in entries}

    def get_resource(self, name):
        return self.entries.get(name)


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
        _SelectiveRegistryStub(["BindingDB", "ChEMBL"]),
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
        _SelectiveRegistryStub(["DrugBank", "BindingDB"]),
        coder_agent=_CoderStub(),
    )

    updated = retriever.execute(state)

    assert "DrugBank" in updated.retrieved_text
    assert "BindingDB" not in updated.retrieved_text


def test_retriever_falls_back_to_registry_skills_when_plan_hints_are_invalid() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What does imatinib target?",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=[
                "drug_target_retrieval",
                "pharmacological_profiling",
                "biochemical_pathway_analysis",
            ],
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
        _SelectiveRegistryStub(["BindingDB", "ChEMBL"]),
        coder_agent=coder,
    )

    updated = retriever.execute(state)

    assert "BindingDB" in updated.retrieved_text or "ChEMBL" in updated.retrieved_text
    assert "drug_target_retrieval" not in updated.retrieved_text
    assert coder.last_skill_names == ["BindingDB", "ChEMBL"]


def test_retriever_normalizes_singular_entity_keys_from_query_plan() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What does imatinib target?",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"], "gene": ["ABL1"]},
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
        _SelectiveRegistryStub(["BindingDB"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_entities == {"drug": ["imatinib"], "gene": ["ABL1"]}


def test_retriever_filters_unavailable_skills_from_query_plan() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What does imatinib target?",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["TTD", "BindingDB"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _AvailabilityRegistryStub({"TTD": False, "BindingDB": True}),
        coder_agent=coder,
    )

    updated = retriever.execute(state)

    assert "BindingDB" in updated.retrieved_text
    assert "TTD" not in updated.retrieved_text
    assert coder.last_skill_names == ["BindingDB"]


def test_retriever_prefers_direct_retrieve_for_simple_target_lookup() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        thinking_mode="simple",
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
            notes=["Prefer direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _SelectiveRegistryStub(["BindingDB"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_prefers_direct_retrieve_for_simple_drug_target_interaction() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        thinking_mode="simple",
        query_plan=QueryPlan(
            question_type="Drug-Target Interaction",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["ChEMBL"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _SelectiveRegistryStub(["ChEMBL"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_prefers_direct_retrieve_for_simple_drug_target_identification() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        thinking_mode="simple",
        query_plan=QueryPlan(
            question_type="drug_target_identification",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["ChEMBL"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _SelectiveRegistryStub(["ChEMBL"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_prefers_direct_retrieve_for_simple_relationship_retrieval_targets() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        thinking_mode="simple",
        query_plan=QueryPlan(
            question_type="Relationship Retrieval",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["ChEMBL"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _SelectiveRegistryStub(["ChEMBL"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_resource_filter_infers_entities_for_label_query() -> None:
    class _EntityExtractionFailLLM:
        def generate_json(self, messages, temperature=0.3):
            raise ValueError("entity extraction unavailable")

    coder = _CoderStub()
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        thinking_mode="simple",
        resource_filter=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
    )

    retriever = RetrieverAgent(
        _EntityExtractionFailLLM(),
        _SelectiveRegistryStub(["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"]),
        coder_agent=coder,
    )

    updated = retriever.execute(state)

    assert updated.current_query_entities == {"drug": ["metformin"]}
    assert coder.last_entities == {"drug": ["metformin"]}


def test_retriever_resource_filter_uses_direct_retrieve_for_simple_label_query() -> None:
    class _EntityExtractionFailLLM:
        def generate_json(self, messages, temperature=0.3):
            raise ValueError("entity extraction unavailable")

    coder = _CoderStub()
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        thinking_mode="simple",
        resource_filter=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
    )

    retriever = RetrieverAgent(
        _EntityExtractionFailLLM(),
        _SelectiveRegistryStub(["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"]),
        coder_agent=coder,
    )

    retriever.execute(state)

    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_resource_filter_reuses_fallback_query_plan_entities_when_llm_returns_empty() -> None:
    class _EmptyEntityLLM:
        def generate_json(self, messages, temperature=0.3):
            return {"drugs": [], "genes": [], "diseases": [], "pathways": []}

    coder = _CoderStub()
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        thinking_mode="simple",
        resource_filter=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
    )

    retriever = RetrieverAgent(
        _EmptyEntityLLM(),
        _SelectiveRegistryStub(["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"]),
        coder_agent=coder,
    )

    updated = retriever.execute(state)

    assert updated.query_plan is not None
    assert updated.query_plan.entities == {"drug": ["metformin"]}
    assert updated.current_query_entities == {"drug": ["metformin"]}
    assert coder.last_entities == {"drug": ["metformin"]}
    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_enriches_empty_planner_query_plan_when_resource_filter_is_present() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What prescribing and safety information is available for metformin?",
        thinking_mode="simple",
        resource_filter=["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
        query_plan=QueryPlan(
            question_type="unknown",
            entities={},
            subquestions=["What prescribing and safety information is available for metformin?"],
            preferred_skills=[],
            preferred_evidence_types=[],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Fallback plan used because planner output was unavailable or invalid."],
        ),
    )

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _SelectiveRegistryStub(["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"]),
        coder_agent=coder,
    )

    updated = retriever.execute(state)

    assert updated.query_plan is not None
    assert updated.query_plan.question_type == "labeling"
    assert updated.query_plan.entities == {"drug": ["metformin"]}
    assert updated.current_query_entities == {"drug": ["metformin"]}
    assert coder.last_entities == {"drug": ["metformin"]}
    assert coder.last_execution_strategy == "direct_retrieve"


def test_retriever_prioritizes_ready_remote_skills_for_fallback() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["TTD", "TarKG"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    class _SkillWithAccess(_AvailabilitySkillStub):
        def __init__(self, available: bool, access_mode: str):
            super().__init__(available)
            self.access_mode = access_mode

    class _RegistryWithAccess(_RegistryStub):
        def __init__(self):
            self.skills = {
                "TTD": _SkillWithAccess(True, "LOCAL_FILE"),
                "TarKG": _SkillWithAccess(True, "LOCAL_FILE"),
                "ChEMBL": _SkillWithAccess(True, "CLI"),
                "Open Targets Platform": _SkillWithAccess(True, "REST_API"),
                "DGIdb": _SkillWithAccess(True, "REST_API"),
            }

        def get_skills_for_query(self, query):
            return ["TTD", "ChEMBL", "TarKG", "Open Targets Platform", "DGIdb"]

        def get_skill(self, skill_name):
            return self.skills.get(skill_name)

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _RegistryWithAccess(),
        coder_agent=coder,
        resource_registry=_ResourceRegistryStub(
            [
                type("Entry", (), {"name": "TTD", "status": "ready", "access_mode": "LOCAL_FILE"})(),
                type("Entry", (), {"name": "TarKG", "status": "ready", "access_mode": "LOCAL_FILE"})(),
                type("Entry", (), {"name": "ChEMBL", "status": "ready", "access_mode": "CLI"})(),
                type("Entry", (), {"name": "Open Targets Platform", "status": "ready", "access_mode": "REST_API"})(),
                type("Entry", (), {"name": "DGIdb", "status": "ready", "access_mode": "REST_API"})(),
            ]
        ),
    )

    updated = retriever.execute(state)

    assert "ChEMBL" in updated.retrieved_text
    assert "Open Targets Platform" in updated.retrieved_text
    assert "DGIdb" in updated.retrieved_text
    assert "TTD" not in updated.retrieved_text
    assert "TarKG" not in updated.retrieved_text
    assert coder.last_skill_names == ["ChEMBL", "DGIdb", "Open Targets Platform"]


def test_retriever_narrows_target_lookup_to_primary_dti_sources() -> None:
    coder = _CoderStub()
    state = AgentState(
        original_query="What are the known drug targets of imatinib?",
        thinking_mode="simple",
        query_plan=QueryPlan(
            question_type="target_lookup",
            entities={"drug": ["imatinib"]},
            subquestions=["What are the known targets of imatinib?"],
            preferred_skills=["ChEMBL", "Molecular Targets", "DRUGMECHDB"],
            preferred_evidence_types=["database_record"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="medium",
            notes=["Prefer direct target databases."],
        ),
    )

    class _TargetSkillStub(_AvailabilitySkillStub):
        def __init__(self, available: bool, access_mode: str):
            super().__init__(available)
            self.access_mode = access_mode

    class _TargetLookupRegistry(_RegistryStub):
        def __init__(self):
            self.skills = {
                "ChEMBL": _TargetSkillStub(True, "CLI"),
                "BindingDB": _TargetSkillStub(True, "REST_API"),
                "DGIdb": _TargetSkillStub(True, "REST_API"),
                "Open Targets Platform": _TargetSkillStub(True, "REST_API"),
                "Molecular Targets": _TargetSkillStub(True, "REST_API"),
                "DRUGMECHDB": _TargetSkillStub(True, "REST_API"),
            }

        def get_skills_for_query(self, query):
            return [
                "ChEMBL",
                "Molecular Targets",
                "DRUGMECHDB",
                "BindingDB",
                "DGIdb",
                "Open Targets Platform",
            ]

        def get_skill(self, skill_name):
            return self.skills.get(skill_name)

    retriever = RetrieverAgent(
        _PlannerBypassLLMStub(),
        _TargetLookupRegistry(),
        coder_agent=coder,
        resource_registry=_ResourceRegistryStub(
            [
                type("Entry", (), {"name": "ChEMBL", "status": "ready", "access_mode": "CLI"})(),
                type("Entry", (), {"name": "BindingDB", "status": "ready", "access_mode": "REST_API"})(),
                type("Entry", (), {"name": "DGIdb", "status": "ready", "access_mode": "REST_API"})(),
                type("Entry", (), {"name": "Open Targets Platform", "status": "ready", "access_mode": "REST_API"})(),
                type("Entry", (), {"name": "Molecular Targets", "status": "ready", "access_mode": "REST_API"})(),
                type("Entry", (), {"name": "DRUGMECHDB", "status": "ready", "access_mode": "REST_API"})(),
            ]
        ),
    )

    updated = retriever.execute(state)

    assert "BindingDB" in updated.retrieved_text
    assert "ChEMBL" in updated.retrieved_text
    assert "DGIdb" in updated.retrieved_text
    assert "Molecular Targets" not in updated.retrieved_text
    assert "DRUGMECHDB" not in updated.retrieved_text
    assert coder.last_skill_names == ["BindingDB", "ChEMBL", "DGIdb"]


class _NoOpAgent:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, state):
        return state


class _RetrieverNodeStub(_NoOpAgent):
    def execute(self, state):
        state.retrieved_text = "retrieved"
        return state


class _VerboseRetrieverNodeStub(_NoOpAgent):
    def execute(self, state):
        print("[Retriever Agent] noisy retriever log")
        state.retrieved_text = "retrieved"
        return state


class _ResponderNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_answer = "answer"
        return state

    def execute_simple(self, state):
        state.current_answer = "answer"
        return state


class _VerboseResponderNodeStub(_NoOpAgent):
    def execute(self, state):
        print("[Responder Agent] noisy responder log")
        state.current_answer = "answer"
        return state

    def execute_simple(self, state):
        print("[Responder Agent] noisy simple responder log")
        state.current_answer = "answer"
        return state


class _ModeAwareResponderNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_answer = "graph-answer"
        return state

    def execute_simple(self, state):
        state.current_answer = "simple-answer"
        return state


class _ReflectContinueNodeStub(_NoOpAgent):
    def execute(self, state):
        state.should_continue = True
        state.max_iterations_reached = False
        state.evidence_sufficient = False
        state.current_reward = 0.4
        state.reflection_feedback = "Need supporting web evidence."
        return state


class _ReflectStopNodeStub(_NoOpAgent):
    def execute(self, state):
        state.should_continue = False
        state.max_iterations_reached = False
        state.evidence_sufficient = True
        state.current_reward = 0.8
        state.reflection_feedback = "Current evidence is sufficient."
        return state


class _WebSearchNodeStub(_NoOpAgent):
    def execute(self, state):
        state.current_answer += "\nweb-evidence"
        return state

    def execute_direct(self, state):
        state.current_answer = "web-only-answer"
        return state


class _WebSkillStub:
    pass


class _RuntimeRegistryStub:
    def get_skill(self, name):
        if name == "WebSearch":
            return _WebSkillStub()
        return None


class _PlannerNodeStub:
    def __init__(self, *args, **kwargs):
        pass

    def plan(self, query, omics_constraints=None):
        return QueryPlan(
            question_type="labeling",
            entities={"drug": ["metformin"]},
            subquestions=["What prescribing information is available for metformin?"],
            preferred_skills=["DailyMed"],
            preferred_evidence_types=["label_text"],
            requires_graph_reasoning=False,
            requires_prediction_sources=False,
            requires_web_fallback=False,
            answer_risk_level="high",
            notes=["Direct label lookup; graph reasoning is unnecessary."],
        )


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


def test_graph_mode_skips_graph_when_plan_and_evidence_do_not_require_it(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "PlannerAgent", _PlannerNodeStub)
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
        "What prescribing and safety information is available for metformin?",
        thinking_mode="graph",
    )

    assert result["success"] is True
    assert "OPTIONAL_GRAPH:skipped" in result["execution_trace"]


def test_query_accepts_thinking_mode_enum_for_simple(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ModeAwareResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _WebSearchNodeStub)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query("What does imatinib target?", thinking_mode=ThinkingMode.SIMPLE)

    assert result["success"] is True
    assert result["mode"] == "simple"
    assert result["answer"] == "simple-answer"


def test_query_accepts_thinking_mode_enum_for_web_only(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ModeAwareResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _WebSearchNodeStub)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query("What does imatinib target?", thinking_mode=ThinkingMode.WEB_ONLY)

    assert result["success"] is True
    assert result["mode"] == "web_only"
    assert result["answer"] == "web-only-answer"


def test_query_accepts_thinking_mode_enum_for_graph(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "PlannerAgent", _PlannerNodeStub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ModeAwareResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _ReflectStopNodeStub)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _WebSearchNodeStub)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query(
        "What prescribing and safety information is available for metformin?",
        thinking_mode=ThinkingMode.GRAPH,
    )

    assert result["success"] is True
    assert result["mode"] == "graph"
    assert result["answer"] == "graph-answer"
    assert "REFLECT" in result["execution_trace"]
    assert "WEB_SEARCH" not in result["execution_trace"]


def test_graph_mode_records_reflect_and_web_search_when_reflection_requests_fallback(monkeypatch) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "PlannerAgent", _PlannerNodeStub)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ModeAwareResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _ReflectContinueNodeStub)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _WebSearchNodeStub)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query(
        "What prescribing and safety information is available for metformin?",
        thinking_mode="graph",
    )

    assert result["success"] is True
    assert "REFLECT" in result["execution_trace"]
    assert "WEB_SEARCH" in result["execution_trace"]
    assert result["answer"].endswith("web-evidence")


def test_query_verbose_false_suppresses_agent_logs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())
    monkeypatch.setattr(main_system_module, "build_default_registry", lambda config: _RuntimeRegistryStub())
    monkeypatch.setattr(main_system_module, "build_resource_registry", lambda registry: object())
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _VerboseRetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _VerboseResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    system = main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    result = system.query("What does imatinib target?", thinking_mode="simple", verbose=False)
    captured = capsys.readouterr()

    assert result["success"] is True
    assert "[Retriever Agent]" not in captured.out
    assert "[Responder Agent]" not in captured.out


def test_system_init_suppresses_registry_startup_noise(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_system_module, "LLMClient", lambda config: object())

    def _noisy_build_default_registry(config):
        print("TTDSkill: file not found — set config['drug_target_tsv']")
        return _RuntimeRegistryStub()

    def _noisy_build_resource_registry(registry):
        print("TarKGSkill: file not found — set config['tsv_path']")
        return object()

    monkeypatch.setattr(main_system_module, "build_default_registry", _noisy_build_default_registry)
    monkeypatch.setattr(main_system_module, "build_resource_registry", _noisy_build_resource_registry)
    monkeypatch.setattr(main_system_module, "CoderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RetrieverAgent", _RetrieverNodeStub)
    monkeypatch.setattr(main_system_module, "GraphBuilderAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "RerankerAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "ResponderAgent", _ResponderNodeStub)
    monkeypatch.setattr(main_system_module, "ReflectorAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "WebSearchAgent", _NoOpAgent)
    monkeypatch.setattr(main_system_module, "wrap_answer_card", lambda answer, result: answer)

    main_system_module.DrugClawSystem(config=object(), enable_logging=False)
    captured = capsys.readouterr()

    assert "file not found" not in captured.out
