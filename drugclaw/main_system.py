"""
Main LangGraph orchestration for DrugClaw — Drug-Specialized Agentic RAG System.

Three thinking modes are supported, selectable per-query:

  GRAPH    (default) — full multi-agent graph reasoning:
                       retrieve → graph_build → rerank → respond → reflect → [web_search] → finalize
  SIMPLE             — one-shot retrieval + direct LLM synthesis (no graph, no loop):
                       retrieve → simple_respond → finalize
  WEB_ONLY           — live web search only, no structured skill retrieval:
                       web_search_direct → finalize

Agent architecture:
  RetrieverAgent   — selects skills, delegates to CoderAgent
  CoderAgent       — writes and executes Python code to query skills
  GraphBuilderAgent— LLM-driven triple extraction (replaces rigid _build_subgraph)
  RerankerAgent    — path scoring and pruning
  ResponderAgent   — answer synthesis
  ReflectorAgent   — evidence sufficiency evaluation
  WebSearchAgent   — PubMed + ClinicalTrials supplementation

Additional query parameters:
  resource_filter : List[str]  — if non-empty, only query the named skills
                                 (bypasses LLM skill selection in the retriever)
  omics_constraints : OmicsConstraints — biological constraints (all modes)
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from .models import AgentState, OmicsConstraints, ReasoningStep, ThinkingMode
from .config import Config
from .llm_client import LLMClient
from .skills import build_default_registry, WebSearchSkill
from .skills.registry import SkillRegistry
from .agent_coder import CoderAgent
from .agent_planner import PlannerAgent
from .agent_retriever import RetrieverAgent
from .agent_graph_builder import GraphBuilderAgent
from .agent_reranker import RerankerAgent
from .agent_responder import ResponderAgent
from .agent_reflector import ReflectorAgent
from .agent_websearch import WebSearchAgent
from .claim_assessment import assess_claims
from .query_logger import QueryLogger, QuerySession
from .response_formatter import wrap_answer_card
from .resource_registry import build_resource_registry


class DrugClawSystem:
    """
    Main orchestration system for drug-specialised agentic RAG.
    Uses LangGraph to manage the multi-agent workflow.

    The RAG retrieval layer is powered by a SkillRegistry plus a resource
    registry summary that expose the current runtime resource set and status.
    Retrieval is handled by a Code Agent that writes custom query code per
    skill, and graph construction is done by an LLM-driven Graph Build Agent.
    """

    def __init__(
        self,
        config: Config,
        enable_logging: bool = True,
        log_dir: str = "./query_logs",
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.config = config

        # Query logger
        self.enable_logging = enable_logging
        self.logger = QueryLogger(log_dir) if enable_logging else None

        # LLM
        self.llm_client = LLMClient(config)

        # Runtime skill registry; the resource registry derives authoritative
        # counts and status from this runtime view.
        if skill_registry is not None:
            self.skill_registry = skill_registry
            self.resource_registry = build_resource_registry(self.skill_registry)
        else:
            sink = StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                self.skill_registry = build_default_registry(self.config)
                self.resource_registry = build_resource_registry(self.skill_registry)
        # Backward-compat alias
        self.kg_manager = self.skill_registry

        # Grab the WebSearchSkill instance for injection into WebSearchAgent
        _web_skill = self.skill_registry.get_skill("WebSearch")

        # Agents
        self.planner = PlannerAgent(
            self.llm_client,
            self.skill_registry,
            resource_registry=self.resource_registry,
        )
        self.coder = CoderAgent(self.llm_client, self.skill_registry)
        self.retriever = RetrieverAgent(
            self.llm_client,
            self.skill_registry,
            coder_agent=self.coder,
            resource_registry=self.resource_registry,
        )
        self.graph_builder = GraphBuilderAgent(self.llm_client)
        self.reranker = RerankerAgent(self.llm_client, config)
        self.responder = ResponderAgent(self.llm_client)
        self.reflector = ReflectorAgent(self.llm_client, config)
        self.web_search = WebSearchAgent(
            self.llm_client,
            web_search_skill=_web_skill,
        )

        # Build LangGraph workflow (handles all 3 modes via conditional routing)
        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # LangGraph workflow
    # ------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """
        Build a single StateGraph that routes between the three thinking modes.

        Graph topology:
          entry_router ──(mode=graph|simple)──► plan
                       ──(mode=web_only)──────► web_search_direct

          plan ──► retrieve ──► normalize_evidence
          normalize_evidence ──(graph)──► optional_graph ──► graph_build ──► rerank ──► assess_claims ──► respond ──► reflect ──► [web_search] ──► finalize
                                           └──────────────► assess_claims
                             ──(simple)──► assess_claims ──► simple_respond ──► finalize

          web_search_direct ──► finalize
          finalize ──► END
        """
        wf = StateGraph(AgentState)

        wf.add_node("entry_router",       self._entry_router_node)
        wf.add_node("plan",               self._plan_node)
        wf.add_node("retrieve",           self._retrieve_node)
        wf.add_node("normalize_evidence", self._normalize_evidence_node)
        wf.add_node("optional_graph",     self._optional_graph_node)
        wf.add_node("graph_build",        self._graph_build_node)
        wf.add_node("rerank",             self._rerank_node)
        wf.add_node("assess_claims",      self._assess_claims_node)
        wf.add_node("respond",            self._respond_node)
        wf.add_node("reflect",           self._reflect_node)
        wf.add_node("web_search",         self._web_search_node)
        wf.add_node("simple_respond",     self._simple_respond_node)
        wf.add_node("web_search_direct",  self._web_search_direct_node)
        wf.add_node("finalize",           self._finalize_node)

        wf.set_entry_point("entry_router")

        # entry_router → mode dispatch
        wf.add_conditional_edges(
            "entry_router",
            self._route_by_mode,
            {"plan": "plan", "web_search_direct": "web_search_direct"},
        )

        wf.add_edge("plan", "retrieve")
        wf.add_edge("retrieve", "normalize_evidence")

        # normalize → next step depends on mode
        wf.add_conditional_edges(
            "normalize_evidence",
            self._after_normalize_evidence,
            {"optional_graph": "optional_graph", "assess_claims": "assess_claims"},
        )
        wf.add_conditional_edges(
            "optional_graph",
            self._after_optional_graph,
            {"graph_build": "graph_build", "assess_claims": "assess_claims"},
        )

        # graph mode pipeline: graph_build → rerank → assess_claims → respond → reflect
        wf.add_edge("graph_build", "rerank")
        wf.add_edge("rerank",     "assess_claims")
        wf.add_conditional_edges(
            "assess_claims",
            self._after_assessment,
            {"respond": "respond", "simple_respond": "simple_respond"},
        )
        wf.add_edge("respond", "reflect")
        wf.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"web_search": "web_search", "finalize": "finalize"},
        )

        # simple / web_only termination
        wf.add_edge("web_search",        "finalize")
        wf.add_edge("simple_respond",    "finalize")
        wf.add_edge("web_search_direct", "finalize")
        wf.add_edge("finalize",          END)

        return wf.compile()

    # ------------------------------------------------------------------
    # Routing helpers (conditional edge functions)
    # ------------------------------------------------------------------

    def _route_by_mode(self, state: AgentState) -> str:
        mode = self._normalize_thinking_mode(
            getattr(state, "thinking_mode", ThinkingMode.GRAPH)
        )
        if mode == ThinkingMode.WEB_ONLY.value:
            return "web_search_direct"
        return "plan"

    def _after_normalize_evidence(self, state: AgentState) -> str:
        mode = self._normalize_thinking_mode(
            getattr(state, "thinking_mode", ThinkingMode.GRAPH)
        )
        if mode == ThinkingMode.GRAPH.value:
            return "optional_graph"
        return "assess_claims"

    def _after_optional_graph(self, state: AgentState) -> str:
        if getattr(state, "graph_decision_reason", "").startswith("run:"):
            return "graph_build"
        return "assess_claims"

    def _after_assessment(self, state: AgentState) -> str:
        mode = self._normalize_thinking_mode(
            getattr(state, "thinking_mode", ThinkingMode.GRAPH)
        )
        if mode == ThinkingMode.SIMPLE.value:
            return "simple_respond"
        return "respond"

    @staticmethod
    def _after_reflect(state: AgentState) -> str:
        if getattr(state, "should_continue", False) and not getattr(
            state, "max_iterations_reached", False
        ):
            return "web_search"
        return "finalize"

    # ------------------------------------------------------------------
    # Node wrappers
    # ------------------------------------------------------------------

    def _entry_router_node(self, state: AgentState) -> AgentState:
        """Passthrough — routing happens in the conditional edge above."""
        mode = self._normalize_thinking_mode(
            getattr(state, "thinking_mode", ThinkingMode.GRAPH)
        )
        rf   = getattr(state, "resource_filter", [])
        print(f"\n[DrugClaw] mode={mode}"
              + (f"  resource_filter={rf}" if rf else ""))
        return state

    def _plan_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "PLAN")
        if state.query_plan is None:
            omics_constraints = self._format_omics_constraints(
                state.omics_constraints
            )
            state.query_plan = self.planner.plan(
                state.original_query,
                omics_constraints=omics_constraints,
            )
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "RETRIEVE")
        return self.retriever.execute(state)

    def _normalize_evidence_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "NORMALIZE_EVIDENCE")
        return state

    def _graph_build_node(self, state: AgentState) -> AgentState:
        """Graph mode: LLM extracts entity triples from retrieval text."""
        return self.graph_builder.execute(state)

    def _rerank_node(self, state: AgentState) -> AgentState:
        return self.reranker.execute(state)

    def _optional_graph_node(self, state: AgentState) -> AgentState:
        should_run, reason = self._should_run_graph(state)
        state.graph_decision_reason = ("run:" if should_run else "skip:") + reason
        self._record_stage(
            state,
            "OPTIONAL_GRAPH:run" if should_run else "OPTIONAL_GRAPH:skipped",
        )
        return state

    def _assess_claims_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "ASSESS_CLAIMS")
        if state.evidence_items:
            state.claim_assessments = assess_claims(state.evidence_items)
        return state

    def _respond_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "ANSWER")
        return self.responder.execute(state)

    def _reflect_node(self, state: AgentState) -> AgentState:
        self._record_stage(state, "REFLECT")
        state = self.reflector.execute(state)
        step = ReasoningStep(
            step_id=state.iteration,
            query=state.original_query,
            subgraph=state.current_subgraph,
            intermediate_answer=state.current_answer,
            evidence_sufficiency=state.current_reward,
            reward=state.current_reward,
            actions_taken=["retrieve", "graph_build", "rerank", "respond", "reflect"],
        )
        state.add_reasoning_step(step)
        return state

    def _web_search_node(self, state: AgentState) -> AgentState:
        """Graph-mode web search (triggered by reflector when evidence is lacking)."""
        self._record_stage(state, "WEB_SEARCH")
        return self.web_search.execute(state)

    def _simple_respond_node(self, state: AgentState) -> AgentState:
        """Simple-mode: synthesize retrieved text directly."""
        self._record_stage(state, "ANSWER")
        return self.responder.execute_simple(state)

    def _web_search_direct_node(self, state: AgentState) -> AgentState:
        """WEB_ONLY mode: search the original query without prior retrieval."""
        return self.web_search.execute_direct(state)

    def _finalize_node(self, state: AgentState) -> AgentState:
        print("\n[DrugClaw] Finalizing answer...")
        state.final_answer = state.current_answer
        # Store raw answer for the formatter (used during logging)
        state.metadata = getattr(state, "metadata", {})
        if state.final_answer_structured is None and state.evidence_items:
            state.final_answer_structured = self.responder._build_final_answer(
                state.original_query,
                state.evidence_items,
            )
        return state

    @staticmethod
    def _record_stage(state: AgentState, stage: str) -> None:
        state.execution_stage = stage
        state.execution_trace.append(stage)

    @staticmethod
    def _should_run_graph(state: AgentState) -> tuple[bool, str]:
        plan = getattr(state, "query_plan", None)
        if plan is None:
            return False, "query plan unavailable"
        if not getattr(plan, "requires_graph_reasoning", False):
            return False, "planner did not recommend graph reasoning"

        entities = getattr(plan, "entities", {}) or {}
        entity_count = sum(len(values) for values in entities.values())
        if entity_count >= 2:
            return True, "multiple entities suggest relational composition"

        question_type = str(getattr(plan, "question_type", ""))
        if question_type in {"ddi_mechanism", "mechanism", "evidence_synthesis", "adr_causal"}:
            return True, f"question type `{question_type}` benefits from graph reasoning"

        if len(getattr(state, "evidence_items", []) or []) >= 2:
            return True, "multiple evidence items available for composition"

        return False, "evidence shape does not justify graph reasoning"

    @staticmethod
    def _format_omics_constraints(constraints: Optional[OmicsConstraints]) -> str:
        if constraints is None:
            return "No specific biological constraints provided."

        parts = []
        if constraints.gene_sets:
            parts.append(f"Genes: {', '.join(constraints.gene_sets[:10])}")
        if constraints.pathway_sets:
            parts.append(f"Pathways: {', '.join(constraints.pathway_sets[:10])}")
        if constraints.disease_terms:
            parts.append(f"Diseases: {', '.join(constraints.disease_terms)}")
        if constraints.tissue_types:
            parts.append(f"Tissues: {', '.join(constraints.tissue_types)}")
        return "\n".join(parts) if parts else "No specific constraints."

    @staticmethod
    def _normalize_thinking_mode(thinking_mode: str | ThinkingMode) -> str:
        if isinstance(thinking_mode, ThinkingMode):
            return thinking_mode.value

        normalized = str(thinking_mode).strip().lower()
        allowed = {mode.value for mode in ThinkingMode}
        if normalized in allowed:
            return normalized

        raise ValueError(
            "Invalid thinking_mode. Expected one of: "
            + ", ".join(sorted(allowed))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        omics_constraints: Optional[OmicsConstraints] = None,
        thinking_mode: str | ThinkingMode = ThinkingMode.GRAPH,
        resource_filter: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        save_html_report: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a drug-related query.

        Parameters
        ----------
        query            : Natural language question.
        omics_constraints: Optional biological constraints (gene sets, pathways, …).
        thinking_mode    : One of "graph" | "simple" | "web_only", or the
                           corresponding ThinkingMode enum value.
        resource_filter  : Optional list of skill names to restrict retrieval
                           (e.g. ["ChEMBL", "DGIdb"]).  When provided, the LLM
                           skill-selection step is bypassed.
        metadata         : Arbitrary metadata to log alongside the query.

        Returns
        -------
        dict with keys: query, answer, mode, iterations, evidence_graph_size,
                        final_reward, reasoning_history, retrieved_content,
                        web_search_results, success, [query_id]
        """
        normalized_mode: str | None = None
        try:
            normalized_mode = self._normalize_thinking_mode(thinking_mode)
            initial_state = AgentState(
                original_query=query,
                omics_constraints=omics_constraints,
                thinking_mode=normalized_mode,
                resource_filter=resource_filter or [],
            )
            if verbose:
                print(f"\n{'='*80}")
                print(f"QUERY [{normalized_mode}]: {query}")
                if resource_filter:
                    print(f"RESOURCE FILTER: {resource_filter}")
                print(f"{'='*80}\n")
                final = self.workflow.invoke(initial_state)
            else:
                sink = StringIO()
                with redirect_stdout(sink), redirect_stderr(sink):
                    final = self.workflow.invoke(initial_state)

            final_answer    = final.get("final_answer", final.get("current_answer", ""))
            final_answer_structured = final.get("final_answer_structured")
            iteration       = final.get("iteration", 0)
            current_reward  = final.get("current_reward", 0.0)
            reasoning_steps = final.get("reasoning_steps", [])
            subgraph        = final.get("current_subgraph")
            subgraph_size   = subgraph.get_size() if subgraph and hasattr(subgraph, "get_size") else 0

            result = {
                "query":              query,
                "answer":             final_answer,
                "final_answer_structured": (
                    final_answer_structured.to_dict()
                    if hasattr(final_answer_structured, "to_dict")
                    else final_answer_structured
                ),
                "query_plan": (
                    final.get("query_plan").to_dict()
                    if getattr(final.get("query_plan"), "to_dict", None)
                    else final.get("query_plan")
                ),
                "claim_assessments": [
                    assessment.to_dict() if hasattr(assessment, "to_dict") else assessment
                    for assessment in final.get("claim_assessments", [])
                ],
                "mode":               normalized_mode,
                "resource_filter":    resource_filter or [],
                "iterations":         iteration,
                "evidence_graph_size": subgraph_size,
                "final_reward":       current_reward,
                "execution_trace":    final.get("execution_trace", []),
                "execution_stage":    final.get("execution_stage", ""),
                "graph_decision_reason": final.get("graph_decision_reason", ""),
                "reasoning_history":  [
                    {
                        "step": s.step_id,
                        "answer": s.intermediate_answer,
                        "reward": s.reward,
                        "evidence_sufficiency": s.evidence_sufficiency,
                    }
                    for s in reasoning_steps
                ],
                "retrieved_content":  final.get("retrieved_content", []),
                "evidence_items": [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in final.get("evidence_items", [])
                ],
                "retrieved_text":     final.get("retrieved_text", ""),
                "reflection_feedback": final.get("reflection_feedback", ""),
                "web_search_results": final.get("web_search_results", []),
                "success":            True,
            }

            # Build rich Markdown report card
            formatted_answer = wrap_answer_card(final_answer, result)
            result["formatted_answer"] = formatted_answer

            if verbose:
                print(f"\n{'='*80}\nFINAL ANSWER\n{'='*80}\n")
                print(formatted_answer)

            if self.enable_logging and self.logger:
                if verbose:
                    result["query_id"] = self.logger.log_query(
                        query,
                        result,
                        metadata,
                        save_html_report=save_html_report,
                    )
                else:
                    sink = StringIO()
                    with redirect_stdout(sink), redirect_stderr(sink):
                        result["query_id"] = self.logger.log_query(
                            query,
                            result,
                            metadata,
                            save_html_report=save_html_report,
                        )
                if save_html_report:
                    html_report_path = self.logger.get_query_report_html_path(
                        result["query_id"]
                    )
                    if html_report_path:
                        result["html_report_path"] = html_report_path

            return result

        except Exception as exc:
            import traceback
            print(f"\n[DrugClaw] Error: {exc}")
            traceback.print_exc()
            return {
                "query":   query,
                "answer":  f"Error: {exc}",
                "mode":    normalized_mode if normalized_mode is not None else str(thinking_mode),
                "success": False,
            }

    # ------------------------------------------------------------------
    # Logging helpers (unchanged API)
    # ------------------------------------------------------------------

    def get_query_history(self, n: int = 10) -> List[Dict[str, Any]]:
        if not self.logger:
            return []
        return self.logger.get_recent_queries(n)

    def get_query_by_id(self, query_id: str, detailed: bool = False):
        if not self.logger:
            return None
        return self.logger.get_query(query_id, detailed)

    def search_queries(self, keyword: str) -> List[Dict[str, Any]]:
        if not self.logger:
            return []
        return self.logger.search_queries(keyword=keyword)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.logger:
            return {}
        return self.logger.get_statistics()

    def visualize_workflow(self, output_path: str = "workflow.png"):
        try:
            png_data = self.workflow.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as fh:
                fh.write(png_data)
            print(f"Workflow saved to {output_path}")
        except Exception as exc:
            print(f"Cannot generate visualization: {exc}")
