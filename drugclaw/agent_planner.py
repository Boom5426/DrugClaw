from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .knowhow_registry import KnowHowRegistry
from .knowhow_retriever import KnowHowRetriever
from .query_plan import (
    QueryPlan,
    build_fallback_query_plan,
    infer_entities_from_query,
    is_direct_target_lookup,
    is_supported_question_type,
    normalize_question_type,
    prioritize_target_lookup_skills,
)


class PlannerAgent:
    """Produce a structured retrieval plan from the raw user query."""

    def __init__(
        self,
        llm_client,
        skill_registry=None,
        resource_registry=None,
        knowhow_registry: KnowHowRegistry | None = None,
        knowhow_retriever: KnowHowRetriever | None = None,
    ):
        self.llm = llm_client
        self.skill_registry = skill_registry
        self.resource_registry = resource_registry
        self.knowhow_registry = knowhow_registry or KnowHowRegistry()
        self.knowhow_retriever = knowhow_retriever or KnowHowRetriever(self.knowhow_registry)

    def get_system_prompt(self) -> str:
        return """You are the Planner Agent for DrugClaw.

Your role is narrow and structured:
- classify the question type
- extract relevant biomedical entities
- decompose the query into short subquestions
- suggest preferred skills and evidence types
- indicate whether graph reasoning may be useful

Do not retrieve evidence. Do not answer the question. Do not invent facts.
Return concise JSON only."""

    def get_planning_prompt(
        self,
        query: str,
        omics_constraints: Optional[str] = None,
    ) -> str:
        omics_text = omics_constraints or "No explicit omics constraints."
        suggested_skills = []
        if self.skill_registry is not None:
            try:
                suggested_skills = self._rank_suggested_skills(
                    self.skill_registry.get_skills_for_query(query),
                    query=query,
                )[:8]
            except Exception:
                suggested_skills = []
        suggestion_text = (
            "Suggested exact runtime skill names:\n- " + "\n- ".join(suggested_skills)
            if suggested_skills
            else "Suggested exact runtime skill names:\n- (none available; leave preferred_skills empty if unsure)"
        )
        return f"""User query: {query}

Omics constraints:
{omics_text}

{suggestion_text}

Return JSON with these fields:
- plan_type
- primary_task
- supporting_tasks
- execution_tasks
- answer_contract
- question_type
- entities
- subquestions
- preferred_skills
- preferred_evidence_types
- requires_graph_reasoning
- requires_prediction_sources
- requires_web_fallback
- answer_risk_level
- notes

Rules:
- `preferred_skills` must contain exact registered runtime skill names only
- do not invent category labels, capability names, or abstract tool names
- if no exact skill name is justified, return an empty list
- prefer the v2 task fields when the query has multiple answer intents
"""

    def _rank_suggested_skills(self, skill_names: List[str], *, query: str) -> List[str]:
        if not skill_names:
            return []
        if is_direct_target_lookup(query=query):
            target_lookup_ranked = prioritize_target_lookup_skills(skill_names)
            if target_lookup_ranked:
                skill_names = target_lookup_ranked
        if self.resource_registry is not None and hasattr(
            self.resource_registry, "prioritize_resource_names"
        ):
            ready_only = self.resource_registry.prioritize_resource_names(
                skill_names,
                ready_only=True,
            )
            if ready_only:
                return ready_only
            ranked = self.resource_registry.prioritize_resource_names(skill_names)
            if ranked:
                return ranked
        return list(dict.fromkeys(skill_names))

    def plan(
        self,
        query: str,
        omics_constraints: Optional[str] = None,
    ) -> QueryPlan:
        if not query.strip():
            return self._attach_knowhow(build_fallback_query_plan(query))

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": self.get_planning_prompt(query, omics_constraints),
            },
        ]
        try:
            result = self.llm.generate_json(messages, temperature=0.2)
        except Exception:
            return self._attach_knowhow(build_fallback_query_plan(query))
        return self._attach_knowhow(self._normalize_query_plan(query, result))

    def _attach_knowhow(self, plan: QueryPlan) -> QueryPlan:
        try:
            return self.knowhow_retriever.enrich_query_plan(plan)
        except Exception:
            return plan

    def _normalize_query_plan(self, query: str, payload: Dict[str, Any]) -> QueryPlan:
        fallback = build_fallback_query_plan(query)
        entities = self._normalize_entities(payload.get("entities"))
        if not entities:
            entities = infer_entities_from_query(query)

        has_v2_structure = any(
            payload.get(key)
            for key in (
                "plan_type",
                "primary_task",
                "supporting_tasks",
                "execution_tasks",
                "answer_contract",
            )
        )

        if has_v2_structure:
            candidate = QueryPlan(
                question_type=str(payload.get("question_type") or fallback.question_type),
                entities=entities or fallback.entities,
                subquestions=self._normalize_list(payload.get("subquestions")) or fallback.subquestions,
                preferred_skills=self._normalize_list(payload.get("preferred_skills")) or fallback.preferred_skills,
                preferred_evidence_types=self._normalize_list(payload.get("preferred_evidence_types")),
                requires_graph_reasoning=bool(payload.get("requires_graph_reasoning", False)),
                requires_prediction_sources=bool(payload.get("requires_prediction_sources", False)),
                requires_web_fallback=bool(payload.get("requires_web_fallback", False)),
                answer_risk_level=str(payload.get("answer_risk_level") or fallback.answer_risk_level),
                notes=self._normalize_list(payload.get("notes")) or fallback.notes,
                knowhow_doc_ids=self._normalize_list(payload.get("knowhow_doc_ids")),
                knowhow_hints=payload.get("knowhow_hints") or [],
                plan_type=str(payload.get("plan_type") or fallback.plan_type),
                primary_task=payload.get("primary_task") or fallback.primary_task,
                supporting_tasks=payload.get("supporting_tasks") or fallback.supporting_tasks,
                execution_tasks=payload.get("execution_tasks") or fallback.execution_tasks,
                answer_contract=payload.get("answer_contract") or fallback.answer_contract,
            )
            candidate = self._sanitize_candidate_plan(
                query,
                candidate,
                fallback=fallback,
            )
            if getattr(candidate.primary_task, "task_type", "unknown") == "unknown":
                return fallback
            return candidate

        raw_question_type = str(payload.get("question_type") or "").strip()
        raw_question_type_key = raw_question_type.lower().replace("-", "_").replace(" ", "_")
        question_type = normalize_question_type(raw_question_type)
        use_fallback_skills = False
        if (
            not raw_question_type
            or (
                normalize_question_type(question_type) == "unknown"
                and fallback.question_type != "unknown"
            )
            or (
                not is_supported_question_type(question_type)
                and fallback.question_type != "unknown"
            )
        ):
            question_type = fallback.question_type
            use_fallback_skills = True
        elif raw_question_type_key != question_type:
            use_fallback_skills = True

        strong_path_question_types = {
            "drug_repurposing",
            "mechanism",
            "pharmacogenomics",
        }
        preferred_skills = self._normalize_list(payload.get("preferred_skills"))
        if (
            use_fallback_skills
            or not preferred_skills
            or question_type in strong_path_question_types
        ):
            preferred_skills = fallback.preferred_skills

        requires_graph_reasoning = bool(payload.get("requires_graph_reasoning", False))
        if use_fallback_skills or question_type in strong_path_question_types:
            requires_graph_reasoning = fallback.requires_graph_reasoning

        if fallback.plan_type == "composite_query":
            promoted_subquestions = self._normalize_list(payload.get("subquestions"))
            if len(promoted_subquestions) < (1 + len(fallback.supporting_tasks)):
                promoted_subquestions = fallback.subquestions

            primary_task = (
                fallback.primary_task.to_dict() if fallback.primary_task is not None else None
            )
            if primary_task is not None:
                primary_task["entities"] = entities or fallback.entities

            supporting_tasks = []
            for task in fallback.supporting_tasks:
                task_payload = task.to_dict()
                task_payload["entities"] = entities or fallback.entities
                supporting_tasks.append(task_payload)

            return QueryPlan(
                question_type=fallback.question_type,
                entities=entities or fallback.entities,
                subquestions=promoted_subquestions,
                preferred_skills=[],
                preferred_evidence_types=self._normalize_list(
                    payload.get("preferred_evidence_types")
                ),
                requires_graph_reasoning=requires_graph_reasoning,
                requires_prediction_sources=bool(
                    payload.get("requires_prediction_sources", False)
                ),
                requires_web_fallback=bool(payload.get("requires_web_fallback", False))
                or fallback.requires_web_fallback,
                answer_risk_level=str(
                    payload.get("answer_risk_level") or fallback.answer_risk_level
                ),
                notes=self._normalize_list(payload.get("notes")) or fallback.notes,
                knowhow_doc_ids=self._normalize_list(payload.get("knowhow_doc_ids")),
                knowhow_hints=payload.get("knowhow_hints") or [],
                plan_type=fallback.plan_type,
                primary_task=primary_task,
                supporting_tasks=supporting_tasks,
                answer_contract=fallback.answer_contract.to_dict()
                if fallback.answer_contract is not None
                else None,
            )

        return QueryPlan(
            question_type=question_type or fallback.question_type,
            entities=entities,
            subquestions=self._normalize_list(payload.get("subquestions")) or fallback.subquestions,
            preferred_skills=preferred_skills,
            preferred_evidence_types=self._normalize_list(payload.get("preferred_evidence_types")),
            requires_graph_reasoning=requires_graph_reasoning,
            requires_prediction_sources=bool(payload.get("requires_prediction_sources", False)),
            requires_web_fallback=bool(payload.get("requires_web_fallback", False)),
            answer_risk_level=str(payload.get("answer_risk_level") or fallback.answer_risk_level),
            notes=self._normalize_list(payload.get("notes")) or fallback.notes,
            knowhow_doc_ids=self._normalize_list(payload.get("knowhow_doc_ids")),
            knowhow_hints=payload.get("knowhow_hints") or [],
        )

    def _sanitize_candidate_plan(
        self,
        query: str,
        candidate: QueryPlan,
        *,
        fallback: QueryPlan,
    ) -> QueryPlan:
        primary_task = (
            candidate.primary_task.to_dict()
            if getattr(candidate, "primary_task", None) is not None
            else None
        )
        primary_task_type = str(
            getattr(getattr(candidate, "primary_task", None), "task_type", "") or ""
        ).strip()

        supporting_tasks = []
        seen_task_types = {primary_task_type} if primary_task_type else set()
        for task in list(getattr(candidate, "supporting_tasks", []) or []):
            task_type = str(getattr(task, "task_type", "") or "").strip()
            if not task_type or task_type == "unknown" or task_type in seen_task_types:
                continue
            seen_task_types.add(task_type)
            supporting_tasks.append(task.to_dict() if hasattr(task, "to_dict") else task)

        if is_direct_target_lookup(query=query, question_type=candidate.question_type):
            primary_task = (
                fallback.primary_task.to_dict()
                if getattr(fallback, "primary_task", None) is not None
                else primary_task
            )
            if primary_task is not None:
                primary_task["entities"] = dict(candidate.entities or fallback.entities)
            supporting_tasks = []

        plan_type = "composite_query" if supporting_tasks else "single_task"
        answer_contract = (
            candidate.answer_contract.to_dict()
            if getattr(candidate, "answer_contract", None) is not None
            else None
        )
        if plan_type == "single_task" and getattr(fallback, "answer_contract", None) is not None:
            answer_contract = fallback.answer_contract.to_dict()

        preferred_skills = list(candidate.preferred_skills)
        if plan_type == "single_task" and is_direct_target_lookup(
            query=query,
            question_type=candidate.question_type,
        ):
            preferred_skills = list(fallback.preferred_skills)

        return QueryPlan(
            question_type=str(candidate.question_type or fallback.question_type),
            entities=dict(candidate.entities or fallback.entities),
            subquestions=list(candidate.subquestions or fallback.subquestions),
            preferred_skills=preferred_skills,
            preferred_evidence_types=list(candidate.preferred_evidence_types),
            requires_graph_reasoning=bool(candidate.requires_graph_reasoning),
            requires_prediction_sources=bool(candidate.requires_prediction_sources),
            requires_web_fallback=bool(candidate.requires_web_fallback),
            answer_risk_level=str(candidate.answer_risk_level or fallback.answer_risk_level),
            notes=list(candidate.notes or fallback.notes),
            knowhow_doc_ids=list(candidate.knowhow_doc_ids),
            knowhow_hints=list(candidate.knowhow_hints),
            plan_type=plan_type,
            primary_task=primary_task,
            supporting_tasks=supporting_tasks,
            answer_contract=answer_contract,
        )

    @staticmethod
    def _normalize_entities(value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, dict):
            return {}

        normalized: Dict[str, List[str]] = {}
        for key, raw in value.items():
            items = PlannerAgent._normalize_list(raw)
            if items:
                normalized[str(key)] = items
        return normalized

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray, dict)):
            return []

        normalized: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def _infer_entities_from_query(query: str) -> Dict[str, List[str]]:
        return infer_entities_from_query(query)
