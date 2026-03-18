from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .query_plan import QueryPlan, build_fallback_query_plan


class PlannerAgent:
    """Produce a structured retrieval plan from the raw user query."""

    def __init__(self, llm_client):
        self.llm = llm_client

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
        return f"""User query: {query}

Omics constraints:
{omics_text}

Return JSON with these fields:
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
"""

    def plan(
        self,
        query: str,
        omics_constraints: Optional[str] = None,
    ) -> QueryPlan:
        if not query.strip():
            return build_fallback_query_plan(query)

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
            return build_fallback_query_plan(query)
        return self._normalize_query_plan(query, result)

    def _normalize_query_plan(self, query: str, payload: Dict[str, Any]) -> QueryPlan:
        fallback = build_fallback_query_plan(query)
        return QueryPlan(
            question_type=str(payload.get("question_type") or fallback.question_type),
            entities=self._normalize_entities(payload.get("entities")),
            subquestions=self._normalize_list(payload.get("subquestions")) or fallback.subquestions,
            preferred_skills=self._normalize_list(payload.get("preferred_skills")),
            preferred_evidence_types=self._normalize_list(payload.get("preferred_evidence_types")),
            requires_graph_reasoning=bool(payload.get("requires_graph_reasoning", False)),
            requires_prediction_sources=bool(payload.get("requires_prediction_sources", False)),
            requires_web_fallback=bool(payload.get("requires_web_fallback", False)),
            answer_risk_level=str(payload.get("answer_risk_level") or fallback.answer_risk_level),
            notes=self._normalize_list(payload.get("notes")) or fallback.notes,
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
