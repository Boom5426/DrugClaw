from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .query_plan import QueryPlan, build_fallback_query_plan


class PlannerAgent:
    """Produce a structured retrieval plan from the raw user query."""

    def __init__(self, llm_client, skill_registry=None, resource_registry=None):
        self.llm = llm_client
        self.skill_registry = skill_registry
        self.resource_registry = resource_registry

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
                    self.skill_registry.get_skills_for_query(query)
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
"""

    def _rank_suggested_skills(self, skill_names: List[str]) -> List[str]:
        if not skill_names:
            return []
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
        entities = self._normalize_entities(payload.get("entities"))
        if not entities:
            entities = self._infer_entities_from_query(query)
        return QueryPlan(
            question_type=str(payload.get("question_type") or fallback.question_type),
            entities=entities,
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

    @staticmethod
    def _infer_entities_from_query(query: str) -> Dict[str, List[str]]:
        lowered = query.strip().lower()
        if not lowered:
            return {}

        ddi_match = re.search(
            r"how does\s+([a-z0-9\-]+)\s+interact with\s+([a-z0-9\-]+)",
            lowered,
        )
        if ddi_match:
            return {"drug": [ddi_match.group(1), ddi_match.group(2)]}

        for pattern in (
            r"targets?\s+of\s+([a-z0-9\-]+)",
            r"information\s+is\s+available\s+for\s+([a-z0-9\-]+)",
            r"available\s+for\s+([a-z0-9\-]+)",
            r"about\s+([a-z0-9\-]+)$",
        ):
            match = re.search(pattern, lowered)
            if match:
                return {"drug": [match.group(1)]}

        tokens = re.findall(r"[a-z0-9\-]+", lowered)
        stopwords = {
            "what", "are", "the", "known", "drug", "drugs", "target", "targets",
            "of", "for", "does", "is", "available", "information", "how",
            "interact", "with", "and", "safety", "prescribing",
        }
        candidates = [token for token in tokens if token not in stopwords and len(token) > 2]
        if candidates:
            return {"drug": [candidates[-1]]}
        return {}
