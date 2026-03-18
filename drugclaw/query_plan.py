from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class QueryPlan:
    question_type: str
    entities: Dict[str, List[str]]
    subquestions: List[str]
    preferred_skills: List[str]
    preferred_evidence_types: List[str]
    requires_graph_reasoning: bool
    requires_prediction_sources: bool
    requires_web_fallback: bool
    answer_risk_level: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_fallback_query_plan(query: str) -> QueryPlan:
    return QueryPlan(
        question_type="unknown",
        entities={},
        subquestions=[query] if query else [],
        preferred_skills=[],
        preferred_evidence_types=[],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level="medium",
        notes=["Fallback plan used because planner output was unavailable or invalid."],
    )
