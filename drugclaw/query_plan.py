from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List


PRIMARY_TARGET_LOOKUP_SKILLS = (
    "BindingDB",
    "ChEMBL",
    "DGIdb",
    "Open Targets Platform",
    "DrugBank",
    "TTD",
    "STITCH",
)

SECONDARY_TARGET_LOOKUP_SKILLS = (
    "Molecular Targets",
    "Molecular Targets Data",
    "TarKG",
    "DRUGMECHDB",
)


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


def normalize_question_type(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_direct_target_lookup(*, query: str = "", question_type: str = "") -> bool:
    normalized_type = normalize_question_type(question_type)
    if any(
        marker in normalized_type
        for marker in (
            "target_lookup",
            "drug_target_interaction",
            "drug_target_identification",
            "relationship_retrieval",
        )
    ):
        return True

    lowered_query = str(query).strip().lower()
    if not lowered_query:
        return False

    if re.search(r"\bdoes\s+[a-z0-9\-]+\s+target\b", lowered_query):
        return True

    return any(
        marker in lowered_query
        for marker in (
            "drug target",
            "drug targets",
            "known target",
            "known targets",
            "target of",
            "targets of",
        )
    )


def prioritize_target_lookup_skills(skill_names: List[str]) -> List[str]:
    unique_names = list(dict.fromkeys(skill_names))
    if not unique_names:
        return []

    primary = [name for name in PRIMARY_TARGET_LOOKUP_SKILLS if name in unique_names]
    secondary = [name for name in SECONDARY_TARGET_LOOKUP_SKILLS if name in unique_names]
    remainder = [
        name
        for name in unique_names
        if name not in primary and name not in secondary
    ]

    # For direct target lookup, keep first-line DTI sources exclusive when the
    # runtime has enough of them. Fall back to broader/specialized sources only
    # when the primary set is sparse.
    if len(primary) >= 3:
        return primary
    if primary:
        return primary + secondary + remainder
    return secondary + remainder
