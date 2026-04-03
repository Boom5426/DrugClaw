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


QUESTION_TYPE_ALIASES = {
    "drug_target_interaction": "target_lookup",
    "drug_target_identification": "target_lookup",
    "relationship_retrieval": "target_lookup",
    "drug_indications_and_repurposing_evidence": "drug_repurposing",
    "drug_indication_and_repurposing_evidence_query": "drug_repurposing",
    "drug_target_and_mechanism_of_action": "mechanism",
    "drug_safety_adverse_reactions": "adr",
    "drug_drug_interaction_mechanism_query_with_clinical_relevance_prioritization": "ddi_mechanism",
    "drug_prescribing_and_clinical_use_summary": "labeling",
    "pharmacogenomics_mechanism_and_clinical_impact_query": "pharmacogenomics",
}


SUPPORTED_QUESTION_TYPES = {
    "unknown",
    "target_lookup",
    "mechanism",
    "drug_repurposing",
    "adr",
    "ddi",
    "ddi_mechanism",
    "labeling",
    "pharmacogenomics",
}


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
    question_type = infer_question_type_from_query(query)
    return QueryPlan(
        question_type=question_type,
        entities=infer_entities_from_query(query),
        subquestions=[query] if query else [],
        preferred_skills=preferred_skills_for_question_type(question_type),
        preferred_evidence_types=[],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level=risk_level_for_question_type(question_type),
        notes=["Fallback plan used because planner output was unavailable or invalid."],
    )


def normalize_question_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if "mechanism_of_action" in normalized and "target" in normalized:
        return "mechanism"
    if normalized in QUESTION_TYPE_ALIASES:
        return QUESTION_TYPE_ALIASES[normalized]

    if "repurpos" in normalized and "indication" in normalized:
        return "drug_repurposing"
    if "pharmacogenom" in normalized or normalized.startswith("pgx"):
        return "pharmacogenomics"
    if ("interaction" in normalized or "ddi" in normalized) and "mechanism" in normalized:
        return "ddi_mechanism"
    if "interaction" in normalized or normalized == "ddi":
        return "ddi"
    if (
        "prescribing" in normalized
        or "clinical_use" in normalized
        or "label" in normalized
    ):
        return "labeling"
    if (
        "adverse_reaction" in normalized
        or "adverse_event" in normalized
        or "safety" in normalized
        or normalized == "adr"
    ):
        return "adr"
    return normalized


def is_supported_question_type(value: str) -> bool:
    return normalize_question_type(value) in SUPPORTED_QUESTION_TYPES


def infer_question_type_from_query(query: str) -> str:
    lowered = str(query).strip().lower()
    if not lowered:
        return "unknown"

    if any(
        marker in lowered
        for marker in (
            "pharmacogenomic",
            "pharmacogenomics",
            "pgx",
            "genotype",
            "genotypes",
            "metabolizer",
            "metaboliser",
            "cyp2c19",
            "cyp2c9",
            "vkorc1",
            "hla-b",
            "dpyd",
        )
    ):
        return "pharmacogenomics"

    if any(
        marker in lowered
        for marker in (
            "drug-drug interaction",
            "drug drug interaction",
            "drug interactions",
            "interactions of",
            "interact with",
            "interaction with",
        )
    ):
        if any(marker in lowered for marker in ("mechanism", "mechanisms")):
            return "ddi_mechanism"
        return "ddi"

    if any(
        marker in lowered
        for marker in (
            "mechanism of action",
            " moa",
            " moa?",
        )
    ):
        return "mechanism"

    if is_direct_target_lookup(query=lowered):
        return "target_lookup"

    if any(
        marker in lowered
        for marker in (
            "repurposing",
            "repositioning",
            "reposition",
            "approved indications",
            "repurposing evidence",
            "new indications",
        )
    ):
        return "drug_repurposing"

    if any(
        marker in lowered
        for marker in (
            "prescribing",
            "contraindication",
            "contraindications",
            "warning",
            "warnings",
            "boxed warning",
            "clinical use",
            "dosing",
            "dose",
            "labeling",
            "label information",
            "patient guidance",
        )
    ):
        return "labeling"

    if any(
        marker in lowered
        for marker in (
            "adverse drug reaction",
            "adverse drug reactions",
            "adverse reaction",
            "adverse reactions",
            "side effect",
            "side effects",
            "safety risk",
            "safety risks",
            "major safety",
            "toxicity",
        )
    ):
        return "adr"

    return "unknown"


def infer_entities_from_query(query: str) -> Dict[str, List[str]]:
    lowered = str(query).strip().lower()
    if not lowered:
        return {}

    ddi_match = re.search(
        r"how does\s+([a-z0-9\-]+)\s+interact with\s+([a-z0-9\-]+)",
        lowered,
    )
    if ddi_match:
        return {"drug": [ddi_match.group(1), ddi_match.group(2)]}

    for pattern in (
        r"does\s+([a-z0-9\-]+)\s+target",
        r"targets?\s+of\s+([a-z0-9\-]+)",
        r"mechanism of action of\s+([a-z0-9\-]+)",
        r"information\s+is\s+available\s+for\s+([a-z0-9\-]+)",
        r"prescribing.*for\s+([a-z0-9\-]+)",
        r"affect\s+([a-z0-9\-]+)\s+(?:efficacy|safety)",
        r"(?:risks?|reactions?|interactions?)\s+of\s+([a-z0-9\-]+)",
        r"of\s+([a-z0-9\-]+)\?$",
        r"for\s+([a-z0-9\-]+)\?$",
    ):
        match = re.search(pattern, lowered)
        if match:
            return {"drug": [match.group(1)]}

    tokens = re.findall(r"[a-z0-9\-]+", lowered)
    stopwords = {
        "what",
        "are",
        "the",
        "known",
        "drug",
        "drugs",
        "target",
        "targets",
        "of",
        "for",
        "does",
        "is",
        "available",
        "information",
        "how",
        "interact",
        "with",
        "and",
        "safety",
        "prescribing",
        "pharmacogenomic",
        "pharmacogenomics",
        "factors",
        "affect",
        "efficacy",
        "major",
        "serious",
        "adverse",
        "reactions",
        "reaction",
        "risk",
        "risks",
        "clinically",
        "important",
        "interactions",
        "mechanisms",
        "mechanism",
        "key",
        "clinical",
        "use",
        "approved",
        "indications",
        "repurposing",
        "evidence",
        "their",
        "action",
    }
    candidates = [token for token in tokens if token not in stopwords and len(token) > 2]
    if candidates:
        return {"drug": [candidates[-1]]}
    return {}


def preferred_skills_for_question_type(question_type: str) -> List[str]:
    normalized = normalize_question_type(question_type)
    mapping = {
        "target_lookup": [
            "BindingDB",
            "ChEMBL",
            "DGIdb",
            "Open Targets Platform",
        ],
        "mechanism": [
            "Open Targets Platform",
            "DRUGMECHDB",
            "BindingDB",
            "ChEMBL",
        ],
        "drug_repurposing": [
            "RepoDB",
            "DrugCentral",
            "DrugBank",
            "Open Targets Platform",
            "DRUGMECHDB",
            "openFDA Human Drug",
        ],
        "adr": [
            "ADReCS",
            "FAERS",
            "nSIDES",
            "SIDER",
        ],
        "ddi": [
            "DDInter",
            "KEGG Drug",
            "MecDDI",
        ],
        "ddi_mechanism": [
            "DDInter",
            "KEGG Drug",
            "MecDDI",
        ],
        "labeling": [
            "DailyMed",
            "openFDA Human Drug",
            "MedlinePlus Drug Info",
        ],
        "pharmacogenomics": [
            "PharmGKB",
            "CPIC",
        ],
    }
    return list(mapping.get(normalized, []))


def risk_level_for_question_type(question_type: str) -> str:
    normalized = normalize_question_type(question_type)
    if normalized in {
        "adr",
        "ddi",
        "ddi_mechanism",
        "labeling",
        "pharmacogenomics",
    }:
        return "high"
    return "medium"


def is_direct_target_lookup(*, query: str = "", question_type: str = "") -> bool:
    normalized_type = normalize_question_type(question_type)
    if normalized_type == "mechanism":
        return False
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
