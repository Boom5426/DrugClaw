from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvidenceItem:
    evidence_id: str
    source_skill: str
    source_type: str
    source_title: str
    source_locator: str
    snippet: str
    structured_payload: Dict[str, Any]
    claim: str
    evidence_kind: str
    support_direction: str
    confidence: float
    retrieval_score: Optional[float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimSummary:
    claim: str
    confidence: float
    evidence_ids: List[str]
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalAnswer:
    answer_text: str
    summary_confidence: float
    key_claims: List[ClaimSummary]
    evidence_items: List[EvidenceItem]
    citations: List[str]
    limitations: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "summary_confidence": self.summary_confidence,
            "key_claims": [claim.to_dict() for claim in self.key_claims],
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "citations": self.citations,
            "limitations": self.limitations,
            "warnings": self.warnings,
        }


def score_evidence_item(item: EvidenceItem) -> float:
    """Rule-based evidence quality score.

    Higher confidence is reserved for curated, direct, multi-field evidence.
    Predictive-only evidence remains usable but should rarely dominate.
    """
    base_by_kind = {
        "database_record": 0.78,
        "label_text": 0.8,
        "literature_statement": 0.82,
        "ontology_relation": 0.72,
        "model_prediction": 0.48,
    }
    score = base_by_kind.get(item.evidence_kind, 0.6)

    if item.structured_payload:
        score += 0.05
    if item.retrieval_score is not None:
        score += max(0.0, min(0.1, item.retrieval_score * 0.1))
    if len(item.snippet or "") < 40:
        score -= 0.08
    else:
        score += 0.03
    if item.support_direction == "neutral":
        score -= 0.1
    return _clamp(score)


def score_claim_confidence(items: Iterable[EvidenceItem]) -> float:
    evidence_items = list(items)
    if not evidence_items:
        return 0.0

    item_scores = [score_evidence_item(item) for item in evidence_items]
    score = sum(item_scores) / len(item_scores)

    unique_sources = {item.source_skill for item in evidence_items}
    if len(unique_sources) >= 2:
        score += 0.1
    elif len(unique_sources) == 1:
        score -= 0.08

    contradictions = sum(1 for item in evidence_items if item.support_direction == "contradicts")
    if contradictions:
        score -= min(0.3, 0.15 * contradictions)

    if all(item.evidence_kind == "model_prediction" for item in evidence_items):
        score -= 0.2

    return _clamp(score)


def score_answer_confidence(claims: Iterable[ClaimSummary]) -> float:
    claim_summaries = list(claims)
    if not claim_summaries:
        return 0.0
    return _clamp(
        sum(summary.confidence for summary in claim_summaries) / len(claim_summaries)
    )


def build_evidence_items_for_skill(
    *,
    skill_name: str,
    records: List[Dict[str, Any]],
    query: str = "",
    skill: Any = None,
) -> List[EvidenceItem]:
    if skill is not None and hasattr(skill, "build_evidence_items"):
        return skill.build_evidence_items(records, query=query)

    items: List[EvidenceItem] = []
    timestamp = _now_iso()
    for index, record in enumerate(records, start=1):
        source_entity = record.get("source_entity", "")
        relationship = record.get("relationship", "related_to")
        target_entity = record.get("target_entity", "")
        claim = _claim_from_record(record, query=query)
        locator = _locator_from_record(record)
        snippet = record.get("evidence_text", "") or claim
        structured_payload = {
            key: value
            for key, value in record.items()
            if key not in {
                "source", "source_entity", "source_type", "target_entity",
                "target_type", "relationship", "evidence_text", "skill_category",
            }
        }
        evidence_kind = _evidence_kind_from_record(record, skill=skill)
        item = EvidenceItem(
            evidence_id=f"{_slug(skill_name)}:{index}",
            source_skill=skill_name,
            source_type=getattr(skill, "resource_type", record.get("source_type", "unknown")),
            source_title=record.get("source", skill_name),
            source_locator=locator,
            snippet=snippet,
            structured_payload=structured_payload,
            claim=claim if claim else f"{source_entity} {relationship} {target_entity}".strip(),
            evidence_kind=evidence_kind,
            support_direction="supports",
            confidence=0.0,
            retrieval_score=_coerce_float(record.get("retrieval_score")),
            timestamp=timestamp,
            metadata={
                "skill_category": record.get("skill_category", getattr(skill, "subcategory", "")),
                "source_entity": source_entity,
                "relationship": relationship,
                "target_entity": target_entity,
                "source_type": record.get("source_type", ""),
                "target_type": record.get("target_type", ""),
            },
        )
        item.confidence = score_evidence_item(item)
        items.append(item)
    return items


def _claim_from_record(record: Dict[str, Any], query: str) -> str:
    source_entity = record.get("source_entity", "")
    relationship = record.get("relationship", "related_to")
    target_entity = record.get("target_entity", "")
    if source_entity or target_entity:
        return f"{source_entity} {relationship} {target_entity}".strip()
    return record.get("evidence_text", "")[:200] or query


def _locator_from_record(record: Dict[str, Any]) -> str:
    sources = record.get("sources", [])
    if isinstance(sources, list) and sources:
        return str(sources[0])
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("url", "pmid", "target_id", "chembl_id", "id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return record.get("source", "unknown")


def _evidence_kind_from_record(record: Dict[str, Any], skill: Any = None) -> str:
    relationship = str(record.get("relationship", ""))
    resource_type = str(getattr(skill, "resource_type", "") or record.get("resource_type", ""))
    if "linked_target" in relationship or "prediction" in relationship:
        return "model_prediction"
    if "drug_info" in relationship or "label" in relationship:
        return "label_text"
    if resource_type.lower() in {"database", "dataset"}:
        return "database_record"
    if resource_type.lower() == "kg":
        return "ontology_relation"
    return "database_record"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
