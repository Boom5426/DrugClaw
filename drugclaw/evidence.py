from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .query_plan import infer_question_type_from_query


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
    task_type: str = ""
    final_outcome: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "summary_confidence": self.summary_confidence,
            "key_claims": [claim.to_dict() for claim in self.key_claims],
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "citations": self.citations,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "task_type": self.task_type,
            "final_outcome": self.final_outcome,
            "diagnostics": self.diagnostics,
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
    question_type = infer_question_type_from_query(query)
    specialized_builder = {
        "ddi": _ddi_claim_from_record,
        "ddi_mechanism": _ddi_claim_from_record,
        "labeling": _labeling_claim_from_record,
        "pharmacogenomics": _pgx_claim_from_record,
        "adr": _adr_claim_from_record,
    }.get(question_type)

    if specialized_builder is not None:
        specialized_claim = specialized_builder(record)
        if specialized_claim:
            return specialized_claim

    source_entity = record.get("source_entity", "")
    relationship = record.get("relationship", "related_to")
    target_entity = record.get("target_entity", "")
    if source_entity or target_entity:
        return f"{source_entity} {relationship} {target_entity}".strip()
    return record.get("evidence_text", "")[:200] or query


def _ddi_claim_from_record(record: Dict[str, Any]) -> str:
    source_entity = str(record.get("source_entity", "")).strip()
    target_entity = str(record.get("target_entity", "")).strip()
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    description = str(
        metadata.get("ddi_description")
        or record.get("ddi_description")
        or metadata.get("description")
        or ""
    ).strip()
    label = str(metadata.get("ddi_label") or record.get("ddi_label") or "").strip()
    snippet = str(record.get("evidence_text", "") or "").strip()

    partner = ""
    if target_entity and not _looks_like_compound_identifier(target_entity):
        partner = target_entity
    if not partner:
        partner = _extract_partner_from_text(description) or _extract_partner_from_text(snippet)
    if partner and _looks_like_compound_identifier(partner):
        partner = ""

    if description.lower().startswith("enzyme:"):
        enzyme = description.split(":", 1)[1].strip()
        if enzyme:
            return f"{source_entity} interaction mechanism involves {enzyme}"

    if source_entity and partner:
        details = description or label
        if details:
            return f"{source_entity} interacts with {partner} ({details})"
        return f"{source_entity} interacts with {partner}"
    if source_entity and description.lower() == "unclassified":
        return f"{source_entity} has unresolved KEGG interaction entries"
    if source_entity and description:
        return f"{source_entity} has a clinically important interaction: {description}"
    return ""


def _labeling_claim_from_record(record: Dict[str, Any]) -> str:
    source_entity = str(record.get("source_entity", "")).strip()
    relationship = str(record.get("relationship", "")).strip().lower()
    snippet = _clean_label_text(str(record.get("evidence_text", "") or ""))
    target_entity = str(record.get("target_entity", "")).strip()

    if relationship == "indicated_for" and snippet:
        return f"{source_entity}: {snippet}"
    if relationship == "has_warning" and snippet:
        return f"{source_entity} warning: {snippet}"
    if relationship == "has_adverse_reaction" and snippet:
        return f"{source_entity} adverse reactions: {snippet}"
    if relationship == "interacts_with" and snippet:
        return f"{source_entity} interaction information: {snippet}"
    if relationship == "has_mechanism" and snippet:
        return f"{source_entity} mechanism: {snippet}"
    if relationship == "has_patient_drug_info" and target_entity:
        return f"{source_entity} has patient guidance: {target_entity}"
    if relationship == "has_official_label":
        if snippet:
            return f"{source_entity} official label summary: {snippet}"
        if target_entity:
            return f"{source_entity} official label available: {target_entity}"
    return ""


def _pgx_claim_from_record(record: Dict[str, Any]) -> str:
    source_entity = str(record.get("source_entity", "")).strip()
    target_entity = str(record.get("target_entity", "")).strip()
    relationship = str(record.get("relationship", "")).strip().lower()
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}

    cpic_level = str(metadata.get("cpiclevel") or record.get("cpiclevel") or "").strip()
    clinpgx_level = str(metadata.get("clinpgxlevel") or record.get("clinpgxlevel") or "").strip()
    pgx_testing = str(metadata.get("pgxtesting") or record.get("pgxtesting") or "").strip()
    actionable = bool(metadata.get("usedforrecommendation") or record.get("usedforrecommendation"))
    mechanism_summary = str(
        metadata.get("mechanism_summary") or record.get("mechanism_summary") or ""
    ).strip()
    guidance_summary = str(
        metadata.get("guidance_summary") or record.get("guidance_summary") or ""
    ).strip()

    if "guideline" in relationship:
        details: List[str] = []
        if cpic_level:
            details.append(f"CPIC level {cpic_level}")
        if clinpgx_level:
            details.append(f"ClinPGx {clinpgx_level}")
        if actionable or pgx_testing.lower().startswith("actionable"):
            details.append("actionable guidance")
        suffix = f" ({'; '.join(details)})" if details else ""
        if mechanism_summary or guidance_summary:
            summary_parts = [part for part in (mechanism_summary, guidance_summary) if part]
            return f"{source_entity} PGx guidance highlights {target_entity}{suffix}: {' '.join(summary_parts)}"
        return f"{source_entity} PGx guidance highlights {target_entity}{suffix}"

    if "pgx" in relationship or "association" in relationship:
        if mechanism_summary:
            return f"{source_entity} has a pharmacogenomic association with {target_entity}: {mechanism_summary}"
        return f"{source_entity} has a pharmacogenomic association with {target_entity}"
    return ""


def _adr_claim_from_record(record: Dict[str, Any]) -> str:
    source_entity = str(record.get("source_entity", "")).strip()
    relationship = str(record.get("relationship", "")).strip().lower()
    target_entity = str(record.get("target_entity", "")).strip()
    if relationship == "causes_adverse_event" and source_entity and target_entity:
        return f"{source_entity} serious safety signal: {target_entity}"
    return ""


def _clean_label_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"^\d+(?:\.\d+)?\s+[A-Z][A-Z\s/&-]{3,40}\s+", "", cleaned)
    cleaned = re.sub(r"^\d+(?:\.\d+)?\s+[A-Z][a-zA-Z\s/&-]{3,40}\s+", "", cleaned)
    for pattern, replacement in (
        (r"\bLactaton\b", "Lactation"),
        (r"\bdoage\b", "dosage"),
    ):
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _looks_like_compound_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"(?:cpd|dr):[A-Z0-9]+", value.strip(), re.IGNORECASE))


def _extract_partner_from_text(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"\bwith\s+([A-Za-z][A-Za-z0-9+/\-\s]{1,80})", text)
    if not match:
        return ""

    candidate = match.group(1)
    candidate = re.split(r"[.;,()]", candidate, maxsplit=1)[0].strip()
    if candidate.lower() in {"dr", "cpd"}:
        return ""
    return candidate


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
