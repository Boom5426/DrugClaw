from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

from .evidence import EvidenceItem, score_claim_confidence


@dataclass
class ClaimAssessment:
    claim: str
    verdict: str
    supporting_evidence_ids: List[str]
    contradicting_evidence_ids: List[str]
    confidence: float
    rationale: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def assess_claims(evidence_items: Iterable[EvidenceItem]) -> List[ClaimAssessment]:
    grouped: Dict[str, List[EvidenceItem]] = {}
    for item in evidence_items:
        grouped.setdefault(item.claim, []).append(item)

    assessments: List[ClaimAssessment] = []
    for claim, items in grouped.items():
        supporting = [item.evidence_id for item in items if item.support_direction == "supports"]
        contradicting = [
            item.evidence_id for item in items if item.support_direction == "contradicts"
        ]
        verdict = _determine_verdict(items, supporting, contradicting)
        limitations = _build_limitations(items, supporting, contradicting)
        assessments.append(
            ClaimAssessment(
                claim=claim,
                verdict=verdict,
                supporting_evidence_ids=supporting,
                contradicting_evidence_ids=contradicting,
                confidence=score_claim_confidence(items),
                rationale=_build_rationale(verdict, supporting, contradicting),
                limitations=limitations,
            )
        )

    assessments.sort(key=lambda item: item.confidence, reverse=True)
    return assessments


def _determine_verdict(
    items: List[EvidenceItem],
    supporting: List[str],
    contradicting: List[str],
) -> str:
    if supporting and contradicting:
        return "uncertain"
    if contradicting and not supporting:
        return "contradicted"
    if supporting:
        return "supported"
    if items:
        return "insufficient"
    return "insufficient"


def _build_rationale(
    verdict: str,
    supporting: List[str],
    contradicting: List[str],
) -> str:
    if verdict == "supported":
        return f"Supported by {len(supporting)} evidence item(s)."
    if verdict == "contradicted":
        return f"Only contradicting evidence was found ({len(contradicting)} item(s))."
    if verdict == "uncertain":
        return (
            f"Conflicting evidence detected: {len(supporting)} supporting and "
            f"{len(contradicting)} contradicting item(s)."
        )
    return "Evidence is too sparse or indirect to support a stronger verdict."


def _build_limitations(
    items: List[EvidenceItem],
    supporting: List[str],
    contradicting: List[str],
) -> List[str]:
    limitations: List[str] = []

    if supporting and len(supporting) == 1 and not contradicting:
        limitations.append("Claim relies on a single supporting evidence item.")
    if contradicting:
        limitations.append("Conflicting evidence is present.")
    if items and all(item.evidence_kind == "model_prediction" for item in items):
        limitations.append("Claim is supported only by prediction-oriented evidence.")
    if any(len(item.snippet or "") < 40 for item in items):
        limitations.append("Some evidence snippets are short or lack context.")

    return limitations
