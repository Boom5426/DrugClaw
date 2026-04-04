from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceItem
from .query_plan import normalize_task_type


@dataclass(frozen=True)
class EvidenceClassification:
    tier: str
    slot: str
    rationale: str = ""


def classify_evidence_item(task_type: str, item: EvidenceItem) -> EvidenceClassification:
    normalized_task_type = normalize_task_type(task_type)
    if normalized_task_type == "direct_targets":
        return _classify_direct_target_item(item)
    if normalized_task_type == "repurposing_evidence":
        return _classify_drug_repurposing_item(item)

    return EvidenceClassification(
        tier="generic_weak_support",
        slot="additional_context",
        rationale=f"no specialized policy for task type {normalized_task_type!r}",
    )


def _classify_direct_target_item(item: EvidenceItem) -> EvidenceClassification:
    source_skill = str(getattr(item, "source_skill", "")).strip()
    relationship = str(item.metadata.get("relationship", "")).strip().lower()
    target_type = str(item.metadata.get("target_type", "")).strip().lower()
    evidence_kind = str(getattr(item, "evidence_kind", "")).strip().lower()

    if target_type not in {"protein", "gene"}:
        return EvidenceClassification(
            tier="generic_weak_support",
            slot="additional_context",
            rationale="Direct target sections only admit protein/gene target evidence.",
        )

    direct_relationship = any(
        marker in relationship
        for marker in (
            "target",
            "bind",
            "activity",
            "inhib",
            "agon",
            "antagon",
            "substr",
            "modulat",
            "block",
            "activat",
        )
    )

    if (
        source_skill in {"BindingDB", "ChEMBL", "DrugBank", "DrugCentral", "IUPHAR", "TTD"}
        and direct_relationship
        and evidence_kind in {"database_record", "literature_statement", "label_text"}
    ):
        return EvidenceClassification(
            tier="strong_direct",
            slot="established_direct_targets",
            rationale="Curated or measured target evidence from direct target resources counts as established direct-target support.",
        )

    if source_skill in {"DGIdb", "Open Targets Platform", "STITCH", "TarKG", "Molecular Targets", "Molecular Targets Data"}:
        return EvidenceClassification(
            tier="association_only",
            slot="association_only_signals",
            rationale="Broad interaction/association resources cannot be promoted to established direct targets without stronger direct support.",
        )

    if evidence_kind == "model_prediction" or relationship == "linked_target":
        return EvidenceClassification(
            tier="association_only",
            slot="association_only_signals",
            rationale="Predictive or linked-target evidence is kept as association-only support.",
        )

    if direct_relationship:
        return EvidenceClassification(
            tier="generic_weak_support",
            slot="association_only_signals",
            rationale="Unmapped target-like evidence is kept separate from established direct targets.",
        )

    return EvidenceClassification(
        tier="generic_weak_support",
        slot="additional_context",
        rationale="Record is not specific enough for direct-target presentation.",
    )


def _classify_drug_repurposing_item(item: EvidenceItem) -> EvidenceClassification:
    source_skill = str(getattr(item, "source_skill", "")).strip()
    relationship = str(item.metadata.get("relationship", "")).strip().lower()
    target_type = str(item.metadata.get("target_type", "")).strip().lower()

    if source_skill == "RepoDB" and relationship == "repurposing_evidence":
        return EvidenceClassification(
            tier="strong_structured",
            slot="repurposing_evidence",
            rationale="RepoDB repurposing rows are the Phase 2A primary source for repurposing evidence.",
        )

    if (
        source_skill in {"DrugBank", "DrugCentral"}
        and relationship == "indicated_for"
        and target_type == "disease"
    ):
        return EvidenceClassification(
            tier="strong_structured",
            slot="approved_indications",
            rationale="Structured disease-level indication rows from primary indication sources count as strong evidence.",
        )

    if source_skill in {"DailyMed", "openFDA Human Drug"} and relationship == "indicated_for":
        return EvidenceClassification(
            tier="secondary_official_support",
            slot="approved_indications",
            rationale="Official label-derived indication support is secondary official support in Phase 2A.",
        )

    if source_skill == "Open Targets Platform" and relationship in {"indicated_for", "investigated_for"}:
        return EvidenceClassification(
            tier="generic_weak_support",
            slot="approved_indications",
            rationale="Open Targets clinical-stage indication rows are useful support but not primary approved-indication evidence in Phase 2A.",
        )

    if source_skill in {"DailyMed", "openFDA Human Drug"}:
        return EvidenceClassification(
            tier="generic_weak_support",
            slot="additional_context",
            rationale="Official label sections outside direct indication support cannot be promoted to repurposing evidence.",
        )

    if source_skill == "DrugCentral" and relationship == "has_approved_entry":
        return EvidenceClassification(
            tier="generic_weak_support",
            slot="approved_indications",
            rationale="Approval roster rows without disease-level indication metadata are weaker than structured indication records.",
        )

    return EvidenceClassification(
        tier="generic_weak_support",
        slot="additional_context",
        rationale="Unmapped evidence defaults to weak contextual support.",
    )
