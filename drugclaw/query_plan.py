from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence


PRIMARY_TARGET_LOOKUP_SKILLS = (
    "BindingDB",
    "ChEMBL",
    "DGIdb",
    "Open Targets Platform",
)

SECONDARY_TARGET_LOOKUP_SKILLS = (
    "DrugBank",
    "TTD",
    "STITCH",
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
    "drug_target_and_mechanism_of_action_lookup": "mechanism",
    "drug_safety_adverse_reactions": "adr",
    "drug_drug_interaction_mechanism_query_with_clinical_relevance_prioritization": "ddi_mechanism",
    "drug_prescribing_and_clinical_use_summary": "labeling",
    "pharmacogenomics_mechanism_and_clinical_impact_query": "pharmacogenomics",
}

TASK_TYPE_ALIASES = {
    "target_lookup": "direct_targets",
    "drug_target_interaction": "direct_targets",
    "drug_target_identification": "direct_targets",
    "relationship_retrieval": "direct_targets",
    "mechanism": "mechanism_of_action",
    "drug_repurposing": "repurposing_evidence",
    "adr": "major_adrs",
    "ddi": "clinically_relevant_ddi",
    "ddi_mechanism": "ddi_mechanism",
    "labeling": "labeling_summary",
    "pharmacogenomics": "pgx_guidance",
}

TASK_TYPE_TO_LEGACY_QUESTION_TYPE = {
    "unknown": "unknown",
    "direct_targets": "target_lookup",
    "target_profile": "target_lookup",
    "mechanism_of_action": "mechanism",
    "labeling_summary": "labeling",
    "major_adrs": "adr",
    "clinically_relevant_ddi": "ddi",
    "ddi_mechanism": "ddi_mechanism",
    "pgx_guidance": "pharmacogenomics",
    "repurposing_evidence": "drug_repurposing",
    "comparative_analysis": "unknown",
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

SUPPORTED_TASK_TYPES = {
    "unknown",
    "direct_targets",
    "target_profile",
    "mechanism_of_action",
    "labeling_summary",
    "major_adrs",
    "clinically_relevant_ddi",
    "ddi_mechanism",
    "pgx_guidance",
    "repurposing_evidence",
    "comparative_analysis",
}

HIGH_RISK_TASK_TYPES = {
    "labeling_summary",
    "major_adrs",
    "clinically_relevant_ddi",
    "ddi_mechanism",
    "pgx_guidance",
}


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _dedupe_strings(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray, dict)):
        return []

    normalized: List[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_entities(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}

    normalized: Dict[str, List[str]] = {}
    for key, raw in value.items():
        items = _dedupe_strings(raw)
        if items:
            normalized[str(key)] = items
    return normalized


def _normalize_knowhow_hints(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return []

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in value:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict):
            continue
        normalized_item = {
            "doc_id": str(item.get("doc_id", "")).strip(),
            "title": str(item.get("title", "")).strip(),
            "task_id": str(item.get("task_id", "")).strip(),
            "task_type": normalize_task_type(item.get("task_type", "")),
            "snippet": str(item.get("snippet", "")).strip(),
            "risk_level": str(item.get("risk_level", "")).strip(),
            "evidence_types": _dedupe_strings(item.get("evidence_types")),
            "declared_by_skills": _dedupe_strings(item.get("declared_by_skills")),
        }
        key = (
            normalized_item["task_id"],
            normalized_item["doc_id"],
            normalized_item["snippet"],
        )
        if key in seen or not normalized_item["doc_id"]:
            continue
        seen.add(key)
        normalized.append(normalized_item)
    return normalized


def normalize_plan_type(value: str) -> str:
    normalized = _normalize_token(value)
    if normalized == "composite_query":
        return "composite_query"
    return "single_task"


def normalize_task_type(value: str) -> str:
    normalized = _normalize_token(value)
    if not normalized:
        return "unknown"
    if normalized in TASK_TYPE_ALIASES:
        return TASK_TYPE_ALIASES[normalized]
    if normalized in SUPPORTED_TASK_TYPES:
        return normalized
    if "mechanism_of_action" in normalized and "target" in normalized:
        return "mechanism_of_action"
    if "repurpos" in normalized:
        return "repurposing_evidence"
    if "pharmacogenom" in normalized or normalized.startswith("pgx"):
        return "pgx_guidance"
    if ("interaction" in normalized or "ddi" in normalized) and "mechanism" in normalized:
        return "ddi_mechanism"
    if "interaction" in normalized or normalized == "ddi":
        return "clinically_relevant_ddi"
    if "label" in normalized or "prescribing" in normalized or "clinical_use" in normalized:
        return "labeling_summary"
    if "adverse" in normalized or "safety" in normalized or normalized == "adr":
        return "major_adrs"
    if "target" in normalized:
        return "direct_targets"
    if "mechanism" in normalized:
        return "mechanism_of_action"
    if "compar" in normalized or normalized in {"vs", "versus"}:
        return "comparative_analysis"
    return "unknown"


def normalize_question_type(value: str) -> str:
    normalized = _normalize_token(value)
    if not normalized:
        return "unknown"
    if normalized in TASK_TYPE_TO_LEGACY_QUESTION_TYPE:
        return TASK_TYPE_TO_LEGACY_QUESTION_TYPE[normalized]
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
    if "prescribing" in normalized or "clinical_use" in normalized or "label" in normalized:
        return "labeling"
    if "adverse_reaction" in normalized or "adverse_event" in normalized or "safety" in normalized or normalized == "adr":
        return "adr"
    return normalized


def is_supported_question_type(value: str) -> bool:
    return normalize_question_type(value) in SUPPORTED_QUESTION_TYPES


def _merge_entities_from_tasks(tasks: Sequence["AnswerTask"]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for task in tasks:
        for entity_type, values in task.entities.items():
            existing = merged.get(entity_type, [])
            for value in values:
                if value not in existing:
                    existing.append(value)
            if existing:
                merged[entity_type] = existing
    return merged


def _aggregate_risk_level(levels: Sequence[str]) -> str:
    normalized = {_normalize_token(level) for level in levels if str(level).strip()}
    if "high" in normalized:
        return "high"
    if "medium" in normalized:
        return "medium"
    if "low" in normalized:
        return "low"
    return "medium"


def legacy_question_type_for_task_type(task_type: str) -> str:
    normalized = normalize_task_type(task_type)
    return TASK_TYPE_TO_LEGACY_QUESTION_TYPE.get(normalized, "unknown")


def legacy_question_type_for_plan(
    primary_task_type: str,
    supporting_task_types: Sequence[str],
) -> str:
    task_types = [normalize_task_type(primary_task_type)] + [
        normalize_task_type(task_type) for task_type in supporting_task_types
    ]
    if "ddi_mechanism" in task_types:
        return "ddi_mechanism"
    if "mechanism_of_action" in task_types and "direct_targets" in task_types:
        return "mechanism"
    return legacy_question_type_for_task_type(primary_task_type)


def preferred_skills_for_task_type(task_type: str) -> List[str]:
    normalized = normalize_task_type(task_type)
    mapping = {
        "direct_targets": [
            "BindingDB",
            "ChEMBL",
            "DGIdb",
            "Open Targets Platform",
        ],
        "target_profile": [
            "BindingDB",
            "ChEMBL",
            "Open Targets Platform",
            "DGIdb",
            "STITCH",
            "TarKG",
        ],
        "mechanism_of_action": [
            "Open Targets Platform",
            "DRUGMECHDB",
            "BindingDB",
            "ChEMBL",
        ],
        "repurposing_evidence": [
            "RepoDB",
            "DrugCentral",
            "DrugBank",
            "DailyMed",
            "openFDA Human Drug",
        ],
        "major_adrs": [
            "ADReCS",
            "FAERS",
            "nSIDES",
            "SIDER",
        ],
        "clinically_relevant_ddi": [
            "DDInter",
            "KEGG Drug",
            "MecDDI",
        ],
        "ddi_mechanism": [
            "DDInter",
            "KEGG Drug",
            "MecDDI",
        ],
        "labeling_summary": [
            "DailyMed",
            "openFDA Human Drug",
            "MedlinePlus Drug Info",
        ],
        "pgx_guidance": [
            "PharmGKB",
            "CPIC",
        ],
        "comparative_analysis": [],
        "unknown": [],
    }
    return list(mapping.get(normalized, []))


def preferred_skills_for_question_type(question_type: str) -> List[str]:
    task_type = normalize_task_type(question_type)
    return preferred_skills_for_task_type(task_type)


def risk_level_for_task_type(task_type: str) -> str:
    normalized = normalize_task_type(task_type)
    if normalized in HIGH_RISK_TASK_TYPES:
        return "high"
    if normalized == "comparative_analysis":
        return "medium"
    return "medium"


def risk_level_for_question_type(question_type: str) -> str:
    task_type = normalize_task_type(question_type)
    return risk_level_for_task_type(task_type)


def _default_requires_graph_reasoning(task_type: str) -> bool:
    return False


def _build_default_answer_contract(task_types: Sequence[str]) -> "AnswerContract":
    normalized = [normalize_task_type(task_type) for task_type in task_types if task_type]
    section_order = ["summary"]
    for task_type in normalized:
        if task_type not in section_order:
            section_order.append(task_type)
    if "limitations" not in section_order:
        section_order.append("limitations")
    return AnswerContract(
        summary_style="direct_answer_first",
        section_order=section_order,
    )


@dataclass
class AnswerTask:
    task_type: str
    question: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    preferred_skills: List[str] = field(default_factory=list)
    preferred_evidence_types: List[str] = field(default_factory=list)
    knowhow_doc_ids: List[str] = field(default_factory=list)
    knowhow_hints: List[Dict[str, Any]] = field(default_factory=list)
    requires_graph_reasoning: bool = False
    requires_prediction_sources: bool = False
    requires_web_fallback: bool = False
    answer_risk_level: str = "medium"
    notes: List[str] = field(default_factory=list)
    task_id: str = ""

    def __post_init__(self) -> None:
        self.task_type = normalize_task_type(self.task_type)
        self.question = str(self.question or "").strip()
        self.entities = _normalize_entities(self.entities)
        self.preferred_skills = _dedupe_strings(self.preferred_skills) or preferred_skills_for_task_type(self.task_type)
        self.preferred_evidence_types = _dedupe_strings(self.preferred_evidence_types)
        self.knowhow_doc_ids = _dedupe_strings(self.knowhow_doc_ids)
        self.knowhow_hints = _normalize_knowhow_hints(self.knowhow_hints)
        self.answer_risk_level = str(self.answer_risk_level or risk_level_for_task_type(self.task_type)).strip() or risk_level_for_task_type(self.task_type)
        self.notes = _dedupe_strings(self.notes)
        if not self.task_id:
            self.task_id = self.task_type or "task"
        if not self.requires_graph_reasoning:
            self.requires_graph_reasoning = _default_requires_graph_reasoning(self.task_type)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTask:
    task_id: str
    priority: int = 100
    task_type: str = ""
    depends_on: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id or "").strip() or "task"
        self.task_type = normalize_task_type(self.task_type) if self.task_type else ""
        try:
            self.priority = int(self.priority)
        except (TypeError, ValueError):
            self.priority = 100
        self.depends_on = _dedupe_strings(self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerContract:
    summary_style: str = "direct_answer_first"
    section_order: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.summary_style = str(self.summary_style or "direct_answer_first").strip() or "direct_answer_first"
        self.section_order = _dedupe_strings(self.section_order)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_answer_task(value: Any, *, default_task_id: str = "") -> AnswerTask | None:
    if value is None:
        return None
    if isinstance(value, AnswerTask):
        if default_task_id and not value.task_id:
            value.task_id = default_task_id
        return value
    if isinstance(value, str):
        return AnswerTask(
            task_id=default_task_id,
            task_type=value,
        )
    if isinstance(value, dict):
        payload = dict(value)
        if "task_type" not in payload:
            if payload.get("task"):
                payload["task_type"] = payload.pop("task")
            elif payload.get("type"):
                payload["task_type"] = payload.pop("type")
        if "task_id" not in payload and payload.get("id"):
            payload["task_id"] = payload.pop("id")
        if default_task_id and not payload.get("task_id"):
            payload["task_id"] = default_task_id
        allowed_fields = {
            "task_type",
            "question",
            "entities",
            "preferred_skills",
            "preferred_evidence_types",
            "knowhow_doc_ids",
            "knowhow_hints",
            "requires_graph_reasoning",
            "requires_prediction_sources",
            "requires_web_fallback",
            "answer_risk_level",
            "notes",
            "task_id",
        }
        sanitized_payload = {
            key: val for key, val in payload.items() if key in allowed_fields
        }
        return AnswerTask(**sanitized_payload)
    raise TypeError(f"Unsupported AnswerTask value: {type(value)!r}")


def _coerce_execution_task(value: Any, *, default_task_id: str = "", default_task_type: str = "", priority: int = 100) -> ExecutionTask:
    if isinstance(value, ExecutionTask):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        if "task_type" not in payload:
            if payload.get("task"):
                payload["task_type"] = payload.pop("task")
            elif payload.get("type"):
                payload["task_type"] = payload.pop("type")
        if "task_id" not in payload and payload.get("id"):
            payload["task_id"] = payload.pop("id")
        if default_task_id and not payload.get("task_id"):
            payload["task_id"] = default_task_id
        if default_task_type and not payload.get("task_type"):
            payload["task_type"] = default_task_type
        if "priority" not in payload:
            payload["priority"] = priority
        allowed_fields = {"task_id", "priority", "task_type", "depends_on"}
        sanitized_payload = {
            key: val for key, val in payload.items() if key in allowed_fields
        }
        return ExecutionTask(**sanitized_payload)
    return ExecutionTask(
        task_id=default_task_id or str(value or "task"),
        priority=priority,
        task_type=default_task_type,
    )


def _coerce_answer_contract(value: Any, *, task_types: Sequence[str]) -> AnswerContract:
    if isinstance(value, AnswerContract):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        if "section_order" not in payload and payload.get("sections"):
            payload["section_order"] = payload.pop("sections")
        allowed_fields = {"summary_style", "section_order"}
        sanitized_payload = {
            key: val for key, val in payload.items() if key in allowed_fields
        }
        return AnswerContract(**sanitized_payload)
    return _build_default_answer_contract(task_types)


def _should_collapse_target_only_projection(
    primary_task: AnswerTask | None,
    supporting_tasks: Sequence[AnswerTask],
) -> bool:
    if primary_task is None:
        return False
    if normalize_task_type(primary_task.task_type) != "direct_targets":
        return False
    if not supporting_tasks:
        return False

    normalized_supporting = [
        normalize_task_type(task.task_type)
        for task in supporting_tasks
        if normalize_task_type(task.task_type) != "unknown"
    ]
    if not normalized_supporting:
        return False

    return all(
        task_type in {"direct_targets", "target_profile"}
        for task_type in normalized_supporting
    )


@dataclass
class QueryPlan:
    question_type: str = "unknown"
    entities: Dict[str, List[str]] = field(default_factory=dict)
    subquestions: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    preferred_evidence_types: List[str] = field(default_factory=list)
    requires_graph_reasoning: bool = False
    requires_prediction_sources: bool = False
    requires_web_fallback: bool = False
    answer_risk_level: str = "medium"
    notes: List[str] = field(default_factory=list)
    knowhow_doc_ids: List[str] = field(default_factory=list)
    knowhow_hints: List[Dict[str, Any]] = field(default_factory=list)
    plan_type: str = "single_task"
    primary_task: AnswerTask | Dict[str, Any] | None = None
    supporting_tasks: List[AnswerTask | Dict[str, Any]] = field(default_factory=list)
    execution_tasks: List[ExecutionTask | Dict[str, Any]] = field(default_factory=list)
    answer_contract: AnswerContract | Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.question_type = normalize_question_type(self.question_type)
        self.entities = _normalize_entities(self.entities)
        self.subquestions = _dedupe_strings(self.subquestions)
        self.preferred_skills = _dedupe_strings(self.preferred_skills)
        self.preferred_evidence_types = _dedupe_strings(self.preferred_evidence_types)
        self.answer_risk_level = str(self.answer_risk_level or "medium").strip() or "medium"
        self.notes = _dedupe_strings(self.notes)
        self.knowhow_doc_ids = _dedupe_strings(self.knowhow_doc_ids)
        self.knowhow_hints = _normalize_knowhow_hints(self.knowhow_hints)
        self.plan_type = normalize_plan_type(
            self.plan_type or ("composite_query" if self.supporting_tasks else "single_task")
        )

        primary_task = _coerce_answer_task(self.primary_task, default_task_id="primary")
        if primary_task is None:
            primary_task = AnswerTask(
                task_id="primary",
                task_type=normalize_task_type(self.question_type),
                question=self.subquestions[0] if self.subquestions else "",
                entities=self.entities,
                preferred_skills=self.preferred_skills,
                preferred_evidence_types=self.preferred_evidence_types,
                requires_graph_reasoning=self.requires_graph_reasoning,
                requires_prediction_sources=self.requires_prediction_sources,
                requires_web_fallback=self.requires_web_fallback,
                answer_risk_level=self.answer_risk_level,
                notes=self.notes,
            )
        self.primary_task = primary_task

        normalized_supporting: List[AnswerTask] = []
        for index, task in enumerate(self.supporting_tasks or [], start=1):
            normalized_supporting.append(
                _coerce_answer_task(task, default_task_id=f"support_{index}")
            )
        if _should_collapse_target_only_projection(primary_task, normalized_supporting):
            normalized_supporting = []
        self.supporting_tasks = normalized_supporting
        if self.supporting_tasks:
            self.plan_type = "composite_query"
        else:
            self.plan_type = "single_task"

        all_tasks = [self.primary_task] + list(self.supporting_tasks)

        if not self.execution_tasks:
            generated_execution_tasks = []
            for index, task in enumerate(all_tasks):
                generated_execution_tasks.append(
                    ExecutionTask(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        priority=max(10, 100 - (index * 20)),
                    )
                )
            self.execution_tasks = generated_execution_tasks
        else:
            normalized_execution: List[ExecutionTask] = []
            for index, value in enumerate(self.execution_tasks):
                task = all_tasks[index] if index < len(all_tasks) else None
                normalized_execution.append(
                    _coerce_execution_task(
                        value,
                        default_task_id=task.task_id if task is not None else f"task_{index + 1}",
                        default_task_type=task.task_type if task is not None else "",
                        priority=max(10, 100 - (index * 20)),
                    )
                )
            self.execution_tasks = normalized_execution

        task_types = [task.task_type for task in all_tasks]
        self.answer_contract = _coerce_answer_contract(self.answer_contract, task_types=task_types)

        merged_entities = _merge_entities_from_tasks(all_tasks)
        if merged_entities:
            self.entities = merged_entities

        if not self.subquestions:
            self.subquestions = [task.question for task in all_tasks if task.question]

        aggregated_skills: List[str] = []
        aggregated_evidence_types: List[str] = []
        aggregated_notes: List[str] = list(self.notes)
        aggregated_knowhow_doc_ids: List[str] = list(self.knowhow_doc_ids)
        aggregated_knowhow_hints: List[Dict[str, Any]] = list(self.knowhow_hints)
        for task in all_tasks:
            for skill in task.preferred_skills:
                if skill not in aggregated_skills:
                    aggregated_skills.append(skill)
            for evidence_type in task.preferred_evidence_types:
                if evidence_type not in aggregated_evidence_types:
                    aggregated_evidence_types.append(evidence_type)
            for note in task.notes:
                if note not in aggregated_notes:
                    aggregated_notes.append(note)
            for doc_id in task.knowhow_doc_ids:
                if doc_id not in aggregated_knowhow_doc_ids:
                    aggregated_knowhow_doc_ids.append(doc_id)
            aggregated_knowhow_hints.extend(task.knowhow_hints)

        if aggregated_skills and not self.preferred_skills:
            self.preferred_skills = aggregated_skills
        if aggregated_evidence_types and not self.preferred_evidence_types:
            self.preferred_evidence_types = aggregated_evidence_types
        self.notes = aggregated_notes
        self.knowhow_doc_ids = aggregated_knowhow_doc_ids
        self.knowhow_hints = _normalize_knowhow_hints(aggregated_knowhow_hints)
        self.requires_graph_reasoning = any(task.requires_graph_reasoning for task in all_tasks)
        self.requires_prediction_sources = any(task.requires_prediction_sources for task in all_tasks)
        self.requires_web_fallback = any(task.requires_web_fallback for task in all_tasks)
        self.answer_risk_level = _aggregate_risk_level(
            [self.answer_risk_level] + [task.answer_risk_level for task in all_tasks]
        )
        self.question_type = legacy_question_type_for_plan(
            self.primary_task.task_type,
            [task.task_type for task in self.supporting_tasks],
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _query_flags(query: str) -> Dict[str, bool]:
    lowered = str(query).strip().lower()
    if not lowered:
        return {
            "pgx": False,
            "ddi": False,
            "mechanism": False,
            "targets": False,
            "repurposing": False,
            "labeling": False,
            "adr": False,
            "comparative": False,
        }

    return {
        "pgx": any(
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
        ),
        "ddi": any(
            marker in lowered
            for marker in (
                "drug-drug interaction",
                "drug drug interaction",
                "drug interactions",
                "interactions of",
                "interact with",
                "interaction with",
            )
        ),
        "mechanism": any(
            marker in lowered
            for marker in (
                "mechanism of action",
                " moa",
                " moa?",
                " mechanism ",
                " mechanisms",
            )
        ),
        "targets": is_direct_target_lookup(query=lowered),
        "repurposing": any(
            marker in lowered
            for marker in (
                "repurposing",
                "repositioning",
                "reposition",
                "approved indications",
                "repurposing evidence",
                "new indications",
            )
        ),
        "labeling": any(
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
        ),
        "adr": any(
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
        ),
        "comparative": any(
            marker in lowered
            for marker in (
                "compare",
                "comparison",
                "versus",
                " vs ",
            )
        ),
    }


def infer_task_type_from_query(query: str) -> str:
    flags = _query_flags(query)
    if flags["pgx"]:
        return "pgx_guidance"
    if flags["ddi"]:
        if flags["mechanism"]:
            return "ddi_mechanism"
        return "clinically_relevant_ddi"
    if flags["comparative"]:
        return "comparative_analysis"
    if flags["repurposing"]:
        return "repurposing_evidence"
    if flags["labeling"]:
        return "labeling_summary"
    if flags["adr"]:
        return "major_adrs"
    if flags["mechanism"]:
        return "mechanism_of_action"
    if flags["targets"]:
        return "direct_targets"
    return "unknown"


def _plan_shape_from_query(query: str) -> tuple[str, str, List[str]]:
    flags = _query_flags(query)

    if flags["ddi"] and flags["mechanism"]:
        return "composite_query", "clinically_relevant_ddi", ["ddi_mechanism"]
    if flags["targets"] and flags["mechanism"]:
        return "composite_query", "direct_targets", ["mechanism_of_action"]

    primary_task_type = infer_task_type_from_query(query)
    return "single_task", primary_task_type, []


def _task_question_from_query(
    *,
    task_type: str,
    query: str,
    entities: Dict[str, List[str]],
) -> str:
    normalized = normalize_task_type(task_type)
    drugs = entities.get("drug", [])
    drug_name = drugs[0] if len(drugs) == 1 else ""

    if normalized == "direct_targets" and drug_name:
        return f"What are the established direct targets of {drug_name}?"
    if normalized == "target_profile" and drug_name:
        return f"What is the broader target profile of {drug_name}?"
    if normalized == "mechanism_of_action" and drug_name:
        return f"What is the mechanism of action of {drug_name}?"
    if normalized == "labeling_summary" and drug_name:
        return f"What prescribing and safety information is available for {drug_name}?"
    if normalized == "major_adrs" and drug_name:
        return f"What are the major adverse reactions of {drug_name}?"
    if normalized == "pgx_guidance" and drug_name:
        return f"What pharmacogenomic guidance is available for {drug_name}?"
    if normalized == "repurposing_evidence" and drug_name:
        return f"What repurposing evidence is available for {drug_name}?"
    return str(query or "").strip()


def _build_task(
    *,
    task_type: str,
    query: str,
    entities: Dict[str, List[str]],
    task_id: str,
    notes: List[str] | None = None,
) -> AnswerTask:
    normalized = normalize_task_type(task_type)
    return AnswerTask(
        task_id=task_id,
        task_type=normalized,
        question=_task_question_from_query(task_type=normalized, query=query, entities=entities),
        entities=entities,
        preferred_skills=preferred_skills_for_task_type(normalized),
        preferred_evidence_types=[],
        requires_graph_reasoning=_default_requires_graph_reasoning(normalized),
        requires_prediction_sources=False,
        requires_web_fallback=normalized in {"mechanism_of_action", "ddi_mechanism", "pgx_guidance"},
        answer_risk_level=risk_level_for_task_type(normalized),
        notes=notes or [],
    )


def build_fallback_query_plan(query: str) -> QueryPlan:
    entities = infer_entities_from_query(query)
    plan_type, primary_task_type, supporting_task_types = _plan_shape_from_query(query)
    primary_task = _build_task(
        task_type=primary_task_type,
        query=query,
        entities=entities,
        task_id="primary",
        notes=["Fallback plan used because planner output was unavailable or invalid."],
    )
    supporting_tasks = [
        _build_task(
            task_type=task_type,
            query=query,
            entities=entities,
            task_id=f"support_{index}",
        )
        for index, task_type in enumerate(supporting_task_types, start=1)
    ]
    return QueryPlan(
        question_type=legacy_question_type_for_plan(
            primary_task.task_type,
            [task.task_type for task in supporting_tasks],
        ),
        entities=entities,
        subquestions=[task.question for task in [primary_task] + supporting_tasks if task.question] or ([query] if query else []),
        preferred_skills=preferred_skills_for_question_type(
            legacy_question_type_for_plan(
                primary_task.task_type,
                [task.task_type for task in supporting_tasks],
            )
        ),
        preferred_evidence_types=[],
        requires_graph_reasoning=False,
        requires_prediction_sources=False,
        requires_web_fallback=False,
        answer_risk_level=risk_level_for_question_type(
            legacy_question_type_for_plan(
                primary_task.task_type,
                [task.task_type for task in supporting_tasks],
            )
        ),
        notes=["Fallback plan used because planner output was unavailable or invalid."],
        plan_type=plan_type,
        primary_task=primary_task,
        supporting_tasks=supporting_tasks,
        answer_contract=_build_default_answer_contract(
            [primary_task.task_type] + [task.task_type for task in supporting_tasks]
        ),
    )


def infer_question_type_from_query(query: str) -> str:
    plan_type, primary_task_type, supporting_task_types = _plan_shape_from_query(query)
    return legacy_question_type_for_plan(primary_task_type, supporting_task_types)


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


def is_direct_target_lookup(*, query: str = "", question_type: str = "") -> bool:
    normalized_task_type = normalize_task_type(question_type)
    if normalized_task_type in {"direct_targets", "target_profile"}:
        return True
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

    if re.search(r"\bdoes\s+[a-z0-9\-]+(?:\s*\([^)]*\))?\s+target\b", lowered_query):
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

    if len(primary) >= 3:
        return primary
    if primary:
        return primary + secondary + remainder
    return secondary + remainder
