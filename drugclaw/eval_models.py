from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .query_plan import normalize_task_type, normalize_question_type


def _dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


@dataclass(frozen=True)
class EvalExpectation:
    expected_evidence_types: List[str] = field(default_factory=list)
    expected_resources: List[str] = field(default_factory=list)
    expected_answer_sections: List[str] = field(default_factory=list)
    scorer: str = "classification_exact"
    gold_notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expected_evidence_types", _dedupe_strings(list(self.expected_evidence_types))
        )
        object.__setattr__(
            self, "expected_resources", _dedupe_strings(list(self.expected_resources))
        )
        object.__setattr__(
            self, "expected_answer_sections", _dedupe_strings(list(self.expected_answer_sections))
        )
        object.__setattr__(self, "scorer", str(self.scorer or "classification_exact").strip())
        object.__setattr__(self, "gold_notes", str(self.gold_notes or "").strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalTaskCase:
    task_id: str
    dataset_name: str
    plan_type: str
    task_type: str
    supporting_task_types: List[str] = field(default_factory=list)
    legacy_question_type: str = ""
    query: str = ""
    expectation: EvalExpectation = field(default_factory=EvalExpectation)

    def __post_init__(self) -> None:
        normalized_task_type = normalize_task_type(self.task_type)
        normalized_supporting = [
            normalize_task_type(task_type)
            for task_type in list(self.supporting_task_types)
        ]
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(self, "dataset_name", str(self.dataset_name or "").strip())
        object.__setattr__(self, "plan_type", str(self.plan_type or "single_task").strip() or "single_task")
        object.__setattr__(self, "task_type", normalized_task_type)
        object.__setattr__(self, "supporting_task_types", _dedupe_strings(normalized_supporting))
        object.__setattr__(
            self,
            "legacy_question_type",
            normalize_question_type(self.legacy_question_type or self.task_type),
        )
        object.__setattr__(self, "query", str(self.query or "").strip())

    @property
    def expected_evidence_types(self) -> List[str]:
        return list(self.expectation.expected_evidence_types)

    @property
    def expected_resources(self) -> List[str]:
        return list(self.expectation.expected_resources)

    @property
    def expected_answer_sections(self) -> List[str]:
        return list(self.expectation.expected_answer_sections)

    @property
    def scorer(self) -> str:
        return self.expectation.scorer

    @property
    def gold_notes(self) -> str:
        return self.expectation.gold_notes

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["expected_evidence_types"] = self.expected_evidence_types
        payload["expected_resources"] = self.expected_resources
        payload["expected_answer_sections"] = self.expected_answer_sections
        payload["scorer"] = self.scorer
        payload["gold_notes"] = self.gold_notes
        return payload


@dataclass(frozen=True)
class EvalScore:
    task_id: str
    dataset_name: str
    plan_type: str
    task_type: str
    legacy_question_type: str
    scorer: str
    score: float
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(self, "dataset_name", str(self.dataset_name or "").strip())
        object.__setattr__(self, "plan_type", str(self.plan_type or "single_task").strip() or "single_task")
        object.__setattr__(self, "task_type", normalize_task_type(self.task_type))
        object.__setattr__(self, "legacy_question_type", normalize_question_type(self.legacy_question_type))
        object.__setattr__(self, "scorer", str(self.scorer or "").strip())
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "error", str(self.error or "").strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalRunSummary:
    total_cases: int
    completed_cases: int
    failed_cases: int
    task_success_rate: float
    evidence_quality_score: float = 0.0
    authority_source_rate: float = 0.0
    knowhow_hit_rate: float = 0.0
    package_ready_rate: float = 0.0
    scores: List[EvalScore] = field(default_factory=list)
    dataset_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dataset_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    task_type_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "failed_cases": self.failed_cases,
            "task_success_rate": self.task_success_rate,
            "evidence_quality_score": self.evidence_quality_score,
            "authority_source_rate": self.authority_source_rate,
            "knowhow_hit_rate": self.knowhow_hit_rate,
            "package_ready_rate": self.package_ready_rate,
            "scores": [score.to_dict() for score in self.scores],
            "dataset_results": dict(self.dataset_results),
            "dataset_breakdown": dict(self.dataset_breakdown),
            "task_type_breakdown": dict(self.task_type_breakdown),
        }
