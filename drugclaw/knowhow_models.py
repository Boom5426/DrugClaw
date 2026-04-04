from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


@dataclass(frozen=True)
class KnowHowDocument:
    doc_id: str
    title: str
    task_types: List[str] = field(default_factory=list)
    evidence_types: List[str] = field(default_factory=list)
    declared_by_skills: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    conflict_policy: str = ""
    answer_template: str = ""
    max_prompt_snippets: int = 1
    body_path: str = ""
    body_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", str(self.doc_id or "").strip())
        object.__setattr__(self, "title", str(self.title or "").strip())
        object.__setattr__(self, "task_types", _dedupe_strings(list(self.task_types)))
        object.__setattr__(self, "evidence_types", _dedupe_strings(list(self.evidence_types)))
        object.__setattr__(self, "declared_by_skills", _dedupe_strings(list(self.declared_by_skills)))
        object.__setattr__(self, "risk_level", str(self.risk_level or "medium").strip() or "medium")
        object.__setattr__(self, "conflict_policy", str(self.conflict_policy or "").strip())
        object.__setattr__(self, "answer_template", str(self.answer_template or "").strip())
        object.__setattr__(self, "body_path", str(self.body_path or "").strip())
        object.__setattr__(self, "body_text", str(self.body_text or "").strip())
        try:
            max_prompt_snippets = int(self.max_prompt_snippets)
        except (TypeError, ValueError):
            max_prompt_snippets = 1
        object.__setattr__(self, "max_prompt_snippets", max(1, max_prompt_snippets))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowHowSnippetSelection:
    doc_id: str
    title: str
    task_id: str
    task_type: str
    snippet: str
    risk_level: str = "medium"
    evidence_types: List[str] = field(default_factory=list)
    declared_by_skills: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", str(self.doc_id or "").strip())
        object.__setattr__(self, "title", str(self.title or "").strip())
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(self, "task_type", str(self.task_type or "").strip())
        object.__setattr__(self, "snippet", str(self.snippet or "").strip())
        object.__setattr__(self, "risk_level", str(self.risk_level or "medium").strip() or "medium")
        object.__setattr__(self, "evidence_types", _dedupe_strings(list(self.evidence_types)))
        object.__setattr__(self, "declared_by_skills", _dedupe_strings(list(self.declared_by_skills)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
