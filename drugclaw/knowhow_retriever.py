from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .knowhow_models import KnowHowDocument, KnowHowSnippetSelection
from .knowhow_registry import KnowHowRegistry


class KnowHowRetriever:
    def __init__(self, registry: KnowHowRegistry | None = None) -> None:
        self.registry = registry or KnowHowRegistry()

    def enrich_query_plan(self, plan):
        if plan is None:
            return plan

        answer_contract = getattr(plan, "answer_contract", None)
        task_hints: List[Dict[str, Any]] = []
        task_doc_ids: List[str] = list(getattr(plan, "knowhow_doc_ids", []) or [])
        tasks = [getattr(plan, "primary_task", None)] + list(getattr(plan, "supporting_tasks", []) or [])
        for task in tasks:
            if task is None:
                continue
            selections = self.select_for_task(task, answer_contract=answer_contract)
            task.knowhow_doc_ids = _dedupe_strings(
                list(getattr(task, "knowhow_doc_ids", []) or [])
                + [selection.doc_id for selection in selections]
            )
            task.knowhow_hints = _dedupe_hint_dicts(
                list(getattr(task, "knowhow_hints", []) or [])
                + [selection.to_dict() for selection in selections]
            )
            task_doc_ids.extend(task.knowhow_doc_ids)
            task_hints.extend(task.knowhow_hints)

        plan.knowhow_doc_ids = _dedupe_strings(task_doc_ids)
        plan.knowhow_hints = _dedupe_hint_dicts(
            list(getattr(plan, "knowhow_hints", []) or []) + task_hints
        )
        return plan

    def select_for_task(self, task, *, answer_contract=None) -> List[KnowHowSnippetSelection]:
        preferred_evidence_types = list(getattr(task, "preferred_evidence_types", []) or [])
        preferred_skills = list(getattr(task, "preferred_skills", []) or [])
        risk_level = str(getattr(task, "answer_risk_level", "") or "").strip()
        section_order = list(getattr(answer_contract, "section_order", []) or [])

        scored: List[tuple[int, KnowHowDocument]] = []
        for doc in self.registry.list_documents():
            score = self._score_document(
                doc,
                task_type=getattr(task, "task_type", ""),
                preferred_evidence_types=preferred_evidence_types,
                preferred_skills=preferred_skills,
                risk_level=risk_level,
                section_order=section_order,
            )
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: (-item[0], item[1].title, item[1].doc_id))
        selections: List[KnowHowSnippetSelection] = []
        for _, doc in scored[:3]:
            snippet = self._extract_snippet(doc)
            if not snippet:
                continue
            selections.append(
                KnowHowSnippetSelection(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    task_id=str(getattr(task, "task_id", "") or ""),
                    task_type=str(getattr(task, "task_type", "") or ""),
                    snippet=snippet,
                    risk_level=doc.risk_level,
                    evidence_types=list(doc.evidence_types),
                    declared_by_skills=list(doc.declared_by_skills),
                )
            )
        return selections

    @staticmethod
    def _score_document(
        doc: KnowHowDocument,
        *,
        task_type: str,
        preferred_evidence_types: List[str],
        preferred_skills: List[str],
        risk_level: str,
        section_order: List[str],
    ) -> int:
        normalized_task_type = str(task_type or "").strip()
        if normalized_task_type not in doc.task_types:
            return 0

        score = 5
        if preferred_evidence_types:
            overlap = set(preferred_evidence_types).intersection(doc.evidence_types)
            if overlap:
                score += 2
        if preferred_skills:
            declared_overlap = set(preferred_skills).intersection(doc.declared_by_skills)
            if declared_overlap:
                score += 1
        if risk_level and str(doc.risk_level or "").strip().lower() == str(risk_level).strip().lower():
            score += 2
        if normalized_task_type in section_order:
            score += 1
        if doc.conflict_policy:
            score += 1
        return score

    @staticmethod
    def _extract_snippet(doc: KnowHowDocument) -> str:
        body_text = str(doc.body_text or "").strip()
        if not body_text and doc.body_path:
            body_path = Path(doc.body_path)
            if body_path.exists():
                try:
                    body_text = body_path.read_text(encoding="utf-8").strip()
                except OSError:
                    body_text = ""

        lines = [
            line.strip()
            for line in body_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        snippet = " ".join(lines[:2]).strip()
        if not snippet:
            snippet = str(doc.conflict_policy or doc.answer_template or "").strip()
        if len(snippet) > 280:
            snippet = snippet[:277].rstrip() + "..."
        return snippet


def _dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _dedupe_hint_dicts(values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for value in values:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, dict):
            continue
        key = (
            str(value.get("task_id", "")).strip(),
            str(value.get("doc_id", "")).strip(),
            str(value.get("snippet", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(value))
    return deduped
