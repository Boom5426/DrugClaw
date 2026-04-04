from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .knowhow_models import KnowHowDocument
from .query_plan import normalize_task_type
from .resource_package_registry import load_package_manifests
from .resource_path_resolver import get_repo_root, resolve_path_value


class KnowHowRegistry:
    def __init__(
        self,
        documents: Iterable[KnowHowDocument] | None = None,
        *,
        repo_root: Path | None = None,
    ) -> None:
        if documents is None:
            documents = self._load_documents(repo_root=repo_root)
        self._documents = [doc for doc in documents if doc.doc_id]
        self._documents_by_id: Dict[str, KnowHowDocument] = {
            doc.doc_id: doc for doc in self._documents
        }

    @classmethod
    def from_documents(cls, documents: Iterable[KnowHowDocument]) -> "KnowHowRegistry":
        return cls(list(documents))

    def list_documents(self) -> List[KnowHowDocument]:
        return list(self._documents)

    def get_document(self, doc_id: str) -> KnowHowDocument | None:
        return self._documents_by_id.get(str(doc_id or "").strip())

    def find_documents(
        self,
        *,
        primary_task_type: str = "",
        supporting_task_types: Sequence[str] | None = None,
        evidence_type: str = "",
        risk_level: str = "",
    ) -> List[KnowHowDocument]:
        candidate_task_types = {
            normalize_task_type(primary_task_type)
        }
        candidate_task_types.update(
            normalize_task_type(task_type)
            for task_type in (supporting_task_types or [])
            if str(task_type or "").strip()
        )
        candidate_task_types.discard("unknown")

        normalized_evidence_type = str(evidence_type or "").strip()
        normalized_risk_level = str(risk_level or "").strip().lower()

        matched: List[KnowHowDocument] = []
        for doc in self._documents:
            doc_task_types = {
                normalize_task_type(task_type)
                for task_type in doc.task_types
                if str(task_type or "").strip()
            }
            if candidate_task_types and not candidate_task_types.intersection(doc_task_types):
                continue
            if normalized_evidence_type:
                doc_evidence_types = {str(value).strip() for value in doc.evidence_types if str(value).strip()}
                if doc_evidence_types and normalized_evidence_type not in doc_evidence_types:
                    continue
            if normalized_risk_level and doc.risk_level.lower() not in {"", "any", normalized_risk_level}:
                continue
            matched.append(doc)

        return matched

    @staticmethod
    def _load_documents(repo_root: Path | None = None) -> List[KnowHowDocument]:
        root = repo_root or get_repo_root()
        base_dir = root / "resources_metadata" / "knowhow"
        if not base_dir.exists():
            return []

        path_to_skill_names = _build_knowhow_path_skill_index(root)
        documents: List[KnowHowDocument] = []
        for metadata_path in sorted(base_dir.rglob("*.json")):
            if not metadata_path.is_file():
                continue
            if "examples" in metadata_path.relative_to(base_dir).parts:
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            body_path = str(payload.get("body_path", "") or "").strip()
            if body_path:
                resolved_body_path = str(resolve_path_value(body_path, root))
            else:
                sibling_markdown = metadata_path.with_suffix(".md")
                resolved_body_path = str(sibling_markdown) if sibling_markdown.exists() else ""

            declared_by_skills: List[str] = []
            for candidate_path in (metadata_path, Path(resolved_body_path) if resolved_body_path else None):
                if candidate_path is None:
                    continue
                candidate_key = str(Path(candidate_path).resolve())
                for skill_name in path_to_skill_names.get(candidate_key, []):
                    if skill_name not in declared_by_skills:
                        declared_by_skills.append(skill_name)

            documents.append(
                KnowHowDocument(
                    doc_id=payload.get("doc_id", metadata_path.stem),
                    title=payload.get("title", metadata_path.stem),
                    task_types=list(payload.get("task_types", []) or []),
                    evidence_types=list(payload.get("evidence_types", []) or []),
                    declared_by_skills=declared_by_skills,
                    risk_level=payload.get("risk_level", "medium"),
                    conflict_policy=payload.get("conflict_policy", ""),
                    answer_template=payload.get("answer_template", ""),
                    max_prompt_snippets=payload.get("max_prompt_snippets", 1),
                    body_path=resolved_body_path,
                )
            )

        return documents


def _build_knowhow_path_skill_index(repo_root: Path) -> Dict[str, List[str]]:
    path_to_skill_names: Dict[str, List[str]] = {}
    for manifest in load_package_manifests(repo_root).values():
        for declared_path in manifest.knowhow_docs:
            resolved_path = str(resolve_path_value(declared_path, repo_root).resolve())
            skill_names = path_to_skill_names.setdefault(resolved_path, [])
            if manifest.skill_name and manifest.skill_name not in skill_names:
                skill_names.append(manifest.skill_name)
    return path_to_skill_names
