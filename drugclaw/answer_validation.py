from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .query_plan import (
    infer_entities_from_query,
    is_direct_target_lookup,
    normalize_question_type,
    normalize_task_type,
)


@dataclass(frozen=True)
class AnswerValidationIssue:
    code: str
    message: str
    severity: str = "warning"


def validate_answer_output(
    *,
    query: str,
    answer_text: str,
    query_plan: Any | None = None,
) -> List[AnswerValidationIssue]:
    issues: List[AnswerValidationIssue] = []
    text = str(answer_text or "")

    for marker in (
        "Established Direct Targets:",
        "Additional Direct Activity Hits:",
        "Association-Only Signals:",
        "Mechanism Coverage:",
    ):
        if text.count(marker) > 1:
            issues.append(
                AnswerValidationIssue(
                    code="duplicate_section_marker",
                    message=f"Answer repeated the section marker {marker!r}.",
                    severity="error",
                )
            )

    primary_task_type = normalize_task_type(
        getattr(getattr(query_plan, "primary_task", None), "task_type", "") or ""
    )
    supporting_task_types = [
        normalize_task_type(getattr(task, "task_type", ""))
        for task in (getattr(query_plan, "supporting_tasks", []) or [])
    ]
    supporting_task_types = [
        task_type
        for task_type in supporting_task_types
        if task_type != "unknown"
    ]
    legacy_question_type = normalize_question_type(
        getattr(query_plan, "question_type", "") or ""
    )

    if is_direct_target_lookup(query=query, question_type=legacy_question_type):
        if supporting_task_types:
            issues.append(
                AnswerValidationIssue(
                    code="unexpected_composite_target_lookup",
                    message="Single-intent target lookup retained supporting tasks.",
                    severity="error",
                )
            )
        if primary_task_type == "direct_targets" and "Short Answer:" in text:
            issues.append(
                AnswerValidationIssue(
                    code="unexpected_short_answer_block",
                    message="Pure direct-target answers should not render a composite short-answer block.",
                    severity="error",
                )
            )

    if legacy_question_type == "labeling" or primary_task_type == "labeling_summary":
        expected_drug_names = _extract_expected_labeling_drug_names(
            query=query,
            query_plan=query_plan,
        )
        labeling_subjects = _extract_labeling_subjects(text)
        lowered_body_text = _strip_query_line(text).lower()

        if expected_drug_names:
            has_expected_subject = any(
                _labeling_subject_matches_expected_drug(subject, expected_drug_names)
                for subject in labeling_subjects
            )
            has_expected_body_mention = any(
                name in lowered_body_text for name in expected_drug_names
            )
            if labeling_subjects and not has_expected_subject:
                issues.append(
                    AnswerValidationIssue(
                        code="canonical_drug_missing_from_labeling_answer",
                        message=(
                            "Labeling answer does not mention the expected drug context "
                            f"({', '.join(repr(name) for name in expected_drug_names)})."
                        ),
                        severity="error",
                    )
                )
            elif not labeling_subjects and not has_expected_body_mention:
                issues.append(
                    AnswerValidationIssue(
                        code="canonical_drug_missing_from_labeling_answer",
                        message=(
                            "Labeling answer does not mention the expected drug context "
                            f"({', '.join(repr(name) for name in expected_drug_names)})."
                        ),
                        severity="error",
                    )
                )

        if (
            expected_drug_names
            and labeling_subjects
            and all(
                not _labeling_subject_matches_expected_drug(
                    subject,
                    expected_drug_names,
                )
                for subject in labeling_subjects[:3]
            )
        ):
            issues.append(
                AnswerValidationIssue(
                    code="wrong_drug_labeling_answer",
                    message=(
                        "Leading labeling claims do not stay aligned with the expected drug context "
                        f"({', '.join(repr(name) for name in expected_drug_names)})."
                    ),
                    severity="error",
                )
            )

    return issues


def _extract_primary_drug_name(*, query: str, query_plan: Any | None = None) -> str:
    plan_entities = getattr(query_plan, "entities", {}) or {}
    plan_drugs = plan_entities.get("drug") or []
    if plan_drugs:
        return str(plan_drugs[0]).strip().lower()

    inferred_entities = infer_entities_from_query(query)
    inferred_drugs = inferred_entities.get("drug") or []
    if inferred_drugs:
        return str(inferred_drugs[0]).strip().lower()
    return ""


def _extract_expected_labeling_drug_names(*, query: str, query_plan: Any | None = None) -> List[str]:
    names: List[str] = []

    primary_drug = _extract_primary_drug_name(query=query, query_plan=query_plan)
    if primary_drug and primary_drug not in names:
        names.append(primary_drug)

    inferred_entities = infer_entities_from_query(query)
    inferred_drugs = inferred_entities.get("drug") or []
    for drug_name in inferred_drugs:
        normalized = str(drug_name).strip().lower()
        if normalized and normalized not in names:
            names.append(normalized)

    return names


def _extract_labeling_subjects(answer_text: str) -> List[str]:
    subjects: List[str] = []
    in_labeling_section = False

    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        if line == "Structured Labeling Findings:":
            in_labeling_section = True
            continue
        if not in_labeling_section:
            continue
        if not line:
            if subjects:
                break
            continue
        if not line.startswith("- "):
            if subjects:
                break
            continue

        body = line[2:].strip()
        subject = body.split(":", 1)[0].strip().lower()
        if subject:
            subjects.append(subject)

    return subjects


def _labeling_subject_matches_expected_drug(subject: str, expected_drug_names: List[str]) -> bool:
    normalized_subject = str(subject or "").strip().lower()
    if not normalized_subject:
        return False
    if _looks_like_combination_subject(normalized_subject):
        return False
    return any(name in normalized_subject for name in expected_drug_names)


def _looks_like_combination_subject(subject: str) -> bool:
    normalized_subject = str(subject or "").strip().lower()
    return any(separator in normalized_subject for separator in (" and ", "/", ";", ","))


def _strip_query_line(answer_text: str) -> str:
    lines = str(answer_text or "").splitlines()
    if lines and lines[0].strip().lower().startswith("query:"):
        return "\n".join(lines[1:])
    return str(answer_text or "")
