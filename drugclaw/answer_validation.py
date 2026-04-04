from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .query_plan import is_direct_target_lookup, normalize_question_type, normalize_task_type


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

    return issues

