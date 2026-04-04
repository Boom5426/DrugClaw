from __future__ import annotations

from drugclaw.answer_validation import validate_answer_output
from drugclaw.query_plan import QueryPlan


def test_answer_validation_flags_duplicate_direct_target_sections() -> None:
    issues = validate_answer_output(
        query="What does imatinib target?",
        answer_text=(
            "Query: What does imatinib target?\n\n"
            "Short Answer:\n- Primary supported answer: ABL1\n\n"
            "Established Direct Targets:\n- imatinib targets ABL1.\n\n"
            "Association-Only Signals:\n- none\n\n"
            "Established Direct Targets:\n- imatinib targets KIT.\n"
        ),
        query_plan=QueryPlan(
            question_type="target_lookup",
            plan_type="composite_query",
            primary_task={"task_type": "direct_targets"},
            supporting_tasks=[{"task_type": "target_profile"}],
        ),
    )

    issue_codes = {issue.code for issue in issues}
    assert "duplicate_section_marker" in issue_codes
    assert "unexpected_short_answer_block" in issue_codes
