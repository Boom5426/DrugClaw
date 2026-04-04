from __future__ import annotations

import pytest

from drugclaw.evidence import EvidenceItem
from drugclaw.eval_models import EvalExpectation, EvalTaskCase
from drugclaw.eval_runner import (
    build_self_bench_task_cases,
    run_eval_cases,
    run_self_bench,
)
from drugclaw.query_plan import normalize_task_type


def _make_evidence_item(
    *,
    source_skill: str,
    relationship: str,
    target_entity: str,
    target_type: str,
    evidence_kind: str = "database_record",
) -> dict:
    item = EvidenceItem(
        evidence_id=f"{source_skill}:{relationship}:{target_entity}",
        source_skill=source_skill,
        source_type="database",
        source_title=f"{source_skill} evidence",
        source_locator=source_skill,
        snippet=f"{source_skill} {relationship} {target_entity}",
        structured_payload={},
        claim=f"imatinib {relationship} {target_entity}",
        evidence_kind=evidence_kind,
        support_direction="supports",
        confidence=0.8,
        retrieval_score=0.9,
        timestamp="2026-04-04T00:00:00Z",
        metadata={
            "source_entity": "imatinib",
            "relationship": relationship,
            "target_entity": target_entity,
            "source_type": "drug",
            "target_type": target_type,
        },
    )
    return item.to_dict()


def test_build_self_bench_task_cases_reuses_query_plan_taxonomy() -> None:
    cases = build_self_bench_task_cases(["ddi_corpus", "drugprot", "ade_corpus"])
    by_dataset = {case.dataset_name: case for case in cases}

    assert by_dataset["ddi_corpus"].task_type == normalize_task_type("ddi")
    assert by_dataset["ddi_corpus"].legacy_question_type == "ddi"
    assert by_dataset["ddi_corpus"].expected_resources == ["DDI Corpus 2013"]

    assert by_dataset["drugprot"].task_type == normalize_task_type("target_lookup")
    assert by_dataset["drugprot"].legacy_question_type == "target_lookup"
    assert by_dataset["drugprot"].expected_resources == ["DrugProt"]

    assert by_dataset["ade_corpus"].task_type == normalize_task_type("adr")
    assert by_dataset["ade_corpus"].legacy_question_type == "adr"


def test_run_eval_cases_filters_by_task_type_and_preserves_dataset_results() -> None:
    def _fake_runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "accuracy": 0.8 if dataset == "ade_corpus" else 0.6,
            "total": 5,
            "correct": 4 if dataset == "ade_corpus" else 3,
        }

    cases = build_self_bench_task_cases(["ade_corpus", "ddi_corpus"])
    summary = run_eval_cases(
        cases,
        dataset_runner=_fake_runner,
        task_types=["major_adrs"],
    )

    assert summary.total_cases == 1
    assert [score.dataset_name for score in summary.scores] == ["ade_corpus"]
    assert summary.dataset_results["ade_corpus"]["accuracy"] == 0.8
    assert summary.task_success_rate == 1.0


def test_run_eval_cases_marks_runner_errors_as_failed_scores() -> None:
    case = EvalTaskCase(
        task_id="custom_major_adrs",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="major_adrs",
        supporting_task_types=[],
        legacy_question_type="adr",
        query="Classify whether this sentence describes an ADR.",
        expectation=EvalExpectation(
            scorer="classification_exact",
            expected_resources=["Custom ADR Dataset"],
        ),
    )

    def _failing_runner(dataset, key_file, max_samples, maskself, log_dir):
        return {"error": "dataset unavailable"}

    summary = run_eval_cases([case], dataset_runner=_failing_runner)

    assert summary.total_cases == 1
    assert summary.failed_cases == 1
    assert summary.task_success_rate == 0.0
    assert summary.scores[0].dataset_name == "custom_dataset"
    assert summary.scores[0].error == "dataset unavailable"
    assert summary.scores[0].score == 0.0


def test_run_eval_cases_rejects_unknown_scorer() -> None:
    case = EvalTaskCase(
        task_id="custom_unknown_scorer",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="major_adrs",
        supporting_task_types=[],
        legacy_question_type="adr",
        query="Classify whether this sentence describes an ADR.",
        expectation=EvalExpectation(
            scorer="unsupported_scorer",
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {"accuracy": 1.0, "total": 1, "correct": 1}

    with pytest.raises(ValueError, match="unsupported scorer"):
        run_eval_cases([case], dataset_runner=_runner)


def test_run_self_bench_wraps_case_building_and_filtering() -> None:
    observed = []

    def _fake_runner(dataset, key_file, max_samples, maskself, log_dir):
        observed.append((dataset, max_samples, maskself))
        return {"accuracy": 0.5, "total": 2, "correct": 1}

    summary = run_self_bench(
        datasets=["ade_corpus", "ddi_corpus"],
        task_types=["major_adrs"],
        max_samples=3,
        maskself=True,
        dataset_runner=_fake_runner,
    )

    assert observed == [("ade_corpus", 3, True)]
    assert summary.total_cases == 1
    assert list(summary.dataset_results) == ["ade_corpus"]


def test_run_eval_cases_computes_scorecard_summary_fields() -> None:
    case = EvalTaskCase(
        task_id="custom_major_adrs",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="major_adrs",
        supporting_task_types=[],
        legacy_question_type="adr",
        query="Classify whether this sentence describes an ADR.",
        expectation=EvalExpectation(
            scorer="classification_exact",
            expected_resources=["ADReCS", "FAERS"],
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "accuracy": 0.75,
            "total": 4,
            "correct": 3,
            "authority_source_rate": 0.5,
            "knowhow_hit_rate": 1.0,
            "package_ready_rate": 0.25,
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.task_success_rate == 1.0
    assert summary.evidence_quality_score == 0.75
    assert summary.authority_source_rate == 0.5
    assert summary.knowhow_hit_rate == 1.0
    assert summary.package_ready_rate == 0.25


def test_run_eval_cases_builds_dataset_and_task_type_breakdowns() -> None:
    cases = [
        EvalTaskCase(
            task_id="case_adr",
            dataset_name="ade_corpus",
            plan_type="single_task",
            task_type="major_adrs",
            supporting_task_types=[],
            legacy_question_type="adr",
            query="Classify whether this sentence describes an ADR.",
            expectation=EvalExpectation(scorer="classification_exact"),
        ),
        EvalTaskCase(
            task_id="case_ddi",
            dataset_name="ddi_corpus",
            plan_type="single_task",
            task_type="clinically_relevant_ddi",
            supporting_task_types=[],
            legacy_question_type="ddi",
            query="Classify whether this sentence describes a DDI.",
            expectation=EvalExpectation(scorer="classification_exact"),
        ),
    ]

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        if dataset == "ade_corpus":
            return {"accuracy": 0.8, "authority_source_rate": 0.5}
        return {"accuracy": 0.6, "authority_source_rate": 1.0}

    summary = run_eval_cases(cases, dataset_runner=_runner)

    assert summary.dataset_breakdown["ade_corpus"]["total_cases"] == 1
    assert summary.dataset_breakdown["ade_corpus"]["evidence_quality_score"] == 0.8
    assert summary.dataset_breakdown["ddi_corpus"]["authority_source_rate"] == 1.0
    assert summary.task_type_breakdown["major_adrs"]["task_success_rate"] == 1.0
    assert summary.task_type_breakdown["clinically_relevant_ddi"]["evidence_quality_score"] == 0.6


def test_run_eval_cases_scores_evidence_coverage_from_expected_contracts() -> None:
    case = EvalTaskCase(
        task_id="coverage_case",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="direct_targets",
        supporting_task_types=[],
        legacy_question_type="target_lookup",
        query="What are the direct targets of imatinib?",
        expectation=EvalExpectation(
            scorer="evidence_coverage",
            expected_evidence_types=["binding", "clinical"],
            expected_resources=["BindingDB", "DrugBank"],
            expected_answer_sections=["summary", "limitations"],
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "evidence_types": ["binding"],
            "used_resources": ["BindingDB", "ChEMBL"],
            "answer_sections": ["summary", "limitations", "citations"],
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.scores[0].score == pytest.approx((0.5 + 0.5 + 1.0) / 3, abs=1e-4)
    assert summary.scores[0].metrics["evidence_type_coverage"] == pytest.approx(0.5)
    assert summary.scores[0].metrics["resource_coverage"] == pytest.approx(0.5)
    assert summary.scores[0].metrics["answer_section_coverage"] == pytest.approx(1.0)


def test_run_eval_cases_scores_source_quality_from_blended_signals() -> None:
    case = EvalTaskCase(
        task_id="source_quality_case",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="direct_targets",
        supporting_task_types=[],
        legacy_question_type="target_lookup",
        query="What are the direct targets of imatinib?",
        expectation=EvalExpectation(
            scorer="source_quality",
            expected_resources=["BindingDB", "DrugBank"],
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "authority_source_rate": 0.8,
            "package_ready_rate": 0.4,
            "used_resources": ["BindingDB"],
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.scores[0].score == pytest.approx((0.8 + 0.4 + 0.5) / 3, abs=1e-4)
    assert summary.scores[0].metrics["authority_source_rate"] == pytest.approx(0.8)
    assert summary.scores[0].metrics["package_ready_rate"] == pytest.approx(0.4)
    assert summary.scores[0].metrics["resource_coverage"] == pytest.approx(0.5)


def test_run_eval_cases_scores_source_quality_from_evidence_tiers_when_summary_metrics_missing() -> None:
    case = EvalTaskCase(
        task_id="source_quality_fallback_case",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="direct_targets",
        supporting_task_types=[],
        legacy_question_type="target_lookup",
        query="What are the direct targets of imatinib?",
        expectation=EvalExpectation(
            scorer="source_quality",
            expected_resources=["BindingDB", "Open Targets Platform"],
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "evidence_items": [
                _make_evidence_item(
                    source_skill="BindingDB",
                    relationship="targets",
                    target_entity="ABL1",
                    target_type="protein",
                ),
                _make_evidence_item(
                    source_skill="Open Targets Platform",
                    relationship="linked_target",
                    target_entity="KIT",
                    target_type="protein",
                ),
            ],
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.scores[0].score == pytest.approx((0.725 + 1.0) / 2, abs=1e-4)
    assert summary.scores[0].metrics["evidence_tier_quality"] == pytest.approx(0.725)
    assert summary.scores[0].metrics["resource_coverage"] == pytest.approx(1.0)


def test_run_eval_cases_scores_conflict_handling_from_structured_answer_signals() -> None:
    case = EvalTaskCase(
        task_id="conflict_case",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="clinically_relevant_ddi",
        supporting_task_types=[],
        legacy_question_type="ddi",
        query="Does warfarin interact with amiodarone?",
        expectation=EvalExpectation(
            scorer="conflict_handling",
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "final_answer_structured": {
                "warnings": ["Evidence conflict between observational and label sources."],
                "limitations": ["Conflicting reports remain unresolved."],
                "final_outcome": "partial_with_weak_support",
            }
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.scores[0].score == pytest.approx(1.0)
    assert summary.scores[0].metrics["conflict_warning_present"] == 1.0
    assert summary.scores[0].metrics["conflict_limitation_present"] == 1.0
    assert summary.scores[0].metrics["conflict_safe_outcome"] == 1.0


def test_run_eval_cases_scores_conflict_handling_from_claim_assessments_when_structured_answer_is_sparse() -> None:
    case = EvalTaskCase(
        task_id="conflict_assessment_case",
        dataset_name="custom_dataset",
        plan_type="single_task",
        task_type="clinically_relevant_ddi",
        supporting_task_types=[],
        legacy_question_type="ddi",
        query="Does warfarin interact with amiodarone?",
        expectation=EvalExpectation(
            scorer="conflict_handling",
        ),
    )

    def _runner(dataset, key_file, max_samples, maskself, log_dir):
        return {
            "claim_assessments": [
                {
                    "claim": "warfarin interacts with amiodarone",
                    "verdict": "uncertain",
                    "rationale": "Conflicting evidence detected: 1 supporting and 1 contradicting item.",
                    "limitations": ["Conflicting evidence is present."],
                }
            ],
            "final_answer_structured": {
                "warnings": [],
                "limitations": [],
                "final_outcome": "honest_gap",
            },
        }

    summary = run_eval_cases([case], dataset_runner=_runner)

    assert summary.scores[0].score == pytest.approx(1.0)
    assert summary.scores[0].metrics["conflict_warning_present"] == 1.0
    assert summary.scores[0].metrics["conflict_limitation_present"] == 1.0
    assert summary.scores[0].metrics["conflict_safe_outcome"] == 1.0
