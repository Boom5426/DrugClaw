from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .evidence import EvidenceItem
from .eval_models import EvalExpectation, EvalRunSummary, EvalScore, EvalTaskCase
from .query_plan import normalize_task_type
from .task_evidence_policy import classify_evidence_item
from self_bench.bench_utils import DATASET_SKILL_MAP


SELF_BENCH_TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ade_corpus": {
        "task_type": "major_adrs",
        "legacy_question_type": "adr",
        "query": "Classify whether the sentence describes an adverse drug event.",
    },
    "ddi_corpus": {
        "task_type": "clinically_relevant_ddi",
        "legacy_question_type": "ddi",
        "query": "Classify the drug-drug interaction type described in the sentence.",
    },
    "drugprot": {
        "task_type": "direct_targets",
        "legacy_question_type": "target_lookup",
        "query": "Classify the drug-protein relation described in the sentence.",
    },
    "phee": {
        "task_type": "major_adrs",
        "legacy_question_type": "adr",
        "query": "Classify the pharmacovigilance event type described in the sentence.",
    },
    "dilirank": {
        "task_type": "major_adrs",
        "legacy_question_type": "adr",
        "query": "Classify the liver injury concern level for the drug.",
    },
    "n2c2_2018": {
        "task_type": "major_adrs",
        "legacy_question_type": "adr",
        "query": "Classify whether the clinical note describes an adverse drug event.",
    },
    "psytar": {
        "task_type": "major_adrs",
        "legacy_question_type": "adr",
        "query": "Classify whether the patient review describes an adverse drug reaction.",
    },
}


DatasetRunner = Callable[[str, str | None, int, bool | None, str | None], Dict[str, Any]]


def default_dataset_runner(
    dataset: str,
    key_file: str | None,
    max_samples: int,
    maskself: bool | None,
    log_dir: str | None,
) -> Dict[str, Any]:
    module = importlib.import_module(f"self_bench.{dataset}.bench")
    return module.run(
        key_file=key_file,
        max_samples=max_samples,
        maskself=maskself,
        log_dir=log_dir,
    )


def build_self_bench_task_cases(datasets: Sequence[str] | None = None) -> List[EvalTaskCase]:
    selected = list(datasets or SELF_BENCH_TASK_CONFIGS.keys())
    cases: List[EvalTaskCase] = []
    for dataset in selected:
        config = SELF_BENCH_TASK_CONFIGS.get(dataset)
        if config is None:
            continue
        cases.append(
            EvalTaskCase(
                task_id=f"self_bench::{dataset}",
                dataset_name=dataset,
                plan_type="single_task",
                task_type=config["task_type"],
                supporting_task_types=[],
                legacy_question_type=config["legacy_question_type"],
                query=config["query"],
                expectation=EvalExpectation(
                    scorer="classification_exact",
                    expected_resources=[DATASET_SKILL_MAP[dataset]]
                    if dataset in DATASET_SKILL_MAP
                    else [],
                ),
            )
        )
    return cases


def run_eval_cases(
    task_cases: Iterable[EvalTaskCase],
    *,
    dataset_runner: DatasetRunner = default_dataset_runner,
    datasets: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    plan_types: Sequence[str] | None = None,
    key_file: str | None = None,
    max_samples: int = 0,
    maskself: bool | None = None,
    log_dir: str | None = None,
) -> EvalRunSummary:
    selected_cases = _filter_cases(
        list(task_cases),
        datasets=datasets,
        task_types=task_types,
        plan_types=plan_types,
    )
    scores: List[EvalScore] = []
    dataset_results: Dict[str, Dict[str, Any]] = {}
    completed_cases = 0
    failed_cases = 0
    evidence_quality_values: List[float] = []
    authority_source_values: List[float] = []
    knowhow_hit_values: List[float] = []
    package_ready_values: List[float] = []

    for case in selected_cases:
        started_at = time.time()
        raw_result = dataset_runner(
            case.dataset_name,
            key_file,
            max_samples,
            maskself,
            log_dir,
        )
        if "elapsed_sec" not in raw_result:
            raw_result = dict(raw_result)
            raw_result["elapsed_sec"] = round(time.time() - started_at, 1)
        dataset_results[case.dataset_name] = dict(raw_result)
        score = _score_case(case, raw_result)
        scores.append(score)
        evidence_quality_values.append(score.score)
        authority_source_values.append(_coerce_metric(raw_result, "authority_source_rate"))
        knowhow_hit_values.append(_coerce_metric(raw_result, "knowhow_hit_rate"))
        package_ready_values.append(_coerce_metric(raw_result, "package_ready_rate"))
        if score.error:
            failed_cases += 1
        else:
            completed_cases += 1

    total_cases = len(selected_cases)
    task_success_rate = (
        round(completed_cases / total_cases, 4)
        if total_cases
        else 0.0
    )
    return EvalRunSummary(
        total_cases=total_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        task_success_rate=task_success_rate,
        evidence_quality_score=_average(evidence_quality_values),
        authority_source_rate=_average(authority_source_values),
        knowhow_hit_rate=_average(knowhow_hit_values),
        package_ready_rate=_average(package_ready_values),
        scores=scores,
        dataset_results=dataset_results,
        dataset_breakdown=_build_breakdown(scores, group_key="dataset_name"),
        task_type_breakdown=_build_breakdown(scores, group_key="task_type"),
    )


def run_self_bench(
    *,
    datasets: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    plan_types: Sequence[str] | None = None,
    key_file: str | None = None,
    max_samples: int = 0,
    maskself: bool | None = None,
    log_dir: str | None = None,
    dataset_runner: DatasetRunner = default_dataset_runner,
) -> EvalRunSummary:
    cases = build_self_bench_task_cases(datasets)
    return run_eval_cases(
        cases,
        dataset_runner=dataset_runner,
        datasets=datasets,
        task_types=task_types,
        plan_types=plan_types,
        key_file=key_file,
        max_samples=max_samples,
        maskself=maskself,
        log_dir=log_dir,
    )


def _filter_cases(
    task_cases: List[EvalTaskCase],
    *,
    datasets: Sequence[str] | None,
    task_types: Sequence[str] | None,
    plan_types: Sequence[str] | None,
) -> List[EvalTaskCase]:
    allowed_datasets = {str(value).strip() for value in (datasets or []) if str(value).strip()}
    allowed_task_types = {
        normalize_task_type(value)
        for value in (task_types or [])
        if str(value).strip()
    }
    allowed_plan_types = {str(value).strip() for value in (plan_types or []) if str(value).strip()}

    filtered: List[EvalTaskCase] = []
    for case in task_cases:
        if allowed_datasets and case.dataset_name not in allowed_datasets:
            continue
        if allowed_task_types and case.task_type not in allowed_task_types:
            continue
        if allowed_plan_types and case.plan_type not in allowed_plan_types:
            continue
        filtered.append(case)
    return filtered


def _score_case(case: EvalTaskCase, raw_result: Dict[str, Any]) -> EvalScore:
    scorer = str(case.scorer or "classification_exact").strip()
    if scorer not in {
        "classification_exact",
        "evidence_coverage",
        "source_quality",
        "conflict_handling",
    }:
        raise ValueError(f"unsupported scorer: {scorer}")

    error = str(raw_result.get("error", "") or "").strip()
    if error:
        return EvalScore(
            task_id=case.task_id,
            dataset_name=case.dataset_name,
            plan_type=case.plan_type,
            task_type=case.task_type,
            legacy_question_type=case.legacy_question_type,
            scorer=scorer,
            score=0.0,
            passed=False,
            metrics=dict(raw_result),
            error=error,
        )

    metrics = dict(raw_result)
    score = float(raw_result.get("accuracy", 0.0) or 0.0)
    if scorer == "evidence_coverage":
        score, derived_metrics = _score_evidence_coverage(case, raw_result)
        metrics.update(derived_metrics)
    elif scorer == "source_quality":
        score, derived_metrics = _score_source_quality(case, raw_result)
        metrics.update(derived_metrics)
    elif scorer == "conflict_handling":
        score, derived_metrics = _score_conflict_handling(raw_result)
        metrics.update(derived_metrics)
    elif scorer != "classification_exact":
        score = float(raw_result.get(scorer, score) or 0.0)

    return EvalScore(
        task_id=case.task_id,
        dataset_name=case.dataset_name,
        plan_type=case.plan_type,
        task_type=case.task_type,
        legacy_question_type=case.legacy_question_type,
        scorer=scorer,
        score=score,
        passed=True,
        metrics=metrics,
        error="",
    )


def _score_evidence_coverage(
    case: EvalTaskCase,
    raw_result: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    derived_metrics: Dict[str, float] = {}
    components: List[float] = []

    if case.expected_evidence_types:
        coverage = _coverage_ratio(case.expected_evidence_types, _extract_evidence_types(raw_result))
        derived_metrics["evidence_type_coverage"] = coverage
        components.append(coverage)
    if case.expected_resources:
        coverage = _coverage_ratio(case.expected_resources, _extract_used_resources(raw_result))
        derived_metrics["resource_coverage"] = coverage
        components.append(coverage)
    if case.expected_answer_sections:
        coverage = _coverage_ratio(case.expected_answer_sections, _extract_answer_sections(raw_result))
        derived_metrics["answer_section_coverage"] = coverage
        components.append(coverage)

    if not components:
        return _coerce_metric(raw_result, "accuracy"), derived_metrics
    return _average(components), derived_metrics


def _score_source_quality(
    case: EvalTaskCase,
    raw_result: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    derived_metrics: Dict[str, float] = {}
    components: List[float] = []

    if "authority_source_rate" in raw_result:
        authority_source_rate = _coerce_metric(raw_result, "authority_source_rate")
        derived_metrics["authority_source_rate"] = authority_source_rate
        components.append(authority_source_rate)
    if "package_ready_rate" in raw_result:
        package_ready_rate = _coerce_metric(raw_result, "package_ready_rate")
        derived_metrics["package_ready_rate"] = package_ready_rate
        components.append(package_ready_rate)
    evidence_tier_quality = _derive_evidence_tier_quality(case.task_type, raw_result)
    if evidence_tier_quality is not None:
        derived_metrics["evidence_tier_quality"] = evidence_tier_quality
        components.append(evidence_tier_quality)
    if case.expected_resources:
        resource_coverage = _coverage_ratio(case.expected_resources, _extract_used_resources(raw_result))
        derived_metrics["resource_coverage"] = resource_coverage
        components.append(resource_coverage)

    if not components:
        return _coerce_metric(raw_result, "source_quality"), derived_metrics
    return _average(components), derived_metrics


def _score_conflict_handling(raw_result: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    structured = _coerce_mapping(raw_result.get("final_answer_structured"))
    assessment_warnings, assessment_limitations = _extract_claim_assessment_conflict_text(raw_result)
    warnings = _string_list(
        raw_result.get("warnings"),
        structured.get("warnings"),
        assessment_warnings,
    )
    limitations = _string_list(
        raw_result.get("limitations"),
        structured.get("limitations"),
        assessment_limitations,
    )
    final_outcome = str(
        raw_result.get("final_outcome")
        or structured.get("final_outcome")
        or ""
    ).strip()

    derived_metrics = {
        "conflict_warning_present": 1.0 if _contains_conflict_signal(warnings) else 0.0,
        "conflict_limitation_present": 1.0 if _contains_conflict_signal(limitations) else 0.0,
        "conflict_safe_outcome": 1.0 if final_outcome in {
            "partial_with_weak_support",
            "honest_gap",
            "conflicting_evidence",
        } else 0.0,
    }
    components = list(derived_metrics.values())
    if "conflict_resolution_quality" in raw_result:
        quality = _coerce_metric(raw_result, "conflict_resolution_quality")
        derived_metrics["conflict_resolution_quality"] = quality
        components.append(quality)
    return _average(components), derived_metrics


def _derive_evidence_tier_quality(task_type: str, raw_result: Dict[str, Any]) -> float | None:
    weights = {
        "strong_direct": 1.0,
        "strong_structured": 1.0,
        "secondary_official_support": 0.85,
        "association_only": 0.45,
        "generic_weak_support": 0.25,
    }
    scored_items: List[float] = []
    for item in _materialize_evidence_items(raw_result):
        classification = classify_evidence_item(task_type, item)
        scored_items.append(weights.get(classification.tier, 0.3))
    if not scored_items:
        return None
    return _average(scored_items)


def _coverage_ratio(expected: Sequence[str], observed: Sequence[str]) -> float:
    expected_values = _normalized_values(expected)
    if not expected_values:
        return 0.0
    observed_values = set(_normalized_values(observed))
    matched = sum(1 for value in expected_values if value in observed_values)
    return round(matched / len(expected_values), 4)


def _extract_evidence_types(raw_result: Dict[str, Any]) -> List[str]:
    explicit_values = _string_list(
        raw_result.get("evidence_types"),
        raw_result.get("observed_evidence_types"),
    )
    if explicit_values:
        return explicit_values

    observed: List[str] = []
    for item in _extract_evidence_items(raw_result):
        metadata = _coerce_mapping(item.get("metadata"))
        observed.extend(
            _string_list(
                item.get("evidence_type"),
                metadata.get("evidence_type"),
                metadata.get("relationship"),
            )
        )
    return observed


def _extract_used_resources(raw_result: Dict[str, Any]) -> List[str]:
    explicit_values = _string_list(
        raw_result.get("used_resources"),
        raw_result.get("selected_resources"),
        raw_result.get("retrieved_resources"),
        raw_result.get("resource_filter"),
    )
    if explicit_values:
        return explicit_values

    observed: List[str] = []
    for item in _extract_evidence_items(raw_result):
        observed.extend(_string_list(item.get("source_skill"), item.get("source_locator")))
    return observed


def _extract_answer_sections(raw_result: Dict[str, Any]) -> List[str]:
    explicit_values = _string_list(raw_result.get("answer_sections"))
    if explicit_values:
        return explicit_values

    structured = _coerce_mapping(raw_result.get("final_answer_structured"))
    observed: List[str] = []
    if raw_result.get("answer") or structured:
        observed.append("summary")
    if _string_list(structured.get("warnings")):
        observed.append("warnings")
    if _string_list(structured.get("limitations")):
        observed.append("limitations")
    if _string_list(structured.get("citations")):
        observed.append("citations")
    if _coerce_sequence(structured.get("key_claims")):
        observed.append("key_claims")
    return observed


def _extract_evidence_items(raw_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured = _coerce_mapping(raw_result.get("final_answer_structured"))
    values = raw_result.get("evidence_items")
    if not values:
        values = structured.get("evidence_items")
    items: List[Dict[str, Any]] = []
    for value in _coerce_sequence(values):
        items.append(_coerce_mapping(value))
    return items


def _materialize_evidence_items(raw_result: Dict[str, Any]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    for payload in _extract_evidence_items(raw_result):
        evidence_item = _coerce_evidence_item(payload)
        if evidence_item is not None:
            items.append(evidence_item)
    return items


def _extract_claim_assessment_conflict_text(raw_result: Dict[str, Any]) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    limitations: List[str] = []
    for payload in _coerce_sequence(raw_result.get("claim_assessments")):
        assessment = _coerce_mapping(payload)
        verdict = str(assessment.get("verdict", "") or "").strip().lower()
        rationale = str(assessment.get("rationale", "") or "").strip()
        if verdict in {"uncertain", "contradicted"}:
            warnings.append(rationale or "Conflicting evidence is present.")
        limitations.extend(_string_list(assessment.get("limitations")))
    return _string_list(warnings), _string_list(limitations)


def _contains_conflict_signal(values: Sequence[str]) -> bool:
    conflict_terms = (
        "conflict",
        "conflicting",
        "discord",
        "disagree",
        "inconsistent",
    )
    return any(any(term in value for term in conflict_terms) for value in _normalized_values(values))


def _coerce_evidence_item(value: Dict[str, Any]) -> EvidenceItem | None:
    if not value:
        return None
    try:
        return EvidenceItem(
            evidence_id=str(value.get("evidence_id", "") or "").strip(),
            source_skill=str(value.get("source_skill", "") or "").strip(),
            source_type=str(value.get("source_type", "") or "").strip(),
            source_title=str(value.get("source_title", "") or "").strip(),
            source_locator=str(value.get("source_locator", "") or "").strip(),
            snippet=str(value.get("snippet", "") or "").strip(),
            structured_payload=_coerce_mapping(value.get("structured_payload")),
            claim=str(value.get("claim", "") or "").strip(),
            evidence_kind=str(value.get("evidence_kind", "") or "").strip(),
            support_direction=str(value.get("support_direction", "") or "").strip(),
            confidence=_coerce_metric(value, "confidence"),
            retrieval_score=_coerce_optional_float(value.get("retrieval_score")),
            timestamp=str(value.get("timestamp", "") or "").strip(),
            metadata=_coerce_mapping(value.get("metadata")),
        )
    except Exception:
        return None


def _normalized_values(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _string_list(*values: Any) -> List[str]:
    observed: List[str] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in observed:
                observed.append(text)
            continue
        for item in _coerce_sequence(value):
            text = str(item or "").strip()
            if text and text not in observed:
                observed.append(text)
    return observed


def _coerce_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return dict(converted)
    return {}


def _coerce_metric(raw_result: Dict[str, Any], key: str) -> float:
    try:
        return float(raw_result.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _build_breakdown(scores: Sequence[EvalScore], *, group_key: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[EvalScore]] = {}
    for score in scores:
        value = str(getattr(score, group_key, "") or "").strip()
        if not value:
            continue
        buckets.setdefault(value, []).append(score)

    breakdown: Dict[str, Dict[str, Any]] = {}
    for value, grouped_scores in buckets.items():
        total_cases = len(grouped_scores)
        completed_cases = sum(1 for score in grouped_scores if not score.error)
        failed_cases = total_cases - completed_cases
        breakdown[value] = {
            "total_cases": total_cases,
            "completed_cases": completed_cases,
            "failed_cases": failed_cases,
            "task_success_rate": round(completed_cases / total_cases, 4) if total_cases else 0.0,
            "evidence_quality_score": _average([score.score for score in grouped_scores]),
            "authority_source_rate": _average(
                [_coerce_metric(score.metrics, "authority_source_rate") for score in grouped_scores]
            ),
            "knowhow_hit_rate": _average(
                [_coerce_metric(score.metrics, "knowhow_hit_rate") for score in grouped_scores]
            ),
            "package_ready_rate": _average(
                [_coerce_metric(score.metrics, "package_ready_rate") for score in grouped_scores]
            ),
        }
    return breakdown
