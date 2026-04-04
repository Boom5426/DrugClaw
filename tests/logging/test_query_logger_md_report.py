from __future__ import annotations

import json
from pathlib import Path

from drugclaw.query_logger import QueryLogger


def _sample_result() -> dict:
    return {
        "query": "What does imatinib target?",
        "normalized_query": "What does imatinib target?",
        "resolved_entities": {"drug": ["imatinib"]},
        "input_resolution": {
            "status": "resolved",
            "canonical_drug_names": ["imatinib"],
            "alias_candidates": ["gleevec"],
            "identifier_resolution": {
                "status": "resolved",
                "detected_identifiers": [
                    {
                        "type": "chembl_id",
                        "raw_text": "CHEMBL941",
                        "normalized_value": "CHEMBL941",
                    }
                ],
                "resolved_records": [
                    {
                        "identifier_type": "chembl_id",
                        "identifier_value": "CHEMBL941",
                        "canonical_drug_name": "imatinib",
                    }
                ],
                "errors": [],
            },
        },
        "answer": "Known Targets:\n- imatinib -> ABL1",
        "mode": "simple",
        "resource_filter": ["BindingDB"],
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "reasoning_history": [],
        "retrieved_content": [],
        "retrieved_text": "",
        "retrieval_diagnostics": [
            {
                "kind": "knowhow",
                "task_id": "primary",
                "task_type": "direct_targets",
                "doc_id": "direct_targets_grounding",
                "title": "Direct target grounding",
                "snippet": "Prioritize established direct binding evidence before association-only target claims.",
                "declared_by_skills": [
                    "BindingDB",
                    "ChEMBL",
                    "DGIdb",
                    "DrugBank",
                    "Open Targets Platform",
                ],
            }
        ],
        "web_search_results": [],
        "query_plan": {
            "plan_type": "composite_query",
            "question_type": "mechanism",
            "knowhow_doc_ids": [
                "direct_targets_grounding",
                "mechanism_explanation",
            ],
            "primary_task": {
                "task_type": "direct_targets",
                "task_id": "primary",
            },
            "supporting_tasks": [
                {
                    "task_type": "mechanism_of_action",
                    "task_id": "support_1",
                }
            ],
            "answer_contract": {
                "section_order": [
                    "summary",
                    "direct_targets",
                    "mechanism_of_action",
                    "limitations",
                ]
            },
            "knowhow_hints": [
                {
                    "doc_id": "direct_targets_grounding",
                    "task_id": "primary",
                    "task_type": "direct_targets",
                    "snippet": "Prioritize established direct binding evidence before association-only target claims.",
                    "declared_by_skills": [
                        "BindingDB",
                        "ChEMBL",
                        "DGIdb",
                        "DrugBank",
                        "Open Targets Platform",
                    ],
                }
            ],
        },
        "success": True,
        "final_answer_structured": {
            "summary_confidence": 0.83,
            "task_type": "mechanism",
            "final_outcome": "partial_with_weak_support",
            "diagnostics": {
                "strong_record_count": 1,
                "weak_support_count": 0,
            },
            "key_claims": [
                {
                    "claim": "Imatinib targets ABL1.",
                    "confidence": 0.83,
                    "evidence_ids": ["bindingdb:1"],
                }
            ],
            "evidence_items": [
                {
                    "evidence_id": "bindingdb:1",
                    "source_skill": "BindingDB",
                    "source_locator": "BindingDB",
                    "metadata": {
                        "source_entity": "imatinib",
                        "relationship": "targets",
                        "target_entity": "ABL1",
                    },
                    "confidence": 0.83,
                }
            ],
            "citations": ["[bindingdb:1] BindingDB — BindingDB"],
            "warnings": [],
            "limitations": [],
        },
    }


def test_query_logger_writes_md_report_when_requested(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What does imatinib target?",
        _sample_result(),
        save_md_report=True,
    )

    report_path = tmp_path / "query_logs" / query_id / "report.md"

    assert report_path.exists()
    markdown = report_path.read_text(encoding="utf-8")
    assert "# DrugClaw Query Report" in markdown
    assert "What does imatinib target?" in markdown
    assert "imatinib -> ABL1" in markdown
    assert "BindingDB" in markdown
    assert "partial_with_weak_support" in markdown


def test_query_logger_skips_md_report_when_not_requested(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What does imatinib target?",
        _sample_result(),
        save_md_report=False,
    )

    report_path = tmp_path / "query_logs" / query_id / "report.md"

    assert not report_path.exists()


def test_query_logger_persists_input_resolution_metadata(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What does Gleevec target?",
        _sample_result(),
        save_md_report=False,
    )

    metadata_path = tmp_path / "query_logs" / query_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["query"] == "What does Gleevec target?"
    assert metadata["normalized_query"] == "What does imatinib target?"
    assert metadata["resolved_entities"] == {"drug": ["imatinib"]}
    assert metadata["input_resolution"]["canonical_drug_names"] == ["imatinib"]
    assert metadata["input_resolution"]["identifier_resolution"]["resolved_records"][0]["identifier_value"] == "CHEMBL941"


def test_query_logger_persists_task_aware_query_plan_and_knowhow_summary(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What are the known drug targets and mechanism of action of imatinib?",
        _sample_result(),
        save_md_report=False,
    )

    metadata_path = tmp_path / "query_logs" / query_id / "metadata.json"
    evidence_path = tmp_path / "query_logs" / query_id / "evidence.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert metadata["query_plan_summary"]["plan_type"] == "composite_query"
    assert metadata["query_plan_summary"]["primary_task_type"] == "direct_targets"
    assert metadata["query_plan_summary"]["supporting_task_types"] == [
        "mechanism_of_action"
    ]
    assert metadata["query_plan_summary"]["knowhow_doc_ids"] == [
        "direct_targets_grounding",
        "mechanism_explanation",
    ]
    assert metadata["query_plan_summary"]["answer_section_order"] == [
        "summary",
        "direct_targets",
        "mechanism_of_action",
        "limitations",
    ]
    assert metadata["query_plan_summary"]["knowhow_hints"] == [
        {
            "doc_id": "direct_targets_grounding",
            "task_id": "primary",
            "task_type": "direct_targets",
            "declared_by_skills": [
                "BindingDB",
                "ChEMBL",
                "DGIdb",
                "DrugBank",
                "Open Targets Platform",
            ],
        }
    ]
    assert evidence["retrieval_diagnostics"][0]["kind"] == "knowhow"
    assert evidence["retrieval_diagnostics"][0]["doc_id"] == "direct_targets_grounding"


def test_query_logger_saves_and_reuses_latest_scorecard_summary(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    scorecard_path = logger.save_scorecard_summary(
        {
            "task_success_rate": 0.8,
            "evidence_quality_score": 0.7,
            "authority_source_rate": 0.6,
            "knowhow_hit_rate": 1.0,
            "package_ready_rate": 0.5,
            "release_gate": {
                "passed": False,
                "failures": ["task_success_rate=0.80 < 0.90"],
                "thresholds": {"min_task_success_rate": 0.9},
            },
        },
        metadata={"datasets": ["ade_corpus", "ddi_corpus"]},
    )

    summary = logger.get_latest_scorecard_summary()

    assert Path(scorecard_path).exists()
    assert summary is not None
    assert summary["summary"]["task_success_rate"] == 0.8
    assert summary["summary"]["release_gate"]["passed"] is False
    assert summary["metadata"]["datasets"] == ["ade_corpus", "ddi_corpus"]

    query_id = logger.log_query(
        "What does imatinib target?",
        _sample_result(),
        save_md_report=False,
    )

    metadata_path = tmp_path / "query_logs" / query_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["recent_scorecard"]["summary"]["knowhow_hit_rate"] == 1.0
    assert metadata["recent_scorecard"]["summary"]["release_gate"]["failures"] == [
        "task_success_rate=0.80 < 0.90"
    ]
    assert metadata["recent_scorecard"]["metadata"]["datasets"] == [
        "ade_corpus",
        "ddi_corpus",
    ]
