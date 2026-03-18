from __future__ import annotations

from pathlib import Path

from drugclaw.query_logger import QueryLogger


def _sample_result() -> dict:
    return {
        "query": "What does imatinib target?",
        "answer": "Known Targets:\n- imatinib -> ABL1",
        "mode": "simple",
        "resource_filter": ["BindingDB"],
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "reasoning_history": [],
        "retrieved_content": [],
        "retrieved_text": "",
        "web_search_results": [],
        "success": True,
        "final_answer_structured": {
            "summary_confidence": 0.83,
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


def test_query_logger_writes_html_report_when_requested(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What does imatinib target?",
        _sample_result(),
        save_html_report=True,
    )

    report_path = tmp_path / "query_logs" / query_id / "report.html"

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "What does imatinib target?" in html
    assert "Imatinib targets ABL1." in html
    assert "BindingDB" in html


def test_query_logger_skips_html_report_when_not_requested(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=str(tmp_path / "query_logs"))

    query_id = logger.log_query(
        "What does imatinib target?",
        _sample_result(),
        save_html_report=False,
    )

    report_path = tmp_path / "query_logs" / query_id / "report.html"

    assert not report_path.exists()
