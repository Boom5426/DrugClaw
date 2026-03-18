from __future__ import annotations

from drugclaw.response_formatter import wrap_answer_card


def test_wrap_answer_card_prefers_structured_evidence_items() -> None:
    result = {
        "query": "What are the known drug targets of imatinib?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": [],
        "retrieved_content": [
            {
                "source": "Molecular Targets",
                "source_entity": "imatinib",
                "relationship": "search_hit",
                "target_entity": "leukemia",
                "sources": [],
            }
        ],
        "final_answer_structured": {
            "evidence_items": [
                {
                    "source_skill": "ChEMBL",
                    "source_locator": "CHEMBL941",
                    "metadata": {
                        "source_entity": "IMATINIB",
                        "relationship": "has_ic50_activity",
                        "target_entity": "ABL1",
                    },
                    "confidence": 0.78,
                }
            ],
            "citations": ["[chembl:1] ChEMBL — CHEMBL941"],
        },
    }

    formatted = wrap_answer_card("Known Targets:\n- IMATINIB -> ABL1", result)

    assert "ABL1" in formatted
    assert "search_hit" not in formatted
    assert "leukemia" not in formatted
