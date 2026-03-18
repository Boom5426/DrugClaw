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


def test_wrap_answer_card_groups_target_lookup_evidence_by_claim() -> None:
    result = {
        "query": "What are the known drug targets of imatinib?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": [],
        "retrieved_content": [],
        "final_answer_structured": {
            "key_claims": [
                {
                    "claim": "imatinib targets KIT.",
                    "confidence": 0.84,
                    "evidence_ids": ["bindingdb:2", "chembl:6"],
                    "citations": [],
                },
                {
                    "claim": "imatinib targets ABL1.",
                    "confidence": 0.83,
                    "evidence_ids": ["bindingdb:1", "chembl:1"],
                    "citations": [],
                },
            ],
            "evidence_items": [
                {
                    "evidence_id": "bindingdb:2",
                    "source_skill": "BindingDB",
                    "source_locator": "BindingDB",
                    "metadata": {
                        "source_entity": "imatinib",
                        "relationship": "targets",
                        "target_entity": "KIT",
                    },
                    "confidence": 0.84,
                },
                {
                    "evidence_id": "chembl:6",
                    "source_skill": "ChEMBL",
                    "source_locator": "CHEMBL941",
                    "metadata": {
                        "source_entity": "IMATINIB",
                        "relationship": "has_ic50_activity",
                        "target_entity": "Mast/stem cell growth factor receptor Kit",
                    },
                    "confidence": 0.86,
                },
                {
                    "evidence_id": "bindingdb:1",
                    "source_skill": "BindingDB",
                    "source_locator": "BindingDB",
                    "metadata": {
                        "source_entity": "imatinib",
                        "relationship": "targets",
                        "target_entity": "ABL1",
                    },
                    "confidence": 0.84,
                },
            ],
            "citations": ["[bindingdb:2] BindingDB — BindingDB"],
        },
    }

    formatted = wrap_answer_card(
        "Known Targets:\n- imatinib -> KIT\n- imatinib -> ABL1",
        result,
    )

    assert "| 1 | BindingDB, ChEMBL | imatinib | targets | KIT | 0.84 | bindingdb:2, chembl:6 |" in formatted
    assert "| 2 | BindingDB | imatinib | targets | ABL1 | 0.83 | bindingdb:1, chembl:1 |" in formatted
    assert "Mast/stem cell growth factor receptor Kit" not in formatted
