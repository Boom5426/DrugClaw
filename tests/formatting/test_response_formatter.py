from __future__ import annotations

from drugclaw.response_formatter import wrap_answer_card


def test_wrap_answer_card_prefers_structured_summary_confidence() -> None:
    result = {
        "query": "What prescribing and safety information is available for metformin?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": ["DailyMed"],
        "retrieved_content": [],
        "final_answer_structured": {
            "summary_confidence": 0.83,
            "key_claims": [],
            "evidence_items": [],
            "citations": [],
            "limitations": [],
            "warnings": [],
        },
    }

    formatted = wrap_answer_card("Metformin labeling answer", result)

    assert "**Confidence**" in formatted
    assert "(0.83)" in formatted
    assert "HIGH" in formatted
    assert "N/A" not in formatted


def test_wrap_answer_card_surfaces_phase_2a_task_outcome_when_present() -> None:
    result = {
        "query": "What are the approved indications and repurposing evidence of metformin?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": ["RepoDB", "DrugCentral", "DrugBank"],
        "retrieved_content": [],
        "final_answer_structured": {
            "summary_confidence": 0.41,
            "task_type": "drug_repurposing",
            "final_outcome": "partial_with_weak_support",
            "key_claims": [],
            "evidence_items": [],
            "citations": [],
            "limitations": [],
            "warnings": [],
        },
    }

    formatted = wrap_answer_card("Metformin answer", result)

    assert "drug_repurposing" in formatted
    assert "partial_with_weak_support" in formatted


def test_wrap_answer_card_truncates_long_source_list() -> None:
    result = {
        "query": "What prescribing and safety information is available for metformin?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": ["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
        "retrieved_content": [],
        "final_answer_structured": {
            "summary_confidence": 0.72,
            "key_claims": [],
            "evidence_items": [],
            "citations": [
                "[src:1] DailyMed — https://example.test/1",
                "[src:2] DailyMed — https://example.test/2",
                "[src:3] DailyMed — https://example.test/3",
                "[src:4] DailyMed — https://example.test/4",
                "[src:5] DailyMed — https://example.test/5",
                "[src:6] openFDA Human Drug — https://example.test/6",
                "[src:7] openFDA Human Drug — https://example.test/7",
                "[src:8] MedlinePlus Drug Info — https://example.test/8",
            ],
            "limitations": [],
            "warnings": [],
        },
    }

    formatted = wrap_answer_card("Metformin labeling answer", result)

    assert "## Sources" in formatted
    assert "[src:1] DailyMed" in formatted
    assert "[src:6] openFDA Human Drug" in formatted
    assert "[src:7] openFDA Human Drug" not in formatted
    assert "[src:8] MedlinePlus Drug Info" not in formatted
    assert "Showing 6 of 8 sources; 2 more omitted." in formatted


def test_wrap_answer_card_prefers_structured_web_citations_over_raw_web_results() -> None:
    result = {
        "query": "What are the approved indications and repurposing evidence of metformin?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": [],
        "retrieved_content": [],
        "web_search_results": [
            {
                "source": "PubMed",
                "title": "Therapeutic effects of metformin on cocaine conditioned place preference and locomotion",
                "url": "https://pubmed.ncbi.nlm.nih.gov/40000001/",
            },
            {
                "source": "PubMed",
                "title": "Metformin repurposing for ovarian cancer",
                "url": "https://pubmed.ncbi.nlm.nih.gov/40000002/",
            },
        ],
        "final_answer_structured": {
            "summary_confidence": 0.59,
            "task_type": "drug_repurposing",
            "final_outcome": "partial_with_weak_support",
            "key_claims": [],
            "evidence_items": [],
            "citations": ["[web:PubMed] https://pubmed.ncbi.nlm.nih.gov/40000002/"],
            "limitations": [],
            "warnings": [],
        },
    }

    formatted = wrap_answer_card("Metformin answer", result)

    assert "https://pubmed.ncbi.nlm.nih.gov/40000002/" in formatted
    assert "cocaine conditioned place preference" not in formatted.lower()


def test_wrap_answer_card_collapses_duplicate_evidence_rows() -> None:
    result = {
        "query": "What prescribing and safety information is available for metformin?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": ["openFDA Human Drug"],
        "retrieved_content": [],
        "final_answer_structured": {
            "summary_confidence": 0.72,
            "key_claims": [],
            "evidence_items": [
                {
                    "source_skill": "openFDA Human Drug",
                    "source_locator": "openFDA Human Drug",
                    "metadata": {
                        "source_entity": "Metformin Hydrochloride",
                        "relationship": "has_adverse_reaction",
                        "target_entity": "adverse reactions",
                    },
                    "confidence": 0.86,
                },
                {
                    "source_skill": "openFDA Human Drug",
                    "source_locator": "openFDA Human Drug",
                    "metadata": {
                        "source_entity": "Metformin Hydrochloride",
                        "relationship": "has_adverse_reaction",
                        "target_entity": "adverse reactions",
                    },
                    "confidence": 0.86,
                },
                {
                    "source_skill": "openFDA Human Drug",
                    "source_locator": "openFDA Human Drug",
                    "metadata": {
                        "source_entity": "Metformin Hydrochloride",
                        "relationship": "has_adverse_reaction",
                        "target_entity": "adverse reactions",
                    },
                    "confidence": 0.86,
                },
            ],
            "citations": ["[openfda:1] openFDA Human Drug — openFDA Human Drug"],
        },
    }

    formatted = wrap_answer_card("Metformin labeling answer", result)

    assert formatted.count("| 1 | openFDA Human Drug | Metformin Hydrochloride | has_adverse_reaction | adverse reactions | 0.86 | openFDA Human Drug (+2 similar records) |") == 1
    assert "| 2 | openFDA Human Drug | Metformin Hydrochloride | has_adverse_reaction | adverse reactions | 0.86 | openFDA Human Drug |" not in formatted


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


def test_wrap_answer_card_places_answer_before_metadata_summary() -> None:
    result = {
        "query": "What are the known drug targets and mechanism of action of imatinib?",
        "mode": "simple",
        "iterations": 0,
        "evidence_graph_size": 0,
        "final_reward": 0.0,
        "resource_filter": [],
        "retrieved_content": [],
        "final_answer_structured": {
            "summary_confidence": 0.79,
            "task_type": "composite_query",
            "final_outcome": "strong_answer",
            "diagnostics": {
                "target_support_count": 36,
                "mechanism_support_count": 4,
            },
            "key_claims": [],
            "evidence_items": [],
            "citations": [],
            "limitations": [],
            "warnings": [],
        },
    }

    formatted = wrap_answer_card(
        "Short Answer:\n- Primary supported answer: inhibits ABL1, inhibits KIT, inhibits PDGFRB",
        result,
    )

    assert formatted.index("## Answer") < formatted.index("Short Answer:")
    assert formatted.index("Short Answer:") < formatted.index("## Run Metadata")
    assert formatted.index("## Run Metadata") < formatted.index("**Query**")
