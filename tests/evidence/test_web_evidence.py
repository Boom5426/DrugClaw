from __future__ import annotations

from drugclaw.web_query_policy import build_simple_search_queries, filter_results_for_question_type
from drugclaw.web_evidence import (
    build_task_aware_web_section,
    build_web_citations,
    summarize_web_results,
)


def test_summarize_web_results_formats_authority_entries() -> None:
    summaries = summarize_web_results(
        [
            {
                "source": "PubMed",
                "title": "Clopidogrel and CYP2C19 guideline evidence",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "snippet": "CPIC guidance highlights reduced efficacy in poor metabolizers.",
            }
        ]
    )

    assert summaries == [
        "- PubMed / pubmed.ncbi.nlm.nih.gov: Clopidogrel and CYP2C19 guideline evidence - CPIC guidance highlights reduced efficacy in poor metabolizers."
    ]


def test_build_web_citations_uses_source_and_url() -> None:
    citations = build_web_citations(
        [
            {
                "source": "PubMed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            }
        ]
    )

    assert citations == ["[web:PubMed] https://pubmed.ncbi.nlm.nih.gov/12345678/"]


def test_build_task_aware_web_section_classifies_pgx_guidance_support() -> None:
    section = build_task_aware_web_section(
        "pharmacogenomics",
        [
            {
                "source": "PubMed",
                "title": "CPIC-guided clopidogrel therapy",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "snippet": "Clinical guidance supports altered antiplatelet strategy in CYP2C19 poor metabolizers.",
            }
        ],
    )

    assert section is not None
    heading, lines = section
    assert heading == "Authority-First PGx Guidance:"
    assert lines == [
        "- Guideline support (PubMed / pubmed.ncbi.nlm.nih.gov): CPIC-guided clopidogrel therapy - Clinical guidance supports altered antiplatelet strategy in CYP2C19 poor metabolizers."
    ]


def test_build_task_aware_web_section_classifies_ddi_mechanistic_support() -> None:
    section = build_task_aware_web_section(
        "ddi_mechanism",
        [
            {
                "source": "DailyMed",
                "title": "Warfarin label interaction precautions",
                "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=example",
                "snippet": "Monitor INR closely when combined with CYP2C9 inhibitors because exposure may increase.",
            }
        ],
    )

    assert section is not None
    heading, lines = section
    assert heading == "Authority-First Clinical Interaction Support:"
    assert lines == [
        "- Mechanistic support (DailyMed / dailymed.nlm.nih.gov): Warfarin label interaction precautions - Monitor INR closely when combined with CYP2C9 inhibitors because exposure may increase."
    ]


def test_build_task_aware_web_section_uses_cross_check_heading_for_target_lookup() -> None:
    section = build_task_aware_web_section(
        "target_lookup",
        [
            {
                "source": "PubMed",
                "title": "Imatinib mesylate",
                "url": "https://pubmed.ncbi.nlm.nih.gov/24756783/",
                "snippet": "Review article describing established kinase targets of imatinib.",
            }
        ],
    )

    assert section is not None
    heading, lines = section
    assert heading == "Authority Cross-Check:"
    assert lines == [
        "- Supporting evidence (PubMed / pubmed.ncbi.nlm.nih.gov): Imatinib mesylate - Review article describing established kinase targets of imatinib."
    ]


def test_build_simple_search_queries_prioritizes_official_label_domains_for_labeling() -> None:
    queries = build_simple_search_queries(
        query="What prescribing and safety information is available for metformin?",
        question_type="labeling",
    )

    assert queries
    assert queries[0]["source"] == "duckduckgo"
    assert "site:dailymed.nlm.nih.gov" in queries[0]["query"]
    assert "site:fda.gov" in queries[0]["query"]
    assert any(query_info["source"] == "pubmed" for query_info in queries)


def test_filter_results_for_labeling_prioritizes_dailymed_and_fda_before_pubmed() -> None:
    results = filter_results_for_question_type(
        "labeling",
        [
            {
                "source": "PubMed",
                "title": "Metformin review",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "snippet": "Review of metformin use.",
            },
            {
                "source": "DailyMed",
                "title": "Metformin prescribing information",
                "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=metformin",
                "snippet": "Official labeling includes warnings and renal monitoring guidance.",
            },
            {
                "source": "FDA",
                "title": "Drug safety communication",
                "url": "https://www.fda.gov/drugs/example",
                "snippet": "FDA safety communication.",
            },
        ],
    )

    assert [result["source"] for result in results[:2]] == ["DailyMed", "FDA"]


def test_build_simple_search_queries_adds_primary_drug_context_for_warfarin_ddi_mechanism() -> None:
    queries = build_simple_search_queries(
        query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        question_type="ddi_mechanism",
        candidate_entities=["warfarin"],
    )

    assert queries[0]["source"] == "duckduckgo"
    assert "site:dailymed.nlm.nih.gov" in queries[0]["query"]
    assert "site:fda.gov" in queries[0]["query"]
    assert "warfarin" in queries[0]["query"].lower()
    assert any(marker in queries[0]["query"].lower() for marker in ("interaction", "inr", "cyp2c9"))

    pubmed_queries = [
        query_info["query"].lower()
        for query_info in queries
        if query_info["source"] == "pubmed"
    ]

    assert pubmed_queries
    assert any("warfarin" in query for query in pubmed_queries)
    assert any(
        any(marker in query for marker in ("inr", "cyp2c9", "bleeding", "interaction", "monitor"))
        for query in pubmed_queries
    )


def test_filter_results_for_ddi_mechanism_drops_off_topic_pubmed_hits() -> None:
    results = filter_results_for_question_type(
        "ddi_mechanism",
        [
            {
                "source": "PubMed",
                "title": "Pharmacokinetic drug interaction profiles of proton pump inhibitors: an update.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/11111111/",
                "snippet": "Review of proton pump inhibitor interactions.",
            },
            {
                "source": "PubMed",
                "title": "Warfarin interactions with CYP2C9 inhibitors and INR management.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/22222222/",
                "snippet": "Warfarin exposure and INR can increase with CYP2C9 inhibition, requiring monitoring.",
            },
            {
                "source": "PubMed",
                "title": "Vericiguat, a novel sGC stimulator: Mechanism of action, clinical, and translational science.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/33333333/",
                "snippet": "Overview of vericiguat pharmacology and clinical development.",
            },
        ],
        query="What are the clinically important drug-drug interactions of warfarin and their mechanisms?",
        candidate_entities=["warfarin"],
    )

    assert [result["title"] for result in results] == [
        "Warfarin interactions with CYP2C9 inhibitors and INR management."
    ]


def test_filter_results_for_adr_keeps_relevant_clozapine_monitoring_support() -> None:
    results = filter_results_for_question_type(
        "adr",
        [
            {
                "source": "PubMed",
                "title": "Clozapine treatment: Ensuring ongoing monitoring during the COVID-19 pandemic.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "snippet": "Clozapine monitoring and ANC surveillance remain essential for severe neutropenia risk.",
            },
            {
                "source": "PubMed",
                "title": "Maintenance treatment in schizophrenia: general considerations.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/87654321/",
                "snippet": "Overview of schizophrenia maintenance strategies in general psychiatric practice.",
            },
        ],
        query="What are the major safety risks and serious adverse reactions of clozapine?",
        candidate_entities=["clozapine"],
    )

    assert [result["title"] for result in results] == [
        "Clozapine treatment: Ensuring ongoing monitoring during the COVID-19 pandemic."
    ]


def test_build_simple_search_queries_focuses_official_adr_query_on_primary_risks() -> None:
    queries = build_simple_search_queries(
        query="What are the major safety risks and serious adverse reactions of clozapine?",
        question_type="adr",
        candidate_entities=["clozapine"],
    )

    assert queries[0]["source"] == "duckduckgo"
    lowered = queries[0]["query"].lower()
    assert "clozapine" in lowered
    assert any(marker in lowered for marker in ("warning", "anc", "neutropenia", "agranulocytosis"))
