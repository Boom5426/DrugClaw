from __future__ import annotations

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
