from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse


HIGH_RISK_QUESTION_TYPES = {
    "adr",
    "ddi",
    "ddi_mechanism",
    "labeling",
    "pharmacogenomics",
}


_AUTHORITY_DOMAINS: Dict[str, set[str]] = {
    "target_lookup": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "opentargets.org",
        "platform.opentargets.org",
        "chembl.ebi.ac.uk",
        "drugbank.com",
    },
    "mechanism": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "opentargets.org",
        "platform.opentargets.org",
        "chembl.ebi.ac.uk",
        "drugbank.com",
        "nih.gov",
    },
    "drug_repurposing": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "clinicaltrials.gov",
        "fda.gov",
        "nih.gov",
    },
    "adr": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "fda.gov",
        "dailymed.nlm.nih.gov",
        "nih.gov",
    },
    "ddi": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "fda.gov",
        "dailymed.nlm.nih.gov",
        "nih.gov",
    },
    "ddi_mechanism": {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "fda.gov",
        "dailymed.nlm.nih.gov",
        "nih.gov",
    },
    "labeling": {
        "dailymed.nlm.nih.gov",
        "fda.gov",
        "nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    },
    "pharmacogenomics": {
        "cpicpgx.org",
        "pharmgkb.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nih.gov",
    },
}


def build_simple_search_queries(
    *,
    query: str,
    question_type: str,
    candidate_entities: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    primary_entity = _primary_candidate_entity(candidate_entities)

    if question_type == "labeling":
        queries.append(
            {
                "query": (
                    f"{query.strip()} "
                    "site:dailymed.nlm.nih.gov OR site:fda.gov"
                ).strip(),
                "purpose": "official labeling and safety context",
                "source": "duckduckgo",
            }
        )
        queries.append(
            {
                "query": query.strip(),
                "purpose": "supplemental labeling literature context",
                "source": "pubmed",
            }
        )
    elif question_type == "pharmacogenomics":
        queries.append(
            {
                "query": query.strip(),
                "purpose": "authority-first PGx guidance",
                "source": "pubmed",
            }
        )
    elif question_type in {"adr", "ddi", "ddi_mechanism"}:
        official_query = query.strip()
        if primary_entity:
            official_query = _build_high_risk_official_query(question_type, primary_entity)
        queries.append(
            {
                "query": (
                    f"{official_query} "
                    "site:dailymed.nlm.nih.gov OR site:fda.gov"
                ).strip(),
                "purpose": "official safety and prescribing context",
                "source": "duckduckgo",
            }
        )
        pubmed_query = query.strip()
        if primary_entity:
            pubmed_query = _build_high_risk_pubmed_query(question_type, primary_entity)
        queries.append(
            {
                "query": pubmed_query,
                "purpose": "authority-first labeling and safety context",
                "source": "pubmed",
            }
        )
    else:
        queries.append(
            {
                "query": query.strip(),
                "purpose": "authority-first biomedical evidence",
                "source": "pubmed",
            }
        )

    queries.append(
        {
            "query": query,
            "purpose": "supplemental authority web evidence",
            "source": "duckduckgo",
        }
    )

    seen = set()
    unique_queries: List[Dict[str, str]] = []
    for query_info in queries:
        key = (query_info["query"], query_info["source"])
        if key in seen:
            continue
        seen.add(key)
        unique_queries.append(query_info)
    return unique_queries


def filter_results_for_question_type(
    question_type: str,
    results: Iterable[Dict[str, str]],
    *,
    query: str = "",
    candidate_entities: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    normalized_type = str(question_type or "").strip().lower()
    results_list = list(results)
    if not results_list:
        return []

    authority_results = [
        result for result in results_list
        if is_authority_result(normalized_type, result)
    ]
    authority_results = sorted(
        authority_results,
        key=lambda result: _authority_priority(normalized_type, result),
    )
    authority_results = _filter_authority_results_for_context(
        normalized_type,
        authority_results,
        query=query,
        candidate_entities=candidate_entities,
    )
    if normalized_type in HIGH_RISK_QUESTION_TYPES:
        return authority_results
    return authority_results or results_list[:5]


def is_authority_result(question_type: str, result: Dict[str, str]) -> bool:
    source = str(result.get("source", "")).strip().lower()
    if source == "pubmed":
        return True
    if source == "clinicaltrials.gov":
        return True

    domain = normalize_result_domain(result)
    if not domain:
        return False

    allowed = _AUTHORITY_DOMAINS.get(question_type) or _AUTHORITY_DOMAINS.get("mechanism", set())
    return any(domain == authority or domain.endswith(f".{authority}") for authority in allowed)


def normalize_result_domain(result: Dict[str, str]) -> str:
    url = str(result.get("url", "")).strip()
    if not url:
        metadata = result.get("metadata", {}) or {}
        url = str(metadata.get("url", "")).strip()
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _authority_priority(question_type: str, result: Dict[str, str]) -> tuple[int, str, str]:
    normalized_type = str(question_type or "").strip().lower()
    source = str(result.get("source", "")).strip().lower()
    domain = normalize_result_domain(result)

    if normalized_type in {"labeling", "adr", "ddi", "ddi_mechanism"}:
        if domain == "dailymed.nlm.nih.gov":
            return (0, domain, source)
        if domain == "fda.gov" or domain.endswith(".fda.gov"):
            return (1, domain, source)
        if domain == "nih.gov" or domain.endswith(".nih.gov"):
            return (2, domain, source)
        if source == "pubmed" or domain.endswith("pubmed.ncbi.nlm.nih.gov") or domain.endswith("ncbi.nlm.nih.gov"):
            return (3, domain, source)
        return (4, domain, source)

    return (0, domain, source)


def _primary_candidate_entity(candidate_entities: Sequence[str] | None) -> str:
    for entity in candidate_entities or []:
        normalized = " ".join(str(entity or "").split()).strip()
        if normalized:
            return normalized.lower()
    return ""


def _build_high_risk_pubmed_query(question_type: str, primary_entity: str) -> str:
    normalized_type = str(question_type or "").strip().lower()
    if normalized_type == "adr":
        return (
            f"{primary_entity} safety adverse reaction monitoring ANC "
            "neutropenia agranulocytosis myocarditis seizure"
        ).strip()
    if normalized_type == "ddi":
        return (
            f"{primary_entity} drug interaction clinically important INR "
            "bleeding monitor management"
        ).strip()
    if normalized_type == "ddi_mechanism":
        return (
            f"{primary_entity} drug interaction mechanism CYP2C9 INR "
            "bleeding monitor management"
        ).strip()
    return primary_entity


def _build_high_risk_official_query(question_type: str, primary_entity: str) -> str:
    normalized_type = str(question_type or "").strip().lower()
    if normalized_type == "adr":
        return (
            f"{primary_entity} warning ANC neutropenia agranulocytosis "
            "myocarditis seizure monitoring"
        ).strip()
    if normalized_type == "ddi":
        return (
            f"{primary_entity} drug interactions INR bleeding monitoring"
        ).strip()
    if normalized_type == "ddi_mechanism":
        return (
            f"{primary_entity} drug interactions CYP2C9 INR bleeding monitoring"
        ).strip()
    return primary_entity


def _filter_authority_results_for_context(
    question_type: str,
    results: Iterable[Dict[str, str]],
    *,
    query: str = "",
    candidate_entities: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    normalized_type = str(question_type or "").strip().lower()
    results_list = list(results)
    if normalized_type not in {"adr", "ddi", "ddi_mechanism"}:
        return results_list

    primary_entity = _primary_candidate_entity(candidate_entities)
    if not primary_entity and not str(query or "").strip():
        return results_list

    filtered = [
        result
        for result in results_list
        if _authority_result_matches_question_context(
            normalized_type,
            result,
            primary_entity=primary_entity,
        )
    ]
    return filtered


def _authority_result_matches_question_context(
    question_type: str,
    result: Dict[str, str],
    *,
    primary_entity: str = "",
) -> bool:
    text = _result_text(result)
    if not text:
        return False

    if primary_entity and not _text_contains_term(text, primary_entity):
        return False

    if question_type in {"ddi", "ddi_mechanism"}:
        return any(
            marker in text
            for marker in (
                "interaction",
                "interactions",
                "drug-drug",
                "cyp",
                "enzyme",
                "metabolism",
                "metabol",
                "inhibit",
                "induc",
                "exposure",
                "inr",
                "bleed",
                "anticoagul",
                "monitor",
                "dose",
                "management",
                "precaution",
                "contraind",
            )
        )

    if question_type == "adr":
        return any(
            marker in text
            for marker in (
                "safety",
                "adverse",
                "serious",
                "reaction",
                "risk",
                "warning",
                "monitor",
                "monitoring",
                "anc",
                "neutropenia",
                "agranulocytosis",
                "myocarditis",
                "seizure",
                "fatal",
            )
        )

    return True


def _result_text(result: Dict[str, str]) -> str:
    metadata = result.get("metadata", {}) or {}
    fields = [
        str(result.get("title", "")).strip().lower(),
        str(result.get("snippet", "")).strip().lower(),
        str(result.get("url", "")).strip().lower(),
        str(metadata.get("url", "")).strip().lower(),
    ]
    return " ".join(field for field in fields if field).strip()


def _text_contains_term(text: str, term: str) -> bool:
    normalized_term = " ".join(str(term or "").lower().split()).strip()
    if not normalized_term:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in normalized_term.split()) + r"\b"
    return re.search(pattern, text) is not None
