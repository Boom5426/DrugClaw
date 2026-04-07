from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse


def summarize_web_results(results: Iterable[Dict[str, str]], *, limit: int = 3) -> List[str]:
    lines: List[str] = []
    for result in list(results)[:limit]:
        source = str(result.get("source", "Web")).strip() or "Web"
        title = str(result.get("title", "")).strip() or "Untitled result"
        url = _result_url(result)
        domain = _normalize_domain(url)
        snippet = " ".join(str(result.get("snippet", "")).split()).strip()
        if snippet:
            lines.append(f"- {source} / {domain}: {title} - {snippet}")
        else:
            lines.append(f"- {source} / {domain}: {title}")
    return lines


def build_web_citations(results: Iterable[Dict[str, str]], *, limit: int = 5) -> List[str]:
    citations: List[str] = []
    seen = set()
    for result in list(results)[:limit]:
        source = str(result.get("source", "Web")).strip() or "Web"
        url = _result_url(result)
        if not url:
            continue
        citation = f"[web:{source}] {url}"
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


def build_task_aware_web_section(
    question_type: str,
    results: Iterable[Dict[str, str]],
    *,
    limit: int = 3,
) -> Tuple[str, List[str]] | None:
    results_list = list(results)[:limit]
    if not results_list:
        return None

    heading = _task_heading(question_type)
    lines = [_format_task_aware_line(question_type, result) for result in results_list]
    return heading, lines


def _result_url(result: Dict[str, str]) -> str:
    url = str(result.get("url", "")).strip()
    if url:
        return url
    metadata = result.get("metadata", {}) or {}
    return str(metadata.get("url", "")).strip()


def _normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.") or "unknown"


def _task_heading(question_type: str) -> str:
    return {
        "ddi": "Authority-First Clinical Interaction Support:",
        "ddi_mechanism": "Authority-First Clinical Interaction Support:",
        "labeling": "Authority-First Labeling Support:",
        "pharmacogenomics": "Authority-First PGx Guidance:",
        "adr": "Authority-First Safety Support:",
        "drug_repurposing": "Authority-First Repurposing Support:",
        "mechanism": "Authority-First Mechanistic Support:",
        "target_lookup": "Authority Cross-Check:",
        "direct_targets": "Authority Cross-Check:",
        "target_profile": "Authority Cross-Check:",
    }.get(str(question_type or "").strip().lower(), "Authority-first web evidence:")


def _format_task_aware_line(question_type: str, result: Dict[str, str]) -> str:
    source = str(result.get("source", "Web")).strip() or "Web"
    title = str(result.get("title", "")).strip() or "Untitled result"
    url = _result_url(result)
    domain = _normalize_domain(url)
    snippet = " ".join(str(result.get("snippet", "")).split()).strip()
    support_label = _support_label(question_type, title=title, snippet=snippet)
    line = f"- {support_label} ({source} / {domain}): {title}"
    if snippet:
        line += f" - {snippet}"
    return line


def _support_label(question_type: str, *, title: str, snippet: str) -> str:
    lowered = f"{title} {snippet}".lower()
    normalized_type = str(question_type or "").strip().lower()

    if normalized_type in {"ddi", "ddi_mechanism"}:
        if any(
            marker in lowered
            for marker in ("cyp", "enzyme", "metabolism", "metabol", "inhibit", "induc", "exposure")
        ):
            return "Mechanistic support"
        if any(
            marker in lowered
            for marker in ("monitor", "contraind", "avoid", "dose", "management", "precaution")
        ):
            return "Management support"
        return "Interaction support"

    if normalized_type == "labeling":
        if any(
            marker in lowered
            for marker in ("warning", "boxed", "precaution", "lactic acidosis", "renal", "monitor")
        ):
            return "Regulatory warning support"
        if any(
            marker in lowered
            for marker in ("indication", "dosing", "dose", "contraindication", "clinical use", "label")
        ):
            return "Label support"
        return "Clinical label support"

    if normalized_type == "pharmacogenomics":
        if any(
            marker in lowered
            for marker in ("cpic", "guideline", "pharmgkb", "recommendation", "actionable", "therapy")
        ):
            return "Guideline support"
        if any(
            marker in lowered
            for marker in ("genotype", "metabolizer", "metaboliser", "variant", "allele", "cyp")
        ):
            return "Genotype impact support"
        return "PGx support"

    if normalized_type == "adr":
        return "Safety support"
    if normalized_type == "drug_repurposing":
        return "Repurposing support"
    return "Supporting evidence"
