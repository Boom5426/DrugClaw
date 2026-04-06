"""
Response Formatter — wraps raw LLM answers into rich, structured Markdown.

Provides multiple formatting skills:
  - format_answer()          : full-featured answer card with metadata badges
  - format_evidence_table()  : tabular evidence summary
  - format_reasoning_trace() : step-by-step reasoning in Markdown
  - format_source_citations(): numbered source list with links
  - format_confidence_badge(): visual confidence indicator
  - wrap_answer_card()       : complete answer card combining all elements
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Confidence visual helpers ────────────────────────────────────────

_CONFIDENCE_BARS = {
    "high":   "████████████████████ HIGH",
    "medium": "████████████░░░░░░░░ MEDIUM",
    "low":    "████████░░░░░░░░░░░░ LOW",
    "none":   "░░░░░░░░░░░░░░░░░░░░ N/A",
}

_CONFIDENCE_EMOJI = {
    "high":   "🟢",
    "medium": "🟡",
    "low":    "🔴",
    "none":   "⚪",
}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_to_level(reward: float) -> str:
    if reward >= 0.7:
        return "high"
    if reward >= 0.4:
        return "medium"
    if reward > 0:
        return "low"
    return "none"


# ── Public formatting skills ─────────────────────────────────────────

def format_confidence_badge(reward: float) -> str:
    """Return a Markdown confidence bar with emoji."""
    level = _reward_to_level(reward)
    emoji = _CONFIDENCE_EMOJI[level]
    bar   = _CONFIDENCE_BARS[level]
    return f"{emoji} **Confidence** `{bar}` ({reward:.2f})"


def format_metadata_header(
    query: str,
    mode: str,
    iterations: int,
    graph_size: int,
    reward: float,
    resource_filter: Optional[List[str]] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """Generate a metadata badge block for the run metadata section."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    mode_badge = f"`{mode.upper()}`"
    lines = [
        f"> **Query** : {query}",
        f"> **Mode** : {mode_badge}  |  **Iterations** : `{iterations}`  |  "
        f"**Evidence Graph** : `{graph_size} entities`",
        f"> **Time** : {ts}",
    ]
    if resource_filter:
        skills = ", ".join(f"`{s}`" for s in resource_filter)
        lines.append(f"> **Resources** : {skills}")
    lines += ["", format_confidence_badge(reward), ""]
    return "\n".join(lines)


def format_task_outcome_block(structured: Dict[str, Any]) -> str:
    task_type = str(structured.get("task_type", "") or "").strip()
    final_outcome = str(structured.get("final_outcome", "") or "").strip()
    diagnostics = structured.get("diagnostics", {}) or {}

    if not task_type and not final_outcome and not diagnostics:
        return ""

    lines: List[str] = []
    summary_bits: List[str] = []
    if task_type:
        summary_bits.append(f"**Task Type** : `{task_type}`")
    if final_outcome:
        summary_bits.append(f"**Final Outcome** : `{final_outcome}`")
    if summary_bits:
        lines.append("> " + "  |  ".join(summary_bits))

    diagnostic_bits: List[str] = []
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, list):
            if not value:
                continue
            shown = ", ".join(str(entry) for entry in value[:3])
            if len(value) > 3:
                shown += ", ..."
            diagnostic_bits.append(f"`{key}={shown}`")
            continue
        diagnostic_bits.append(f"`{key}={value}`")
    if diagnostic_bits:
        lines.append("> **Diagnostics** : " + "  ".join(diagnostic_bits[:6]))

    lines.append("")
    return "\n".join(lines)


def format_evidence_table(
    evidence_items: List[Dict[str, Any]],
    max_rows: int = 20,
) -> str:
    """Render evidence records as a Markdown table."""
    if not evidence_items:
        return ""

    merged_items = _merge_evidence_rows(evidence_items)

    lines = [
        "",
        "## Evidence Summary",
        "",
        "| # | Source | Entity A | Relation | Entity B | Confidence | Reference |",
        "|---|--------|----------|----------|----------|------------|-----------|",
    ]
    for i, item in enumerate(merged_items[:max_rows], 1):
        src  = item.get("source", "—")
        se   = item.get("source_entity", "—")
        rel  = item.get("relationship", "→")
        te   = item.get("target_entity", "—")
        conf = item.get("confidence", "—")
        refs = item.get("sources", [])
        ref_str = ", ".join(refs[:2]) if refs else "—"
        similar_records = int(item.get("similar_records", 0) or 0)
        if similar_records > 0:
            ref_str = (
                f"{ref_str} (+{similar_records} similar records)"
                if ref_str != "—"
                else f"+{similar_records} similar records"
            )
        lines.append(f"| {i} | {src} | {se} | {rel} | {te} | {conf} | {ref_str} |")

    if len(merged_items) > max_rows:
        lines.append(f"\n> *Showing {max_rows} of {len(merged_items)} evidence records.*")
    lines.append("")
    return "\n".join(lines)


def _merge_evidence_rows(evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_key: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}

    for item in evidence_items:
        source = str(item.get("source", "—")).strip()
        source_entity = str(item.get("source_entity", "—")).strip()
        relationship = str(item.get("relationship", "→")).strip()
        target_entity = str(item.get("target_entity", "—")).strip()
        key = (
            source.lower(),
            source_entity.lower(),
            relationship.lower(),
            target_entity.lower(),
        )

        existing = by_key.get(key)
        if existing is None:
            merged_item = dict(item)
            merged_item["sources"] = list(dict.fromkeys(item.get("sources", []) or []))
            merged_item["similar_records"] = int(item.get("similar_records", 0) or 0)
            by_key[key] = merged_item
            merged.append(merged_item)
            continue

        existing["similar_records"] = int(existing.get("similar_records", 0) or 0) + 1

        merged_sources = list(existing.get("sources", []) or [])
        for source_ref in item.get("sources", []) or []:
            if source_ref not in merged_sources:
                merged_sources.append(source_ref)
        existing["sources"] = merged_sources

        existing_conf = _coerce_float(existing.get("confidence"), -1.0)
        candidate_conf = _coerce_float(item.get("confidence"), -1.0)
        if candidate_conf > existing_conf:
            existing["confidence"] = item.get("confidence", existing.get("confidence", "—"))

    return merged


def _structured_evidence_to_table_rows(
    evidence_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in evidence_items:
        metadata = item.get("metadata", {}) or {}
        rows.append(
            {
                "source": item.get("source_skill", "—"),
                "source_entity": metadata.get("source_entity", "—"),
                "relationship": metadata.get("relationship", "→"),
                "target_entity": metadata.get("target_entity", "—"),
                "confidence": f"{float(item.get('confidence', 0.0)):.2f}",
                "sources": [item.get("source_locator", "—")],
            }
        )
    return rows


def _claim_to_target_label(claim: str) -> str:
    text = str(claim or "").strip().rstrip(".")
    if " targets " in text:
        return text.split(" targets ", 1)[1].strip()
    return text


def _canonical_source_entity(items: List[Dict[str, Any]]) -> str:
    labels = []
    for item in items:
        metadata = item.get("metadata", {}) or {}
        label = str(metadata.get("source_entity", "")).strip()
        if label:
            labels.append(label)
    if not labels:
        return "—"
    lowered = {label.lower() for label in labels}
    if len(lowered) == 1:
        return labels[0].lower()
    return labels[0]


def _target_claim_rows(structured: Dict[str, Any]) -> List[Dict[str, Any]]:
    key_claims = structured.get("key_claims", []) or []
    evidence_items = structured.get("evidence_items", []) or []
    evidence_by_id = {
        str(item.get("evidence_id", "")): item
        for item in evidence_items
        if item.get("evidence_id")
    }

    rows: List[Dict[str, Any]] = []
    for claim in key_claims:
        claim_text = str(claim.get("claim", "")).strip()
        evidence_ids = [str(evidence_id) for evidence_id in claim.get("evidence_ids", []) if evidence_id]
        claim_items = [evidence_by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_by_id]
        if not claim_items or " targets " not in claim_text:
            continue

        sources = list(dict.fromkeys(str(item.get("source_skill", "—")) for item in claim_items))
        rows.append(
            {
                "source": ", ".join(sources),
                "source_entity": _canonical_source_entity(claim_items),
                "relationship": "targets",
                "target_entity": _claim_to_target_label(claim_text),
                "confidence": f"{float(claim.get('confidence', 0.0)):.2f}",
                "sources": evidence_ids[:2],
            }
        )
    return rows


def format_reasoning_trace(reasoning_history: List[Dict[str, Any]]) -> str:
    """Render reasoning steps as a Markdown timeline."""
    if not reasoning_history:
        return ""

    lines = [
        "",
        "## Reasoning Trace",
        "",
    ]
    for step in reasoning_history:
        step_id = step.get("step", 0)
        reward  = step.get("reward", 0.0)
        suff    = step.get("evidence_sufficiency", 0.0)
        answer  = step.get("answer", "")
        preview = (answer[:200] + "...") if len(answer) > 200 else answer
        level   = _reward_to_level(reward)
        emoji   = _CONFIDENCE_EMOJI[level]

        lines += [
            f"### Step {step_id}  {emoji}  reward={reward:.3f}  sufficiency={suff:.3f}",
            f"",
            f"<details>",
            f"<summary>Intermediate answer preview</summary>",
            f"",
            f"{preview}",
            f"",
            f"</details>",
            f"",
        ]
    return "\n".join(lines)


def format_source_citations(
    retrieved_content: List[Dict[str, Any]],
    web_results: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[str]] = None,
    max_items: int = 6,
) -> str:
    """Collect and deduplicate sources into a numbered citation list."""
    seen: set[str] = set()
    sources: List[str] = []

    if citations:
        for citation in citations:
            if citation and citation not in seen:
                seen.add(citation)
                sources.append(citation)

    for item in retrieved_content:
        src = item.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    if web_results:
        for item in web_results:
            title = item.get("title", "")
            url   = item.get("url", "")
            label = f"[{title}]({url})" if url else title
            if label and label not in seen:
                seen.add(label)
                sources.append(label)

    if not sources:
        return ""

    lines = ["", "## Sources", ""]
    displayed_sources = sources[:max_items] if max_items > 0 else sources
    for i, src in enumerate(displayed_sources, 1):
        lines.append(f"{i}. {src}")
    if max_items > 0 and len(sources) > max_items:
        omitted = len(sources) - max_items
        lines.extend(
            [
                "",
                f"> *Showing {max_items} of {len(sources)} sources; {omitted} more omitted.*",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _collect_source_citations(
    retrieved_content: List[Dict[str, Any]],
    web_results: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[str]] = None,
) -> List[str]:
    seen: set[str] = set()
    sources: List[str] = []

    if citations:
        for citation in citations:
            if citation and citation not in seen:
                seen.add(citation)
                sources.append(citation)

    for item in retrieved_content:
        src = item.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)

    if web_results:
        for item in web_results:
            title = item.get("title", "")
            url = item.get("url", "")
            label = f"{title} — {url}" if title and url else title or url
            if label and label not in seen:
                seen.add(label)
                sources.append(label)

    return sources


def _display_confidence(result: Dict[str, Any], structured: Dict[str, Any]) -> float:
    if "summary_confidence" in structured:
        return _coerce_float(structured.get("summary_confidence"), 0.0)
    return _coerce_float(result.get("final_reward", 0.0), 0.0)


def _result_evidence_rows(
    result: Dict[str, Any],
    structured: Dict[str, Any],
    query: str,
) -> List[Dict[str, Any]]:
    structured_evidence = structured.get("evidence_items", []) or []
    target_claim_rows = _target_claim_rows(structured) if "target" in str(query).lower() else []
    return target_claim_rows or (
        _structured_evidence_to_table_rows(structured_evidence)
        if structured_evidence else result.get("retrieved_content", [])
    )


def format_skills_used(resource_filter: Optional[List[str]] = None) -> str:
    """Show which skills/resources were queried."""
    if not resource_filter:
        return ""
    badges = "  ".join(f"`{s}`" for s in resource_filter)
    return f"\n## Skills Used\n\n{badges}\n"


def wrap_answer_card(
    answer: str,
    result: Dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Wrap a raw LLM answer string into a full Markdown report card.

    Parameters
    ----------
    answer   : The raw LLM answer text.
    result   : The full result dict from DrugClawSystem.query().
    timestamp: Optional override timestamp.

    Returns
    -------
    A rich Markdown string ready for display or file storage.
    """
    query          = result.get("query", "")
    mode           = result.get("mode", "graph")
    iterations     = result.get("iterations", 0)
    graph_size     = result.get("evidence_graph_size", 0)
    resource_filter = result.get("resource_filter", [])
    reasoning      = result.get("reasoning_history", [])
    retrieved      = result.get("retrieved_content", [])
    web_results    = result.get("web_search_results", [])
    structured     = result.get("final_answer_structured", {}) or {}
    confidence     = _display_confidence(result, structured)
    structured_citations = structured.get("citations", []) or []
    evidence_rows  = _result_evidence_rows(result, structured, query)
    citation_rows = [] if structured_citations else retrieved
    source_web_results = [] if structured_citations else web_results

    sections = [
        "# DrugClaw Query Report",
        "",
        "## Answer",
        "",
        answer,
        "",
        "---",
        format_evidence_table(evidence_rows),
        format_reasoning_trace(reasoning),
        format_source_citations(citation_rows, source_web_results, citations=structured_citations),
        format_skills_used(resource_filter),
        "## Run Metadata",
        "",
        format_task_outcome_block(structured),
        format_metadata_header(
            query, mode, iterations, graph_size, confidence,
            resource_filter, timestamp,
        ),
        "---",
        f"*Generated by DrugClaw — Drug-Specialized Agentic RAG System*",
        "",
    ]
    return "\n".join(sections)
