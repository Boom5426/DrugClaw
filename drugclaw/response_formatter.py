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
from html import escape
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
    """Generate a metadata badge block at the top of the answer."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    mode_badge = f"`{mode.upper()}`"
    lines = [
        f"# DrugClaw Query Report",
        f"",
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


def _confidence_tone(reward: float) -> str:
    level = _reward_to_level(reward)
    return {
        "high": "good",
        "medium": "warn",
        "low": "bad",
        "none": "muted",
    }[level]


def _render_html_list(items: List[str], empty_text: str = "None") -> str:
    if not items:
        return f"<p class=\"muted\">{escape(empty_text)}</p>"
    rendered = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<ul>{rendered}</ul>"


def _render_html_evidence_table(evidence_rows: List[Dict[str, Any]]) -> str:
    if not evidence_rows:
        return "<p class=\"muted\">No structured evidence table available.</p>"

    rows = _merge_evidence_rows(evidence_rows)
    body = []
    for item in rows[:20]:
        refs = item.get("sources", []) or []
        ref_text = ", ".join(str(ref) for ref in refs[:2]) if refs else "—"
        similar_records = int(item.get("similar_records", 0) or 0)
        if similar_records > 0:
            ref_text = f"{ref_text} (+{similar_records} similar records)"
        body.append(
            "<tr>"
            f"<td>{escape(str(item.get('source', '—')))}</td>"
            f"<td>{escape(str(item.get('source_entity', '—')))}</td>"
            f"<td>{escape(str(item.get('relationship', '→')))}</td>"
            f"<td>{escape(str(item.get('target_entity', '—')))}</td>"
            f"<td>{escape(str(item.get('confidence', '—')))}</td>"
            f"<td>{escape(ref_text)}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr>"
        "<th>Source</th><th>Entity A</th><th>Relation</th>"
        "<th>Entity B</th><th>Confidence</th><th>Reference</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
    if len(rows) > 20:
        table += (
            f"<p class=\"muted\">Showing 20 of {len(rows)} evidence records.</p>"
        )
    return table


def _render_html_reasoning_trace(reasoning_history: List[Dict[str, Any]]) -> str:
    if not reasoning_history:
        return "<p class=\"muted\">No multi-step reasoning trace for this run.</p>"

    blocks = []
    for step in reasoning_history:
        preview = str(step.get("answer", "") or "")
        if len(preview) > 400:
            preview = preview[:400] + "..."
        blocks.append(
            "<details class=\"trace-step\">"
            f"<summary>Step {int(step.get('step', 0))} · reward={float(step.get('reward', 0.0)):.2f} · sufficiency={float(step.get('evidence_sufficiency', 0.0)):.2f}</summary>"
            f"<div class=\"trace-body\">{escape(preview)}</div>"
            "</details>"
        )
    return "".join(blocks)


def render_answer_report_html(
    result: Dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> str:
    query = str(result.get("query", "") or "")
    mode = str(result.get("mode", "graph") or "graph").upper()
    iterations = int(result.get("iterations", 0) or 0)
    graph_size = int(result.get("evidence_graph_size", 0) or 0)
    resource_filter = result.get("resource_filter", []) or []
    reasoning = result.get("reasoning_history", []) or []
    structured = result.get("final_answer_structured", {}) or {}
    confidence = _display_confidence(result, structured)
    answer = str(result.get("answer", "") or "")
    structured_citations = structured.get("citations", []) or []
    citations = _collect_source_citations(
        [] if structured_citations else result.get("retrieved_content", []),
        result.get("web_search_results", []) or [],
        citations=structured_citations,
    )
    evidence_rows = _result_evidence_rows(result, structured, query)
    key_claims = structured.get("key_claims", []) or []
    warnings = structured.get("warnings", []) or []
    limitations = structured.get("limitations", []) or []
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    resource_badges = "".join(
        f"<span class=\"badge\">{escape(str(skill))}</span>"
        for skill in resource_filter
    )
    key_claim_items = [
        (
            f"{claim.get('claim', '')} "
            f"(confidence {float(claim.get('confidence', 0.0)):.2f})"
        ).strip()
        for claim in key_claims[:6]
    ]
    confidence_bar_width = max(0, min(100, round(confidence * 100)))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DrugClaw Query Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --text: #1f2933;
      --muted: #5b6472;
      --line: #ded5c4;
      --accent: #0f766e;
      --accent-soft: #d7f0ed;
      --good: #1d7a46;
      --warn: #b7791f;
      --bad: #b42318;
      --muted-tone: #667085;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, #efe3cc 0, transparent 28rem),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 12px 30px rgba(54, 42, 27, 0.08);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 20px;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    h1, h2 {{
      margin: 0 0 14px;
      font-family: "IBM Plex Serif", "Georgia", serif;
    }}
    h1 {{ font-size: clamp(28px, 4vw, 40px); line-height: 1.1; }}
    h2 {{ font-size: 22px; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 20px;
    }}
    .meta-card {{
      padding: 14px 16px;
      border-radius: 14px;
      background: #fbf8f2;
      border: 1px solid var(--line);
    }}
    .meta-label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .meta-value {{
      font-weight: 700;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .badge {{
      display: inline-flex;
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }}
    .confidence {{
      padding: 16px 18px;
      border-radius: 16px;
      background: #fcfaf6;
      border: 1px solid var(--line);
    }}
    .confidence-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    .tone-good {{ color: var(--good); }}
    .tone-warn {{ color: var(--warn); }}
    .tone-bad {{ color: var(--bad); }}
    .tone-muted {{ color: var(--muted-tone); }}
    .bar {{
      height: 12px;
      border-radius: 999px;
      background: #ebe4d8;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #0f766e, #14b8a6);
      width: {confidence_bar_width}%;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 20px;
      margin-top: 20px;
    }}
    .panel {{
      padding: 22px;
    }}
    .answer-body {{
      white-space: pre-wrap;
      line-height: 1.7;
      font-size: 15px;
    }}
    .stack > * + * {{
      margin-top: 18px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.7;
    }}
    .muted {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .trace-step + .trace-step {{
      margin-top: 10px;
    }}
    .trace-step summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .trace-body {{
      white-space: pre-wrap;
      color: var(--muted);
      margin-top: 10px;
    }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    @media (max-width: 900px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 20px 14px 40px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <p class="eyebrow">DrugClaw Visual Report</p>
      <h1>{escape(query)}</h1>
      <div class="meta">
        <div class="meta-card"><span class="meta-label">Mode</span><span class="meta-value">{escape(mode)}</span></div>
        <div class="meta-card"><span class="meta-label">Iterations</span><span class="meta-value">{iterations}</span></div>
        <div class="meta-card"><span class="meta-label">Evidence Graph</span><span class="meta-value">{graph_size} entities</span></div>
        <div class="meta-card"><span class="meta-label">Generated</span><span class="meta-value">{escape(ts)}</span></div>
      </div>
      <div class="confidence">
        <div class="confidence-head">
          <strong>Confidence</strong>
          <span class="tone-{_confidence_tone(confidence)}">{confidence:.2f}</span>
        </div>
        <div class="bar"><div class="bar-fill"></div></div>
      </div>
      <div class="badges">{resource_badges or '<span class="muted">No explicit resource filter.</span>'}</div>
    </section>

    <div class="grid">
      <section class="panel stack">
        <div>
          <h2>Answer</h2>
          <div class="answer-body">{escape(answer)}</div>
        </div>
        <div>
          <h2>Evidence Summary</h2>
          {_render_html_evidence_table(evidence_rows)}
        </div>
        <div>
          <h2>Reasoning Trace</h2>
          {_render_html_reasoning_trace(reasoning)}
        </div>
      </section>

      <aside class="panel stack">
        <div>
          <h2>Key Claims</h2>
          {_render_html_list(key_claim_items, empty_text="No structured key claims available.")}
        </div>
        <div>
          <h2>Warnings</h2>
          {_render_html_list([str(item) for item in warnings], empty_text="No warnings.")}
        </div>
        <div>
          <h2>Limitations</h2>
          {_render_html_list([str(item) for item in limitations], empty_text="No explicit limitations.")}
        </div>
        <div>
          <h2>Sources</h2>
          {_render_html_list([str(item) for item in citations[:10]], empty_text="No source list available.")}
          {f'<p class="muted">Showing 10 of {len(citations)} sources.</p>' if len(citations) > 10 else ''}
        </div>
      </aside>
    </div>

    <p class="footer">Generated by DrugClaw — Drug-Specialized Agentic RAG System</p>
  </div>
</body>
</html>
"""


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

    sections = [
        format_metadata_header(
            query, mode, iterations, graph_size, confidence,
            resource_filter, timestamp,
        ),
        "---",
        "",
        "## Answer",
        "",
        answer,
        "",
        "---",
        format_evidence_table(evidence_rows),
        format_reasoning_trace(reasoning),
        format_source_citations(citation_rows, web_results, citations=structured_citations),
        format_skills_used(resource_filter),
        "---",
        f"*Generated by DrugClaw — Drug-Specialized Agentic RAG System*",
        "",
    ]
    return "\n".join(sections)
