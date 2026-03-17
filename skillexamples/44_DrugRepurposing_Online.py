"""
DrugRepurposing Online - Drug Repurposing Resource Aggregation
Category: Drug-centric | Type: DB | Subcategory: Drug Repurposing
Link: https://drugrepurposing.info/

DrugRepurposing.info aggregates computational and experimental drug repurposing
results from multiple sources with a searchable interface.

Access method: Web scraping (no public API).
"""

import urllib.request
import urllib.parse
import re
import json
import time
from typing import Union

BASE_URL = "https://drugrepurposing.info"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
DELAY = 1.0  # polite delay between requests (seconds)


# ── helpers ──────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 20) -> str:
    """Fetch a URL and return decoded HTML."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_tables(html: str) -> list[list[list[str]]]:
    """
    Extract all HTML tables as list-of-rows (each row = list of cell texts).
    Returns a list of tables; each table is a list of rows.
    """
    tables = []
    for table_match in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.S | re.I):
        table_html = table_match.group(1)
        rows = []
        for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_html, re.S | re.I):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_match.group(1), re.S | re.I)
            cells = [_strip_html(c) for c in cells]
            if any(c for c in cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _extract_links(html: str, pattern: str = "") -> list[dict]:
    """Extract <a> links from HTML, optionally filtering href by pattern."""
    links = []
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S | re.I):
        href, text = m.group(1), _strip_html(m.group(2))
        if pattern and pattern.lower() not in href.lower():
            continue
        if text:
            links.append({"href": href, "text": text})
    return links


# ── core query functions ─────────────────────────────────────────────

def query_entity(entity: str) -> dict:
    """
    Query DrugRepurposing.info for a single entity (drug name).

    Returns:
        dict with keys:
          - entity: query string
          - url: search URL used
          - tables: list of extracted tables (list of rows)
          - links: relevant links found on the results page
          - snippet: first 1000 chars of visible text (fallback)
          - error: error message if request failed, else None
    """
    params = urllib.parse.urlencode({"q": entity})
    url = f"{BASE_URL}/search?{params}"
    result = {
        "entity": entity,
        "url": url,
        "tables": [],
        "links": [],
        "snippet": "",
        "error": None,
    }
    try:
        html = _fetch(url)
        result["tables"] = _extract_tables(html)
        result["links"] = _extract_links(html, pattern="drug")[:20]
        result["snippet"] = _strip_html(html)[:1000]
    except Exception as e:
        result["error"] = str(e)
    return result


def query_entities(entities: Union[str, list[str]], delay: float = DELAY) -> list[dict]:
    """
    Query DrugRepurposing.info for one or more entities.

    Args:
        entities: a single drug name (str) or list of drug names.
        delay: seconds to wait between requests (default 1.0).

    Returns:
        List of result dicts (see query_entity for schema).
    """
    if isinstance(entities, str):
        entities = [entities]
    results = []
    for i, entity in enumerate(entities):
        results.append(query_entity(entity))
        if i < len(entities) - 1:
            time.sleep(delay)
    return results


def format_results(results: list[dict]) -> str:
    """Format query results into LLM-readable plain text."""
    lines = []
    for r in results:
        lines.append(f"## {r['entity']}")
        lines.append(f"Search URL: {r['url']}")
        if r["error"]:
            lines.append(f"Error: {r['error']}")
            lines.append("")
            continue
        if r["tables"]:
            for ti, table in enumerate(r["tables"]):
                lines.append(f"Table {ti + 1} ({len(table)} rows):")
                for row in table[:30]:  # cap display rows
                    lines.append("  | " + " | ".join(row) + " |")
                if len(table) > 30:
                    lines.append(f"  ... ({len(table) - 30} more rows)")
        else:
            lines.append("No tables found.")
        if r["links"]:
            lines.append("Relevant links:")
            for lk in r["links"][:10]:
                lines.append(f"  - {lk['text']}: {lk['href']}")
        if not r["tables"] and not r["links"]:
            lines.append(f"Text snippet: {r['snippet'][:500]}")
        lines.append("")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept entities from command-line args or use defaults
    if len(sys.argv) > 1:
        entities = sys.argv[1:]
    else:
        entities = ["aspirin", "metformin"]

    print(f"Querying DrugRepurposing.info for: {entities}\n")
    results = query_entities(entities)
    print(format_results(results))

    # Also dump JSON for programmatic use
    out_path = "44_DrugRepurposing_Online_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to {out_path}")