"""
Skill 16 – GDKD (Gene-Drug Knowledge Database) query module.

Query disease–gene–variant–drug associations from the GDKD
Knowledge_database_v20.0.xlsx curated by Dienstmann et al.
Covers ~700 variant-specific gene–drug interactions with consensus
or emerging therapeutic relevance across solid tumours.

Source: https://www.synapse.org/#!Synapse:syn2370773
Paper : Dienstmann et al., Cancer Discovery 2015;5(2):118-123
"""

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

# ── data path ────────────────────────────────────────────────────────
DATA_PATH = (
    "/blue/qsong1/wang.qing/AgentLLM/DrugClaw/"
    "resources_metadata/dti/GDKD/Knowledge_database_v20.0.xlsx"
)

# ── column constants ─────────────────────────────────────────────────
CORE_COLS = ["Disease", "Gene", "Variant", "Description", "Effect"]
MAX_ASSOC = 8  # association slots 1‑8


# ── loader ───────────────────────────────────────────────────────────
def load_gdkd(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the GDKD xlsx and normalise column names."""
    df = pd.read_excel(path, engine="openpyxl")
    # strip whitespace from headers (source file has trailing spaces)
    df.columns = [str(c).strip() for c in df.columns]
    # strip cell values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": None, "None": None, "": None})
    return df


# ── association unpacking ────────────────────────────────────────────
def _unpack_associations(row: pd.Series) -> list[dict]:
    """Return a list of dicts, one per non-empty association slot."""
    assocs = []
    for i in range(1, MAX_ASSOC + 1):
        # handle slight naming irregularities (trailing space, underscore)
        def _col(prefix: str) -> Optional[str]:
            for suffix in [f"_{i}", f" _{i}", f"_{i} "]:
                cand = f"{prefix}{suffix}".strip()
                if cand in row.index:
                    v = row[cand]
                    if v is not None:
                        return str(v).strip()
            return None

        assoc_type = _col("Association")
        if assoc_type is None:
            continue
        assocs.append({
            "association": assoc_type,
            "therapeutic_context": _col("Therapeutic context"),
            "status": _col("Status"),
            "evidence": _col("Evidence"),
            "pmid": _col("PMID"),
        })
    return assocs


# ── entity detection ─────────────────────────────────────────────────
_GENE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,10}$")  # e.g. BRAF, ABL1, EGFR
_VARIANT_RE = re.compile(
    r"^[A-Z]\d+[A-Z]$"          # e.g. T315I, V600E
    r"|^(p\.)?"                  # optional p. prefix
    r"|amplification$"
    r"|deletion$"
    r"|overexpression$"
    r"|mutation$",
    re.IGNORECASE,
)


def _detect_type(entity: str) -> str:
    """Heuristic: gene symbol / variant / free text (disease or drug).

    Variant patterns (e.g. V600E, T315I) are checked *before* gene
    symbols because they also match the all-uppercase gene regex.
    """
    e = entity.strip()
    # variant first: single letter + digits + single letter  (V600E, T315I)
    if re.match(r"^[A-Z]\d+[A-Z]$", e):
        return "variant"
    if _GENE_RE.match(e):
        return "gene"
    if re.match(r"(?i)^(amplification|deletion|overexpression|fusion)", e):
        return "variant_keyword"
    return "text"


# ── search ───────────────────────────────────────────────────────────
def search(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Search GDKD by a single entity string. Auto-detects type.

    Returns a DataFrame subset of matching rows.
    """
    e = entity.strip()
    etype = _detect_type(e)
    el = e.lower()

    if etype == "gene":
        mask = df["Gene"].str.upper() == e.upper()
    elif etype == "variant":
        mask = df["Variant"].str.upper() == e.upper()
    elif etype == "variant_keyword":
        mask = df["Description"].str.lower().str.contains(el, na=False)
    else:
        # free text: search Disease, Gene, Description, and all
        # Therapeutic context columns
        mask = (
            df["Disease"].str.lower().str.contains(el, na=False)
            | df["Gene"].str.lower().str.contains(el, na=False)
            | df["Description"].str.lower().str.contains(el, na=False)
        )
        # also search therapeutic-context columns for drug names
        for i in range(1, MAX_ASSOC + 1):
            for col in df.columns:
                if col.startswith("Therapeutic context") and col.endswith(str(i)):
                    mask = mask | df[col].str.lower().str.contains(el, na=False)
                    break

    return df.loc[mask].reset_index(drop=True)


def search_batch(
    df: pd.DataFrame, entities: list[str]
) -> dict[str, pd.DataFrame]:
    """Search for a list of entities. Returns {entity: DataFrame}."""
    return {e: search(df, e) for e in entities}


# ── summarize ────────────────────────────────────────────────────────
def summarize(hits: pd.DataFrame, entity: str = "") -> str:
    """Compact LLM-readable summary of search results."""
    if hits.empty:
        return f"GDKD | {entity}: no results"

    lines = [f"GDKD | {entity} ({len(hits)} records)"]
    for _, row in hits.iterrows():
        core = (
            f"  {row['Disease']} | {row['Gene']} {row['Variant']} "
            f"({row['Description']}) [{row['Effect']}]"
        )
        assocs = _unpack_associations(row)
        if assocs:
            parts = []
            for a in assocs:
                ctx = a["therapeutic_context"] or "?"
                st = a["status"] or "?"
                ev = a["evidence"] or "?"
                parts.append(f"{a['association']}→{ctx}({st},{ev})")
            core += " :: " + "; ".join(parts)
        lines.append(core)
    return "\n".join(lines)


# ── JSON export ──────────────────────────────────────────────────────
def to_json(hits: pd.DataFrame) -> list[dict]:
    """Structured output for pipeline consumption."""
    records = []
    for _, row in hits.iterrows():
        rec = {c: row.get(c) for c in CORE_COLS}
        rec["associations"] = _unpack_associations(row)
        records.append(rec)
    return records


# ── main demo ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    df = load_gdkd()
    print(f"Loaded {len(df)} GDKD records, {df['Gene'].nunique()} genes, "
          f"{df['Disease'].nunique()} diseases\n")

    # 1. Search by gene symbol
    hits = search(df, "BRAF")
    print(summarize(hits, "BRAF"))
    print()

    # 2. Search by variant
    hits = search(df, "V600E")
    print(summarize(hits, "V600E"))
    print()

    # 3. Search by disease (free text)
    hits = search(df, "melanoma")
    print(summarize(hits, "melanoma"))
    print()

    # 4. Search by drug name in therapeutic context
    hits = search(df, "nilotinib")
    print(summarize(hits, "nilotinib"))
    print()

    # 5. Batch search
    results = search_batch(df, ["EGFR", "T790M", "lung"])
    for ent, h in results.items():
        print(summarize(h, ent))
        print()

    # 6. JSON output
    hits = search(df, "ABL1")
    print(json.dumps(to_json(hits)[:2], indent=2))