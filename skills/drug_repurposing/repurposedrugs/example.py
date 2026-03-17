"""
Skill 16 – RepurposeDrugs (single-agent monotherapy dataset)
Query drug-disease repurposing associations from the RepurposeDrugs database.

Data source
-----------
Ianevski A et al. "RepurposeDrugs: an interactive web-portal and predictive
platform for repurposing mono- and combination therapies."
Briefings in Bioinformatics, 2024.  https://repurposedrugs.org/

Local file: dataset_single.xlsx
Columns: Drug_name | Disease_name | Phase | Merged_RefNew
"""

from __future__ import annotations

import re
import json
from typing import Optional

import pandas as pd

# ── data path (HPC absolute) ────────────────────────────────────────────────
DATA_PATH = (
    "/blue/qsong1/wang.qing/AgentLLM/DrugClaw/"
    "resources_metadata/drug_repurposing/RepurposeDrugs/dataset_single.xlsx"
)

# ── helpers ──────────────────────────────────────────────────────────────────

_NCT_RE = re.compile(r"NCT\d{5,}", re.IGNORECASE)


def _extract_ncts(url: Optional[str]) -> list[str]:
    """Pull all NCT IDs from a Merged_RefNew URL string."""
    if not isinstance(url, str):
        return []
    return _NCT_RE.findall(url)


# ── core API ─────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset_single.xlsx into a DataFrame.

    Adds a derived ``NCT_IDs`` column (list of NCT identifiers parsed from
    the Merged_RefNew URL).
    """
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["NCT_IDs"] = df["Merged_RefNew"].apply(_extract_ncts)
    return df


def search(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Search for a single entity.  Auto-detects query type:

    * ``NCT\\d+``       → match inside NCT_IDs list
    * ``\\d`` only      → treated as phase number (1-4)
    * anything else     → case-insensitive substring on Drug_name OR Disease_name
    """
    e = entity.strip()
    if not e:
        return df.iloc[0:0]

    # NCT ID
    if re.match(r"^NCT\d+$", e, re.IGNORECASE):
        mask = df["NCT_IDs"].apply(lambda ids: e.upper() in [i.upper() for i in ids])
        return df.loc[mask].copy()

    # Phase number
    if re.match(r"^[1-4]$", e):
        return df.loc[df["Phase"].astype(str) == e].copy()

    # Free-text (drug or disease name)
    pat = re.escape(e).replace(r"\ ", ".*")
    mask_drug = df["Drug_name"].str.contains(pat, case=False, na=False)
    mask_dis = df["Disease_name"].str.contains(pat, case=False, na=False)
    return df.loc[mask_drug | mask_dis].copy()


def search_batch(df: pd.DataFrame, entities: list[str]) -> dict[str, pd.DataFrame]:
    """Search for multiple entities. Returns ``{entity: DataFrame}``."""
    return {e: search(df, e) for e in entities}


def summarize(hits: pd.DataFrame, entity: str = "") -> str:
    """Return an LLM-readable compact summary string.

    Format per row:  ``DRUG → DISEASE (Phase N) [NCTxxxxxxxx, …]``
    """
    if hits.empty:
        return f"No results for '{entity}'." if entity else "No results."

    lines: list[str] = []
    if entity:
        lines.append(f"== {entity}: {len(hits)} association(s) ==")
    for _, r in hits.iterrows():
        ncts = ", ".join(r["NCT_IDs"]) if r["NCT_IDs"] else "no NCT"
        lines.append(f"{r['Drug_name']} → {r['Disease_name']} (Phase {r['Phase']}) [{ncts}]")
    return "\n".join(lines)


def to_json(hits: pd.DataFrame) -> list[dict]:
    """Convert hit rows to a list of plain dicts (JSON-serialisable)."""
    out = []
    for _, r in hits.iterrows():
        out.append({
            "drug_name": r["Drug_name"],
            "disease_name": r["Disease_name"],
            "phase": int(r["Phase"]) if pd.notna(r["Phase"]) else None,
            "nct_ids": r["NCT_IDs"],
            "ref_url": r.get("Merged_RefNew", ""),
        })
    return out


# ── runnable examples ────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['Drug_name'].nunique()} unique drugs, "
          f"{df['Disease_name'].nunique()} unique diseases.\n")

    # 1. Single drug name
    hits = search(df, "aspirin")
    print(summarize(hits, "aspirin"))
    print()

    # 2. Single disease name
    hits = search(df, "Pulmonary Hypertension")
    print(summarize(hits, "Pulmonary Hypertension"))
    print()

    # 3. NCT ID lookup
    hits = search(df, "NCT01856868")
    print(summarize(hits, "NCT01856868"))
    print()

    # 4. Batch search
    results = search_batch(df, ["metformin", "semaglutide", "Friedreich Ataxia"])
    for ent, h in results.items():
        print(summarize(h, ent))
        print()

    # 5. JSON output
    hits = search(df, "(-)-Epicatechin")
    print(json.dumps(to_json(hits), indent=2))