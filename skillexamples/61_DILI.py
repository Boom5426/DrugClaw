"""
DILI - Drug-Induced Liver Injury (DILIrank 2.0 & DILIst)
Category: Drug-centric | Type: Local Dataset | Subcategory: Drug Toxicity
Source: FDA Liver Toxicity Knowledge Base (LTKB)
Link: https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset

Queries locally downloaded DILIrank / DILIst datasets.
  - DILIrank: FDA severity classification (Most / Less / No-DILI-Concern / Ambiguous)
  - DILIst:  Extended compound-level DILI annotations

Setup:
  pip install pandas openpyxl
  Set DATA_DIR to the folder containing the downloaded .xlsx / .csv files.
"""

import os, sys, json, glob
import pandas as pd
from typing import Union

# ── configure path ──────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "DILI_DATA_DIR",
    "/blue/qsong1/wang.qing/AgentLLM/Survey100/resources_metadata/drug_toxicity/DILI",
)


# ── helpers ─────────────────────────────────────────────────────────
def _load_all_tables(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load every .xlsx / .csv in *data_dir* into {filename: DataFrame}."""
    tables: dict[str, pd.DataFrame] = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "*"))):
        ext = os.path.splitext(fp)[1].lower()
        name = os.path.basename(fp)
        try:
            if ext == ".xlsx":
                xls = pd.ExcelFile(fp)
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    key = f"{name}::{sheet}" if len(xls.sheet_names) > 1 else name
                    tables[key] = df
            elif ext == ".csv":
                tables[name] = pd.read_csv(fp)
            elif ext == ".tsv":
                tables[name] = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"[WARN] skip {name}: {e}", file=sys.stderr)
    return tables


def _guess_name_cols(df: pd.DataFrame) -> list[str]:
    """Heuristically pick columns that likely hold drug / compound names."""
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(kw in cl for kw in ("compound", "drug", "name", "molecule", "ingredient")):
            candidates.append(c)
    return candidates or [df.columns[0]]  # fallback: first column


def _search_df(df: pd.DataFrame, queries: list[str], name_cols: list[str]) -> pd.DataFrame:
    """Case-insensitive substring match across *name_cols*."""
    q_lower = [q.lower().strip() for q in queries]
    mask = pd.Series(False, index=df.index)
    for col in name_cols:
        col_lower = df[col].astype(str).str.lower()
        for q in q_lower:
            mask |= col_lower.str.contains(q, na=False)
    return df.loc[mask]


# ── public API ──────────────────────────────────────────────────────
def query_dili(
    entities: Union[str, list[str]],
    data_dir: str = DATA_DIR,
) -> list[dict]:
    """
    Query DILIrank / DILIst for one or more drug entities.

    Parameters
    ----------
    entities : str or list[str]
        Drug name(s) to search (case-insensitive substring match).
    data_dir : str
        Path to the folder with DILIrank / DILIst files.

    Returns
    -------
    list[dict]
        Each dict: {"source": filename, "matches": [row_dict, ...]}.
    """
    if isinstance(entities, str):
        entities = [entities]

    tables = _load_all_tables(data_dir)
    if not tables:
        return [{"error": f"No data files found in {data_dir}"}]

    results = []
    for tname, df in tables.items():
        name_cols = _guess_name_cols(df)
        hits = _search_df(df, entities, name_cols)
        if not hits.empty:
            results.append({
                "source": tname,
                "match_count": len(hits),
                "matches": hits.head(50).to_dict(orient="records"),
            })
    return results


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: query three well-known hepatotoxic drugs
    entities = ["acetaminophen", "isoniazid", "troglitazone"]

    print(f"Querying DILI data for: {entities}")
    print(f"Data dir: {DATA_DIR}\n")

    results = query_dili(entities)
    if not results:
        print("No matches found.")
    for blk in results:
        if "error" in blk:
            print(f"[ERROR] {blk['error']}")
            continue
        print(f"── {blk['source']}  ({blk['match_count']} hits) ──")
        for row in blk["matches"][:5]:
            print(json.dumps(row, ensure_ascii=False, default=str))
        if blk["match_count"] > 5:
            print(f"  ... and {blk['match_count'] - 5} more rows")
        print()