"""
DrugCombDB — Drug Combination Synergy Data (local CSV)
Category : Drug-centric | Type: DB | Subcategory: Drug Combination/Synergy
Link     : http://drugcombdb.denglab.org/
Paper    : https://doi.org/10.1093/nar/gkz1007  (NAR 2020)

DrugCombDB integrates 448 555 drug combinations (2 887 drugs, 124 cancer
cell lines) with synergy scores computed by four reference models (ZIP,
Bliss, Loewe, HSA).  Data sourced from NCI-ALMANAC, NCATS Matrix, and
literature.

Access method: Local CSV files downloaded from the DrugCombDB website.
No API key required.
"""

import csv
import json
import os
import re
from typing import Union

# ── data path (edit to match your environment) ────────────────────────

DATA_DIR = (
    "/blue/qsong1/wang.qing/AgentLLM/Survey100/"
    "resources_metadata/drug_combination/DrugCombDB/drugcombdb_data"
)

# ── lazy-load cache ───────────────────────────────────────────────────

_cache: dict[str, list[dict]] = {}


def _load_csv(filename: str) -> list[dict]:
    """Load a CSV once into _cache; return list[dict]."""
    if filename in _cache:
        return _cache[filename]
    path = os.path.join(DATA_DIR, filename)
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    _cache[filename] = rows
    return rows


def _load_scored() -> list[dict]:
    """drugcombs_scored.csv → ID, Drug1, Drug2, Cell line, ZIP, Bliss, Loewe, HSA"""
    return _load_csv("drugcombs_scored.csv")


def _load_drugs() -> list[dict]:
    """drug_chemical_info.csv → drugName, cIds, drugNameOfficial, molecularWeight, smilesString"""
    return _load_csv("drug_chemical_info.csv")


def _load_cell_lines() -> list[dict]:
    """cell_Line.csv → cellName, cosmicId, tag"""
    return _load_csv("cell_Line.csv")


def _load_classification(model: str = "zip") -> list[dict]:
    """Syner&Antag_{model}.csv → ID, Drug1, Drug2, Cell line, {MODEL}, classification"""
    fname = f"Syner&Antag_{model}.csv"
    return _load_csv(fname)


def _load_three_drug() -> list[dict]:
    """ThreeDrugCombs.csv → Drug1, Drug2, Drug3, concentrations, viability, ..."""
    return _load_csv("ThreeDrugCombs.csv")


# ── entity-type detection ─────────────────────────────────────────────

_PAT_ID    = re.compile(r"^\d+$")
_PAT_CID   = re.compile(r"^CID[Ss]?\d+$", re.I)
_PAT_CELL  = re.compile(
    r"^(MCF|MDA|A549|A375|A2058|HCT|HT29|PC3|786|NCI|SK|SW|U2OS|BT|LNCAP|PANC|DU145)",
    re.I,
)


def _detect(entity: str) -> str:
    """Heuristic entity-type detection.
    Returns: 'id' | 'cid' | 'cell_line' | 'drug'
    """
    e = entity.strip()
    if _PAT_ID.match(e):
        return "id"
    if _PAT_CID.match(e):
        return "cid"
    if _PAT_CELL.match(e):
        return "cell_line"
    return "drug"


# ── core query functions ──────────────────────────────────────────────

def search_drug(drug_name: str, limit: int = 20) -> list[dict]:
    """Find combinations involving *drug_name* (case-insensitive substring).

    Searches Drug1 and Drug2 columns in drugcombs_scored.csv.
    Returns list of dicts with synergy scores.
    """
    rows = _load_scored()
    q = drug_name.strip().lower()
    hits = [
        r for r in rows
        if q in r.get("Drug1", "").lower() or q in r.get("Drug2", "").lower()
    ]
    return hits[:limit]


def search_cell_line(cell_line: str, limit: int = 20) -> list[dict]:
    """Find combinations tested on *cell_line* (substring match)."""
    rows = _load_scored()
    q = cell_line.strip().lower()
    hits = [r for r in rows if q in r.get("Cell line", "").lower()]
    return hits[:limit]


def search_drug_pair(drug1: str, drug2: str, limit: int = 50) -> list[dict]:
    """Find synergy records for a specific drug pair (order-agnostic)."""
    rows = _load_scored()
    d1, d2 = drug1.strip().lower(), drug2.strip().lower()
    hits = [
        r for r in rows
        if (d1 in r.get("Drug1", "").lower() and d2 in r.get("Drug2", "").lower())
        or (d2 in r.get("Drug1", "").lower() and d1 in r.get("Drug2", "").lower())
    ]
    return hits[:limit]


def get_by_id(record_id: str) -> list[dict]:
    """Get scored record(s) by numeric ID."""
    rows = _load_scored()
    return [r for r in rows if r.get("ID") == record_id]


def get_drug_info(drug_name: str) -> list[dict]:
    """Look up chemical info for a drug (substring match on drugName)."""
    rows = _load_drugs()
    q = drug_name.strip().lower()
    return [r for r in rows if q in r.get("drugName", "").lower()]


def get_drug_info_by_cid(cid: str) -> list[dict]:
    """Look up chemical info by PubChem CID (e.g. 'CIDs00065628')."""
    rows = _load_drugs()
    q = cid.strip().lower()
    return [r for r in rows if q in r.get("cIds", "").lower()]


def list_cell_lines() -> list[dict]:
    """Return all cell lines with COSMIC IDs."""
    return _load_cell_lines()


def get_classification(drug_name: str, model: str = "zip",
                       limit: int = 20) -> list[dict]:
    """Get synergy/antagonism classification for a drug under a given model.

    model: 'zip' | 'bliss' | 'loewe' | 'hsa' | 'voting'
    """
    rows = _load_classification(model)
    q = drug_name.strip().lower()
    hits = [
        r for r in rows
        if q in r.get("Drug1", "").lower() or q in r.get("Drug2", "").lower()
    ]
    return hits[:limit]


def search_three_drug(drug_name: str, limit: int = 20) -> list[dict]:
    """Search three-drug combinations involving *drug_name*."""
    rows = _load_three_drug()
    q = drug_name.strip().lower()
    hits = [
        r for r in rows
        if q in r.get("Drug1 ", "").lower()    # note: trailing space in header
        or q in r.get("Drug2", "").lower()
        or q in r.get("Drug3", "").lower()
    ]
    return hits[:limit]


# ── unified search interface ──────────────────────────────────────────

def search(entity: str, limit: int = 20) -> list[dict]:
    """Auto-detect entity type and route to the right query.

    | Input             | Detected As | Action                        |
    |-------------------|-------------|-------------------------------|
    | `12345`           | record ID   | exact match on ID             |
    | `CIDs00065628`    | PubChem CID | drug_chemical_info lookup     |
    | `A549`, `MCF7`    | cell line   | combinations on that cell line|
    | anything else     | drug name   | substring on Drug1/Drug2      |
    """
    etype = _detect(entity)
    if etype == "id":
        return get_by_id(entity)
    if etype == "cid":
        return get_drug_info_by_cid(entity)
    if etype == "cell_line":
        return search_cell_line(entity, limit=limit)
    return search_drug(entity, limit=limit)


def search_batch(entities: list[str], limit: int = 10) -> dict[str, list[dict]]:
    """Query multiple entities. Returns {entity: [results]}."""
    out: dict[str, list[dict]] = {}
    for e in entities:
        try:
            out[e] = search(e, limit=limit)
        except Exception as exc:
            out[e] = [{"error": str(exc)}]
    return out


# ── compact summariser (LLM-friendly) ────────────────────────────────

def summarize(results: list[dict], entity: str) -> str:
    """One-line-per-hit summary for LLM consumption."""
    if not results:
        return f"No results for '{entity}'."
    lines = [f"=== DrugCombDB results for '{entity}' ({len(results)} hit(s)) ==="]
    for r in results[:15]:
        # Scored combination record
        if "Drug1" in r and "ZIP" in r:
            d1   = r.get("Drug1") or r.get("Drug1 ", "?")
            d2   = r.get("Drug2", "?")
            cell = r.get("Cell line", "?")
            zp   = r.get("ZIP", "")
            bl   = r.get("Bliss", "")
            lw   = r.get("Loewe", "")
            hs   = r.get("HSA", "")
            clf  = r.get("classification", "")
            scores = f"ZIP={zp} Bliss={bl} Loewe={lw} HSA={hs}"
            tag = f" [{clf}]" if clf else ""
            lines.append(f"  {d1} + {d2} | {cell} | {scores}{tag}")
        # Drug chemical info record
        elif "drugName" in r:
            lines.append(
                f"  {r.get('drugName')} (official: {r.get('drugNameOfficial')}) "
                f"MW={r.get('molecularWeight')} CID={r.get('cIds')}"
            )
        # Cell line record
        elif "cellName" in r:
            lines.append(
                f"  {r.get('cellName')} COSMIC={r.get('cosmicId')}"
            )
        # Three-drug combo
        elif "Drug3" in r:
            lines.append(
                f"  {r.get('Drug1 ', '?').strip()} + {r.get('Drug2', '?')} + "
                f"{r.get('Drug3', '?')} | {r.get('Cell Name', '?')} | "
                f"viability={r.get('Mean Relative Viability', '?')}"
            )
        else:
            lines.append(f"  {r}")
    return "\n".join(lines)


# ── JSON output ───────────────────────────────────────────────────────

def to_json(results: list[dict]) -> list[dict]:
    """Passthrough; results are already dicts."""
    return results


# ── runnable examples ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json as _json
    if len(sys.argv) > 1:

        _cli_entities = sys.argv[1:]
        for _e in _cli_entities:
            _hits = search(_e, limit=20)
            print(summarize(_hits, _e))
        sys.exit(0)

    # --- original demo below ---
    print("=" * 70)
    print("1) Search by drug name — 5-FU")
    print("=" * 70)
    hits = search("5-FU", limit=5)
    print(summarize(hits, "5-FU"))

    print()
    print("=" * 70)
    print("2) Search by cell line — A2058")
    print("=" * 70)
    hits2 = search("A2058", limit=5)
    print(summarize(hits2, "A2058"))

    print()
    print("=" * 70)
    print("3) Drug pair — 5-FU + ABT-888")
    print("=" * 70)
    hits3 = search_drug_pair("5-FU", "ABT-888", limit=5)
    print(summarize(hits3, "5-FU + ABT-888"))

    print()
    print("=" * 70)
    print("4) Drug chemical info lookup")
    print("=" * 70)
    info = get_drug_info("Bendamustine")
    print(summarize(info, "Bendamustine"))

    print()
    print("=" * 70)
    print("5) Synergy classification (ZIP model)")
    print("=" * 70)
    clf = get_classification("5-FU", model="zip", limit=5)
    print(summarize(clf, "5-FU (ZIP classification)"))

    print()
    print("=" * 70)
    print("6) Three-drug combinations")
    print("=" * 70)
    tri = search_three_drug("Vemurafenib", limit=3)
    print(summarize(tri, "Vemurafenib (three-drug)"))

    print()
    print("=" * 70)
    print("7) Batch search — multiple drugs")
    print("=" * 70)
    batch = search_batch(["Paclitaxel", "Cisplatin", "Doxorubicin"], limit=3)
    for ent, res in batch.items():
        print(summarize(res, ent))
        print()

    print("=" * 70)
    print("8) JSON output (first 2 hits)")
    print("=" * 70)
    j = to_json(hits[:2])
    print(json.dumps(j, indent=2)[:500])