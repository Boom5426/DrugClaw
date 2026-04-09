"""
entity_audit.py  —  DrugClaw evidence-layer health-check
=========================================================

Usage
-----
# Single query (layers 1-4 + payload dump)
python entity_audit.py single "What are the drug targets of imatinib?" \
    --skills ChEMBL DGIdb "Open Targets Platform"

# Six-probe matrix (DTI / ADR / DDI / Repo / PGx / LABEL)
python entity_audit.py matrix

# Both
python entity_audit.py all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

PROBES: List[Tuple[str, str, List[str]]] = [
    ("DTI",   "What are the known drug targets of imatinib?",
               ["ChEMBL", "DGIdb", "Open Targets Platform"]),
    ("ADR",   "What are the adverse effects of atorvastatin?",
               ["SIDER", "FAERS"]),
    ("DDI",   "Does warfarin interact with aspirin?",
               ["DDInter", "MecDDI"]),
    ("REPO",  "Which approved drugs may be repurposed for triple-negative breast cancer?",
               ["RepoDB", "DRKG"]),
    ("PGx",   "What pharmacogenomic guidance exists for clopidogrel and CYP2C19?",
               ["PharmGKB"]),
    ("LABEL", "What prescribing and safety information is available for metformin?",
               ["DailyMed", "openFDA Human Drug"]),
]

# Skills where empty source_entity / target_entity is expected by design
DOCUMENT_SKILLS = {
    "DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info", "RxList",
    "AskAPatient", "Drugs.com Reviews", "WebMD",
    "LiverTox", "DILI", "DILIRank", "UniTox",
    "ADE Corpus", "CADEC", "DDI Corpus", "DrugEHRQA", "DrugProt",
    "N2C2 2018", "PHEE", "PsyTAR", "TAC 2017",
    "Web Search",
}

# Per-skill native payload keys that *must* be present for the record to be
# considered properly structured. Only relational / KG skills are listed here.
SKILL_EXPECTED_KEYS: Dict[str, List[str]] = {
    # DTI
    "ChEMBL":                  ["chembl_id", "target_chembl_id"],
    "DGIdb":                   ["gene_name"],
    "BindingDB":               ["ligand_name", "target_name"],
    "Open Targets Platform":   ["targetId"],
    "DTC":                     ["target_id"],
    "TTD":                     ["TargetID"],
    "STITCH":                  ["chemical", "protein"],
    # ADR
    "SIDER":                   ["side_effect_name"],
    "FAERS":                   ["reaction"],
    "ADRECS":                  ["adr_id"],
    "NSIDEs":                  ["concept_name"],
    # DDI
    "DDInter":                 ["drug1_name", "drug2_name"],
    "MecDDI":                  ["mechanism"],
    "KEGG Drug":               ["drug1", "drug2"],
    # Drug Repurposing
    "RepoDB":                  ["drug_name", "ind_name", "status"],
    "DRKG":                    ["relation"],
    "CancerDR":                ["drug_name"],
    "Repurposing Hub":         ["name"],
    # PGx
    "PharmGKB":                ["pharmgkb_id"],
    "CPIC":                    ["drug", "gene"],
    # Drug Knowledge
    "DrugBank":                ["drugbank_id"],
    "DrugCentral":             ["struct_id"],
    "ChEBI":                   ["chebi_id"],
    "RxNorm":                  ["rxcui"],
    # Drug-Disease
    "SemaTyp":                 ["disease_name"],
    # Combination
    "DrugComb":                ["drug_row", "drug_col"],
    "CDCDB":                   ["drug_a", "drug_b"],
    # Mechanism
    "DrugMechDB":              ["drug", "mechanism_text"],
}


# ---------------------------------------------------------------------------
# Helper – load system lazily so import errors surface as clear messages
# ---------------------------------------------------------------------------

def _load_system():
    try:
        from drugclaw.config import Config
        from drugclaw.main_system import DrugClawSystem
        from drugclaw.models import ThinkingMode
    except ImportError as exc:
        sys.exit(f"[entity_audit] cannot import DrugClaw: {exc}")

    cfg_path = Path("api_keys.json")
    if not cfg_path.exists():
        sys.exit("[entity_audit] api_keys.json not found in current directory")

    cfg = Config(key_file=str(cfg_path))
    system = DrugClawSystem(cfg)
    return system, ThinkingMode


# ---------------------------------------------------------------------------
# Evidence parsing helpers
# ---------------------------------------------------------------------------

def _items_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return evidence_items as a list of plain dicts."""
    raw = result.get("evidence_items", [])
    out = []
    for it in raw:
        if hasattr(it, "to_dict"):
            out.append(it.to_dict())
        elif isinstance(it, dict):
            out.append(it)
    return out


def _payload_has_expected_keys(payload: Dict[str, Any], skill: str) -> bool:
    expected = SKILL_EXPECTED_KEYS.get(skill)
    if not expected:
        return True  # no expectation → can't fail
    return any(k in payload for k in expected)


def _entity_fill(item: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return (has_source_entity, has_target_entity) from metadata or top-level."""
    meta = item.get("metadata") or {}
    src = meta.get("source_entity") or item.get("source_entity") or ""
    tgt = meta.get("target_entity") or item.get("target_entity") or ""
    return bool(src), bool(tgt)


def _diag_map(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """skill_name → diagnostics dict"""
    out = {}
    for d in result.get("retrieval_diagnostics", []):
        if isinstance(d, dict) and "skill" in d:
            out[d["skill"]] = d
    return out


# ---------------------------------------------------------------------------
# Report: single query
# ---------------------------------------------------------------------------

def run_single(
    system,
    ThinkingMode,
    query: str,
    skills: Optional[List[str]],
    out_dir: Path,
) -> Dict[str, Any]:
    print("=" * 72)
    print(f"[SINGLE AUDIT] {query}")
    t0 = time.time()
    result = system.query(
        query,
        thinking_mode=ThinkingMode.SIMPLE,
        resource_filter=skills or [],
    )
    elapsed = time.time() - t0

    items = _items_from_result(result)
    diag = _diag_map(result)

    print(f"query: {query}")
    print(f"mode:  simple   items: {len(items)}   elapsed: {elapsed:.1f}s")
    print("=" * 72)

    # ── Layer-by-layer entity trace ─────────────────────────────────────────
    print("-- entity by stage --")
    ir = result.get("input_resolution") or {}
    print(f"  [1] detected_identifiers : {ir.get('detected_identifiers', [])}")
    re_ = result.get("resolved_entities") or {}
    print(f"  [2] resolved_entities    : {re_}")
    qp = result.get("query_plan") or {}
    print(f"  [3] query_plan.entities  : {qp.get('entities', {})}")

    # ── Per-skill breakdown ─────────────────────────────────────────────────
    by_skill: Dict[str, List[Dict]] = defaultdict(list)
    for it in items:
        by_skill[it.get("source_skill") or it.get("source") or "?"].append(it)

    requested = set(skills or [])
    present = set(by_skill.keys())
    absent = requested - present

    print("\n-- fill rate by skill --")
    print(f"  {'skill':<28} {'cat':<22} {'ent_fill':>8} {'payload':>8} {'native_keys':>11} {'diag_records':>12}")

    all_ok = True
    for skill, sitems in sorted(by_skill.items()):
        cat = (sitems[0].get("metadata") or {}).get("skill_category", "?")
        n = len(sitems)
        src_filled = sum(1 for it in sitems if _entity_fill(it)[0])
        tgt_filled = sum(1 for it in sitems if _entity_fill(it)[1])
        pay_filled = sum(1 for it in sitems if it.get("structured_payload"))
        native_ok  = sum(
            1 for it in sitems
            if _payload_has_expected_keys(it.get("structured_payload") or {}, skill)
        )
        d = diag.get(skill, {})
        diag_rec = d.get("records", "?")
        doc = skill in DOCUMENT_SKILLS

        ent_label = f"{src_filled}/{n}src {tgt_filled}/{n}tgt"
        if doc:
            verdict = "ok (doc)"
        elif src_filled == 0 and tgt_filled == 0:
            verdict = "WARN-entity"
            all_ok = False
        else:
            verdict = "ok"

        nat_label = f"{native_ok}/{n}"
        if not doc and native_ok == 0 and skill in SKILL_EXPECTED_KEYS:
            nat_label += " WARN"
            all_ok = False

        print(f"  {skill:<28} {cat:<22} {ent_label:>8} {pay_filled:>3}/{n:<3}    {nat_label:>8}   {str(diag_rec):>8}")

    if absent:
        print("\n-- ABSENT skills (requested but 0 evidence items returned) --")
        for sk in sorted(absent):
            d = diag.get(sk, {})
            error = d.get("error") or "(not in diagnostics)"
            strategy = d.get("strategy", "?")
            diag_rec = d.get("records", "?")
            print(f"  {sk:<30}  strategy={strategy}  records={diag_rec}  error={error[:120]}")

    # ── Sample payload ──────────────────────────────────────────────────────
    if items:
        first = items[0]
        pay = first.get("structured_payload") or {}
        skill_name = first.get("source_skill") or first.get("source") or "?"
        print(f"\n-- sample structured_payload ({skill_name}, item 0) --")
        sample_str = json.dumps(pay, ensure_ascii=False, default=str)
        print("  " + sample_str[:600])
        if len(sample_str) > 600:
            print("  ... (truncated)")

    # ── Save full JSON ──────────────────────────────────────────────────────
    safe_q = "".join(c if c.isalnum() else "_" for c in query[:40])
    out_path = out_dir / f"single_{safe_q}.json"
    _save_json(out_path, result)

    return {"all_ok": all_ok, "items": len(items), "absent": list(absent)}


# ---------------------------------------------------------------------------
# Report: matrix scan
# ---------------------------------------------------------------------------

def run_matrix(system, ThinkingMode, out_dir: Path) -> None:
    print("=" * 72)
    print("[MATRIX SCAN] 6 probes")
    print("=" * 72)

    rows = []
    for qtype, query, skills in PROBES:
        print(f"\n--- [{qtype}] {query}")
        print(f"    skills: {skills}")
        t0 = time.time()
        result = system.query(
            query,
            thinking_mode=ThinkingMode.SIMPLE,
            resource_filter=skills,
        )
        elapsed = time.time() - t0

        items = _items_from_result(result)
        diag = _diag_map(result)
        requested = set(skills)
        present_skills = {it.get("source_skill") or it.get("source") for it in items}
        absent = requested - present_skills

        # Print per-skill summary
        by_skill: Dict[str, List[Dict]] = defaultdict(list)
        for it in items:
            by_skill[it.get("source_skill") or it.get("source") or "?"].append(it)

        for skill, sitems in sorted(by_skill.items()):
            n = len(sitems)
            cat = (sitems[0].get("metadata") or {}).get("skill_category", "?")
            src_fill = sum(1 for it in sitems if _entity_fill(it)[0]) / n
            tgt_fill = sum(1 for it in sitems if _entity_fill(it)[1]) / n
            pay_fill = sum(1 for it in sitems if it.get("structured_payload")) / n
            nat_ok = sum(
                1 for it in sitems
                if _payload_has_expected_keys(it.get("structured_payload") or {}, skill)
            ) / n
            d = diag.get(skill, {})
            print(f"    {skill:<28} n={n:>3}  ent={src_fill:.0%}/{tgt_fill:.0%}"
                  f"  payload={pay_fill:.0%}  native={nat_ok:.0%}"
                  f"  [{d.get('strategy','?')}]")
            rows.append({
                "qtype": qtype, "skill": skill, "cat": cat, "n": n,
                "src_fill": src_fill, "tgt_fill": tgt_fill,
                "payload_fill": pay_fill, "native_fill": nat_ok,
                "strategy": d.get("strategy", "?"),
                "error": d.get("error", ""),
            })

        if absent:
            print(f"    ABSENT: {sorted(absent)}")
            for sk in sorted(absent):
                d = diag.get(sk, {})
                rows.append({
                    "qtype": qtype, "skill": sk, "cat": "ABSENT", "n": 0,
                    "src_fill": 0.0, "tgt_fill": 0.0,
                    "payload_fill": 0.0, "native_fill": 0.0,
                    "strategy": d.get("strategy", "?"),
                    "error": (d.get("error") or "not in diagnostics")[:120],
                })

        print(f"    elapsed: {elapsed:.1f}s")
        _save_json(out_dir / f"matrix_{qtype}.json", result)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("[MATRIX SUMMARY]")
    print("=" * 72)
    hdr = f"{'qtype':<6} {'skill':<28} {'cat':<22} {'n':>4}  {'ent src':>7}  {'ent tgt':>7}  {'payload':>7}  {'native':>7}  {'strategy':<18}  error"
    print(hdr)
    for r in rows:
        flag = ""
        doc = r["skill"] in DOCUMENT_SKILLS
        if r["cat"] == "ABSENT":
            flag = " ← ABSENT"
        elif not doc and r["src_fill"] == 0 and r["tgt_fill"] == 0 and r["n"] > 0:
            flag = " ← WARN entity"
        elif not doc and r["native_fill"] == 0 and r["skill"] in SKILL_EXPECTED_KEYS and r["n"] > 0:
            flag = " ← WARN native"
        err = (r["error"] or "")[:40]
        print(
            f"{r['qtype']:<6} {r['skill']:<28} {r['cat']:<22} {r['n']:>4}"
            f"  {r['src_fill']:>6.0%}  {r['tgt_fill']:>6.0%}"
            f"  {r['payload_fill']:>6.0%}  {r['native_fill']:>6.0%}"
            f"  {r['strategy']:<18}  {err}{flag}"
        )

    # ── Write TSV ────────────────────────────────────────────────────────────
    tsv_path = out_dir / "summary_matrix.tsv"
    with tsv_path.open("w") as fh:
        fh.write("qtype\tskill\tcat\tn\tsrc_fill\ttgt_fill\tpayload_fill\tnative_fill\tstrategy\terror\n")
        for r in rows:
            fh.write(
                f"{r['qtype']}\t{r['skill']}\t{r['cat']}\t{r['n']}"
                f"\t{r['src_fill']:.2f}\t{r['tgt_fill']:.2f}"
                f"\t{r['payload_fill']:.2f}\t{r['native_fill']:.2f}"
                f"\t{r['strategy']}\t{r['error']}\n"
            )
    print(f"\n[entity_audit] wrote {tsv_path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_json(path: Path, obj: Any) -> None:
    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dataclass_fields__"):
            import dataclasses
            return dataclasses.asdict(o)
        return str(o)

    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)


def _make_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("entity_audit_out") / ts
    d.mkdir(parents=True, exist_ok=True)
    print(f"[entity_audit] output dir: {d}")
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DrugClaw entity audit tool")
    sub = parser.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Audit a single query")
    p_single.add_argument("query", help="Query string")
    p_single.add_argument("--skills", nargs="*", default=None,
                          help="Resource filter (space-separated skill names)")

    sub.add_parser("matrix", help="Run the 6-probe matrix scan")
    sub.add_parser("all",    help="Run single (imatinib DTI) + matrix")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    out_dir = _make_out_dir()
    system, ThinkingMode = _load_system()

    if args.cmd == "single":
        run_single(system, ThinkingMode, args.query, args.skills, out_dir)

    elif args.cmd == "matrix":
        run_matrix(system, ThinkingMode, out_dir)

    elif args.cmd == "all":
        run_single(
            system, ThinkingMode,
            "What are the known drug targets of imatinib?",
            ["ChEMBL", "DGIdb", "Open Targets Platform"],
            out_dir,
        )
        run_matrix(system, ThinkingMode, out_dir)


if __name__ == "__main__":
    main()
