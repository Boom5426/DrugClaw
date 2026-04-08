#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/data/boom/Agent/DrugClaw")
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "resources_metadata" / "drug_nlp" / "DrugProt" / "drugprot.tsv"


@dataclass
class Report:
    status: str
    commands_executed: list[str]
    checks_performed: list[str]
    sample_returned_records: dict[str, Any]
    problems_found: list[str]
    minimal_fixes_suggested: list[str]


def rr_view(item: Any) -> dict[str, Any]:
    d = item.to_dict() if hasattr(item, "to_dict") else asdict(item)
    return {"source_entity": d.get("source_entity"), "target_entity": d.get("target_entity"), "relationship": d.get("relationship"), "evidence_text": d.get("evidence_text"), "metadata": getattr(item, "metadata", d.get("metadata", {})), "source": d.get("source")}


def validate(results: list[Any], label: str) -> list[str]:
    problems = []
    if not results:
        return [f"{label}: result list is empty"]
    for i, item in enumerate(results):
        meta = getattr(item, "metadata", None)
        if not getattr(item, "evidence_text", None):
            problems.append(f"{label}: result[{i}] missing evidence_text")
        if not isinstance(meta, dict) or "relation_type" not in meta:
            problems.append(f"{label}: result[{i}] metadata missing relation_type")
    return problems


def main() -> int:
    from skills.drug_nlp.drugprot import DrugProtSkill
    checks = ["import DrugProtSkill", f"instantiate DrugProtSkill(tsv_path={FIXTURE})", "call is_available()", "standard query: drug=imatinib", "edge query: drug=zzz_not_a_real_drug_zzz", "validate evidence_text and metadata"]
    skill = DrugProtSkill(config={"tsv_path": str(FIXTURE)})
    sample = {"is_available": skill.is_available()}
    standard = skill.retrieve({"drug": ["imatinib"]}, query="drug-protein relation", max_results=5)
    edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query="drug-protein relation", max_results=5)
    sample["result_count"] = len(standard); sample["edge_count"] = len(edge); sample["sample_results"] = [rr_view(r) for r in standard[:2]]
    problems = ([] if sample["is_available"] else ["is_available() returned False for a fixture-backed DrugProt skill."]) + validate(standard, "standard_query")
    if edge: problems.append("edge_query: expected empty list for non-existent drug")
    status = "PASS" if not problems else "FAIL"
    fixes = [] if not problems else ["Keep DrugProtSkill aligned with the local-file contract used by maintainers/smoke/test_skill_31_drugprot.py."]
    print(json.dumps(asdict(Report(status, ["python maintainers/smoke/test_skill_31_drugprot.py"], checks, sample, problems, fixes)), indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
