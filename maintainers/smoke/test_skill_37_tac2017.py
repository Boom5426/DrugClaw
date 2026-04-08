#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/data/boom/Agent/DrugClaw")
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "resources_metadata" / "drug_nlp" / "TAC_2017_ADR" / "tac2017.tsv"


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
        if not isinstance(meta, dict) or "section" not in meta:
            problems.append(f"{label}: result[{i}] metadata missing section")
    return problems


def main() -> int:
    from skills.drug_nlp.tac2017 import TAC2017ADRSkill
    checks = ["import TAC2017ADRSkill", f"instantiate TAC2017ADRSkill(tsv_path={FIXTURE})", "call is_available()", "standard query: drug=aspirin", "edge query: drug=zzz_not_a_real_drug_zzz", "validate evidence_text and metadata"]
    skill = TAC2017ADRSkill(config={"tsv_path": str(FIXTURE)})
    sample = {"is_available": skill.is_available()}
    standard = skill.retrieve({"drug": ["aspirin"]}, query="adr", max_results=5)
    edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query="adr", max_results=5)
    sample["result_count"] = len(standard); sample["edge_count"] = len(edge); sample["sample_results"] = [rr_view(r) for r in standard[:2]]
    problems = ([] if sample["is_available"] else ["is_available() returned False for a fixture-backed TAC2017 ADR skill."]) + validate(standard, "standard_query")
    if edge: problems.append("edge_query: expected empty list for non-existent drug")
    status = "PASS" if not problems else "FAIL"
    fixes = [] if not problems else ["Keep TAC2017ADRSkill aligned with the local-file contract used by maintainers/smoke/test_skill_37_tac2017.py."]
    print(json.dumps(asdict(Report(status, ["python maintainers/smoke/test_skill_37_tac2017.py"], checks, sample, problems, fixes)), indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
