#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/data/boom/Agent/DrugClaw")
sys.path.insert(0, str(ROOT))


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
    return {
        "source_entity": d.get("source_entity"),
        "target_entity": d.get("target_entity"),
        "relationship": d.get("relationship"),
        "evidence_text": d.get("evidence_text"),
        "metadata": getattr(item, "metadata", d.get("metadata", {})),
        "source": d.get("source"),
    }


def validate(results: list[Any], label: str) -> list[str]:
    problems: list[str] = []
    if not results:
        return [f"{label}: result list is empty"]
    for i, item in enumerate(results):
        if not getattr(item, "evidence_text", None):
            problems.append(f"{label}: result[{i}] missing evidence_text")
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict) or not metadata:
            problems.append(f"{label}: result[{i}] metadata empty or not dict")
        elif "pharmgkb_id" not in metadata and "pharmgkb_gene_id" not in metadata:
            problems.append(f"{label}: result[{i}] metadata missing PharmGKB business id")
    return problems


def main() -> int:
    from skills.pharmacogenomics.pharmgkb import PharmGKBSkill

    checks = [
        "import PharmGKBSkill",
        "instantiate PharmGKBSkill(timeout=20)",
        "call is_available()",
        "standard query: drug=clopidogrel and gene=CYP2C19",
        "edge query: drug=zzz_not_a_real_drug_zzz",
        "validate evidence_text and metadata",
    ]
    sample: dict[str, Any] = {}
    problems: list[str] = []
    fixes: list[str] = []

    skill = PharmGKBSkill(config={"timeout": 20})
    sample["is_available"] = skill.is_available()
    standard = skill.retrieve({"drug": ["clopidogrel"], "gene": ["CYP2C19"]}, query="pharmacogenomics", max_results=5)
    edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query="pharmacogenomics", max_results=5)
    sample["result_count"] = len(standard)
    sample["edge_count"] = len(edge)
    sample["sample_results"] = [rr_view(r) for r in standard[:2]]

    if not sample["is_available"]:
        problems.append("is_available() returned False for an implemented PharmGKB skill.")
    problems.extend(validate(standard, "standard_query"))
    if edge:
        problems.append("edge_query: expected empty list for non-existent drug")

    status = "PASS" if not problems else "FAIL"
    if problems:
        fixes.append("Keep PharmGKBSkill aligned with the example script and current offline/live fallback contract.")

    print(json.dumps(asdict(Report(
        status=status,
        commands_executed=["python maintainers/smoke/test_skill_49_pharmgkb.py"],
        checks_performed=checks,
        sample_returned_records=sample,
        problems_found=problems,
        minimal_fixes_suggested=fixes,
    )), indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
