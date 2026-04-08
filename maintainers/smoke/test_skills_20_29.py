#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/data/boom/Agent/DrugClaw")
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)


@dataclass
class SkillCase:
    number: str
    label: str
    module: str
    class_name: str
    implemented: bool
    config: dict[str, Any]
    entities: dict[str, list[str]]
    query: str


@dataclass
class SkillReport:
    name: str
    status: str
    commands_executed: list[str]
    checks_performed: list[str]
    sample_outputs: dict[str, Any]
    problems_found: list[str]
    minimal_fixes_suggested: list[str]


CASES = [
    SkillCase("20", "DDInter", "skills.ddi.ddinter", "DDInterSkill", True, {"timeout": 20}, {"drug": ["aspirin"]}, "interaction"),
    SkillCase("21", "DrugCombDB", "skills.drug_combination.drugcombdb", "DrugCombDBSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_combination" / "DrugCombDB" / "drugcombdb.csv")}, {"drug": ["imatinib"]}, "combination"),
    SkillCase("22", "CDCDB", "skills.drug_combination.cdcdb", "CDCDBSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_combination" / "CDCDB" / "cdcdb.csv")}, {"drug": ["imatinib"]}, "combination"),
    SkillCase("23", "VigiAccess", "skills.adr.vigiaccess", "VigiAccessSkill", False, {"timeout": 20}, {"drug": ["aspirin"]}, "adverse reactions"),
    SkillCase("24", "OREGANO", "skills.drug_repurposing.oregano", "OREGANOSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_repurposing" / "OREGANO" / "oregano.csv")}, {"drug": ["imatinib"]}, "repurposing"),
    SkillCase("25", "DRKG", "skills.drug_repurposing.drkg", "DRKGSkill", True, {"drkg_tsv": str(ROOT / "resources_metadata" / "drug_repurposing" / "DRKG" / "drkg.tsv")}, {"drug": ["imatinib"]}, "knowledge graph"),
    SkillCase("26", "BindingDB", "skills.dti.bindingdb", "BindingDBSkill", True, {"timeout": 20}, {"drug": ["imatinib"]}, "binding"),
    SkillCase("27", "ATC/DDD", "skills.drug_ontology.atc", "ATCSkill", False, {}, {"drug": ["aspirin"]}, "classification"),
    SkillCase("28", "UniTox", "skills.drug_toxicity.unitox", "UniToxSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_toxicity" / "UniTox" / "unitox.csv")}, {"drug": ["acetaminophen"]}, "toxicity"),
    SkillCase("29", "Drug Repurposing Hub", "skills.drug_repurposing.repurposing_hub", "RepurposingHubSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_repurposing" / "Repurposing_Hub" / "repurposing_hub.csv")}, {"drug": ["imatinib"]}, "repurposing"),
]


def rr_view(item: Any) -> dict[str, Any]:
    d = item.to_dict() if hasattr(item, "to_dict") else item.__dict__
    return {
        "source_entity": d.get("source_entity"),
        "target_entity": d.get("target_entity"),
        "relationship": d.get("relationship"),
        "evidence_text": d.get("evidence_text"),
        "metadata": getattr(item, "metadata", d.get("metadata", {})),
        "source": d.get("source"),
    }


def validate_results(results: list[Any], label: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(results, list):
        return [f"{label}: result is not a list"]
    if not results:
        return [f"{label}: result list is empty"]
    for i, item in enumerate(results):
        if not getattr(item, "evidence_text", None):
            problems.append(f"{label}: result[{i}] missing evidence_text")
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict) or not metadata:
            problems.append(f"{label}: result[{i}] metadata empty or not dict")
    return problems


def run_case(case: SkillCase) -> SkillReport:
    commands = ["python maintainers/smoke/test_skills_20_29.py"]
    checks = [
        f"import {case.class_name}",
        f"instantiate {case.class_name}",
        "call is_available()",
    ]
    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    mod = importlib.import_module(case.module)
    skill_cls = getattr(mod, case.class_name)
    skill = skill_cls(config=case.config)
    available = skill.is_available()
    sample["is_available"] = available
    sample["implemented_contract"] = case.implemented

    if not isinstance(available, bool):
        problems.append("is_available() did not return bool")

    if case.implemented:
        checks += [
            f"retrieve standard query {case.entities}",
            "validate evidence_text and metadata",
        ]
        results = skill.retrieve(case.entities, query=case.query, max_results=5)
        edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query=case.query, max_results=5)
        sample["result_count"] = len(results)
        sample["edge_count"] = len(edge)
        sample["sample_results"] = [rr_view(r) for r in results[:2]]
        problems.extend(validate_results(results, "standard_query"))
        if edge:
            problems.append("edge_query: expected empty list for non-existent entity")
    else:
        checks.append("validate stub skill imports and exposes availability without crashing")
        sample["note"] = "Stub skill validation only; no retrieve() assertion yet."

    status = "PASS" if not problems else "FAIL"
    if problems:
        fixes.append("Align the runtime skill behavior with the grouped smoke-test contract in maintainers/smoke/test_skills_20_29.py.")
    return SkillReport(f"{case.number} {case.label}", status, commands, checks, sample, problems, fixes)


def main() -> int:
    reports = [run_case(case) for case in CASES]
    overall = "PASS" if all(r.status == "PASS" for r in reports) else "FAIL"
    print(json.dumps({"status": overall, "reports": [asdict(r) for r in reports]}, indent=2, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
