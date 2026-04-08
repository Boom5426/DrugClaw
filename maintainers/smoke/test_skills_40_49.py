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
    SkillCase("40", "Drug Reviews Drugs.com", "skills.drug_review.drugs_com_reviews", "DrugsComReviewsSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_review" / "Drugs_com_Reviews" / "drugs_com_reviews.tsv")}, {"drug": ["aspirin"]}, "reviews"),
    SkillCase("41", "TarKG", "skills.dti.tarkg", "TarKGSkill", True, {"tsv_path": str(ROOT / "resources_metadata" / "dti" / "TarKG" / "tarkg.tsv")}, {"drug": ["imatinib"]}, "target"),
    SkillCase("42", "DrugRepoBank", "skills.drug_repurposing.drugrepobank", "DrugRepoBankSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_repurposing" / "DrugRepoBank" / "drugrepobank.csv")}, {"drug": ["imatinib"]}, "repurposing"),
    SkillCase("43", "RepurposeDrugs", "skills.drug_repurposing.repurposedrugs", "RepurposeDrugsSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_repurposing" / "RepurposeDrugs" / "repurposedrugs.csv")}, {"drug": ["imatinib"]}, "repurposing"),
    SkillCase("44", "DrugRepurposing Online", "skills.drug_repurposing.drugrepurposing_online", "DrugRepurposingOnlineSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_repurposing" / "DrugRepurposing_Online" / "drugrepurposing_online.csv")}, {"drug": ["imatinib"]}, "repurposing"),
    SkillCase("45", "STITCH", "skills.dti.stitch", "STITCHSkill", True, {"timeout": 20}, {"drug": ["imatinib"]}, "target"),
    SkillCase("46", "PROMISCUOUS", "skills.dti.promiscuous", "PROMISCUOUSSkill", False, {}, {"drug": ["imatinib"]}, "target"),
    SkillCase("47", "LiverTox", "skills.drug_toxicity.livertox", "LiverToxSkill", True, {"fixture_path": str(ROOT / "resources_metadata" / "drug_toxicity" / "LiverTox" / "livertox.json"), "timeout": 20}, {"drug": ["acetaminophen"]}, "toxicity"),
    SkillCase("48", "DILIrank", "skills.drug_toxicity.dilirank", "DILIrankSkill", True, {"csv_path": str(ROOT / "resources_metadata" / "drug_toxicity" / "DILIrank" / "dilirank.csv")}, {"drug": ["acetaminophen"]}, "toxicity"),
    SkillCase("49", "PharmGKB ClinPGx", "skills.pharmacogenomics.pharmgkb", "PharmGKBSkill", True, {"timeout": 20}, {"drug": ["clopidogrel"], "gene": ["CYP2C19"]}, "pharmacogenomics"),
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
    commands = ["python maintainers/smoke/test_skills_40_49.py"]
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
        fixes.append("Align the runtime skill behavior with the grouped smoke-test contract in maintainers/smoke/test_skills_40_49.py.")
    return SkillReport(f"{case.number} {case.label}", status, commands, checks, sample, problems, fixes)


def main() -> int:
    reports = [run_case(case) for case in CASES]
    overall = "PASS" if all(r.status == "PASS" for r in reports) else "FAIL"
    print(json.dumps({"status": overall, "reports": [asdict(r) for r in reports]}, indent=2, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
