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
    config: dict[str, Any]


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
    SkillCase("30", "DDI Corpus 2013", "skills.drug_nlp.ddi_corpus", "DDICorpusSkill", {"tsv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "DDI_Corpus_2013" / "ddi_corpus.tsv")}),
    SkillCase("31", "DrugProt", "skills.drug_nlp.drugprot", "DrugProtSkill", {"tsv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "DrugProt" / "drugprot.tsv")}),
    SkillCase("32", "ADE Corpus", "skills.drug_nlp.ade_corpus", "ADECorpusSkill", {"csv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "ADE_Corpus" / "ade_corpus.csv")}),
    SkillCase("33", "n2c2 2018 Track2", "skills.drug_nlp.n2c2_2018", "N2C22018Skill", {"tsv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "n2c2_2018" / "n2c2_2018.tsv")}),
    SkillCase("34", "CADEC", "skills.drug_nlp.cadec", "CADECSkill", {"csv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "CADEC" / "cadec.csv")}),
    SkillCase("35", "AskAPatient", "skills.drug_review.askapatient", "AskAPatientSkill", {"csv_path": str(ROOT / "resources_metadata" / "drug_review" / "AskAPatient" / "askapatient.csv")}),
    SkillCase("36", "PsyTAR", "skills.drug_nlp.psytar", "PsyTARSkill", {"csv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "PsyTAR" / "psytar.csv")}),
    SkillCase("37", "TAC 2017 ADR", "skills.drug_nlp.tac2017", "TAC2017ADRSkill", {"tsv_path": str(ROOT / "resources_metadata" / "drug_nlp" / "TAC_2017_ADR" / "tac2017.tsv")}),
    SkillCase("38", "PHEE", "skills.drug_nlp.phee", "PHEESkill", {"json_path": str(ROOT / "resources_metadata" / "drug_nlp" / "PHEE" / "phee.json")}),
    SkillCase("39", "Drugs.com", "skills.drug_knowledgebase.drugs_com", "DrugsComSkill", {"timeout": 20}),
]


def run_case(case: SkillCase) -> SkillReport:
    commands = ["python maintainers/smoke/test_skills_30_39.py"]
    checks = [
        f"import {case.class_name}",
        f"instantiate {case.class_name}",
        "call is_available()",
    ]
    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    try:
        mod = importlib.import_module(case.module)
        skill_cls = getattr(mod, case.class_name)
        skill = skill_cls(config=case.config)
        available = skill.is_available()
        sample["is_available"] = available
        if not isinstance(available, bool):
            problems.append("is_available() did not return bool")
        if case.number == "39":
            sample["implemented_contract"] = False
            sample["note"] = "Structural smoke test only; retrieve() assertions are not enforced for this unregistered skill."
            checks.append("validate stub skill imports and exposes availability without crashing")
        else:
            checks += [
                "retrieve standard query",
                "validate evidence_text and metadata",
            ]
            entity = {
                "30": {"drug": ["aspirin"]},
                "31": {"drug": ["imatinib"]},
                "32": {"drug": ["aspirin"]},
                "33": {"drug": ["warfarin"]},
                "34": {"drug": ["lipitor"]},
                "35": {"drug": ["metformin"]},
                "36": {"drug": ["sertraline"]},
                "37": {"drug": ["aspirin"]},
                "38": {"drug": ["aspirin"]},
            }[case.number]
            results = skill.retrieve(entity, query="smoke", max_results=5)
            edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query="smoke", max_results=5)
            sample["implemented_contract"] = True
            sample["result_count"] = len(results)
            sample["edge_count"] = len(edge)
            sample["sample_results"] = [
                {
                    "source_entity": getattr(r, "source_entity", None),
                    "target_entity": getattr(r, "target_entity", None),
                    "relationship": getattr(r, "relationship", None),
                    "evidence_text": getattr(r, "evidence_text", None),
                    "metadata": getattr(r, "metadata", None),
                    "source": getattr(r, "source", None),
                }
                for r in results[:2]
            ]
            if not available:
                problems.append("implemented dataset skill returned False from is_available()")
            if not results:
                problems.append("implemented dataset skill returned empty results for the standard query")
            for i, item in enumerate(results):
                if not getattr(item, "evidence_text", None):
                    problems.append(f"result[{i}] missing evidence_text")
                metadata = getattr(item, "metadata", None)
                if not isinstance(metadata, dict) or not metadata:
                    problems.append(f"result[{i}] metadata empty or not dict")
            if edge:
                problems.append("edge_query: expected empty list for non-existent entity")
    except Exception as exc:
        problems.append(f"runtime exception: {type(exc).__name__}: {exc}")

    status = "PASS" if not problems else "FAIL"
    if problems:
        fixes.append("Keep the grouped smoke-test contract aligned with the current runtime skill behavior.")
    return SkillReport(f"{case.number} {case.label}", status, commands, checks, sample, problems, fixes)


def main() -> int:
    reports = [run_case(case) for case in CASES]
    overall = "PASS" if all(r.status == "PASS" for r in reports) else "FAIL"
    print(json.dumps({"status": overall, "reports": [asdict(r) for r in reports]}, indent=2, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
