#!/usr/bin/env python3
"""
Runtime smoke tests for skills 66, 67, 68.

Checks performed per skill:
  - import skill class
  - instantiate skill
  - call is_available()
  - call retrieve(...) with a representative input from README/example
  - verify non-empty list
  - verify every item has evidence_text and metadata

This script prefers real external/resource paths over synthetic fixtures.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/data/boom/Agent/DrugClaw")


@dataclass
class SkillReport:
    name: str
    status: str
    commands_executed: list[str]
    checks_performed: list[str]
    sample_outputs: dict[str, Any]
    problems_found: list[str]
    minimal_fixes_suggested: list[str]


def _rr_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        d = item.to_dict()
    else:
        d = asdict(item)
    return {
        "source_entity": d.get("source_entity"),
        "target_entity": d.get("target_entity"),
        "relationship": d.get("relationship"),
        "evidence_text": d.get("evidence_text"),
        "metadata": d.get("metadata", {}),
        "source": d.get("source"),
    }


def _all_have_fields(results: list[Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for i, item in enumerate(results):
        evidence = getattr(item, "evidence_text", None)
        metadata = getattr(item, "metadata", None)
        if not evidence:
            problems.append(f"result[{i}] missing evidence_text")
        if metadata is None:
            problems.append(f"result[{i}] missing metadata")
    return (len(problems) == 0, problems)


def _format_report_failure(report: SkillReport) -> str:
    payload = {
        "name": report.name,
        "status": report.status,
        "problems_found": report.problems_found,
        "minimal_fixes_suggested": report.minimal_fixes_suggested,
        "sample_outputs": report.sample_outputs,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _assert_report_passes(report: SkillReport) -> None:
    assert report.status == "PASS", _format_report_failure(report)


def _run_sematyp() -> SkillReport:
    commands = [
        "python maintainers/smoke/test_skills_66_68.py",
        "discover local SemaTyP resource via skills/drug_disease/sematyp/example.py or resources_metadata/",
    ]
    checks = [
        "import SemaTyPSkill",
        "instantiate with real local file config if available",
        "call is_available()",
        "call retrieve(...) with README-style drug input",
        "validate evidence_text and metadata",
    ]
    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    mod = importlib.import_module("skills.drug_disease.sematyp")
    skill_cls = getattr(mod, "SemaTyPSkill")

    example_mod = importlib.import_module("skills.drug_disease.sematyp.example")
    csv_candidate: str | None = None

    candidate_roots = []
    example_data_dir = Path(getattr(example_mod, "DATA_DIR", "") or "")
    if str(example_data_dir):
        candidate_roots.append(example_data_dir)
    candidate_roots.append(ROOT / "resources_metadata" / "drug_disease" / "SemaTyP")

    try:
        sample["candidate_roots"] = [str(path) for path in candidate_roots]

        for root in candidate_roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not (p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}):
                    continue
                try:
                    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
                except Exception:
                    continue
                joined = " | ".join(head).lower()
                if not any(
                    tok in joined for tok in ["subject", "predicate", "object", "drug", "disease", "head", "tail"]
                ):
                    continue
                csv_candidate = str(p)
                sample["candidate_preview"] = head
                sample["selected_root"] = str(root)
                break
            if csv_candidate:
                break

        sample["csv_candidate"] = csv_candidate

        config = {"csv_path": csv_candidate} if csv_candidate else {}
        if csv_candidate and csv_candidate.endswith(".tsv"):
            config["delimiter"] = "\t"
        skill = skill_cls(config=config)
        available = skill.is_available()
        sample["is_available"] = available

        results = skill.retrieve(entities={"drug": ["imatinib"]}, query="mechanism", max_results=5)
        sample["result_count"] = len(results)
        sample["sample_results"] = [_rr_to_dict(r) for r in results[:2]]

        if not csv_candidate:
            problems.append("No real triplet file matching the runtime skill schema was found in local SemaTyP resources.")
            fixes.append("Point 66_SemaTyP.py / sematyp example docs to the repo-local triplet file used by SemaTyPSkill.")
            fixes.append("Alternatively add a loader config example showing the exact CSV/TSV path and delimiter.")
            return SkillReport("66 SemaTyP", "FAIL", commands, checks, sample, problems, fixes)

        if not available:
            problems.append("Skill imported but is_available() returned False with the discovered local resource.")
        if not results:
            problems.append("retrieve(...) returned an empty list with README-style input.")
        else:
            ok, field_problems = _all_have_fields(results)
            if not ok:
                problems.extend(field_problems)

        status = "PASS" if not problems else "FAIL"
        if problems:
            fixes.append("Verify local SemaTyP resources contain triplets with headers expected by SemaTyPSkill.")
        return SkillReport("66 SemaTyP", status, commands, checks, sample, problems, fixes)
    finally:
        pass


def test_sematyp() -> None:
    _assert_report_passes(_run_sematyp())


def _run_cpic() -> SkillReport:
    commands = ["python maintainers/smoke/test_skills_66_68.py"]
    checks = [
        "import CPICSkill",
        "instantiate skill",
        "call is_available()",
        "call retrieve(...) using representative CPIC drug input",
        "validate evidence_text and metadata",
        "validate live CPIC API path indirectly through retrieve(...)",
    ]
    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    mod = importlib.import_module("skills.drug_knowledgebase.cpic")
    skill_cls = getattr(mod, "CPICSkill")
    skill = skill_cls(config={"timeout": 20})
    sample["is_available"] = skill.is_available()

    # Use the example's representative drug rather than the weak README example.
    results = skill.retrieve(entities={"drug": ["clopidogrel"]}, query="guideline", max_results=5)
    sample["result_count"] = len(results)
    sample["sample_results"] = [_rr_to_dict(r) for r in results[:2]]

    if not sample["is_available"]:
        problems.append("is_available() returned False for an implemented REST skill.")
    if not results:
        problems.append("retrieve(...) returned an empty list for clopidogrel against the live CPIC API.")
        fixes.append("Align CPICSkill with the verified PostgREST schema used in skills/drug_knowledgebase/cpic/example.py.")
        fixes.append("Current implementation queries /v1/drug?name=<drug> and expects nested guidelines/genes fields that the live API does not expose.")
    else:
        ok, field_problems = _all_have_fields(results)
        if not ok:
            problems.extend(field_problems)

    status = "PASS" if not problems else "FAIL"
    return SkillReport("67 CPIC", status, commands, checks, sample, problems, fixes)


def test_cpic() -> None:
    _assert_report_passes(_run_cpic())


def _run_kegg() -> SkillReport:
    commands = ["python maintainers/smoke/test_skills_66_68.py"]
    checks = [
        "import KEGGDrugSkill",
        "instantiate skill",
        "call is_available()",
        "call retrieve(...) using representative drug input",
        "validate non-empty list",
        "validate evidence_text and metadata",
        "validate real KEGG REST fallback if bioservices is unavailable",
    ]
    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    mod = importlib.import_module("skills.ddi.kegg_drug")
    skill_cls = getattr(mod, "KEGGDrugSkill")
    skill = skill_cls(config={"timeout": 20})
    sample["is_available"] = skill.is_available()

    # README uses imatinib; examples show warfarin as a stronger DDI case.
    results = skill.retrieve(entities={"drug": ["warfarin"]}, query="interactions", max_results=5)
    sample["result_count"] = len(results)
    sample["sample_results"] = [_rr_to_dict(r) for r in results[:2]]

    if not sample["is_available"]:
        problems.append("is_available() returned False even though KEGGDrugSkill is marked implemented.")
    if not results:
        problems.append("retrieve(...) returned an empty list for warfarin.")
        fixes.append("If bioservices is absent, ensure REST fallback remains reachable and parses INTERACTION lines correctly.")
    else:
        ok, field_problems = _all_have_fields(results)
        if not ok:
            problems.extend(field_problems)

    status = "PASS" if not problems else "FAIL"
    return SkillReport("68 KEGG Drug", status, commands, checks, sample, problems, fixes)


def test_kegg() -> None:
    _assert_report_passes(_run_kegg())


def main() -> int:
    sys.path.insert(0, str(ROOT))
    reports = [_run_sematyp(), _run_cpic(), _run_kegg()]
    overall_status = "PASS" if all(r.status == "PASS" for r in reports) else "FAIL"
    payload = {
        "status": overall_status,
        "reports": [asdict(r) for r in reports],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
