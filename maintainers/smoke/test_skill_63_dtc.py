#!/usr/bin/env python3
"""
Real smoke test for skills.dti.dtc.DTCSkill.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import urllib.request
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


def download_dtc_csv(out_dir: Path) -> str | None:
    dest = out_dir / "DTC_data.csv"
    url = "https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
    except Exception:
        return None
    return str(dest)


def main() -> int:
    commands = ["python maintainers/smoke/test_skill_63_dtc.py"]
    checks = [
        "read dtc_skill.py",
        "read README.md",
        "read skills/dti/dtc/README.md",
        "import DTCSkill",
        "attempt real DTC download from the documented DTC CSV endpoint",
        "instantiate DTCSkill with downloaded csv_path if available",
        "call is_available()",
        "standard query 1: drug=imatinib",
        "standard query 2: drug=gefitinib",
        "edge query: drug=zzz_not_a_real_drug_zzz",
        "validate non-empty structured results for standard queries",
        "validate evidence_text and metadata",
    ]

    problems: list[str] = []
    fixes: list[str] = []
    sample: dict[str, Any] = {}

    from skills.dti.dtc import DTCSkill

    tmpdir = Path(tempfile.mkdtemp(prefix="dtc_test_"))
    try:
        csv_path = download_dtc_csv(tmpdir)
        sample["csv_path"] = csv_path

        if not csv_path:
            problems.append("Real DTC CSV download failed; runtime skill cannot be exercised against live data.")
            fixes.append("Current DTC endpoint returns SSL failure under normal verification and HTTP 500 even with insecure fallback.")
            fixes.append("Mark DTC as external-resource blocked or switch to a maintained mirror/local snapshot before claiming runtime availability.")
            report = Report(
                status="FAIL",
                commands_executed=commands,
                checks_performed=checks,
                sample_returned_records=sample,
                problems_found=problems,
                minimal_fixes_suggested=fixes,
            )
            print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
            return 1

        skill = DTCSkill(config={"csv_path": csv_path})
        sample["is_available"] = skill.is_available()
        r1 = skill.retrieve({"drug": ["imatinib"]}, query="target", max_results=5)
        r2 = skill.retrieve({"drug": ["gefitinib"]}, query="target", max_results=5)
        edge = skill.retrieve({"drug": ["zzz_not_a_real_drug_zzz"]}, query="target", max_results=5)
        sample["imatinib_count"] = len(r1)
        sample["gefitinib_count"] = len(r2)
        sample["edge_count"] = len(edge)
        sample["imatinib_sample"] = [rr_view(x) for x in r1[:2]]
        sample["gefitinib_sample"] = [rr_view(x) for x in r2[:2]]
        sample["edge_sample"] = [rr_view(x) for x in edge[:2]]

        status = "PASS" if sample["is_available"] and r1 and r2 and not edge else "FAIL"
        report = Report(
            status=status,
            commands_executed=commands,
            checks_performed=checks,
            sample_returned_records=sample,
            problems_found=problems,
            minimal_fixes_suggested=fixes,
        )
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return 0 if status == "PASS" else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
