from __future__ import annotations

import csv
import re
from pathlib import Path

from ._xlsx import load_xlsx_records


_CANDIDATE_INPUTS = (
    "repurposedrugs.csv",
    "repurposedrugs.tsv",
    "repurpose_export.csv",
    "repurpose_export.tsv",
    "dataset_single.xlsx",
)


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"RepurposeDrugs normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}"
    )


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if input_path.suffix.lower() == ".tsv" else ","
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"RepurposeDrugs input has no rows: {input_path}")
    return rows


def _extract_pmid(reference: str) -> str:
    lowered = reference.lower()
    if "pubmed" not in lowered and "ncbi.nlm.nih.gov" not in lowered:
        return ""
    match = re.search(r"(\d{5,9})", reference)
    return match.group(1) if match else ""


def _read_xlsx_rows(input_path: Path) -> list[dict[str, str]]:
    rows = load_xlsx_records(
        input_path,
        sheet_name="Drug Disease Sources",
        required_headers=("Drug_name", "Disease_name"),
    )
    if not rows:
        raise ValueError(f"RepurposeDrugs input has no rows: {input_path}")
    return rows


def normalize_repurposedrugs(source_path: Path, output_path: Path) -> None:
    input_path = _pick_input_file(source_path)
    rows = _read_xlsx_rows(input_path) if input_path.suffix.lower() == ".xlsx" else _read_rows(input_path)

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug = (
            row.get("drug", "")
            or row.get("Drug", "")
            or row.get("compound", "")
            or row.get("Drug_name", "")
        ).strip()
        disease = (
            row.get("disease", "")
            or row.get("Disease", "")
            or row.get("indication", "")
            or row.get("Disease_name", "")
        ).strip()
        score = (row.get("score", "") or row.get("confidence", "")).strip()
        raw_status = (row.get("status", "") or row.get("Status", "") or row.get("Phase", "")).strip()
        status = raw_status if not raw_status.isdigit() else f"Phase {raw_status}"
        pmid = (row.get("pmid", "") or row.get("PMID", "")).strip() or _extract_pmid(
            str(row.get("Merged_RefNew", "")).strip()
        )
        if not drug or not disease:
            continue
        normalized.append(
            {
                "drug": drug,
                "disease": disease,
                "score": score,
                "status": status,
                "pmid": pmid,
            }
        )
    if not normalized:
        raise ValueError(
            f"RepurposeDrugs input missing required columns drug/disease: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["drug", "disease", "score", "status", "pmid"],
        )
        writer.writeheader()
        writer.writerows(normalized)
