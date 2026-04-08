from __future__ import annotations

import csv
import re
from pathlib import Path

from ._xlsx import load_xlsx_records


_CANDIDATE_INPUTS = (
    "gdkd.csv",
    "CCLE_drug_data.csv",
    "ccle_drug_data.csv",
    "Knowledge_database_v20.0.xlsx",
)


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"GDKD normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}"
    )


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"GDKD input has no rows: {input_path}")
    return rows


def _split_therapeutic_context(context: str) -> list[str]:
    cleaned = context.strip()
    if not cleaned:
        return []
    results: list[str] = []
    for token in re.split(r"\s*[,;]\s*", cleaned):
        value = token.strip().strip("-").strip()
        if value.lower().startswith("and "):
            value = value[4:].strip()
        if value:
            results.append(value)
    return results


def _read_xlsx_rows(input_path: Path) -> list[dict[str, str]]:
    rows = load_xlsx_records(input_path, required_headers=("Gene",))
    if not rows:
        raise ValueError(f"GDKD input has no rows: {input_path}")

    normalized: list[dict[str, str]] = []
    for row in rows:
        gene = str(row.get("Gene", "")).strip()
        if not gene:
            continue
        for index in range(1, 9):
            context = str(row.get(f"Therapeutic context_{index}", "")).strip()
            if not context:
                continue
            score = str(row.get(f"Evidence_{index}", "")).strip()
            status = str(row.get(f"Status_{index}", "")).strip()
            source = " | ".join(part for part in (status, score) if part)
            for drug in _split_therapeutic_context(context):
                normalized.append(
                    {
                        "drug": drug,
                        "gene": gene,
                        "score": score,
                        "source": source or input_path.name,
                    }
                )
    if not normalized:
        raise ValueError(f"GDKD input missing therapeutic context rows: {input_path}")
    return normalized


def normalize_gdkd(source_path: Path, output_path: Path) -> None:
    input_path = _pick_input_file(source_path)
    rows = _read_xlsx_rows(input_path) if input_path.suffix.lower() == ".xlsx" else _read_rows(input_path)

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug = (row.get("drug", "") or row.get("compound", "")).strip()
        gene = (row.get("gene", "") or row.get("target", "")).strip()
        if not drug or not gene:
            continue
        normalized.append(
            {
                "drug": drug,
                "gene": gene,
                "score": str(row.get("score", "")).strip(),
                "source": str(row.get("source", input_path.name)).strip() or input_path.name,
            }
        )
    if not normalized:
        raise ValueError(f"GDKD input missing required columns drug/gene: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["drug", "gene", "score", "source"])
        writer.writeheader()
        writer.writerows(normalized)
