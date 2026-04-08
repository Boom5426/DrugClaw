from __future__ import annotations

import csv
from pathlib import Path


_CANDIDATE_INPUTS = (
    "drugcombdb.csv",
    "combination_data.tsv",
    "combination_data.csv",
    "drugcombdb_data.csv",
    "Syner&Antag_zip.csv",
    "drugcombs_scored.csv",
)


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
        nested = next(source_path.rglob(name), None)
        if nested is not None and nested.is_file():
            return nested
    raise ValueError(
        f"DrugCombDB normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}"
    )


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if input_path.suffix.lower() == ".tsv" else ","
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"DrugCombDB input has no rows: {input_path}")
    return rows


def normalize_drugcombdb(source_path: Path, output_path: Path) -> None:
    input_path = _pick_input_file(source_path)
    rows = _read_rows(input_path)

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug1 = (row.get("Drug1", "") or row.get("drug1", "") or row.get("Drug_1", "")).strip()
        drug2 = (row.get("Drug2", "") or row.get("drug2", "") or row.get("Drug_2", "")).strip()
        cell = (
            row.get("Cell", "")
            or row.get("CellLine", "")
            or row.get("cell_line", "")
            or row.get("Cell line", "")
        ).strip()
        synergy = (
            row.get("Synergy", "")
            or row.get("Score", "")
            or row.get("score", "")
            or row.get("synergy_score", "")
            or row.get("ZIP", "")
        ).strip()
        synergy_type = (
            row.get("SynergyType", "")
            or row.get("Combination_type", "")
            or row.get("synergy_type", "")
            or row.get("classification", "")
        ).strip()
        pmid = (row.get("PMID", "") or row.get("pmid", "")).strip()
        if not drug1 or not drug2:
            continue
        normalized.append(
            {
                "Drug1": drug1,
                "Drug2": drug2,
                "Cell": cell,
                "Synergy": synergy,
                "SynergyType": synergy_type,
                "PMID": pmid,
            }
        )
    if not normalized:
        raise ValueError(
            f"DrugCombDB input missing required columns Drug1/Drug2 or variants: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Drug1", "Drug2", "Cell", "Synergy", "SynergyType", "PMID"],
        )
        writer.writeheader()
        writer.writerows(normalized)
