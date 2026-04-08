from __future__ import annotations

import csv
from pathlib import Path


_CANDIDATE_INPUTS = ("drugcomb.csv", "drugcomb_data_v1.5.csv", "summary_table_v1.4.csv")


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"DrugComb normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}"
    )


def normalize_drugcomb(source_path: Path, output_path: Path) -> None:
    input_path = _pick_input_file(source_path)
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"DrugComb input has no rows: {input_path}")

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug_row = (row.get("drug_row", "") or row.get("Drug1", "") or row.get("drug1", "")).strip()
        drug_col = (row.get("drug_col", "") or row.get("Drug2", "") or row.get("drug2", "")).strip()
        cell_line = (
            row.get("cell_line_name", "") or row.get("CellLine", "") or row.get("cell_line", "")
        ).strip()
        synergy_zip = (
            row.get("synergy_zip", "")
            or row.get("Score", "")
            or row.get("score", "")
            or row.get("synergy_score", "")
        ).strip()
        synergy_bliss = (row.get("synergy_bliss", "")).strip()
        if not drug_row or not drug_col:
            continue
        normalized.append(
            {
                "block_id": str(row.get("block_id", "")).strip(),
                "drug_row": drug_row,
                "drug_col": drug_col,
                "cell_line_name": cell_line,
                "drug_row_cid": str(row.get("drug_row_cid", "") or row.get("cid_row", "")).strip(),
                "drug_col_cid": str(row.get("drug_col_cid", "") or row.get("cid_col", "")).strip(),
                "synergy_zip": synergy_zip,
                "synergy_bliss": synergy_bliss,
                "synergy_loewe": str(row.get("synergy_loewe", "")).strip(),
                "synergy_hsa": str(row.get("synergy_hsa", "")).strip(),
                "css": str(row.get("css", "")).strip(),
                "css_ri": str(row.get("css_ri", "")).strip(),
                "S_mean": str(row.get("S_mean", "") or row.get("S", "")).strip(),
                "S_max": str(row.get("S_max", "")).strip(),
                "ic50_row": str(row.get("ic50_row", "")).strip(),
                "ic50_col": str(row.get("ic50_col", "")).strip(),
                "study_name": str(row.get("study_name", "") or row.get("source", "")).strip(),
            }
        )
    if not normalized:
        raise ValueError(
            f"DrugComb input missing required columns drug_row/drug_col or variants: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "block_id",
                "drug_row",
                "drug_col",
                "cell_line_name",
                "drug_row_cid",
                "drug_col_cid",
                "synergy_zip",
                "synergy_bliss",
                "synergy_loewe",
                "synergy_hsa",
                "css",
                "css_ri",
                "S_mean",
                "S_max",
                "ic50_row",
                "ic50_col",
                "study_name",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized)
