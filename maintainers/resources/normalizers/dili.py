from __future__ import annotations

import csv
from pathlib import Path

from ._xlsx import load_xlsx_records


_CANDIDATE_INPUTS = ("dili.csv", "dili_raw.csv")
_DILIST_XLSX = "DILIst Supplementary Table.xlsx"
_DILIRANK_XLSX = "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"DILI normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}")


def _load_xlsx_rows(source_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    dilirank_path = source_path / _DILIRANK_XLSX
    dilist_path = source_path / _DILIST_XLSX

    if dilirank_path.is_file():
        for row in load_xlsx_records(
            dilirank_path,
            required_headers=("LTKBID", "CompoundName"),
        ):
            drug = str(row.get("CompoundName", "")).strip()
            warning = str(row.get("vDILI-Concern", "")).strip()
            if drug and warning:
                rows.append(
                    {
                        "drug": drug,
                        "warning_type": warning,
                        "molecule_chembl_id": "",
                    }
                )

    if dilist_path.is_file():
        label_map = {"1": "DILI-positive", "0": "DILI-negative"}
        for row in load_xlsx_records(
            dilist_path,
            required_headers=("DILIST_ID", "CompoundName"),
        ):
            drug = str(row.get("CompoundName", "")).strip()
            raw_label = str(
                row.get("DILIst Classification ", "") or row.get("DILIst Classification", "")
            ).strip()
            warning = label_map.get(raw_label, raw_label)
            if drug and warning:
                rows.append(
                    {
                        "drug": drug,
                        "warning_type": warning,
                        "molecule_chembl_id": "",
                    }
                )

    if not rows:
        raise ValueError(
            f"DILI normalizer expected CSV or {_DILIRANK_XLSX!r}/{_DILIST_XLSX!r} under {source_path}"
        )
    return rows


def normalize_dili(source_path: Path, output_path: Path) -> None:
    try:
        input_path = _pick_input_file(source_path)
        with input_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"DILI input has no rows: {input_path}")
    except ValueError:
        rows = _load_xlsx_rows(source_path)
        input_path = source_path

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug = str(row.get("drug", "")).strip()
        warning_type = str(
            row.get("warning_type", "") or row.get("WarningType", "") or row.get("warning", "")
        ).strip()
        molecule_chembl_id = str(
            row.get("molecule_chembl_id", "") or row.get("molecule_id", "")
        ).strip()
        if not drug:
            continue
        normalized.append(
            {
                "drug": drug,
                "warning_type": warning_type,
                "molecule_chembl_id": molecule_chembl_id,
            }
        )
    if not normalized:
        raise ValueError(f"DILI input missing required column drug: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["drug", "warning_type", "molecule_chembl_id"],
        )
        writer.writeheader()
        writer.writerows(normalized)
