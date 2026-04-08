from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET


_CANDIDATE_INPUTS = ("livertox.json", "livertox_fixture.json", "livertox.jsonl")


def _pick_input_file(source_path: Path) -> Path:
    for name in _CANDIDATE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"LiverTox normalizer expected one of {_CANDIDATE_INPUTS} under {source_path}"
    )


def _parse_rows(input_path: Path) -> list[dict[str, str]]:
    if input_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"LiverTox input must be a JSON list: {input_path}")
        rows = payload
    if not rows:
        raise ValueError(f"LiverTox input has no rows: {input_path}")
    return rows


def _parse_nxml_rows(source_path: Path) -> list[dict[str, str]]:
    candidates = sorted((source_path / "livertox_NBK547852").glob("*.nxml"))
    if not candidates:
        candidates = sorted(source_path.rglob("*.nxml"))
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            root = ET.fromstring(candidate.read_text(encoding="utf-8", errors="ignore"))
        except ET.ParseError:
            continue
        title = ""
        for element in root.findall(".//title-group/title"):
            text = "".join(element.itertext()).strip()
            if text:
                title = text
                break
        if not title:
            title = candidate.stem.replace("_", " ").strip()
        if not title:
            continue
        rows.append({"drug": title, "title": title, "ncbi_book_id": "NBK547852"})
    if not rows:
        raise ValueError(f"LiverTox input missing .nxml chapters: {source_path}")
    return rows


def normalize_livertox(source_path: Path, output_path: Path) -> None:
    try:
        input_path = _pick_input_file(source_path)
        rows = _parse_rows(input_path)
    except ValueError:
        rows = _parse_nxml_rows(source_path)
        input_path = source_path

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug = str(row.get("drug", "")).strip()
        title = str(row.get("title", "")).strip()
        book_id = str(row.get("ncbi_book_id", "")).strip()
        if not drug or not title:
            continue
        normalized.append({"drug": drug, "title": title, "ncbi_book_id": book_id})
    if not normalized:
        raise ValueError(
            f"LiverTox input missing required fields drug/title: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
