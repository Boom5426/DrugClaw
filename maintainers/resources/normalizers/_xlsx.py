from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

_SPREADSHEET_NS = {
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_ref_to_index(ref: str) -> int:
    index = 0
    for char in ref:
        if char.isalpha():
            index = index * 26 + (ord(char.upper()) - 64)
    return index - 1


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("x:si", _SPREADSHEET_NS):
        values.append(
            "".join(text.text or "" for text in item.iterfind(".//x:t", _SPREADSHEET_NS))
        )
    return values


def _sheet_targets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall("pr:Relationship", _SPREADSHEET_NS)
    }
    sheets = workbook.find("x:sheets", _SPREADSHEET_NS)
    if sheets is None:
        return []
    targets: list[tuple[str, str]] = []
    for sheet in sheets:
        rel_id = sheet.attrib.get(f"{{{_REL_NS}}}id")
        if not rel_id or rel_id not in rel_targets:
            continue
        targets.append((sheet.attrib.get("name", ""), "xl/" + rel_targets[rel_id].lstrip("/")))
    return targets


def load_xlsx_rows(path: Path, *, sheet_name: str | None = None) -> list[list[str]]:
    with ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_targets = _sheet_targets(archive)
        if not sheet_targets:
            return []

        target_path = sheet_targets[0][1]
        if sheet_name is not None:
            for current_name, current_target in sheet_targets:
                if current_name == sheet_name:
                    target_path = current_target
                    break

        worksheet = ET.fromstring(archive.read(target_path))
        rows: list[list[str]] = []
        for row in worksheet.findall(".//x:sheetData/x:row", _SPREADSHEET_NS):
            values: dict[int, str] = {}
            for cell in row.findall("x:c", _SPREADSHEET_NS):
                ref = cell.attrib.get("r", "A1")
                column_index = _column_ref_to_index("".join(ch for ch in ref if ch.isalpha()))
                cell_type = cell.attrib.get("t")
                value = cell.find("x:v", _SPREADSHEET_NS)
                inline = cell.find("x:is", _SPREADSHEET_NS)

                text = ""
                if cell_type == "s" and value is not None and value.text is not None:
                    text = shared_strings[int(value.text)]
                elif cell_type == "inlineStr" and inline is not None:
                    text = "".join(
                        node.text or "" for node in inline.iterfind(".//x:t", _SPREADSHEET_NS)
                    )
                elif value is not None and value.text is not None:
                    text = value.text

                values[column_index] = text.strip()

            if not values:
                continue
            width = max(values) + 1
            rows.append([values.get(index, "") for index in range(width)])
        return rows


def load_xlsx_records(
    path: Path,
    *,
    sheet_name: str | None = None,
    required_headers: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    rows = load_xlsx_rows(path, sheet_name=sheet_name)
    if not rows:
        return []

    header_index = 0
    if required_headers:
        required = {header.strip() for header in required_headers}
        for index, row in enumerate(rows):
            present = {cell.strip() for cell in row if cell.strip()}
            if required.issubset(present):
                header_index = index
                break

    header_row = rows[header_index]
    headers: list[str] = []
    for index, value in enumerate(header_row):
        header = value.strip() or f"col_{index}"
        if header in headers:
            header = f"{header}_{index}"
        headers.append(header)

    records: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(headers) - len(row))
        records.append({headers[i]: padded[i].strip() for i in range(len(headers))})
    return records
