from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from maintainers.resources.build_full_package import build_full_package
from maintainers.resources.validate_full_package import (
    validate_archive,
    validate_staging_tree,
)


def _load_contract() -> dict:
    return json.loads(
        Path("maintainers/resources/full_package_contract.json").read_text(encoding="utf-8")
    )


def _write_delimited_table(
    path: Path,
    *,
    header: list[str],
    rows: list[dict[str, str]],
    delimiter: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _excel_column_name(index: int) -> str:
    column = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        column = chr(65 + remainder) + column
    return column


def _write_minimal_xlsx(path: Path, *, sheet_name: str, rows: list[list[str]]) -> None:
    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    sheet_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row):
            if value == "":
                continue
            cell_ref = f"{_excel_column_name(column_index)}{row_index}"
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{_escape(str(value))}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook = f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{_escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
    worksheet = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {''.join(sheet_rows)}
  </sheetData>
</worksheet>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _materialize_contract_compliant_staging(parent: Path) -> Path:
    contract = _load_contract()
    staging_root = parent / contract["archive_root"]
    staging_root.mkdir(parents=True, exist_ok=True)

    for entry in contract["resources"]["direct"]:
        (staging_root / entry["canonical_path"]).mkdir(parents=True, exist_ok=True)

    for entry in contract["resources"]["rename_only"]:
        canonical = staging_root / entry["canonical_path"]
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.suffix:
            canonical.write_text("placeholder\n", encoding="utf-8")
        else:
            canonical.mkdir(parents=True, exist_ok=True)

    for entry in contract["resources"]["normalized"]:
        for rel_output in entry["canonical_outputs"]:
            output = staging_root / rel_output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("normalized\n", encoding="utf-8")

    return staging_root


def _materialize_synthetic_raw_source_tree(parent: Path) -> Path:
    contract = _load_contract()
    source_root = parent / "source"
    staging_root = source_root / contract["archive_root"]
    staging_root.mkdir(parents=True, exist_ok=True)

    for entry in contract["resources"]["direct"]:
        (staging_root / entry["canonical_path"]).mkdir(parents=True, exist_ok=True)

    for entry in contract["resources"]["rename_only"]:
        source = staging_root / entry["source_path"]
        source.mkdir(parents=True, exist_ok=True)
        if entry["resource_id"] == "webmd_drug_reviews":
            _write_delimited_table(
                source / "webmd_reviews.csv",
                header=["drug", "review"],
                rows=[{"drug": "aspirin", "review": "helpful"}],
                delimiter=",",
            )
        else:
            (source / "raw.txt").write_text(f"{entry['resource_id']}\n", encoding="utf-8")

    _write_minimal_xlsx(
        staging_root / "dti" / "GDKD" / "Knowledge_database_v20.0.xlsx",
        sheet_name="Sheet1",
        rows=[
            [
                "Disease",
                "Gene",
                "Variant",
                "Description",
                "Effect",
                "Association_1",
                "Therapeutic context_1",
                "Status_1",
                "Evidence_1",
            ],
            [
                "chronic_myeloid_leukemia",
                "ABL1",
                "T315I",
                "missense mutation",
                "gain-of-function",
                "response",
                "imatinib, dasatinib",
                "guideline",
                "consensus",
            ],
        ],
    )
    _write_delimited_table(
        staging_root / "dti" / "TarKG" / "Compound_nodes.csv",
        header=["index", "unify_id", "kind", "dbid", "db_source", "name"],
        rows=[
            {
                "index": "1",
                "unify_id": "TC1",
                "kind": "Compound",
                "dbid": "DB00619",
                "db_source": "DrugBank",
                "name": "Imatinib",
            }
        ],
        delimiter=",",
    )
    _write_delimited_table(
        staging_root / "dti" / "TarKG" / "Gene_nodes.csv",
        header=["index", "unify_id", "kind", "dbid", "db_source", "name"],
        rows=[
            {
                "index": "2",
                "unify_id": "P00519",
                "kind": "Gene",
                "dbid": "25",
                "db_source": "UniProt",
                "name": "ABL1_HUMAN",
            }
        ],
        delimiter=",",
    )
    _write_delimited_table(
        staging_root / "dti" / "TarKG" / "TarKG_edges.csv",
        header=["index", "node1", "node1_type", "relation", "node2", "node2_type"],
        rows=[
            {
                "index": "10",
                "node1": "TC1",
                "node1_type": "Compound",
                "relation": "inhibits",
                "node2": "P00519",
                "node2_type": "Gene",
            }
        ],
        delimiter=",",
    )
    _write_minimal_xlsx(
        staging_root / "drug_repurposing" / "RepurposeDrugs" / "dataset_single.xlsx",
        sheet_name="Drug Disease Sources",
        rows=[
            ["Drug_name", "Disease_name", "Phase", "Merged_RefNew"],
            [
                "imatinib",
                "systemic sclerosis",
                "2",
                "https://pubmed.ncbi.nlm.nih.gov/34567890/",
            ],
        ],
    )
    _write_delimited_table(
        staging_root / "drug_combination" / "DrugCombDB" / "drugcombdb_data" / "Syner&Antag_zip.csv",
        header=["ID", "Drug1", "Drug2", "Cell line", "ZIP", "classification"],
        rows=[
            {
                "ID": "1",
                "Drug1": "imatinib",
                "Drug2": "dasatinib",
                "Cell line": "K562",
                "ZIP": "18.4",
                "classification": "synergy",
            }
        ],
        delimiter=",",
    )
    _write_delimited_table(
        staging_root / "drug_combination" / "DrugComb" / "summary_table_v1.4.csv",
        header=[
            "block_id",
            "drug_row",
            "drug_col",
            "cell_line_name",
            "css",
            "synergy_zip",
            "synergy_bliss",
            "synergy_loewe",
            "synergy_hsa",
            "ic50_row",
            "ic50_col",
            "S",
        ],
        rows=[
            {
                "block_id": "7",
                "drug_row": "Erlotinib",
                "drug_col": "Rapamycin",
                "cell_line_name": "A549",
                "css": "22.2",
                "synergy_zip": "12.5",
                "synergy_bliss": "10.1",
                "synergy_loewe": "8.4",
                "synergy_hsa": "9.5",
                "ic50_row": "1.1",
                "ic50_col": "2.2",
                "S": "17.8",
            }
        ],
        delimiter=",",
    )
    liver_dir = staging_root / "drug_toxicity" / "LiverTox" / "livertox_NBK547852"
    liver_dir.mkdir(parents=True, exist_ok=True)
    (liver_dir / "Acetaminophen.nxml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<book-part-wrapper id="Acetaminophen">
  <book-part>
    <book-part-meta>
      <title-group>
        <title>Acetaminophen</title>
      </title-group>
    </book-part-meta>
  </book-part>
</book-part-wrapper>
""",
        encoding="utf-8",
    )
    _write_minimal_xlsx(
        staging_root / "drug_toxicity" / "DILI" / "DILIst Supplementary Table.xlsx",
        sheet_name="DILIst",
        rows=[
            ["DILIST_ID", "CompoundName", "DILIst Classification ", "Routs of Administration "],
            ["1", "acetaminophen", "1", "Oral"],
        ],
    )
    _write_minimal_xlsx(
        staging_root
        / "drug_toxicity"
        / "DILI"
        / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx",
        sheet_name="version 2",
        rows=[
            ["Drug Induced Liver Injury Rank (DILIrank) Dataset Ver 2.0 | FDA", "", "", ""],
            ["LTKBID", "CompoundName", "SeverityClass", "vDILI-Concern"],
            ["LT00001", "imatinib", "8", "vMOST-DILI-concern"],
        ],
    )

    return source_root


def _canonicalize_archive_name(name: str) -> str:
    normalized = name
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


def _archive_member_names(archive_path: Path) -> set[str]:
    with tarfile.open(archive_path, mode="r:gz") as handle:
        return {
            _canonicalize_archive_name(member.name)
            for member in handle.getmembers()
            if _canonicalize_archive_name(member.name) not in ("", ".")
        }


def _archive_member_text(archive_path: Path, member_name: str) -> str:
    with tarfile.open(archive_path, mode="r:gz") as handle:
        member = handle.getmember(member_name)
        extracted = handle.extractfile(member)
        assert extracted is not None
        return extracted.read().decode("utf-8")


def _archive_contains_path(member_names: set[str], relative_path: str) -> bool:
    expected = f"resources_metadata/{relative_path}"
    return any(name == expected or name.startswith(f"{expected}/") for name in member_names)


def _write_archive_from_staging(staging_root: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(staging_root, arcname=staging_root.name)
    return archive_path


def test_validate_staging_tree_rejects_wrong_root_name(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong_root"
    wrong.mkdir(parents=True)

    errors = validate_staging_tree(wrong)

    assert errors
    assert any("must be named resources_metadata" in error for error in errors)


def test_validate_staging_tree_rejects_missing_normalized_output(tmp_path: Path) -> None:
    staging_root = _materialize_contract_compliant_staging(tmp_path)
    missing = staging_root / "dti" / "GDKD" / "gdkd.csv"
    missing.unlink()

    errors = validate_staging_tree(staging_root)

    assert errors
    assert any("dti/GDKD/gdkd.csv" in error for error in errors)


def test_validate_staging_tree_accepts_contract_compliant_tree(tmp_path: Path) -> None:
    staging_root = _materialize_contract_compliant_staging(tmp_path)

    errors = validate_staging_tree(staging_root)

    assert errors == []


def test_validate_archive_rejects_non_overlay_archive_root(tmp_path: Path) -> None:
    wrong_root = tmp_path / "wrong_root"
    wrong_root.mkdir(parents=True)
    (wrong_root / "placeholder.txt").write_text("x\n", encoding="utf-8")
    archive_path = _write_archive_from_staging(
        wrong_root, tmp_path / "dist" / "resources_metadata_full.tar.gz"
    )

    errors = validate_archive(archive_path)

    assert errors
    assert any("top-level root" in error for error in errors)


def test_validate_archive_rejects_forbidden_runtime_root_nested_under_archive_root(
    tmp_path: Path,
) -> None:
    staging_root = _materialize_contract_compliant_staging(tmp_path)
    nested_forbidden = staging_root / "resources_metadata_full" / "payload.txt"
    nested_forbidden.parent.mkdir(parents=True, exist_ok=True)
    nested_forbidden.write_text("x\n", encoding="utf-8")
    archive_path = _write_archive_from_staging(
        staging_root, tmp_path / "dist" / "resources_metadata_full.tar.gz"
    )

    errors = validate_archive(archive_path)

    assert errors
    assert any("resources_metadata_full" in error for error in errors)


def test_build_full_package_rename_only_mappings_are_applied(tmp_path: Path) -> None:
    source_root = _materialize_synthetic_raw_source_tree(tmp_path)

    tar_path = build_full_package(output_dir=tmp_path / "dist", source_root=source_root)

    assert validate_archive(tar_path) == []
    names = _archive_member_names(tar_path)
    assert _archive_contains_path(names, "drug_knowledgebase/WHO_EML")
    assert _archive_contains_path(names, "drug_repurposing/Repurposing_Hub")
    assert _archive_contains_path(names, "drug_nlp/ADE_Corpus")
    assert _archive_contains_path(names, "drug_nlp/DDI_Corpus_2013")
    assert _archive_contains_path(names, "drug_nlp/TAC_2017_ADR")
    assert "resources_metadata/drug_review/WebMDDrugReviews/webmd.csv" in names
    assert "drug,review" in _archive_member_text(
        tar_path, "resources_metadata/drug_review/WebMDDrugReviews/webmd.csv"
    )


def test_build_full_package_normalizer_handlers_emit_canonical_outputs(
    tmp_path: Path,
) -> None:
    source_root = _materialize_synthetic_raw_source_tree(tmp_path)

    tar_path = build_full_package(output_dir=tmp_path / "dist", source_root=source_root)

    assert validate_archive(tar_path) == []
    names = _archive_member_names(tar_path)
    expected_outputs = (
        "dti/GDKD/gdkd.csv",
        "dti/TarKG/tarkg.tsv",
        "drug_repurposing/RepurposeDrugs/repurposedrugs.csv",
        "drug_combination/DrugCombDB/drugcombdb.csv",
        "drug_combination/DrugComb/drugcomb.csv",
        "drug_toxicity/LiverTox/livertox.json",
        "drug_toxicity/DILI/dili.csv",
    )
    for output in expected_outputs:
        assert f"resources_metadata/{output}" in names

    assert _archive_member_text(
        tar_path, "resources_metadata/dti/GDKD/gdkd.csv"
    ).splitlines()[0] == "drug,gene,score,source"
    assert _archive_member_text(
        tar_path, "resources_metadata/dti/TarKG/tarkg.tsv"
    ).splitlines()[0] == "drug\ttarget\trelation\tdisease\tpathway"
    assert _archive_member_text(
        tar_path,
        "resources_metadata/drug_repurposing/RepurposeDrugs/repurposedrugs.csv",
    ).splitlines()[0] == "drug,disease,score,status,pmid"
    assert _archive_member_text(
        tar_path, "resources_metadata/drug_combination/DrugCombDB/drugcombdb.csv"
    ).splitlines()[0] == "Drug1,Drug2,Cell,Synergy,SynergyType,PMID"
    drugcomb_header = _archive_member_text(
        tar_path, "resources_metadata/drug_combination/DrugComb/drugcomb.csv"
    ).splitlines()[0].split(",")
    assert {"drug_row", "drug_col", "cell_line_name", "synergy_zip", "synergy_bliss"}.issubset(
        set(drugcomb_header)
    )
    livertox_payload = json.loads(
        _archive_member_text(
            tar_path, "resources_metadata/drug_toxicity/LiverTox/livertox.json"
        )
    )
    assert livertox_payload[0]["drug"].lower() == "acetaminophen"
    assert _archive_member_text(
        tar_path, "resources_metadata/drug_toxicity/DILI/dili.csv"
    ).splitlines()[0] == "drug,warning_type,molecule_chembl_id"


def test_build_full_package_writes_overlay_compatible_tarball(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    staging_root = _materialize_contract_compliant_staging(source_root)

    tar_path = build_full_package(output_dir=tmp_path / "dist", source_root=source_root)

    assert tar_path.name.endswith(".tar.gz")
    assert tar_path.exists()
    assert validate_archive(tar_path) == []
    with tarfile.open(tar_path, mode="r:gz") as handle:
        names = [member.name for member in handle.getmembers()]

    assert names
    assert all(
        name == "resources_metadata" or name.startswith("resources_metadata/")
        for name in names
    )
    assert staging_root.exists()


def test_build_full_package_fails_if_contract_outputs_are_missing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    staging_root = _materialize_contract_compliant_staging(source_root)
    (staging_root / "dti" / "TarKG" / "tarkg.tsv").unlink()

    with pytest.raises(ValueError, match="contract validation failed"):
        build_full_package(output_dir=tmp_path / "dist", source_root=source_root)
