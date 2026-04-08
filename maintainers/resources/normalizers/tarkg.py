from __future__ import annotations

import csv
from pathlib import Path

_READY_MADE_INPUTS = ("tarkg.tsv", "TarKG.tsv", "TarKG.csv", "relations.tsv", "relations.csv")
_RAW_EDGE_FILE = "TarKG_edges.csv"
_COMPOUND_NODE_FILE = "Compound_nodes.csv"
_GENE_NODE_FILE = "Gene_nodes.csv"
_ALLOWED_RELATIONS = {
    "target",
    "binds",
    "interacts with",
    "inhibits",
    "activates",
    "downregulates",
    "upregulates",
    "agonism",
    "antagonism",
    "blocker",
    "carrier",
    "channels",
    "enzyme",
    "enzyme activity",
    "transporter",
}


def _pick_input_file(source_path: Path) -> Path:
    for name in _READY_MADE_INPUTS:
        candidate = source_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"TarKG normalizer expected one of {_READY_MADE_INPUTS} under {source_path}"
    )


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if input_path.suffix.lower() == ".tsv" else ","
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"TarKG input has no rows: {input_path}")
    return rows


def _normalize_gene_name(value: str, *, fallback: str) -> str:
    candidate = value.strip()
    if not candidate:
        return fallback
    if "_" in candidate:
        prefix = candidate.split("_", 1)[0].strip()
        if prefix:
            return prefix
    return candidate


def _load_node_names(path: Path, *, node_type: str) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("kind", "") != node_type:
                continue
            unify_id = str(row.get("unify_id", "")).strip()
            if not unify_id or unify_id in names:
                continue
            raw_name = str(row.get("name", "")).strip()
            if node_type == "Gene":
                raw_name = _normalize_gene_name(
                    raw_name,
                    fallback=str(row.get("dbid", "")).strip() or unify_id,
                )
            else:
                raw_name = raw_name or str(row.get("dbid", "")).strip() or unify_id
            if raw_name:
                names[unify_id] = raw_name
    return names


def _normalize_from_raw_graph(source_path: Path, output_path: Path) -> None:
    edge_path = source_path / _RAW_EDGE_FILE
    compound_path = source_path / _COMPOUND_NODE_FILE
    gene_path = source_path / _GENE_NODE_FILE
    if not edge_path.is_file() or not compound_path.is_file() or not gene_path.is_file():
        raise ValueError(
            "TarKG normalizer expected TarKG_edges.csv, Compound_nodes.csv, and Gene_nodes.csv "
            f"under {source_path}"
        )

    compound_names = _load_node_names(compound_path, node_type="Compound")
    gene_names = _load_node_names(gene_path, node_type="Gene")
    if not compound_names or not gene_names:
        raise ValueError(f"TarKG node tables did not yield compound/gene names under {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with edge_path.open(newline="", encoding="utf-8", errors="ignore") as source_handle:
        reader = csv.DictReader(source_handle)
        with output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=["drug", "target", "relation", "disease", "pathway"],
                delimiter="\t",
            )
            writer.writeheader()
            wrote_rows = False
            for row in reader:
                if row.get("node1_type") != "Compound" or row.get("node2_type") != "Gene":
                    continue
                relation = str(row.get("relation", "")).strip().lower()
                if relation not in _ALLOWED_RELATIONS:
                    continue
                drug = compound_names.get(str(row.get("node1", "")).strip(), "")
                target = gene_names.get(str(row.get("node2", "")).strip(), "")
                if not drug or not target:
                    continue
                writer.writerow(
                    {
                        "drug": drug,
                        "target": target,
                        "relation": relation,
                        "disease": "",
                        "pathway": "",
                    }
                )
                wrote_rows = True

    if not wrote_rows:
        raise ValueError(f"TarKG raw graph did not yield any compound->gene rows: {source_path}")


def normalize_tarkg(source_path: Path, output_path: Path) -> None:
    try:
        input_path = _pick_input_file(source_path)
    except ValueError:
        _normalize_from_raw_graph(source_path, output_path)
        return

    rows = _read_rows(input_path)

    normalized: list[dict[str, str]] = []
    for row in rows:
        drug = (row.get("drug", "") or row.get("source", "")).strip()
        target = (row.get("target", "") or row.get("destination", "")).strip()
        relation = str(row.get("relation", "drug_target_interaction")).strip()
        disease = str(row.get("disease", "")).strip()
        pathway = str(row.get("pathway", "")).strip()
        if not drug or not target:
            continue
        normalized.append(
            {
                "drug": drug,
                "target": target,
                "relation": relation or "drug_target_interaction",
                "disease": disease,
                "pathway": pathway,
            }
        )
    if not normalized:
        raise ValueError(
            f"TarKG input missing required columns drug/target or source/destination: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["drug", "target", "relation", "disease", "pathway"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(normalized)
