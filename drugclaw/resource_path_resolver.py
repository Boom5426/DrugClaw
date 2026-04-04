from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


_PATH_KEY_SUFFIXES = (
    "_path",
    "_csv",
    "_tsv",
    "_json",
    "_xml",
    "_graphml",
    "_dir",
    "_folder",
)


_DEFAULT_SKILL_PATHS: Dict[str, Dict[str, str]] = {
    "TarKG": {
        "tsv_path": "resources_metadata/dti/TarKG/tarkg.tsv",
    },
    "GDKD": {
        "csv_path": "resources_metadata/dti/GDKD/gdkd.csv",
    },
    "DTC": {
        "csv_path": "resources_metadata/dti/DTC/DTC_data.csv",
    },
    "PharmKG": {
        "train_tsv": "resources_metadata/drug_knowledgebase/PharmKG/train.tsv",
    },
    "WHO Essential Medicines List": {
        "csv_path": "resources_metadata/drug_knowledgebase/WHO_EML/medicines.csv",
    },
    "RepoDB": {
        "csv_path": "resources_metadata/drug_repurposing/RepoDB/full.csv",
    },
    "DRKG": {
        "drkg_tsv": "resources_metadata/drug_repurposing/DRKG/drkg.tsv",
    },
    "OREGANO": {
        "csv_path": "resources_metadata/drug_repurposing/OREGANO/oregano.csv",
    },
    "Drug Repurposing Hub": {
        "csv_path": "resources_metadata/drug_repurposing/Repurposing_Hub/repurposing_hub.csv",
    },
    "DrugRepoBank": {
        "csv_path": "resources_metadata/drug_repurposing/DrugRepoBank/drugrepobank.csv",
    },
    "RepurposeDrugs": {
        "csv_path": "resources_metadata/drug_repurposing/RepurposeDrugs/repurposedrugs.csv",
    },
    "CancerDR": {
        "csv_path": "resources_metadata/drug_repurposing/CancerDR/cancerdr.csv",
    },
    "EK-DRD": {
        "csv_path": "resources_metadata/drug_repurposing/EK_DRD/ek_drd.csv",
    },
    "UniTox": {
        "csv_path": "resources_metadata/drug_toxicity/UniTox/unitox.csv",
    },
    "DILIrank": {
        "csv_path": "resources_metadata/drug_toxicity/DILIrank/dilirank.csv",
    },
    "SIDER": {
        "se_tsv": "resources_metadata/adr/SIDER/meddra_all_se.tsv",
    },
    "MecDDI": {
        "csv_path": "resources_metadata/ddi/MecDDI/mecddi.csv",
    },
    "LiverTox": {
        "json_path": "resources_metadata/drug_toxicity/LiverTox/livertox.json",
    },
    "DrugCombDB": {
        "csv_path": "resources_metadata/drug_combination/DrugCombDB/drugcombdb.csv",
    },
    "DrugComb": {
        "csv_path": "resources_metadata/drug_combination/DrugComb/drugcomb.csv",
    },
    "GDSC": {
        "csv_path": "resources_metadata/drug_molecular_property/GDSC/screened_compounds_rel_8.4.csv",
    },
    "SemaTyP": {
        "csv_path": "resources_metadata/drug_disease/SemaTyP/train.tsv",
    },
    "DDI Corpus 2013": {
        "tsv_path": "resources_metadata/drug_nlp/DDI_Corpus_2013/ddi_corpus.tsv",
    },
    "DrugProt": {
        "tsv_path": "resources_metadata/drug_nlp/DrugProt/drugprot.tsv",
    },
    "ADE Corpus": {
        "csv_path": "resources_metadata/drug_nlp/ADE_Corpus/ade_corpus.csv",
    },
    "CADEC": {
        "csv_path": "resources_metadata/drug_nlp/CADEC/cadec.csv",
    },
    "PsyTAR": {
        "csv_path": "resources_metadata/drug_nlp/PsyTAR/psytar.csv",
    },
    "PHEE": {
        "json_path": "resources_metadata/drug_nlp/PHEE/phee.json",
    },
    "TAC 2017 ADR": {
        "tsv_path": "resources_metadata/drug_nlp/TAC_2017_ADR/tac2017.tsv",
    },
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_package_manifest_dir(repo_root: Path | None = None) -> Path:
    return (repo_root or get_repo_root()) / "resources_metadata" / "packages"


def is_path_key(key: str) -> bool:
    return str(key).lower().endswith(_PATH_KEY_SUFFIXES)


def resolve_path_value(value: str | Path, repo_root: Path | None = None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root or get_repo_root()) / candidate


def discover_package_manifest_paths(repo_root: Path | None = None) -> List[Path]:
    packages_dir = get_package_manifest_dir(repo_root)
    if not packages_dir.exists():
        return []
    return sorted(
        path for path in packages_dir.glob("*.json")
        if path.is_file()
    )


def resolve_package_component_paths(
    values: Iterable[str | Path],
    repo_root: Path | None = None,
) -> List[str]:
    root = repo_root or get_repo_root()
    resolved: List[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate:
            continue
        resolved.append(str(resolve_path_value(candidate, root)))
    return resolved


def default_skill_paths(skill_name: str, repo_root: Path | None = None) -> Dict[str, str]:
    root = repo_root or get_repo_root()
    defaults = _DEFAULT_SKILL_PATHS.get(skill_name, {})
    return {
        key: str(resolve_path_value(relative_path, root))
        for key, relative_path in defaults.items()
    }


def resolve_skill_config_paths(
    skill_name: str,
    config: Dict[str, Any] | None,
    repo_root: Path | None = None,
) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    resolved: Dict[str, Any] = dict(config or {})
    defaults = default_skill_paths(skill_name, root)

    for key, default_value in defaults.items():
        current_value = resolved.get(key, "")
        if isinstance(current_value, str) and current_value.strip():
            continue
        resolved[key] = default_value

    for key, value in list(resolved.items()):
        if isinstance(value, str) and value.strip() and is_path_key(key):
            resolved[key] = str(resolve_path_value(value, root))
        elif isinstance(value, dict):
            resolved[key] = _resolve_nested_mapping(value, root)

    return resolved


def collect_required_metadata_paths(
    skill_name: str,
    config: Dict[str, Any] | None,
    repo_root: Path | None = None,
) -> List[str]:
    resolved = resolve_skill_config_paths(skill_name, config, repo_root=repo_root)
    return sorted(dict.fromkeys(_collect_paths(resolved)))


def _resolve_nested_mapping(value: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    resolved_nested: Dict[str, Any] = {}
    for nested_key, nested_value in value.items():
        if isinstance(nested_value, str) and nested_value.strip() and is_path_key(str(nested_key)):
            resolved_nested[nested_key] = str(resolve_path_value(nested_value, repo_root))
        else:
            resolved_nested[nested_key] = nested_value
    return resolved_nested


def _collect_paths(config: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key, value in config.items():
        if isinstance(value, str) and value.strip() and is_path_key(str(key)):
            paths.append(value)
        elif isinstance(value, dict):
            paths.extend(_collect_nested_paths(value.items()))
    return paths


def _collect_nested_paths(items: Iterable[tuple[Any, Any]]) -> List[str]:
    nested_paths: List[str] = []
    for key, value in items:
        if isinstance(value, str) and value.strip() and is_path_key(str(key)):
            nested_paths.append(value)
    return nested_paths
