from __future__ import annotations

from pathlib import Path

from drugclaw.resource_path_resolver import default_skill_paths, get_repo_root
from skills.drug_combination.drugcomb import example as drugcomb_example
from skills.drug_combination.drugcombdb import example as drugcombdb_example
from skills.drug_knowledgebase.drugbank import example as drugbank_example
from skills.drug_knowledgebase.drugcentral import example as drugcentral_example
from skills.drug_repurposing.repurposedrugs import example as repurposedrugs_example
from skills.drug_toxicity.dili import example as dili_example
from skills.drug_toxicity.livertox import example as livertox_example
from skills.dti.gdkd import example as gdkd_example
from skills.dti.tarkg import example as tarkg_example


def test_drugbank_example_uses_repo_local_resource_paths_by_default() -> None:
    repo_root = get_repo_root()
    vocab_path = Path(drugbank_example.VOCAB_PATH)
    data_path = Path(drugbank_example.DATA_PATH)

    assert vocab_path.is_absolute()
    assert data_path.is_absolute()
    assert repo_root in vocab_path.parents
    assert repo_root in data_path.parents
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(vocab_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(data_path)


def test_drugcentral_example_uses_repo_local_resource_paths_by_default() -> None:
    repo_root = get_repo_root()
    structures_path = Path(drugcentral_example.STRUCTURES_FILE)
    dti_path = Path(drugcentral_example.DTI_FILE)

    assert structures_path.is_absolute()
    assert dti_path.is_absolute()
    assert repo_root in structures_path.parents
    assert repo_root in dti_path.parents
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(structures_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(dti_path)


def test_canonical_resource_examples_use_packaged_output_paths() -> None:
    expected = {
        "gdkd": (
            Path(gdkd_example.DATA_PATH),
            "resources_metadata/dti/GDKD/gdkd.csv",
        ),
        "tarkg": (
            Path(tarkg_example.DATA_PATH),
            "resources_metadata/dti/TarKG/tarkg.tsv",
        ),
        "repurposedrugs": (
            Path(repurposedrugs_example.DATA_PATH),
            "resources_metadata/drug_repurposing/RepurposeDrugs/repurposedrugs.csv",
        ),
        "drugcombdb": (
            Path(drugcombdb_example.DATA_PATH),
            "resources_metadata/drug_combination/DrugCombDB/drugcombdb.csv",
        ),
        "drugcomb": (
            Path(drugcomb_example.DATA_PATH),
            "resources_metadata/drug_combination/DrugComb/drugcomb.csv",
        ),
        "livertox": (
            Path(livertox_example.DATA_PATH),
            "resources_metadata/drug_toxicity/LiverTox/livertox.json",
        ),
        "dili": (
            Path(dili_example.DATA_PATH),
            "resources_metadata/drug_toxicity/DILI/dili.csv",
        ),
    }

    for _, (path_obj, rel_suffix) in expected.items():
        repo_root = get_repo_root()
        path_str = str(path_obj)
        assert path_obj.is_absolute()
        assert repo_root in path_obj.parents
        assert path_str.endswith(rel_suffix)
        assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in path_str


def test_default_skill_paths_use_canonical_packaged_outputs() -> None:
    repo_root = get_repo_root()
    expected = {
        "GDKD": ("csv_path", "resources_metadata/dti/GDKD/gdkd.csv"),
        "TarKG": ("tsv_path", "resources_metadata/dti/TarKG/tarkg.tsv"),
        "RepurposeDrugs": (
            "csv_path",
            "resources_metadata/drug_repurposing/RepurposeDrugs/repurposedrugs.csv",
        ),
        "DrugCombDB": (
            "csv_path",
            "resources_metadata/drug_combination/DrugCombDB/drugcombdb.csv",
        ),
        "DrugComb": (
            "csv_path",
            "resources_metadata/drug_combination/DrugComb/drugcomb.csv",
        ),
        "LiverTox": ("json_path", "resources_metadata/drug_toxicity/LiverTox/livertox.json"),
    }

    for skill_name, (config_key, relative_path) in expected.items():
        defaults = default_skill_paths(skill_name)
        assert defaults[config_key] == str(repo_root / relative_path)
