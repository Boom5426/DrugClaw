from __future__ import annotations

from pathlib import Path

from skills.drug_knowledgebase.drugbank import example as drugbank_example
from skills.drug_knowledgebase.drugcentral import example as drugcentral_example


def test_drugbank_example_uses_repo_local_resource_paths_by_default() -> None:
    vocab_path = Path(drugbank_example.VOCAB_PATH)
    data_path = Path(drugbank_example.DATA_PATH)

    assert vocab_path.exists()
    assert data_path.exists()
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(vocab_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(data_path)


def test_drugcentral_example_uses_repo_local_resource_paths_by_default() -> None:
    structures_path = Path(drugcentral_example.STRUCTURES_FILE)
    dti_path = Path(drugcentral_example.DTI_FILE)

    assert structures_path.exists()
    assert dti_path.exists()
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(structures_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(dti_path)
