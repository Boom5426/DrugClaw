from __future__ import annotations

from pathlib import Path

from skills.drug_knowledgebase.drugbank import example as drugbank_example
from skills.drug_knowledgebase.drugcentral import example as drugcentral_example


def test_drugbank_example_uses_repo_local_resource_paths_by_default() -> None:
    vocab_path = Path(drugbank_example.VOCAB_PATH)
    data_path = Path(drugbank_example.DATA_PATH)

    assert str(vocab_path).startswith("/data/boom/Agent/DrugClaw/")
    assert str(data_path).startswith("/data/boom/Agent/DrugClaw/")
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(vocab_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(data_path)


def test_drugcentral_example_uses_repo_local_resource_paths_by_default() -> None:
    structures_path = Path(drugcentral_example.STRUCTURES_FILE)
    dti_path = Path(drugcentral_example.DTI_FILE)

    assert str(structures_path).startswith("/data/boom/Agent/DrugClaw/")
    assert str(dti_path).startswith("/data/boom/Agent/DrugClaw/")
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(structures_path)
    assert "/blue/qsong1/wang.qing/AgentLLM/DrugClaw" not in str(dti_path)
