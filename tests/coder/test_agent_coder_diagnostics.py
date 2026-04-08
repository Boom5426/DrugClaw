from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from drugclaw.agent_coder import CoderAgent
from drugclaw.skills.base import AccessMode, RetrievalResult


class _RegistryStub:
    def __init__(self, skill):
        self._skill = skill

    def get_skill_info_for_coder(self, name: str) -> str:
        return f"Skill info for {name}"

    def get_skill(self, name: str):
        if name == self._skill.name:
            return self._skill
        return None


class _LLMShouldNotRun:
    def generate(self, messages, temperature=0.3):
        raise AssertionError("LLM should not run for deterministic execution")


@dataclass
class _SkillStub:
    name: str = "DemoSkill"
    config: Dict[str, Any] = None
    access_mode: str = AccessMode.LOCAL_FILE
    should_raise: bool = False
    return_records: bool = True

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = {}

    def retrieve(self, entities, query="", max_results=50, **kwargs):
        if self.should_raise:
            raise RuntimeError("structured retrieve blew up")
        if not self.return_records:
            return []
        return [
            RetrievalResult(
                source_entity="imatinib",
                source_type="drug",
                target_entity="ABL1",
                target_type="gene",
                relationship="targets",
                weight=1.0,
                source=self.name,
                evidence_text="Imatinib targets ABL1.",
                sources=["PMID:1"],
                skill_category="dti",
            )
        ][:max_results]


def test_deterministic_first_skips_llm_when_structured_retrieve_succeeds() -> None:
    skill = _SkillStub()
    agent = CoderAgent(_LLMShouldNotRun(), _RegistryStub(skill))

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
        execution_strategy="deterministic_first",
    )

    info = result["per_skill"][skill.name]
    diagnostics = info["diagnostics"]

    assert info["strategy"] == "deterministic_first"
    assert diagnostics["structured_status"] == "success"
    assert diagnostics["llm_attempted"] is False
    assert diagnostics["final_status"] == "success_structured"


def test_deterministic_only_marks_text_only_script_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = _SkillStub(name="TextOnlySkill", return_records=False)
    agent = CoderAgent(_LLMShouldNotRun(), _RegistryStub(skill))

    skill_dir = tmp_path / "text_only_skill"
    skill_dir.mkdir()
    (skill_dir / "skill_impl.py").write_text("# stub\n", encoding="utf-8")
    (skill_dir / "retrieve.py").write_text("print('text-only fallback output')\n", encoding="utf-8")

    monkeypatch.setattr("drugclaw.agent_coder.inspect.getfile", lambda cls: str(skill_dir / "skill_impl.py"))
    monkeypatch.setattr(
        "drugclaw.agent_coder.subprocess.run",
        lambda *args, **kwargs: type(
            "CompletedProcess",
            (),
            {"stdout": "text-only fallback output", "stderr": "", "returncode": 0},
        )(),
    )

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
        execution_strategy="deterministic_only",
    )

    info = result["per_skill"][skill.name]
    diagnostics = info["diagnostics"]

    assert info["output"] == "text-only fallback output"
    assert diagnostics["script_status"] == "success"
    assert diagnostics["record_count"] == 0
    assert diagnostics["final_status"] == "success_text_only"


def test_deterministic_only_distinguishes_structured_error_from_empty() -> None:
    skill = _SkillStub(name="BrokenSkill", should_raise=True)
    agent = CoderAgent(_LLMShouldNotRun(), _RegistryStub(skill))

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
        execution_strategy="deterministic_only",
    )

    info = result["per_skill"][skill.name]
    diagnostics = info["diagnostics"]

    assert diagnostics["structured_status"] == "error"
    assert "retrieve() error" in diagnostics["structured_error"]
    assert diagnostics["final_status"] == "deterministic_failed"
