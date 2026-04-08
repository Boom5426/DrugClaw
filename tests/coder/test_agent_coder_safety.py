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


class _LLMStub:
    def __init__(self, responses: List[str]):
        self._responses = list(responses)

    def generate(self, messages, temperature=0.3):
        if not self._responses:
            raise AssertionError("No LLM response queued")
        return self._responses.pop(0)


class _LLMShouldNotRun:
    def generate(self, messages, temperature=0.3):
        raise AssertionError("LLM should not be called in deterministic mode")


@dataclass
class _SkillStub:
    name: str = "DemoSkill"
    config: Dict[str, Any] = None
    access_mode: str = AccessMode.LOCAL_FILE

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = {}

    def retrieve(self, entities, query="", max_results=50, **kwargs):
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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import os\nprint('x')", "forbidden import"),
        ("import subprocess\nprint('x')", "forbidden import"),
        ("print(open('secret.txt').read())", "forbidden call"),
        ("print(os.environ.get('SECRET_KEY'))", "forbidden attribute"),
        ("import requests\nprint(requests.get('https://example.com'))", "forbidden import"),
        ("print(eval('1+1'))", "forbidden call"),
        ("print(exec('1+1'))", "forbidden call"),
        ("print(__import__('os'))", "forbidden call"),
        ("while True:\n    print('x')", "forbidden syntax"),
    ],
)
def test_validate_generated_code_rejects_unsafe_code(code: str, expected: str) -> None:
    error = CoderAgent._validate_generated_code(code)
    assert expected in error


def test_execute_code_allows_safe_skill_query_and_transform() -> None:
    agent = CoderAgent(_LLMStub([]), _RegistryStub(_SkillStub()))

    output, error = agent._execute_code(
        code=(
            "records = safe_query(max_results=5)\n"
            "genes = [record['target_entity'] for record in records if record['relationship'] == 'targets']\n"
            "print(', '.join(sorted(genes)))\n"
        ),
        skill_name="DemoSkill",
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        timeout_seconds=5,
    )

    assert error == ""
    assert output.strip() == "ABL1"


def test_generate_and_execute_falls_back_when_code_is_rejected() -> None:
    skill = _SkillStub()
    agent = CoderAgent(
        _LLMStub(
            [
                '{"approach": "use direct shell access", "operations": ["unsafe"]}',
                "import os\nprint(os.listdir('.'))",
            ]
        ),
        _RegistryStub(skill),
    )

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
    )

    assert skill.name in result["per_skill"]
    assert result["per_skill"][skill.name]["strategy"] == "fallback_retrieve"
    assert "Imatinib targets ABL1." in result["per_skill"][skill.name]["output"]


def test_generate_and_execute_falls_back_when_execution_raises() -> None:
    skill = _SkillStub()
    agent = CoderAgent(
        _LLMStub(
            [
                '{"approach": "retrieve then print missing variable", "operations": ["retrieve", "format"], "focus_fields": ["target_entity"], "output_style": "plain", "needs_imports": []}',
                "records = safe_query(max_results=5)\nprint(undefined_name)\n",
            ]
        ),
        _RegistryStub(skill),
    )

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
    )

    assert result["per_skill"][skill.name]["strategy"] == "fallback_retrieve"
    assert "Imatinib targets ABL1." in result["per_skill"][skill.name]["output"]


def test_generate_and_execute_can_use_deterministic_only_strategy() -> None:
    skill = _SkillStub()
    agent = CoderAgent(_LLMShouldNotRun(), _RegistryStub(skill))

    result = agent.generate_and_execute(
        skill_names=[skill.name],
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results_per_skill=5,
        execution_strategy="deterministic_only",
    )

    assert result["per_skill"][skill.name]["strategy"] == "deterministic_only"
    assert "Imatinib targets ABL1." in result["per_skill"][skill.name]["output"]


def test_fallback_retrieve_skips_retrieve_py_for_rest_api_skills(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = _SkillStub(name="RestSkill", access_mode=AccessMode.REST_API)
    agent = CoderAgent(_LLMStub([]), _RegistryStub(skill))

    skill_dir = tmp_path / "rest_skill"
    skill_dir.mkdir()
    (skill_dir / "skill_impl.py").write_text("# stub\n", encoding="utf-8")
    (skill_dir / "retrieve.py").write_text(
        "raise SystemExit('retrieve.py should not run for REST skills')\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("drugclaw.agent_coder.inspect.getfile", lambda cls: str(skill_dir / "skill_impl.py"))
    monkeypatch.setattr(
        "drugclaw.agent_coder.subprocess.run",
        lambda *args, **kwargs: type(
            "CompletedProcess",
            (),
            {"stdout": "retrieve.py output should not be used", "stderr": "", "returncode": 0},
        )(),
    )

    output, error, records = agent._fallback_retrieve(
        skill_name=skill.name,
        entities={"drug": ["imatinib"]},
        query="What does imatinib target?",
        max_results=5,
    )

    assert error == ""
    assert output != "retrieve.py output should not be used"
    assert "Imatinib targets ABL1." in output
    assert records
    assert records[0]["target_entity"] == "ABL1"
