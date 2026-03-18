from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FILES_TO_SCAN = [
    ROOT / "README.md",
    ROOT / "README_CN.md",
    ROOT / "drugclaw" / "main_system.py",
    ROOT / "drugclaw" / "agent_retriever.py",
    ROOT / "drugclaw" / "agent_responder.py",
    ROOT / "skills" / "__init__.py",
    ROOT / "skills" / "skill_tree.py",
    ROOT / "drugclaw" / "config.py",
]

FORBIDDEN_SNIPPETS = [
    "68 curated drug knowledge resources",
    "70 curated drug resources",
    "57 implemented skills",
    "25 with example.py + SKILL.md",
]


def test_registry_is_the_only_authoritative_source_for_counts() -> None:
    for path in FILES_TO_SCAN:
        content = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            assert snippet not in content
