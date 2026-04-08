from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

AUTHORITATIVE_COUNT_PATTERNS = (
    r"\b\d+\s+implemented skills\b",
    r"\b\d+\s+curated drug resources\b",
    r"\b\d+\s+个可用 skill\b",
    r"\b\d+\s+个精选药物资源\b",
)


def test_readmes_do_not_contain_authoritative_resource_counts() -> None:
    for relative_path in ("README.md", "README_CN.md"):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in AUTHORITATIVE_COUNT_PATTERNS:
            assert re.search(pattern, content, re.IGNORECASE) is None


def test_docs_reference_navigator_keys_and_venv() -> None:
    for relative_path in ("README.md", "README_CN.md", "docs/index.html"):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "navigator_api_keys.json" in content
        assert "python3 -m venv .venv" in content
        assert ".venv/bin/activate" in content
        assert "python -m pip install --upgrade pip" in content
        assert "python -m pip install --no-build-isolation -r requirements.txt" in content

    for relative_path in ("README.md", "README_CN.md"):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "--key-file api_keys.json" in content
        sanitized = content.replace("navigator_api_keys.json", "")
        sanitized = sanitized.replace("--key-file api_keys.json", "")
        sanitized = sanitized.replace("`api_keys.json`", "")
        assert "api_keys.json" not in sanitized

    docs_index = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    assert "api_keys.json" not in docs_index.replace("navigator_api_keys.json", "")
