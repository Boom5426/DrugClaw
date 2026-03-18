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
