from __future__ import annotations

from types import SimpleNamespace

from drugclaw import cli
from drugclaw.skills import build_default_registry
from drugclaw.resource_registry import build_resource_registry


def _config_stub() -> SimpleNamespace:
    return SimpleNamespace(SKILL_CONFIGS={}, KG_ENDPOINTS={})


def test_resource_registry_enabled_count_matches_registered_skills() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)
    summary = resource_registry.summarize_registry()

    assert summary["enabled_resources"] == len(skill_registry.get_registered_skills())
    assert sum(summary["status_counts"].values()) == summary["total_resources"]


def test_cli_registry_summary_matches_resource_registry_counts() -> None:
    skill_registry = build_default_registry(_config_stub())
    resource_registry = build_resource_registry(skill_registry)
    summary = resource_registry.summarize_registry()

    lines = cli._registry_summary_lines(summary)

    assert any(
        line == f"[OK] registry_total_resources: {summary['total_resources']}"
        for line in lines
    )
    assert any(
        line == f"[OK] registry_enabled_resources: {summary['enabled_resources']}"
        for line in lines
    )
