from __future__ import annotations

from types import SimpleNamespace

from drugclaw import cli
from drugclaw.config import Config
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


def test_resource_registry_marks_unconfigured_local_file_skills_as_missing_metadata() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)

    sider = resource_registry.get_resource("SIDER")
    mecddi = resource_registry.get_resource("MecDDI")

    assert sider is not None
    assert sider.status == "missing_metadata"
    assert mecddi is not None
    assert mecddi.status == "missing_metadata"


def test_resource_registry_marks_unconfigured_repodb_as_missing_metadata() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)

    repodb = resource_registry.get_resource("RepoDB")

    assert repodb is not None
    assert repodb.status == "missing_metadata"


def test_resource_registry_marks_unconfigured_drugcentral_and_drugbank_as_missing_metadata() -> None:
    skill_registry = build_default_registry(Config())

    resource_registry = build_resource_registry(skill_registry)

    drugcentral = resource_registry.get_resource("DrugCentral")
    drugbank = resource_registry.get_resource("DrugBank")

    assert drugcentral is not None
    assert drugbank is not None
    assert drugcentral.status == "missing_metadata"
    assert drugbank.status == "missing_metadata"
