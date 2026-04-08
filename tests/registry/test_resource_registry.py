from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from drugclaw import cli
from drugclaw.config import Config
from drugclaw.skills import build_default_registry
from drugclaw.skills.base import AccessMode, RAGSkill
from drugclaw.skills.registry import SkillRegistry
from drugclaw.resource_registry import ResourceEntry, ResourceRegistry, build_resource_registry
from drugclaw.resource_path_resolver import get_repo_root, get_resources_metadata_root


def _config_stub() -> SimpleNamespace:
    return SimpleNamespace(SKILL_CONFIGS={}, KG_ENDPOINTS={})


def _missing_metadata_config(tmp_path: Path) -> SimpleNamespace:
    missing_root = tmp_path / "missing_resources"
    return SimpleNamespace(
        SKILL_CONFIGS={
            "SIDER": {
                "se_tsv": str(missing_root / "adr" / "SIDER" / "meddra_all_se.tsv"),
            },
            "MecDDI": {
                "csv_path": str(missing_root / "ddi" / "MecDDI" / "mecddi.csv"),
            },
            "RepoDB": {
                "csv_path": str(
                    missing_root / "drug_repurposing" / "RepoDB" / "full.csv"
                ),
            },
            "DrugCentral": {
                "structures_path": str(
                    missing_root
                    / "drug_knowledgebase"
                    / "DrugCentral"
                    / "structures.smiles.tsv"
                ),
                "targets_path": str(
                    missing_root
                    / "drug_knowledgebase"
                    / "DrugCentral"
                    / "drug.target.interaction.tsv"
                ),
                "approved_path": str(
                    missing_root
                    / "drug_knowledgebase"
                    / "DrugCentral"
                    / "FDA+EMA+PMDA_Approved.csv"
                ),
            },
            "DrugBank": {
                "vocab_csv_path": str(
                    missing_root
                    / "drug_knowledgebase"
                    / "DrugBank"
                    / "drugbank vocabulary.csv"
                ),
                "xml_path": str(
                    missing_root
                    / "drug_knowledgebase"
                    / "DrugBank"
                    / "full database.xml"
                ),
            },
        },
        KG_ENDPOINTS={},
    )


class _PackageAwareDummySkill(RAGSkill):
    name = "PackageAware"
    subcategory = "dti"
    resource_type = "Dataset"
    access_mode = AccessMode.LOCAL_FILE
    aim = "Test package overlays"
    data_range = "fixture"
    _implemented = True

    def retrieve(
        self,
        entities: Dict[str, List[str]],
        query: str = "",
        max_results: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        return []

    def is_available(self) -> bool:
        return True


def test_resource_registry_enabled_count_matches_registered_skills() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)
    summary = resource_registry.summarize_registry()

    assert summary["enabled_resources"] == len(skill_registry.get_registered_skills())
    assert sum(summary["status_counts"].values()) == summary["total_resources"]


def test_resource_registry_summary_includes_package_status_and_capability_counts() -> None:
    registry = ResourceRegistry(
        [
            ResourceEntry(
                id="bindingdb",
                name="BindingDB",
                category="dti",
                description="binding evidence",
                entrypoint="skills.dti.bindingdb:BindingDBSkill",
                enabled=True,
                requires_metadata=False,
                required_metadata_paths=[],
                required_dependencies=[],
                supports_code_generation=True,
                fallback_retrieve_supported=True,
                status="ready",
                status_reason="available",
                access_mode="REST_API",
                resource_type="Database",
                package_id="bindingdb_core",
                package_status="ready",
                package_components=[],
                missing_components=[],
                has_knowhow=True,
                gateway_declared=True,
                gateway_ready=True,
            ),
            ResourceEntry(
                id="sider",
                name="SIDER",
                category="adr",
                description="label-derived adverse effects",
                entrypoint="skills.adr.sider:SIDERSkill",
                enabled=True,
                requires_metadata=True,
                required_metadata_paths=["/tmp/missing.tsv"],
                required_dependencies=[],
                supports_code_generation=True,
                fallback_retrieve_supported=True,
                status="missing_metadata",
                status_reason="missing local metadata: /tmp/missing.tsv",
                access_mode="LOCAL_FILE",
                resource_type="Database",
                package_id="sider_safety",
                package_status="missing_metadata",
                package_components=[],
                missing_components=["dataset_bundle"],
                has_knowhow=True,
                gateway_declared=True,
                gateway_ready=False,
            ),
        ]
    )

    summary = registry.summarize_registry()

    assert summary["package_status_counts"]["ready"] == 1
    assert summary["package_status_counts"]["missing_metadata"] == 1
    assert summary["resources_with_knowhow"] == 2
    assert summary["gateway_declared_resources"] == 2
    assert summary["gateway_ready_resources"] == 1
    assert summary["missing_component_counts"] == {"dataset_bundle": 1}


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


def test_resource_registry_marks_explicitly_missing_local_file_skills_as_missing_metadata(
    tmp_path: Path,
) -> None:
    skill_registry = build_default_registry(_missing_metadata_config(tmp_path))

    resource_registry = build_resource_registry(skill_registry)

    sider = resource_registry.get_resource("SIDER")
    mecddi = resource_registry.get_resource("MecDDI")

    assert sider is not None
    assert sider.status == "missing_metadata"
    assert mecddi is not None
    assert mecddi.status == "missing_metadata"


def test_resource_registry_marks_explicitly_missing_repodb_as_missing_metadata(
    tmp_path: Path,
) -> None:
    skill_registry = build_default_registry(_missing_metadata_config(tmp_path))

    resource_registry = build_resource_registry(skill_registry)

    repodb = resource_registry.get_resource("RepoDB")

    assert repodb is not None
    assert repodb.status == "missing_metadata"
    assert str(tmp_path / "missing_resources" / "drug_repurposing" / "RepoDB" / "full.csv") in repodb.status_reason


def test_resource_registry_marks_explicitly_missing_drugcentral_and_drugbank_as_missing_metadata(
    tmp_path: Path,
) -> None:
    skill_registry = build_default_registry(_missing_metadata_config(tmp_path))

    resource_registry = build_resource_registry(skill_registry)

    drugcentral = resource_registry.get_resource("DrugCentral")
    drugbank = resource_registry.get_resource("DrugBank")

    assert drugcentral is not None
    assert drugbank is not None
    assert drugcentral.status == "missing_metadata"
    assert drugbank.status == "missing_metadata"


def test_resource_registry_marks_repo_local_skills_ready_when_default_metadata_exists() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)

    drkg = resource_registry.get_resource("DRKG")
    tarkg = resource_registry.get_resource("TarKG")
    unitox = resource_registry.get_resource("UniTox")

    assert drkg is not None
    assert tarkg is not None
    assert unitox is not None

    assert drkg.status == "ready"
    assert tarkg.status == "ready"
    assert unitox.status == "ready"


def test_resource_registry_required_metadata_paths_use_single_repo_runtime_root() -> None:
    skill_registry = build_default_registry(_config_stub())
    resource_registry = build_resource_registry(skill_registry)
    resources_root = get_resources_metadata_root()

    for name in ("RepoDB", "SIDER", "TarKG", "DRKG", "UniTox"):
        entry = resource_registry.get_resource(name)
        assert entry is not None
        assert entry.required_metadata_paths
        for raw_path in entry.required_metadata_paths:
            resolved = Path(raw_path)
            assert resolved.is_absolute()
            assert resources_root in (resolved, *resolved.parents)
            assert "resources_metadata_full" not in resolved.parts


def test_registry_skills_keep_config_derived_repo_local_paths_under_resources_metadata(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "navigator_api_keys.json"
    key_file.write_text("{}", encoding="utf-8")
    skill_registry = build_default_registry(Config(key_file=str(key_file)))
    resources_root = get_resources_metadata_root()
    skills = {skill.name: skill for skill in skill_registry.get_registered_skills()}

    targets = [
        skills["DrugBank"].config["vocab_csv_path"],
        skills["DrugBank"].config["xml_path"],
        skills["DrugCentral"].config["structures_path"],
        skills["DrugCentral"].config["targets_path"],
        skills["DrugCentral"].config["approved_path"],
        skills["UniD3"].config["UniD3_Level1_DDM"],
        skills["UniD3"].config["UniD3_Level1_DEA"],
        skills["UniD3"].config["UniD3_Level1_DTA"],
        skills["UniD3"].config["UniD3_Level2_DDM"],
        skills["UniD3"].config["UniD3_Level2_DEA"],
        skills["UniD3"].config["UniD3_Level2_DTA"],
    ]
    for raw_path in targets:
        path = Path(raw_path)
        assert path.is_absolute()
        assert resources_root in (path, *path.parents)
        assert "resources_metadata_full" not in path.parts


def test_resource_registry_applies_package_overlay_and_planner_profile_tags(tmp_path) -> None:
    dataset_path = tmp_path / "resources_metadata" / "dti" / "PackageAware" / "records.csv"
    protocol_path = tmp_path / "resources_metadata" / "packages" / "package_aware_protocol.md"
    how_to_path = tmp_path / "resources_metadata" / "packages" / "package_aware_how_to.md"
    manifest_path = tmp_path / "resources_metadata" / "packages" / "package_aware.json"
    dataset_path.parent.mkdir(parents=True)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("id,value\n1,test\n", encoding="utf-8")
    protocol_path.write_text("# Protocol\n", encoding="utf-8")
    how_to_path.write_text("# How To\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "package_id": "package_aware_bundle",
                "skill_name": "PackageAware",
                "dataset_bundle": [str(dataset_path.relative_to(tmp_path))],
                "protocol_docs": [str(protocol_path.relative_to(tmp_path))],
                "how_to_docs": [str(how_to_path.relative_to(tmp_path))],
                "software_dependencies": ["json"],
                "knowhow_docs": ["resources_metadata/knowhow/dti/package_aware.md"],
            }
        ),
        encoding="utf-8",
    )

    assert "repo_root" in inspect.signature(build_resource_registry).parameters

    skill_registry = SkillRegistry()
    skill = _PackageAwareDummySkill()
    skill_registry.register(skill)

    resource_registry = build_resource_registry(skill_registry, repo_root=tmp_path)
    entry = resource_registry.get_resource("PackageAware")

    assert entry is not None
    assert hasattr(entry, "package_status")
    assert entry.package_id == "package_aware_bundle"
    assert entry.status == "degraded"
    assert entry.package_status == "degraded"
    assert entry.gateway_ready is True
    assert entry.has_knowhow is False
    assert "knowhow_docs" in entry.missing_components
    assert any(
        component["component_type"] == "knowhow_docs"
        and component["status"] == "missing_metadata"
        for component in entry.package_components
    )

    profile = skill.planner_profile()
    assert profile["has_dataset_bundle"] is True
    assert profile["has_protocol"] is True
    assert profile["has_how_to"] is True
    assert profile["has_knowhow"] is False
    assert profile["has_software_dependency"] is True
    assert "has_dataset_bundle" in profile["tags"]
    assert "has_protocol" in profile["tags"]
    assert "has_how_to" in profile["tags"]
    assert "has_software_dependency" in profile["tags"]


def test_resource_registry_without_manifest_keeps_legacy_status_behavior(tmp_path) -> None:
    assert "repo_root" in inspect.signature(build_resource_registry).parameters

    skill_registry = SkillRegistry()
    skill = _PackageAwareDummySkill()
    skill_registry.register(skill)

    resource_registry = build_resource_registry(skill_registry, repo_root=tmp_path)
    entry = resource_registry.get_resource("PackageAware")

    assert entry is not None
    assert hasattr(entry, "package_status")
    assert entry.status == "ready"
    assert entry.package_status == "ready"
    assert entry.gateway_ready is True
    assert entry.package_components == []
    assert entry.missing_components == []


def test_resource_registry_exposes_live_package_manifests_for_priority_resources() -> None:
    skill_registry = build_default_registry(_config_stub())

    resource_registry = build_resource_registry(skill_registry)

    bindingdb = resource_registry.get_resource("BindingDB")
    chembl = resource_registry.get_resource("ChEMBL")
    dgidb = resource_registry.get_resource("DGIdb")
    open_targets = resource_registry.get_resource("Open Targets Platform")
    stitch = resource_registry.get_resource("STITCH")
    tarkg = resource_registry.get_resource("TarKG")
    drugmechdb = resource_registry.get_resource("DRUGMECHDB")
    ddinter = resource_registry.get_resource("DDInter")
    kegg_drug = resource_registry.get_resource("KEGG Drug")
    mecddi = resource_registry.get_resource("MecDDI")
    repodb = resource_registry.get_resource("RepoDB")
    cpic = resource_registry.get_resource("CPIC")
    dailymed = resource_registry.get_resource("DailyMed")
    medlineplus = resource_registry.get_resource("MedlinePlus Drug Info")
    adrecs = resource_registry.get_resource("ADReCS")
    faers = resource_registry.get_resource("FAERS")
    nsides = resource_registry.get_resource("nSIDES")
    sider = resource_registry.get_resource("SIDER")
    openfda = resource_registry.get_resource("openFDA Human Drug")
    pharmgkb = resource_registry.get_resource("PharmGKB")
    drugcentral = resource_registry.get_resource("DrugCentral")
    drugbank = resource_registry.get_resource("DrugBank")

    for entry, expected_package_id in (
        (bindingdb, "bindingdb_core"),
        (chembl, "chembl_core"),
        (dgidb, "dgidb_core"),
        (open_targets, "open_targets_core"),
        (stitch, "stitch_target_profile"),
        (tarkg, "tarkg_target_profile"),
        (drugmechdb, "drugmechdb_mechanism"),
        (ddinter, "ddinter_clinical"),
        (kegg_drug, "kegg_drug_ddi"),
        (mecddi, "mecddi_mechanistic_ddi"),
        (repodb, "repodb_repurposing"),
        (cpic, "cpic_pgx"),
        (dailymed, "dailymed_labeling"),
        (medlineplus, "medlineplus_labeling"),
        (adrecs, "adrecs_safety"),
        (faers, "faers_safety"),
        (nsides, "nsides_safety"),
        (sider, "sider_safety"),
        (openfda, "openfda_labeling"),
        (pharmgkb, "pharmgkb_pgx"),
        (drugcentral, "drugcentral_repurposing"),
        (drugbank, "drugbank_multirole"),
    ):
        assert entry is not None
        assert entry.package_id == expected_package_id
        assert entry.has_knowhow is True
        assert "knowhow_docs" not in entry.missing_components
        assert any(
            component["component_type"] == "knowhow_docs"
            and component["status"] == "ready"
            for component in entry.package_components
        )
