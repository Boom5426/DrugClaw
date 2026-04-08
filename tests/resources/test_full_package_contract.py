from __future__ import annotations

import json
from pathlib import Path

from drugclaw.resource_path_resolver import (
    canonical_local_resource_paths,
    default_skill_paths,
    get_package_manifest_dir,
    get_repo_root,
    get_resources_metadata_root,
    resolve_package_component_paths,
)


def _load_full_package_contract_manifest() -> dict:
    return json.loads(
        Path("maintainers/resources/full_package_contract.json").read_text(encoding="utf-8")
    )


def test_full_package_contract_uses_single_runtime_root_for_defaults() -> None:
    resources_root = get_resources_metadata_root()

    for skill_name in ("RepoDB", "SIDER", "TarKG", "DRKG"):
        defaults = default_skill_paths(skill_name)
        assert defaults
        for value in defaults.values():
            path = Path(value)
            assert path.is_absolute()
            assert resources_root in (path, *path.parents)
            assert "resources_metadata_full" not in path.parts


def test_full_package_contract_package_manifests_share_same_runtime_root() -> None:
    resources_root = get_resources_metadata_root()

    manifest_dir = get_package_manifest_dir()

    assert manifest_dir == resources_root / "packages"
    assert "resources_metadata_full" not in manifest_dir.parts


def test_full_package_contract_includes_representative_overlay_targets() -> None:
    repo_root = get_repo_root()
    resources_root = repo_root / "resources_metadata"
    targets = canonical_local_resource_paths(repo_root=repo_root)

    assert set(targets) == {"DrugBank", "SIDER", "TTD"}
    for target in targets.values():
        assert target.is_absolute()
        assert resources_root in (target, *target.parents)


def test_full_package_contract_constrains_repo_relative_package_paths(tmp_path: Path) -> None:
    canonical_absolute = tmp_path / "resources_metadata" / "packages" / "absolute_protocol.md"
    outside_absolute = tmp_path / "outside_root" / "absolute_protocol.md"
    resolved = resolve_package_component_paths(
        [
            "resources_metadata/packages/repodb_protocol.md",
            "resources_metadata_full/packages/repodb_protocol.md",
            "outside_root/readme.md",
            str(canonical_absolute),
            str(outside_absolute),
        ],
        repo_root=tmp_path,
    )
    resources_root = tmp_path / "resources_metadata"

    assert Path(resolved[0]) == resources_root / "packages" / "repodb_protocol.md"
    assert Path(resolved[1]).is_relative_to(
        resources_root / "__invalid_package_component_path__"
    )
    assert Path(resolved[2]).is_relative_to(
        resources_root / "__invalid_package_component_path__"
    )
    assert Path(resolved[3]) == canonical_absolute
    assert Path(resolved[4]).is_relative_to(
        resources_root / "__invalid_package_component_path__"
    )


def test_full_package_contract_manifest_has_required_sections() -> None:
    payload = _load_full_package_contract_manifest()

    assert payload["archive_root"] == "resources_metadata"
    assert payload["archive"]["root"] == "resources_metadata"
    assert payload["resources"]
    assert "direct" in payload["resources"]
    assert "rename_only" in payload["resources"]
    assert "normalized" in payload["resources"]


def test_full_package_contract_manifest_rename_only_entries_map_source_to_canonical() -> None:
    payload = _load_full_package_contract_manifest()
    mappings = payload["resources"]["rename_only"]

    assert mappings
    expected_keys = {"resource_id", "source_path", "canonical_path"}
    expected_mappings = {
        "who_eml": (
            "drug_knowledgebase/WHO Essential Medicines List",
            "drug_knowledgebase/WHO_EML",
        ),
        "drug_repurposing_hub": (
            "drug_repurposing/DrugRepurposingHub",
            "drug_repurposing/Repurposing_Hub",
        ),
        "ade_corpus": (
            "drug_nlp/ADECorpus",
            "drug_nlp/ADE_Corpus",
        ),
        "ddi_corpus_2013": (
            "drug_nlp/DDICorpus2013",
            "drug_nlp/DDI_Corpus_2013",
        ),
        "tac_2017_adr": (
            "drug_nlp/TAC2017ADR",
            "drug_nlp/TAC_2017_ADR",
        ),
        "drug_reviews_drugs_com": (
            "drug_review/DrugReviews",
            "drug_review/Drugs_com_Reviews",
        ),
        "webmd_drug_reviews": (
            "drug_review/WebMDDrugReviews",
            "drug_review/WebMDDrugReviews/webmd.csv",
        ),
    }

    by_id = {}
    for mapping in mappings:
        assert set(mapping) == expected_keys
        assert mapping["resource_id"]
        assert mapping["source_path"]
        assert mapping["canonical_path"]
        assert not mapping["source_path"].startswith("/")
        assert not mapping["canonical_path"].startswith("/")
        assert not mapping["source_path"].startswith("resources_metadata/")
        assert not mapping["canonical_path"].startswith("resources_metadata/")
        by_id[mapping["resource_id"]] = mapping

    assert set(by_id) == set(expected_mappings)
    for resource_id, (expected_source, expected_target) in expected_mappings.items():
        assert by_id[resource_id]["source_path"] == expected_source
        assert by_id[resource_id]["canonical_path"] == expected_target
    assert by_id["webmd_drug_reviews"]["canonical_path"].endswith(
        "WebMDDrugReviews/webmd.csv"
    )


def test_full_package_contract_manifest_normalized_entries_declare_canonical_outputs() -> None:
    payload = _load_full_package_contract_manifest()
    normalized = payload["resources"]["normalized"]

    assert normalized
    by_id = {entry["resource_id"]: entry for entry in normalized}
    for resource_id in ("gdkd", "tarkg", "repurposedrugs", "drugcombdb", "livertox"):
        assert resource_id in by_id
        outputs = by_id[resource_id]["canonical_outputs"]
        assert outputs
        for output in outputs:
            assert not output.startswith("/")
            assert not output.startswith("resources_metadata/")
