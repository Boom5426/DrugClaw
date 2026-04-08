from __future__ import annotations

import json
from pathlib import Path

import drugclaw.resource_path_resolver as resolver
from drugclaw.config import Config
from drugclaw.resource_path_resolver import (
    canonical_local_resource_paths,
    get_repo_root,
    get_resources_metadata_root,
    resolve_path_value,
    resolve_skill_config_paths,
)


def test_resolve_path_value_uses_repo_root_for_relative_paths() -> None:
    repo_root = get_repo_root()

    resolved = resolve_path_value("resources_metadata/dti/TarKG/tarkg.tsv")

    assert resolved == repo_root / "resources_metadata" / "dti" / "TarKG" / "tarkg.tsv"


def test_resolve_skill_config_paths_uses_repo_local_defaults_for_known_skills() -> None:
    repo_root = get_repo_root()

    resolved = resolve_skill_config_paths("DRKG", {})

    assert resolved["drkg_tsv"] == str(
        repo_root / "resources_metadata" / "drug_repurposing" / "DRKG" / "drkg.tsv"
    )


def test_resolve_skill_config_paths_preserves_explicit_absolute_paths(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.csv"

    resolved = resolve_skill_config_paths("RepoDB", {"csv_path": str(explicit)})

    assert resolved["csv_path"] == str(explicit)


def test_discover_package_manifest_paths_scans_repo_local_packages_dir(tmp_path: Path) -> None:
    packages_dir = tmp_path / "resources_metadata" / "packages"
    packages_dir.mkdir(parents=True)
    manifest_path = packages_dir / "example.json"
    manifest_path.write_text('{"package_id":"example","skill_name":"Example"}', encoding="utf-8")

    assert hasattr(resolver, "discover_package_manifest_paths")

    paths = resolver.discover_package_manifest_paths(repo_root=tmp_path)

    assert paths == [manifest_path]


def test_resolve_package_component_paths_uses_repo_root_for_relative_entries(tmp_path: Path) -> None:
    explicit = tmp_path / "resources_metadata" / "packages" / "already_absolute.md"

    assert hasattr(resolver, "resolve_package_component_paths")

    resolved = resolver.resolve_package_component_paths(
        [
            "resources_metadata/knowhow/adr/rules.md",
            str(explicit),
        ],
        repo_root=tmp_path,
    )

    assert resolved == [
        str(tmp_path / "resources_metadata" / "knowhow" / "adr" / "rules.md"),
        str(explicit),
    ]


def test_resolve_package_component_paths_constrains_repo_relative_entries_to_resources_metadata(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "already_absolute.md"

    resolved = resolver.resolve_package_component_paths(
        [
            "resources_metadata_full/adr/SIDER/meddra_all_se.tsv",
            "other_root/notes.md",
            str(explicit),
        ],
        repo_root=tmp_path,
    )

    invalid_root = tmp_path / "resources_metadata" / "__invalid_package_component_path__"
    assert Path(resolved[0]).is_relative_to(invalid_root)
    assert Path(resolved[1]).is_relative_to(invalid_root)
    assert Path(resolved[2]).is_relative_to(invalid_root)


def test_get_resources_metadata_root_points_to_repo_local_single_runtime_root() -> None:
    repo_root = get_repo_root()

    resources_root = get_resources_metadata_root()

    assert resources_root == repo_root / "resources_metadata"
    assert "resources_metadata_full" not in resources_root.parts


def test_canonical_local_resource_paths_resolve_under_repo_resources_metadata() -> None:
    resources_root = get_resources_metadata_root()

    for path in canonical_local_resource_paths().values():
        assert path.is_absolute()
        assert resources_root in (path, *path.parents)


def test_config_repo_local_paths_stay_under_canonical_resources_metadata(tmp_path: Path) -> None:
    key_file = tmp_path / "navigator_api_keys.json"
    key_file.write_text(json.dumps({}), encoding="utf-8")
    config = Config(key_file=str(key_file))
    resources_root = get_resources_metadata_root()

    targets = [
        config.SKILL_CONFIGS["DrugBank"]["vocab_csv_path"],
        config.SKILL_CONFIGS["DrugBank"]["xml_path"],
        config.SKILL_CONFIGS["DrugCentral"]["structures_path"],
        config.SKILL_CONFIGS["DrugCentral"]["targets_path"],
        config.SKILL_CONFIGS["DrugCentral"]["approved_path"],
        *config.SKILL_CONFIGS["UniD3"].values(),
    ]

    for raw in targets:
        path = Path(raw)
        assert path.is_absolute()
        assert resources_root in (path, *path.parents)
        assert "resources_metadata_full" not in path.parts
