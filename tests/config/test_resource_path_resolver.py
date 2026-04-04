from __future__ import annotations

from pathlib import Path

import drugclaw.resource_path_resolver as resolver
from drugclaw.resource_path_resolver import (
    get_repo_root,
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
    explicit = tmp_path / "already_absolute.md"

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
