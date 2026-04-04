from __future__ import annotations

import json
from pathlib import Path

from drugclaw.resource_package_models import ResourcePackageManifest
from drugclaw.resource_package_registry import build_package_snapshot, load_package_manifests


def test_package_snapshot_marks_gateway_ready_when_endpoint_and_auth_are_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_path = tmp_path / "resources_metadata" / "packages" / "gateway_protocol.md"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("# Gateway Protocol\n", encoding="utf-8")
    monkeypatch.setenv("OPENFDA_GATEWAY_TOKEN", "test-token")

    manifest = ResourcePackageManifest.from_dict(
        {
            "package_id": "openfda_gateway_bundle",
            "skill_name": "openFDA Human Drug",
            "protocol_docs": [str(protocol_path.relative_to(tmp_path))],
            "gateway": {
                "transport": "rest_api",
                "endpoint": "https://api.fda.gov",
                "tool_namespace": "openfda.human_drug",
                "auth_env_vars": ["OPENFDA_GATEWAY_TOKEN"],
            },
        }
    )

    snapshot = build_package_snapshot(manifest, repo_root=tmp_path)

    assert snapshot.gateway_declared is True
    assert snapshot.gateway_ready is True
    assert snapshot.gateway_status == "ready"
    assert snapshot.gateway_transport == "rest_api"
    assert snapshot.gateway_tool_namespace == "openfda.human_drug"
    assert snapshot.gateway_missing_env == []


def test_package_snapshot_surfaces_missing_gateway_auth_env_vars(tmp_path: Path) -> None:
    protocol_path = tmp_path / "resources_metadata" / "packages" / "gateway_protocol.md"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("# Gateway Protocol\n", encoding="utf-8")

    manifest = ResourcePackageManifest.from_dict(
        {
            "package_id": "secure_gateway_bundle",
            "skill_name": "Secure API Skill",
            "protocol_docs": [str(protocol_path.relative_to(tmp_path))],
            "gateway": {
                "transport": "mcp",
                "endpoint": "http://127.0.0.1:8765/mcp",
                "tool_namespace": "secure.api",
                "auth_env_vars": ["SECURE_GATEWAY_TOKEN", "SECURE_GATEWAY_SECRET"],
            },
        }
    )

    snapshot = build_package_snapshot(manifest, repo_root=tmp_path)

    assert snapshot.gateway_declared is True
    assert snapshot.gateway_ready is False
    assert snapshot.gateway_status == "missing_auth"
    assert snapshot.gateway_missing_env == [
        "SECURE_GATEWAY_SECRET",
        "SECURE_GATEWAY_TOKEN",
    ]


def test_package_snapshot_preserves_gateway_read_only_flag(tmp_path: Path) -> None:
    protocol_path = tmp_path / "resources_metadata" / "packages" / "gateway_protocol.md"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("# Gateway Protocol\n", encoding="utf-8")

    manifest = ResourcePackageManifest.from_dict(
        {
            "package_id": "write_enabled_gateway_bundle",
            "skill_name": "Write Enabled Skill",
            "protocol_docs": [str(protocol_path.relative_to(tmp_path))],
            "gateway": {
                "transport": "rest_api",
                "endpoint": "https://example.com/api",
                "tool_namespace": "write.enabled",
                "read_only": False,
            },
        }
    )

    snapshot = build_package_snapshot(manifest, repo_root=tmp_path)

    assert snapshot.gateway_declared is True
    assert snapshot.gateway_ready is True
    assert snapshot.gateway_read_only is False


def test_load_package_manifests_skips_examples_directory(tmp_path: Path) -> None:
    packages_dir = tmp_path / "resources_metadata" / "packages"
    examples_dir = packages_dir / "examples"
    packages_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    (packages_dir / "real_package.json").write_text(
        json.dumps(
            {
                "package_id": "real_package",
                "skill_name": "Real Skill",
                "protocol_docs": ["resources_metadata/packages/real_protocol.md"],
            }
        ),
        encoding="utf-8",
    )
    (examples_dir / "minimal_package.json").write_text(
        json.dumps(
            {
                "package_id": "example_package",
                "skill_name": "Example Skill",
                "protocol_docs": ["resources_metadata/packages/examples/minimal_protocol.md"],
            }
        ),
        encoding="utf-8",
    )

    manifests = load_package_manifests(tmp_path)

    assert sorted(manifests) == ["Real Skill"]
