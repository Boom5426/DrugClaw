from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .resource_package_models import (
    PackageComponentStatus,
    ResourcePackageGateway,
    ResourcePackageManifest,
    ResourcePackageSnapshot,
)
from .resource_path_resolver import (
    discover_package_manifest_paths,
    resolve_package_component_paths,
)


def load_package_manifests(repo_root: Path | None = None) -> Dict[str, ResourcePackageManifest]:
    manifests: Dict[str, ResourcePackageManifest] = {}
    for path in discover_package_manifest_paths(repo_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest = ResourcePackageManifest.from_dict(payload)
        if manifest.skill_name:
            manifests[manifest.skill_name] = manifest
    return manifests


def build_resource_package_snapshots(
    skill_names: Iterable[str],
    *,
    repo_root: Path | None = None,
) -> Dict[str, ResourcePackageSnapshot]:
    manifests = load_package_manifests(repo_root)
    snapshots: Dict[str, ResourcePackageSnapshot] = {}
    for skill_name in skill_names:
        manifest = manifests.get(skill_name)
        if manifest is None:
            snapshots[skill_name] = build_fallback_package_snapshot(skill_name)
            continue
        snapshots[skill_name] = build_package_snapshot(manifest, repo_root=repo_root)
    return snapshots


def attach_package_snapshots(
    skills: Iterable[Any],
    snapshots: Dict[str, ResourcePackageSnapshot],
) -> None:
    for skill in skills:
        snapshot = snapshots.get(getattr(skill, "name", ""))
        if snapshot is not None:
            setattr(skill, "resource_package_snapshot", snapshot)


def build_fallback_package_snapshot(skill_name: str) -> ResourcePackageSnapshot:
    return ResourcePackageSnapshot(
        package_id=_normalize_id(skill_name),
        skill_name=skill_name,
        status="ready",
        status_reason="no resource package manifest; using legacy skill-only package view",
        components=[],
        missing_components=[],
    )


def build_package_snapshot(
    manifest: ResourcePackageManifest,
    *,
    repo_root: Path | None = None,
) -> ResourcePackageSnapshot:
    components: List[PackageComponentStatus] = []
    components.extend(
        _path_components(
            "dataset_bundle",
            manifest.dataset_bundle,
            repo_root=repo_root,
            required=True,
        )
    )
    components.extend(
        _path_components(
            "protocol_docs",
            manifest.protocol_docs,
            repo_root=repo_root,
            required=True,
        )
    )
    components.extend(
        _path_components(
            "how_to_docs",
            manifest.how_to_docs,
            repo_root=repo_root,
            required=False,
        )
    )
    components.extend(
        _path_components(
            "knowhow_docs",
            manifest.knowhow_docs,
            repo_root=repo_root,
            required=False,
        )
    )
    components.extend(_dependency_components(manifest.software_dependencies))

    status, status_reason = _summarize_components(components)
    missing_components = sorted(
        {
            component.component_type
            for component in components
            if component.status != "ready"
        }
    )
    gateway = _summarize_gateway(manifest.gateway)

    return ResourcePackageSnapshot(
        package_id=manifest.package_id or _normalize_id(manifest.skill_name),
        skill_name=manifest.skill_name,
        status=status,
        status_reason=status_reason,
        components=components,
        missing_components=missing_components,
        has_dataset_bundle=_has_ready_component(components, "dataset_bundle"),
        has_protocol=_has_ready_component(components, "protocol_docs"),
        has_how_to=_has_ready_component(components, "how_to_docs"),
        has_knowhow=_has_ready_component(components, "knowhow_docs"),
        has_software_dependency=_has_ready_component(components, "software_dependencies"),
        gateway_declared=gateway["gateway_declared"],
        gateway_ready=gateway["gateway_ready"],
        gateway_status=gateway["gateway_status"],
        gateway_reason=gateway["gateway_reason"],
        gateway_transport=gateway["gateway_transport"],
        gateway_endpoint=gateway["gateway_endpoint"],
        gateway_tool_namespace=gateway["gateway_tool_namespace"],
        gateway_missing_env=gateway["gateway_missing_env"],
        gateway_read_only=gateway["gateway_read_only"],
    )


def _path_components(
    component_type: str,
    values: List[str],
    *,
    repo_root: Path | None,
    required: bool,
) -> List[PackageComponentStatus]:
    components: List[PackageComponentStatus] = []
    for resolved in resolve_package_component_paths(values, repo_root=repo_root):
        exists = Path(resolved).exists()
        components.append(
            PackageComponentStatus(
                component_type=component_type,
                path_or_name=resolved,
                status="ready" if exists else "missing_metadata",
                reason=(
                    "present in the current workspace"
                    if exists
                    else f"missing local metadata: {resolved}"
                ),
                required=required,
            )
        )
    return components


def _dependency_components(values: List[str]) -> List[PackageComponentStatus]:
    components: List[PackageComponentStatus] = []
    for dependency in values:
        module_name = dependency.replace("-", "_")
        available = find_spec(module_name) is not None
        components.append(
            PackageComponentStatus(
                component_type="software_dependencies",
                path_or_name=dependency,
                status="ready" if available else "missing_dependency",
                reason=(
                    "dependency importable in current environment"
                    if available
                    else f"missing dependency: {dependency}"
                ),
                required=True,
            )
        )
    return components


def _summarize_components(
    components: List[PackageComponentStatus],
) -> tuple[str, str]:
    if not components:
        return "ready", "no package components declared"

    missing_dependency = next(
        (component for component in components if component.status == "missing_dependency"),
        None,
    )
    if missing_dependency is not None:
        return "missing_dependency", missing_dependency.reason

    required_missing = next(
        (
            component
            for component in components
            if component.required and component.status == "missing_metadata"
        ),
        None,
    )
    if required_missing is not None:
        return "missing_metadata", required_missing.reason

    optional_missing = next(
        (
            component
            for component in components
            if not component.required and component.status == "missing_metadata"
        ),
        None,
    )
    if optional_missing is not None:
        return "degraded", optional_missing.reason

    return "ready", "all declared package components are available"


def _has_ready_component(
    components: List[PackageComponentStatus],
    component_type: str,
) -> bool:
    return any(
        component.component_type == component_type and component.status == "ready"
        for component in components
    )


def _normalize_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _summarize_gateway(gateway: ResourcePackageGateway) -> Dict[str, Any]:
    if not gateway.declared:
        return {
            "gateway_declared": False,
            "gateway_ready": True,
            "gateway_status": "not_declared",
            "gateway_reason": "no external gateway declared",
            "gateway_transport": "",
            "gateway_endpoint": "",
            "gateway_tool_namespace": "",
            "gateway_missing_env": [],
            "gateway_read_only": True,
        }

    endpoint = gateway.endpoint or os.getenv(gateway.endpoint_env or "", "").strip()
    missing_auth_env = sorted(
        env_name
        for env_name in gateway.auth_env_vars
        if not str(os.getenv(env_name, "") or "").strip()
    )

    if not gateway.transport or not gateway.tool_namespace:
        return {
            "gateway_declared": True,
            "gateway_ready": False,
            "gateway_status": "misconfigured",
            "gateway_reason": "gateway transport and tool namespace are required",
            "gateway_transport": gateway.transport,
            "gateway_endpoint": endpoint,
            "gateway_tool_namespace": gateway.tool_namespace,
            "gateway_missing_env": missing_auth_env,
            "gateway_read_only": gateway.read_only,
        }
    if not endpoint:
        return {
            "gateway_declared": True,
            "gateway_ready": False,
            "gateway_status": "missing_endpoint",
            "gateway_reason": (
                f"missing gateway endpoint: set {gateway.endpoint_env}"
                if gateway.endpoint_env
                else "missing gateway endpoint"
            ),
            "gateway_transport": gateway.transport,
            "gateway_endpoint": "",
            "gateway_tool_namespace": gateway.tool_namespace,
            "gateway_missing_env": missing_auth_env,
            "gateway_read_only": gateway.read_only,
        }
    if missing_auth_env:
        return {
            "gateway_declared": True,
            "gateway_ready": False,
            "gateway_status": "missing_auth",
            "gateway_reason": "missing gateway auth environment variables",
            "gateway_transport": gateway.transport,
            "gateway_endpoint": endpoint,
            "gateway_tool_namespace": gateway.tool_namespace,
            "gateway_missing_env": missing_auth_env,
            "gateway_read_only": gateway.read_only,
        }

    return {
        "gateway_declared": True,
        "gateway_ready": True,
        "gateway_status": "ready",
        "gateway_reason": "external gateway configuration is usable",
        "gateway_transport": gateway.transport,
        "gateway_endpoint": endpoint,
        "gateway_tool_namespace": gateway.tool_namespace,
        "gateway_missing_env": [],
        "gateway_read_only": gateway.read_only,
    }
