from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .skills.registry import SkillRegistry
from .skills.skill_tree import SkillTree
from .resource_package_models import ResourcePackageSnapshot
from .resource_package_registry import (
    attach_package_snapshots,
    build_resource_package_snapshots,
)
from .resource_path_resolver import collect_required_metadata_paths, is_path_key


RESOURCE_STATUSES = (
    "ready",
    "missing_metadata",
    "missing_dependency",
    "degraded",
    "disabled",
)


@dataclass(frozen=True)
class ResourceEntry:
    id: str
    name: str
    category: str
    description: str
    entrypoint: str
    enabled: bool
    requires_metadata: bool
    required_metadata_paths: List[str]
    required_dependencies: List[str]
    supports_code_generation: bool
    fallback_retrieve_supported: bool
    status: str
    status_reason: str
    access_mode: str = "REST_API"
    resource_type: str = "unknown"
    package_id: str = ""
    package_status: str = "ready"
    package_components: List[Dict[str, Any]] = field(default_factory=list)
    missing_components: List[str] = field(default_factory=list)
    has_knowhow: bool = False
    gateway_declared: bool = False
    gateway_ready: bool = True
    gateway_status: str = "not_declared"
    gateway_reason: str = ""
    gateway_transport: str = ""
    gateway_endpoint: str = ""
    gateway_tool_namespace: str = ""
    gateway_missing_env: List[str] = field(default_factory=list)
    gateway_read_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceRegistry:
    def __init__(self, entries: Iterable[ResourceEntry]):
        self._entries = sorted(entries, key=lambda entry: (entry.category, entry.name))
        self._entries_by_name = {entry.name: entry for entry in self._entries}

    def get_all_resources(self) -> List[ResourceEntry]:
        return list(self._entries)

    def get_enabled_resources(self) -> List[ResourceEntry]:
        return [entry for entry in self._entries if entry.enabled]

    def get_resource(self, name: str) -> Optional[ResourceEntry]:
        return self._entries_by_name.get(name)

    def prioritize_resource_names(
        self,
        names: Iterable[str],
        *,
        ready_only: bool = False,
    ) -> List[str]:
        seen = set()
        ranked: List[tuple[int, int, int, str]] = []
        for index, name in enumerate(names):
            if name in seen:
                continue
            seen.add(name)
            entry = self.get_resource(name)
            if entry is None:
                continue
            if ready_only and entry.status != "ready":
                continue
            ranked.append(
                (
                    _status_priority(entry.status),
                    _access_priority(entry.access_mode),
                    index,
                    entry.name,
                )
            )
        return [name for _, _, _, name in sorted(ranked)]

    def summarize_registry(self) -> Dict[str, Any]:
        status_counts = {status: 0 for status in RESOURCE_STATUSES}
        package_status_counts = {status: 0 for status in RESOURCE_STATUSES}
        category_counts: Dict[str, int] = {}
        missing_component_counts: Dict[str, int] = {}
        resources_with_knowhow = 0
        gateway_declared_resources = 0
        gateway_ready_resources = 0
        for entry in self._entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
            package_status_counts[entry.package_status] = (
                package_status_counts.get(entry.package_status, 0) + 1
            )
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            if entry.has_knowhow:
                resources_with_knowhow += 1
            if entry.gateway_declared:
                gateway_declared_resources += 1
            if entry.gateway_declared and entry.gateway_ready:
                gateway_ready_resources += 1
            for component_name in entry.missing_components:
                missing_component_counts[component_name] = (
                    missing_component_counts.get(component_name, 0) + 1
                )

        return {
            "total_resources": len(self._entries),
            "enabled_resources": len(self.get_enabled_resources()),
            "status_counts": status_counts,
            "category_counts": dict(sorted(category_counts.items())),
            "package_status_counts": package_status_counts,
            "resources_with_knowhow": resources_with_knowhow,
            "gateway_declared_resources": gateway_declared_resources,
            "gateway_ready_resources": gateway_ready_resources,
            "missing_component_counts": dict(sorted(missing_component_counts.items())),
        }


def build_resource_registry(
    skill_registry: SkillRegistry,
    *,
    repo_root: Path | None = None,
) -> ResourceRegistry:
    tree = getattr(skill_registry, "skill_tree", None) or SkillTree()
    category_by_name: Dict[str, str] = {}
    node_by_name: Dict[str, Any] = {}
    for subcategory in tree.subcategories:
        for node in subcategory.skills:
            category_by_name[node.name] = subcategory.key
            node_by_name[node.name] = node

    runtime_skills = {
        skill.name: skill for skill in skill_registry.get_registered_skills()
    }
    names = sorted(set(node_by_name) | set(runtime_skills))
    package_snapshots = build_resource_package_snapshots(names, repo_root=repo_root)
    attach_package_snapshots(runtime_skills.values(), package_snapshots)
    entries = [
        _build_resource_entry(
            name=name,
            category=category_by_name.get(name, getattr(runtime_skills.get(name), "subcategory", "unknown")),
            node=node_by_name.get(name),
            skill=runtime_skills.get(name),
            repo_root=repo_root,
            package_snapshot=package_snapshots.get(name),
        )
        for name in names
    ]
    return ResourceRegistry(entries)


def _build_resource_entry(
    *,
    name: str,
    category: str,
    node: Any,
    skill: Any,
    repo_root: Path | None,
    package_snapshot: ResourcePackageSnapshot | None,
) -> ResourceEntry:
    description = ""
    entrypoint = ""
    access_mode = "REST_API"
    resource_type = "unknown"
    required_metadata_paths: List[str] = []
    required_dependencies: List[str] = []

    if skill is not None:
        description = getattr(skill, "aim", "") or getattr(node, "aim", "")
        entrypoint = f"{skill.__class__.__module__}:{skill.__class__.__name__}"
        access_mode = getattr(skill, "access_mode", access_mode)
        resource_type = getattr(skill, "resource_type", resource_type)
        required_metadata_paths = _infer_required_metadata_paths(skill, repo_root=repo_root)
        required_dependencies = _infer_required_dependencies(skill)
    elif node is not None:
        description = getattr(node, "aim", "")
        access_mode = getattr(node, "access_mode", access_mode)

    requires_metadata = access_mode in {"LOCAL_FILE", "DATASET"} or bool(required_metadata_paths)
    enabled = skill is not None
    fallback_retrieve_supported = bool(skill is not None and hasattr(skill, "retrieve"))
    supports_code_generation = bool(enabled and name != "WebSearch")

    status, status_reason = _determine_status(
        enabled=enabled,
        skill=skill,
        access_mode=access_mode,
        requires_metadata=requires_metadata,
        required_metadata_paths=required_metadata_paths,
        required_dependencies=required_dependencies,
    )
    package_snapshot = package_snapshot or ResourcePackageSnapshot(
        package_id=_resource_id(name),
        skill_name=name,
        status="ready",
        status_reason="no resource package manifest; using legacy skill-only package view",
    )
    status, status_reason = _merge_resource_and_package_status(
        resource_status=status,
        resource_reason=status_reason,
        package_snapshot=package_snapshot,
    )

    return ResourceEntry(
        id=_resource_id(name),
        name=name,
        category=category,
        description=description,
        entrypoint=entrypoint,
        enabled=enabled,
        requires_metadata=requires_metadata,
        required_metadata_paths=required_metadata_paths,
        required_dependencies=required_dependencies,
        supports_code_generation=supports_code_generation,
        fallback_retrieve_supported=fallback_retrieve_supported,
        status=status,
        status_reason=status_reason,
        access_mode=access_mode,
        resource_type=resource_type,
        package_id=package_snapshot.package_id,
        package_status=package_snapshot.status,
        package_components=[component.to_dict() for component in package_snapshot.components],
        missing_components=list(package_snapshot.missing_components),
        has_knowhow=package_snapshot.has_knowhow,
        gateway_declared=package_snapshot.gateway_declared,
        gateway_ready=package_snapshot.gateway_ready,
        gateway_status=package_snapshot.gateway_status,
        gateway_reason=package_snapshot.gateway_reason,
        gateway_transport=package_snapshot.gateway_transport,
        gateway_endpoint=package_snapshot.gateway_endpoint,
        gateway_tool_namespace=package_snapshot.gateway_tool_namespace,
        gateway_missing_env=list(package_snapshot.gateway_missing_env),
        gateway_read_only=package_snapshot.gateway_read_only,
    )


def _determine_status(
    *,
    enabled: bool,
    skill: Any,
    access_mode: str,
    requires_metadata: bool,
    required_metadata_paths: List[str],
    required_dependencies: List[str],
) -> tuple[str, str]:
    if not enabled:
        return "disabled", "not enabled in the runtime skill registry"

    try:
        available = bool(skill.is_available())
    except Exception as exc:
        return "degraded", f"availability check raised {type(exc).__name__}: {exc}"

    if available:
        return "ready", "available in the current environment"

    if requires_metadata:
        if not required_metadata_paths:
            return "missing_metadata", "requires local metadata but no metadata path is configured"
        missing = [path for path in required_metadata_paths if not Path(path).expanduser().exists()]
        if missing:
            missing_preview = ", ".join(missing[:3])
            return "missing_metadata", f"missing local metadata: {missing_preview}"

    if required_dependencies:
        missing_deps = [
            dependency for dependency in required_dependencies
            if find_spec(dependency.replace("-", "_")) is None
        ]
        if missing_deps:
            return "missing_dependency", f"missing dependency: {', '.join(missing_deps)}"

    if access_mode == "CLI":
        return "missing_dependency", "CLI-backed resource is unavailable in the current environment"

    return "degraded", "registered but not currently usable"


def _merge_resource_and_package_status(
    *,
    resource_status: str,
    resource_reason: str,
    package_snapshot: ResourcePackageSnapshot,
) -> tuple[str, str]:
    if resource_status == "disabled":
        return resource_status, resource_reason

    if _status_priority(package_snapshot.status) > _status_priority(resource_status):
        return package_snapshot.status, package_snapshot.status_reason

    return resource_status, resource_reason


def _infer_required_metadata_paths(
    skill: Any,
    *,
    repo_root: Path | None = None,
) -> List[str]:
    return collect_required_metadata_paths(
        getattr(skill, "name", ""),
        getattr(skill, "config", {}) or {},
        repo_root=repo_root,
    )


def _infer_required_dependencies(skill: Any) -> List[str]:
    cli_package_name = getattr(skill, "cli_package_name", "") or ""
    return [cli_package_name] if cli_package_name else []


def _looks_like_path_key(key: str) -> bool:
    return is_path_key(key)


def _resource_id(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _status_priority(status: str) -> int:
    order = {
        "ready": 0,
        "degraded": 1,
        "missing_dependency": 2,
        "missing_metadata": 3,
        "disabled": 4,
    }
    return order.get(status, 99)


def _access_priority(access_mode: str) -> int:
    order = {
        "REST_API": 0,
        "CLI": 0,
        "LOCAL_FILE": 1,
        "DATASET": 2,
    }
    return order.get(access_mode, 3)
