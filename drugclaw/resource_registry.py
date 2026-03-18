from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .skills.registry import SkillRegistry
from .skills.skill_tree import SkillTree


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceRegistry:
    def __init__(self, entries: Iterable[ResourceEntry]):
        self._entries = sorted(entries, key=lambda entry: (entry.category, entry.name))

    def get_all_resources(self) -> List[ResourceEntry]:
        return list(self._entries)

    def get_enabled_resources(self) -> List[ResourceEntry]:
        return [entry for entry in self._entries if entry.enabled]

    def summarize_registry(self) -> Dict[str, Any]:
        status_counts = {status: 0 for status in RESOURCE_STATUSES}
        category_counts: Dict[str, int] = {}
        for entry in self._entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1

        return {
            "total_resources": len(self._entries),
            "enabled_resources": len(self.get_enabled_resources()),
            "status_counts": status_counts,
            "category_counts": dict(sorted(category_counts.items())),
        }


def build_resource_registry(skill_registry: SkillRegistry) -> ResourceRegistry:
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
    entries = [
        _build_resource_entry(
            name=name,
            category=category_by_name.get(name, getattr(runtime_skills.get(name), "subcategory", "unknown")),
            node=node_by_name.get(name),
            skill=runtime_skills.get(name),
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
        required_metadata_paths = _infer_required_metadata_paths(skill)
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


def _infer_required_metadata_paths(skill: Any) -> List[str]:
    paths: List[str] = []
    config = getattr(skill, "config", {}) or {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        key_lower = key.lower()
        if isinstance(value, str) and value.strip() and _looks_like_path_key(key_lower):
            paths.append(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if (
                    isinstance(nested_key, str)
                    and isinstance(nested_value, str)
                    and nested_value.strip()
                    and _looks_like_path_key(nested_key.lower())
                ):
                    paths.append(nested_value)
    return sorted(dict.fromkeys(paths))


def _infer_required_dependencies(skill: Any) -> List[str]:
    cli_package_name = getattr(skill, "cli_package_name", "") or ""
    return [cli_package_name] if cli_package_name else []


def _looks_like_path_key(key: str) -> bool:
    return key.endswith(("_path", "_csv", "_tsv", "_json", "_xml", "_graphml", "_dir", "_folder"))


def _resource_id(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")
