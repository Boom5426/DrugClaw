from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


@dataclass(frozen=True)
class PackageComponentStatus:
    component_type: str
    path_or_name: str
    status: str
    reason: str
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePackageGateway:
    transport: str = ""
    endpoint: str = ""
    endpoint_env: str = ""
    tool_namespace: str = ""
    auth_env_vars: List[str] = field(default_factory=list)
    docs_url: str = ""
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport", str(self.transport or "").strip())
        object.__setattr__(self, "endpoint", str(self.endpoint or "").strip())
        object.__setattr__(self, "endpoint_env", str(self.endpoint_env or "").strip())
        object.__setattr__(self, "tool_namespace", str(self.tool_namespace or "").strip())
        object.__setattr__(self, "auth_env_vars", _dedupe_strings(list(self.auth_env_vars)))
        object.__setattr__(self, "docs_url", str(self.docs_url or "").strip())
        object.__setattr__(self, "read_only", bool(self.read_only))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ResourcePackageGateway":
        payload = dict(payload or {})
        return cls(
            transport=payload.get("transport", ""),
            endpoint=payload.get("endpoint", ""),
            endpoint_env=payload.get("endpoint_env", ""),
            tool_namespace=payload.get("tool_namespace", ""),
            auth_env_vars=list(payload.get("auth_env_vars", []) or []),
            docs_url=payload.get("docs_url", ""),
            read_only=payload.get("read_only", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def declared(self) -> bool:
        return any(
            (
                self.transport,
                self.endpoint,
                self.endpoint_env,
                self.tool_namespace,
                self.auth_env_vars,
                self.docs_url,
            )
        )


@dataclass(frozen=True)
class ResourcePackageManifest:
    package_id: str
    skill_name: str
    dataset_bundle: List[str] = field(default_factory=list)
    protocol_docs: List[str] = field(default_factory=list)
    how_to_docs: List[str] = field(default_factory=list)
    software_dependencies: List[str] = field(default_factory=list)
    knowhow_docs: List[str] = field(default_factory=list)
    gateway: ResourcePackageGateway | Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_id", str(self.package_id or "").strip())
        object.__setattr__(self, "skill_name", str(self.skill_name or "").strip())
        object.__setattr__(self, "dataset_bundle", _dedupe_strings(list(self.dataset_bundle)))
        object.__setattr__(self, "protocol_docs", _dedupe_strings(list(self.protocol_docs)))
        object.__setattr__(self, "how_to_docs", _dedupe_strings(list(self.how_to_docs)))
        object.__setattr__(self, "software_dependencies", _dedupe_strings(list(self.software_dependencies)))
        object.__setattr__(self, "knowhow_docs", _dedupe_strings(list(self.knowhow_docs)))
        gateway = self.gateway
        if isinstance(gateway, ResourcePackageGateway):
            normalized_gateway = gateway
        else:
            normalized_gateway = ResourcePackageGateway.from_dict(
                gateway if isinstance(gateway, dict) else None
            )
        object.__setattr__(self, "gateway", normalized_gateway)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResourcePackageManifest":
        return cls(
            package_id=payload.get("package_id", ""),
            skill_name=payload.get("skill_name", ""),
            dataset_bundle=list(payload.get("dataset_bundle", []) or []),
            protocol_docs=list(payload.get("protocol_docs", []) or []),
            how_to_docs=list(payload.get("how_to_docs", []) or []),
            software_dependencies=list(payload.get("software_dependencies", []) or []),
            knowhow_docs=list(payload.get("knowhow_docs", []) or []),
            gateway=payload.get("gateway"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePackageSnapshot:
    package_id: str
    skill_name: str
    status: str
    status_reason: str
    components: List[PackageComponentStatus] = field(default_factory=list)
    missing_components: List[str] = field(default_factory=list)
    has_dataset_bundle: bool = False
    has_protocol: bool = False
    has_how_to: bool = False
    has_knowhow: bool = False
    has_software_dependency: bool = False
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
        return {
            "package_id": self.package_id,
            "skill_name": self.skill_name,
            "status": self.status,
            "status_reason": self.status_reason,
            "components": [component.to_dict() for component in self.components],
            "missing_components": list(self.missing_components),
            "has_dataset_bundle": self.has_dataset_bundle,
            "has_protocol": self.has_protocol,
            "has_how_to": self.has_how_to,
            "has_knowhow": self.has_knowhow,
            "has_software_dependency": self.has_software_dependency,
            "gateway_declared": self.gateway_declared,
            "gateway_ready": self.gateway_ready,
            "gateway_status": self.gateway_status,
            "gateway_reason": self.gateway_reason,
            "gateway_transport": self.gateway_transport,
            "gateway_endpoint": self.gateway_endpoint,
            "gateway_tool_namespace": self.gateway_tool_namespace,
            "gateway_missing_env": list(self.gateway_missing_env),
            "gateway_read_only": self.gateway_read_only,
        }
