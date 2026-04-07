from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol

from .gateway_invoker import GatewayInvocationError, GatewayInvoker
from .gateway_registry import GatewayDescriptor, GatewayRegistry, build_gateway_registry
from .resource_package_models import PackageComponentStatus


class ExternalResourceGatewayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalGatewayResource:
    resource_name: str
    package_id: str
    category: str
    transport: str
    endpoint: str
    tool_namespace: str
    ready: bool
    status: str
    reason: str
    read_only: bool
    missing_env: List[str]

    @classmethod
    def from_descriptor(cls, gateway: GatewayDescriptor) -> "ExternalGatewayResource":
        return cls(
            resource_name=gateway.resource_name,
            package_id=gateway.package_id,
            category=gateway.category,
            transport=gateway.transport,
            endpoint=gateway.endpoint,
            tool_namespace=gateway.tool_namespace,
            ready=gateway.ready,
            status=gateway.status,
            reason=gateway.reason,
            read_only=gateway.read_only,
            missing_env=list(gateway.missing_env),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalGatewayHealth:
    resource_name: str
    package_id: str
    tool_namespace: str
    ready: bool
    status: str
    reason: str
    missing_env: List[str]

    @classmethod
    def from_descriptor(cls, gateway: GatewayDescriptor) -> "ExternalGatewayHealth":
        return cls(
            resource_name=gateway.resource_name,
            package_id=gateway.package_id,
            tool_namespace=gateway.tool_namespace,
            ready=gateway.ready,
            status=gateway.status,
            reason=gateway.reason,
            missing_env=list(gateway.missing_env),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalGatewayDependencySummary:
    resource_name: str
    package_id: str
    transport: str
    tool_namespace: str
    endpoint: str
    ready: bool
    status: str
    reason: str
    missing_env: List[str]
    read_only: bool

    @classmethod
    def from_descriptor(cls, gateway: GatewayDescriptor) -> "ExternalGatewayDependencySummary":
        return cls(
            resource_name=gateway.resource_name,
            package_id=gateway.package_id,
            transport=gateway.transport,
            tool_namespace=gateway.tool_namespace,
            endpoint=gateway.endpoint,
            ready=gateway.ready,
            status=gateway.status,
            reason=gateway.reason,
            missing_env=list(gateway.missing_env),
            read_only=gateway.read_only,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_package_component(self) -> PackageComponentStatus:
        component_status = "ready" if self.ready else "missing_dependency"
        reason = self.reason or (
            "external gateway configuration is usable"
            if self.ready
            else f"gateway is not ready: {self.status or 'unknown'}"
        )
        return PackageComponentStatus(
            component_type="gateway_capability",
            path_or_name=self.tool_namespace or self.resource_name,
            status=component_status,
            reason=reason,
            required=False,
        )


class ExternalResourceGateway(Protocol):
    def list_resources(self, *, ready_only: bool = False) -> List[ExternalGatewayResource]:
        ...

    def healthcheck(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> ExternalGatewayHealth:
        ...

    def dependency_summary(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> ExternalGatewayDependencySummary:
        ...

    def invoke(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
        path: str = "",
        params: Optional[Dict[str, Any]] = None,
        query: str = "",
        variables: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ...


class ManagedExternalResourceGateway:
    def __init__(
        self,
        gateway_registry: GatewayRegistry,
        *,
        gateway_invoker: Optional[GatewayInvoker] = None,
    ) -> None:
        self.gateway_registry = gateway_registry
        self.gateway_invoker = gateway_invoker or GatewayInvoker(gateway_registry)

    @classmethod
    def from_resource_registry(
        cls,
        resource_registry: Any,
        *,
        opener: Optional[Any] = None,
        ready_only: bool = False,
    ) -> "ManagedExternalResourceGateway":
        gateway_registry = build_gateway_registry(resource_registry, ready_only=ready_only)
        return cls(
            gateway_registry,
            gateway_invoker=GatewayInvoker(gateway_registry, opener=opener),
        )

    def list_resources(self, *, ready_only: bool = False) -> List[ExternalGatewayResource]:
        resources = [
            ExternalGatewayResource.from_descriptor(gateway)
            for gateway in self.gateway_registry.get_all_gateways()
        ]
        if ready_only:
            return [resource for resource in resources if resource.ready]
        return resources

    def healthcheck(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> ExternalGatewayHealth:
        gateway = self._require_gateway(resource_name=resource_name, tool_namespace=tool_namespace)
        return ExternalGatewayHealth.from_descriptor(gateway)

    def dependency_summary(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> ExternalGatewayDependencySummary:
        gateway = self._require_gateway(resource_name=resource_name, tool_namespace=tool_namespace)
        return ExternalGatewayDependencySummary.from_descriptor(gateway)

    def invoke(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
        path: str = "",
        params: Optional[Dict[str, Any]] = None,
        query: str = "",
        variables: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        try:
            return self.gateway_invoker.invoke(
                resource_name=resource_name,
                tool_namespace=tool_namespace,
                path=path,
                params=params,
                query=query,
                variables=variables,
                timeout=timeout,
                headers=headers,
            )
        except GatewayInvocationError as exc:
            raise ExternalResourceGatewayError(str(exc)) from exc

    def _require_gateway(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> GatewayDescriptor:
        gateway = self.gateway_registry.get_gateway(
            resource_name=resource_name,
            tool_namespace=tool_namespace,
        )
        if gateway is None:
            raise ExternalResourceGatewayError("unknown gateway")
        return gateway
