from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GatewayDescriptor:
    resource_name: str
    package_id: str
    category: str
    transport: str
    endpoint: str
    tool_namespace: str
    ready: bool
    status: str
    reason: str
    read_only: bool = True
    missing_env: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GatewayRegistry:
    def __init__(self, gateways: Iterable[GatewayDescriptor]):
        self._gateways = sorted(
            list(gateways),
            key=lambda gateway: (
                0 if gateway.ready else 1,
                gateway.resource_name.casefold(),
            ),
        )
        self._by_resource_name = {
            gateway.resource_name: gateway for gateway in self._gateways
        }
        self._by_tool_namespace = {
            gateway.tool_namespace: gateway
            for gateway in self._gateways
            if gateway.tool_namespace
        }

    def get_all_gateways(self) -> List[GatewayDescriptor]:
        return list(self._gateways)

    def get_gateway(
        self,
        *,
        resource_name: str = "",
        tool_namespace: str = "",
    ) -> Optional[GatewayDescriptor]:
        if resource_name:
            gateway = self._by_resource_name.get(resource_name)
            if gateway is None:
                return None
            if tool_namespace and gateway.tool_namespace != tool_namespace:
                return None
            return gateway
        if tool_namespace:
            return self._by_tool_namespace.get(tool_namespace)
        return None


def build_gateway_registry(
    resource_registry: Any,
    *,
    ready_only: bool = False,
) -> GatewayRegistry:
    get_all_resources = getattr(resource_registry, "get_all_resources", None)
    resources = get_all_resources() if callable(get_all_resources) else []
    gateways: List[GatewayDescriptor] = []
    for entry in resources:
        if not bool(getattr(entry, "gateway_declared", False)):
            continue
        if ready_only and not bool(getattr(entry, "gateway_ready", False)):
            continue
        gateways.append(
            GatewayDescriptor(
                resource_name=str(getattr(entry, "name", "") or "").strip(),
                package_id=str(getattr(entry, "package_id", "") or "").strip(),
                category=str(getattr(entry, "category", "") or "").strip(),
                transport=str(getattr(entry, "gateway_transport", "") or "").strip(),
                endpoint=str(getattr(entry, "gateway_endpoint", "") or "").strip(),
                tool_namespace=str(getattr(entry, "gateway_tool_namespace", "") or "").strip(),
                ready=bool(getattr(entry, "gateway_ready", False)),
                status=str(getattr(entry, "gateway_status", "") or "").strip(),
                reason=str(getattr(entry, "gateway_reason", "") or "").strip(),
                read_only=bool(getattr(entry, "gateway_read_only", True)),
                missing_env=list(getattr(entry, "gateway_missing_env", []) or []),
            )
        )
    return GatewayRegistry(gateways)
