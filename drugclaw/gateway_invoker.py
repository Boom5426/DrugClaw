from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .gateway_registry import GatewayDescriptor, GatewayRegistry


class GatewayInvocationError(RuntimeError):
    pass


class GatewayInvoker:
    def __init__(
        self,
        gateway_registry: GatewayRegistry,
        *,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.gateway_registry = gateway_registry
        self._opener = opener or urlopen

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
        gateway = self.gateway_registry.get_gateway(
            resource_name=resource_name,
            tool_namespace=tool_namespace,
        )
        if gateway is None:
            raise GatewayInvocationError("unknown gateway")
        if not gateway.ready:
            raise GatewayInvocationError(f"gateway is not ready: {gateway.status}")
        if not gateway.read_only:
            raise GatewayInvocationError("gateway invocation requires a read-only gateway")

        if gateway.transport == "rest_api":
            return self._invoke_rest(
                gateway,
                path=path,
                params=params,
                timeout=timeout,
                headers=headers,
            )
        if gateway.transport == "graphql":
            return self._invoke_graphql(
                gateway,
                query=query,
                variables=variables,
                timeout=timeout,
                headers=headers,
            )
        raise GatewayInvocationError(
            f"gateway transport is not yet supported: {gateway.transport or 'unknown'}"
        )

    def _invoke_rest(
        self,
        gateway: GatewayDescriptor,
        *,
        path: str,
        params: Optional[Dict[str, Any]],
        timeout: float,
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        base_url = gateway.endpoint.rstrip("/") + "/"
        target_path = str(path or "").lstrip("/")
        url = urljoin(base_url, target_path)
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(
            url=url,
            headers={"Accept": "application/json", **dict(headers or {})},
            method="GET",
        )
        payload = self._read_response(request, timeout=timeout)
        return self._build_result(gateway, payload)

    def _invoke_graphql(
        self,
        gateway: GatewayDescriptor,
        *,
        query: str,
        variables: Optional[Dict[str, Any]],
        timeout: float,
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise GatewayInvocationError("graphql gateway invocation requires a query")
        if not _is_read_only_graphql(normalized_query):
            raise GatewayInvocationError("graphql gateway invocation must be read-only")

        request_payload = {
            "query": normalized_query,
            "variables": dict(variables or {}),
        }
        request = Request(
            url=gateway.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                **dict(headers or {}),
            },
            method="POST",
        )
        payload = self._read_response(request, timeout=timeout)
        return self._build_result(gateway, payload)

    def _read_response(self, request: Request, *, timeout: float) -> Any:
        with self._opener(request, timeout=timeout) as response:
            body = response.read()
            content_type = str(getattr(response, "headers", {}).get("Content-Type", "") or "")
        text = body.decode("utf-8")
        if "json" in content_type.lower():
            return json.loads(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _build_result(self, gateway: GatewayDescriptor, data: Any) -> Dict[str, Any]:
        return {
            "resource_name": gateway.resource_name,
            "package_id": gateway.package_id,
            "transport": gateway.transport,
            "tool_namespace": gateway.tool_namespace,
            "endpoint": gateway.endpoint,
            "status": gateway.status,
            "data": data,
        }


def _is_read_only_graphql(query: str) -> bool:
    normalized = query.lstrip()
    lowered = normalized.casefold()
    if lowered.startswith("mutation") or lowered.startswith("subscription"):
        return False
    return lowered.startswith("query") or lowered.startswith("{")
