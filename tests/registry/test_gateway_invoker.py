from __future__ import annotations

import json

import pytest

from drugclaw.gateway_invoker import GatewayInvocationError, GatewayInvoker
from drugclaw.gateway_registry import build_gateway_registry
from drugclaw.resource_registry import ResourceEntry, ResourceRegistry


class _FakeResponse:
    def __init__(self, body: str, *, content_type: str = "application/json") -> None:
        self._body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _resource_entry(
    name: str,
    *,
    gateway_transport: str,
    gateway_tool_namespace: str,
    gateway_endpoint: str,
    gateway_ready: bool = True,
    gateway_status: str = "ready",
    gateway_read_only: bool = True,
) -> ResourceEntry:
    return ResourceEntry(
        id=name.lower().replace(" ", "_"),
        name=name,
        category="test",
        description="fixture",
        entrypoint="tests.fixture:Skill",
        enabled=True,
        requires_metadata=False,
        required_metadata_paths=[],
        required_dependencies=[],
        supports_code_generation=True,
        fallback_retrieve_supported=True,
        status="ready",
        status_reason="available",
        access_mode="REST_API",
        resource_type="Database",
        package_id=f"{name.lower().replace(' ', '_')}_bundle",
        package_status="ready",
        package_components=[],
        missing_components=[],
        has_knowhow=False,
        gateway_declared=True,
        gateway_ready=gateway_ready,
        gateway_status=gateway_status,
        gateway_reason="" if gateway_ready else gateway_status,
        gateway_transport=gateway_transport,
        gateway_endpoint=gateway_endpoint,
        gateway_tool_namespace=gateway_tool_namespace,
        gateway_read_only=gateway_read_only,
    )


def test_gateway_invoker_rejects_unknown_gateway() -> None:
    registry = build_gateway_registry(ResourceRegistry([]))
    invoker = GatewayInvoker(registry)

    with pytest.raises(GatewayInvocationError, match="unknown gateway"):
        invoker.invoke(resource_name="MissingGateway")


def test_gateway_invoker_rejects_non_ready_gateways() -> None:
    registry = build_gateway_registry(
        ResourceRegistry(
            [
                _resource_entry(
                    "Open Targets Platform",
                    gateway_transport="graphql",
                    gateway_tool_namespace="open_targets.platform",
                    gateway_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
                    gateway_ready=False,
                    gateway_status="missing_auth",
                )
            ]
        )
    )
    invoker = GatewayInvoker(registry)

    with pytest.raises(GatewayInvocationError, match="not ready"):
        invoker.invoke(tool_namespace="open_targets.platform", query="{ target { id } }")


def test_gateway_invoker_rejects_non_read_only_gateways() -> None:
    registry = build_gateway_registry(
        ResourceRegistry(
            [
                _resource_entry(
                    "Write Enabled Skill",
                    gateway_transport="rest_api",
                    gateway_tool_namespace="write.enabled",
                    gateway_endpoint="https://example.com/api",
                    gateway_read_only=False,
                )
            ]
        )
    )
    invoker = GatewayInvoker(registry)

    with pytest.raises(GatewayInvocationError, match="read-only"):
        invoker.invoke(resource_name="Write Enabled Skill", path="/records")


def test_gateway_invoker_executes_rest_get_and_returns_normalized_payload() -> None:
    captured: dict[str, str] = {}

    def _fake_opener(request, timeout=0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        return _FakeResponse('{"results": [{"id": "drug-1"}]}')

    registry = build_gateway_registry(
        ResourceRegistry(
            [
                _resource_entry(
                    "openFDA Human Drug",
                    gateway_transport="rest_api",
                    gateway_tool_namespace="openfda.human_drug",
                    gateway_endpoint="https://api.fda.gov",
                )
            ]
        )
    )
    invoker = GatewayInvoker(registry, opener=_fake_opener)

    result = invoker.invoke(
        resource_name="openFDA Human Drug",
        path="/drug/label.json",
        params={"search": "set_id:123", "limit": 1},
    )

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.fda.gov/drug/label.json?search=set_id%3A123&limit=1"
    assert result["resource_name"] == "openFDA Human Drug"
    assert result["tool_namespace"] == "openfda.human_drug"
    assert result["transport"] == "rest_api"
    assert result["data"] == {"results": [{"id": "drug-1"}]}


def test_gateway_invoker_rejects_graphql_mutations() -> None:
    registry = build_gateway_registry(
        ResourceRegistry(
            [
                _resource_entry(
                    "Open Targets Platform",
                    gateway_transport="graphql",
                    gateway_tool_namespace="open_targets.platform",
                    gateway_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
                )
            ]
        )
    )
    invoker = GatewayInvoker(registry)

    with pytest.raises(GatewayInvocationError, match="read-only"):
        invoker.invoke(
            tool_namespace="open_targets.platform",
            query="mutation UpdateTarget { updateTarget(id: \"ENSG000001\") { id } }",
        )


def test_gateway_invoker_executes_graphql_query_post_and_returns_normalized_payload() -> None:
    captured: dict[str, str] = {}

    def _fake_opener(request, timeout=0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = (request.data or b"").decode("utf-8")
        return _FakeResponse(json.dumps({"data": {"target": {"id": "ENSG000001"}}}))

    registry = build_gateway_registry(
        ResourceRegistry(
            [
                _resource_entry(
                    "Open Targets Platform",
                    gateway_transport="graphql",
                    gateway_tool_namespace="open_targets.platform",
                    gateway_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
                )
            ]
        )
    )
    invoker = GatewayInvoker(registry, opener=_fake_opener)

    result = invoker.invoke(
        tool_namespace="open_targets.platform",
        query="query TargetLookup($id: String!) { target(ensemblId: $id) { id } }",
        variables={"id": "ENSG000001"},
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.platform.opentargets.org/api/v4/graphql"
    assert json.loads(captured["body"]) == {
        "query": "query TargetLookup($id: String!) { target(ensemblId: $id) { id } }",
        "variables": {"id": "ENSG000001"},
    }
    assert result["resource_name"] == "Open Targets Platform"
    assert result["transport"] == "graphql"
    assert result["data"] == {"data": {"target": {"id": "ENSG000001"}}}
