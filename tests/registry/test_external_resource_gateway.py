from __future__ import annotations

from drugclaw.gateway_registry import build_gateway_registry
from drugclaw.resource_package_models import PackageComponentStatus
from drugclaw.resource_registry import ResourceEntry, ResourceRegistry


def _resource_entry(
    name: str,
    *,
    gateway_ready: bool,
    gateway_status: str,
    gateway_reason: str = "",
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
        status="ready" if gateway_ready else "degraded",
        status_reason="available" if gateway_ready else gateway_status,
        access_mode="REST_API",
        resource_type="Database",
        package_id=f"{name.lower().replace(' ', '_')}_bundle",
        package_status="ready" if gateway_ready else "degraded",
        package_components=[],
        missing_components=[],
        has_knowhow=False,
        gateway_declared=True,
        gateway_ready=gateway_ready,
        gateway_status=gateway_status,
        gateway_reason=gateway_reason,
        gateway_transport="graphql",
        gateway_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
        gateway_tool_namespace="open_targets.platform" if gateway_ready else "broken.gateway",
        gateway_read_only=True,
        gateway_missing_env=["OPEN_TARGETS_TOKEN"] if not gateway_ready else [],
    )


def test_managed_external_resource_gateway_lists_resources_and_reports_dependency_summary() -> None:
    from drugclaw.external_resource_gateway import ManagedExternalResourceGateway

    resource_registry = ResourceRegistry(
        [
            _resource_entry(
                "Open Targets Platform",
                gateway_ready=True,
                gateway_status="ready",
            ),
            _resource_entry(
                "Broken Gateway",
                gateway_ready=False,
                gateway_status="missing_auth",
                gateway_reason="missing gateway auth environment variables",
            ),
        ]
    )
    gateway = ManagedExternalResourceGateway(build_gateway_registry(resource_registry))

    resources = gateway.list_resources()

    assert [resource.resource_name for resource in resources] == [
        "Open Targets Platform",
        "Broken Gateway",
    ]
    health = gateway.healthcheck(resource_name="Broken Gateway")
    assert health.ready is False
    assert health.status == "missing_auth"
    dependency = gateway.dependency_summary(resource_name="Broken Gateway")
    assert dependency.to_package_component() == PackageComponentStatus(
        component_type="gateway_capability",
        path_or_name="broken.gateway",
        status="missing_dependency",
        reason="missing gateway auth environment variables",
        required=False,
    )


def test_managed_external_resource_gateway_invokes_through_gateway_invoker() -> None:
    from drugclaw.external_resource_gateway import ManagedExternalResourceGateway

    class _GatewayInvokerStub:
        def __init__(self):
            self.calls = []

        def invoke(self, **kwargs):
            self.calls.append(dict(kwargs))
            return {"resource_name": "Open Targets Platform", "data": {"ok": True}}

    resource_registry = ResourceRegistry(
        [
            _resource_entry(
                "Open Targets Platform",
                gateway_ready=True,
                gateway_status="ready",
            ),
        ]
    )
    invoker = _GatewayInvokerStub()
    gateway = ManagedExternalResourceGateway(
        build_gateway_registry(resource_registry),
        gateway_invoker=invoker,
    )

    result = gateway.invoke(
        tool_namespace="open_targets.platform",
        query="{ target { id } }",
        variables={"id": "ENSG000001"},
        timeout=4.5,
    )

    assert result == {"resource_name": "Open Targets Platform", "data": {"ok": True}}
    assert invoker.calls == [
        {
            "resource_name": "",
            "tool_namespace": "open_targets.platform",
            "path": "",
            "params": None,
            "query": "{ target { id } }",
            "variables": {"id": "ENSG000001"},
            "timeout": 4.5,
            "headers": None,
        }
    ]
