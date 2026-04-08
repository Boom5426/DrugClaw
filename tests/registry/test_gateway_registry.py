from __future__ import annotations

from drugclaw.gateway_registry import build_gateway_registry
from drugclaw.resource_registry import ResourceEntry, ResourceRegistry


def _resource_entry(
    name: str,
    *,
    gateway_declared: bool,
    gateway_ready: bool,
    gateway_transport: str = "",
    gateway_tool_namespace: str = "",
    gateway_endpoint: str = "",
    gateway_status: str = "not_declared",
    gateway_reason: str = "",
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
        gateway_declared=gateway_declared,
        gateway_ready=gateway_ready,
        gateway_status=gateway_status,
        gateway_reason=gateway_reason,
        gateway_transport=gateway_transport,
        gateway_endpoint=gateway_endpoint,
        gateway_tool_namespace=gateway_tool_namespace,
        gateway_read_only=gateway_read_only,
    )


def test_build_gateway_registry_filters_to_declared_gateways_and_supports_namespace_lookup() -> None:
    resource_registry = ResourceRegistry(
        [
            _resource_entry(
                "openFDA Human Drug",
                gateway_declared=True,
                gateway_ready=True,
                gateway_status="ready",
                gateway_transport="rest_api",
                gateway_tool_namespace="openfda.human_drug",
                gateway_endpoint="https://api.fda.gov",
            ),
            _resource_entry(
                "Open Targets Platform",
                gateway_declared=True,
                gateway_ready=False,
                gateway_status="missing_auth",
                gateway_reason="missing gateway auth environment variables",
                gateway_transport="graphql",
                gateway_tool_namespace="open_targets.platform",
                gateway_endpoint="https://api.platform.opentargets.org/api/v4/graphql",
            ),
            _resource_entry(
                "BindingDB",
                gateway_declared=False,
                gateway_ready=True,
            ),
        ]
    )

    registry = build_gateway_registry(resource_registry)

    assert [gateway.resource_name for gateway in registry.get_all_gateways()] == [
        "openFDA Human Drug",
        "Open Targets Platform",
    ]
    assert registry.get_gateway(resource_name="openFDA Human Drug") is not None
    assert registry.get_gateway(tool_namespace="openfda.human_drug") is not None
    assert registry.get_gateway(resource_name="BindingDB") is None

    ready_only_registry = build_gateway_registry(resource_registry, ready_only=True)

    assert [gateway.resource_name for gateway in ready_only_registry.get_all_gateways()] == [
        "openFDA Human Drug"
    ]
    assert ready_only_registry.get_gateway(tool_namespace="open_targets.platform") is None


def test_resource_registry_summary_only_counts_declared_ready_gateways() -> None:
    resource_registry = ResourceRegistry(
        [
            _resource_entry(
                "Declared Ready",
                gateway_declared=True,
                gateway_ready=True,
                gateway_status="ready",
                gateway_transport="rest_api",
                gateway_tool_namespace="declared.ready",
                gateway_endpoint="https://example.com/ready",
            ),
            _resource_entry(
                "Undeclared",
                gateway_declared=False,
                gateway_ready=True,
            ),
        ]
    )

    summary = resource_registry.summarize_registry()

    assert summary["gateway_declared_resources"] == 1
    assert summary["gateway_ready_resources"] == 1
