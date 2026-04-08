from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from drugclaw import cli
from drugclaw.server_app import create_app


_REQUEST_TIMEOUT_SECONDS = 1.0


def _make_client(runtime: object) -> httpx.AsyncClient:
    app = create_app(runtime=runtime)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    **kwargs,
) -> httpx.Response:
    request_fn = getattr(client, method)
    return await asyncio.wait_for(
        request_fn(path, **kwargs),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def test_cli_parser_supports_serve_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "8000"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_query_request_validation() -> None:
    from drugclaw.server_models import QueryRequest

    with pytest.raises(Exception):
        QueryRequest(query="", mode="simple")

    with pytest.raises(Exception):
        QueryRequest(query="x" * 5001, mode="simple")

    with pytest.raises(Exception):
        QueryRequest(query="What does imatinib target?", mode="bad-mode")


def test_query_request_defaults_mode_to_none() -> None:
    from drugclaw.server_models import QueryRequest

    request = QueryRequest(query="What does imatinib target?")

    assert request.mode is None


def test_gateway_invoke_request_validation() -> None:
    from drugclaw.server_models import GatewayInvokeRequest

    with pytest.raises(Exception):
        GatewayInvokeRequest(resource_name="", tool_namespace="")

    with pytest.raises(Exception):
        GatewayInvokeRequest(tool_namespace="open_targets.platform", timeout_seconds=0)

    request = GatewayInvokeRequest(
        tool_namespace="open_targets.platform",
        query="{ target { id } }",
        variables={"id": "ENSG000001"},
    )

    assert request.tool_namespace == "open_targets.platform"
    assert request.resource_name is None
    assert request.variables == {"id": "ENSG000001"}


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_resources_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/resources")

    assert response.status_code == 200
    assert response.json()["total_resources"] == 1


@pytest.mark.anyio
async def test_resources_endpoint_includes_details() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/resources")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["resources"], list)
    assert data["resources"][0]["name"] == "ChEMBL"


@pytest.mark.anyio
async def test_query_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(
            client,
            "post",
            "/api/query",
            json={
                "query": "What are the known drug targets of imatinib?",
                "mode": "simple",
                "resource_filter": ["ChEMBL"],
                "save_md_report": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["query_id"] == "query_1"
    assert body["normalized_query"] == "What are the known drug targets of imatinib?"
    assert body["resource_filter"] == ["ChEMBL"]
    assert body["save_md_report"] is True


@pytest.mark.anyio
async def test_query_endpoint_uses_runtime_default_mode_when_omitted() -> None:
    runtime = _RuntimeStub(default_mode="web_only")
    async with _make_client(runtime) as client:
        response = await _request(
            client,
            "post",
            "/api/query",
            json={
                "query": "What are the known drug targets of imatinib?",
                "resource_filter": ["ChEMBL"],
            },
        )

    assert response.status_code == 200
    assert runtime.calls[0]["mode"] == "web_only"
    assert response.json()["mode"] == "web_only"


@pytest.mark.anyio
async def test_gateway_invoke_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(
            client,
            "post",
            "/api/gateways/invoke",
            json={
                "tool_namespace": "open_targets.platform",
                "query": "{ target { id } }",
                "variables": {"id": "ENSG000001"},
                "timeout_seconds": 4.5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["resource_name"] == "Open Targets Platform"
    assert body["tool_namespace"] == "open_targets.platform"
    assert body["data"] == {"data": {"target": {"id": "ENSG000001"}}}


@pytest.mark.anyio
async def test_gateway_invoke_endpoint_returns_bad_request_for_runtime_validation_errors() -> None:
    runtime = _RuntimeStub()
    runtime.gateway_error = "unknown gateway"
    async with _make_client(runtime) as client:
        response = await _request(
            client,
            "post",
            "/api/gateways/invoke",
            json={"resource_name": "Missing Gateway"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "unknown gateway"


@pytest.mark.anyio
async def test_query_detail_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/api/queries/query_1")

    assert response.status_code == 200
    assert response.json()["query_id"] == "query_1"


@pytest.mark.anyio
async def test_query_report_endpoint() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/api/queries/query_1/report")

    assert response.status_code == 200
    assert "# DrugClaw Query Report" in response.text


@pytest.mark.anyio
async def test_root_serves_chat_page() -> None:
    async with _make_client(_RuntimeStub()) as client:
        response = await _request(client, "get", "/")

    assert response.status_code == 200
    assert "DrugClaw" in response.text


class _RuntimeStub:
    def __init__(self, default_mode: str = "simple"):
        self.config = type(
            "_Config",
            (),
            {"SERVER_DEFAULT_MODE": default_mode},
        )()
        self.calls = []
        self.gateway_calls = []
        self.gateway_error = ""

    def health(self):
        return {
            "status": "ok",
            "model": "test-model",
            "default_mode": self.config.SERVER_DEFAULT_MODE,
            "active_requests": 0,
        }

    def resources(self):
        return {
            "total_resources": 1,
            "enabled_resources": 1,
            "resources": [{"name": "ChEMBL", "status": "ready"}],
        }

    def run_query(self, query, mode, resource_filter, save_md_report):
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "resource_filter": resource_filter,
                "save_md_report": save_md_report,
            }
        )
        return {
            "success": True,
            "query": query,
            "normalized_query": query,
            "answer": "ok",
            "query_id": "query_1",
            "mode": mode,
            "resource_filter": resource_filter,
            "save_md_report": save_md_report,
        }

    def get_query(self, query_id):
        return {"query_id": query_id, "query": "demo"}

    def get_query_report(self, query_id):
        return "# DrugClaw Query Report\n\nreport"

    def invoke_gateway(
        self,
        *,
        resource_name="",
        tool_namespace="",
        path="",
        params=None,
        query="",
        variables=None,
        timeout_seconds=10.0,
    ):
        if self.gateway_error:
            raise ValueError(self.gateway_error)
        self.gateway_calls.append(
            {
                "resource_name": resource_name,
                "tool_namespace": tool_namespace,
                "path": path,
                "params": params,
                "query": query,
                "variables": variables,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "resource_name": "Open Targets Platform",
            "tool_namespace": tool_namespace,
            "transport": "graphql",
            "data": {"data": {"target": {"id": variables["id"]}}},
        }
