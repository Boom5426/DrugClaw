from __future__ import annotations

from pathlib import Path

import pytest

from drugclaw import cli


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


def test_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_resources_endpoint() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/resources")

    assert response.status_code == 200
    assert response.json()["total_resources"] == 1


def test_resources_endpoint_includes_details() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/resources")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["resources"], list)
    assert data["resources"][0]["name"] == "ChEMBL"


def test_query_endpoint() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.post(
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


def test_query_endpoint_uses_runtime_default_mode_when_omitted() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    runtime = _RuntimeStub(default_mode="web_only")
    client = TestClient(create_app(runtime=runtime))
    response = client.post(
        "/api/query",
        json={
            "query": "What are the known drug targets of imatinib?",
            "resource_filter": ["ChEMBL"],
        },
    )

    assert response.status_code == 200
    assert runtime.calls[0]["mode"] == "web_only"
    assert response.json()["mode"] == "web_only"


def test_query_detail_endpoint() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/api/queries/query_1")

    assert response.status_code == 200
    assert response.json()["query_id"] == "query_1"


def test_query_report_endpoint() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/api/queries/query_1/report")

    assert response.status_code == 200
    assert "# DrugClaw Query Report" in response.text


def test_root_serves_chat_page() -> None:
    from fastapi.testclient import TestClient

    from drugclaw.server_app import create_app

    client = TestClient(create_app(runtime=_RuntimeStub()))
    response = client.get("/")

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
