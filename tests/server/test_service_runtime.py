from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from drugclaw.config import Config
from drugclaw.models import ThinkingMode


def _write_key_file(path: Path, **overrides) -> None:
    payload = {
        "api_key": "",
        "base_url": "https://example.com/v1",
        "model": "test-model",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_rejects_empty_query(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=object(),
    )

    with pytest.raises(ValueError, match="query"):
        runtime.validate_request(
            query="",
            mode=ThinkingMode.SIMPLE.value,
            resource_filter=[],
        )

def test_config_exposes_server_defaults(tmp_path: Path) -> None:
    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)

    config = Config(key_file=str(key_file))

    assert config.SERVER_HOST == "127.0.0.1"
    assert config.SERVER_PORT == 8000
    assert config.SERVER_MAX_CONCURRENCY == 1
    assert config.SERVER_QUERY_TIMEOUT_SECONDS == 120
    assert config.SERVER_DEFAULT_MODE == ThinkingMode.SIMPLE.value
    assert config.SERVER_ENABLE_GRAPH_MODE is False


def test_runtime_rejects_invalid_mode(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=object(),
    )

    with pytest.raises(ValueError, match="mode"):
        runtime.validate_request(
            query="What are the known drug targets of imatinib?",
            mode="not-a-real-mode",
            resource_filter=[],
        )


def test_runtime_rejects_graph_mode_when_disabled(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=object(),
    )

    with pytest.raises(ValueError, match="graph"):
        runtime.validate_request(
            query="Explain the mechanism of imatinib.",
            mode=ThinkingMode.GRAPH.value,
            resource_filter=[],
        )


def test_runtime_rejects_overlong_query(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=object(),
    )

    with pytest.raises(ValueError, match="too long"):
        runtime.validate_request(
            query="x" * 5001,
            mode=ThinkingMode.SIMPLE.value,
            resource_filter=[],
        )


def test_runtime_rejects_unknown_resource_filter_entries(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    class _RegistryStub:
        def get_resource(self, name):
            if name == "ChEMBL":
                return object()
            return None

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=_RegistryStub(),
    )

    with pytest.raises(ValueError, match="resource_filter"):
        runtime.validate_request(
            query="What are the known drug targets of imatinib?",
            mode=ThinkingMode.SIMPLE.value,
            resource_filter=["MissingSkill"],
        )


def test_runtime_runs_query_and_returns_result(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    class _SystemStub:
        def __init__(self):
            self.calls = []

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            self.calls.append(
                {
                    "query": query,
                    "thinking_mode": thinking_mode,
                    "resource_filter": resource_filter,
                    "verbose": verbose,
                    "save_md_report": save_md_report,
                }
            )
            return {
                "success": True,
                "query": query,
                "normalized_query": query,
                "answer": "ok",
                "query_id": "query_1",
            }

    class _RegistryStub:
        def get_resource(self, name):
            return object()

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    system = _SystemStub()
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=system,
        resource_registry=_RegistryStub(),
    )

    result = runtime.run_query(
        query="What are the known drug targets of imatinib?",
        mode=ThinkingMode.SIMPLE.value,
        resource_filter=["ChEMBL"],
        save_md_report=True,
    )

    assert result["success"] is True
    assert result["query_id"] == "query_1"
    assert system.calls[0]["verbose"] is False
    assert system.calls[0]["save_md_report"] is True


def test_runtime_releases_semaphore_after_run_query(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    class _SystemStub:
        def __init__(self):
            self.calls = []

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            self.calls.append(
                {
                    "query": query,
                    "thinking_mode": thinking_mode,
                    "resource_filter": resource_filter,
                    "verbose": verbose,
                    "save_md_report": save_md_report,
                }
            )
            return {
                "success": True,
                "query": query,
                "normalized_query": query,
                "answer": "ok",
                "query_id": "query_2",
            }

    class _RegistryStub:
        def get_resource(self, name):
            return object()

    class _SemaphoreStub:
        def __init__(self):
            self.acquired = False
            self.released = False

        def acquire(self, blocking=True):
            self.acquired = True
            return True

        def release(self):
            self.released = True

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=_SystemStub(),
        resource_registry=_RegistryStub(),
    )
    semaphore = _SemaphoreStub()
    runtime._semaphore = semaphore

    runtime.run_query(
        query="How does imatinib work?",
        mode=ThinkingMode.SIMPLE.value,
        resource_filter=["ChEMBL"],
        save_md_report=False,
    )

    assert semaphore.acquired is True
    assert semaphore.released is True


def test_runtime_reports_busy_when_concurrency_limit_is_reached(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=object(),
        resource_registry=object(),
    )

    acquired = runtime._semaphore.acquire(blocking=False)
    assert acquired is True
    try:
        with pytest.raises(RuntimeError, match="busy"):
            runtime._acquire_slot()
    finally:
        runtime._semaphore.release()


def test_runtime_keeps_slot_occupied_until_timed_out_query_finishes(tmp_path: Path) -> None:
    from drugclaw.service_runtime import DrugClawServiceRuntime

    class _BlockingSystemStub:
        def __init__(self):
            self.started = threading.Event()
            self.finish = threading.Event()

        def query(
            self,
            query,
            thinking_mode,
            resource_filter,
            verbose=True,
            save_md_report=False,
        ):
            self.started.set()
            self.finish.wait(timeout=5)
            return {
                "success": True,
                "query": query,
                "normalized_query": query,
                "answer": "ok",
                "query_id": "query_timeout",
            }

    class _RegistryStub:
        def get_resource(self, name):
            return object()

    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(
        key_file,
        server_query_timeout_seconds=1,
        server_max_concurrency=1,
    )
    system = _BlockingSystemStub()
    runtime = DrugClawServiceRuntime(
        config=Config(key_file=str(key_file)),
        system=system,
        resource_registry=_RegistryStub(),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        runtime.run_query(
            query="What are the known drug targets of imatinib?",
            mode=ThinkingMode.SIMPLE.value,
            resource_filter=["ChEMBL"],
            save_md_report=False,
        )

    assert system.started.wait(timeout=1) is True
    assert runtime.health()["active_requests"] == 1

    with pytest.raises(RuntimeError, match="busy"):
        runtime.run_query(
            query="What are the known drug targets of imatinib?",
            mode=ThinkingMode.SIMPLE.value,
            resource_filter=["ChEMBL"],
            save_md_report=False,
        )

    system.finish.set()

    deadline = time.time() + 2
    while runtime.health()["active_requests"] != 0 and time.time() < deadline:
        time.sleep(0.01)

    assert runtime.health()["active_requests"] == 0

    result = runtime.run_query(
        query="What are the known drug targets of imatinib?",
        mode=ThinkingMode.SIMPLE.value,
        resource_filter=["ChEMBL"],
        save_md_report=False,
    )

    assert result["success"] is True
