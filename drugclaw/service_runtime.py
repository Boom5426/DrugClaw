from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
from typing import Any, Optional

from .config import Config
from .main_system import DrugClawSystem
from .models import ThinkingMode


class DrugClawServiceRuntime:
    def __init__(
        self,
        *,
        config: Config,
        system: Optional[Any] = None,
        resource_registry: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.system = system or DrugClawSystem(config)
        self.resource_registry = resource_registry or getattr(
            self.system,
            "resource_registry",
            None,
        )
        self._max_concurrency = max(1, int(config.SERVER_MAX_CONCURRENCY))
        self._timeout_seconds = max(1, int(config.SERVER_QUERY_TIMEOUT_SECONDS))
        self._max_query_chars = max(1, int(getattr(config, "SERVER_MAX_QUERY_CHARS", 5000)))
        self._semaphore = threading.Semaphore(self._max_concurrency)
        self._active_requests = 0
        self._active_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)

    @classmethod
    def from_key_file(cls, key_file: str = "navigator_api_keys.json") -> "DrugClawServiceRuntime":
        return cls(config=Config(key_file=key_file))

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.MODEL_NAME,
            "default_mode": self.config.SERVER_DEFAULT_MODE,
            "active_requests": self._active_requests,
        }

    def resources(self) -> dict[str, Any]:
        if self.resource_registry is None:
            return {
                "total_resources": 0,
                "enabled_resources": 0,
                "resources": [],
            }

        summary = self.resource_registry.summarize_registry()
        resources = []
        for entry in self.resource_registry.get_all_resources():
            to_dict = getattr(entry, "to_dict", None)
            resources.append(to_dict() if callable(to_dict) else entry.__dict__)
        return {
            "total_resources": summary.get("total_resources", 0),
            "enabled_resources": summary.get("enabled_resources", 0),
            "resources": resources,
        }

    def validate_request(
        self,
        *,
        query: str,
        mode: str,
        resource_filter: list[str],
    ) -> None:
        text = str(query).strip()
        if not text:
            raise ValueError("query cannot be empty")
        if len(text) > self._max_query_chars:
            raise ValueError("query is too long")

        normalized_mode = str(mode).strip().lower()
        allowed_modes = {
            ThinkingMode.SIMPLE.value,
            ThinkingMode.WEB_ONLY.value,
            ThinkingMode.GRAPH.value,
        }
        if normalized_mode not in allowed_modes:
            raise ValueError("mode is invalid")
        if (
            normalized_mode == ThinkingMode.GRAPH.value
            and not bool(self.config.SERVER_ENABLE_GRAPH_MODE)
        ):
            raise ValueError("graph mode is disabled")

        if not resource_filter:
            return

        if len(resource_filter) > 20:
            raise ValueError("resource_filter is too large")

        get_resource = getattr(self.resource_registry, "get_resource", None)
        if not callable(get_resource):
            return

        unknown = [name for name in resource_filter if get_resource(name) is None]
        if unknown:
            raise ValueError(
                "resource_filter contains unknown resources: "
                + ", ".join(unknown)
            )

    def _acquire_slot(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise RuntimeError("service is busy")
        with self._active_lock:
            self._active_requests += 1

    def _release_slot(self) -> None:
        with self._active_lock:
            self._active_requests = max(0, self._active_requests - 1)
        self._semaphore.release()

    def run_query(
        self,
        *,
        query: str,
        mode: str,
        resource_filter: list[str],
        save_md_report: bool,
    ) -> dict[str, Any]:
        self.validate_request(
            query=query,
            mode=mode,
            resource_filter=resource_filter,
        )
        self._acquire_slot()
        callback_registered = False
        try:
            future = self._executor.submit(
                self.system.query,
                query,
                thinking_mode=mode,
                resource_filter=resource_filter,
                verbose=False,
                save_md_report=save_md_report,
            )
            future.add_done_callback(lambda _: self._release_slot())
            callback_registered = True
            try:
                return future.result(timeout=self._timeout_seconds)
            except FuturesTimeoutError as exc:
                raise RuntimeError("query timed out") from exc
        finally:
            if not callback_registered:
                self._release_slot()

    def get_query(self, query_id: str) -> dict[str, Any]:
        logger = getattr(self.system, "logger", None)
        if logger is None:
            raise FileNotFoundError(query_id)
        result = logger.get_query(query_id)
        if result is None:
            raise FileNotFoundError(query_id)
        return result

    def get_query_report(self, query_id: str) -> str:
        logger = getattr(self.system, "logger", None)
        if logger is None:
            raise FileNotFoundError(query_id)
        path = logger.get_query_report_md_path(query_id)
        if not path:
            raise FileNotFoundError(query_id)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
