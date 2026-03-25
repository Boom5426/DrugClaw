from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .server_models import QueryRequest
from .service_runtime import DrugClawServiceRuntime


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    runtime: DrugClawServiceRuntime | None = None,
    key_file: str = "navigator_api_keys.json",
) -> FastAPI:
    app = FastAPI(title="DrugClaw Service")
    app.state.runtime = runtime or DrugClawServiceRuntime.from_key_file(key_file)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/health")
    def health() -> dict:
        return app.state.runtime.health()

    @app.get("/resources")
    def resources() -> dict:
        return app.state.runtime.resources()

    @app.post("/api/query")
    def query(request: QueryRequest) -> dict:
        runtime_config = getattr(app.state.runtime, "config", None)
        default_mode = getattr(runtime_config, "SERVER_DEFAULT_MODE", "simple")
        effective_mode = request.mode or default_mode
        try:
            return app.state.runtime.run_query(
                query=request.query,
                mode=effective_mode,
                resource_filter=request.resource_filter,
                save_md_report=request.save_md_report,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            status_code = 503 if "busy" in str(exc).lower() else 504 if "timed out" in str(exc).lower() else 500
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @app.get("/api/queries/{query_id}")
    def get_query(query_id: str) -> dict:
        try:
            return app.state.runtime.get_query(query_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="query not found") from exc

    @app.get("/api/queries/{query_id}/report", response_class=PlainTextResponse)
    def get_query_report(query_id: str) -> str:
        try:
            return app.state.runtime.get_query_report(query_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report not found") from exc

    @app.get("/", response_class=FileResponse)
    def root():
        return FileResponse(_STATIC_DIR / "chat.html")

    return app
