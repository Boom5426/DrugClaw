from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import ThinkingMode


_ALLOWED_MODES = {mode.value for mode in ThinkingMode}


class QueryRequest(BaseModel):
    query: str
    mode: str | None = None
    resource_filter: list[str] = Field(default_factory=list)
    save_md_report: bool = False

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("query cannot be empty")
        if len(text) > 5000:
            raise ValueError("query is too long")
        return text

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized not in _ALLOWED_MODES:
            raise ValueError("invalid mode")
        return normalized

    @field_validator("resource_filter")
    @classmethod
    def _validate_resource_filter(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                raise ValueError("resource_filter entries must be non-empty")
            if text not in cleaned:
                cleaned.append(text)
        return cleaned


class QueryResponse(BaseModel):
    success: bool
    query: str
    normalized_query: str = ""
    answer: str = ""
    query_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model: str
    default_mode: str
    active_requests: int = 0


class ResourceSummaryResponse(BaseModel):
    total_resources: int
    enabled_resources: int
    resources: list[dict[str, Any]] = Field(default_factory=list)


class GatewayInvokeRequest(BaseModel):
    resource_name: str | None = None
    tool_namespace: str | None = None
    path: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    query: str = ""
    variables: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 10.0

    @field_validator("resource_name", "tool_namespace", "path", "query")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout_seconds(cls, value: float) -> float:
        timeout = float(value)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        return timeout

    @model_validator(mode="after")
    def _validate_identifier(self) -> "GatewayInvokeRequest":
        if not self.resource_name and not self.tool_namespace:
            raise ValueError("resource_name or tool_namespace is required")
        return self
