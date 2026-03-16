import json
from pathlib import Path

from drugclaw.config import Config
from drugclaw.llm_client import LLMClient


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_config_supports_legacy_llm_keys(tmp_path: Path) -> None:
    key_file = tmp_path / "legacy.json"
    _write_json(
        key_file,
        {
            "OPENAI_API_KEY": "legacy-key",
            "base_url": "https://legacy.example.com/v1",
        },
    )

    config = Config(key_file=str(key_file))

    assert config.OPENAI_API_KEY == "legacy-key"
    assert config.base_url == "https://legacy.example.com/v1"
    assert config.MODEL_NAME == "gpt-oss-120b"
    assert config.MAX_TOKENS == 2000
    assert config.TEMPERATURE == 0.7
    assert config.TIMEOUT == 60


def test_config_prefers_new_llm_keys_and_runtime_overrides(tmp_path: Path) -> None:
    key_file = tmp_path / "new.json"
    _write_json(
        key_file,
        {
            "OPENAI_API_KEY": "legacy-key",
            "api_key": "new-key",
            "base_url": "https://provider.example.com/v1",
            "model": "gemini-3-pro-all",
            "max_tokens": 40000,
            "timeout": 600,
            "temperature": 0.3,
        },
    )

    config = Config(key_file=str(key_file))

    assert config.OPENAI_API_KEY == "new-key"
    assert config.MODEL_NAME == "gemini-3-pro-all"
    assert config.MAX_TOKENS == 40000
    assert config.TIMEOUT == 600
    assert config.TEMPERATURE == 0.3
    assert config.get_llm_config()["timeout"] == 600


def test_llm_client_uses_file_driven_timeout(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    key_file = tmp_path / "client.json"
    _write_json(
        key_file,
        {
            "api_key": "new-key",
            "base_url": "https://provider.example.com/v1",
            "timeout": 123,
            "model": "custom-model",
        },
    )

    monkeypatch.setattr("drugclaw.llm_client.openai.OpenAI", FakeOpenAI)

    config = Config(key_file=str(key_file))
    client = LLMClient(config)

    assert captured["api_key"] == "new-key"
    assert captured["base_url"] == "https://provider.example.com/v1"
    assert captured["timeout"] == 123
    assert client.model == "custom-model"
