# File-Driven LLM Configuration Design

**Date:** 2026-03-16

## Goal

Move LLM provider configuration out of hard-coded defaults and into `navigator_api_keys.json`, while preserving compatibility with the existing legacy key format.

## Scope

- Support both legacy and new key names in `navigator_api_keys.json`
- Allow `base_url`, `model`, `temperature`, `max_tokens`, and `timeout` to be defined in the file
- Keep existing users working without having to rewrite their config immediately
- Update CLI validation and user-facing docs to match the new behavior

## Supported File Formats

### Legacy format

```json
{
  "OPENAI_API_KEY": "sk-...",
  "base_url": "https://provider.example.com/v1"
}
```

### New format

```json
{
  "api_key": "sk-...",
  "base_url": "https://provider.example.com/v1",
  "model": "gemini-3-pro-all",
  "max_tokens": 40000,
  "timeout": 600,
  "temperature": 0.7
}
```

## Design

### Config parsing

- `Config` becomes the single place that resolves LLM settings.
- Field resolution priority:
  1. new key names from JSON
  2. legacy key names from JSON
  3. current code defaults
- `api_key` and `OPENAI_API_KEY` should populate the same runtime field.
- `base_url` should remain required for OpenAI-compatible providers.

### Runtime behavior

- `LLMClient` should use the resolved config values instead of hard-coded constants.
- Request timeout should be passed into the OpenAI-compatible client.
- `get_llm_config()` should expose the resolved runtime values, including timeout.

### CLI and docs

- `doctor` should recognize both `api_key` and `OPENAI_API_KEY`.
- The example config file and README should show the recommended new format and mention legacy compatibility.

## Verification

- Config parsing tests for legacy and new formats
- CLI doctor tests for both key styles
- LLM client tests confirming runtime values come from config
