# File-Driven LLM Configuration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLM provider settings file-driven and compatible with both legacy and new `navigator_api_keys.json` formats.

**Architecture:** Centralize LLM setting resolution in `Config`, keep `LLMClient` thin, and back the change with focused pytest coverage for config parsing and CLI doctor behavior.

**Tech Stack:** Python, pytest, Markdown docs

---

## Chunk 1: Parsing and Runtime

### Task 1: Add failing tests for legacy and new LLM config formats

**Files:**
- Create: `tests/config/test_llm_config.py`
- Create: `tests/cli/test_cli_doctor.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run focused tests and confirm they fail**
- [ ] **Step 3: Commit**

### Task 2: Implement file-driven LLM config resolution

**Files:**
- Modify: `drugclaw/config.py`
- Modify: `drugclaw/llm_client.py`

- [ ] **Step 1: Resolve legacy and new key names in `Config`**
- [ ] **Step 2: Make `LLMClient` use resolved timeout/model/token settings**
- [ ] **Step 3: Run focused tests and confirm they pass**
- [ ] **Step 4: Commit**

## Chunk 2: CLI and Docs

### Task 3: Update doctor, example config, and README

**Files:**
- Modify: `drugclaw/cli.py`
- Modify: `navigator_api_keys.example.json`
- Modify: `README.md`
- Modify: `README_CN.md`

- [ ] **Step 1: Make doctor accept both key styles**
- [ ] **Step 2: Update examples and docs to recommend the new format**
- [ ] **Step 3: Run `pytest tests -q`**
- [ ] **Step 4: Commit**
