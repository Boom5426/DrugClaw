# Markdown Query Report Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--save-md-report` flag for `drugclaw run --query ...` that saves a local Markdown report under `query_logs/<query_id>/report.md`.

**Architecture:** Reuse the existing query logging pipeline instead of creating a separate export path. The CLI will expose the new flag, `DrugClawSystem.query()` will pass the intent into logging, and `QueryLogger` will optionally write `report.md` using the existing Markdown answer card content.

**Tech Stack:** Python, argparse, pathlib, pytest, existing DrugClaw query logger and response formatter.

---

## File Map

- Modify: `drugclaw/cli.py`
  Add the `--save-md-report` flag and print the generated Markdown path when available.
- Modify: `drugclaw/main_system.py`
  Thread `save_md_report` through query execution and logging.
- Modify: `drugclaw/query_logger.py`
  Optionally write `report.md` and expose the generated path.
- Modify: `tests/cli/test_cli_output.py`
  Cover parser and CLI output behavior for the new flag.
- Create: `tests/logging/test_query_logger_md_report.py`
  Cover conditional Markdown generation and file contents.
- Modify: `README.md`
  Document the new CLI flag.
- Modify: `README_CN.md`
  Document the new CLI flag in Chinese.

## Chunk 1: CLI and Logging Contract

### Task 1: Add parser and CLI coverage for `--save-md-report`

**Files:**
- Modify: `drugclaw/cli.py`
- Test: `tests/cli/test_cli_output.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_query_can_print_saved_md_report_path(...):
    ...
    assert "Markdown report saved to" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py::test_run_query_can_print_saved_md_report_path -q`
Expected: FAIL because the flag and output do not exist yet.

- [ ] **Step 3: Implement the minimal CLI changes**

Implement:
- `run` parser gets `--save-md-report`
- `_run_query(...)` accepts the new boolean
- CLI prints `md_report_path` when present

- [ ] **Step 4: Run CLI tests**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py -q`
Expected: PASS

## Chunk 2: Query Logger HTML Export

### Task 2: Add failing logger tests for conditional Markdown generation

**Files:**
- Modify: `drugclaw/query_logger.py`
- Create: `tests/logging/test_query_logger_md_report.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_query_logger_writes_md_report_when_requested(tmp_path):
    ...
    assert (query_dir / "report.md").exists()

def test_query_logger_skips_md_report_when_not_requested(tmp_path):
    ...
    assert not (query_dir / "report.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/logging/test_query_logger_md_report.py -q`
Expected: FAIL because Markdown export does not exist yet.

- [ ] **Step 3: Implement minimal logger and formatter changes**

Implement:
- `QueryLogger.log_query(..., save_md_report=False)`
- write `report.md` only when requested

- [ ] **Step 4: Run logger tests**

Run: `./.venv/bin/python -m pytest tests/logging/test_query_logger_md_report.py -q`
Expected: PASS

## Chunk 3: Core Query Plumbing and Docs

### Task 3: Thread `save_md_report` through `DrugClawSystem.query()`

**Files:**
- Modify: `drugclaw/main_system.py`
- Modify: `drugclaw/query_logger.py`
- Test: `tests/cli/test_cli_output.py`

- [ ] **Step 1: Write or extend failing integration coverage**

```python
def test_run_query_passes_save_md_report_to_system(...):
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py -q`
Expected: FAIL until the parameter is threaded through.

- [ ] **Step 3: Implement minimal plumbing**

Implement:
- `DrugClawSystem.query(..., save_md_report=False)`
- set `result["md_report_path"]` when generated

- [ ] **Step 4: Run targeted tests**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py tests/logging/test_query_logger_md_report.py -q`
Expected: PASS

### Task 4: Document the new usage

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`

- [ ] **Step 1: Add a short usage example**

Document:

```bash
python -m drugclaw run --query "What does imatinib target?" --save-md-report
```

- [ ] **Step 2: Verify docs mention the saved Markdown report path behavior**

## Chunk 4: Final Verification and Delivery

### Task 5: Run regression verification and ship

**Files:**
- Modify: any files touched above

- [ ] **Step 1: Run focused tests**

Run:
- `./.venv/bin/python -m pytest tests/cli/test_cli_output.py tests/logging/test_query_logger_md_report.py -q`

- [ ] **Step 2: Run the full test suite**

Run:
- `./.venv/bin/python -m pytest -q`

- [ ] **Step 3: Commit**

```bash
git add drugclaw tests README.md README_CN.md docs/superpowers/specs/2026-03-18-html-query-report-design.md docs/superpowers/plans/2026-03-18-html-query-report-plan.md
git commit -m "feat: add optional markdown query reports"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```
