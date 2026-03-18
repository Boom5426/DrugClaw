# HTML Query Report Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--save-html-report` flag for `drugclaw run --query ...` that saves a local visual HTML report under `query_logs/<query_id>/report.html`.

**Architecture:** Reuse the existing query logging pipeline instead of creating a separate export path. The CLI will expose the new flag, `DrugClawSystem.query()` will pass the intent into logging, `QueryLogger` will optionally write `report.html`, and formatting code will generate a self-contained HTML page from the existing result payload.

**Tech Stack:** Python, argparse, pathlib, pytest, existing DrugClaw query logger and response formatter.

---

## File Map

- Modify: `drugclaw/cli.py`
  Add the `--save-html-report` flag and print the generated HTML path when available.
- Modify: `drugclaw/main_system.py`
  Thread `save_html_report` through query execution and logging.
- Modify: `drugclaw/query_logger.py`
  Optionally write `report.html` and expose the generated path.
- Modify: `drugclaw/response_formatter.py`
  Add HTML report rendering for the saved local document.
- Modify: `tests/cli/test_cli_output.py`
  Cover parser and CLI output behavior for the new flag.
- Create: `tests/logging/test_query_logger_html_report.py`
  Cover conditional HTML generation and page contents.
- Modify: `README.md`
  Document the new CLI flag.
- Modify: `README_CN.md`
  Document the new CLI flag in Chinese.

## Chunk 1: CLI and Logging Contract

### Task 1: Add parser and CLI coverage for `--save-html-report`

**Files:**
- Modify: `drugclaw/cli.py`
- Test: `tests/cli/test_cli_output.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_query_can_print_saved_html_report_path(...):
    ...
    assert "HTML report saved to" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py::test_run_query_can_print_saved_html_report_path -q`
Expected: FAIL because the flag and output do not exist yet.

- [ ] **Step 3: Implement the minimal CLI changes**

Implement:
- `run` parser gets `--save-html-report`
- `_run_query(...)` accepts the new boolean
- CLI prints `html_report_path` when present

- [ ] **Step 4: Run CLI tests**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py -q`
Expected: PASS

## Chunk 2: Query Logger HTML Export

### Task 2: Add failing logger tests for conditional HTML generation

**Files:**
- Modify: `drugclaw/query_logger.py`
- Modify: `drugclaw/response_formatter.py`
- Create: `tests/logging/test_query_logger_html_report.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_query_logger_writes_html_report_when_requested(tmp_path):
    ...
    assert (query_dir / "report.html").exists()

def test_query_logger_skips_html_report_when_not_requested(tmp_path):
    ...
    assert not (query_dir / "report.html").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/logging/test_query_logger_html_report.py -q`
Expected: FAIL because HTML export does not exist yet.

- [ ] **Step 3: Implement minimal logger and formatter changes**

Implement:
- `QueryLogger.log_query(..., save_html_report=False)`
- HTML formatter in `response_formatter.py`
- write `report.html` only when requested

- [ ] **Step 4: Run logger tests**

Run: `./.venv/bin/python -m pytest tests/logging/test_query_logger_html_report.py -q`
Expected: PASS

## Chunk 3: Core Query Plumbing and Docs

### Task 3: Thread `save_html_report` through `DrugClawSystem.query()`

**Files:**
- Modify: `drugclaw/main_system.py`
- Modify: `drugclaw/query_logger.py`
- Test: `tests/cli/test_cli_output.py`

- [ ] **Step 1: Write or extend failing integration coverage**

```python
def test_run_query_passes_save_html_report_to_system(...):
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py -q`
Expected: FAIL until the parameter is threaded through.

- [ ] **Step 3: Implement minimal plumbing**

Implement:
- `DrugClawSystem.query(..., save_html_report=False)`
- set `result["html_report_path"]` when generated

- [ ] **Step 4: Run targeted tests**

Run: `./.venv/bin/python -m pytest tests/cli/test_cli_output.py tests/logging/test_query_logger_html_report.py -q`
Expected: PASS

### Task 4: Document the new usage

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`

- [ ] **Step 1: Add a short usage example**

Document:

```bash
python -m drugclaw run --query "What does imatinib target?" --save-html-report
```

- [ ] **Step 2: Verify docs mention the saved HTML report path behavior**

## Chunk 4: Final Verification and Delivery

### Task 5: Run regression verification and ship

**Files:**
- Modify: any files touched above

- [ ] **Step 1: Run focused tests**

Run:
- `./.venv/bin/python -m pytest tests/cli/test_cli_output.py tests/logging/test_query_logger_html_report.py -q`

- [ ] **Step 2: Run the full test suite**

Run:
- `./.venv/bin/python -m pytest -q`

- [ ] **Step 3: Commit**

```bash
git add drugclaw tests README.md README_CN.md docs/superpowers/specs/2026-03-18-html-query-report-design.md docs/superpowers/plans/2026-03-18-html-query-report-plan.md
git commit -m "feat: add optional html query reports"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```
