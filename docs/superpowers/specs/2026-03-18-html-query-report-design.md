# HTML Query Report Design

**Date:** 2026-03-18

## Goal

Allow users to explicitly save a local visual HTML report for their own ad hoc CLI questions, without changing the default behavior of demo-oriented commands.

## Scope

This change applies only to:

- `python -m drugclaw run --query "..."`
- `drugclaw run --query "..."`

and only when the user passes a dedicated flag:

- `--save-html-report`

This change does **not** auto-generate HTML for:

- `drugclaw demo`
- `drugclaw list`
- `drugclaw doctor`

## Product Behavior

When a user runs:

```bash
python -m drugclaw run --query "..." --save-html-report
```

DrugClaw should:

1. print the normal CLI answer to the terminal
2. keep writing the existing per-query log artifacts under `query_logs/`
3. additionally save a visual HTML report at:

```text
query_logs/<query_id>/report.html
```

4. print the saved HTML path after the query finishes

When the flag is absent, no HTML report should be generated.

## Design Choice

Reuse the existing query logging pipeline instead of building a second CLI-only export path.

Current behavior already persists:

- `answer.md`
- `metadata.json`
- `reasoning_trace.md`
- `evidence.json`
- `full_result.pkl`

The new HTML report should be generated from the same result payload during query logging. This keeps the terminal output, Markdown report, JSON logs, and HTML report aligned.

## HTML Report Requirements

The generated HTML report should be:

- a single self-contained file
- readable when opened directly from disk
- dependency-free at runtime
- styled with inline CSS
- structured around the same answer sections users already see in CLI output

Recommended sections:

- title and query metadata
- confidence badge
- answer body
- key claims / warnings / limitations when structured output exists
- evidence summary table
- sources
- reasoning trace when available

## Architecture

### CLI layer

Add a new `run`-only flag:

- `--save-html-report`

The CLI should pass this intent into the core query execution path. Demo and utility commands must remain unchanged.

### Core query execution

Extend the query execution path so that the system can request HTML export when logging is enabled.

The returned result should include:

- `query_id`
- `html_report_path` when generated

### Query logger

Extend `QueryLogger.log_query(...)` to optionally write `report.html` beside the existing Markdown and JSON artifacts.

### Formatter

Add an HTML formatter function that renders a visually readable report directly from the result payload. Avoid introducing a Markdown-to-HTML dependency.

## Non-Goals

- no browser auto-open behavior
- no HTML export for demo/list/doctor
- no static asset pipeline
- no web server
- no change to default CLI output behavior

## Testing Strategy

Add focused regression coverage for:

- parser support for `--save-html-report`
- CLI run path showing the saved HTML path
- HTML file generation only when requested
- HTML output containing core query report content
- absence of HTML generation when the flag is not provided

## Risks and Mitigations

### Risk: duplicated formatting logic

Mitigation:

- keep HTML rendering in formatter-focused code, not ad hoc inside CLI

### Risk: noisy behavior for demo users

Mitigation:

- limit feature to explicit `run --query ... --save-html-report`

### Risk: brittle file output tests

Mitigation:

- test for presence of key HTML fragments and expected file paths rather than exact full-page snapshots
