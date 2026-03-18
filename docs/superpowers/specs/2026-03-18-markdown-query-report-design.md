# Markdown Query Report Design

**Date:** 2026-03-18

## Goal

Allow users to explicitly save a local Markdown report for their own ad hoc CLI questions, without changing the default behavior of demo-oriented commands.

## Scope

This change applies only to:

- `python -m drugclaw run --query "..."`
- `drugclaw run --query "..."`

and only when the user passes a dedicated flag:

- `--save-md-report`

This change does **not** auto-generate HTML for:

- `drugclaw demo`
- `drugclaw list`
- `drugclaw doctor`

## Product Behavior

When a user runs:

```bash
python -m drugclaw run --query "..." --save-md-report
```

DrugClaw should:

1. print the normal CLI answer to the terminal
2. keep writing the existing per-query log artifacts under `query_logs/`
3. additionally save a Markdown report at:

```text
query_logs/<query_id>/report.md
```

4. print the saved HTML path after the query finishes

When the flag is absent, no extra Markdown report should be generated.

## Design Choice

Reuse the existing query logging pipeline instead of building a second CLI-only export path.

Current behavior already persists:

- `answer.md`
- `metadata.json`
- `reasoning_trace.md`
- `evidence.json`
- `full_result.pkl`

The new Markdown report should be generated from the same result payload during query logging. This keeps the terminal output, Markdown report, JSON logs, and the saved report aligned.

## Markdown Report Requirements

The generated Markdown report should be:

- a single file
- readable directly in editors and markdown viewers
- dependency-free at runtime
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

- `--save-md-report`

The CLI should pass this intent into the core query execution path. Demo and utility commands must remain unchanged.

### Core query execution

Extend the query execution path so that the system can request HTML export when logging is enabled.

The returned result should include:

- `query_id`
- `md_report_path` when generated

### Query logger

Extend `QueryLogger.log_query(...)` to optionally write `report.md` beside the existing Markdown and JSON artifacts.

### Formatter

Reuse the existing Markdown answer card as the saved report content instead of introducing a second renderer.

## Non-Goals

- no browser auto-open behavior
- no Markdown export for demo/list/doctor
- no change to default CLI output behavior

## Testing Strategy

Add focused regression coverage for:

- parser support for `--save-md-report`
- CLI run path showing the saved Markdown path
- `report.md` generation only when requested
- saved Markdown containing the core query report content
- absence of `report.md` when the flag is not provided

## Risks and Mitigations

### Risk: duplicated report files

Mitigation:

- keep `answer.md` as the always-on log artifact
- keep `report.md` as the explicit user-requested export path only

### Risk: noisy behavior for demo users

Mitigation:

- limit feature to explicit `run --query ... --save-md-report`

### Risk: brittle file output tests

Mitigation:

- test for presence of key Markdown content and expected file paths rather than exact full-file snapshots
