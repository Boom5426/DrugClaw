from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import List

from .config import Config
from .eval_runner import run_self_bench
from .models import ThinkingMode
from .query_logger import QueryLogger
from .skills import build_default_registry
from .resource_registry import build_resource_registry


DEMO_PRESETS = {
    "adr": {
        "query": "What are the known adverse drug reactions of aspirin?",
        "mode": ThinkingMode.SIMPLE.value,
        "resource_filter": ["SIDER", "FAERS"],
        "description": "ADR query with the most stable default resources.",
    },
    "dti": {
        "query": "What are the known drug targets of imatinib?",
        "mode": ThinkingMode.SIMPLE.value,
        "resource_filter": ["ChEMBL", "DGIdb", "Open Targets Platform"],
        "description": "Drug-target query across three public sources.",
    },
    "label": {
        "query": "What prescribing and safety information is available for metformin?",
        "mode": ThinkingMode.SIMPLE.value,
        "resource_filter": ["DailyMed", "openFDA Human Drug", "MedlinePlus Drug Info"],
        "description": "Drug labeling and patient information query.",
    },
}


def _parse_resource_filter(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drugclaw",
        description="CLI entrypoint for fast DrugClaw demos and manual queries.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a custom query with configurable mode and resource filter.",
    )
    run_parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query to send to DrugClaw.",
    )
    run_parser.add_argument(
        "--mode",
        choices=[ThinkingMode.GRAPH.value, ThinkingMode.SIMPLE.value, ThinkingMode.WEB_ONLY.value],
        default=ThinkingMode.SIMPLE.value,
        help="Thinking mode. SIMPLE is the safest default for first-time usage.",
    )
    run_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )
    run_parser.add_argument(
        "--resource-filter",
        type=_parse_resource_filter,
        default=None,
        help="Comma-separated skill names, e.g. 'SIDER,FAERS'.",
    )
    run_parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print structured evidence, key claims, and confidence after the answer.",
    )
    run_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the structured query plan summary after the answer.",
    )
    run_parser.add_argument(
        "--show-claims",
        action="store_true",
        help="Print structured claim assessment summaries after the answer.",
    )
    run_parser.add_argument(
        "--debug-agents",
        action="store_true",
        help="Print internal agent routing and execution logs.",
    )
    run_parser.add_argument(
        "--save-md-report",
        action="store_true",
        help="Save a local Markdown report under query_logs/ for this custom query.",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a curated demo query intended for first-time users.",
    )
    demo_parser.add_argument(
        "--preset",
        choices=sorted(DEMO_PRESETS),
        default="label",
        help="Choose a built-in demo scenario.",
    )
    demo_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )
    demo_parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print structured evidence, key claims, and confidence after the answer.",
    )
    demo_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the structured query plan summary after the answer.",
    )
    demo_parser.add_argument(
        "--show-claims",
        action="store_true",
        help="Print structured claim assessment summaries after the answer.",
    )
    demo_parser.add_argument(
        "--debug-agents",
        action="store_true",
        help="Print internal agent routing and execution logs.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check whether the local setup is ready for first-time usage.",
    )
    doctor_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )

    subparsers.add_parser(
        "list",
        help="List built-in demos, modes, and recommended first commands.",
    )
    list_parser = subparsers.choices["list"]
    list_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json. Use --key-file api_keys.json to override.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the DrugClaw web/API service.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the local server to.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the local server to.",
    )
    serve_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )

    scorecard_parser = subparsers.add_parser(
        "scorecard",
        help="Run the unified self-bench scorecard summary.",
    )
    scorecard_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional self-bench dataset names to include.",
    )
    scorecard_parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Optional QueryPlan-aligned task types to include.",
    )
    scorecard_parser.add_argument(
        "--plan-types",
        nargs="+",
        default=None,
        help="Optional plan types to include.",
    )
    scorecard_parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per dataset (0 = use all available).",
    )
    scorecard_parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )
    scorecard_parser.add_argument(
        "--maskself",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Optional self-bench maskself setting.",
    )
    scorecard_parser.add_argument(
        "--log-dir",
        default="./query_logs",
        help="Directory used to persist the latest scorecard summary.",
    )
    scorecard_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the scorecard summary as JSON.",
    )
    scorecard_parser.add_argument(
        "--min-task-success-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for task_success_rate.",
    )
    scorecard_parser.add_argument(
        "--min-evidence-quality-score",
        type=float,
        default=None,
        help="Optional release gate threshold for evidence_quality_score.",
    )
    scorecard_parser.add_argument(
        "--min-authority-source-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for authority_source_rate.",
    )
    scorecard_parser.add_argument(
        "--min-knowhow-hit-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for knowhow_hit_rate.",
    )
    scorecard_parser.add_argument(
        "--min-package-ready-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for package_ready_rate.",
    )

    return parser


def _status_line(label: str, ok: bool, detail: str, *, level: str | None = None) -> str:
    status = level or ("OK" if ok else "FAIL")
    return f"[{status}] {label}: {detail}"


def _doctor_check_key_file(key_file: str) -> List[str]:
    lines: List[str] = []
    key_path = Path(key_file).expanduser()
    if not key_path.is_absolute():
        key_path = (Path.cwd() / key_path).resolve()

    if not key_path.exists():
        return [_status_line("key_file", False, f"not found at {key_path}")]

    try:
        data = json.loads(key_path.read_text())
    except Exception as exc:
        return [_status_line("key_file", False, f"invalid JSON ({exc})")]

    api_key = data.get("api_key") or data.get("OPENAI_API_KEY")
    has_api_key = bool(api_key)
    has_base_url = bool(data.get("base_url"))
    lines.append(_status_line("key_file", True, str(key_path)))
    lines.append(_status_line("OPENAI_API_KEY", has_api_key, "present" if has_api_key else "missing"))
    lines.append(_status_line("base_url", has_base_url, data.get("base_url", "missing")))
    if "model" in data:
        lines.append(_status_line("model", True, str(data["model"])))
    return lines


def _doctor_check_imports() -> List[str]:
    lines: List[str] = []
    try:
        import langgraph  # noqa: F401
        lines.append(_status_line("langgraph", True, "importable"))
    except Exception as exc:
        lines.append(_status_line("langgraph", False, f"import failed ({exc})"))

    try:
        import openai  # noqa: F401
        lines.append(_status_line("openai", True, "importable"))
    except Exception as exc:
        lines.append(_status_line("openai", False, f"import failed ({exc})"))
    return lines


def _load_registry_for_cli(key_file: str, *, strict_config: bool) -> tuple[object, object]:
    try:
        config = Config(key_file=key_file)
    except Exception:
        if strict_config:
            raise
        config = SimpleNamespace(SKILL_CONFIGS={}, KG_ENDPOINTS={})

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        skill_registry = build_default_registry(config)
        resource_registry = build_resource_registry(skill_registry)
    finally:
        logging.disable(previous_disable_level)
    return skill_registry, resource_registry


def _build_system(key_file: str):
    from .main_system import DrugClawSystem

    config = Config(key_file=key_file)
    return DrugClawSystem(config)


def _registry_summary_lines(summary: dict) -> List[str]:
    lines = [
        _status_line("registry_total_resources", True, str(summary["total_resources"])),
        _status_line("registry_enabled_resources", True, str(summary["enabled_resources"])),
    ]
    for status_name in (
        "ready",
        "degraded",
        "missing_metadata",
        "missing_dependency",
        "disabled",
    ):
        lines.append(
            _status_line(
                f"registry_status:{status_name}",
                True,
                str(summary["status_counts"].get(status_name, 0)),
            )
        )
    for status_name in (
        "ready",
        "degraded",
        "missing_metadata",
        "missing_dependency",
        "disabled",
    ):
        lines.append(
            _status_line(
                f"registry_package_status:{status_name}",
                True,
                str(summary.get("package_status_counts", {}).get(status_name, 0)),
            )
        )
    lines.append(
        _status_line(
            "registry_resources_with_knowhow",
            True,
            str(summary.get("resources_with_knowhow", 0)),
        )
    )
    lines.append(
        _status_line(
            "registry_gateway_declared_resources",
            True,
            str(summary.get("gateway_declared_resources", 0)),
        )
    )
    lines.append(
        _status_line(
            "registry_gateway_ready_resources",
            True,
            str(summary.get("gateway_ready_resources", 0)),
        )
    )
    for component_name, count in sorted(
        dict(summary.get("missing_component_counts", {})).items()
    ):
        lines.append(
            _status_line(
                f"registry_missing_component:{component_name}",
                True,
                str(count),
            )
        )
    return lines


def _doctor_check_registry(key_file: str) -> List[str]:
    try:
        _, resource_registry = _load_registry_for_cli(key_file, strict_config=False)
    except Exception as exc:
        return [_status_line("registry", False, f"cannot build resource registry ({exc})")]

    lines = _registry_summary_lines(resource_registry.summarize_registry())
    for entry in resource_registry.get_all_resources():
        detail = (
            f"category={entry.category}; access={entry.access_mode}; "
            f"status={entry.status}; reason={entry.status_reason}"
        )
        package_id = str(getattr(entry, "package_id", "") or "").strip()
        package_status = getattr(entry, "package_status", "")
        missing_components = list(getattr(entry, "missing_components", []) or [])
        has_knowhow = bool(getattr(entry, "has_knowhow", False))
        gateway_declared = bool(getattr(entry, "gateway_declared", False))
        gateway_ready = bool(getattr(entry, "gateway_ready", True))
        gateway_status = str(getattr(entry, "gateway_status", "") or "").strip()
        gateway_transport = str(getattr(entry, "gateway_transport", "") or "").strip()
        gateway_tool_namespace = str(getattr(entry, "gateway_tool_namespace", "") or "").strip()
        gateway_missing_env = list(getattr(entry, "gateway_missing_env", []) or [])
        if package_id:
            detail += f"; package_id={package_id}"
        if package_status:
            detail += f"; package_status={package_status}"
        if missing_components:
            detail += f"; missing_components={','.join(missing_components)}"
        detail += f"; has_knowhow={'yes' if has_knowhow else 'no'}"
        detail += f"; gateway_declared={'yes' if gateway_declared else 'no'}"
        detail += f"; gateway_ready={'yes' if gateway_ready else 'no'}"
        if gateway_status:
            detail += f"; gateway_status={gateway_status}"
        if gateway_transport:
            detail += f"; gateway_transport={gateway_transport}"
        if gateway_tool_namespace:
            detail += f"; gateway_tool_namespace={gateway_tool_namespace}"
        if gateway_missing_env:
            detail += f"; gateway_missing_env={','.join(gateway_missing_env)}"
        if not entry.enabled or entry.status == "ready":
            lines.append(_status_line(f"resource:{entry.name}", True, detail))
            continue
        lines.append(_status_line(f"resource:{entry.name}", False, detail, level="WARN"))
    return lines


def _doctor_check_presets(key_file: str) -> List[str]:
    lines: List[str] = []
    try:
        registry, resource_registry = _load_registry_for_cli(key_file, strict_config=True)
    except Exception as exc:
        return [_status_line("config", False, f"cannot initialize Config ({exc})")]

    for preset_name, preset in DEMO_PRESETS.items():
        required = preset["resource_filter"]
        unavailable = []
        available_count = 0
        details = []
        for skill_name in required:
            skill = registry.get_skill(skill_name)
            if skill is None:
                unavailable.append(skill_name)
                details.append(f"{skill_name}=missing")
                continue
            entry = resource_registry.get_resource(skill_name) if resource_registry is not None else None
            access = getattr(entry, "access_mode", getattr(skill, "access_mode", "unknown"))
            available = bool(skill.is_available())
            path_note = "available" if available else getattr(entry, "status_reason", "unavailable")
            if access == "CLI" and hasattr(skill, "_cli_available"):
                cli_ok = skill._cli_available()  # type: ignore[attr-defined]
                details.append(f"{skill_name}={access}:{'cli' if cli_ok else 'rest-fallback'}")
            else:
                details.append(f"{skill_name}={access}:{path_note}")
            if not available:
                unavailable.append(skill_name)
            else:
                available_count += 1
        ok = available_count > 0
        lines.append(
            _status_line(
                f"demo:{preset_name}",
                ok,
                ", ".join(details) if details else "no resources",
            )
        )
    return lines


def _doctor_check_install_hint() -> List[str]:
    current_script_dir = Path(sys.executable).expanduser().parent
    bundled_script = any(
        current_script_dir.joinpath(candidate).exists()
        for candidate in ("drugclaw", "drugclaw.exe", "drugclaw-script.py")
    )
    installed_as_script = any(
        Path(path).joinpath("drugclaw").exists()
        for path in os.environ.get("PATH", "").split(os.pathsep)
        if path
    )
    installed_as_script = installed_as_script or bundled_script
    detail = (
        "command available for the current Python environment"
        if bundled_script
        else "command found on PATH"
        if installed_as_script
        else "use `python -m drugclaw ...` or `python -m pip install -e .[dev] --no-build-isolation`"
    )
    return [_status_line("cli_command", installed_as_script, detail)]


def _doctor_check_git_safety() -> List[str]:
    lines: List[str] = []
    tracked = os.system("git ls-files --error-unmatch navigator_api_keys.json >/dev/null 2>&1") == 0
    lines.append(
        _status_line(
            "tracked_key_file",
            not tracked,
            "not tracked" if not tracked else "navigator_api_keys.json is still tracked by Git",
        )
    )

    hooks_path = os.popen("git config --get core.hooksPath").read().strip()
    hooks_enabled = hooks_path == ".githooks"
    hook_detail = hooks_path or "not configured; run `git config core.hooksPath .githooks`"
    lines.append(_status_line("hooks_path", hooks_enabled, hook_detail))
    return lines


def _run_doctor(key_file: str) -> int:
    sections = [
        ("Configuration", _doctor_check_key_file(key_file)),
        ("Imports", _doctor_check_imports()),
        ("Resource Registry", _doctor_check_registry(key_file)),
        ("Demo Presets", _doctor_check_presets(key_file)),
        ("CLI", _doctor_check_install_hint()),
        ("Git Safety", _doctor_check_git_safety()),
    ]

    print("[DrugClaw doctor] local readiness check")
    failed = False
    for title, lines in sections:
        print(f"\n== {title} ==")
        for line in lines:
            print(line)
            if line.startswith("[FAIL]"):
                failed = True

    if failed:
        print("\nDoctor result: setup is incomplete.")
        return 1

    print("\nDoctor result: setup looks usable.")
    return 0


def _run_list(key_file: str = "navigator_api_keys.json") -> int:
    print("[DrugClaw list] quick navigation")
    _, resource_registry = _load_registry_for_cli(key_file, strict_config=False)
    summary = resource_registry.summarize_registry()

    print("\n== Recommended First Commands ==")
    print("python -m drugclaw doctor")
    print("python -m drugclaw demo")
    print('python -m drugclaw run --query "What are the known drug targets of imatinib?"')

    print("\n== Registry Summary ==")
    print("- The resource registry is the source of truth for resource counts and status.")
    for line in _registry_summary_lines(summary):
        print(line)

    print("\n== Built-in Demos ==")
    for name, preset in DEMO_PRESETS.items():
        print(f"- {name}: {preset['description']}")
        print(f"  mode={preset['mode']}  resources={', '.join(preset['resource_filter'])}")
        print(f"  query={preset['query']}")

    print("\n== Thinking Modes ==")
    print("- graph: full retrieve -> graph build -> rerank -> respond -> reflect")
    print("- simple: fastest default for first-time users")
    print("- web_only: only web and literature search")

    print("\n== Common Resource Filters ==")
    print("- ADR: SIDER,FAERS")
    print("- DTI: ChEMBL,DGIdb,Open Targets Platform")
    print("- Labeling: DailyMed,openFDA Human Drug,MedlinePlus Drug Info")
    print("- DDI: DDInter,MecDDI,KEGG Drug")

    print("\n== Notes ==")
    print("- Prefer `demo` or `run --mode simple` for the first experience.")
    print("- Use `doctor` first if you are unsure whether local resources are ready.")
    print("- Use the resource status below to diagnose empty or degraded retrieval.")

    print("\n== Resource Status ==")
    for entry in resource_registry.get_all_resources():
        print(
            f"- {entry.name} [{entry.category}]"
            f" access={entry.access_mode}"
            f" enabled={entry.enabled}"
            f" status={entry.status}"
        )
        print(f"  reason={entry.status_reason}")
    return 0


def _run_query(
    *,
    query: str,
    thinking_mode: str,
    key_file: str,
    resource_filter: List[str] | None,
    show_evidence: bool = False,
    show_plan: bool = False,
    show_claims: bool = False,
    debug_agents: bool = False,
    save_md_report: bool = False,
) -> int:
    system = _build_system(key_file)

    if not debug_agents:
        print(f"\n{'='*80}")
        print(f"QUERY [{thinking_mode}]: {query}")
        if resource_filter:
            print(f"RESOURCE FILTER: {resource_filter}")
        print(f"{'='*80}\n")

    result = system.query(
        query,
        thinking_mode=thinking_mode,
        resource_filter=resource_filter or [],
        verbose=debug_agents,
        save_md_report=save_md_report,
    )

    if not debug_agents:
        formatted_answer = result.get("formatted_answer") or result.get("answer", "")
        print(f"\n{'='*80}\nFINAL ANSWER\n{'='*80}\n")
        print(formatted_answer)

    if show_evidence:
        _print_evidence_summary(result)
    if show_plan:
        _print_plan_summary(result)
    if show_claims:
        _print_claim_summary(result)
    if result.get("md_report_path"):
        print(f"\nMarkdown report saved to {result['md_report_path']}")

    return 0 if result.get("success") else 1


def _run_serve(host: str, port: int, key_file: str) -> int:
    import uvicorn

    from .server_app import create_app

    uvicorn.run(
        create_app(key_file=key_file),
        host=host,
        port=port,
    )
    return 0


def _run_scorecard(
    *,
    datasets: List[str] | None,
    task_types: List[str] | None,
    plan_types: List[str] | None,
    key_file: str,
    max_samples: int,
    maskself: bool | None,
    log_dir: str,
    json_output: bool,
    min_task_success_rate: float | None,
    min_evidence_quality_score: float | None,
    min_authority_source_rate: float | None,
    min_knowhow_hit_rate: float | None,
    min_package_ready_rate: float | None,
) -> int:
    summary = run_self_bench(
        datasets=datasets,
        task_types=task_types,
        plan_types=plan_types,
        key_file=key_file,
        max_samples=max_samples,
        maskself=maskself,
        log_dir=None,
    )
    release_gate_failures = _evaluate_scorecard_release_gates(
        summary,
        min_task_success_rate=min_task_success_rate,
        min_evidence_quality_score=min_evidence_quality_score,
        min_authority_source_rate=min_authority_source_rate,
        min_knowhow_hit_rate=min_knowhow_hit_rate,
        min_package_ready_rate=min_package_ready_rate,
    )
    release_gate_thresholds = {
        "min_task_success_rate": min_task_success_rate,
        "min_evidence_quality_score": min_evidence_quality_score,
        "min_authority_source_rate": min_authority_source_rate,
        "min_knowhow_hit_rate": min_knowhow_hit_rate,
        "min_package_ready_rate": min_package_ready_rate,
    }
    scorecard_payload = _build_scorecard_payload(
        summary,
        release_gate_failures=release_gate_failures,
        release_gate_thresholds=release_gate_thresholds,
    )
    logger = QueryLogger(log_dir=log_dir)
    scorecard_path = logger.save_scorecard_summary(
        scorecard_payload,
        metadata={
            "datasets": list(datasets or []),
            "task_types": list(task_types or []),
            "plan_types": list(plan_types or []),
            "max_samples": max_samples,
            "maskself": maskself,
            "release_gates": release_gate_thresholds,
        },
    )

    if json_output:
        print(json.dumps(scorecard_payload, indent=2, ensure_ascii=False))
    else:
        print("[DrugClaw scorecard]")
        print(f"total_cases={summary.total_cases}")
        print(f"completed_cases={summary.completed_cases}")
        print(f"failed_cases={summary.failed_cases}")
        print(f"task_success_rate={summary.task_success_rate:.2f}")
        print(f"evidence_quality_score={summary.evidence_quality_score:.2f}")
        print(f"authority_source_rate={summary.authority_source_rate:.2f}")
        print(f"knowhow_hit_rate={summary.knowhow_hit_rate:.2f}")
        print(f"package_ready_rate={summary.package_ready_rate:.2f}")
        task_type_breakdown = getattr(summary, "task_type_breakdown", {}) or {}
        if task_type_breakdown:
            print("[By task type]")
            for task_type, metrics in sorted(task_type_breakdown.items()):
                print(
                    f"- {task_type}: success={float(metrics.get('task_success_rate', 0.0)):.2f} "
                    f"evidence={float(metrics.get('evidence_quality_score', 0.0)):.2f} "
                    f"total_cases={int(metrics.get('total_cases', 0))}"
                )
        dataset_breakdown = getattr(summary, "dataset_breakdown", {}) or {}
        if dataset_breakdown:
            print("[By dataset]")
            for dataset_name, metrics in sorted(dataset_breakdown.items()):
                print(
                    f"- {dataset_name}: success={float(metrics.get('task_success_rate', 0.0)):.2f} "
                    f"evidence={float(metrics.get('evidence_quality_score', 0.0)):.2f} "
                    f"total_cases={int(metrics.get('total_cases', 0))}"
                )
        for dataset_name, result in sorted(summary.dataset_results.items()):
            if "error" in result:
                print(f"- {dataset_name}: ERROR={result['error']}")
                continue
            print(
                f"- {dataset_name}: accuracy={result.get('accuracy', 0.0)} "
                f"total={result.get('total', 0)}"
            )
        if release_gate_failures:
            print("Release gate failures:")
            for failure in release_gate_failures:
                print(f"- {failure}")
    print(f"Scorecard summary saved to {scorecard_path}")
    return 0 if summary.failed_cases == 0 and not release_gate_failures else 1


def _build_scorecard_payload(
    summary,
    *,
    release_gate_failures: List[str],
    release_gate_thresholds: dict,
) -> dict:
    payload = summary.to_dict()
    payload["release_gate"] = {
        "passed": not release_gate_failures,
        "failures": list(release_gate_failures),
        "thresholds": dict(release_gate_thresholds),
        "by_dataset": _build_release_gate_breakdown(
            payload.get("dataset_breakdown"),
            release_gate_thresholds=release_gate_thresholds,
        ),
        "by_task_type": _build_release_gate_breakdown(
            payload.get("task_type_breakdown"),
            release_gate_thresholds=release_gate_thresholds,
        ),
    }
    return payload


def _evaluate_scorecard_release_gates(
    summary,
    *,
    min_task_success_rate: float | None,
    min_evidence_quality_score: float | None,
    min_authority_source_rate: float | None,
    min_knowhow_hit_rate: float | None,
    min_package_ready_rate: float | None,
) -> List[str]:
    thresholds = [
        ("task_success_rate", _read_scorecard_metric(summary, "task_success_rate"), min_task_success_rate),
        (
            "evidence_quality_score",
            _read_scorecard_metric(summary, "evidence_quality_score"),
            min_evidence_quality_score,
        ),
        (
            "authority_source_rate",
            _read_scorecard_metric(summary, "authority_source_rate"),
            min_authority_source_rate,
        ),
        ("knowhow_hit_rate", _read_scorecard_metric(summary, "knowhow_hit_rate"), min_knowhow_hit_rate),
        (
            "package_ready_rate",
            _read_scorecard_metric(summary, "package_ready_rate"),
            min_package_ready_rate,
        ),
    ]
    failures: List[str] = []
    for name, observed, minimum in thresholds:
        if minimum is None:
            continue
        if float(observed) < float(minimum):
            failures.append(f"{name}={float(observed):.2f} < {float(minimum):.2f}")
    return failures


def _read_scorecard_metric(summary, key: str) -> float:
    if isinstance(summary, dict):
        value = summary.get(key, 0.0)
    else:
        value = getattr(summary, key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_release_gate_breakdown(
    breakdown: dict | None,
    *,
    release_gate_thresholds: dict,
) -> dict:
    rendered: dict = {}
    for key, metrics in sorted((breakdown or {}).items()):
        failures = _evaluate_scorecard_release_gates(
            metrics or {},
            min_task_success_rate=release_gate_thresholds.get("min_task_success_rate"),
            min_evidence_quality_score=release_gate_thresholds.get("min_evidence_quality_score"),
            min_authority_source_rate=release_gate_thresholds.get("min_authority_source_rate"),
            min_knowhow_hit_rate=release_gate_thresholds.get("min_knowhow_hit_rate"),
            min_package_ready_rate=release_gate_thresholds.get("min_package_ready_rate"),
        )
        rendered[str(key)] = {
            "passed": not failures,
            "failures": failures,
        }
    return rendered


def _print_evidence_summary(result: dict) -> None:
    structured = result.get("final_answer_structured") or {}
    if not structured:
        print("\n[DrugClaw evidence] no structured evidence available")
        return

    print("\n== Evidence Summary ==")
    print(f"summary_confidence={structured.get('summary_confidence', 0.0):.2f}")
    if structured.get("task_type"):
        print(f"task_type={structured.get('task_type')}")
    if structured.get("final_outcome"):
        print(f"final_outcome={structured.get('final_outcome')}")

    diagnostics = structured.get("diagnostics") or {}
    if diagnostics:
        rendered_pairs = []
        for key, value in diagnostics.items():
            if isinstance(value, list):
                shown = ", ".join(str(entry) for entry in value[:3])
                if len(value) > 3:
                    shown += ", ..."
                rendered_pairs.append(f"{key}={shown}")
                continue
            if isinstance(value, dict):
                continue
            rendered_pairs.append(f"{key}={value}")
        if rendered_pairs:
            print("diagnostics=" + "; ".join(rendered_pairs[:6]))

    for claim in structured.get("key_claims", [])[:5]:
        evidence_ids = ", ".join(claim.get("evidence_ids", []))
        print(
            f"- {claim.get('claim', '')} "
            f"(confidence={claim.get('confidence', 0.0):.2f}; evidence={evidence_ids})"
        )

    warnings = structured.get("warnings", [])
    if warnings:
        print("warnings=" + " | ".join(warnings[:5]))

    limitations = structured.get("limitations", [])
    if limitations:
        print("limitations=" + " | ".join(limitations[:5]))


def _print_plan_summary(result: dict) -> None:
    plan = result.get("query_plan") or {}
    if not plan:
        print("\n[DrugClaw plan] no structured query plan available")
        return

    print("\n== Query Plan ==")
    if plan.get("plan_type"):
        print(f"plan_type={plan.get('plan_type')}")
    print(f"question_type={plan.get('question_type', 'unknown')}")

    primary_task = plan.get("primary_task") or {}
    if primary_task:
        print(f"primary_task={primary_task.get('task_type', 'unknown')}")

    supporting_tasks = plan.get("supporting_tasks") or []
    if supporting_tasks:
        supporting_labels = [
            str(task.get("task_type", "")).strip()
            for task in supporting_tasks
            if str(task.get("task_type", "")).strip()
        ]
        if supporting_labels:
            print("supporting_tasks=" + ", ".join(supporting_labels))

    preferred_skills = plan.get("preferred_skills", [])
    if preferred_skills:
        print("preferred_skills=" + ", ".join(preferred_skills))

    print(
        "requires_graph_reasoning="
        + str(plan.get("requires_graph_reasoning", False))
    )

    graph_reason = result.get("graph_decision_reason", "")
    if graph_reason:
        print(f"graph_decision_reason={graph_reason}")


def _print_claim_summary(result: dict) -> None:
    assessments = result.get("claim_assessments") or []
    if not assessments:
        print("\n[DrugClaw claims] no structured claim assessments available")
        return

    print("\n== Claim Assessments ==")
    for assessment in assessments[:5]:
        support_ids = ", ".join(assessment.get("supporting_evidence_ids", []))
        contradict_ids = ", ".join(assessment.get("contradicting_evidence_ids", []))
        print(
            f"- {assessment.get('claim', '')} "
            f"(verdict={assessment.get('verdict', 'unknown')}; "
            f"confidence={assessment.get('confidence', 0.0):.2f}; "
            f"support={support_ids or '-'}; contradict={contradict_ids or '-'})"
        )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "run":
        return _run_query(
            query=args.query,
            thinking_mode=args.mode,
            key_file=args.key_file,
            resource_filter=args.resource_filter,
            show_evidence=args.show_evidence,
            show_plan=args.show_plan,
            show_claims=args.show_claims,
            debug_agents=args.debug_agents,
            save_md_report=args.save_md_report,
        )

    if args.command == "doctor":
        return _run_doctor(args.key_file)

    if args.command == "list":
        return _run_list(args.key_file)

    if args.command == "serve":
        return _run_serve(args.host, args.port, args.key_file)

    if args.command == "scorecard":
        maskself = None
        if args.maskself is not None:
            maskself = args.maskself.lower() == "true"
        return _run_scorecard(
            datasets=args.datasets,
            task_types=args.task_types,
            plan_types=args.plan_types,
            key_file=args.key_file,
            max_samples=args.max_samples,
            maskself=maskself,
            log_dir=args.log_dir,
            json_output=args.json,
            min_task_success_rate=args.min_task_success_rate,
            min_evidence_quality_score=args.min_evidence_quality_score,
            min_authority_source_rate=args.min_authority_source_rate,
            min_knowhow_hit_rate=args.min_knowhow_hit_rate,
            min_package_ready_rate=args.min_package_ready_rate,
        )

    preset = DEMO_PRESETS[args.preset]
    print(f"[DrugClaw demo] preset={args.preset} - {preset['description']}")
    print(f"[DrugClaw demo] query={preset['query']}")
    print(f"[DrugClaw demo] resource_filter={preset['resource_filter']}")
    return _run_query(
        query=preset["query"],
        thinking_mode=preset["mode"],
        key_file=args.key_file,
        resource_filter=preset["resource_filter"],
        show_evidence=args.show_evidence,
        show_plan=args.show_plan,
        show_claims=args.show_claims,
        debug_agents=args.debug_agents,
    )
