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
from .main_system import DrugClawSystem
from .models import ThinkingMode
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
        help="Path to api_keys.json or navigator_api_keys.json.",
    )

    return parser


def _status_line(label: str, ok: bool, detail: str) -> str:
    status = "OK" if ok else "FAIL"
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
    return lines


def _doctor_check_registry(key_file: str) -> List[str]:
    try:
        _, resource_registry = _load_registry_for_cli(key_file, strict_config=False)
    except Exception as exc:
        return [_status_line("registry", False, f"cannot build resource registry ({exc})")]

    lines = _registry_summary_lines(resource_registry.summarize_registry())
    for entry in resource_registry.get_all_resources():
        usable = entry.status == "ready"
        detail = (
            f"category={entry.category}; access={entry.access_mode}; "
            f"status={entry.status}; reason={entry.status_reason}"
        )
        lines.append(_status_line(f"resource:{entry.name}", usable or not entry.enabled, detail))
    return lines


def _doctor_check_presets(key_file: str) -> List[str]:
    lines: List[str] = []
    try:
        config, _ = _load_registry_for_cli(key_file, strict_config=True)
    except Exception as exc:
        return [_status_line("config", False, f"cannot initialize Config ({exc})")]

    registry = build_default_registry(config)
    for preset_name, preset in DEMO_PRESETS.items():
        required = preset["resource_filter"]
        unavailable = []
        details = []
        for skill_name in required:
            skill = registry.get_skill(skill_name)
            if skill is None:
                unavailable.append(skill_name)
                details.append(f"{skill_name}=missing")
                continue
            access = getattr(skill, "access_mode", "unknown")
            available = skill.is_available()
            local_paths: List[Path] = []
            if access in {"LOCAL_FILE", "DATASET"}:
                for value in skill.config.values():
                    if isinstance(value, str) and value:
                        path = Path(value).expanduser()
                        if not path.is_absolute():
                            path = (Path.cwd() / path).resolve()
                        local_paths.append(path)
                    elif isinstance(value, dict):
                        for nested in value.values():
                            if isinstance(nested, str) and nested:
                                path = Path(nested).expanduser()
                                if not path.is_absolute():
                                    path = (Path.cwd() / path).resolve()
                                local_paths.append(path)
                if local_paths:
                    existing_paths = [path for path in local_paths if path.exists()]
                    available = bool(existing_paths)
                    path_note = f"{len(existing_paths)}/{len(local_paths)} files"
                else:
                    available = False
                    path_note = "no local path configured"
            else:
                path_note = "available" if available else "unavailable"
            if access == "CLI" and hasattr(skill, "_cli_available"):
                cli_ok = skill._cli_available()  # type: ignore[attr-defined]
                details.append(f"{skill_name}={access}:{'cli' if cli_ok else 'rest-fallback'}")
            else:
                details.append(f"{skill_name}={access}:{path_note}")
            if not available:
                unavailable.append(skill_name)
        ok = not unavailable
        lines.append(
            _status_line(
                f"demo:{preset_name}",
                ok,
                ", ".join(details) if details else "no resources",
            )
        )
    return lines


def _doctor_check_install_hint() -> List[str]:
    installed_as_script = any(
        Path(path).joinpath("drugclaw").exists()
        for path in os.environ.get("PATH", "").split(os.pathsep)
        if path
    )
    detail = "command found on PATH" if installed_as_script else "use `python -m drugclaw ...` or `pip install -e . --no-build-isolation`"
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
) -> int:
    config = Config(key_file=key_file)
    system = DrugClawSystem(config)

    result = system.query(
        query,
        thinking_mode=thinking_mode,
        resource_filter=resource_filter or [],
    )

    if show_evidence:
        _print_evidence_summary(result)
    if show_plan:
        _print_plan_summary(result)
    if show_claims:
        _print_claim_summary(result)

    return 0 if result.get("success") else 1


def _print_evidence_summary(result: dict) -> None:
    structured = result.get("final_answer_structured") or {}
    if not structured:
        print("\n[DrugClaw evidence] no structured evidence available")
        return

    print("\n== Evidence Summary ==")
    print(f"summary_confidence={structured.get('summary_confidence', 0.0):.2f}")

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
    print(f"question_type={plan.get('question_type', 'unknown')}")

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
        )

    if args.command == "doctor":
        return _run_doctor(args.key_file)

    if args.command == "list":
        return _run_list(args.key_file)

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
    )
