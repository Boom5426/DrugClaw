#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drugclaw.cli import _run_scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified DrugClaw self-bench scorecard and persist the latest snapshot.",
    )
    parser.add_argument(
        "--suite",
        choices=["self_bench", "cli_usability_min_pack"],
        default="self_bench",
        help="Scorecard suite to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional self-bench dataset names to include.",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Optional QueryPlan-aligned task types to include.",
    )
    parser.add_argument(
        "--plan-types",
        nargs="+",
        default=None,
        help="Optional plan types to include.",
    )
    parser.add_argument(
        "--key-file",
        default="navigator_api_keys.json",
        help="Path to navigator_api_keys.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per dataset (0 = use all available).",
    )
    parser.add_argument(
        "--maskself",
        choices=["true", "false"],
        default=None,
        help="Pass true/false to benchmark with DrugClaw RAG enabled or masked.",
    )
    parser.add_argument(
        "--log-dir",
        default="query_logs",
        help="Directory used to persist the latest scorecard summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the scorecard summary as JSON.",
    )
    parser.add_argument(
        "--min-task-success-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for task_success_rate.",
    )
    parser.add_argument(
        "--min-evidence-quality-score",
        type=float,
        default=None,
        help="Optional release gate threshold for evidence_quality_score.",
    )
    parser.add_argument(
        "--min-authority-source-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for authority_source_rate.",
    )
    parser.add_argument(
        "--min-knowhow-hit-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for knowhow_hit_rate.",
    )
    parser.add_argument(
        "--min-package-ready-rate",
        type=float,
        default=None,
        help="Optional release gate threshold for package_ready_rate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    maskself: bool | None = None
    if args.maskself is not None:
        maskself = args.maskself == "true"
    return _run_scorecard(
        suite=args.suite,
        datasets=list(args.datasets or []),
        task_types=list(args.task_types or []),
        plan_types=list(args.plan_types or []),
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


if __name__ == "__main__":
    raise SystemExit(main())
