#!/usr/bin/env python3
"""
run_all.py — Execute all self-bench classification benchmarks and report results.

Usage:
    python -m self_bench.run_all                          # all datasets
    python -m self_bench.run_all --datasets ade_corpus dilirank  # specific ones
    python -m self_bench.run_all --max-samples 50         # limit per dataset
    python -m self_bench.run_all --key-file /path/to/keys.json
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

ALL_DATASETS = [
    "ade_corpus",
    "ddi_corpus",
    "drugprot",
    "phee",
    "dilirank",
    "n2c2_2018",
    "psytar",
]


def run_one(dataset: str, key_file: str | None, max_samples: int) -> dict:
    """Import and run a single dataset benchmark, return its results dict."""
    mod = importlib.import_module(f"self_bench.{dataset}.bench")
    return mod.run(key_file=key_file, max_samples=max_samples)


def main():
    parser = argparse.ArgumentParser(description="DrugClaw self-bench runner")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Dataset names to benchmark (default: all)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples per dataset (0 = use all available)",
    )
    parser.add_argument(
        "--key-file", type=str, default=None,
        help="Path to API key JSON file (default: auto-detect via Config)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results (default: stdout summary only)",
    )
    args = parser.parse_args()

    all_results = {}
    for ds in args.datasets:
        if ds not in ALL_DATASETS:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {ds}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = run_one(ds, args.key_file, args.max_samples)
            result["elapsed_sec"] = round(time.time() - t0, 1)
            all_results[ds] = result
            print(f"  Accuracy: {result.get('accuracy', 'N/A')}")
            if "f1_macro" in result:
                print(f"  F1 (macro): {result['f1_macro']}")
            print(f"  Samples: {result.get('total', '?')}  "
                  f"Correct: {result.get('correct', '?')}  "
                  f"Time: {result['elapsed_sec']}s")
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results[ds] = {"error": str(e)}

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        if "error" in res:
            print(f"  {ds:20s}  ERROR: {res['error']}")
        else:
            acc = res.get("accuracy", "N/A")
            f1 = res.get("f1_macro", "")
            f1_str = f"  F1={f1}" if f1 else ""
            print(f"  {ds:20s}  Acc={acc}  N={res.get('total', '?')}{f1_str}")

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
