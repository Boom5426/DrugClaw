"""
Drug Sensitivity — GDSC2.

Task  : Predict whether a cancer cell line is sensitive or resistant to a drug.
Labels: Sensitive, Resistant
Data  : new_bench_data/gdsc/gdsc2_benchmark.json
        JSON array [{drug, cell_line, cancer_type, target, pathway,
                     label (0=Resistant/1=Sensitive), label_name,
                     ln_ic50, threshold, query}, ...]
        Balanced: 500 Sensitive + 500 Resistant.
QA    : Q = "Is the cancer cell line {cell_line} ({cancer_type}) sensitive or
              resistant to {drug} (targets: {target}; pathway: {pathway})?"
        A = "Sensitive" / "Resistant"
Metric: Accuracy, F1 (macro), per-class precision/recall/F1
"""

import json
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/gdsc/gdsc2_benchmark.json"
)

LABELS = ["Sensitive", "Resistant"]

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Cancer drug sensitivity prediction (GDSC2)

The Genomics of Drug Sensitivity in Cancer (GDSC) project systematically profiles \
drug sensitivity across hundreds of cancer cell lines using in vitro dose-response \
experiments. Sensitivity is determined by the natural log of the IC50 (ln IC50) \
relative to a drug-specific threshold derived from the distribution across all \
tested cell lines:
- Sensitive: the cell line's ln IC50 is below the threshold — the drug inhibits \
cell growth effectively.
- Resistant:  the cell line's ln IC50 is at or above the threshold — the drug \
has limited anti-proliferative effect.

Given a drug (with its molecular targets and signalling pathway) and a cancer \
cell line (with cancer type), predict whether the cell line is Sensitive or \
Resistant to that drug.

Respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: Sensitive, Resistant\
"""


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        gold = r.get("label_name", "").strip()
        if gold not in LABELS:
            gold = "Sensitive" if int(r.get("label", 0)) == 1 else "Resistant"
        rows.append({
            "drug":        r["drug"],
            "cell_line":   r["cell_line"],
            "cancer_type": r.get("cancer_type", "Unknown"),
            "target":      r.get("target", "Unknown"),
            "pathway":     r.get("pathway", "Unknown"),
            "gold":        gold,
        })

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _format_prompt(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND}\n\n"
        f"Is the cancer cell line {s['cell_line']} ({s['cancer_type']}) "
        f"sensitive or resistant to {s['drug']} "
        f"(targets: {s['target']}; pathway: {s['pathway']})?"
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_bench

    return run_bench(
        dataset_name="gdsc",
        labels=LABELS,
        default_label="Resistant",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=_format_prompt,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
