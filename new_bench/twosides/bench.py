"""
Drug-pair ADR — TWOSIDES.

Task  : Given a drug pair and an adverse event, predict whether the combination
        causes that event (yes / no).
Data  : new_bench_data/2sides/twosides_benchmark.json
        JSON array [{drug_1, drug_2, condition, label (0/1), PRR, query}, ...]
        label 1 = positive (pair causes AE), 0 = negative.
        Dataset is balanced: 500 positives + 500 negatives.
QA    : Q = query field ("Does the combination of X and Y cause Z?")
        A = "yes" / "no"
Metric: Precision, Recall, F1, AUROC
"""

import json
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/2sides/twosides_benchmark.json"
)

LABELS = ["yes", "no"]

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Pharmacovigilance — drug-pair adverse event prediction (TWOSIDES)

TWOSIDES is a large-scale pharmacovigilance database derived from the FDA Adverse \
Event Reporting System (FAERS). It records statistically significant associations \
between drug combinations and adverse events, identified using the Proportional \
Reporting Ratio (PRR). Each sample presents a drug pair and a specific adverse \
event; your task is to determine whether the combination is associated with \
that adverse event.

Respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: yes, no\
"""


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        label_int = int(r["label"])
        gold = "yes" if label_int == 1 else "no"
        rows.append({
            "query":     r["query"],
            "gold":      gold,
            "_label":    label_int,
        })

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_binary_bench

    return run_binary_bench(
        dataset_name="twosides",
        pos_label="yes",
        neg_label="no",
        default_label="no",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=lambda s: f"{TASK_BACKGROUND}\n\n{s['query']}",
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
