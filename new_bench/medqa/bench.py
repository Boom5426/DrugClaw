"""
Clinical MCQ — MedQA drug-focused test set.

Task  : Multiple-choice medical question; select the single best answer letter.
Labels: A, B, C, D, E
Data  : new_bench_data/medqa/medqa_drug_test.json
        JSON array [{meta_info, question, answer_idx, answer, options}, ...]
        options: [{"key": "A", "value": "..."}, ...]
QA    : Q = question stem + formatted options
        A = answer letter (A / B / C / D / E)
Metric: Accuracy (exact letter match)
"""

import json
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/medqa/medqa_drug_test.json"
)

LABELS = ["A", "B", "C", "D", "E"]

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Clinical pharmacology exam (MedQA — drug-focused subset)

MedQA is a multiple-choice medical question answering benchmark based on the \
United States Medical Licensing Examination (USMLE). This subset focuses on \
questions involving drug mechanisms, adverse effects, pharmacokinetics, drug \
interactions, and clinical drug use.

Read the clinical question carefully and select the single best answer from \
the options provided.

Respond with ONLY a JSON object: {"answer": "<letter>"}
where <letter> is one of: A, B, C, D, E\
"""


def _format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['key']}. {o['value']}" for o in options)


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        gold = str(r.get("answer_idx", "")).strip().upper()
        if gold not in LABELS:
            continue
        rows.append({
            "question": r["question"],
            "options":  r["options"],
            "gold":     gold,
        })

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _format_prompt(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND}\n\n"
        f"{s['question']}\n\n"
        f"Options:\n{_format_options(s['options'])}"
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_bench, extract_mcq_answer

    samples = load_data(max_samples)
    valid_keys = sorted({s["gold"] for s in samples} | set(LABELS))

    return run_bench(
        dataset_name="medqa",
        labels=LABELS,
        default_label="A",
        system_prompt=SYSTEM_PROMPT,
        samples=samples,
        format_prompt=_format_prompt,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
        answer_extractor=lambda text: extract_mcq_answer(text, valid_keys),
    )
